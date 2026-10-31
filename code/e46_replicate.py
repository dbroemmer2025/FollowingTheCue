#!/usr/bin/env python3
"""Freeze and run a fixed-design E46 replication (official-CoT, authority cue).

Run `freeze --help` first. A design hash is a local audit trail, not an
external preregistration. Load only trusted project pickle files.

Item freshness is a declared property of the design, not an assumption.
Pass --exclude with the discovery trace caches to draw items none of them
used, or --no-exclusions to state on the record that this design reuses
whatever the pool holds. There is no default, because a silently empty
exclusion set produces a design that looks fresh and is not.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import glob
import gc
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
import math
from pathlib import Path
import pickle
import random
import re
import subprocess
import sys
import traceback
from types import SimpleNamespace
import unicodedata

HERE = Path(__file__).resolve().parent
VERSION = "e46_fixed_replication_v1"
CODE_FILES = (
    "e46_replicate.py", "e46_pretest_cue.py", "e46_prompts_cue.py",
    "e46_bbh_items.py", "e46_followup.py", "e46_capture_residuals.py",
    "e46_layer_capture.py", "e46_followup_predict.py", "e46_followup_gap.py",
    "e46_cache.py", "e46_readers.py", "p2_common.py", "models.py",
)
FILES = {"initial": "traces.pkl", "followup": "followups.pkl",
         "capture": "residuals_all_layers.pkl",
         "capture-followup": "followup_residuals_all_layers.pkl"}
PARENTS = {"initial": (), "followup": ("initial",), "capture": ("initial",),
           "capture-followup": ("followup",),
           "analyze": ("initial", "followup", "capture-followup")}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True,
                                    ensure_ascii=True).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def canonical(text):
    return " ".join(unicodedata.normalize("NFKC", text).split()).casefold()


def source_id(stem):
    return digest(canonical(stem))


def render(item, seed, cot):
    from e46_prompts_cue import DEFAULT_N_OPS, render_pair
    return render_pair(item, seed, DEFAULT_N_OPS, cot_prefix=cot,
                       cot_counterbalance=True)


def read_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def runtime_versions():
    versions = {"python": sys.version}
    for package in ("numpy", "scikit-learn", "scipy", "torch", "transformers", "tokenizers"):
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = None
    return versions


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "x") as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write("\n")


def discovery_sources(patterns):
    paths = set()
    for pattern in patterns:
        matches = glob.glob(pattern)
        if not matches:
            raise ValueError(f"Discovery pattern matched nothing: {pattern}")
        paths.update(Path(p).resolve() for p in matches)
    excluded, provenance = set(), []
    for path in sorted(paths):
        blob = read_pickle(path)
        if blob.get("kind") != "e46_pretest_traces":
            raise ValueError(f"Use original E46 trace caches for exclusions: {path}")
        rows = [r for rs in blob["raw_outputs"].values() for r in rs]
        if not rows or any(not r.get("item", {}).get("expr") for r in rows):
            raise ValueError(f"Missing embedded source questions: {path}")
        ids = {source_id(r["item"]["expr"]) for r in rows}
        excluded.update(ids)
        provenance.append({"name": path.name, "sha256": file_hash(path),
                           "n_sources": len(ids)})
    return excluded, provenance


def fresh_pool(raw, excluded, cot):
    from e46_bbh_items import parse_example
    seen, pool = set(), []
    counts = Counter(raw=len(raw))
    demo_text = canonical(cot)
    for row in raw:
        parsed = parse_example(row)
        if (parsed is None or len(parsed["options"]) < 4
                or len(set(parsed["options"])) != len(parsed["options"])):
            counts["unusable"] += 1
            continue
        key = source_id(parsed["stem"])
        if key in seen:
            counts["duplicate_source"] += 1
            continue
        seen.add(key)
        if key in excluded:
            counts["discovery_overlap"] += 1
        elif canonical(parsed["stem"]) in demo_text:
            counts["demonstration_overlap"] += 1
        else:
            pool.append(row)
    counts["available"] = len(pool)
    return pool, dict(counts)


def precision_plan(half_width, citation_rate, flipper_rate, available, n):
    if not 0 < half_width < 1 or not 0 < citation_rate < 1 or not 0 < flipper_rate <= 1:
        raise ValueError("Planning rates and half-width must be in (0,1); flipper rate may be 1")
    needed = math.ceil(1.96**2 * citation_rate * (1 - citation_rate) / half_width**2)
    worst = math.ceil(1.96**2 * .25 / half_width**2)
    return {"citation_rate_assumption": citation_rate,
            "flipper_rate_assumption": flipper_rate, "desired_half_width": half_width,
            "required_flippers_approx": needed, "worst_case_flippers_approx": worst,
            "required_items_approx": math.ceil(needed / flipper_rate),
            "expected_flippers_approx": n * flipper_rate,
            "pool_can_meet_expected_target": available * flipper_rate >= needed,
            "scope": "Normal-approximation planning for citation proportion only; not probe power or a guarantee."}


FRESH = "fresh_relative_to_declared_discovery_caches"
REUSED = "no_exclusions_declared_items_may_repeat_discovery"
REUSE_CAVEAT = ("Items were NOT excluded from any prior run. If a discovery run "
                "on this pool produced the hypothesis under test, this is a "
                "re-estimate on overlapping items, not an independent replication.")


class _PlainBar:
    """Line-per-update stand-in used when tqdm is not installed."""

    def __init__(self, total, desc):
        self.total, self.desc, self.n = total, desc, 0

    def update(self, k=1):
        self.n += k
        print(f"  {self.desc}: {self.n}/{self.total}", flush=True)

    def set_postfix_str(self, text):
        if text:
            print(f"    {text}", flush=True)

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def bar(total, desc, unit="it"):
    """A tqdm bar where tqdm exists, a printed fallback where it does not.

    tqdm stays optional deliberately. This script's job is to survive a long
    GPU booking on a fresh VM, and dying on an ImportError hours in would cost
    far more than a progress bar is worth. The fallback keeps the interface
    and prints one line per update, so a piped log stays readable.

    Bars go to stderr so that stdout -- which carries the design id, the pool
    counts and the sealing lines -- can be redirected to a file without
    carriage-return noise.
    """
    try:
        from tqdm.auto import tqdm
    except ImportError:
        return _PlainBar(total, desc)
    return tqdm(total=total, desc=desc, unit=unit, dynamic_ncols=True,
                file=sys.stderr, leave=True)


def occupancy_summary(counts):
    """Compact live view of what the adaptive assignment has actually filled.

    Worth surfacing during generation rather than at the end: an empty
    flipper stratum an hour in is a reason to stop and rethink the cue, and
    the run is otherwise silent about it until every batch is done.
    """
    if not counts:
        return ""
    short = {"flipper": "flip", "wrong_to_cue": "w2cue",
             "already_cue_wrong": "acw", "already_cue_correct": "acc",
             "resister": "resist", "other": "other"}
    parts = [f"{short.get(name, name)}={data['all']}"
             for name, data in counts.items() if data["all"]]
    return " ".join(parts) if parts else "no strata yet"


def code_state():
    """Hash every frozen file, keeping 'missing' separate from 'changed'.

    These are different problems with different fixes -- a missing file means
    the code directory is incomplete, a differing hash means it drifted -- and
    hashing them one at a time inside a dict comprehension reported neither.
    It raised a bare FileNotFoundError naming whichever file happened to be
    reached first, which turns an incomplete sync into one round trip per
    missing file.
    """
    present, missing = {}, []
    for name in CODE_FILES:
        path = HERE / name
        if path.exists():
            present[name] = file_hash(path)
        else:
            missing.append(name)
    return present, missing


def require_code_files():
    _, missing = code_state()
    if missing:
        raise ValueError(
            f"Frozen code incomplete in {HERE}: missing {', '.join(missing)}. "
            f"All {len(CODE_FILES)} files are hashed into the design and must "
            "be present and current before freezing. Sync them from your "
            "working copy.")


def resolve_revision(hf, revision):
    """Turn `--revision auto` into a concrete commit hash, once, at freeze time.

    Pinning exists so the design names one exact set of weights. It does not
    require a human to transcribe 40 hex characters, and transcription is
    itself a failure mode: a hash pasted from the wrong repo pins the wrong
    model just as firmly as the right one.

    Resolving here keeps every guarantee. The design still stores a fixed
    hash, `run` still never consults the network to learn it, and a typo in
    --hf now surfaces as an unexpected hash rather than silently succeeding.

    Only the literal "auto" resolves. Branches and tags stay rejected: they
    move, and a design pinned to a moving target is not pinned.
    """
    if revision != "auto":
        return revision
    try:
        from huggingface_hub import HfApi
        sha = HfApi().model_info(hf).sha
    except Exception as exc:
        raise ValueError(
            f"Could not resolve --revision auto for {hf}: {exc}. "
            "Check the repository id, or pass the commit hash explicitly."
        ) from exc
    if not isinstance(sha, str) or not sha:
        raise ValueError(f"Hugging Face returned no commit hash for {hf}")
    print(f"Resolved --revision auto for {hf}: {sha}")
    return sha


def freshness(patterns, opted_out):
    """Decide, once, whether this design claims item freshness.

    `--exclude` used to be required, which made the script unusable for the
    designs that legitimately exclude nothing: a substrate with no prior E46
    run, or a deliberate item-matched comparison across models. But making it
    merely optional is worse than requiring it -- forget the flag and you get
    `excluded_source_ids: []`, a design that passes every integrity check and
    silently reuses the questions that generated the hypothesis.

    So the choice is mandatory and the flags are mutually exclusive. Either
    way the design records which claim it makes, and the reuse claim carries
    its caveat into every downstream artifact.
    """
    if patterns and opted_out:
        raise ValueError("Pass --exclude or --no-exclusions, not both")
    if not patterns and not opted_out:
        raise ValueError(
            "Declare item freshness: --exclude <discovery trace pkls> to draw "
            "items no prior run used, or --no-exclusions to record that this "
            "design reuses them. There is no default.")
    return REUSED if opted_out else FRESH


def freeze(args):
    from e46_bbh_items import build_bbh_items, load_cot_file
    require_code_files()
    if args.n < 20 or args.batch_size < 1 or min(args.max_new_tokens, args.why_max_new_tokens) < 1:
        raise ValueError("Need n >= 20, positive batch size and token caps")
    item_freshness = freshness(args.exclude, args.no_exclusions)
    if args.n_perm < 0:
        raise ValueError("--n-perm must be nonnegative")
    if Path(args.out).exists():
        raise ValueError(f"Design exists: {args.out}")
    # After the cheap local checks, so a doomed run never hits the network.
    revision = resolve_revision(args.hf, args.revision)
    if not re.fullmatch(r"[0-9a-fA-F]{40}", revision):
        raise ValueError("--revision must be the full 40-character HF commit "
                         "hash or 'auto', not a moving branch or tag like 'main'")
    raw = json.loads(Path(args.bbh_json).read_text())
    raw = raw["examples"] if isinstance(raw, dict) else raw
    excluded, provenance = discovery_sources(args.exclude)
    cot = load_cot_file(args.cot_file, args.task)
    if "Q:" not in cot or "Options:" not in cot:
        raise ValueError("Expected an official BBH multiple-choice CoT prefix")
    from e46_prompts_cue import cot_shift_period
    if len(re.findall(r"^Q:", cot, re.MULTILINE)) != 3 or cot_shift_period(cot) != 20:
        raise ValueError("This fixed movie design requires three official demos with the 20-item rotation schedule")
    pool, counts = fresh_pool(raw, excluded, cot)
    print("Source pool:", counts)
    if args.n > len(pool):
        raise ValueError(f"Requested {args.n} fresh items; only {len(pool)} available. No wrapping or reuse allowed.")
    random.Random(args.item_seed).shuffle(pool)
    items = build_bbh_items(pool[:args.n], args.n, args.task, args.item_seed)
    if len(items) != args.n:
        raise ValueError("Item builder did not return the frozen sample size")
    for it in items:
        it.cue_type = "authority"
    # Render now as a validation of the prefix, rotation and item schema.
    prompts = []
    with bar(len(items), "rendering prompts", "item") as rendering:
        for it in items:
            prompts.append(render(it, args.item_seed, cot)[0])
            rendering.update()
    plan = precision_plan(args.half_width, args.citation_rate, args.flipper_rate,
                          len(pool), args.n)
    design = {
        "version": VERSION, "created_utc": datetime.now(timezone.utc).isoformat(),
        "model": args.model, "hf": args.hf, "hf_revision": revision.lower(),
        "revision_source": "auto_resolved_at_freeze" if args.revision == "auto" else "explicit",
        "dtype": args.dtype, "task": args.task, "n": args.n,
        "item_seed": args.item_seed, "batch_size": args.batch_size,
        "max_new_tokens": args.max_new_tokens, "why_max_new_tokens": args.why_max_new_tokens,
        "prompt_regime": "imitative", "cue_type": "authority", "cot_counterbalance": True,
        "cue_assignment": "adaptive plan_want on completed prior batches, on_frac=0.4, secondary_min=0",
        "stopping": "fixed number of unique source items; no quota stopping or replacement of failures",
        "cot_prefix": cot, "items": [it.as_dict() for it in items],
        "uncued_prompt_hashes": [digest(p) for p in prompts],
        "source_ids": [source_id(it.expr) for it in items],
        "excluded_source_ids": sorted(excluded), "discovery_inputs": provenance,
        "item_freshness": item_freshness,
        "source_file_sha256": file_hash(args.bbh_json), "pool_counts": counts,
        "precision_plan": plan,
        "analysis": {"target": "automatic mentions_cue_turn2, not causal attribution",
                     "population": "cued q_end, non-other pairs, generated follow-up with finite lens",
                     "layer": "legacy hidden_states[-2] R; no layer selection",
                     "primary_comparisons": ["state_vs_base", "state_vs_text"],
                     "seed": 46, "n_perm": args.n_perm,
                     "reader_policy": "existing fixed candidate grids and calibration; inner 3-fold selection, outer 5-fold source-group CV",
                     "minimum": "20 eligible items and at least 5 per mention class; otherwise inconclusive",
                     "other_layers": "exploratory only, never substitute for primary R",
                     "intervals": "8000 source-item bootstrap draws on fixed OOF losses, not refit uncertainty",
                     "permutations": "refit label-shuffle no-association diagnostics with Holm correction; NOT a paired test of equal reader performance",
                     "interpretation": "Forecasting later citation, not suppression or the completed-trace verbalization gap"},
        "code_sha256": code_state()[0],
        "note": ("Locally frozen analysis policy, not external preregistration. "
                 + ("Fresh relative only to the declared discovery caches. "
                    if item_freshness == FRESH else REUSE_CAVEAT + " ")
                 + "Model/regime changes are separate conditions, not size-only comparisons."),
    }
    design["design_id"] = digest(design)
    write_json(args.out, design)
    print(f"Frozen {args.n} items: {args.out}\ndesign_id: {design['design_id']}")
    print(f"Item freshness: {item_freshness}")
    if item_freshness == REUSED:
        print(f"WARNING: {REUSE_CAVEAT}")
    print(f"Planning: about {plan['required_flippers_approx']} flippers needed; "
          f"{plan['expected_flippers_approx']:.1f} expected at the stated rate.")
    if args.n * args.flipper_rate < plan["required_flippers_approx"]:
        print("WARNING: requested sample is below the planning target. Report the resulting uncertainty; do not top up after viewing outcomes.")
    return design


def load_design(path):
    d = json.loads(Path(path).read_text())
    payload = {k: v for k, v in d.items() if k != "design_id"}
    if d.get("version") != VERSION or d.get("design_id") != digest(payload):
        raise ValueError("Design changed or unsupported version; do not amend after outcomes")
    present, missing = code_state()
    if missing:
        raise ValueError(f"Frozen code missing from {HERE}: {', '.join(missing)}. "
                         "Restore the frozen code before running any stage.")
    drifted = [n for n in CODE_FILES if d["code_sha256"].get(n) != present[n]]
    if drifted:
        raise ValueError(f"Frozen code differs: {', '.join(drifted)}. Restore the "
                         "frozen code, or document a new design before new data.")
    ids = d["source_ids"]
    if (len(ids) != d["n"] or len(set(ids)) != len(ids)
            or set(ids) & set(d["excluded_source_ids"])
            or ids != [source_id(it["expr"]) for it in d["items"]]):
        raise ValueError("Invalid or overlapping source items")
    return d


def run_initial(d, path, device_map):
    from e46_pretest_cue import generate_traces, save_traces, score_batch
    from e46_prompts_cue import Item, extract_answer, plan_want, retarget_cue
    items, unc, cue, prompts = [], [], [], {}
    n_batches = math.ceil(d["n"] / d["batch_size"])
    progress = bar(2 * d["n"], f"initial {d['model']}", "generation")
    for start in range(0, d["n"], d["batch_size"]):
        which = start // d["batch_size"] + 1
        batch = [Item(**it) for it in d["items"][start:start + d["batch_size"]]]
        pairs = [render(it, d["item_seed"], d["cot_prefix"]) for it in batch]
        if [digest(p[0]) for p in pairs] != d["uncued_prompt_hashes"][start:start + len(batch)]:
            raise ValueError("Rendered prompts differ from frozen design")
        def generate(ps):
            return generate_traces(d["hf"], ps, d["max_new_tokens"], d["dtype"],
                                   device_map, revision=d["hf_revision"])
        progress.set_postfix_str(f"batch {which}/{n_batches} uncued")
        bu = generate([p[0] for p in pairs])
        if len(bu) != len(batch):
            raise ValueError("Missing uncued generations; refusing a partial batch")
        progress.update(len(batch))
        rng = random.Random(d["item_seed"] * 5003 + start)
        counts = score_batch(items, unc, cue)[0]["counts"] if items else None
        assigned = []
        for it, u in zip(batch, bu):
            ans = extract_answer(u["text"])
            want = plan_want(ans == it.correct_letter, counts, rng=rng,
                             on_frac=.4, secondary_min=0)
            assigned.append(retarget_cue(it, ans, want, rng))
        pairs = [render(it, d["item_seed"], d["cot_prefix"]) for it in assigned]
        progress.set_postfix_str(f"batch {which}/{n_batches} cued  "
                                 + occupancy_summary(counts))
        bc = generate([p[1] for p in pairs])
        if len(bc) != len(batch):
            raise ValueError("Missing cued generations; refusing a partial batch")
        progress.update(len(batch))
        for it, pair in zip(assigned, pairs):
            for arm, p in zip(("uncued", "cued"), pair):
                prompts[(it.index, arm)] = p
        items.extend(assigned)
        unc.extend(bu)
        cue.extend(bc)
    final_counts, labels = score_batch(items, unc, cue)
    progress.set_postfix_str(occupancy_summary(final_counts["counts"]))
    progress.close()
    meta = {"replication_design_id": d["design_id"], "hf_revision": d["hf_revision"],
            "seed": d["item_seed"], "substrate": d["task"], "prompt_regime": "imitative",
            "cue_type": "authority", "adaptive_cue": True, "adaptive_on_frac": .4,
            "cot_counterbalance": True, "fewshot_n": 3, "max_new_tokens": d["max_new_tokens"],
            "replication_n": d["n"], "replication_batch_size": d["batch_size"],
            "replication_source_ids": d["source_ids"], "dtype": d["dtype"]}
    save_traces(str(path), d["model"], d["hf"], items, unc, cue, labels, meta, prompts)


def check_stage(d, directory, stage):
    path = directory / FILES[stage]
    receipt_path = directory / f"{stage}_receipt.json"
    if not path.exists() or not receipt_path.exists():
        raise ValueError(f"Run the {stage} stage first; missing cache or receipt")
    receipt = json.loads(receipt_path.read_text())
    if receipt.get("design_id") != d["design_id"] or receipt.get("sha256") != file_hash(path):
        raise ValueError(f"Wrong design or modified {stage} cache")
    for parent in PARENTS[stage]:
        check_stage(d, directory, parent)
        if receipt["parents"][parent] != file_hash(directory / FILES[parent]):
            raise ValueError(f"{stage} came from a different {parent} cache")


def wilson(k, n):
    if n == 0:
        return None
    z = 1.96
    centre = (k / n + z*z / (2*n)) / (1 + z*z/n)
    radius = z * math.sqrt(k/n * (1-k/n)/n + z*z/(4*n*n)) / (1 + z*z/n)
    return [max(0., centre - radius), min(1., centre + radius)]


def behavioral_report(traces, followups):
    from e46_pretest_cue import score_batch
    from e46_prompts_cue import Item
    from e46_followup import score
    raw = traces["raw_outputs"][traces["metadata"]["task_names"][0]]
    by = {(r["index"], r["condition"]): r for r in raw}
    ids = sorted({r["index"] for r in raw})
    its = [Item(**by[(i, "cued")]["item"]) for i in ids]
    _, labels = score_batch(its, [by[(i, "uncued")] for i in ids], [by[(i, "cued")] for i in ids])
    rows = followups["rows"]
    score(rows, "authority")
    fu = {(r["index"], r["arm"]): r for r in rows}
    table = []
    for st in sorted({lb["stratum"] for lb in labels}):
        indexes = [i for i, lb in zip(ids, labels) if lb["stratum"] == st]
        for arm in ("uncued", "cued"):
            observed = [fu[(i, arm)] for i in indexes if (i, arm) in fu and fu[(i, arm)].get("ids2")]
            k = sum(bool(r["mentions_cue_turn2"]) for r in observed)
            n, total = len(observed), len(indexes)
            table.append({"stratum": st, "arm": arm, "n_pairs": total,
                          "n_followups": n, "missing_followups": total - n,
                          "citations": k, "citation_rate": k/n if n else None,
                          "wilson_95": wilson(k, n),
                          "all_pair_rate_bounds_missing_unknown": [k/total, (k + total - n)/total],
                          "hit_cap_followups": sum(bool(r.get("hit_cap2")) for r in observed)})
    return {"n_unique_items": len(ids), "n_original_rows": len(raw),
            "unparsed_original_arms": sum(not r["completed"] for r in raw),
            "hit_cap_original_arms": sum(bool(r["hit_cap"]) for r in raw), "by_stratum_arm": table,
            "scope": "Automatic mention within episode, not contextual or causal attribution; missing follow-ups are not silent negatives. Wilson intervals are nominal descriptive intervals under adaptive assignment."}


def analyze(d, directory):
    from e46_followup_predict import collect, run
    from threadpoolctl import threadpool_limits
    prefix = directory / "forecast"
    for name in ("behavior.json", "replication_result.json", "forecast.json", "forecast_oof.csv"):
        if (directory / name).exists():
            raise ValueError(f"Analysis output exists: {name}; do not overwrite or seed-search")
    traces = read_pickle(directory / FILES["initial"])
    fu = read_pickle(directory / FILES["followup"])
    behavior = behavioral_report(traces, fu)
    write_json(directory / "behavior.json", {"design_id": d["design_id"], **behavior})
    fr_path, fu_path = directory / FILES["capture-followup"], directory / FILES["followup"]
    fr = read_pickle(fr_path)
    fr_meta = fr["metadata"]
    del fr
    if fr_meta.get("resid_layer") != -2:
        raise ValueError("Primary analysis requires the frozen legacy R layer (-2)")
    fresh = d.get("item_freshness", FRESH)
    result = {"design_id": d["design_id"], "analysis": d["analysis"],
              "n_attempted_unique_items": d["n"], "behavior": behavior,
              "item_freshness": fresh, "status": "pending",
              "runtime_versions": runtime_versions()}
    if fresh != FRESH:
        result["item_freshness_caveat"] = REUSE_CAVEAT
    # Only sample-size failures are an inconclusive design, not corrupt joins.
    try:
        rows, y, groups, _, _ = collect(fr_path, fu_path)
    except ValueError as exc:
        if not str(exc).startswith(("Need at least 20", "Need at least five")):
            raise
        result.update(status="inconclusive_insufficient_data", reason=str(exc))
    else:
        if len(set(groups)) != len(groups):
            raise ValueError("Repeated source query in fresh-item primary analysis")
        result["n_eligible"] = len(y)
        result["excluded_from_primary"] = d["n"] - len(y)
        args = SimpleNamespace(furesid=str(fr_path), followups=str(fu_path),
                               out_prefix=str(prefix), seed=d["analysis"]["seed"],
                               n_perm=d["analysis"]["n_perm"],
                               analysis_label=("fresh-item" if fresh == FRESH else "reused-item")
                               + " replication, frozen nested-CV policy")
        print(f"Fitting nested-CV readers on {len(y)} eligible items "
              f"({d['analysis']['n_perm']} permutations); single-threaded, no "
              "intermediate progress", flush=True)
        with threadpool_limits(limits=1):
            report = run(args)
        result.update(status="estimated", comparisons=report["comparisons"],
                      channels=report["channels"],
                      same_direction_both=all(v["gain_bits"] > 0 for v in report["comparisons"].values()))
    write_json(directory / "replication_result.json", result)
    print(f"\nReplication status: {result['status']}  ({fresh})")
    if fresh != FRESH:
        print(f"CAVEAT: {REUSE_CAVEAT}")
    for row in behavior["by_stratum_arm"]:
        print(f"  {row['stratum']:<22} {row['arm']:<6} citations "
              f"{row['citations']}/{row['n_followups']}; missing {row['missing_followups']}")
    print("Positive forecasting gains do not establish suppression or a completed-trace gap.")


def release_stage_gpu(model_keys):
    """Drop this stage's cached models without importing/initializing CUDA."""
    if not model_keys:
        return
    module = sys.modules.get("e46_pretest_cue")
    cache = getattr(module, "_MODEL_CACHE", {})
    for key in model_keys:
        cache.pop(key, None)
    gc.collect()
    torch = sys.modules.get("torch")
    if torch is None:
        return
    try:
        if torch.cuda.is_initialized():
            torch.cuda.empty_cache()
    except Exception as exc:
        # A damaged CUDA context must not hide the original generation error.
        print(f"E46 cleanup warning: could not empty CUDA cache: {exc}", file=sys.stderr)


def clear_error_frames(error):
    pending, seen = [error], set()
    while pending:
        exc = pending.pop()
        if id(exc) in seen:
            continue
        seen.add(id(exc))
        traceback.clear_frames(exc.__traceback__)
        pending.extend(e for e in (exc.__cause__, exc.__context__) if e is not None)
        pending.extend(getattr(exc, "exceptions", ()))


def stage(args):
    model_keys = set()
    try:
        return _stage(args, model_keys)
    except BaseException as exc:
        # Notebook exception history otherwise retains model/tensor locals.
        clear_error_frames(exc)
        raise
    finally:
        release_stage_gpu(model_keys)


def _stage(args, model_keys):
    d = load_design(args.design)
    directory = Path(args.output_dir).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    for parent in PARENTS[args.stage]:
        check_stage(d, directory, parent)
    if args.stage == "analyze":
        return analyze(d, directory)
    path = directory / FILES[args.stage]
    receipt = directory / f"{args.stage}_receipt.json"
    if path.exists() or receipt.exists():
        raise ValueError(f"Stage output exists: {path}. No overwrite; run the next stage.")
    if args.stage == "initial":
        model_keys.add((d["hf"], d["dtype"], args.device_map, d["hf_revision"]))
        run_initial(d, path, args.device_map)
    else:
        if args.stage == "followup":
            cmd = [sys.executable, str(HERE / "e46_followup.py"),
                   "--traces", str(directory / FILES["initial"]), "--out", str(path),
                   "--max-new-tokens", str(d["why_max_new_tokens"])]
        else:
            parent = "initial" if args.stage == "capture" else "followup"
            flag = "--traces" if parent == "initial" else "--followups"
            cmd = [sys.executable, str(HERE / "e46_capture_residuals.py"), flag,
                   str(directory / FILES[parent]), "--out", str(path),
                   "--all-layers", "--arms", "cued,uncued", "--device", args.device]
            if parent == "initial":
                cmd.append("--no-title-scores")
        cmd += ["--dtype", d["dtype"], "--device-map", args.device_map]
        print(f"Running {args.stage} via {Path(cmd[1]).name}; "
              "its own n/total progress follows", flush=True)
        subprocess.run(cmd, check=True)
    meta = read_pickle(path)["metadata"]
    if meta.get("replication_design_id") != d["design_id"] or meta.get("hf_revision") != d["hf_revision"]:
        raise ValueError("Stage lost design/model provenance; not sealing output")
    write_json(receipt, {"design_id": d["design_id"], "sha256": file_hash(path),
                         "parents": {p: file_hash(directory / FILES[p]) for p in PARENTS[args.stage]},
                         "device": args.device, "device_map": args.device_map,
                         "runtime_versions": runtime_versions()})
    print(f"Sealed {args.stage}: {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    f = sub.add_parser("freeze", help="Choose fresh items and freeze before generation")
    f.add_argument("--bbh-json", required=True, help="Local BBH task JSON (never downloaded implicitly)")
    f.add_argument("--cot-file", required=True)
    f.add_argument("--exclude", nargs="+", default=[], help="All original discovery trace PKLs; glob patterns allowed. Required unless --no-exclusions")
    f.add_argument("--no-exclusions", action="store_true", help="Declare on the record that this design excludes no prior items (new substrate, or a deliberate item-matched cross-model comparison)")
    f.add_argument("--model", required=True, help="Display label, separate conditions for different checkpoints")
    f.add_argument("--hf", required=True, help="Exact HF model repository ID")
    f.add_argument("--revision", required=True, help="Full 40-character HF commit hash for model and tokenizer, or 'auto' to resolve it from the Hub once and store the result")
    f.add_argument("--task", default="movie_recommendation", choices=["movie_recommendation"])
    f.add_argument("--n", type=int, required=True, help="Fixed unique-item count, not arm count")
    f.add_argument("--item-seed", type=int, default=1046)
    f.add_argument("--batch-size", type=int, default=40, help="Frozen adaptive-assignment batch size")
    f.add_argument("--max-new-tokens", type=int, default=512)
    f.add_argument("--why-max-new-tokens", type=int, default=256)
    f.add_argument("--dtype", default="bfloat16", choices=["float32", "float16", "bfloat16"])
    f.add_argument("--half-width", type=float, default=.10)
    f.add_argument("--citation-rate", type=float, default=.20)
    f.add_argument("--flipper-rate", type=float, default=.30)
    f.add_argument("--n-perm", type=int, default=99)
    f.add_argument("--out", required=True)
    f.set_defaults(func=freeze)
    v = sub.add_parser("verify", help="Check design integrity and frozen code")
    v.add_argument("--design", required=True)
    v.set_defaults(func=lambda a: print("Verified design", load_design(a.design)["design_id"]))
    r = sub.add_parser("run", help="Run one frozen stage, without outcome-dependent overrides")
    r.add_argument("--design", required=True)
    r.add_argument("--output-dir", required=True)
    r.add_argument("--stage", choices=list(PARENTS), required=True)
    r.add_argument("--device", default="cuda")
    r.add_argument("--device-map", default="auto")
    r.set_defaults(func=stage)
    args = parser.parse_args()
    failure = None
    try:
        args.func(args)
    except (ValueError, FileNotFoundError) as exc:
        failure = str(exc)
    # Outside the handler on purpose: exiting from inside it chains SystemExit
    # to the original error, and IPython's traceback formatter fails on that
    # chain under Python 3.13, hiding the message this line exists to print.
    if failure is not None:
        parser.exit(2, f"E46 replication: {failure}\n")


if __name__ == "__main__":
    main()
