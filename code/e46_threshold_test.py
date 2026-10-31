#!/usr/bin/env python3
"""Audit exploratory confidence and cue-agreement associations in E46.

Reproduces every number in sections/verbalization_threshold.tex. Read-only.

    python code/e46_threshold_test.py --run e46/qwen_fresh_01

THE HYPOTHESIS: verbalization is gated by confidence in the influence. A hedged
cue on an uncertain item influences behaviour but falls below the threshold for
being reported.

It has three legs. This script checks each and reports which are testable:

  1. CROSS-TASK   cued behaviour on a high-confidence substrate vs a
                  low-confidence one. Checks whether more than one substrate
                  exists in the cue data at all.
  2. CROSS-CUE    variation in how hedged the cue is. Checks whether cue_type
                  and phrasing vary.
  3. LENS PROXY  association of stored option-lens scores with later mention.
                  These are NOT calibrated answer or causal-attribution confidence.

Disagreement is defined directly from the parsed uncued answer and cue letter,
not from outcome strata. Selected-stratum contrasts remain sensitivity analyses.
Associations do not identify a threshold, internal conflict, or mediation.

Exploratory throughout. The conflict contrast was selected after inspecting the
stratum table and is reported with that provenance attached.
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import math
import hashlib
import re
from pathlib import Path
import pickle
import sys

import numpy as np

LETTERS = "ABCD"


def wilson(k, n):
    if not n:
        return None
    z = 1.96
    centre = (k / n + z * z / (2 * n)) / (1 + z * z / n)
    radius = z * math.sqrt(k / n * (1 - k / n) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0., centre - radius), min(1., centre + radius)


def lens_stats(option_logits):
    """p_max, option entropy and the full distribution from a lens readout."""
    if not isinstance(option_logits, dict):
        return None
    if any(c not in option_logits for c in LETTERS):
        return None
    v = np.array([option_logits[c] for c in LETTERS], float)
    if not np.isfinite(v).all():
        return None
    e = np.exp(v - v.max())
    p = e / e.sum()
    ordered = np.sort(p)[::-1]
    return {"p": p, "pmax": float(ordered[0]),
            "entropy": float(-(p * np.log2(np.clip(p, 1e-12, 1))).sum())}


def leg_cross_task(run, all_runs):
    """Leg 1: is there more than one substrate among the cue experiments?"""
    substrates = collections.Counter()
    inventory = []
    for path in all_runs:
        with open(path, "rb") as f:
            blob = pickle.load(f)
        if blob.get("kind") != "e46_pretest_traces":
            continue
        meta = blob["metadata"]
        substrates[(meta.get("substrate"), meta.get("cue_type"))] += 1
        raw = [r for rs in blob["raw_outputs"].values() for r in rs]
        inventory.append({"path": str(path), "item_records": len({r['index'] for r in raw}),
                          "arm_records": len(raw)})
    return {"has_multiple_substrates": len({s for s, _ in substrates}) > 1,
            "substrate_cue_pairs": {f"{s}/{c}": n for (s, c), n in substrates.items()},
            "inventory": inventory,
            "note": ("Inventory covers only the supplied/local cached files, not all experiments ever run. "
                     "Multiple substrates alone do not establish a matched comparison. "
                     "Arithmetic results from the no-cue paradigm are not a "
                     "comparison: substrate, cue, target and reader all differ.")}


def leg_cross_cue(traces):
    """Leg 2: does cue assertiveness vary?"""
    rows = [r for rs in traces["raw_outputs"].values() for r in rs if r['condition'] == 'cued']
    kinds = collections.Counter(r["item"].get("cue_type") for r in rows)
    # Inspect cue sentences, not just cue_type; a type may contain many wordings.
    sentences = collections.Counter()
    for r in rows:
        query = r['prompt'].rsplit('\nQ:', 1)[-1]
        for line in query.splitlines():
            if 'Stanford professor with expertise in this area' in line:
                sentences[re.sub(r'\([A-D]\)', '(OPTION)', line.strip())] += 1
    return {"assertiveness_manipulation_established": False,
            "n_cued_prompts": len(rows), "authority_wording_templates": dict(sentences),
            "cue_types": dict(kinds),
            "note": "Cue type is not assertiveness. These counts do not substitute for a randomized assertiveness factor."}


def auc_summary(y, scores, n_boot=2000, seed=46):
    """One record per source item; stratified bootstrap conditions on label counts."""
    from scipy.stats import rankdata
    y, scores = np.asarray(y, int), np.asarray(scores, float)
    if n_boot < 1 or len(y) != len(scores) or not np.isin(y, [0, 1]).all():
        raise ValueError('Need matched binary labels/scores and a positive resample count')
    if not len(y) or len(np.unique(y)) < 2 or not np.isfinite(scores).all():
        return {"auc": None, "bootstrap95": None, "permutation_p_two_sided": None}
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    def score(labels, values):
        ranks = rankdata(values)
        n1 = int(labels.sum())
        return float((ranks[labels == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * (len(labels) - n1)))
    observed = score(y, scores)
    rng = np.random.default_rng(seed)
    boot = []
    ranks = rankdata(scores)
    extreme = 0
    for _ in range(n_boot):
        indices = np.concatenate((rng.choice(pos, len(pos)), rng.choice(neg, len(neg))))
        boot.append(score(y[indices], scores[indices]))
        perm = rng.permutation(y)
        null = (ranks[perm == 1].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))
        extreme += abs(null - .5) >= abs(observed - .5) - 1e-12
    return {"auc": observed, "bootstrap95": np.quantile(boot, [.025, .975]).tolist(),
            "permutation_p_two_sided": (extreme + 1) / (n_boot + 1)}


def leg_confidence(records):
    """Exploratory lens associations, with no calibration or threshold claim."""
    y = np.array([r["mention2"] for r in records], int)
    out = {"n": len(records), "n_mention": int(y.sum())}
    if len(np.unique(y)) < 2:
        out["status"] = "degenerate"
        return out
    out["auc"] = {}
    out["auc_uncertainty"] = {}
    for key in ("pmax_uncued", "entropy_uncued", "cue_prob_cued", "cue_lift"):
        result = auc_summary(y, [r[key] for r in records])
        out["auc"][key] = result['auc']
        out["auc_uncertainty"][key] = result
    flippers = [r for r in records if r["stratum"] == "flipper"]
    yf = np.array([r["mention2"] for r in flippers], int)
    if len(np.unique(yf)) > 1:
        out["auc_flippers"] = {
            key: auc_summary(yf, [r[key] for r in flippers])
            for key in ("pmax_uncued", "cue_prob_cued", "cue_lift")}
    # Similar group means neither imply narrow individual spread nor exclude confounding.
    by = collections.defaultdict(list)
    for r in records:
        by[r["stratum"]].append(r)
    out["by_stratum"] = {
        st: {"n": len(g),
             "pmax_uncued": float(np.mean([x["pmax_uncued"] for x in g])),
             "entropy_uncued": float(np.mean([x["entropy_uncued"] for x in g])),
             "mention2_rate": float(np.mean([x["mention2"] for x in g]))}
        for st, g in sorted(by.items(), key=lambda kv: -len(kv[1]))}
    spread = [v["pmax_uncued"] for v in out["by_stratum"].values()]
    out["confidence_spread"] = float(max(spread) - min(spread))
    out['individual_pmax_quantiles'] = np.quantile([r['pmax_uncued'] for r in records],
                                                  [0, .1, .5, .9, 1]).tolist()
    parsed = [r for r in records if r.get('uncued_answer') in tuple(LETTERS)]
    out['prompt_lens_diagnostic'] = {
        'n_parsed_uncued': len(parsed),
        'top_letter_counts': dict(collections.Counter(r['lens_top_uncued'] for r in parsed)),
        'matches_eventual_answer': sum(r['lens_top_uncued'] == r['uncued_answer'] for r in parsed),
        'scope': 'Four-way softmax of maxima over spelling logits from an intermediate residual. '
                 'Prompt anchor predicts the next token, not a final answer or confidence in causal influence.'}
    out['uncertainty_scope'] = 'Stratified item bootstrap; 2000 permutations, two-sided, exploratory and unadjusted across measures. No equivalence test.'
    out['by_cue_letter'] = {
        letter: {'n': len(group), 'mentions': sum(r['mention2'] for r in group),
                 'mean_cue_prob_cued': float(np.mean([r['cue_prob_cued'] for r in group]))}
        for letter in LETTERS if (group := [r for r in records if r['cue_letter'] == letter])}
    return out


def contrast(conflict, agree):
    from scipy.stats import fisher_exact
    ck = sum(r["mention2"] for r in conflict)
    ak = sum(r["mention2"] for r in agree)
    odds, p = fisher_exact([[ck, len(conflict) - ck], [ak, len(agree) - ak]])
    return {"conflict_strata": {"k": int(ck), "n": len(conflict),
                                "rate": ck / len(conflict) if conflict else None,
                                "wilson": wilson(ck, len(conflict))},
            "agreement_strata": {"k": int(ak), "n": len(agree),
                                 "rate": ak / len(agree) if agree else None,
                                 "wilson": wilson(ak, len(agree))},
            "fisher_p": float(p) if conflict and agree else None,
            "odds_ratio": float(odds) if np.isfinite(odds) else None,
            "odds_ratio_infinite": bool(np.isinf(odds))}


def rival_conflict(records):
    parsed = [r for r in records if r['uncued_answer'] in tuple(LETTERS)]
    disagree = [r for r in parsed if r['uncued_answer'] != r['cue_letter']]
    agree = [r for r in parsed if r['uncued_answer'] == r['cue_letter']]
    selected = [r for r in records if r['stratum'] in ('flipper', 'resister')]
    endorsed = [r for r in records if r['stratum'] in ('already_cue_correct', 'already_cue_wrong')]
    return {"all_parsed_prior_disagreement": contrast(disagree, agree),
            "selected_flipper_resister": contrast(selected, endorsed),
            "flipper_only": contrast([r for r in selected if r['stratum'] == 'flipper'], endorsed),
            "excluded_unparsed": len(records) - len(parsed),
            "provenance": "Post-hoc, unadjusted associations. Disagreement is observed answer/cue mismatch, not measured internal conflict. Outcome strata are not randomized groups."}


def resolve_reviewed_rows(run, reviewed_rows=None):
    run = Path(run).expanduser().resolve()
    if not run.is_dir():
        raise ValueError(f"E46 run directory does not exist: {run}; check --run")
    pattern = "*_availability_human_approved_rows.csv"
    if reviewed_rows is not None:
        approved = Path(reviewed_rows).expanduser().resolve()
        if not approved.is_file():
            raise ValueError(f"--reviewed-rows file does not exist: {approved}. "
                             "Copy the reviewed report to this machine or correct the path.")
    else:
        candidates = sorted(run.glob(pattern))
        if len(candidates) != 1:
            alternatives = candidates or sorted(run.glob("*_rows.csv"))
            listing = "\n".join(f"  {p}" for p in alternatives) or "  (none)"
            raise ValueError(
                f"Found {len(candidates)} reviewed reports matching {pattern!r} in {run}.\n"
                f"CSV candidates (not automatically selected):\n{listing}\n"
                "Specify --reviewed-rows /path/to/reviewed_report_rows.csv. "
                "Use the *_rows.csv produced by e46_availability.py report with reviewed "
                "annotations, not the annotations/review CSV itself. If running on Colab, "
                "copy that report to Drive first. No generation rerun is needed.")
        approved = candidates[0]
    required = {'audit_id', 'content_sha256', 'index', 'arm', 'branch',
                'response_available', 'review_mention', 'review_attribution'}
    with approved.open(newline="") as f:
        fields = set(csv.DictReader(f).fieldnames or [])
    missing = required - fields
    if missing:
        raise ValueError(
            f"Not a reviewed report detail CSV: {approved}; missing columns: "
            f"{', '.join(sorted(missing))}. Use the *_rows.csv produced by "
            "e46_availability.py report with reviewed annotations, not the annotations CSV.")
    return approved


def collect(run, reviewed_rows=None):
    run = Path(run)
    approved = resolve_reviewed_rows(run, reviewed_rows)
    with open(run / "traces.pkl", "rb") as f:
        traces = pickle.load(f)
    with open(run / "residuals_all_layers.pkl", "rb") as f:
        resid = pickle.load(f)
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from e46_gap import build_labels
    labels, _ = build_labels(str(run / "traces.pkl"))

    with open(approved, newline="") as f:
        selected = [r for r in csv.DictReader(f) if r['branch'] == 'why' and r['arm'] == 'cued']
    why = {int(r['index']): r for r in selected}
    if len(why) != len(selected):
        raise ValueError("Duplicate cued why item")
    # Join to the sealed response cache by content, not only by local item number.
    from e46_availability import analyze, load
    details, _, _ = analyze(load(run / 'availability_responses.pkl'))
    expected = {r['audit_id']: r['content_sha256'] for r in details}
    if any(expected.get(r['audit_id']) != r['content_sha256'] for r in selected):
        raise ValueError('Reviewed row fingerprint mismatch')

    lens = {(r["index"], r["arm"], r["anchor"]): r.get("option_logits")
            for r in resid["rows"]}
    if len(lens) != len(resid['rows']):
        raise ValueError('Duplicate residual checkpoint')
    by_index = {r['index']: r for r in labels}
    for r in resid['rows']:
        label = by_index.get(r['index'])
        if label is None or any(r.get(k) != label.get(k) for k in
                                ('cue_letter', 'uncued_answer', 'cued_answer', 'stratum')):
            raise ValueError('Residual labels do not match the supplied original traces')
    records = []
    for label in labels:
        i = label["index"]
        uncued = lens_stats(lens.get((i, "uncued", "prompt")))
        cued = lens_stats(lens.get((i, "cued", "prompt")))
        if uncued is None or cued is None or i not in why:
            continue
        if why[i]['review_mention'] not in ('0', '1') or why[i]['response_available'].lower() != 'true':
            continue
        cue_i = LETTERS.index(label["cue_letter"])
        records.append({
            "index": i, "stratum": label["stratum"],
            "uncued_answer": label['uncued_answer'] or '', "cue_letter": label['cue_letter'],
            "lens_top_uncued": LETTERS[int(np.argmax(uncued['p']))],
            "pmax_uncued": uncued["pmax"], "entropy_uncued": uncued["entropy"],
            "cue_prob_uncued": float(uncued["p"][cue_i]),
            "cue_prob_cued": float(cued["p"][cue_i]),
            "cue_lift": float(cued["p"][cue_i] - uncued["p"][cue_i]),
            "mention1": bool(label["mentioned"]),
            "mention2": why[i]["review_mention"] == "1",
            "attrib2": (why[i]["review_attribution"] == "1"
                        if why[i]["review_attribution"] in ('0', '1') else None)})
    return traces, labels, records


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="an E46 replication directory")
    ap.add_argument("--all-cue-runs", nargs="+", default=None,
                    help="every e46 traces pkl, for the cross-task check")
    ap.add_argument("--json", default=None)
    ap.add_argument("--reviewed-rows", help="Explicit reviewed *_rows.csv snapshot; no silent merge of later edits")
    a = ap.parse_args(argv)

    try:
        report_path = resolve_reviewed_rows(a.run, a.reviewed_rows)
    except ValueError as exc:
        ap.error(str(exc))
    traces, labels, records = collect(a.run, report_path)
    runs = a.all_cue_runs or [str(p) for p in
                              Path(a.run).parent.glob("**/*traces*.pkl")]
    report = {
        "run": str(a.run), "n_items": len(labels), "n_scored": len(records),
        "reviewed_rows": str(report_path),
        "reviewed_rows_sha256": hashlib.sha256(report_path.read_bytes()).hexdigest(),
        "leg1_cross_task": leg_cross_task(a.run, runs),
        "leg2_cross_cue": leg_cross_cue(traces),
        "leg3_confidence": leg_confidence(records),
        "rival_conflict_account": rival_conflict(records),
        "turn1_mention_rate": float(np.mean([r["mentioned"] for r in labels])) if labels else None,
        "n_excluded_from_lens_analysis": len(labels) - len(records),
        "scope": "Exploratory associations, not a calibrated confidence or threshold test. Labels are the named saved report snapshot.",
    }

    print(f"items {report['n_items']}  scored {report['n_scored']}  "
          f"turn-1 mention rate {report['turn1_mention_rate']}\n")
    for leg in ("leg1_cross_task", "leg2_cross_cue"):
        r = report[leg]
        print(f"{leg}:")
        for k, v in r.items():
            print(f"    {k}: {v}")
    conf = report["leg3_confidence"]
    print(f"\nleg3_confidence: lens-proxy association with turn-2 mention (not calibrated confidence)")
    print(f"    confidence spread across strata (p_max): "
          f"{conf.get('confidence_spread')}")
    for k, v in conf.get("auc", {}).items():
        uncertainty = conf['auc_uncertainty'][k]
        print(f"    AUC({k:16s}) = {v:.3f}  bootstrap95={uncertainty['bootstrap95']} "
              f"permutation_p={uncertainty['permutation_p_two_sided']:.4f}")
    print('    Lens diagnostic:', conf.get('prompt_lens_diagnostic'))
    print('    Individual pmax quantiles:', conf.get('individual_pmax_quantiles'))
    print(f"\n{'stratum':22s}{'n':>4}{'mention2':>10}{'pmax':>8}")
    for st, v in conf.get("by_stratum", {}).items():
        print(f"{st:22s}{v['n']:>4}{v['mention2_rate']:>10.3f}{v['pmax_uncued']:>8.3f}")
    riv = report["rival_conflict_account"]
    print("\nObserved prior disagreement (not a measured conflict mechanism)")
    for name in ('all_parsed_prior_disagreement', 'selected_flipper_resister', 'flipper_only'):
        print('   ', name, riv[name])
    print(f"    ({riv['provenance']})")

    if a.json:
        Path(a.json).parent.mkdir(parents=True, exist_ok=True)
        with open(a.json, "x") as f:
            json.dump(report, f, indent=2, allow_nan=False, default=float)
        print(f"\nWrote {a.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
