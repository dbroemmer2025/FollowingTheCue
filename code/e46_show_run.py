#!/usr/bin/env python3
# =============================================================================
# E46 / E47 -- ONE-PAGE VIEW OF A WHOLE RUN
#
# Reads the artifacts produced by either protocol and prints one dashboard.
# E46 includes strata, availability, margins, and branched follow-ups. E47
# includes its factorial behavior report, frozen design, capture manifest,
# bounded readers, controls, causal patching, and neutral why turn.
#
# STRICTLY READ-ONLY. It opens files, prints, and exits. It imports nothing
# from the analysis modules, so it cannot be broken by -- or break -- a change
# to how any of them compute. Every field is read with `.get`, so a renamed or
# newly added key degrades to "--" instead of a traceback: these artefacts are
# actively being revised and a viewer that dies on an unfamiliar schema is
# worse than useless.
#
# WHAT IT DELIBERATELY DOES NOT DO
#   * rerun analyses or write artifacts. The one display-only fallback is to
#     aggregate approved E47 review labels when report.json predates manual
#     review. Paired contrasts still require a reviewed/rescored report.
#   * interpret. Where a control decides whether a number is readable, the
#     control is printed NEXT TO it rather than folded into a verdict.
#
#     python e46_show_run.py --dir e46
#     python e46_show_run.py --dir e46 --model qwen2.5_7b_quota
#     python e46_show_run.py --dir e46 --section strata,followup
#     python e46_show_run.py --run-dir e47/qwen_run
#     python e46_show_run.py --selftest
# =============================================================================
from __future__ import annotations

import argparse
import csv
import datetime
import glob
import json
import os
import pickle
import sys
from typing import Any

W = 84
SECTIONS = ("config", "behaviour", "strata", "mentions", "provenance",
            "replication", "captures", "readers", "margins", "followup",
            "forecast", "availability", "files")


# ------------------------------------------------------------------ helpers

def rule(title: str = "", ch: str = "-") -> None:
    if title:
        print(f"\n{ch * 2} {title} " + ch * max(0, W - len(title) - 4))
    else:
        print(ch * W)


def fmt(v, nd: int = 3, dash: str = "--") -> str:
    if v is None:
        return dash
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def ci(pair, nd: int = 2) -> str:
    if not pair or len(pair) != 2 or pair[0] is None:
        return ""
    try:
        return "[{:+.{n}f},{:+.{n}f}]".format(pair[0], pair[1], n=nd)
    except (TypeError, ValueError):
        return ""


def pct(numerator, denominator, nd: int = 1) -> str:
    try:
        if numerator is None or denominator in (None, 0):
            return "--"
        return f"{100 * float(numerator) / float(denominator):+.{nd}f}%"
    except (TypeError, ValueError, ZeroDivisionError):
        return "--"


def reviewed_ratio(cell, field):
    """Format one approved LLM/manual review metric as positive/reviewed."""
    reviewed = cell.get(field + "_reviewed")
    if reviewed is None:
        return "--"
    return f"{fmt(cell.get(field + '_positive', 0))}/{fmt(reviewed)}"


E47_REVIEW_ANY_FIELDS = (
    "review_mention", "review_source_present", "review_source_correct")


def approved_review_counts(path):
    """Aggregate approved E47 labels for display without rerunning analysis."""
    cells = {}
    seen = set()
    with open(path, newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        required = {"audit_id", "branch", "condition", "review_status",
                    *E47_REVIEW_ANY_FIELDS}
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(
                f"Approved review CSV is missing columns: {sorted(missing)}")
        for row in reader:
            if row.get("review_status") != "approved":
                continue
            audit_id = row.get("audit_id") or ""
            if not audit_id:
                raise ValueError("Approved review CSV contains an empty audit_id")
            if audit_id in seen:
                raise ValueError(
                    f"Approved review CSV contains duplicate audit_id {audit_id}")
            seen.add(audit_id)
            branch = row.get("branch") or ""
            condition = row.get("condition") or ""
            if not branch or not condition:
                raise ValueError(
                    f"Approved review row {audit_id} lacks branch or condition")
            cell = cells.setdefault(f"{branch}/{condition}", {})
            field = "review_cue_acknowledgment"
            cell[field + "_reviewed"] = cell.get(field + "_reviewed", 0) + 1
            cell[field + "_positive"] = (
                cell.get(field + "_positive", 0)
                + int(any((row.get(field) or "").strip() == "1"
                          for field in E47_REVIEW_ANY_FIELDS)))
    return {"cells": cells, "approved_rows": len(seen)}


def print_table(headers, rows, indent="  "):
    if not rows:
        return
    widths = [len(h) for h in headers]
    for row in rows:
        for i, value in enumerate(row):
            widths[i] = max(widths[i], len(str(value)))
    print(indent + "  ".join(headers[i].ljust(widths[i]) for i in range(len(headers))))
    print(indent + "  ".join("-" * widths[i] for i in range(len(headers))))
    for row in rows:
        print(indent + "  ".join(str(row[i]).ljust(widths[i]) for i in range(len(headers))))


def binary_auc(y, score):
    """Tie-aware binary AUC without importing the analysis stack."""
    pairs = sorted(zip(score, y), key=lambda pair: pair[0])
    n_pos = sum(int(v == 1) for v in y)
    n_neg = len(y) - n_pos
    if not n_pos or not n_neg:
        return None
    rank_sum = 0.0
    i = 0
    while i < len(pairs):
        j = i + 1
        while j < len(pairs) and pairs[j][0] == pairs[i][0]:
            j += 1
        average_rank = ((i + 1) + j) / 2.0
        rank_sum += average_rank * sum(int(v == 1) for _, v in pairs[i:j])
        i = j
    return (rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def item_mean(rows, field):
    """Match E47's item-weighted summaries when an item has multiple rows."""
    by_item = {}
    for row in rows:
        by_item.setdefault(row["source_id"], []).append(float(row[field]))
    if not by_item:
        return None
    return sum(sum(values) / len(values) for values in by_item.values()) / len(by_item)


def oof_reader_diagnostics(path):
    """Display-only summaries from persisted OOF predictions.

    These never refit or select a reader. In particular, outcome-conditioned
    losses use readers trained on the full prespecified population.
    """
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    by_checkpoint = {}
    for row in rows:
        try:
            row["target"] = int(row["target"])
            for reader in ("prompt", "text", "state", "joint"):
                row[reader + "_p1"] = float(row[reader + "_p1"])
                row[reader + "_loss_bits"] = float(row[reader + "_loss_bits"])
            row["gap_bits"] = float(row["gap_bits"])
        except (KeyError, TypeError, ValueError):
            continue
        by_checkpoint.setdefault(row["checkpoint"], []).append(row)

    checkpoint_order = {
        "initial/prompt": 0, "initial/frac25": 1, "initial/frac50": 2,
        "initial/frac75": 3, "initial/pre_answer": 4,
        "initial/complete": 5, "why/prompt": 6, "why/frac25": 7,
        "why/frac50": 8, "why/frac75": 9, "why/complete": 10,
    }
    auc_rows = []
    outcome_rows = []
    for checkpoint, checkpoint_rows in sorted(
            by_checkpoint.items(),
            key=lambda kv: (checkpoint_order.get(kv[0], 99), kv[0])):
        for scope, selected in (
                ("all", checkpoint_rows),
                ("disagree", [r for r in checkpoint_rows
                              if r["condition"].startswith("disagree_")])):
            y = [r["target"] for r in selected]
            auc_rows.append([checkpoint, scope, len(selected)] + [
                fmt(binary_auc(y, [r[reader + "_p1"] for r in selected]))
                for reader in ("prompt", "text", "state", "joint")])

        for outcome, label in ((1, "switched_to_cue"),
                               (0, "did_not_switch")):
            # Switching is only a meaningful outcome when the cue conflicts
            # with the uncued answer. Agree arms are structurally target=0 and
            # would make the negative comparison artificially easy.
            selected = [r for r in checkpoint_rows
                        if r["condition"].startswith("disagree_")
                        and r["target"] == outcome]
            ce_text = item_mean(selected, "text_loss_bits")
            ce_state = item_mean(selected, "state_loss_bits")
            ce_joint = item_mean(selected, "joint_loss_bits")
            gap = item_mean(selected, "gap_bits")
            outcome_rows.append([
                checkpoint, label, len(selected),
                len({r["source_id"] for r in selected}),
                fmt(ce_text), fmt(ce_state), fmt(ce_joint), fmt(gap),
                pct(gap, ce_text),
            ])
    return auc_rows, outcome_rows


def load_json(path):
    try:
        with open(path) as fh:
            return json.load(fh)
    except Exception:                                          # noqa: BLE001
        return None


def load_pkl_meta(path):
    """Keep metadata/shapes only; pickle still deserializes the full cache.

    Use --skip-residuals to avoid loading large activation arrays altogether.
    """
    try:
        with open(path, "rb") as fh:
            blob = pickle.load(fh)
    except Exception:                                          # noqa: BLE001
        return None
    if not isinstance(blob, dict):
        return None
    n = None
    rows = blob.get("rows")
    if isinstance(rows, list):
        n = len(rows)
    elif isinstance(blob.get("raw_outputs"), dict):
        n = sum(len(v) for v in blob["raw_outputs"].values())
    out = {"kind": blob.get("kind"), "metadata": blob.get("metadata", {}),
           "n_rows": n, "plan_id": blob.get("plan_id")}
    for key in ("version", "design_id", "complete", "parents", "runtime",
                "layer_metadata"):
        if key in blob:
            out[key] = blob[key]
    if isinstance(rows, list):
        from collections import Counter
        item_ids = {r.get("index") for r in rows if r.get("index") is not None}
        out["n_items"] = len(item_ids) if item_ids else None
        out["conditions"] = dict(sorted(Counter(
            r.get("condition") for r in rows if r.get("condition") is not None).items()))
        out["stages"] = dict(sorted(Counter(
            r.get("stage") for r in rows if r.get("stage") is not None).items()))
        out["hit_cap"] = sum(bool(r.get("hit_cap")) for r in rows)
        out["parsed"] = sum(r.get("answer") is not None for r in rows if "answer" in r)
    for key in ("R", "R_layers"):
        arr = blob.get(key)
        if hasattr(arr, "shape"):
            out[key + "_shape"] = list(arr.shape)
            out[key + "_dtype"] = str(arr.dtype)
    if str(blob.get("kind", "")).startswith("e46_availability"):
        rows = blob.get("rows", [])
        out["n_reused_why"] = sum(bool(r.get("reused_why")) for r in rows)
        out["n_pending"] = sum("text2" not in r for r in rows)
        out["n_skipped_arms"] = len(blob.get("skipped_arms", []))
        out["n_items"] = len(blob.get("assignment", {}))
        out["questions"] = sorted({(r.get("branch"), r.get("wording"), r.get("question"))
                                   for r in rows if r.get("branch") != "why"})
    return out


def mtime(path) -> str:
    try:
        return datetime.datetime.fromtimestamp(
            os.path.getmtime(path)).strftime("%m-%d %H:%M")
    except OSError:
        return "?"


# ---------------------------------------------------------------- discovery

KINDS = (("traces", "e46_traces_{}.json"), ("traces_pkl", "e46_traces_{}.pkl"),
         ("gap", "e46_gap_{}.json"), ("margins", "e46_margins_{}.json"),
         ("answer_state", "e46_answer_state_{}.json"),
         ("resid", "e46_resid_{}.pkl"),
         ("followup", "e46_followup_{}.json"),
         ("followup_pkl", "e46_followup_{}.pkl"),
         ("furesid", "e46_furesid_{}.pkl"),
         ("followup_gap", "e46_followup_gap_{}.json"),
         ("availability_plan", "e46_availability_{}_plan.pkl"),
         ("availability_responses", "e46_availability_{}_responses.pkl"),
         ("availability", "e46_availability_{}.json"))

BUNDLE = {"design": "design.json", "traces_pkl": "traces.pkl",
          "behaviour_report": "behavior.json", "replication": "replication_result.json",
          "followup_pkl": "followups.pkl", "followup": "followup.json",
          "resid": "residuals_all_layers.pkl", "furesid": "followup_residuals_all_layers.pkl",
          "forecast": "forecast.json", "forecast_oof": "forecast_oof.csv", "margins": "margins.json",
          "followup_gap": "followup_gap.json", "availability_plan": "availability_plan.pkl",
          "availability_responses": "availability_responses.pkl", "availability": "availability.json",
          "availability_rows": "availability_rows.csv", "availability_review": "availability_review.csv",
          "capture_manifest": "capture.pkl", "e47_analysis": "analysis.json",
          "e47_controls": "controls.json", "e47_patch": "patch.json",
          "e47_causal_interchange": "causal_interchange.json",
          "e47_report": "report.json", "e47_oof": "oof_losses.csv",
          "e47_review_approved": "review_approved.csv"}

E47_BUNDLE_KEYS = ("design", "traces_pkl", "followup_pkl", "capture_manifest",
                   "e47_controls", "e47_analysis", "e47_patch", "e47_report",
                   "e47_causal_interchange",
                   "e47_active_report", "e47_target_battery",
                   "e47_target_matched_audit",
                   "e47_cue_letter_audit",
                   "e47_accuracy_gap",
                   "e47_correct_location_causal", "e47_settling_curve",
                   "e47_residual_trajectory", "e47_oof")
E47_BUNDLE_KEYS = E47_BUNDLE_KEYS + ("e47_review_approved",)
E46_BUNDLE_KEYS = tuple(
    key for key in BUNDLE
    if key not in set(E47_BUNDLE_KEYS) - {"design", "traces_pkl", "followup_pkl"})


def attach(d, key, path, skip_residuals=False, required=False):
    if not os.path.isfile(path):
        if required:
            raise ValueError(f"File not found: {path}")
        return
    d["paths"][key] = path
    if path.endswith(".csv"):
        d[key] = {"artifact_only": True}
        return
    if skip_residuals and key in ("resid", "furesid"):
        d[key] = {"metadata": {}, "not_loaded": True}
        return
    blob = load_json(path) if path.endswith(".json") else load_pkl_meta(path)
    if blob is None:
        raise ValueError(f"Could not read {path}; use the project environment and trusted caches")
    d[key] = blob


def collect_bundle(directory, design=None, availability_report=None, skip_residuals=False):
    d = {"slug": os.path.basename(os.path.abspath(directory)), "dir": directory,
         "paths": {}, "bundle": True}
    for key, name in BUNDLE.items():
        attach(d, key, os.path.join(directory, name), skip_residuals)
    if str((d.get("design") or {}).get("version", "")).startswith("e47_v"):
        candidates = ([os.path.join(directory, "report_reviewed.json")]
                      + glob.glob(os.path.join(directory, "report_rescored_*.json")))
        candidates = [p for p in candidates
                      if os.path.isfile(p) and not p.endswith('.unparsed.json')]
        if candidates:
            selected = max(candidates, key=os.path.getmtime)
            attach(d, "e47_active_report", selected)
            d["e47_report_alternatives"] = [os.path.basename(p) for p in sorted(candidates)]
        batteries = glob.glob(os.path.join(directory, "target_battery_*.json"))
        if batteries:
            attach(d, "e47_target_battery", max(batteries, key=os.path.getmtime))
        matched_audits = glob.glob(os.path.join(
            directory, "target_matched_audit_*.json"))
        if matched_audits:
            attach(d, "e47_target_matched_audit",
                   max(matched_audits, key=os.path.getmtime))
        letter_audits = glob.glob(os.path.join(
            directory, "cue_letter_audit_*.json"))
        if letter_audits:
            attach(d, "e47_cue_letter_audit",
                   max(letter_audits, key=os.path.getmtime))
        accuracy_gap = glob.glob(os.path.join(
            directory, "accuracy_gap_*.json"))
        if accuracy_gap:
            attach(d, "e47_accuracy_gap",
                   max(accuracy_gap, key=os.path.getmtime))
        correct_causal = glob.glob(os.path.join(
            directory, "correct_location_causal_*.json"))
        if correct_causal:
            attach(d, "e47_correct_location_causal",
                   max(correct_causal, key=os.path.getmtime))
        settling = [
            os.path.join(directory, "settling_curve_with_leads.json"),
            os.path.join(directory, "settling_curve.json"),
        ]
        settling = [path for path in settling if os.path.isfile(path)]
        if settling:
            # Prefer the lead-enabled fixed-population result when available.
            selected = next((path for path in settling
                             if path.endswith("settling_curve_with_leads.json")),
                            max(settling, key=os.path.getmtime))
            attach(d, "e47_settling_curve", selected)
        attach(d, "e47_residual_trajectory",
               os.path.join(directory, "residual_trajectory.json"))
        review_path = d["paths"].get("e47_review_approved")
        if review_path:
            d["e47_review_counts"] = approved_review_counts(review_path)
    if design:
        attach(d, "design", design, required=True)
    elif "design" not in d:
        attach(d, "design", os.path.join(directory, "..", "design.json"))
    if availability_report:
        attach(d, "availability", availability_report, required=True)
        prefix = os.path.splitext(availability_report)[0]
        attach(d, "availability_rows", prefix + "_rows.csv")
        attach(d, "availability_review", prefix + "_review.csv")
    else:
        # BUNDLE maps `availability` to the plain `availability.json` the
        # availability run writes, which holds NO reviewed labels. A reviewed
        # export is a differently named file, and picking one automatically
        # would silently swap an unreviewed annotation set for a reviewed one.
        # So the default stands -- but the alternatives are recorded, because
        # rendering an all-zero MANUAL LABELS table while an approved export
        # sits beside it is the least helpful of the available failures.
        d["availability_alternatives"] = sorted(
            os.path.basename(p)
            for p in glob.glob(os.path.join(directory, "*availability*.json"))
            if os.path.basename(p) != BUNDLE["availability"])
    for stage in ("initial", "followup", "capture", "capture-followup"):
        attach(d, stage + "_receipt", os.path.join(directory, stage + "_receipt.json"))
    if not d["paths"]:
        raise ValueError(f"No fixed-run artifacts found in {directory}")
    expected = (d.get("design") or {}).get("design_id")
    ids = set()
    for key, blob in d.items():
        if key in ("paths", "design") or not isinstance(blob, dict):
            continue
        value = blob.get("design_id") or (blob.get("metadata") or {}).get("replication_design_id")
        if value:
            ids.add(value)
    if expected:
        ids.add(expected)
    if len(ids) > 1:
        raise ValueError("Artifacts have different replication design IDs; do not combine runs")
    plan_ids = {b.get("plan_id") for k in ("availability", "availability_plan", "availability_responses")
                if isinstance((b := d.get(k)), dict) and b.get("plan_id")}
    if len(plan_ids) > 1:
        raise ValueError("Availability report/plan/responses have different plan IDs")
    return d


def is_e47_run(d) -> bool:
    return str((d.get("design") or {}).get("version", "")).startswith("e47_v")


def discover(directory: str) -> dict[str, list[str]]:
    """Run slug -> the files that belong to it.

    The slug is whatever sits between the known prefix and the extension, so a
    directory holding several models or several n's separates cleanly without
    the viewer needing to know the naming convention in advance.
    """
    runs: dict[str, set] = {}
    # LONGEST PREFIX WINS. `e46_followup_gap_X.json` also matches the shorter
    # `e46_followup_{}.json` pattern, and first-match order split it into a
    # phantom run called "gap_X". Sorting by prefix length makes the match
    # order independent of how KINDS happens to be written.
    ordered = sorted(KINDS, key=lambda kp: -len(kp[1].split("{}")[0]))
    for path in sorted(glob.glob(os.path.join(directory, "e46_*"))):
        base = os.path.basename(path)
        for _, pat in ordered:
            pre, suf = pat.split("{}")
            if base.startswith(pre) and base.endswith(suf):
                runs.setdefault(base[len(pre):len(base) - len(suf)],
                                set()).add(base)
                break
    return {k: sorted(v) for k, v in runs.items()}


# A replication directory names its artefacts by STAGE, not by model slug, so
# none of the KINDS patterns match and `discover` finds nothing real. What it
# finds instead is any analysis JSON that happens to sit there: an
# `e46_gap_quota_nested_v2.json` parses as a run called "quota_nested_v2".
# Detecting the layout is better than letting the user read three phantom runs
# as three models.
BUNDLE_MARKERS = ("traces.pkl", "design.json", "replication_result.json",
                  "initial_receipt.json")

# A slug is only a RUN if something generative is on disk for it. An analysis
# output alone describes a run; it is not one.
PRIMARY_KINDS = ("traces", "traces_pkl", "resid")


def looks_like_bundle(directory: str) -> bool:
    return any(os.path.exists(os.path.join(directory, name))
               for name in BUNDLE_MARKERS)


def split_runs(runs: dict[str, list[str]]) -> tuple[dict, dict]:
    """Separate real runs from slugs seen only in analysis outputs."""
    primary = {pat.split("{}")[0] for key, pat in KINDS if key in PRIMARY_KINDS}
    real, analysis_only = {}, {}
    for slug, files in runs.items():
        has = any(f.startswith(p) for f in files for p in primary)
        (real if has else analysis_only)[slug] = files
    return real, analysis_only


def collect(directory: str, slug: str, skip_residuals=False) -> dict[str, Any]:
    out = {"slug": slug, "dir": directory, "paths": {}}
    for key, pat in KINDS:
        p = os.path.join(directory, pat.format(slug))
        if not os.path.exists(p):
            continue
        attach(out, key, p, skip_residuals)
    return out


# ----------------------------------------------------------------- sections

def sec_config(d):
    t = d.get("traces") or d.get("design") or {}
    tp = (d.get("traces_pkl") or {}).get("metadata", {})
    rule("RUN", "=")
    if is_e47_run(d):
        design = d["design"]
        print(f"  protocol     E47 ({fmt(design.get('version'))})"
              f"   phase {fmt(design.get('phase'))}")
        print(f"  model        {fmt(design.get('model'))}"
              f"        hf   {fmt(design.get('hf'))}")
        print(f"  HF revision  {fmt(design.get('revision'))}")
        print(f"  task         {fmt(design.get('task'))}"
              f"   prompt mode {fmt(design.get('prompt_mode'))}"
              f"   thinking {fmt(design.get('chat_thinking'))}")
        print(f"  n            {fmt(design.get('n'))}"
              f"   seed {fmt(design.get('seed'))}"
              f"   dtype {fmt(design.get('dtype'))}")
        print(f"  caps         initial {fmt(design.get('max_new_tokens'))}"
              f"   why {fmt(design.get('why_max_new_tokens'))}")
        print(f"  design ID    {fmt(design.get('design_id'))}")
        print(f"  policy ID    {fmt(design.get('policy_id'))}")
        return
    g = lambda k: t.get(k, tp.get(k))                          # noqa: E731
    print(f"  model        {fmt(g('model'))}"
          f"        hf   {fmt(g('hf'))}")
    print(f"  HF revision  {fmt(g('hf_revision'))}")
    if not g("hf_revision"):
        print("    Exact checkpoint revision is not recorded.")
    print(f"  substrate    {fmt(g('substrate'))}"
          f"   cue  {fmt(g('cue_type'))}")
    reg = g("prompt_regime")
    print(f"  regime       {fmt(reg)}"
          + (f"   thinking {fmt(g('enable_thinking'))}"
             if reg == "think" else
             f"   fewshot_n {fmt(g('fewshot_n'))}"
             f"   cot_counterbalance {fmt(g('cot_counterbalance'))}"))
    print(f"  adaptive_cue {fmt(g('adaptive_cue'))}"
          f"   seed {fmt(g('seed'))}"
          f"   cue_letter {fmt(g('cue_letter'))}")


def sec_behaviour(d):
    if is_e47_run(d):
        report = d.get("e47_active_report") or d.get("e47_report") or {}
        if not report:
            return
        rule("E47 BEHAVIOUR / VERBALIZATION")
        source_key = "e47_active_report" if d.get("e47_active_report") else "e47_report"
        print(f"  report       {os.path.basename(d['paths'].get(source_key, 'report.json'))}")
        if d.get("e47_active_report"):
            print("  NOTE         displaying the newest reviewed/rescored report; report.json remains unchanged")
        print(f"  phase {fmt(report.get('phase'))}   planned items {fmt(report.get('planned_items'))}")
        rows = []
        review_fallback = (d.get("e47_review_counts") or {}).get("cells") or {}
        fallback_used = False
        order = ("uncued", "agree_authority", "disagree_authority",
                 "agree_sycophancy", "disagree_sycophancy",
                 "agree_tentative", "agree_assertive",
                 "disagree_tentative", "disagree_assertive")
        for stage in ("initial", "why"):
            for condition in order:
                cell_key = f"{stage}/{condition}"
                c = (report.get("cells") or {}).get(cell_key)
                if not c:
                    continue
                c = dict(c)
                fallback = review_fallback.get(cell_key) or {}
                field = "review_cue_acknowledgment"
                reviewed_key = field + "_reviewed"
                if (c.get(reviewed_key) is None
                        and fallback.get(reviewed_key) is not None):
                    c[reviewed_key] = fallback[reviewed_key]
                    c[field + "_positive"] = fallback.get(
                        field + "_positive", 0)
                    fallback_used = True
                switch = (f"{fmt(c.get('switch_to_cue'))}/{fmt(c.get('eligible_target'))}"
                          if c.get("eligible_target") is not None else "--")
                rows.append([
                    stage + "/" + condition,
                    fmt(c.get("responses")),
                    fmt(c.get("parsed")),
                    fmt(c.get("echo")),
                    fmt(c.get("correct")),
                    fmt(c.get("cue_following")),
                    switch,
                    fmt(c.get("automatic_mentions")),
                    reviewed_ratio(c, "review_cue_acknowledgment"),
                    fmt(c.get("hit_cap")),
                ])
        if fallback_used:
            print("  review labels review_approved.csv (display-only fallback; report JSON predates review)")
        print_table(
            ["stage/condition", "n", "parsed", "echo", "correct",
             "follow cue", "switch/cue", "auto mention", "review cue ack",
             "cap"],
            rows)
        print("  Automatic mention is a screen; review cue ack uses approved labels.")
        print("  review cue ack = 1 when review_mention, review_source_present, or review_source_correct is 1.")
        print("  For an approved row, blanks and uncertain components do not count as positive.")
        if fallback_used:
            print("  Paired review contrasts are not recomputed by this viewer; regenerate/rescore the report to persist them.")
        unparsed_meta = report.get("unparsed_samples_report") or {}
        unparsed_path = unparsed_meta.get("file")
        if unparsed_path:
            if not os.path.isabs(unparsed_path):
                unparsed_path = os.path.join(d["dir"], unparsed_path)
            unparsed = load_json(unparsed_path)
            print("\n  unparsed samples")
            print(f"  report       {os.path.basename(unparsed_path)}")
            if unparsed is None:
                print("  status       missing or unreadable")
            else:
                print(f"  count        {fmt(unparsed.get('count'))}"
                      f"   by condition {fmt(unparsed.get('by_condition'))}"
                      f"   by reason {fmt(unparsed.get('by_reason'))}")
                samples = unparsed.get("rows") or []
                if samples:
                    print("  index  condition             reason             tok  response")
                    for row in samples[:25]:
                        episode = " ".join(str(row.get("episode") or "").split())
                        if len(episode) > 62:
                            episode = episode[:59] + "..."
                        print(f"  {fmt(row.get('index')):>5}  "
                              f"{str(row.get('condition') or '--'):<20} "
                              f"{str(row.get('reason') or '--'):<18} "
                              f"{fmt(row.get('generated_tokens')):>4}  {episode}")
                    if len(samples) > 25:
                        print(f"  ... {len(samples) - 25} more; see the sidecar report")
        contrasts = report.get("paired_reporting_contrasts") or {}
        for key in ("initial/automatic_mention", "why/automatic_mention",
                    "initial/review_cue_acknowledgment",
                    "why/review_cue_acknowledgment"):
            if fallback_used and "/review_" in key:
                continue
            block = contrasts.get(key)
            if not block:
                continue
            if block.get("status"):
                print(f"  {key}: {block['status']}")
                continue
            values = []
            effect_order = (
                "assertiveness", "disagreement", "interaction",
                "disagreement_authority", "disagreement_sycophancy",
                "sycophancy_minus_authority_interaction")
            for effect in effect_order:
                value = block.get(effect)
                if not isinstance(value, dict):
                    continue
                values.append(
                    f"{effect} {fmt(value.get('mean'))} "
                    f"{ci(value.get('ci95'), 3)}")
            print(f"  {key}: " + "; ".join(values))
        return
    br = d.get("behaviour_report") or (d.get("replication") or {}).get("behavior") or {}
    if br:
        rule("FIXED-RUN BEHAVIOUR / COVERAGE")
        print(f"  unique items {fmt(br.get('n_unique_items'))}  original rows {fmt(br.get('n_original_rows'))}")
        print(f"  unparsed original arms {fmt(br.get('unparsed_original_arms'))}"
              f"  original cap hits {fmt(br.get('hit_cap_original_arms'))}")
        print("  stratum / arm                       why n/pairs  cites   rate       95% CI       missing cap")
        for c in br.get("by_stratum_arm", []):
            name = f"{c.get('stratum')}/{c.get('arm')}"
            print(f"  {name:<34} {fmt(c.get('n_followups')):>4}/{fmt(c.get('n_pairs')):<4}"
                  f" {fmt(c.get('citations')):>5} {fmt(c.get('citation_rate')):>7}"
                  f" {ci(c.get('wilson_95')):>18} {fmt(c.get('missing_followups')):>5}"
                  f" {fmt(c.get('hit_cap_followups')):>3}")
        print("  Missing follow-ups are not silent negatives; full missing-data bounds are in behavior.json.")
    s = (d.get("traces") or {}).get("stats") or {}
    if not s:
        return
    rule("BEHAVIOUR")
    print(f"  n pairs                {fmt(s.get('n'))}"
          f"          pair complete  {fmt(s.get('pair_complete_rate'), 2)}")
    print(f"  accuracy  uncued       {fmt(s.get('acc_uncued'), 3)}"
          f"          cued           {fmt(s.get('acc_cued'), 3)}"
          f"   (drop {fmt((s.get('acc_uncued') or 0) - (s.get('acc_cued') or 0), 3)})")
    print(f"  CUE-PULL rate          {fmt(s.get('cue_pull_rate'))}"
          f"          over n={fmt(s.get('n_pullable'))} pullable")
    print(f"    moves {fmt(s.get('n_moved'))}, to the cue "
          f"{fmt(s.get('moves_to_cue'))} "
          f"({fmt(s.get('move_to_cue_share'), 2)}); chance 0.33")
    print(f"  FLIP-pull rate         {fmt(s.get('flip_pull_rate'))}"
          f"          over n={fmt(s.get('n_flip_eligible'))} flip-eligible")
    # The three rates collapsing to one number is a real property of deficit
    # steering, not a coincidence, and is easy to miss in a table.
    tri = {round(s.get(k), 6) for k in
           ("cue_pull_rate", "follow_rate", "flip_pull_rate")
           if isinstance(s.get(k), float)}
    if len(tri) == 1:
        print("    NOTE  pull / follow / flip rates are IDENTICAL: every "
              "retargeted-off item\n          had a correct uncued answer, so "
              "the denominators coincide.")
    print(f"  answered  uncued {fmt(s.get('answer_rate_uncued'), 2)}"
          f"  cued {fmt(s.get('answer_rate_cued'), 2)}"
          f"     hit_cap {fmt(s.get('hit_cap_uncued'), 2)}"
          f"/{fmt(s.get('hit_cap_cued'), 2)}"
          f"     ran_on {fmt(s.get('ran_on_cued'), 2)}")
    print(f"  onset  value-first {fmt(s.get('onset_kind_value_frac'), 2)}"
          f"   agrees with final {fmt(s.get('onset_agrees'), 2)}")


def sec_strata(d):
    t = d.get("traces") or {}
    s = t.get("stats") or {}
    c = s.get("counts") or {}
    if not c:
        return
    proj = t.get("projection") or {}
    need = proj.get("need_items_per_stratum") or {}
    rule("STRATA")
    print(f"  {'stratum':<22}{'all':>5}{'unment':>8}{'ment':>6}"
          f"{'items to 25':>14}")
    for k in ("flipper", "wrong_to_cue", "already_cue_wrong",
              "already_cue_correct", "resister", "other"):
        v = c.get(k) or {}
        print(f"  {k:<22}{fmt(v.get('all')):>5}{fmt(v.get('unmentioned')):>8}"
              f"{fmt(v.get('mentioned')):>6}{fmt(need.get(k)):>14}")
    if (c.get("wrong_to_cue") or {}).get("all") == 0:
        print("    NOTE  wrong_to_cue empty is EXPECTED under deficit "
              "steering, not a fault:\n          every wrong uncued answer is "
              "spent on already_cue_wrong until the\n          gate closes, "
              "and the loop stops the moment it does.")
    gl = proj.get("go_line") or {}
    print(f"\n  occupancy met {fmt(proj.get('occupancy_met'))}"
          f"   fillable {fmt(proj.get('fillable_under_cap'))}"
          f"   -> {'GO' if proj.get('go') else 'NO-GO'}")
    if gl:
        print(f"  FROZEN LINE (scaled to n={fmt(gl.get('scaled_to'))}): "
              f"flipper {fmt(gl.get('flipper_unmentioned_per60'), 1)}"
              f"   pushed {fmt(gl.get('pushed_unmentioned_per60'), 1)}"
              f"   mention|moved {fmt(gl.get('mention_given_moved'), 2)}"
              f"   -> {'PASS' if gl.get('pass') else 'MISS'}")
    for n in proj.get("notes") or []:
        print(f"    * {n}")


def sec_mentions(d):
    s = (d.get("traces") or {}).get("stats") or {}
    if not s:
        return
    rule("MENTIONS (turn 1, spontaneous)")
    print(f"  mention | moved to cue   {fmt(s.get('mention_given_moved_to_cue'))}"
          f"      <- the headline silence number")
    print(f"  mention | followers      {fmt(s.get('mention_rate_followers'))}")
    print(f"  mention | all cued       {fmt(s.get('mention_rate_all_cued'))}")


def sec_provenance(d):
    s = (d.get("traces") or {}).get("stats") or {}
    rt = s.get("retarget")
    m = d.get("margins") or {}
    if not rt and not m:
        return
    rule("PROVENANCE & PROMPT HYGIENE")
    if rt:
        h = rt.get("honoured")
        print(f"  retarget  on {fmt(rt.get('n_on'))}"
              f"  off {fmt(rt.get('n_off'))}"
              f"  not retargeted {fmt(rt.get('n_not_retargeted'))}"
              f"  letter changed {fmt(rt.get('cue_letter_changed'))}")
        print(f"    direction honoured {fmt(h, 2)}"
              + ("   <- must be 1.00" if h is not None and h < 1.0 else ""))
    dl = (m.get("demo_letter_bias") or {}).get("option_logits")
    if dl is not None or m.get("demo_letters"):
        print(f"  demo answer letters {fmt(m.get('demo_letters'))}"
              + (f"   contaminated {fmt((dl or {}).get('contaminated'))}"
                 if dl else ""))
        if dl and dl.get("contaminated"):
            print("    the cue's boost depends on whether its letter matches "
                  "a demo answer;\n    counterbalancing is what removes this.")


def sec_readers(d):
    if is_e47_run(d):
        analysis = d.get("e47_analysis") or {}
        controls = d.get("e47_controls") or {}
        patch = d.get("e47_patch") or {}
        battery = d.get("e47_target_battery") or {}
        if not analysis and not controls and not patch and not battery:
            return
        rule("E47 BOUNDED READERS / CONTROLS / PATCHING")
        print(f"  analysis status  {fmt(analysis.get('status'), dash='not run')}"
              f"   primary {fmt(analysis.get('primary'))}")
        print(f"  target           {fmt(analysis.get('target'))}")
        print(f"  reader contract  {fmt(analysis.get('reader_contract'))}")
        policy = (d.get("design") or {}).get("analysis") or {}
        if policy.get("primary_anchor") == "pre_answer":
            print("  headline         initial/pre_answer prospective state-text gap")
            print("  completed trace  target-visibility / reader sanity control")
        print("\n  checkpoints")
        checkpoint_rows = []
        occupancy_notes = []
        checkpoint_order = {
            "initial/prompt": 0,
            "initial/frac25": 1,
            "initial/frac50": 2,
            "initial/frac75": 3,
            "initial/lead40": 4,
            "initial/lead20": 5,
            "initial/lead10": 6,
            "initial/lead5": 7,
            "initial/pre_answer": 8,
            "initial/complete": 9,
            "why/prompt": 10,
            "why/frac25": 11,
            "why/frac50": 12,
            "why/frac75": 13,
            "why/complete": 14,
        }
        checkpoints = analysis.get("checkpoints") or {}
        for name, cell in sorted(checkpoints.items(),
                                 key=lambda kv: (checkpoint_order.get(kv[0], 99), kv[0])):
            occ = cell.get("occupancy") or {}
            gap = cell.get("gap") or {}
            classes = occ.get("class_items")
            if occ.get("reasons"):
                occupancy_notes.append((name, ", ".join(map(str, occ["reasons"]))))
            joint_gain = cell.get("joint_gain_over_prior") or {}
            checkpoint_rows.append([
                name,
                fmt(cell.get("status")),
                fmt(occ.get("n_items")),
                fmt(classes),
                fmt(gap.get("mean")),
                pct(gap.get("mean"), cell.get("ce_text")),
                ci(gap.get("ci95"), 3),
                fmt(cell.get("ce_prior")),
                fmt(cell.get("ce_text")),
                fmt(cell.get("ce_state")),
                fmt(cell.get("ce_joint")),
                fmt(joint_gain.get("mean")),
                ci(joint_gain.get("ci95"), 3),
            ])
        print_table([
            "checkpoint", "status", "items", "classes", "G_bits", "rel_G%",
            "G_95CI", "CE_prior", "CE_text", "CE_state", "CE_joint",
            "joint_gain", "gain_95CI",
        ], checkpoint_rows)
        for name, note in occupancy_notes:
            print(f"    {name} occupancy: {note}")
        cue_rows = []
        moderation_rows = []
        for checkpoint, cell in sorted(
                checkpoints.items(),
                key=lambda kv: (checkpoint_order.get(kv[0], 99), kv[0])):
            for cue_type, values in sorted(
                    (cell.get("by_cue_type") or {}).items()):
                gap = values.get("gap") or {}
                cue_rows.append([
                    checkpoint, cue_type, fmt(values.get("n_items")),
                    fmt(values.get("class_rows")), fmt(values.get("ce_text")),
                    fmt(values.get("ce_state")), fmt(values.get("ce_joint")),
                    fmt(gap.get("mean")),
                    pct(gap.get("mean"), values.get("ce_text")),
                    ci(gap.get("ci95"), 3),
                ])
            moderation = cell.get("cue_type_gap_moderation") or {}
            difference = moderation.get("gap_difference") or {}
            if difference:
                moderation_rows.append([
                    checkpoint, fmt(moderation.get("n_items")),
                    fmt(difference.get("mean")),
                    ci(difference.get("ci95"), 3),
                ])
        if cue_rows:
            print("\n  held-out gap by cue type")
            print_table([
                "checkpoint", "cue_type", "items", "classes", "CE_text",
                "CE_state", "CE_joint", "G_bits", "rel_G%", "G_95CI",
            ], cue_rows)
        if moderation_rows:
            print("\n  paired cue-type gap moderation")
            print_table([
                "checkpoint", "items", "syc-auth G", "95CI",
            ], moderation_rows)
            print("    Positive values mean a larger held-out state-text gap "
                  "under sycophancy than authority.")
        accuracy_gap = d.get("e47_accuracy_gap") or {}
        if accuracy_gap:
            stratum_rows = []
            contrast_rows = []
            for cue_type, values in (
                    accuracy_gap.get("by_cue_type") or {}).items():
                accuracy = values.get("accuracy") or {}
                contrast = values.get("correct_minus_incorrect_gap") or {}
                correct = values.get("correct") or {}
                incorrect = values.get("incorrect") or {}
                contrast_rows.append([
                    cue_type, fmt(values.get("n_items")),
                    pct(accuracy.get("mean"), 1),
                    fmt(correct.get("n_rows")),
                    fmt(incorrect.get("n_rows")),
                    fmt((correct.get("gap") or {}).get("mean")),
                    fmt((incorrect.get("gap") or {}).get("mean")),
                    fmt(contrast.get("mean")),
                    ci(contrast.get("ci95"), 3),
                ])
                for outcome, subset in (("correct", correct),
                                        ("incorrect", incorrect)):
                    gap = subset.get("gap") or {}
                    stratum_rows.append([
                        cue_type, outcome, fmt(subset.get("n_rows")),
                        fmt(subset.get("n_items")),
                        pct((subset.get("switch_rate") or {}).get("mean"), 1),
                        fmt(subset.get("text_ce_bits")),
                        fmt(subset.get("state_ce_bits")),
                        fmt(subset.get("joint_ce_bits")),
                        fmt(gap.get("mean")),
                        pct(gap.get("mean"), subset.get("text_ce_bits")),
                        ci(gap.get("ci95"), 3),
                    ])
            print("\n  BBH correctness versus held-out gap")
            print_table([
                "cue_type", "outcome", "rows", "items", "switch%",
                "CE_text", "CE_state", "CE_joint", "G_bits", "rel_G%",
                "G_95CI",
            ], stratum_rows)
            print("\n  correctness moderation contrast")
            print_table([
                "cue_type", "items", "accuracy", "n_correct", "n_wrong",
                "G_correct", "G_wrong", "correct-wrong G", "95CI",
            ], contrast_rows)
            print("    Fixed full-population OOF readers are stratified after "
                  "evaluation; positive contrast means a larger gap on "
                  "correct responses.")
            print("    Correctness is post-treatment and coupled to cue "
                  "following; this is descriptive, not causal.")
        letter_audit = d.get("e47_cue_letter_audit") or {}
        letter_checkpoints = letter_audit.get("checkpoints") or {}
        control_rows = []
        letter_rows = []
        for checkpoint, cell in sorted(
                checkpoints.items(),
                key=lambda kv: (checkpoint_order.get(kv[0], 99), kv[0])):
            sensitivity = letter_checkpoints.get(checkpoint) or {
                "controls": cell.get("cue_letter_controls") or {},
                "by_letter": cell.get("by_cue_letter") or {},
            }
            for control, values in sorted(
                    (sensitivity.get("controls") or {}).items()):
                gains = values.get("gains") or {}
                joint_gain = gains.get("joint") or {}
                control_rows.append([
                    checkpoint, control, fmt(values.get("ce")),
                    fmt(values.get("auc")), fmt(joint_gain.get("mean")),
                    ci(joint_gain.get("ci95"), 3),
                ])
            for letter, values in sorted(
                    (sensitivity.get("by_letter") or {}).items()):
                gap = values.get("gap") or {}
                letter_rows.append([
                    checkpoint, letter, fmt(values.get("n_items")),
                    fmt(values.get("class_items")),
                    pct(values.get("positive_rate"), 1),
                    fmt(values.get("ce_cue_letter_control")),
                    fmt(values.get("ce_text")), fmt(values.get("ce_state")),
                    fmt(values.get("ce_joint")), fmt(gap.get("mean")),
                    pct(gap.get("mean"), values.get("ce_text")),
                    ci(gap.get("ci95"), 3), fmt(values.get("auc_state")),
                    fmt(values.get("auc_joint")),
                ])
        if control_rows:
            print("\n  fold-local cue-letter nuisance controls")
            print_table([
                "checkpoint", "control", "CE", "AUC", "joint gain",
                "gain 95CI",
            ], control_rows)
            print("    Rates are estimated from the other OOF folds only; "
                  "positive joint gain means the joint reader exceeds the "
                  "categorical nuisance baseline.")
        if letter_rows:
            print("\n  held-out gap by cue letter")
            print_table([
                "checkpoint", "cue", "items", "classes", "switch%",
                "CE_ctl", "CE_text", "CE_state", "CE_joint", "G_bits",
                "rel_G%", "G_95CI", "state_AUC", "joint_AUC",
            ], letter_rows)
            print("    Fixed grouped OOF predictions are stratified after "
                  "evaluation; readers are not refit by cue letter.")
            if letter_audit:
                print("    Source: signed supplemental cue-letter audit; the "
                      "frozen primary analysis is unchanged.")
        trajectory = d.get("e47_residual_trajectory") or {}
        trajectory_cells = trajectory.get("cells") or []
        trajectory_primary = trajectory.get("primary_layer")
        trajectory_target = ((trajectory.get("targets")
                              or ["switch_to_cue"])[0])
        trajectory_rows = []
        for cell in trajectory_cells:
            if (cell.get("target") != trajectory_target
                    or cell.get("stage") != "initial"
                    or cell.get("layer") != trajectory_primary):
                continue
            trajectory_rows.append([
                f"initial/{cell.get('anchor')}",
                fmt(cell.get("n_items")),
                fmt(cell.get("text_ce_bits")),
                fmt(cell.get("state_ce_bits")),
                fmt(cell.get("joint_ce_bits")),
                fmt(cell.get("conditional_gap_bits")),
                pct(cell.get("conditional_gap_bits"),
                    cell.get("text_ce_bits")),
                fmt(cell.get("state_auc")),
                fmt(cell.get("joint_auc")),
            ])
        if trajectory_rows:
            trajectory_rows.sort(
                key=lambda row: checkpoint_order.get(row[0], 99))
            print("\n  exploratory fixed-reader trajectory at primary layer")
            print_table([
                "checkpoint", "items", "CE_text", "CE_state", "CE_joint",
                "G_bits", "rel_G%", "state_AUC", "joint_AUC",
            ], trajectory_rows)
            reader = trajectory.get("reader") or {}
            print(f"    layer {fmt(trajectory_primary)}; "
                  f"text={fmt(reader.get('text'))}; "
                  f"state={fmt(reader.get('state'))}")
            print("    These are the heatmap cells. They are exploratory "
                  "fixed-reader diagnostics, not the frozen checkpoint "
                  "estimates above.")
        settling = d.get("e47_settling_curve") or {}
        settling_targets = settling.get("targets") or {}
        if settling_targets:
            print("\n  fixed-population residual settling curve")
            coverage = settling.get("coverage") or {}
            control_context = settling.get("control_context") or {}
            print(f"    artifact {os.path.basename(d.get('paths', {}).get('e47_settling_curve', ''))}  "
                  f"rows {fmt(coverage.get('kept'))}  "
                  f"population {fmt(coverage.get('population'))}  "
                  f"controls {fmt(control_context.get('status'))}")
            anchors = settling.get("anchors") or []
            layers = settling.get("layers") or []
            primary_layer = str(layers[-1]) if layers else None
            settling_rows = []
            for target_name, target_values in settling_targets.items():
                layer_values = ((target_values.get("layers") or {}).get(
                    primary_layer, {}) if primary_layer is not None else {})
                cells = layer_values.get("anchors") or {}
                for anchor in anchors:
                    cell = cells.get(anchor) or {}
                    if not cell:
                        continue
                    gain = cell.get("state_gain_over_prior") or {}
                    simultaneous = cell.get("state_gain_simultaneous") or {}
                    change = cell.get("state_gain_change_from_previous") or {}
                    sensitivity = cell.get("sensitivity_floor") or {}
                    settling_rows.append([
                        target_name, anchor, fmt(gain.get("n_items")),
                        fmt(cell.get("ce_prior")), fmt(cell.get("ce_state")),
                        fmt(gain.get("mean")), pct(cell.get("rho_state"), 1),
                        ci(gain.get("ci95"), 3),
                        ci(simultaneous.get("ci95_simultaneous"), 3),
                        fmt(change.get("mean")), ci(change.get("ci95"), 3),
                        fmt(sensitivity.get("floor_strength")),
                    ])
            print_table([
                "target", "anchor", "items", "CE_prior", "CE_state",
                "state_gain", "rho%", "gain_95CI", "simult_95CI",
                "delta_gain", "delta_95CI", "floor",
            ], settling_rows)
            print("    state_gain = CE_prior - CE_state; this is residual "
                  "decodability, not the E47 text-to-joint gap.")
            print("    delta_gain compares each anchor with the preceding "
                  "anchor; simultaneous intervals cover the anchor family.")
        outcome_cells = []
        for checkpoint, cell in checkpoints.items():
            outcome = cell.get("observed_outcome_subgroups") or {}
            for name, values in (outcome.get("groups") or {}).items():
                if values.get("status") == "no_rows":
                    continue
                gap = values.get("gap") or {}
                gain = values.get("joint_gain_over_prior") or {}
                outcome_cells.append([
                    checkpoint, name, fmt(values.get("n_items")),
                    fmt(values.get("ce_text")), fmt(values.get("ce_state")),
                    fmt(values.get("ce_joint")), fmt(gap.get("mean")),
                    pct(gap.get("mean"), values.get("ce_text")),
                    ci(gap.get("ci95"), 3), fmt(gain.get("mean")),
                    ci(gain.get("ci95"), 3),
                ])
        if outcome_cells:
            print("\n  conflicting-cue OOF losses by observed outcome")
            print_table([
                "checkpoint", "outcome", "items", "CE_text", "CE_state",
                "CE_joint", "G_bits", "rel_G%", "G_95CI",
                "joint_gain", "gain_95CI",
            ], outcome_cells)
            print("    Readers were fit on the full frozen population and "
                  "evaluated out of fold before splitting by outcome.")
            print("    These are descriptive subgroup losses; no switched-only "
                  "decoder was fit.")
        if d.get("reader_diagnostics"):
            oof_path = d.get("paths", {}).get("e47_oof")
            if not oof_path:
                print("\n  OOF reader diagnostics requested, but oof_losses.csv is missing.")
            else:
                try:
                    auc_rows, outcome_rows = oof_reader_diagnostics(oof_path)
                except (OSError, csv.Error, KeyError, ValueError) as exc:
                    print(f"\n  Could not read OOF diagnostics: {exc}")
                else:
                    print("\n  OOF ranking diagnostics (AUC)")
                    print_table([
                        "checkpoint", "scope", "rows", "prompt", "text",
                        "state", "joint",
                    ], auc_rows)
                    print("    AUC 0.5 = chance ranking; 1.0 = every switch ranks "
                          "above every non-switch.")
                    print("    The disagreement scope is the meaningful cue-conflict "
                          "comparison; all-factorial AUC also includes agree arms, "
                          "whose switch target is structurally zero.")

                    print("\n  Disagreement-only OOF losses conditional on observed outcome")
                    print_table([
                        "checkpoint", "outcome", "rows", "items", "CE_text",
                        "CE_state", "CE_joint", "G_bits", "rel_G%",
                    ], outcome_rows)
                    print("    Readers were fit on the full prespecified population, "
                          "then scored within each outcome.")
                    print("    A switched-only AUC is undefined because every selected "
                          "target equals 1; this table is descriptive, not a "
                          "switched-only decoder fit.")
        targets = battery.get("supplemental_targets") or {}
        if targets:
            print("\n  supplemental target battery")
            print(f"    schema {fmt(battery.get('schema'))}")
            target_rows = []
            condition_rows = []
            for name, cell in targets.items():
                if cell.get("kind") == "continuous":
                    gap = cell.get("gap_mse") or {}
                    family = cell.get("gap_mse_familywise") or {}
                    target_rows.append([
                        name, fmt(cell.get("status")), fmt(cell.get("n_items")),
                        fmt(cell.get("mse_prior")), fmt(cell.get("mse_text")),
                        fmt(cell.get("mse_joint")), fmt(gap.get("mean")),
                        ci(gap.get("ci95"), 3), ci(family.get("ci"), 3)])
                    for condition, values in (
                            cell.get("condition_strata") or {}).items():
                        condition_rows.append([
                            name, condition, fmt(values.get("n_items")),
                            fmt(values.get("mse_prior")),
                            fmt(values.get("mse_text")),
                            fmt(values.get("mse_joint")),
                            fmt(values.get("gap_mse"))])
                else:
                    gap = cell.get("gap") or {}
                    family = cell.get("gap_familywise") or {}
                    target_rows.append([
                        name, fmt(cell.get("status")),
                        fmt((cell.get("occupancy") or {}).get("n_items")),
                        fmt(cell.get("ce_prior")), fmt(cell.get("ce_text")),
                        fmt(cell.get("ce_joint")), fmt(gap.get("mean")),
                        ci(gap.get("ci95"), 3), ci(family.get("ci"), 3)])
            print_table(["target", "status", "items", "prior", "text", "joint",
                         "gap", "95CI", "familyCI"], target_rows)
            attribution = targets.get("causal_attribution_gap") or {}
            if attribution:
                channels = attribution.get("channels") or {}
                if channels:
                    print("\n  causal-attribution answer-free reader channels")
                    print_table(["channel", "CE_bits"], [
                        [name, fmt(value)] for name, value in channels.items()
                    ])
                    gap = attribution.get("gap") or {}
                    print("    primary answer-free gap: "
                          f"{fmt(gap.get('mean'))} bits "
                          f"{ci(gap.get('ci95'), 3)}")
                    print("    response-only is diagnostic; the matched primary "
                          "comparison conditions both readers on the prompt.")
                redacted = attribution.get("answer_content_redacted") or {}
                if redacted:
                    redacted_gap = redacted.get("gap") or {}
                    redacted_gain = redacted.get("joint_gain_over_prior") or {}
                    print("\n  answer-content-redacted pre-answer robustness")
                    print_table(
                        ["status", "items", "prior", "text", "state",
                         "joint", "gap", "G_95CI", "gain", "gain_95CI"],
                        [[fmt(redacted.get("status")),
                          fmt((redacted.get("occupancy") or {}).get("n_items")),
                          fmt(redacted.get("ce_prior")),
                          fmt(redacted.get("ce_text")),
                          fmt(redacted.get("ce_state")),
                          fmt(redacted.get("ce_joint")),
                          fmt(redacted_gap.get("mean")),
                          ci(redacted_gap.get("ci95"), 3),
                          fmt(redacted_gain.get("mean")),
                          ci(redacted_gain.get("ci95"), 3)]])
                    print("    replacement coverage: "
                          f"{fmt(redacted.get('rows_with_replacements'))} rows, "
                          f"{fmt(redacted.get('total_replacements'))} total; "
                          "Transformer redaction "
                          f"{fmt(redacted.get('transformer_included'))}")
                    print("    lexical exact-match control; semantic paraphrases "
                          "remain visible.")
                disclosure = attribution.get("direct_disclosure") or {}
                disclosure_rows = []
                for label, family in (
                        ("initial automatic mention",
                         disclosure.get("automatic_mention") or {}),
                        ("initial mention", disclosure.get("mention") or {}),
                        ("initial cue acknowledgment",
                         disclosure.get("cue_acknowledgment") or {}),
                        ("why mention", disclosure.get("why_mention") or {}),
                        ("why automatic mention",
                         disclosure.get("why_automatic_mention") or {}),
                        ("why cue acknowledgment",
                         disclosure.get("why_cue_acknowledgment") or {})):
                    for outcome, values in family.items():
                        yes = values.get("disclosed") or {}
                        no = values.get("not_disclosed") or {}
                        disclosure_rows.append([
                            label, outcome, fmt(yes.get("successes")),
                            fmt(yes.get("total")), pct(yes.get("rate"), 1),
                            pct(no.get("rate"), 1), ci(no.get("ci95"), 3),
                        ])
                if disclosure_rows:
                    print("\n  direct reviewed cue acknowledgment")
                    print_table([
                        "review label", "outcome", "disclosed", "reviewed",
                        "disclose%", "omit%", "omit_95CI",
                    ], disclosure_rows)
                    false_mention = disclosure.get("uncued_false_mention") or {}
                    print("    uncued false mention: "
                          f"{fmt(false_mention.get('successes'))}/"
                          f"{fmt(false_mention.get('total'))} "
                          f"({pct(false_mention.get('rate'), 1)})")
                    automatic_false = disclosure.get(
                        "uncued_automatic_false_mention") or {}
                    print("    uncued automatic false mention: "
                          f"{fmt(automatic_false.get('successes'))}/"
                          f"{fmt(automatic_false.get('total'))} "
                          f"({pct(automatic_false.get('rate'), 1)})")
                    persistence = disclosure.get(
                        "elicitation_persistence") or {}
                    if persistence:
                        print("\n  causal-disclosure persistence after neutral why")
                        persistence_rows = []
                        for label, key in (
                                ("initial disclosure", "initial_disclosure"),
                                ("why disclosure", "why_disclosure"),
                                ("ever disclosed", "ever_disclosed_by_end_of_why"),
                                ("new in why", "newly_disclosed_in_why"),
                                ("persistently omitted", "persistent_non_disclosure")):
                            value = persistence.get(key) or {}
                            persistence_rows.append([
                                label, fmt(value.get("successes")),
                                fmt(value.get("total")),
                                pct(value.get("rate"), 1),
                                ci(value.get("ci95"), 3)])
                        print_table(["stage", "count", "reviewed", "rate",
                                     "95CI"], persistence_rows)
                        exposure = (persistence.get(
                            "token_exposure_persistent_non_disclosure") or {})
                        why_tokens = exposure.get("why") or {}
                        combined = exposure.get("combined") or {}
                        print("    persistent omissions emitted "
                              f"{fmt(why_tokens.get('total'))} additional why "
                              "tokens; median per case "
                              f"{fmt(why_tokens.get('median'))}. Combined "
                              "initial+why median "
                              f"{fmt(combined.get('median'))} tokens.")
                        print("    This is elicitation-resistant non-disclosure, "
                              "not a monotonic-gap theorem.")
            if condition_rows:
                print("    continuous prior = fold-local agreement x assertiveness mean")
                print("\n  continuous targets by condition")
                print_table(["target", "condition", "items", "prior", "text",
                             "joint", "gap"], condition_rows)
            switched_rows = []
            for name, cell in targets.items():
                subset = cell.get("switched_case_evaluation") or {}
                if not subset:
                    continue
                gap = subset.get("gap") or subset.get("gap_mse") or {}
                switched_rows.append([
                    name, fmt(subset.get("status")), fmt(subset.get("n_rows")),
                    fmt(subset.get("n_items")),
                    fmt(subset.get("ce_text", subset.get("mse_text"))),
                    fmt(subset.get("ce_state", subset.get("mse_state"))),
                    fmt(subset.get("ce_joint", subset.get("mse_joint"))),
                    fmt(gap.get("mean")), ci(gap.get("ci95"), 3)])
            if switched_rows:
                print("\n  supplemental targets on switched disagreement rows")
                print_table(["target", "status", "rows", "items", "text",
                             "state", "joint", "gap", "95CI"],
                            switched_rows)
                print("    Full-population OOF readers scored after outcome "
                      "filtering; no switched-only refit.")
            ledger = battery.get("ledger_audit") or {}
            if ledger:
                print("\n  strict written-state ledger audit")
                print(f"    status {fmt(ledger.get('status'))}  parsed "
                      f"{fmt(ledger.get('parsed'))}/{fmt(ledger.get('rows'))} "
                      f"({fmt(ledger.get('parse_rate'))})")
                ledger_rows = []
                for condition, values in (ledger.get("by_condition") or {}).items():
                    ledger_rows.append([
                        condition, fmt(values.get("rows")),
                        fmt(values.get("parsed")),
                        fmt(values.get("ledger_correct")),
                        fmt(values.get("ledger_matches_final")),
                        fmt(values.get("ledger_correct_final_wrong"))])
                print_table(["condition", "rows", "parsed", "correct",
                             "matches_final", "correct/final_wrong"],
                            ledger_rows)
                switched = ledger.get("switched_cases") or {}
                print("    switched cases: rows " + fmt(switched.get("rows"))
                      + " parsed " + fmt(switched.get("parsed"))
                      + " ledger-correct " + fmt(switched.get("ledger_correct"))
                      + " ledger-matches-final "
                      + fmt(switched.get("ledger_matches_final")))
        correct_causal = d.get("e47_correct_location_causal") or {}
        if correct_causal:
            print("\n  correct-location causal sensitivity")
            coverage = correct_causal.get("coverage") or {}
            primary = correct_causal.get("primary_contrast") or {}
            print(f"    status {fmt(correct_causal.get('status'))}  "
                  f"direction accuracy {fmt(correct_causal.get('direction_oof_accuracy'))}  "
                  f"macro AUC {fmt(correct_causal.get('direction_oof_macro_auc'))}")
            print(f"    intervened {fmt(coverage.get('intervened_rows'))}/"
                  f"{fmt(coverage.get('eligible_switched_rows'))} switched rows  "
                  f"({fmt(coverage.get('intervened_items'))} items)  "
                  f"primary margin contrast {fmt(primary.get('mean'))} "
                  f"{ci(primary.get('ci95'), 3)}")
            probability = (correct_causal.get("primary_probability_contrast")
                           or {}).get("correct_probability_delta") or {}
            toward = (correct_causal.get("probability_summaries") or {}).get(
                "1.0/toward_correct", {})
            argmax = toward.get("correct_argmax_rate") or {}
            cue_argmax = toward.get("cue_argmax_rate") or {}
            raw_probability = toward.get("correct_probability_delta") or {}
            baseline = correct_causal.get("baseline_distribution") or {}
            baseline_correct = baseline.get("correct_probability") or {}
            baseline_cue = baseline.get("cue_probability") or {}
            if baseline_correct or baseline_cue:
                print(f"    baseline correct probability "
                      f"{fmt(baseline_correct.get('mean'))}  cue probability "
                      f"{fmt(baseline_cue.get('mean'))}")
            if probability or argmax:
                print(f"    controlled correct-probability change "
                      f"{fmt(probability.get('mean'))} "
                      f"{ci(probability.get('ci95'), 4)}  "
                      f"raw toward-correct change "
                      f"{fmt(raw_probability.get('mean'))} "
                      f"{ci(raw_probability.get('ci95'), 4)}")
                print(f"    post-patch correct argmax {fmt(argmax.get('mean'))} "
                      f"{ci(argmax.get('ci95'), 3)}  cue argmax "
                      f"{fmt(cue_argmax.get('mean'))} "
                      f"{ci(cue_argmax.get('ci95'), 3)}")
        profile = battery.get("layer_profile") or {}
        sites = battery.get("layer_sites") or []
        profiled = profile.get("targets") or {}
        if sites and profiled:
            print("\n  exploratory fixed-reader layer diagnostic (not headline estimates)")
            headers = ["target"] + [
                f"{site.get('label')}[{site.get('index')}]"
                + ("*" if site.get("primary") else "") for site in sites]
            rows = []
            for name, cells in profiled.items():
                values = []
                for site in sites:
                    value = (cells.get(str(site.get("index"))) or {}).get(
                        "relative_gap")
                    values.append("--" if value is None else f"{100 * value:+.1f}%")
                rows.append([name] + values)
            print_table(headers, rows)
            print("    * frozen primary; all cells use the fixed grouped profile reader")
            print("    continuous cells are condition-controlled; compare layers, not reader ladders")
        print(f"\n  synthetic controls  {fmt(controls.get('status'), dash='not run')}"
              f"   rows {fmt(len(controls.get('rows') or []))}")
        occ = controls.get("occupancy") or {}
        if occ.get("reasons"):
            print("    control occupancy: " + ", ".join(map(str, occ["reasons"])))
        for row in controls.get("rows") or []:
            gain = row.get("gain_over_prior") or {}
            print(f"    {fmt(row.get('placement')):<8} strength {fmt(row.get('strength'))}"
                  f" repeat {fmt(row.get('repeat'))} AUC {fmt(row.get('auc'))}"
                  f" gain {fmt(gain.get('mean'))} {ci(gain.get('ci95'), 3)}"
                  f" detected {fmt(row.get('detected'))}")
        matched = controls.get("target_matched_controls") or {}
        if matched:
            print(f"    target-matched controls: required {fmt(matched.get('required'))}"
                  f" pass {fmt(matched.get('pass'))}"
                  f" endpoint {fmt(matched.get('endpoint'))}")
            for gate in matched.get("gates") or []:
                gain = gate.get("gain_over_prior") or {}
                print(f"      {fmt(gate.get('target') or gate.get('name')):<42}"
                      f" AUC {fmt(gate.get('auc'))}"
                      f" (min {fmt(gate.get('min_auc'))})"
                      f" gain {fmt(gain.get('mean'))} {ci(gain.get('ci95'), 3)}"
                      f" detected {fmt(gate.get('detected'))}")
        corrected = d.get("e47_target_matched_audit") or {}
        if corrected:
            print(f"    corrected supplemental target-matched audit: "
                  f"pass {fmt(corrected.get('pass'))}"
                  f" endpoint {fmt(corrected.get('endpoint'))}")
            for gate in corrected.get("gates") or []:
                gain = gate.get("gain_over_prior") or {}
                visible_rule = gate.get("deterministic_visible_rule") or {}
                rule_text = (f" visible-rule AUC {fmt(visible_rule.get('auc'))}"
                             if visible_rule else "")
                print(f"      {fmt(gate.get('target') or gate.get('name')):<42}"
                      f" AUC {fmt(gate.get('auc'))}"
                      f" (min {fmt(gate.get('min_auc'))})"
                      f" gain {fmt(gain.get('mean'))} {ci(gain.get('ci95'), 3)}"
                      f" detected {fmt(gate.get('detected'))}{rule_text}")
            if corrected.get("changed_code"):
                print("      supplemental audit preserves the frozen outputs; "
                      "changed code is recorded in its signed artifact")
        print(f"  causal patch       {fmt(patch.get('status'), dash='not run')}"
              f"   rows {fmt(len(patch.get('rows') or []))}"
              f"   class items {fmt(patch.get('class_items'))}")
        causal = d.get("e47_causal_interchange") or {}
        if causal:
            coverage = causal.get("coverage") or {}
            print("\n  causal interchange")
            print(f"    protocol {fmt(causal.get('protocol', causal.get('version')))}  "
                  f"status {fmt(causal.get('status'))}  "
                  f"intervened {fmt(coverage.get('intervened_rows'))}/"
                  f"{fmt(coverage.get('eligible_rows'))}")
            if causal.get("matched_class_items"):
                print("    matched class items: "
                      f"{fmt(causal.get('matched_class_items'))}")
            causal_rows = []
            for name, value in (causal.get("class_effects") or {}).items():
                argmax = (causal.get("argmax_effects") or {}).get(name, {})
                causal_rows.append([
                    name, fmt(value.get("n_items")), fmt(value.get("mean")),
                    ci(value.get("ci95"), 3),
                    fmt(argmax.get("changed_to_counterfactual_argmax")),
                    pct(argmax.get("changed_to_counterfactual_argmax"),
                        argmax.get("n_items")),
                ])
            print_table(["direction", "items", "margin_delta", "95CI",
                         "greedy_changes", "change_rate"], causal_rows)
            for name, value in (causal.get("primary_contrasts") or {}).items():
                if value:
                    print(f"    {name}: {fmt(value.get('mean'))} "
                          f"{ci(value.get('ci95'), 3)}")
            skipped = coverage.get("skipped") or {}
            if skipped:
                print("    unmatched/skipped: " + ", ".join(
                    f"{key}={value}" for key, value in sorted(skipped.items())))
            adequacy = causal.get("adequacy") or {}
            if adequacy:
                print("    adequacy: " + ", ".join(
                    f"{key}={fmt(value)}" for key, value in adequacy.items()))
        print("  A nonpositive or unresolved G is not evidence that state contains no signal.")
        return
    g = d.get("gap")
    if not g:
        return
    rule("READER LADDER (Z = flipper vs already_cue_wrong, unmentioned)")
    print(f"  contrast {fmt(g.get('contrast'))}   n={fmt(g.get('n'))}"
          f" ({fmt(g.get('n_pos'))}/{fmt(g.get('n_neg'))})"
          f"   H(Z)={fmt(g.get('H_bits'))} bits")
    print(f"\n  {'reader':<30}{'rho':>8}{'95% CI':>18}{'n':>6}")
    for k, v in (g.get("readers") or {}).items():
        if not isinstance(v, dict):
            continue
        print(f"  {k:<30}{fmt(v.get('rho')):>8}"
              f"{ci([v.get('rho_lo'), v.get('rho_hi')], 3):>18}"
              f"{fmt(v.get('n', g.get('n'))):>6}")
    inj = g.get("injection_control") or {}
    if inj:
        print(f"\n  ADEQUACY  a spliced mention is detected by: "
              f"{fmt(inj.get('adequate_readers'))}")
        for nm, r in (inj.get("readers") or {}).items():
            print(f"    {nm:<12}rho {fmt(r.get('rho'))}"
                  f"  {ci([r.get('rho_lo'), r.get('rho_hi')], 2)}"
                  f"   detects {fmt(r.get('detects'))}")
    cap = g.get("capacity_control") or {}
    if cap:
        print(f"  CAPACITY  oracle feature rho {fmt(cap.get('rho'))}"
              f"   estimator_ok {fmt(cap.get('estimator_ok'))}")
    gp = g.get("gap") or {}
    if gp:
        print(f"\n  GAP  rho_state {fmt(gp.get('rho_state'))}"
              f" ({fmt(gp.get('best_anchor'))})"
              f"  -  rho_trace_prompt {fmt(gp.get('rho_trace_prompt'))}"
              f"  =  {fmt(gp.get('gap_trace_prompt'))}")
        print(f"    CI separated {fmt(gp.get('ci_separated'))}"
              f"   commitment gate {fmt((gp.get('commitment_gate') or {}).get('pass'))}"
              f"   susceptibility {fmt((gp.get('susceptibility_control') or {}).get('pass'))}")
    sa = g.get("state_adequacy") or {}
    if sa:
        print(f"  STATE ADEQUACY (cue letter from the residual): "
              f"adequate {fmt(sa.get('adequate'))}")
        for k, v in sa.items():
            if isinstance(v, dict):
                print(f"    {k:<10}rho {fmt(v.get('rho'))}"
                      f"  {ci([v.get('rho_lo'), v.get('rho_hi')], 2)}"
                      f"  n={fmt(v.get('n'))}")


def sec_margins(d):
    m = d.get("margins")
    if not m:
        return
    rule("OPTION MARGINS (logit lens)")
    ds = (m.get("decision_specificity") or {}).get("option_logits")
    if ds:
        print("  DECISION SPECIFICITY -- read before the deltas. A gain alone "
              "is not evidence\n  of a decision-specific shift: the cued "
              "prompt asserts a letter.")
        print(f"    {'anchor':<11}{'flipper':>9}{'acw':>8}{'diff':>8}"
              f"{'p':>8}  Holm")
        for r in ds.get("rows") or []:
            print(f"    {fmt(r.get('anchor')):<11}{fmt(r.get('flipper_gain'), 2):>9}"
                  f"{fmt(r.get('acw_gain'), 2):>8}{fmt(r.get('difference'), 2):>8}"
                  f"{fmt(r.get('p')):>8}"
                  f"  {'SIG' if r.get('holm_significant') else 'ns'}")
        print(f"    any decision-specific: "
              f"{fmt(ds.get('any_decision_specific'))}")
    for lens in m.get("lenses") or []:
        blk = (m.get("by_lens") or {}).get(lens) or {}
        if not blk:
            continue
        print(f"\n  --- {lens} ---")
        print(f"    {'anchor':<11}{'delta':>8}{'delta CI':>18}"
              f"{'lens_ok':>9}{'aband lead':>12}{'placebo':>10}{'resister':>10}")
        for a, c in blk.items():
            f = c.get("flipper") or {}
            pl = c.get("placebo_already_cue_correct") or {}
            rs = c.get("resister_rejected_cue") or {}
            print(f"    {a:<11}{fmt(f.get('delta_margin'), 2):>8}"
                  f"{ci(f.get('delta_margin_ci')):>18}"
                  f"{fmt(f.get('lens_ok')):>9}"
                  f"{fmt(f.get('cued_abandoned_lead'), 2):>12}"
                  f"{fmt(pl.get('cued_abandoned_lead'), 2):>10}"
                  f"{fmt(rs.get('cued_abandoned_lead'), 2):>10}")


def sec_followup(d):
    f = d.get("followup")
    fg = d.get("followup_gap")
    if not f and not fg:
        return
    rule("FOLLOW-UP  \"Why did you say the answer was (Y)?\"")
    if f:
        print(f"  {'stratum / arm':<32}{'n':>5}{'cites the cue':>16}")
        for k in sorted(f.get("by_cell") or {}):
            c = f["by_cell"][k]
            print(f"  {k:<32}{fmt(c.get('n')):>5}{fmt(c.get('rate')):>16}")
        print(f"\n  flippers  cued {fmt(f.get('flipper_cued_rate'))}"
              f" (n={fmt(f.get('n_flipper_cued'))})"
              f"   uncued FLOOR {fmt(f.get('flipper_uncued_rate'))}"
              f" (n={fmt(f.get('n_flipper_uncued'))})")
        print(f"  excess over the floor  {fmt(f.get('excess_over_control'))}")
    if fg:
        ver = fg.get("analysis_version")
        verdict = fg.get("mechanism_verdict")
        print(f"\n  cue-vocabulary lens at "
              f"{fmt(fg.get('headline_anchor'))}"
              + (f"   [{ver}]" if ver else "")
              + (f"   mechanism: {verdict}" if verdict else ""))
        blk = (fg.get("by_anchor") or {}).get(fg.get("headline_anchor")) or {}
        for k, c in (blk.get("cells") or {}).items():
            print(f"    {k:<30}{fmt(c.get('n')):>5}"
                  f"{fmt(c.get('mean'), 2):>9}  {ci(c.get('ci'))}")
        ln = blk.get("lens_adequacy") or {}
        if ln:
            print(f"    adequacy  cited {fmt(ln.get('cited_mean'), 2)}"
                  f" vs silent {fmt(ln.get('silent_mean'), 2)}"
                  f"   diff {fmt(ln.get('difference'), 2)} {ci(ln.get('ci'))}"
                  f"   ok {fmt(ln.get('ok'))}")
        sh = (blk.get("silent_flipper_cue_shift")
              or blk.get("silent_flipper_suppression") or {})
        if sh:
            print(f"    silent flippers, paired cued-uncued "
                  f"{fmt(sh.get('mean'), 2)} {ci(sh.get('ci'))}"
                  f"  n={fmt(sh.get('n'))}")
        pc = fg.get("lens_positive_control") or {}
        if pc:
            print(f"    positive control  while WRITING the cue "
                  f"{fmt(pc.get('mention_mean'), 2)}"
                  f"  vs q_end {fmt(pc.get('q_end_cued_mean'), 2)}"
                  f"   diff {fmt(pc.get('difference'), 2)}"
                  f"   ok {fmt(pc.get('ok'))}")


def sec_replication(d):
    design, result = d.get("design") or {}, d.get("replication") or {}
    if not design and not result:
        return
    if is_e47_run(d):
        rule("FROZEN E47 DESIGN / ARTIFACT COVERAGE")
        analysis = design.get("analysis") or {}
        print(f"  design ID       {fmt(design.get('design_id'))}")
        print(f"  assignment      {fmt(design.get('assignment'))}")
        print(f"  cue types       {fmt(design.get('cue_types') or ['authority'])}")
        print(f"  decoding        {fmt(design.get('decoding'))}")
        print(f"  freshness       {fmt(design.get('freshness'))}")
        print(f"  source pool     {fmt(design.get('pool_audit'))}")
        print(f"  primary         initial/{fmt(analysis.get('primary_anchor'))}"
              f"   layer {fmt(analysis.get('layer'))} ({fmt(analysis.get('layer_rule'))})")
        print(f"  occupancy gates min_items={fmt(analysis.get('min_items'))}"
              f" min_class_items={fmt(analysis.get('min_class_items'))}"
              f" folds={fmt(analysis.get('folds'))} bootstrap={fmt(analysis.get('bootstrap'))}")
        for key, label in (("traces_pkl", "initial traces"),
                           ("followup_pkl", "why traces"),
                           ("capture_manifest", "capture manifest")):
            artifact = d.get(key) or {}
            if not artifact:
                print(f"  {label:<18} not present")
                continue
            print(f"  {label:<18} rows {fmt(artifact.get('n_rows'))}"
                  f" items {fmt(artifact.get('n_items'))}"
                  f" complete {fmt(artifact.get('complete'))}"
                  f" caps {fmt(artifact.get('hit_cap'))}")
            if artifact.get("conditions"):
                print(f"    conditions {fmt(artifact.get('conditions'))}")
            if artifact.get("stages"):
                print(f"    stages {fmt(artifact.get('stages'))}")
        print("  Artifact design IDs are checked before sections are rendered.")
        return
    rule("FROZEN DESIGN / REPLICATION")
    print(f"  design ID {fmt(design.get('design_id', result.get('design_id')))}")
    for key in ("n", "batch_size", "item_seed", "max_new_tokens", "why_max_new_tokens", "dtype"):
        print(f"  {key:<23} {fmt(design.get(key))}")
    print(f"  source pool {fmt(design.get('pool_counts'))}")
    print(f"  stopping    {fmt(design.get('stopping'))}")
    a = design.get("analysis") or result.get("analysis") or {}
    print(f"  primary layer {fmt(a.get('layer'))}")
    print(f"  analysis seed {fmt(a.get('seed'))}  permutations {fmt(a.get('n_perm'))}")
    pp = design.get("precision_plan") or {}
    print(f"  planning: expected flippers {fmt(pp.get('expected_flippers_approx'), 1)}"
          f"  target approx {fmt(pp.get('required_flippers_approx'))} (not a power guarantee)")
    print(f"  analysis status {fmt(result.get('status'), dash='not run')}")
    if result.get("reason"):
        print(f"  reason: {result['reason']}")
    print(f"  eligible {fmt(result.get('n_eligible'))}"
          f"  excluded {fmt(result.get('excluded_from_primary'))}")
    for key in ("initial", "followup", "capture", "capture-followup"):
        receipt = d.get(key + "_receipt")
        if receipt:
            print(f"  {key:<18} receipt recorded; device={fmt(receipt.get('device'))}"
                  f" map={fmt(receipt.get('device_map'))}")
    print("  Viewer checks recorded IDs, not cache hashes; the stage runner validates receipts.")


def sec_captures(d):
    if is_e47_run(d):
        manifest = d.get("capture_manifest") or {}
        if not manifest:
            return
        rule("E47 RESIDUAL CAPTURE MANIFEST")
        if manifest.get("not_loaded"):
            print("  capture manifest present, not loaded (--skip-residuals)")
            return
        layers = manifest.get("layer_metadata") or {}
        print(f"  shard rows {fmt(manifest.get('n_rows'))}"
              f"   stages {fmt(manifest.get('stages'))}"
              f"   complete {fmt(manifest.get('complete'))}")
        print(f"  transformer blocks {fmt(layers.get('n_layers'))}"
              f"   layer sites {fmt(len(layers.get('layer_sites') or []))}"
              f"   capture {fmt(layers.get('layer_capture'))}")
        print(f"  legacy R layer-axis index {fmt(layers.get('legacy_R_layer_axis'))}")
        print("  The dashboard reads only capture.pkl, not the individual activation shards.")
        return
    if not any(d.get(k) for k in ("resid", "furesid")):
        return
    rule("RESIDUAL CAPTURES (checkpoints, not every token)")
    for key in ("resid", "furesid"):
        b = d.get(key)
        if not b:
            continue
        if b.get("not_loaded"):
            print(f"  {key}: present, not loaded (--skip-residuals)")
            continue
        m = b.get("metadata") or {}
        print(f"  {key}: rows {fmt(b.get('n_rows'))}  transformer blocks {fmt(m.get('n_layers'))}")
        print(f"    R {fmt(b.get('R_shape'))} {fmt(b.get('R_dtype'))}"
              f"   R_layers {fmt(b.get('R_layers_shape'))} {fmt(b.get('R_layers_dtype'))}")
        print(f"    hidden_states index {fmt(m.get('resid_layer'))}"
              f"   legacy R layer-axis index {fmt(m.get('legacy_R_layer_axis'))}")
        print(f"    arms {fmt(m.get('arms'))}   anchors {fmt(m.get('anchors'))}")
        print(f"    torch {fmt(m.get('capture_torch_version'))}"
              f"   transformers {fmt(m.get('capture_transformers_version'))}")
    print("  Capturing extra layers does not add independent examples or change the primary reader.")


def sec_forecast(d):
    f = d.get("forecast") or {}
    result = d.get("replication") or {}
    channels = f.get("channels") or result.get("channels") or {}
    if not channels:
        return
    rule("WHY CITATION FORECAST (cued q_end, legacy R)")
    print(f"  {fmt(f.get('analysis'))}")
    print(f"  eligible n={fmt(f.get('n', result.get('n_eligible')))}"
          f"  citations={fmt(f.get('n_positive'))}  source groups={fmt(f.get('n_source_groups'))}")
    print("  reader                CE bits    AUC    bal acc    gain/base        95% CI")
    for name, c in channels.items():
        print(f"  {name:<20} {fmt(c.get('ce_bits')):>7} {fmt(c.get('auc')):>7}"
              f" {fmt(c.get('balanced_accuracy')):>9} {fmt(c.get('gain_over_base_bits')):>12}"
              f" {ci(c.get('gain_ci'), 3):>21}")
    for name, c in (f.get("comparisons") or result.get("comparisons") or {}).items():
        print(f"  {name}: gain={fmt(c.get('gain_bits'))} bits {ci(c.get('ci'), 3)}"
              f"  shuffle p={fmt(c.get('permutation_p'))} Holm={fmt(c.get('holm_p'))}")
    print(f"  CI scope: {fmt(f.get('ci_scope'))}")
    print("  Label shuffles test no association, not equal reader performance under shared signal.")
    print("  Forecasting citation is not evidence of suppression or a completed-trace gap.")


def sec_availability(d):
    plan = d.get("availability_plan") or {}
    response = d.get("availability_responses") or {}
    report = d.get("availability") or {}
    if not plan and not response and not report:
        return
    rule("AVAILABILITY / EXPLANATION (independent branches)")
    b = response or plan
    meta = b.get("metadata") or {}
    print(f"  plan ID {fmt(b.get('plan_id', report.get('plan_id')))}")
    print(f"  items {fmt(b.get('n_items', report.get('n_attempted_unique_items')))}"
          f"  branches {fmt(b.get('n_rows', report.get('n_rows')))}"
          f"  reused why {fmt(b.get('n_reused_why'))}  pending {fmt(b.get('n_pending'))}"
          f"  skipped arms {fmt(b.get('n_skipped_arms'))}")
    print(f"  seed {fmt(meta.get('availability_seed'))}"
          f"  cap {fmt(meta.get('availability_max_new_tokens'))}"
          f"  dtype {fmt(meta.get('availability_dtype'))}")
    for branch, wording, question in b.get("questions", []):
        print(f"  {branch}/w{wording}: {question}")
    if not report:
        print("  Availability analysis not present; generate responses and run report.")
        return
    print("\n  cell                                           available  missing cap   mention")
    for key, c in sorted((report.get("by_cell") or {}).items()):
        print(f"  {key:<46} {fmt(c.get('n_available')):>4}/{fmt(c.get('n_planned')):<4}"
              f" {fmt(c.get('n_missing')):>6} {fmt(c.get('n_hit_cap')):>3}"
              f" {fmt(c.get('mention_rate')):>9}")
    print("\n  MANUAL LABELS: positives / reviewed (blank or uncertain is not negative)")
    print("  cell                                            source  correct    attr  denial changed")
    reviewed_total = 0
    for key, c in sorted((report.get("by_cell") or {}).items()):
        values = []
        for field in ("source_present", "source_correct", "attribution", "denial", "answer_changed"):
            v = (c.get("manual") or {}).get("review_" + field) or {}
            reviewed_total += int(v.get("reviewed_n", 0) or 0)
            values.append(f"{v.get('positive', 0)}/{v.get('reviewed_n', 0)}")
        print(f"  {key:<46}" + "".join(f" {v:>7}" for v in values))
    if not reviewed_total:
        alternatives = d.get("availability_alternatives") or []
        print("\n  This report carries NO reviewed labels: every cell is 0/0.")
        if alternatives:
            print("  Reviewed exports are NOT selected automatically, so that an "
                  "unreviewed and a\n  reviewed annotation set are never silently "
                  "interchanged. Other availability\n  reports in this directory:")
            for name in alternatives:
                print(f"      {name}")
            print("  Re-run with  --availability-report <dir>/<one of the above>")
        else:
            print("  No other availability report is present; run the review "
                  "and approval steps first.")
    for arm, c in (report.get("manual_source_presence_accuracy") or {}).items():
        print(f"  source-presence accuracy {arm}: {fmt(c.get('correct'))}/{fmt(c.get('reviewed_n'))}"
              f"  {fmt(c.get('accuracy'))} {ci(c.get('wilson95'))}")
    for name, c in (report.get("paired_manual") or {}).items():
        print(f"  {name}: {fmt(c.get('n'))}/{fmt(c.get('n_reviewed_recalled'))} reviewed recalled items")
    print("  Source recall can reread the prompt; influence answers are self-reports, not causal access.")
    print("  Answer-change labels need review: quoting the source's letter is not a revision.")


def sec_files(d):
    rule("FILES")
    if d.get("bundle"):
        expected = list(E47_BUNDLE_KEYS if is_e47_run(d) else E46_BUNDLE_KEYS)
    else:
        expected = [k for k, _ in KINDS]
    for key in dict.fromkeys(expected + list(d["paths"])):
        p = d["paths"].get(key)
        if p:
            extra = ""
            blob = d.get(key)
            if isinstance(blob, dict) and blob.get("n_rows") is not None:
                extra = f"  {blob['n_rows']} rows"
            print(f"  [x] {key:<16}{mtime(p)}  "
                  f"{os.path.basename(p)}{extra}")
        else:
            print(f"  [ ] {key:<16}{'':<12}  (not present)")


RENDER = {"config": sec_config, "behaviour": sec_behaviour,
          "strata": sec_strata, "mentions": sec_mentions,
          "provenance": sec_provenance, "readers": sec_readers,
          "margins": sec_margins, "followup": sec_followup,
          "replication": sec_replication, "captures": sec_captures,
          "forecast": sec_forecast, "availability": sec_availability,
          "files": sec_files}


def show(d, sections=SECTIONS):
    print("\n" + "=" * W)
    protocol = "E47" if is_e47_run(d) else "E46"
    print(f" {protocol} RUN   {d['slug']}")
    print("=" * W)
    for s in sections:
        RENDER[s](d)
    print("\n" + "=" * W)


# ----------------------------------------------------------------- selftest

def _selftest() -> int:
    import tempfile
    ok = [0, 0]

    def chk(name, cond):
        ok[1] += 1
        ok[0] += bool(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}")

    chk("missing values render as a dash, never a crash", fmt(None) == "--")
    chk("booleans read as words", fmt(True) == "yes" and fmt(False) == "no")
    chk("a CI renders signed", ci([-0.1, 0.2]) == "[-0.10,+0.20]")
    chk("and a malformed CI is dropped rather than raising",
        ci(None) == "" and ci(["x", "y"]) == "")

    tmp = tempfile.mkdtemp()
    json.dump({"model": "m1", "substrate": "movie_recommendation",
               "cue_type": "authority", "prompt_regime": "imitative",
               "stats": {"n": 100, "acc_uncued": 0.74, "acc_cued": 0.43,
                         "cue_pull_rate": 0.77, "n_pullable": 39,
                         "mention_given_moved_to_cue": 0.0,
                         "counts": {"flipper": {"all": 30, "unmentioned": 30,
                                                "mentioned": 0},
                                    "wrong_to_cue": {"all": 0,
                                                     "unmentioned": 0,
                                                     "mentioned": 0}},
                         "retarget": {"n_on": 60, "honoured": 1.0}},
               "projection": {"occupancy_met": True, "go": True,
                              "go_line": {"pass": True, "scaled_to": 60},
                              "notes": ["a note"]}},
              open(os.path.join(tmp, "e46_traces_slug_a.json"), "w"))
    json.dump({"by_cell": {"flipper/cued": {"n": 30, "rate": 0.2}},
               "flipper_cued_rate": 0.2, "flipper_uncued_rate": 0.0,
               "excess_over_control": 0.2},
              open(os.path.join(tmp, "e46_followup_slug_a.json"), "w"))
    json.dump({"stats": {}}, open(os.path.join(tmp, "e46_traces_slug_b.json"),
                                  "w"))

    runs = discover(tmp)
    chk("runs are discovered by slug, so one directory can hold several",
        set(runs) == {"slug_a", "slug_b"})
    # The phantom-run bug: a longer prefix must win over a shorter one that
    # also matches, or e46_followup_gap_X becomes a run called "gap_X".
    json.dump({}, open(os.path.join(tmp, "e46_followup_gap_slug_a.json"), "w"))
    chk("a longer prefix wins, so followup_gap is not read as a run of its "
        "own", set(discover(tmp)) == {"slug_a", "slug_b"})
    chk("and it is filed under the right run",
        "e46_followup_gap_slug_a.json" in discover(tmp)["slug_a"])
    # A bundle directory must never be read as a set of slug-named runs.
    bundle = os.path.join(tmp, "bundle")
    os.makedirs(bundle, exist_ok=True)
    open(os.path.join(bundle, "traces.pkl"), "wb").close()
    json.dump({}, open(os.path.join(bundle, "e46_gap_quota_nested_v2.json"), "w"))
    chk("a replication directory is recognised as a bundle, not as runs",
        looks_like_bundle(bundle) and not looks_like_bundle(tmp))
    real, only = split_runs(discover(bundle))
    chk("and an analysis JSON alone is never reported as a run",
        real == {} and set(only) == {"quota_nested_v2"})
    real2, only2 = split_runs(discover(tmp))
    chk("while a real slug with a trace cache still counts as a run",
        "slug_b" in real2 and "slug_b" not in only2)

    chk("and each run lists only its own files",
        len(runs["slug_a"]) == 2 and len(runs["slug_b"]) == 1)

    d = collect(tmp, "slug_a")
    chk("present artefacts are loaded", d.get("traces") and d.get("followup"))
    chk("absent ones are simply absent, not empty stubs",
        "gap" not in d and "margins" not in d)

    import io
    import contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        show(d)
    out = buf.getvalue()
    chk("the dashboard prints the headline behaviour", "0.770" in out)
    chk("and the strata table", "flipper" in out and "30" in out)
    chk("and the follow-up excess over its floor", "0.200" in out)
    chk("an empty wrong_to_cue is explained, not left looking like a bug",
        "EXPECTED under deficit steering" in out)
    chk("missing analyses are listed as missing rather than omitted silently",
        "[ ] gap" in out and "[ ] margins" in out)

    buf2 = io.StringIO()
    with contextlib.redirect_stdout(buf2):
        show(collect(tmp, "slug_b"))
    chk("a run with almost nothing in it still renders",
        "slug_b" in buf2.getvalue())

    # A file whose schema this viewer has never seen must not kill it.
    json.dump({"totally": {"unexpected": ["shape"]}},
              open(os.path.join(tmp, "e46_margins_slug_a.json"), "w"))
    buf3 = io.StringIO()
    with contextlib.redirect_stdout(buf3):
        show(collect(tmp, "slug_a"))
    chk("an unfamiliar schema degrades to blanks instead of raising -- these "
        "artefacts are actively being revised", "E46 RUN" in buf3.getvalue())
    chk("selectable sections", "STRATA" not in io_capture(d, ["config"]))

    print(f"\n  {ok[0]}/{ok[1]} passed")
    return 0 if ok[0] == ok[1] else 1


def io_capture(d, sections):
    import io
    import contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        show(d, sections)
    return buf.getvalue()


def main() -> int:
    ap = argparse.ArgumentParser(
        description="One-page view of an E46 or E47 run. Read-only.")
    ap.add_argument("--dir", default="e46", help="directory of E46 artifacts")
    ap.add_argument("--run-dir", help="fixed E46/E47 run directory with design.json and traces.pkl")
    ap.add_argument("--design", help="explicit design.json for --run-dir")
    ap.add_argument("--availability-report", help="explicit availability JSON (e.g. reviewed report); never silently selects newest")
    ap.add_argument("--skip-residuals", action="store_true", help="do not unpickle large residual arrays")
    ap.add_argument(
        "--reader-diagnostics", action="store_true",
        help=("derive AUC and outcome-conditioned loss tables from the saved "
              "E47 OOF predictions; does not refit readers"))
    ap.add_argument("--model", default=None,
                    help="run slug; omit to list what is available, or to "
                         "show it when only one run is present")
    ap.add_argument("--section", default=None,
                    help="comma-separated subset of: " + ",".join(SECTIONS))
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return _selftest()
    if a.run_dir:
        if a.model:
            ap.error("--run-dir selects one run; do not also specify --model")
        try:
            d = collect_bundle(a.run_dir, a.design, a.availability_report, a.skip_residuals)
        except ValueError as exc:
            ap.error(str(exc))
        d["reader_diagnostics"] = a.reader_diagnostics
        sections = [s.strip() for s in a.section.split(",") if s.strip()] if a.section else SECTIONS
        if any(s not in RENDER for s in sections):
            ap.error("Unknown section; choose from " + ",".join(SECTIONS))
        show(d, sections)
        return 0
    if a.design:
        ap.error("--design requires --run-dir")
    if not os.path.isdir(a.dir):
        raise SystemExit(f"{a.dir} is not a directory")
    if looks_like_bundle(a.dir):
        print(f"  {a.dir} is a fixed-run (replication) directory, whose files "
              "are named by\n  stage rather than by model slug. Showing it as "
              "one run; the equivalent\n  explicit command is --run-dir.\n")
        try:
            d = collect_bundle(a.dir, a.design, a.availability_report,
                               a.skip_residuals)
        except ValueError as exc:
            ap.error(str(exc))
        d["reader_diagnostics"] = a.reader_diagnostics
        sections = [s.strip() for s in a.section.split(",") if s.strip()] if a.section else SECTIONS
        if any(s not in RENDER for s in sections):
            ap.error("Unknown section; choose from " + ",".join(SECTIONS))
        show(d, sections)
        return 0
    runs, analysis_only = split_runs(discover(a.dir))
    if not runs:
        if analysis_only:
            raise SystemExit(
                f"no run caches in {a.dir}; only analysis outputs for: "
                + ", ".join(sorted(analysis_only))
                + ". A run needs its traces or residual cache to view.")
        raise SystemExit(f"no E46/E47 artifacts found in {a.dir}")
    slug = a.model
    if slug is None:
        if len(runs) == 1:
            slug = next(iter(runs))
        else:
            print(f"\n  {len(runs)} runs in {a.dir}:\n")
            for k, v in runs.items():
                print(f"    {k:<44}{len(v)} file(s)")
            print("\n  pass --model <slug> to view one.")
            if analysis_only:
                print("  (analysis-only, not runs: "
                      + ", ".join(sorted(analysis_only)) + ")")
            return 0
    if slug not in runs:
        raise SystemExit(
            f"no run named {slug!r} in {a.dir}. Available: "
            + ", ".join(runs) or "(none)")
    sections = SECTIONS
    if a.section:
        want = [s.strip() for s in a.section.split(",") if s.strip()]
        bad = [s for s in want if s not in RENDER]
        if bad:
            raise SystemExit(f"unknown section(s): {bad}. "
                             f"Choose from: {', '.join(SECTIONS)}")
        sections = want
    d = collect(a.dir, slug, a.skip_residuals)
    d["reader_diagnostics"] = a.reader_diagnostics
    if a.availability_report:
        attach(d, "availability", a.availability_report, required=True)
    show(d, sections)
    return 0


if __name__ == "__main__":
    sys.exit(main())
