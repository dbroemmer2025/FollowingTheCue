#!/usr/bin/env python3
"""Compare E47 behavior, verbalization, losses and gaps across runs.

Edit DEFAULT_RUN_DIRS to change the ordinary report, or replace it at runtime:

    python code/show_e47_gap_summary.py
    python code/show_e47_gap_summary.py --runs e47/run_a e47/run_b
    python code/show_e47_gap_summary.py --checkpoint initial/complete
    python code/show_e47_gap_summary.py --csv reports/e47_gap_summary.csv
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DE_RESULTS_DIR = ROOT / "reports" / "de_readers" / "results"
REVIEW_FIELDS = (
    "review_mention", "review_source_present", "review_source_correct")

# These defaults are deliberately visible and easy to edit. --runs replaces
# the complete list without requiring a code change.
from run_dirs import DEFAULT_RUN_DIRS


def fmt(value, digits=3, missing="--"):
    if value is None:
        return missing
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def percent(numerator, denominator):
    if numerator is None or denominator in (None, 0):
        return "--"
    return f"{100 * numerator / denominator:+.1f}%"


def ratio(numerator, denominator):
    if numerator is None or denominator in (None, 0):
        return "--"
    return f"{numerator}/{denominator} ({100 * numerator / denominator:.1f}%)"


def ci_text(pair):
    if not pair or len(pair) != 2:
        return "--"
    return f"[{pair[0]:+.3f},{pair[1]:+.3f}]"


def print_table(headers, rows):
    rendered = [[fmt(value) for value in row] for row in rows]
    widths = [len(header) for header in headers]
    for row in rendered:
        for index, value in enumerate(row):
            widths[index] = max(widths[index], len(value))
    print("  " + "  ".join(
        header.ljust(widths[index]) for index, header in enumerate(headers)))
    print("  " + "  ".join("-" * width for width in widths))
    for row in rendered:
        print("  " + "  ".join(
            value.ljust(widths[index]) for index, value in enumerate(row)))


def resolve_run(value):
    path = Path(value).expanduser()
    if path.exists():
        return path.resolve()
    rooted = ROOT / path
    return rooted.resolve()


def direct_effect_cell_name(run):
    """Return the direct-effect result key for a standard E47 run directory."""
    name = Path(run).name
    # Direct-effect exports deliberately omit the experimental-regime suffix.
    # Keep this conservative: an unfamiliar run simply has no matching row.
    for suffix in ("_thinking_v10", "_imitative_v10"):
        if name.endswith(suffix):
            return name[:-len(suffix)]
    return name


def load_direct_effect_result(run, results_dir, nollm=False):
    """Load the post-hoc factual-output control, if available for ``run``.

    The main E47 summary must remain usable for partial runs and for arbitrary
    ``--runs`` selections. Missing direct-effect artifacts therefore appear as
    missing values in the display rather than aborting the primary summary.
    """
    cell = direct_effect_cell_name(run)
    suffix = "nollm" if nollm else "output"
    path = Path(results_dir) / f"{cell}__{suffix}.json"
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path}: invalid direct-effect JSON: {exc}") from exc
    if payload.get("cell") != cell:
        raise ValueError(
            f"{path}: cell {payload.get('cell')!r} does not match {cell!r}")
    flag = ("transformer_in_candidate_grid" if nollm else
            "output_access_transformer_in_candidate_grid")
    if nollm and not payload.get("nollm_ablation"):
        raise ValueError(f"{path}: missing no-LLM ablation provenance")
    if not payload.get(flag):
        raise ValueError(
            f"{path}: output-access Transformer was not in the candidate grid")
    return {
        "path": str(path),
        "transformer_output": True,
        "targets": payload.get("targets") or {},
    }


def newest_behavior_report(run):
    candidates = [run / "report_reviewed.json"]
    candidates.extend(
        path for path in run.glob("report_rescored_*.json")
        if not path.name.endswith(".unparsed.json"))
    candidates = [path for path in candidates if path.is_file()]
    if candidates:
        return max(candidates,
                   key=lambda path: (path.stat().st_mtime_ns, path.name))
    path = run / "report.json"
    if not path.is_file():
        raise ValueError(f"{run}: no behavior report found")
    return path


def newest_artifact(run, pattern):
    candidates = [path for path in run.glob(pattern) if path.is_file()]
    if not candidates:
        return None
    return max(candidates,
               key=lambda path: (path.stat().st_mtime_ns, path.name))


def target_battery_acknowledgment_by_cue(run):
    """Read the broad disclosure audit, which does not require logit capture."""
    path = newest_artifact(run, "target_battery_*.json")
    if path is None:
        return {}, None
    try:
        payload = json.loads(path.read_text())
        rows = payload["supplemental_targets"]["causal_attribution_gap"][
            "direct_disclosure"]["by_cue_type"]
    except (KeyError, TypeError, json.JSONDecodeError):
        return {}, None
    counts = {}
    for cue_type, row in rows.items():
        initial = row["initial_cue_acknowledgment_among_switchers"][
            "disclosed"]
        why = row["why_cue_acknowledgment_among_switchers"]["disclosed"]
        counts[cue_type] = {
            "switchers": row["switch_to_cue"]["successes"],
            "initial_positive": initial["successes"],
            "initial_reviewed": initial["total"],
            "why_positive": why["successes"],
            "why_reviewed": why["total"],
        }
    return counts, str(path)


def switcher_acknowledgment_by_cue(run):
    """Return approved acknowledgment rates among switch-to-cue rows.

    Prefer the target battery's direct audit because it retains valid reviewed
    responses without requiring the uncued option distribution used by the
    susceptibility analysis. Current what-causes rows remain a legacy fallback.
    """
    counts, source = target_battery_acknowledgment_by_cue(run)
    if counts:
        return counts, source
    path = newest_artifact(run, "what_causes_switch_*.rows.csv")
    if path is None:
        return {}, None
    counts = {}
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = {
            "cue_type", "switch_to_cue",
            "initial_review_cue_acknowledgment",
            "why_review_cue_acknowledgment",
        }
        missing = required - set(reader.fieldnames or ())
        if missing:
            # Legacy exports contain mention/attribution fields but predate the
            # broader approved acknowledgment endpoint. Do not silently treat
            # those narrower labels as equivalent.
            return {}, str(path)
        for row in reader:
            if (row.get("switch_to_cue") or "").strip() != "1":
                continue
            cue_type = (row.get("cue_type") or "unknown").strip() or "unknown"
            cell = counts.setdefault(cue_type, {
                "switchers": 0,
                "initial_positive": 0,
                "initial_reviewed": 0,
                "why_positive": 0,
                "why_reviewed": 0,
            })
            cell["switchers"] += 1
            for stage in ("initial", "why"):
                value = (row.get(
                    f"{stage}_review_cue_acknowledgment") or "").strip()
                if value not in ("0", "1"):
                    continue
                cell[f"{stage}_reviewed"] += 1
                cell[f"{stage}_positive"] += int(value)
    return counts, str(path)


def approved_review_cells(path):
    """Match show_run's approved-label fallback at cell granularity."""
    cells = {}
    seen = set()
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = {"audit_id", "branch", "condition", "review_status",
                    *REVIEW_FIELDS}
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(
                f"{path}: approved review CSV is missing {sorted(missing)}")
        for row in reader:
            if row.get("review_status") != "approved":
                continue
            audit_id = row.get("audit_id") or ""
            if not audit_id or audit_id in seen:
                raise ValueError(f"{path}: empty or duplicate audit_id {audit_id!r}")
            seen.add(audit_id)
            branch = row.get("branch") or ""
            condition = row.get("condition") or ""
            if not branch or not condition:
                raise ValueError(
                    f"{path}: approved row {audit_id} lacks branch/condition")
            cell = cells.setdefault(f"{branch}/{condition}",
                                    {"reviewed": 0, "positive": 0})
            cell["reviewed"] += 1
            cell["positive"] += int(any(
                (row.get(field) or "").strip() == "1"
                for field in REVIEW_FIELDS))
    return cells


def total(cells, field):
    values = [cell.get(field) for cell in cells
              if cell.get(field) is not None]
    return sum(values) if values else None


def aggregate_acknowledgment(rows):
    """Pool the displayed verbalization counts by task and overall.

    Each numerator is summed only with its matching denominator. A group stays
    missing when none of its constituent runs contains that measurement.
    """
    fields = (
        ("initial_auto_mentions", "initial_responses"),
        ("initial_review_positive", "initial_reviewed"),
        ("why_auto_mentions", "why_responses"),
        ("why_review_positive", "why_reviewed"),
    )
    groups = []
    for task in sorted({row["task"] for row in rows}):
        groups.append((task, [row for row in rows if row["task"] == task]))
    groups.append(("ALL", rows))

    output = []
    for label, members in groups:
        values = []
        for numerator, denominator in fields:
            pairs = [(row.get(numerator), row.get(denominator))
                     for row in members
                     if row.get(numerator) is not None
                     and row.get(denominator) is not None]
            if not pairs:
                values.append((None, None))
            else:
                values.append((sum(pair[0] for pair in pairs),
                               sum(pair[1] for pair in pairs)))
        output.append((label, len(members), values))
    return output


def aggregate_acknowledgment_by_cue(rows):
    """Pool cue-bearing response counts by task, cue family, and stage."""
    output = []
    scopes = [(task, [row for row in rows if row["task"] == task])
              for task in sorted({row["task"] for row in rows})]
    scopes.append(("ALL", rows))
    fields = (
        ("initial_auto_mentions", "initial_responses"),
        ("initial_review_positive", "initial_reviewed"),
        ("why_auto_mentions", "why_responses"),
        ("why_review_positive", "why_reviewed"),
    )
    for scope, members in scopes:
        cue_types = sorted({cue_type for row in members
                            for cue_type in row.get("ack_by_cue", {})})
        for cue_type in cue_types:
            cells = [row["ack_by_cue"][cue_type] for row in members
                     if cue_type in row.get("ack_by_cue", {})]
            values = []
            for numerator, denominator in fields:
                pairs = [(cell.get(numerator), cell.get(denominator))
                         for cell in cells
                         if cell.get(numerator) is not None
                         and cell.get(denominator) is not None]
                values.append(
                    (sum(pair[0] for pair in pairs),
                     sum(pair[1] for pair in pairs)) if pairs
                    else (None, None))
            output.append((scope, cue_type, len(cells), values))
    return output


def aggregate_switching(rows):
    """Pool switch-to-cue outcomes by task and cue family.

    A switch is defined only on conflicting (``disagree_*``) initial-response
    cells. Numerators are always paired with their corresponding eligible-row
    denominators, so a run with a missing cue family cannot contribute a zero.
    """
    scopes = [(task, [row for row in rows if row["task"] == task])
              for task in sorted({row["task"] for row in rows})]
    scopes.append(("ALL", rows))

    by_task = []
    for scope, members in scopes:
        pairs = [(row.get("switch_to_cue"), row.get("switch_eligible"))
                 for row in members
                 if row.get("switch_to_cue") is not None
                 and row.get("switch_eligible") is not None]
        by_task.append((scope, len(pairs),
                        sum(pair[0] for pair in pairs) if pairs else None,
                        sum(pair[1] for pair in pairs) if pairs else None))

    by_cue = []
    for cue_type in ("authority", "sycophancy"):
        cells = [row.get("switch_by_cue", {}).get(cue_type) for row in rows]
        pairs = [(cell.get("switch_to_cue"), cell.get("switch_eligible"))
                 for cell in cells if cell is not None
                 and cell.get("switch_to_cue") is not None
                 and cell.get("switch_eligible") is not None]
        by_cue.append((cue_type, len(pairs),
                       sum(pair[0] for pair in pairs) if pairs else None,
                       sum(pair[1] for pair in pairs) if pairs else None))
    return by_task, by_cue


def aggregate_switcher_acknowledgment(rows):
    """Pool approved cue acknowledgments within the switcher population."""
    def summarize(cells):
        switchers = [cell.get("switchers") for cell in cells
                     if cell.get("switchers") is not None]
        initial = [(cell.get("initial_positive"), cell.get("initial_reviewed"))
                   for cell in cells
                   if cell.get("initial_positive") is not None
                   and cell.get("initial_reviewed") is not None]
        why = [(cell.get("why_positive"), cell.get("why_reviewed"))
               for cell in cells
               if cell.get("why_positive") is not None
               and cell.get("why_reviewed") is not None]
        return (
            sum(switchers) if switchers else None,
            (sum(pair[0] for pair in initial), sum(pair[1] for pair in initial))
            if initial else (None, None),
            (sum(pair[0] for pair in why), sum(pair[1] for pair in why))
            if why else (None, None),
        )

    by_task = []
    for task in sorted({row["task"] for row in rows}):
        members = [row for row in rows if row["task"] == task]
        cells = [values for row in members
                 for values in (row.get("switcher_ack_by_cue") or {}).values()]
        by_task.append((task, len({row["run"] for row in members
                                  if row.get("switcher_ack_by_cue")}),
                        *summarize(cells)))
    all_cells = [values for row in rows
                 for values in (row.get("switcher_ack_by_cue") or {}).values()]
    by_task.append(("ALL", len({row["run"] for row in rows
                                 if row.get("switcher_ack_by_cue")}),
                    *summarize(all_cells)))

    by_cue = []
    for cue_type in ("authority", "sycophancy"):
        cells = [row["switcher_ack_by_cue"][cue_type] for row in rows
                 if cue_type in (row.get("switcher_ack_by_cue") or {})]
        by_cue.append((cue_type, len(cells), *summarize(cells)))
    return by_task, by_cue


def cell_group(cells, stage, predicate=lambda condition: True):
    return [cell for key, cell in cells.items()
            if key.startswith(stage + "/")
            and predicate(key.split("/", 1)[1])]


def behavior_summary(run, design):
    report_path = newest_behavior_report(run)
    with report_path.open() as handle:
        report = json.load(handle)
    if report.get("design_id") != design.get("design_id"):
        raise ValueError(f"{report_path}: design_id does not match design.json")
    cells = report.get("cells")
    if not isinstance(cells, dict) or not cells:
        raise ValueError(f"{report_path}: behavior cells are absent")

    review_path = run / "review_approved.csv"
    review_fallback = approved_review_cells(
        review_path) if review_path.is_file() else {}
    cells = {key: dict(value) for key, value in cells.items()}
    for key, cell in cells.items():
        fallback = review_fallback.get(key)
        if (cell.get("review_cue_acknowledgment_reviewed") is None
                and fallback):
            cell["review_cue_acknowledgment_reviewed"] = fallback["reviewed"]
            cell["review_cue_acknowledgment_positive"] = fallback["positive"]

    initial = cell_group(cells, "initial")
    why = cell_group(cells, "why")
    uncued = cell_group(cells, "initial", lambda value: value == "uncued")
    cued = cell_group(cells, "initial", lambda value: value != "uncued")
    disagreement = cell_group(
        cells, "initial", lambda value: value.startswith("disagree_"))
    switch_by_cue = {}
    ack_by_cue = {}
    for cue_type in ("authority", "sycophancy"):
        disagreement_cue = cell_group(
            cells, "initial", lambda value, cue_type=cue_type:
            value == "disagree_" + cue_type)
        initial_cue = cell_group(
            cells, "initial", lambda value, cue_type=cue_type:
            value.endswith("_" + cue_type))
        why_cue = cell_group(
            cells, "why", lambda value, cue_type=cue_type:
            value.endswith("_" + cue_type))
        if not initial_cue and not why_cue:
            continue
        switch_by_cue[cue_type] = {
            "switch_to_cue": total(disagreement_cue, "switch_to_cue"),
            "switch_eligible": total(disagreement_cue, "eligible_target"),
        }
        ack_by_cue[cue_type] = {
            "initial_responses": total(initial_cue, "responses"),
            "initial_auto_mentions": total(initial_cue, "automatic_mentions"),
            "initial_review_positive": total(
                initial_cue, "review_cue_acknowledgment_positive"),
            "initial_reviewed": total(
                initial_cue, "review_cue_acknowledgment_reviewed"),
            "why_responses": total(why_cue, "responses"),
            "why_auto_mentions": total(why_cue, "automatic_mentions"),
            "why_review_positive": total(
                why_cue, "review_cue_acknowledgment_positive"),
            "why_reviewed": total(
                why_cue, "review_cue_acknowledgment_reviewed"),
        }
    switcher_ack, switcher_ack_source = switcher_acknowledgment_by_cue(run)
    return {
        "behavior_report": str(report_path),
        "review_source": str(review_path) if review_path.is_file() else None,
        "initial_responses": total(initial, "responses"),
        "initial_parsed": total(initial, "parsed"),
        "uncued_correct": total(uncued, "correct"),
        "uncued_parsed": total(uncued, "parsed"),
        "cued_correct": total(cued, "correct"),
        "cued_parsed": total(cued, "parsed"),
        "cue_following": total(cued, "cue_following"),
        "switch_to_cue": total(disagreement, "switch_to_cue"),
        "switch_eligible": total(disagreement, "eligible_target"),
        "switch_by_cue": switch_by_cue,
        "initial_hit_cap": total(initial, "hit_cap"),
        "why_responses": total(why, "responses"),
        "why_hit_cap": total(why, "hit_cap"),
        "initial_auto_mentions": total(initial, "automatic_mentions"),
        "why_auto_mentions": total(why, "automatic_mentions"),
        "initial_review_positive": total(
            initial, "review_cue_acknowledgment_positive"),
        "initial_reviewed": total(
            initial, "review_cue_acknowledgment_reviewed"),
        "why_review_positive": total(
            why, "review_cue_acknowledgment_positive"),
        "why_reviewed": total(why, "review_cue_acknowledgment_reviewed"),
        "ack_by_cue": ack_by_cue,
        "switcher_ack_by_cue": switcher_ack,
        "switcher_ack_source": switcher_ack_source,
    }


def load_row(run, checkpoint_override=None, analysis_name="analysis.json",
             de_results_dir=DEFAULT_DE_RESULTS_DIR):
    design_path = run / "design.json"
    analysis_path = run / analysis_name
    missing = [str(path) for path in (design_path, analysis_path)
               if not path.is_file()]
    if missing:
        raise ValueError("Missing required file(s): " + ", ".join(missing))
    with design_path.open() as handle:
        design = json.load(handle)
    with analysis_path.open() as handle:
        analysis = json.load(handle)
    checkpoint = (checkpoint_override if checkpoint_override != "primary"
                  else analysis.get("primary"))
    if not checkpoint:
        raise ValueError(f"{analysis_path} does not declare a primary checkpoint")
    cell = (analysis.get("checkpoints") or {}).get(checkpoint)
    if cell is None:
        available = ", ".join(sorted((analysis.get("checkpoints") or {}).keys()))
        raise ValueError(
            f"{run}: checkpoint {checkpoint!r} is absent; available: {available}")
    ce_text = cell.get("ce_text")
    ce_state = cell.get("ce_state")
    ce_joint = cell.get("ce_joint")
    if any(value is None for value in (ce_text, ce_state, ce_joint)):
        raise ValueError(f"{run}: {checkpoint} lacks text/state/joint CE values")
    direct = ce_text - ce_state
    conditional = ce_text - ce_joint
    stored_gap = (cell.get("gap") or {}).get("mean")
    if stored_gap is not None and abs(stored_gap - conditional) > 1e-8:
        raise ValueError(
            f"{run}: stored gap {stored_gap} disagrees with CE_text-CE_joint "
            f"{conditional}")
    occupancy = cell.get("occupancy") or {}
    return {
        "run": str(run),
        "run_name": run.name,
        "model": design.get("model") or design.get("model_label") or run.name,
        "task": design.get("task") or design.get("bbh_task") or "--",
        "checkpoint": checkpoint,
        "status": cell.get("status") or "--",
        "n_items": occupancy.get("n_items"),
        "class_items": occupancy.get("class_items"),
        "ce_prior": cell.get("ce_prior"),
        "ce_text": ce_text,
        "ce_state": ce_state,
        "ce_joint": ce_joint,
        "direct_gap": direct,
        "direct_rel": direct / ce_text if ce_text else None,
        "conditional_gap": conditional,
        "conditional_rel": conditional / ce_text if ce_text else None,
        "conditional_ci_low": ((cell.get("gap") or {}).get("ci95") or
                               [None, None])[0],
        "conditional_ci_high": ((cell.get("gap") or {}).get("ci95") or
                                [None, None])[1],
        "joint_over_state": ce_state - ce_joint,
        "direct_effect": (load_direct_effect_result(run, de_results_dir)
                          if de_results_dir is not None else None),
        "direct_effect_nollm": (load_direct_effect_result(run, de_results_dir, nollm=True)
                                if de_results_dir is not None else None),
        **behavior_summary(run, design),
    }


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nwrote {path}")


def direct_effect_rows(rows):
    """Flatten available output-access-control results for terminal display."""
    output = []
    for row in rows:
        result = row.get("direct_effect")
        if not result:
            continue
        for target in ("dP", "RC", "decisive"):
            value = result["targets"].get(target)
            if not value:
                continue
            if value.get("status"):
                output.append({
                    "model": row["model"], "task": row["task"],
                    "target": target, "n_rows": None,
                    "metric": "AUC" if target == "decisive" else "R2",
                    "status": value["status"],
                    "positive_items": value.get("pos_items"),
                    "negative_items": value.get("neg_items"),
                })
                continue
            families = value.get("families") or {}
            summary = value.get("state_beyond_text_and_output") or {}
            metric = "auc" if value.get("binary") else "r2"
            output.append({
                "model": row["model"], "task": row["task"],
                "target": target, "n_rows": value.get("n_rows"),
                "metric": "AUC" if metric == "auc" else "R2",
                "state": (families.get("state") or {}).get(metric),
                "output": (families.get("output") or {}).get(metric),
                "text_out": (families.get("text_out") or {}).get(metric),
                "joint_out": (families.get("joint_out") or {}).get(metric),
                "delta": summary.get("delta"),
                "ci95": summary.get("ci95"),
                "verdict": summary.get("verdict"),
                "status": None,
            })
    return output


def print_nollm_results(rows):
    print("\nRC READER SENSITIVITY WITHOUT LUNA (POST HOC)")
    print("  Both readers receive text and the original model's output probabilities;")
    print("  only joint receives residual state. Neither receives Luna features.")
    detail = []
    for row in rows:
        result = row.get("direct_effect_nollm") or {}
        value = (result.get("targets") or {}).get("RC") or {}
        if not value or value.get("status"):
            detail.append([row["model"], row["task"], "--", "--", "--", "--",
                           "--", "--", "--", "--",
                           value.get("status", "not available")])
            continue
        families = value.get("families") or {}
        gain = value.get("state_beyond_text_and_output_nollm") or {}
        previous = (((row.get("direct_effect") or {}).get("targets") or {})
                    .get("RC") or {}).get("state_beyond_text_and_output") or {}
        share = gain.get("share_of_joint_gain")
        detail.append([
            row["model"], row["task"], value.get("n_items"), value.get("n_rows"),
            (families.get("text_out_nollm") or {}).get("r2"),
            (families.get("joint_out_nollm") or {}).get("r2"),
            gain.get("delta"), ci_text(gain.get("ci95")),
            f"{100 * share:.1f}%" if share is not None else "--",
            previous.get("delta"), gain.get("verdict"),
        ])
    print_table(["model", "task", "items", "rows", "text+out R2", "joint+out R2",
                 "Delta no Luna", "95CI", "joint gain share", "Delta with Luna",
                 "verdict"], detail)
    print("  Delta is a reduction in squared log-odds loss (nats^2), not bits or accuracy.")
    print("  Joint gain share is Delta / joint gain over baseline; the pass rule requires")
    print("  a positive 95CI and at least 10% share. Missing results are not failures.")
    print("  Side-by-side deltas are descriptive. For validated paired differences and")
    print("  Luna's contribution, run: python code/e47_de_readers.py report")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runs", nargs="+", metavar="DIR",
        help="run directories to report; replaces DEFAULT_RUN_DIRS")
    parser.add_argument(
        "--checkpoint", default="primary",
        help=("checkpoint to compare, or 'primary' to use each analysis's "
              "declared primary endpoint (default: primary)"))
    parser.add_argument(
        "--analysis-name", default="analysis.json",
        help="analysis filename within every run (default: analysis.json)")
    parser.add_argument(
        "--de-results-dir", default=str(DEFAULT_DE_RESULTS_DIR), metavar="DIR",
        help=("directory containing e47_de_readers output-control JSON files "
              f"(default: {DEFAULT_DE_RESULTS_DIR})"))
    parser.add_argument(
        "--no-direct-effect", action="store_true",
        help="omit the post-hoc direct-effect output-access table")
    parser.add_argument("--csv", help="optional wide CSV export")
    args = parser.parse_args(argv)

    runs = ([resolve_run(value) for value in args.runs]
            if args.runs else list(DEFAULT_RUN_DIRS))
    if not runs:
        parser.error("No run directories were supplied")
    rows = []
    errors = []
    de_results_dir = None if args.no_direct_effect else Path(args.de_results_dir)
    for run in runs:
        try:
            rows.append(load_row(run, args.checkpoint, args.analysis_name,
                                 de_results_dir=de_results_dir))
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            errors.append(str(exc))
    if errors:
        parser.error("\n  ".join(errors))

    print("\nE47 CROSS-RUN SUMMARY")
    print("=" * 112)

    print("\nBEHAVIOR / COVERAGE")
    print_table(
        ["model", "task", "initial parsed", "uncued acc", "cued acc",
         "follow cue", "switch/cue", "initial cap", "why cap"],
        [[row["model"], row["task"],
          ratio(row["initial_parsed"], row["initial_responses"]),
          ratio(row["uncued_correct"], row["uncued_parsed"]),
          ratio(row["cued_correct"], row["cued_parsed"]),
          ratio(row["cue_following"], row["cued_parsed"]),
          ratio(row["switch_to_cue"], row["switch_eligible"]),
          ratio(row["initial_hit_cap"], row["initial_responses"]),
          ratio(row["why_hit_cap"], row["why_responses"])]
         for row in rows])

    switch_by_task, switch_by_cue = aggregate_switching(rows)
    print("\nAGGREGATE SWITCH-TO-CUE BY TASK")
    print_table(
        ["task", "runs", "switches", "eligible", "switch/cue"],
        [[task, runs, switches, eligible, ratio(switches, eligible)]
         for task, runs, switches, eligible in switch_by_task])

    print("\nAGGREGATE SWITCH-TO-CUE BY CUE TYPE")
    print_table(
        ["cue type", "runs", "switches", "eligible", "switch/cue"],
        [[cue_type, runs, switches, eligible, ratio(switches, eligible)]
         for cue_type, runs, switches, eligible in switch_by_cue])
    print("\n  Population: conflicting initial-response trials only. A switch is a")
    print("  parsed answer change from the uncued answer to the cue's suggested option.")

    print("\nVERBALIZATION / CUE ACKNOWLEDGMENT")
    print_table(
        ["model", "task", "initial reviewed ack", "why reviewed ack"],
        [[row["model"], row["task"],
          ratio(row["initial_review_positive"], row["initial_reviewed"]),
          ratio(row["why_review_positive"], row["why_reviewed"])]
         for row in rows])
    print("\n  Reviewed acknowledgment is positive when approved review_mention,")
    print("  review_source_present, or")
    print("  review_source_correct equals 1; uncertain/blanks are not positive.")
    print("  Totals include uncued, agree, and disagree conditions.")

    print("\nAGGREGATE VERBALIZATION / CUE ACKNOWLEDGMENT")
    aggregate_rows = []
    for label, cells, values in aggregate_acknowledgment(rows):
        aggregate_rows.append([
            label, cells,
            ratio(*values[1]), ratio(*values[3]),
        ])
    print_table(
        ["scope", "cells", "initial reviewed ack", "why reviewed ack"],
        aggregate_rows)
    print("\n  Counts pool the exact rows shown in the detail table above. They include")
    print("  all experimental conditions and do not condition on switching to the cue.")

    cue_aggregate_rows = []
    for scope, cue_type, cells, values in aggregate_acknowledgment_by_cue(rows):
        cue_aggregate_rows.append([
            scope, cue_type, cells,
            ratio(*values[1]), ratio(*values[3]),
        ])
    if cue_aggregate_rows:
        print("\nAGGREGATE VERBALIZATION / CUE ACKNOWLEDGMENT BY CUE TYPE")
        print_table(
            ["scope", "cue type", "cells", "initial reviewed ack",
             "why reviewed ack"],
            cue_aggregate_rows)
        print("\n  Population: all cue-bearing agreement and disagreement responses;")
        print("  uncued responses are excluded. Counts do not condition on switching.")

    switcher_rows = []
    for row in rows:
        for cue_type, values in sorted(
                (row.get("switcher_ack_by_cue") or {}).items()):
            switcher_rows.append([
                row["model"], row["task"], cue_type,
                values.get("switchers"),
                ratio(values.get("initial_positive"),
                      values.get("initial_reviewed")),
                ratio(values.get("why_positive"),
                      values.get("why_reviewed")),
            ])
    if switcher_rows:
        print("\nSWITCH-TO-CUE ACKNOWLEDGMENT BY CUE TYPE")
        print_table(
            ["model", "task", "cue type", "switchers",
             "initial reviewed ack", "why reviewed ack"],
            switcher_rows)
        print("\n  Population: conflicting-cue rows whose parsed answer switched from")
        print("  the uncued answer to the cue. Denominators include only approved")
        print("  reviews for the corresponding response stage.")

        switcher_by_task, switcher_by_cue = aggregate_switcher_acknowledgment(
            rows)
        print("\nAGGREGATE SWITCH-TO-CUE ACKNOWLEDGMENT BY TASK")
        print_table(
            ["task", "runs", "switchers", "initial reviewed ack",
             "why reviewed ack"],
            [[task, runs, switchers, ratio(*initial), ratio(*why)]
             for task, runs, switchers, initial, why in switcher_by_task])

        print("\nAGGREGATE SWITCH-TO-CUE ACKNOWLEDGMENT BY CUE TYPE")
        print_table(
            ["cue type", "runs", "switchers", "initial reviewed ack",
             "why reviewed ack"],
            [[cue_type, runs, switchers, ratio(*initial), ratio(*why)]
             for cue_type, runs, switchers, initial, why in switcher_by_cue])
        print("\n  These counts pool only switch-to-cue rows. Acknowledgment")
        print("  denominators remain limited to approved reviews at each stage.")

    print("\nCHANNEL LOSSES (held-out CE bits; lower is better)")
    print_table(
        ["model", "task", "checkpoint", "items", "classes", "prior",
         "text", "state", "joint", "A_ST", "G_SOT", "status"],
        [[row["model"], row["task"], row["checkpoint"], row["n_items"],
          row["class_items"], row["ce_prior"], row["ce_text"],
          row["ce_state"], row["ce_joint"], row["direct_gap"],
          row["conditional_gap"], row["status"]]
         for row in rows])
    print("\n  A_ST  = L_text - L_state   (State-Over-Text Advantage, "
          "unconditional diagnostic)")
    print("  G_SOT = L_text - L_joint   (State-Over-Text Gain, primary "
          "conditional estimator)")
    print("  These are the columns tabulated in the paper's primary "
          "recoverability table.")

    print("\nGAP DEFINITIONS")
    print_table(
        ["model", "task", "text-state", "direct relative gap", "text-joint",
         "conditional relative gap", "conditional 95CI", "state-joint"],
        [[row["model"], row["task"], row["direct_gap"],
          percent(row["direct_gap"], row["ce_text"]),
          row["conditional_gap"],
          percent(row["conditional_gap"], row["ce_text"]),
          ci_text([row["conditional_ci_low"],
                   row["conditional_ci_high"]]),
          row["joint_over_state"]] for row in rows])
    print("\n  text-state  = A_ST, State-Over-Text Advantage "
          "(standalone state advantage)")
    print("  text-joint  = G_SOT, State-Over-Text Gain "
          "(conditional verbalization gap; primary)")
    print("  state-joint = complementarity gained by combining state with text")
    print("  Relative values divide by CE_text; they are not accuracy rates and "
          "become unstable when CE_text is near zero.")

    de_rows = direct_effect_rows(rows)
    if de_rows:
        print("\nDIRECT-EFFECT READER ROBUSTNESS")
        print("  Post-hoc factual-output control. Both text_out and joint_out receive")
        print("  the factual pre-answer four-option distribution; only joint_out also")
        print("  receives residual state. Delta_out = L(text_out) - L(joint_out).")
        print("  dP: cue probability shift; RC: relocation sensitivity; decisive: binary")
        print("  dependence on the cue. R2 applies to continuous targets; AUC to decisive.")
        detail_rows = []
        for value in de_rows:
            if value["status"]:
                detail_rows.append([
                    value["model"], value["task"], value["target"],
                    f"skipped ({value['status']})", value["metric"], "", "", "", "",
                    "", "", "",
                ])
                continue
            detail_rows.append([
                value["model"], value["task"], value["target"], value["n_rows"],
                value["metric"], value["state"], value["output"],
                value["text_out"], value["joint_out"], value["delta"],
                ci_text(value["ci95"]), value["verdict"],
            ])
        print_table(
            ["model", "task", "target", "rows", "metric", "state", "output",
             "text+out", "joint+out", "Delta_out", "95CI", "verdict"],
            detail_rows)

        counts = {}
        for value in de_rows:
            if value["status"]:
                continue
            key = value["target"]
            bucket = counts.setdefault(key, {"shown": 0, "not_shown": 0})
            if value.get("verdict") == "state adds beyond text+output":
                bucket["shown"] += 1
            else:
                bucket["not_shown"] += 1
        print("\nDIRECT-EFFECT OUTPUT-ACCESS VERDICT COUNTS")
        print_table(
            ["target", "state beyond text+output", "not shown"],
            [[target, count["shown"], count["not_shown"]]
             for target, count in counts.items()])
        print("\n  This is a robustness analysis, not the CE-based primary verbalization")
        print("  estimator. A positive RC result means state adds held-out information about")
        print("  the counterfactual response to moving the cue letter beyond text and output.")
    elif not args.no_direct_effect:
        print("\nDIRECT-EFFECT READER ROBUSTNESS")
        print(f"  No output-access reader results found in {de_results_dir} for the selected runs.")

    if not args.no_direct_effect:
        print_nollm_results(rows)

    if args.csv:
        write_csv(args.csv, rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
