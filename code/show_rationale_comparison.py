#!/usr/bin/env python3
"""Display E48 initial-vs-why rationale comparison results as ASCII tables.

Examples:

    python show_rationale_comparison.py
    python show_rationale_comparison.py ../e47/my_run
    python show_rationale_comparison.py --runs e47/run_a e47/run_b
    python show_rationale_comparison.py --aggregate --strata cue
    python show_rationale_comparison.py --csv ../reports/e48_summary.csv

With no run arguments, all E47 directories containing
``rationale_comparison_report.json`` are shown.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
E47_ROOT = ROOT / "e47"
SCHEMA = "e48_rationale_comparison_v1_report"
RATE_KEYS = (
    "decision_changed",
    "same_decision_materially_different_reason",
    "same_decision_elaboration",
    "why_has_no_substantive_reasoning",
)
RATE_LABELS = {
    "decision_changed": "changed decision",
    "same_decision_materially_different_reason": "same decision/different reason",
    "same_decision_elaboration": "same decision/elaborated",
    "why_has_no_substantive_reasoning": "why no reason",
}
TRANSITION_ORDER = (
    "same_decision_same_core_reasoning",
    "same_decision_elaborated_reasoning",
    "same_decision_different_reasoning",
    "same_decision_contradictory_reasoning",
    "same_decision_why_has_no_substantive_reasoning",
    "changed_decision_same_core_reasoning",
    "changed_decision_elaborated_reasoning",
    "changed_decision_different_reasoning",
    "changed_decision_contradictory_reasoning",
    "changed_decision_reasoning_incomparable",
    "same_decision_reasoning_incomparable",
    "incomparable_or_unclear",
)


def print_table(headers, rows, *, indent="  "):
    rendered = [[str(value) for value in row] for row in rows]
    widths = [len(header) for header in headers]
    for row in rendered:
        for index, value in enumerate(row):
            widths[index] = max(widths[index], len(value))
    print(indent + "  ".join(
        header.ljust(widths[index]) for index, header in enumerate(headers)))
    print(indent + "  ".join("-" * width for width in widths))
    for row in rendered:
        print(indent + "  ".join(
            value.ljust(widths[index]) for index, value in enumerate(row)))


def resolve_run(value):
    path = Path(value).expanduser()
    candidates = (path, ROOT / path, E47_ROOT / path)
    for candidate in candidates:
        if candidate.is_dir():
            return candidate.resolve()
    raise ValueError(f"Run directory not found: {value}")


def discover_runs():
    if not E47_ROOT.is_dir():
        return []
    return sorted(
        path.parent.resolve()
        for path in E47_ROOT.glob("*/rationale_comparison_report.json")
        if path.is_file())


def check_rate(metric, context):
    if not isinstance(metric, dict):
        raise ValueError(f"{context}: rate record is absent")
    count, n, rate = metric.get("count"), metric.get("n"), metric.get("rate")
    if not isinstance(count, int) or not isinstance(n, int) or count < 0 or n < 0 or count > n:
        raise ValueError(f"{context}: invalid count/denominator")
    expected = count / n if n else None
    if ((rate is None) != (expected is None) or
            (rate is not None and not math.isclose(rate, expected, abs_tol=1e-12))):
        raise ValueError(f"{context}: stored rate disagrees with count/denominator")
    interval = metric.get("item_cluster_bootstrap_95ci")
    if interval is not None and (not isinstance(interval, list) or len(interval) != 2):
        raise ValueError(f"{context}: invalid confidence interval")


def load_run(run, artifact_name):
    design_path = run / "design.json"
    report_path = run / artifact_name
    missing = [str(path) for path in (design_path, report_path) if not path.is_file()]
    if missing:
        raise ValueError("Missing required file(s): " + ", ".join(missing))
    with design_path.open() as handle:
        design = json.load(handle)
    with report_path.open() as handle:
        report = json.load(handle)
    if report.get("schema") != SCHEMA:
        raise ValueError(f"{report_path}: unsupported schema {report.get('schema')!r}")
    strata = report.get("strata")
    if not isinstance(strata, dict) or "all" not in strata:
        raise ValueError(f"{report_path}: missing all stratum")
    for name, summary in strata.items():
        rates = summary.get("rates") or {}
        for key in RATE_KEYS:
            check_rate(rates.get(key), f"{report_path}:{name}:{key}")
        transitions = summary.get("transition")
        if not isinstance(transitions, dict) or any(
                not isinstance(value, int) or value < 0 for value in transitions.values()):
            raise ValueError(f"{report_path}:{name}: invalid transition counts")
        if sum(transitions.values()) != summary.get("n"):
            raise ValueError(f"{report_path}:{name}: transitions do not sum to n")
    status = report.get("status") or {}
    if any(not isinstance(value, int) or value < 0 for value in status.values()):
        raise ValueError(f"{report_path}: invalid status counts")
    if sum(status.values()) < strata["all"]["n"]:
        raise ValueError(f"{report_path}: status count is below reported rows")
    return {
        "run": run,
        "run_name": run.name,
        "model": design.get("model") or design.get("model_label") or run.name,
        "task": design.get("task") or design.get("bbh_task") or "--",
        "report_path": report_path,
        "report": report,
    }


def pct(value, digits=1):
    return "--" if value is None else f"{100 * value:.{digits}f}%"


def rate_cell(metric):
    return f"{metric['count']}/{metric['n']} ({pct(metric['rate'])})"


def ci_cell(metric):
    interval = metric.get("item_cluster_bootstrap_95ci")
    if interval is None:
        return "--"
    return f"[{pct(interval[0])}, {pct(interval[1])}]"


def stratum_label(name):
    prefixes = {
        "condition=": "cond:",
        "cue_type=": "cue:",
        "switch_to_cue=": "switch:",
    }
    for prefix, replacement in prefixes.items():
        if name.startswith(prefix):
            return replacement + name[len(prefix):]
    return name


def selected_strata(strata, selection):
    if selection == "all":
        return list(strata.items())
    prefixes = {
        "overall": ("all",),
        "condition": ("all", "condition="),
        "cue": ("all", "cue_type="),
        "switch": ("all", "switch_to_cue="),
    }[selection]
    return [(name, value) for name, value in strata.items()
            if any(name == prefix or name.startswith(prefix) for prefix in prefixes)]


def status_cell(report):
    status = report.get("status") or {}
    proposed = status.get("proposed", 0)
    total = sum(status.values())
    state = "complete" if report.get("complete") else "partial"
    return f"{state} {proposed}/{total}"


def print_run(record, selection):
    report = record["report"]
    overall = report["strata"]["all"]
    print("\nE48 RATIONALE COMPARISON")
    print("=" * 120)
    print(f"  run     : {record['run_name']}")
    print(f"  model   : {record['model']}")
    print(f"  task    : {record['task']}")
    print(f"  status  : {status_cell(report)}; {overall['items']} source items")
    print(f"  artifact: {record['report_path']}")

    print("\nKEY RATES BY STRATUM")
    rows = []
    for name, summary in selected_strata(report["strata"], selection):
        rates = summary["rates"]
        rows.append([
            stratum_label(name), summary["n"], summary["items"],
            rate_cell(rates["decision_changed"]),
            rate_cell(rates["same_decision_materially_different_reason"]),
            rate_cell(rates["same_decision_elaboration"]),
            rate_cell(rates["why_has_no_substantive_reasoning"]),
        ])
    print_table(
        ["stratum", "rows", "items", "changed decision",
         "same decision/different reason", "same decision/elaborated",
         "why no reason"], rows)

    print("\nOVERALL ITEM-CLUSTER BOOTSTRAP INTERVALS")
    rows = []
    for key in RATE_KEYS:
        metric = overall["rates"][key]
        rows.append([RATE_LABELS[key], metric["count"], metric["n"],
                     pct(metric["rate"]), ci_cell(metric)])
    print_table(["metric", "count", "eligible", "rate", "95% CI"], rows)

    print("\nOVERALL SEMANTIC TRANSITIONS")
    transitions = overall["transition"]
    transition_rows = []
    seen = set()
    for name in TRANSITION_ORDER:
        if name in transitions:
            transition_rows.append([name, transitions[name],
                                    pct(transitions[name] / overall["n"])])
            seen.add(name)
    for name in sorted(set(transitions) - seen):
        transition_rows.append([name, transitions[name],
                                pct(transitions[name] / overall["n"])])
    print_table(["transition", "count", "% of rows"], transition_rows)

    print("\n  Each rate cell is count/eligible (percent); 'rows' is not necessarily")
    print("  that metric's denominator. Different reason combines different and")
    print("  contradictory reasoning only when the decision stayed the same.")
    print("  These are LLM-proposed textual labels, not human-validated causal claims.")
    if not report.get("complete"):
        pending = sum(value for key, value in report.get("status", {}).items()
                      if key != "proposed")
        print(f"  WARNING: partial report; {pending} rows are not proposed labels.")


def aggregate_summary(records, selection):
    """Pool count/denominator metrics without combining per-run CIs.

    Each run's source items are a distinct model--task evaluation population,
    so this is a descriptive response-weighted summary. Item-cluster bootstrap
    intervals remain meaningful only within their originating run and are
    intentionally not averaged here.
    """
    groups = [("ALL", records)]
    for task in sorted({str(record["task"]) for record in records}):
        groups.append((f"task={task}", [record for record in records
                                         if str(record["task"]) == task]))

    print("\nAGGREGATED E48 RATIONALE COMPARISON")
    print("=" * 120)
    print("  Pooled counts are descriptive across model--task cells; per-run item-cluster")
    print("  bootstrap intervals are not pooled or averaged.")

    for label, subset in groups:
        available = set.intersection(*(set(record["report"]["strata"])
                                       for record in subset))
        ordered = [(name, summary) for name, summary in
                   selected_strata(subset[0]["report"]["strata"], selection)
                   if name in available]
        if not ordered:
            continue
        print(f"\n  {label}: {len(subset)} cell(s)")
        rows = []
        for name, _ in ordered:
            summaries = [record["report"]["strata"][name] for record in subset]
            metric_cells = []
            for key in RATE_KEYS:
                metrics = [summary["rates"][key] for summary in summaries]
                count = sum(metric["count"] for metric in metrics)
                denominator = sum(metric["n"] for metric in metrics)
                metric_cells.append(f"{count}/{denominator} ({pct(count / denominator if denominator else None)})")
            rows.append([
                stratum_label(name),
                sum(summary["n"] for summary in summaries),
                sum(summary["items"] for summary in summaries),
                *metric_cells,
            ])
        print_table(
            ["stratum", "rows", "source-item records", "changed decision",
             "same decision/different reason", "same decision/elaborated",
             "why no reason"], rows)


def flat_rows(record):
    output = []
    for name, summary in record["report"]["strata"].items():
        row = {
            "run": str(record["run"]), "run_name": record["run_name"],
            "model": record["model"], "task": record["task"],
            "complete": record["report"].get("complete"),
            "stratum": name, "rows": summary["n"], "items": summary["items"],
        }
        for key in RATE_KEYS:
            metric = summary["rates"][key]
            row[f"{key}_count"] = metric["count"]
            row[f"{key}_eligible"] = metric["n"]
            row[f"{key}_rate"] = metric["rate"]
            interval = metric.get("item_cluster_bootstrap_95ci") or [None, None]
            row[f"{key}_ci_low"] = interval[0]
            row[f"{key}_ci_high"] = interval[1]
        output.append(row)
    return output


def write_csv(path, records):
    rows = [row for record in records for row in flat_rows(record)]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nwrote {path}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dirs", nargs="*", metavar="RUN",
                        help="run names or paths; default: auto-discover E48 reports")
    parser.add_argument("--runs", nargs="+", metavar="RUN",
                        help="alternate explicit run list")
    parser.add_argument(
        "--strata", choices=("all", "overall", "condition", "cue", "switch"),
        default="all", help="strata to display (default: all)")
    parser.add_argument(
        "--artifact-name", default="rationale_comparison_report.json",
        help="report filename within each run")
    parser.add_argument("--csv", help="optional denominator-explicit CSV export")
    parser.add_argument(
        "--aggregate", action="store_true",
        help=("also print response-weighted all-cell and task-level pooled "
              "metrics; per-run bootstrap intervals are not pooled"))
    args = parser.parse_args(argv)
    if args.run_dirs and args.runs:
        parser.error("Use positional run directories or --runs, not both")
    supplied = args.runs or args.run_dirs
    try:
        runs = [resolve_run(value) for value in supplied] if supplied else discover_runs()
        if not runs:
            raise ValueError("No E48 rationale comparison reports found under e47/")
        records = [load_run(run, args.artifact_name) for run in runs]
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    for record in records:
        print_run(record, args.strata)
    if args.aggregate:
        aggregate_summary(records, args.strata)
    if args.csv:
        write_csv(args.csv, records)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
