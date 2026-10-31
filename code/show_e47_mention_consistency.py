#!/usr/bin/env python3
"""Summarize reviewed initial-to-why cue-acknowledgment transitions.

The population is intervention-defined switch-to-cue cases with approved
cue acknowledgment labels for both the initial response and its neutral why
response. The label is one when any approved ``review_mention``,
``review_source_present``, or ``review_source_correct`` field is one. Missing
reviews are excluded; uncertain component values do not count as positive.

Edit DEFAULT_RUN_DIRS to change the ordinary report, or replace it at runtime:

    python code/show_e47_mention_consistency.py
    python code/show_e47_mention_consistency.py --runs e47/run_a e47/run_b
    python code/show_e47_mention_consistency.py --csv reports/e47_mentions.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

from run_dirs import DEFAULT_RUN_DIRS

def fmt(value, digits=3, missing="--"):
    if value is None:
        return missing
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def pct(value):
    return "--" if value is None else f"{100 * value:.1f}%"


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
    return (ROOT / path).resolve()


def newest_artifact(run, pattern):
    paths = [path for path in run.glob(pattern) if path.is_file()]
    if not paths:
        raise ValueError(f"{run}: no artifact matches {pattern!r}")
    return max(paths, key=lambda path: (path.stat().st_mtime_ns, path.name))


def exact_mcnemar_p(initial_only, why_only):
    """Two-sided exact binomial test of discordant paired outcomes."""
    discordant = initial_only + why_only
    if discordant == 0:
        return 1.0
    smaller = min(initial_only, why_only)
    lower_tail = sum(math.comb(discordant, k)
                     for k in range(smaller + 1)) / (2 ** discordant)
    return min(1.0, 2 * lower_tail)


def cohen_kappa(n00, n01, n10, n11):
    n = n00 + n01 + n10 + n11
    if not n:
        return None
    observed = (n00 + n11) / n
    initial_positive = (n10 + n11) / n
    why_positive = (n01 + n11) / n
    expected = (initial_positive * why_positive
                + (1 - initial_positive) * (1 - why_positive))
    return None if math.isclose(expected, 1.0) else (
        observed - expected) / (1 - expected)


def load_row(run, artifact_pattern):
    design_path = run / "design.json"
    if not design_path.is_file():
        raise ValueError(f"{run}: missing design.json")
    artifact_path = newest_artifact(run, artifact_pattern)
    with design_path.open() as handle:
        design = json.load(handle)
    with artifact_path.open() as handle:
        artifact = json.load(handle)
    disclosure = artifact.get("disclosure_plan") or {}
    if disclosure.get("status") == "annotations_required":
        raise ValueError(f"{run}: approved review annotations are required")
    transitions = disclosure.get("cue_acknowledgment_transition_counts")
    if transitions is None:
        raise ValueError(
            f"{artifact_path}: cue_acknowledgment_transition_counts is absent; "
            "rerun e47_what_causes_switch.py with the current code")

    n00 = int(transitions.get("initial_0_why_0", 0))
    n01 = int(transitions.get("initial_0_why_1", 0))
    n10 = int(transitions.get("initial_1_why_0", 0))
    n11 = int(transitions.get("initial_1_why_1", 0))
    if min(n00, n01, n10, n11) < 0:
        raise ValueError(f"{artifact_path}: transition counts cannot be negative")
    paired = n00 + n01 + n10 + n11
    initial_acknowledgments = n10 + n11
    why_acknowledgments = n01 + n11
    initial_silent = n00 + n01
    initial_positive = n10 + n11
    return {
        "run": str(run),
        "run_name": run.name,
        "artifact": str(artifact_path),
        "model": design.get("model") or design.get("model_label") or run.name,
        "task": design.get("task") or design.get("bbh_task") or "--",
        "status": disclosure.get("status") or "--",
        "paired_switchers": paired,
        "initial_acknowledgments": initial_acknowledgments,
        "why_acknowledgments": why_acknowledgments,
        "neither": n00,
        "new_in_why": n01,
        "initial_only": n10,
        "both": n11,
        "agreement": ((n00 + n11) / paired if paired else None),
        "initial_acknowledgment_rate": (
            initial_acknowledgments / paired if paired else None),
        "why_acknowledgment_rate": (
            why_acknowledgments / paired if paired else None),
        "retained_after_initial_acknowledgment": (
            n11 / initial_positive if initial_positive else None),
        "new_after_initial_silence": (
            n01 / initial_silent if initial_silent else None),
        "cohen_kappa": cohen_kappa(n00, n01, n10, n11),
        "mcnemar_exact_p": exact_mcnemar_p(n10, n01),
    }


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nwrote {path}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runs", nargs="+", metavar="DIR",
        help="run directories to report; replaces DEFAULT_RUN_DIRS")
    parser.add_argument(
        "--artifact-pattern", default="what_causes_switch*.json",
        help=("artifact filename or glob within every run "
              "(default: what_causes_switch*.json)"))
    parser.add_argument("--csv", help="optional CSV export")
    args = parser.parse_args(argv)

    runs = ([resolve_run(value) for value in args.runs]
            if args.runs else list(DEFAULT_RUN_DIRS))
    if not runs:
        parser.error("No run directories were supplied")
    rows = []
    errors = []
    for run in runs:
        try:
            rows.append(load_row(run, args.artifact_pattern))
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            errors.append(str(exc))
    if errors:
        parser.error("\n  ".join(errors))

    print("\nE47 REVIEWED CUE-ACKNOWLEDGMENT CORRESPONDENCE")
    print("=" * 120)
    print("\nPAIRED TRANSITIONS AMONG SWITCH-TO-CUE CASES")
    print_table(
        ["model", "task", "paired", "initial ack", "why ack",
         "neither", "new in why", "initial only", "both", "agreement"],
        [[row["model"], row["task"], row["paired_switchers"],
          row["initial_acknowledgments"], row["why_acknowledgments"],
          row["neither"],
          row["new_in_why"], row["initial_only"], row["both"],
          pct(row["agreement"])] for row in rows])

    print("\nCORRESPONDENCE DIAGNOSTICS")
    print_table(
        ["model", "initial rate", "why rate", "ack retained",
         "new after silence", "kappa", "McNemar p"],
        [[row["model"], pct(row["initial_acknowledgment_rate"]),
          pct(row["why_acknowledgment_rate"]),
          pct(row["retained_after_initial_acknowledgment"]),
          pct(row["new_after_initial_silence"]), row["cohen_kappa"],
          row["mcnemar_exact_p"]] for row in rows])
    print("\n  neither      = no reviewed cue acknowledgment in either response")
    print("  new in why   = acknowledged only after the neutral why prompt")
    print("  initial only = acknowledged initially but not in the why response")
    print("  both         = acknowledged in both responses")
    print("  Cue acknowledgment is mention OR source-present OR source-correct.")
    print("  Agreement and kappa describe correspondence, not causal attribution.")
    print("  McNemar p tests whether new-in-why and initial-only transitions are "
          "symmetric.")
    print("  Missing reviews are excluded; uncertain component labels do not count "
          "as positive.")

    if args.csv:
        write_csv(args.csv, rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
