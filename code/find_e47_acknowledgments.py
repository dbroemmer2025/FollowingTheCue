#!/usr/bin/env python3
"""List reviewed E47 cue-acknowledgment examples.

Examples:
  python code/find_e47_acknowledgments.py --run e47/mistral_nemo_instruct_2407_temporal_sequences_thinking_v10
  python code/find_e47_acknowledgments.py --all-runs --condition agree_authority --limit 12
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parent.parent
ACK_FIELDS = ("review_mention", "review_source_present", "review_source_correct")


def positive(value: str | None) -> bool:
    return (value or "").strip() == "1"


def compact(text: str, width: int = 220) -> str:
    text = " ".join((text or "").split())
    if len(text) <= width:
        return text
    return text[: width - 3].rstrip() + "..."


def review_files(args: argparse.Namespace) -> list[Path]:
    if args.all_runs:
        return sorted((ROOT / "e47").glob("*/review_approved.csv"))
    return [args.run / "review_approved.csv"]


def rows_from(path: Path, args: argparse.Namespace) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"No approved review file: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    selected = []
    for row in rows:
        if row.get("condition") != args.condition:
            continue
        if row.get("branch") != args.branch:
            continue
        if args.cue_type and row.get("cue_type") != args.cue_type:
            continue
        if not any(positive(row.get(field)) for field in ACK_FIELDS):
            continue
        selected.append(row)
    return selected


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--run", type=Path, help="one E47 run directory")
    group.add_argument("--all-runs", action="store_true", help="search every e47/*/review_approved.csv")
    parser.add_argument("--condition", default="agree_authority",
                        choices=("agree_authority", "agree_sycophancy", "disagree_authority", "disagree_sycophancy"))
    parser.add_argument("--branch", default="initial", choices=("initial", "why"))
    parser.add_argument("--cue-type", choices=("authority", "sycophancy"),
                        help="optional redundant cue-family filter")
    parser.add_argument("--limit", type=int, default=20)
    args = parser.parse_args()

    matches: list[tuple[Path, dict[str, str]]] = []
    for path in review_files(args):
        for row in rows_from(path, args):
            matches.append((path.parent, row))
    matches.sort(key=lambda item: (item[0].name, int(item[1]["index"])))

    print(f"Reviewed acknowledgments: {len(matches)} matching rows")
    print(f"condition={args.condition} branch={args.branch}")
    for run, row in matches[:args.limit]:
        labels = "/".join(
            field.removeprefix("review_").replace("_", "=")
            for field in ACK_FIELDS if positive(row.get(field)))
        print(f"\n{run.name} | item {row['index']} | {row.get('cue_type') or '-'} | {labels}")
        print(f"  {compact(row.get('response_episode', ''))}")
        print("  viewer: "
              f"./code/show_with_why.sh {row['index']} --run-dir {run} "
              f"--condition {args.condition} --arm cued")
    if len(matches) > args.limit:
        print(f"\nShowing {args.limit}/{len(matches)}. Increase --limit to see more.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
