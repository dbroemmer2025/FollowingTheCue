#!/usr/bin/env python3
"""Display E47 neutral-insertion sidecar results without recomputing them."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUNS_ROOT = ROOT / "e47"
SCHEMA = "e47_neutral_insertion_v1"
AUDIT_SCHEMA = "e47_uncued_replay_audit_v1"
FRAME_ORDER = {"authority": 0, "sycophancy": 1}


def table(headers, rows):
    widths = [max(len(str(header)), *(len(str(row[i])) for row in rows))
              for i, header in enumerate(headers)]
    print("  " + "  ".join(str(header).ljust(widths[i])
                          for i, header in enumerate(headers)))
    print("  " + "  ".join("-" * width for width in widths))
    for row in rows:
        print("  " + "  ".join(str(value).ljust(widths[i])
                              for i, value in enumerate(row)))


def pct(value, *, signed=False):
    if value is None:
        return "--"
    return f"{100 * value:+.1f}" if signed else f"{100 * value:.1f}%"


def ci(value):
    if value is None:
        return "--"
    if (not isinstance(value, list) or len(value) != 2 or
            any(not isinstance(v, (int, float)) or not math.isfinite(v)
                for v in value) or value[0] > value[1]):
        raise ValueError(f"Invalid confidence interval: {value!r}")
    return f"[{pct(value[0], signed=True)},{pct(value[1], signed=True)}]"


def resolve_run(value, runs_root):
    path = Path(value).expanduser()
    for candidate in (path, ROOT / path, runs_root / path):
        if candidate.is_dir():
            return candidate.resolve()
    raise ValueError(f"Run directory not found: {value}")


def load_summary(run, sidecar_name="neutral_insertion"):
    path = run / sidecar_name / "summary.json"
    if not path.is_file():
        raise ValueError(f"{run.name}: missing {path}; run the neutral-insertion report stage")
    summary = json.loads(path.read_text())
    if summary.get("schema") != SCHEMA:
        raise ValueError(f"{path}: unsupported schema {summary.get('schema')!r}")
    if not isinstance(summary.get("frames"), dict):
        raise ValueError(f"{path}: missing frames")
    return summary


def load_audit(run, sidecar_name, replay_name):
    path = run / sidecar_name / replay_name / "neutral_comparison.json"
    if not path.is_file():
        return None
    audit = json.loads(path.read_text())
    if audit.get("schema") != AUDIT_SCHEMA or audit.get("cell") != run.name:
        raise ValueError(f"{path}: audit schema or cell does not match")
    if (audit.get("sidecar_name") != sidecar_name or
            audit.get("replay_name") != replay_name):
        raise ValueError(f"{path}: audit sidecar or replay name does not match")
    return audit


def count_rate(entry, denominator):
    if denominator == 0:
        return "0/0 (--)"
    return f'{entry["n"]}/{denominator} ({pct(entry["rate"])})'


def audit_rows(audits, subset=False):
    rows = []
    for run, audit in audits:
        summary = audit["summary"]
        if subset:
            summary = summary["cue_matched_subset"]
        n = summary["valid_triplets"]
        if n == 0:
            rows.append((run.name, 0, "--", "--", "--", "--", "--", "--"))
            continue
        rows.append((run.name, n,
                     count_rate(summary["neutral_away_replay"], n),
                     count_rate(summary["replay_away_frozen"], n),
                     count_rate(summary["neutral_only_change"], n),
                     count_rate(summary["shared_change"], n),
                     pct(summary["away_difference"]["mean"], signed=True),
                     ci(summary["away_difference"]["ci95"])))
    return rows


def display_audits(audits, missing):
    if not audits:
        if missing:
            print("\nUNCUED REPLAY AUDIT: not found for " + ", ".join(missing))
        return
    print()
    print("UNCUED REPLAY AUDIT")
    table(["cell", "neutral", "uncued", "valid", "token match", "prompt mismatch",
           "same session", "same runtime", "same code"], [
        (run.name,
         f'{a["neutral_completed"]}/{a["planned_neutral"]}',
         f'{a["replay_completed"]}/{a["planned_uncued"]}',
         f'{a["summary"]["valid_triplets"]}/{a["summary"]["neutral_replay_pairs"]}',
         f'{a["uncued_token_matches"]}/{a["replay_completed"]}',
         a["prompt_id_mismatches"],
         str(a["provenance"]["same_recorded_session"]),
         str(a["provenance"]["same_runtime"]),
         str(a["same_recorded_generation_code"]))
        for run, a in audits
    ])
    print("  True means recorded neutral and replay environments match; token match")
    print("  compares the replay with the frozen uncued response, not the neutral response.")
    if any(a["provenance"]["same_recorded_session"] is not True or
           a["provenance"]["same_runtime"] is not True or
           a["same_recorded_generation_code"] is not True or
           a["neutral_completed"] != a["planned_neutral"] or
           a["replay_completed"] != a["planned_uncued"] or
           a["prompt_id_mismatches"] for _, a in audits):
        print("  WARNING: one or more audits lack complete same-session provenance;")
        print("  do not describe those cells as matched-session results.")
    print()
    headers = ["cell", "valid", "neutral away", "replay away", "neutral only",
               "shared change", "diff pp", "95% CI pp"]
    print("PAIRED ANSWER CHANGE (ALL VALID TRIPLETS)")
    table(headers, audit_rows(audits))
    print("  Away means answer differs from the frozen uncued answer.")
    print("  Diff = neutral-away minus replay-away on the same rows; CI resamples source items.")
    print()
    print("PAIRED ANSWER CHANGE (VALID FROZEN CUED ANSWER)")
    table(headers, audit_rows(audits, subset=True))
    print("  The cue-valid subset can differ from the older matched denominator after answer rescoring.")
    if missing:
        print("  No replay audit found for: " + ", ".join(missing))


def display(summaries, sidecar_name="neutral_insertion", replay_name="full_uncued"):
    records = []
    for run, summary in summaries:
        for frame, entry in summary["frames"].items():
            n = entry["matched_valid_n"]
            records.append((run.name, summary, frame, entry, n))
    records.sort(key=lambda r: (r[0], FRAME_ORDER.get(r[2], 99)))

    print(f"E47 NEUTRAL INSERTION (post hoc; sidecar={sidecar_name})")
    print("  Cue-versus-neutral rates use the same matched valid items in each row.")
    print("  Delta = cue minus neutral, in percentage points; CI is source-item bootstrap 95%.")
    print()
    print("DIRECTIONAL CUE FOLLOWING")
    table(["model", "task", "frame", "matched", "cue->letter", "neutral->letter",
           "delta pp", "95% CI pp", "cue-only", "neutral-only", "tok diff"], [
        (s.get("model", "--"), s.get("task", "--"), frame, n,
         pct(e["cue_to_assigned_rate"]), pct(e["neutral_to_assigned_rate"]),
         pct(e["directional_excess"], signed=True), ci(e["directional_excess_ci95"]),
         e.get("cue_only_assigned_n", "--"), e.get("neutral_only_assigned_n", "--"),
         "--" if e.get("prompt_token_delta_mean") is None
         else f'{e["prompt_token_delta_mean"]:+.1f}')
        for _, s, frame, e, n in records
    ])
    print("  cue-only / neutral-only = matched items choosing the assigned letter under only")
    print("  that prompt (paired discordant counts). tok diff = mean neutral minus cue prompt tokens.")
    print()
    print("ANSWER INSTABILITY / COVERAGE")
    table(["model", "task", "frame", "generated", "matched", "cue-away",
           "neutral-away", "delta pp", "95% CI pp", "cap", "ambig"], [
        (s.get("model", "--"), s.get("task", "--"), frame,
         f'{e["generated_n"]}/{e["planned_n"]}', n,
         pct(e["cue_away_rate"]), pct(e["neutral_away_matched_rate"]),
         pct(e["away_excess"], signed=True), ci(e["away_excess_ci95"]),
         e["neutral_capped_n"], e["neutral_ambiguous_n"])
        for _, s, frame, e, n in records
    ])
    print("  Away = answer differs from the uncued answer. All rates above use matched items.")
    replays = []
    for run, summary in summaries:
        for path in sorted((run / sidecar_name).glob("*/replay_summary.json")):
            data = json.loads(path.read_text())
            for kind, entry in data["conditions"].items():
                replays.append((summary.get("model", "--"), summary.get("task", "--"),
                                path.parent.name, kind,
                                f'{entry["identical_tokens"]}/{entry["n"]}',
                                f'{entry["answer_agree"]}/{entry["answer_parsed_both_n"]}',
                                "yes" if data.get("complete") else "PARTIAL"))
    print()
    if replays:
        print("REPRODUCIBILITY (verbatim replay of frozen prompts)")
        table(["model", "task", "replay", "condition", "identical tokens",
               "same answer (same parser)", "complete"], replays)
    else:
        print("REPRODUCIBILITY: no replay_summary.json found; run the sidecar `replay` command.")
    incomplete = [run.name for run, summary in summaries if not summary.get("complete")]
    if incomplete:
        print("  PARTIAL reports: " + ", ".join(incomplete))
    audits = [(run, load_audit(run, sidecar_name, replay_name))
              for run, _ in summaries]
    if sidecar_name != "neutral_insertion" or any(audit is not None for _, audit in audits):
        display_audits([(run, audit) for run, audit in audits if audit is not None],
                       [run.name for run, audit in audits if audit is None])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="*", help="Run directories or names; default: all available sidecars")
    parser.add_argument("--runs-root", type=Path, default=DEFAULT_RUNS_ROOT)
    parser.add_argument("--sidecar-name", default="neutral_insertion",
                        help="sidecar directory within each run (default: archived neutral_insertion)")
    parser.add_argument("--replay-name", default="full_uncued",
                        help="replay subdirectory containing neutral_comparison.json")
    args = parser.parse_args(argv)
    runs_root = args.runs_root.expanduser().resolve()
    runs = ([resolve_run(value, runs_root) for value in args.runs]
            if args.runs else sorted(path.parent.parent for path in
                                     runs_root.glob(f"*/{args.sidecar_name}/summary.json")))
    if not runs:
        parser.error(f"No neutral-insertion summaries found under {runs_root}")
    try:
        display([(run, load_summary(run, args.sidecar_name)) for run in runs],
                args.sidecar_name, args.replay_name)
    except (KeyError, TypeError, ValueError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
