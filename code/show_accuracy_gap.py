#!/usr/bin/env python3
"""Show E47 correctness-versus-verbalization-gap metrics across model runs.

By default this uses ``DEFAULT_RUN_DIRS`` from ``run_dirs.py``.  The newest
``accuracy_gap_*.json`` in each run is selected, matching ``show_run.sh``.

Examples:

    python show_accuracy_gap.py
    python show_accuracy_gap.py --runs ../e47/run_a ../e47/run_b
    python show_accuracy_gap.py --overall-only
    python show_accuracy_gap.py --csv ../reports/e47_accuracy_gap.csv
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

from run_dirs import DEFAULT_RUN_DIRS


ROOT = Path(__file__).resolve().parents[1]
E47_ROOT = ROOT / "e47"
SCHEMA = "e47_accuracy_gap_v1"
CUE_ORDER = {"all": 0, "authority": 1, "sycophancy": 2}


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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
    for candidate in (path, ROOT / path, E47_ROOT / path):
        if candidate.is_dir():
            return candidate.resolve()
    raise ValueError(f"Run directory not found: {value}")


def newest_artifact(run, pattern):
    paths = [path for path in run.glob(pattern) if path.is_file()]
    if not paths:
        raise ValueError(f"{run}: no artifact matches {pattern!r}")
    return max(paths, key=lambda path: (path.stat().st_mtime_ns, path.name)), paths


def require_number(value, context, *, minimum=None, maximum=None):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{context}: expected a finite number")
    if minimum is not None and value < minimum:
        raise ValueError(f"{context}: value is below {minimum}")
    if maximum is not None and value > maximum:
        raise ValueError(f"{context}: value is above {maximum}")
    return value


def validate_interval(value, context):
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError(f"{context}: expected a two-value interval")
    low = require_number(value[0], context)
    high = require_number(value[1], context)
    if low > high:
        raise ValueError(f"{context}: reversed interval")
    return value


def validate_cell(cell, context):
    n_rows = cell.get("n_rows")
    n_items = cell.get("n_items")
    if not isinstance(n_rows, int) or not isinstance(n_items, int) or min(n_rows, n_items) < 1:
        raise ValueError(f"{context}: invalid row/item occupancy")
    accuracy = cell.get("accuracy") or {}
    require_number(accuracy.get("mean"), context + ":accuracy", minimum=0, maximum=1)
    validate_interval(accuracy.get("ci95"), context + ":accuracy CI")
    correct = cell.get("correct") or {}
    wrong = cell.get("incorrect") or {}
    contrast = cell.get("correct_minus_incorrect_gap") or {}
    correct_n, wrong_n = correct.get("n_rows"), wrong.get("n_rows")
    if not isinstance(correct_n, int) or not isinstance(wrong_n, int) or (
            correct_n < 1 or wrong_n < 1 or correct_n + wrong_n != n_rows):
        raise ValueError(f"{context}: correct/wrong rows do not partition n_rows")
    if (contrast.get("n_correct_rows") != correct_n or
            contrast.get("n_incorrect_rows") != wrong_n):
        raise ValueError(f"{context}: contrast occupancy disagrees with subsets")
    for label, subset in (("correct", correct), ("wrong", wrong)):
        for key in ("text_ce_bits", "state_ce_bits", "joint_ce_bits"):
            require_number(subset.get(key), f"{context}:{label}:{key}")
        gap = subset.get("gap") or {}
        gap_mean = require_number(gap.get("mean"), f"{context}:{label}:gap")
        validate_interval(gap.get("ci95"), f"{context}:{label}:gap CI")
        if not math.isclose(gap_mean,
                            subset["text_ce_bits"] - subset["joint_ce_bits"],
                            abs_tol=1e-10):
            raise ValueError(f"{context}:{label}: gap differs from text minus joint")
        relative = subset.get("relative_gap")
        expected = gap_mean / subset["text_ce_bits"] if subset["text_ce_bits"] else None
        if relative is None or not math.isclose(relative, expected, abs_tol=1e-10):
            raise ValueError(f"{context}:{label}: relative gap is inconsistent")
        switch = subset.get("switch_rate") or {}
        require_number(switch.get("mean"), f"{context}:{label}:switch", minimum=0, maximum=1)
    contrast_mean = require_number(contrast.get("mean"), context + ":contrast")
    validate_interval(contrast.get("ci95"), context + ":contrast CI")
    expected_contrast = correct["gap"]["mean"] - wrong["gap"]["mean"]
    if not math.isclose(contrast_mean, expected_contrast, abs_tol=1e-10):
        raise ValueError(f"{context}: contrast differs from G_correct - G_wrong")


def validate_inputs(run, artifact):
    inputs = artifact.get("inputs") or {}
    required = ("design.json", "traces.pkl", "analysis.json")
    for name in required:
        if name not in inputs:
            raise ValueError(f"{run}: accuracy artifact does not record {name}")
    for name, expected in inputs.items():
        if name == "script":
            continue
        path = run / name
        if not path.is_file():
            raise ValueError(f"{run}: recorded accuracy-gap input is missing: {name}")
        if file_hash(path) != expected:
            raise ValueError(f"{run}: accuracy-gap input changed since analysis: {name}")


def load_run(run, artifact_pattern="accuracy_gap_*.json"):
    run = Path(run)
    design_path = run / "design.json"
    if not design_path.is_file():
        raise ValueError(f"{run}: missing design.json")
    artifact_path, candidates = newest_artifact(run, artifact_pattern)
    with design_path.open() as handle:
        design = json.load(handle)
    with artifact_path.open() as handle:
        artifact = json.load(handle)
    if artifact.get("schema") != SCHEMA:
        raise ValueError(f"{artifact_path}: unsupported schema {artifact.get('schema')!r}")
    if artifact.get("design_id") != design.get("design_id"):
        raise ValueError(f"{artifact_path}: design_id does not match design.json")
    if artifact.get("gap_definition") != "text_loss_bits - joint_loss_bits":
        raise ValueError(f"{artifact_path}: unexpected gap definition")
    validate_inputs(run, artifact)
    by_cue = artifact.get("by_cue_type")
    if not isinstance(by_cue, dict) or "all" not in by_cue:
        raise ValueError(f"{artifact_path}: missing all cue stratum")
    for cue, cell in by_cue.items():
        validate_cell(cell, f"{artifact_path}:{cue}")
    return {
        "run": run.resolve(), "run_name": run.name,
        "model": design.get("model") or design.get("model_label") or run.name,
        "task": design.get("task") or design.get("bbh_task") or "--",
        "artifact_path": artifact_path.resolve(),
        "artifact_candidates": len(candidates),
        "checkpoint": artifact.get("checkpoint") or "--",
        "population": artifact.get("population"),
        "target": artifact.get("target"),
        "gap_definition": artifact.get("gap_definition"),
        "post_outcome_warning": artifact.get("post_outcome_warning"),
        "by_cue_type": by_cue,
    }


def pct(value, digits=1):
    return "--" if value is None else f"{100 * value:.{digits}f}%"


def number(value, digits=3):
    return "--" if value is None else f"{value:.{digits}f}"


def ci(value, digits=3):
    return "--" if not value else f"[{value[0]:+.{digits}f},{value[1]:+.{digits}f}]"


def direction(contrast):
    interval = contrast.get("ci95")
    if not interval:
        return "inconclusive"
    if interval[0] > 0:
        return "correct > wrong"
    if interval[1] < 0:
        return "correct < wrong"
    return "inconclusive"


def overall_row(record):
    cell = record["by_cue_type"]["all"]
    correct, wrong = cell["correct"], cell["incorrect"]
    contrast = cell["correct_minus_incorrect_gap"]
    return [
        record["model"], record["task"], record["checkpoint"],
        cell["n_items"], cell["n_rows"], pct(cell["accuracy"]["mean"]),
        correct["n_rows"], wrong["n_rows"],
        number(correct["gap"]["mean"]), number(wrong["gap"]["mean"]),
        number(contrast["mean"]), ci(contrast["ci95"]), direction(contrast),
    ]


def cue_rows(record):
    output = []
    for cue, cell in sorted(record["by_cue_type"].items(),
                            key=lambda item: (CUE_ORDER.get(item[0], 99), item[0])):
        correct, wrong = cell["correct"], cell["incorrect"]
        contrast = cell["correct_minus_incorrect_gap"]
        output.append([
            record["model"], record["task"], cue, cell["n_items"],
            pct(cell["accuracy"]["mean"]), correct["n_rows"], wrong["n_rows"],
            number(correct["gap"]["mean"]), number(wrong["gap"]["mean"]),
            number(contrast["mean"]), ci(contrast["ci95"]), direction(contrast),
        ])
    return output


def outcome_rows(record):
    output = []
    cell = record["by_cue_type"]["all"]
    for label, subset in (("correct", cell["correct"]),
                          ("wrong", cell["incorrect"])):
        output.append([
            record["model"], record["task"], label, subset["n_rows"],
            pct(subset["switch_rate"]["mean"]),
            number(subset["text_ce_bits"]), number(subset["state_ce_bits"]),
            number(subset["joint_ce_bits"]), number(subset["gap"]["mean"]),
            pct(subset["relative_gap"]), ci(subset["gap"]["ci95"]),
        ])
    return output


def check_comparability(records):
    keys = ("checkpoint", "population", "target", "gap_definition")
    for key in keys:
        values = {record[key] for record in records}
        if len(values) != 1:
            raise ValueError(f"Cross-run accuracy-gap comparison differs in {key}: {values}")


def flat_rows(records):
    output = []
    for record in records:
        for cue, cell in record["by_cue_type"].items():
            correct, wrong = cell["correct"], cell["incorrect"]
            contrast = cell["correct_minus_incorrect_gap"]
            output.append({
                "run": str(record["run"]), "run_name": record["run_name"],
                "model": record["model"], "task": record["task"],
                "checkpoint": record["checkpoint"], "cue_type": cue,
                "n_items": cell["n_items"], "n_rows": cell["n_rows"],
                "item_weighted_accuracy": cell["accuracy"]["mean"],
                "n_correct_rows": correct["n_rows"],
                "n_wrong_rows": wrong["n_rows"],
                "correct_switch_rate": correct["switch_rate"]["mean"],
                "wrong_switch_rate": wrong["switch_rate"]["mean"],
                "correct_gap_bits": correct["gap"]["mean"],
                "wrong_gap_bits": wrong["gap"]["mean"],
                "correct_relative_gap": correct["relative_gap"],
                "wrong_relative_gap": wrong["relative_gap"],
                "correct_minus_wrong_gap_bits": contrast["mean"],
                "contrast_ci_low": contrast["ci95"][0],
                "contrast_ci_high": contrast["ci95"][1],
                "direction": direction(contrast),
                "artifact": str(record["artifact_path"]),
                **{f"{label}_{channel}_ce_bits": subset[f"{channel}_ce_bits"]
                   for label, subset in (("correct", correct), ("wrong", wrong))
                   for channel in ("text", "state", "joint")},
                **{f"{label}_gap_ci_{bound}": subset["gap"]["ci95"][index]
                   for label, subset in (("correct", correct), ("wrong", wrong))
                   for index, bound in enumerate(("low", "high"))},
            })
    return output


def write_csv(path, records):
    rows = flat_rows(records)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nwrote {path}")


MODEL_LABELS = {
    "qwen2.5-7b": "Qwen2.5-7B",
    "qwen2.5-7b-instruct": "Qwen2.5-7B-IT",
    "gemma-2-9b-it": "Gemma-2-9B-IT",
    "olmo-2-1124-7b-instruct": "OLMo-2-7B-IT",
    "mistral-nemo-instruct-2407": "Mistral-Nemo-IT",
}


def write_figure(prefix, records):
    """Fixed-OOF strata and within-model contrasts; no cross-model regression."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    prefix = Path(prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 9, "pdf.fonttype": 42,
                         "ps.fonttype": 42, "axes.spines.top": False,
                         "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(10, max(3.5, .6 * len(records))),
                             layout="constrained", sharey=True)
    def point(ax, y, mean, bounds, color, marker, label):
        ax.hlines(y, *bounds, color=color, linewidth=1.3)
        ax.plot(mean, y, marker, color=color, markersize=5, label=label)
    for i, record in enumerate(records):
        cell = record["by_cue_type"]["all"]
        for offset, key, color, marker in (
                (-.13, "correct", "#0072B2", "o"),
                (.13, "incorrect", "#D55E00", "s")):
            gap = cell[key]["gap"]
            point(axes[0], i + offset, gap["mean"], gap["ci95"],
                  color, marker, key.capitalize() if i == 0 else None)
        for offset, cue, color, marker in (
                (-.20, "all", "#222222", "o"),
                (0, "authority", "#009E73", "s"),
                (.20, "sycophancy", "#CC79A7", "^")):
            if cue not in record["by_cue_type"]:
                continue
            contrast = record["by_cue_type"][cue]["correct_minus_incorrect_gap"]
            point(axes[1], i + offset, contrast["mean"], contrast["ci95"],
                  color, marker, cue.capitalize() if i == 0 else None)
    labels = [MODEL_LABELS.get(r["model"], r["model"]) for r in records]
    if len({r["task"] for r in records}) > 1:
        labels = [f'{label} / {r["task"]}' for label, r in zip(labels, records)]
    axes[0].set_yticks(range(len(records)), labels)
    axes[0].invert_yaxis()
    for ax in axes:
        ax.axvline(0, color=".5", linewidth=.8, linestyle="--")
        ax.grid(axis="x", alpha=.18)
        ax.legend(loc="upper center", bbox_to_anchor=(.5, -.18),
                  ncol=3, frameon=False, fontsize=8)
    axes[0].set_title("(a) Gap by answer correctness")
    axes[0].set_xlabel("State-Beyond-Text Gain (bits)")
    axes[1].set_title("(b) Correct minus incorrect gap")
    axes[1].set_xlabel("Difference in gain (bits)")
    for suffix in ("pdf", "png"):
        path = prefix.with_suffix("." + suffix)
        fig.savefig(path, dpi=220, bbox_inches="tight")
        print("wrote", path)
    plt.close(fig)
    prefix.with_suffix(".provenance.json").write_text(json.dumps({
        "checkpoint": records[0]["checkpoint"],
        "intervals": "95% source-item bootstrap of fixed OOF losses; pointwise, no refits",
        "sources": [{"path": str(r["artifact_path"]),
                     "sha256": file_hash(r["artifact_path"])} for r in records],
        "script_sha256": file_hash(__file__),
    }, indent=2) + "\n")


def write_latex_table(path, records):
    def escape(value):
        return str(value).replace("_", r"\_").replace("%", r"\%")
    lines = [r"% Generated by code/show_accuracy_gap.py; fixed OOF losses.",
             r"\begin{tabular}{lrrrrr}", r"\hline",
             r"Model & Correct/wrong & Acc. (\%) & $G_C$ & $G_I$ & $G_C-G_I$ [95\% CI]\\",
             r"\hline"]
    for r in records:
        cell = r["by_cue_type"]["all"]
        c, w = cell["correct"], cell["incorrect"]
        d = cell["correct_minus_incorrect_gap"]
        lines.append(
            f'{escape(MODEL_LABELS.get(r["model"], r["model"]))} & '
            f'{c["n_rows"]}/{w["n_rows"]} & {100 * cell["accuracy"]["mean"]:.1f} & '
            f'{c["gap"]["mean"]:.3f} & {w["gap"]["mean"]:.3f} & '
            f'{d["mean"]:+.3f} [{d["ci95"][0]:+.3f}, {d["ci95"][1]:+.3f}]' + r"\\")
    lines.extend([r"\hline", r"\end{tabular}"])
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
    print("wrote", path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", metavar="RUN",
                        help="run names or paths; replaces DEFAULT_RUN_DIRS")
    parser.add_argument("--artifact-pattern", default="accuracy_gap_*.json")
    parser.add_argument("--overall-only", action="store_true",
                        help="omit cue-type and outcome-detail tables")
    parser.add_argument("--csv", help="optional wide CSV export")
    parser.add_argument("--task", help="filter runs by exact frozen task name")
    parser.add_argument("--figure-prefix", type=Path,
                        help="write PDF, PNG and provenance for the contrast figure")
    parser.add_argument("--latex-table", type=Path,
                        help="write a tabular fragment for optional paper inclusion")
    args = parser.parse_args(argv)
    try:
        runs = ([resolve_run(value) for value in args.runs]
                if args.runs else [Path(path).resolve() for path in DEFAULT_RUN_DIRS])
        if args.task:
            runs = [run for run in runs
                    if json.loads((run / "design.json").read_text()).get("task") == args.task]
        if not runs:
            raise ValueError("No E47 run directories were supplied")
        records = [load_run(run, args.artifact_pattern) for run in runs]
        check_comparability(records)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))

    print("\nE47 ACCURACY-GAP SUMMARY")
    print("=" * 152)
    print("\nOVERALL CORRECTNESS MODERATION")
    print_table(
        ["model", "task", "checkpoint", "items", "responses", "item acc",
         "correct", "wrong", "G_correct", "G_wrong", "correct-wrong G",
         "95% CI", "direction"],
        [overall_row(record) for record in records])

    if not args.overall_only:
        print("\nBY CUE TYPE")
        print_table(
            ["model", "task", "cue", "items", "item acc", "correct", "wrong",
             "G_correct", "G_wrong", "correct-wrong G", "95% CI", "direction"],
            [row for record in records for row in cue_rows(record)])

        print("\nOVERALL OUTCOME STRATA")
        print_table(
            ["model", "task", "outcome", "rows", "switch%", "CE_text",
             "CE_state", "CE_joint", "G_bits", "rel_G%", "G_95CI"],
            [row for record in records for row in outcome_rows(record)])

    print(f'\n  G = fixed-OOF CE_text - CE_joint at {records[0]["checkpoint"]}; readers are not refit')
    print("  within correctness strata. The all-cue accuracy is item-weighted, while")
    print("  correct/wrong are response-row counts and therefore need not equal accuracy*n.")
    print("  Positive correct-wrong G means a larger gap on correct responses.")
    print("  Correctness is post-treatment and coupled to cue following: this is")
    print("  descriptive moderation, not a causal effect of gap on accuracy.")
    duplicates = [record for record in records if record["artifact_candidates"] > 1]
    if duplicates:
        print("  Newest artifact selected for runs with multiple files: " + ", ".join(
            f"{record['run_name']} ({record['artifact_path'].name})" for record in duplicates))

    if args.csv:
        write_csv(args.csv, records)
    if args.figure_prefix:
        write_figure(args.figure_prefix, records)
    if args.latex_table:
        write_latex_table(args.latex_table, records)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
