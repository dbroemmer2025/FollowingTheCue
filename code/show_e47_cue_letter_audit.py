#!/usr/bin/env python3
"""Display the complete E47 cue-letter sensitivity audit across runs.

By default this reports every checkpoint for the ten paper cells listed in
``run_dirs.DEFAULT_RUN_DIRS``. Examples:

    python code/show_e47_cue_letter_audit.py
    python code/show_e47_cue_letter_audit.py --checkpoint primary
    python code/show_e47_cue_letter_audit.py --checkpoint initial/pre_answer
    python code/show_e47_cue_letter_audit.py --runs e47/run_a e47/run_b
    python code/show_e47_cue_letter_audit.py --csv reports/cue_letters.csv

The audit reuses fixed grouped out-of-fold predictions. It does not refit a
reader for each letter, so letter-specific intervals are descriptive
sensitivity estimates rather than independent confirmatory tests.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from run_dirs import DEFAULT_RUN_DIRS


ROOT = Path(__file__).resolve().parents[1]


def fmt(value, digits=3, missing="--"):
    if value is None:
        return missing
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    if isinstance(value, (list, tuple)):
        return "[" + ",".join(fmt(item, digits, missing) for item in value) + "]"
    return str(value)


def pct(value):
    return "--" if value is None else f"{100.0 * value:.1f}%"


def rel_pct(numerator, denominator):
    if numerator is None or denominator in (None, 0):
        return "--"
    return f"{100.0 * numerator / denominator:+.1f}%"


def ci_text(value):
    if not value or len(value) != 2:
        return "--"
    return f"[{value[0]:+.3f},{value[1]:+.3f}]"


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


def newest_audit(run):
    candidates = [path for path in run.glob("cue_letter_audit_*.json")
                  if path.is_file()]
    if not candidates:
        raise ValueError(f"{run}: no cue_letter_audit_*.json file found")
    return max(candidates,
               key=lambda path: (path.stat().st_mtime_ns, path.name))


def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_run(run):
    design_path = run / "design.json"
    if not design_path.is_file():
        raise ValueError(f"{run}: design.json is missing")
    with design_path.open() as handle:
        design = json.load(handle)
    audit_path = newest_audit(run)
    with audit_path.open() as handle:
        audit = json.load(handle)
    if audit.get("schema") != "e47_cue_letter_audit_v1":
        raise ValueError(f"{audit_path}: unsupported schema {audit.get('schema')!r}")
    if audit.get("design_id") != design.get("design_id"):
        raise ValueError(f"{audit_path}: design_id does not match design.json")
    oof_names = [name for name in (audit.get("inputs") or {})
                 if name.endswith(".csv")]
    if len(oof_names) != 1:
        raise ValueError(
            f"{audit_path}: expected exactly one OOF CSV input, found {oof_names}")
    oof_path = run / oof_names[0]
    if not oof_path.is_file():
        raise ValueError(f"{audit_path}: OOF input is missing: {oof_path}")
    expected_hash = audit["inputs"][oof_names[0]]
    if expected_hash and file_hash(oof_path) != expected_hash:
        raise ValueError(f"{audit_path}: OOF input hash differs: {oof_path}")
    return {
        "run": run,
        "design": design,
        "audit": audit,
        "audit_path": audit_path,
        "oof_path": oof_path,
        "model": (design.get("model_label") or design.get("model") or
                  run.name),
        "task": (design.get("task") or design.get("bbh_task") or "--"),
    }


def selected_checkpoints(bundle, requested):
    checkpoints = bundle["audit"].get("checkpoints") or {}
    if requested == "all":
        return list(checkpoints)
    if requested == "primary":
        name = bundle["audit"].get("primary")
        if not name:
            raise ValueError(f"{bundle['audit_path']}: primary checkpoint absent")
        requested = name
    if requested not in checkpoints:
        available = ", ".join(checkpoints)
        raise ValueError(
            f"{bundle['run']}: checkpoint {requested!r} is absent; "
            f"available: {available}")
    return [requested]


def flatten(bundles, checkpoint):
    metadata = []
    controls = []
    letters = []
    for bundle in bundles:
        audit = bundle["audit"]
        metadata.append({
            "model": bundle["model"], "task": bundle["task"],
            "run": bundle["run"].name, "audit": bundle["audit_path"].name,
            "primary": audit.get("primary"),
            "changed_code": ",".join(audit.get("changed_code") or ()),
            "signature": audit.get("signature"),
        })
        for checkpoint_name in selected_checkpoints(bundle, checkpoint):
            sensitivity = audit["checkpoints"][checkpoint_name]
            for control_name, values in sorted(
                    (sensitivity.get("controls") or {}).items()):
                row = {
                    "model": bundle["model"], "task": bundle["task"],
                    "checkpoint": checkpoint_name, "control": control_name,
                    "ce": values.get("ce"), "auc": values.get("auc"),
                }
                for channel in ("text", "state", "joint"):
                    gain = (values.get("gains") or {}).get(channel) or {}
                    row[f"{channel}_gain"] = gain.get("mean")
                    row[f"{channel}_gain_ci95"] = gain.get("ci95")
                    row[f"{channel}_gain_n_items"] = gain.get("n_items")
                controls.append(row)
            for letter, values in sorted(
                    (sensitivity.get("by_letter") or {}).items()):
                gap = values.get("gap") or {}
                joint_gain = values.get(
                    "joint_gain_over_cue_letter_control") or {}
                letters.append({
                    "model": bundle["model"], "task": bundle["task"],
                    "checkpoint": checkpoint_name, "cue_letter": letter,
                    "status": values.get("status"),
                    "n_rows": values.get("n_rows"),
                    "n_items": values.get("n_items"),
                    "class_rows": values.get("class_rows"),
                    "class_items": values.get("class_items"),
                    "positive_rate": values.get("positive_rate"),
                    "ce_control": values.get("ce_cue_letter_control"),
                    "ce_text": values.get("ce_text"),
                    "ce_state": values.get("ce_state"),
                    "ce_joint": values.get("ce_joint"),
                    "auc_text": values.get("auc_text"),
                    "auc_state": values.get("auc_state"),
                    "auc_joint": values.get("auc_joint"),
                    "gap": gap.get("mean"), "gap_ci95": gap.get("ci95"),
                    "gap_n_items": gap.get("n_items"),
                    "joint_gain_over_control": joint_gain.get("mean"),
                    "joint_gain_over_control_ci95": joint_gain.get("ci95"),
                    "joint_gain_over_control_n_items": joint_gain.get("n_items"),
                })
    return metadata, controls, letters


def fold_local_letter_loss(rows):
    """Reconstruct the signed audit's model-local cue-letter control loss."""
    target = np.asarray([int(row["target"]) for row in rows], dtype=int)
    folds = np.asarray([int(row["fold"]) for row in rows], dtype=int)
    letters = np.asarray([row["cue_letter"].upper() for row in rows])
    prediction = np.empty(len(rows), dtype=float)
    for fold in np.unique(folds):
        test = folds == fold
        train = ~test
        global_rate = (float(target[train].sum()) + .5) / (int(train.sum()) + 1.)
        rates = {}
        for letter in np.unique(letters[train]):
            selected = train & (letters == letter)
            rates[letter] = ((float(target[selected].sum()) + .5) /
                             (int(selected.sum()) + 1.))
        prediction[test] = [rates.get(letter, global_rate)
                            for letter in letters[test]]
    prediction = np.clip(prediction, 1e-6, 1 - 1e-6)
    return -(target * np.log2(prediction) +
             (1 - target) * np.log2(1 - prediction))


def grouped_interval(values, groups, seed, n_boot):
    values = np.asarray(values, dtype=float)
    groups = np.asarray(groups)
    means = np.asarray([
        values[groups == group].mean() for group in np.unique(groups)])
    rng = np.random.default_rng(seed)
    draws = rng.choice(means, (n_boot, len(means)), replace=True).mean(axis=1)
    return {
        "mean": float(means.mean()),
        "ci95": np.quantile(draws, [.025, .975]).tolist(),
        "n_items": int(len(means)),
    }


def aggregate_letter_gaps(bundles, checkpoint):
    """Pool fixed OOF losses across model-item groups for each task/letter."""
    cells = {}
    for model_index, bundle in enumerate(bundles):
        wanted = set(selected_checkpoints(bundle, checkpoint))
        with bundle["oof_path"].open(newline="") as handle:
            rows = [row for row in csv.DictReader(handle)
                    if row["checkpoint"] in wanted]
        by_checkpoint = {}
        for row in rows:
            by_checkpoint.setdefault(row["checkpoint"], []).append(row)
        for checkpoint_name, selected in by_checkpoint.items():
            control_loss = fold_local_letter_loss(selected)
            for row, ctl_loss in zip(selected, control_loss):
                key = (bundle["task"], checkpoint_name,
                       row["cue_letter"].upper())
                cell = cells.setdefault(key, {
                    "models": set(), "groups": [], "text": [], "gap": [],
                    "joint_gain_over_control": []})
                cell["models"].add(bundle["model"])
                cell["groups"].append(
                    f"{model_index}:{bundle['run'].name}:{row['source_id']}")
                text_loss = float(row["text_loss_bits"])
                joint_loss = float(row["joint_loss_bits"])
                cell["text"].append(text_loss)
                cell["gap"].append(text_loss - joint_loss)
                cell["joint_gain_over_control"].append(ctl_loss - joint_loss)

    aggregates = []
    for (task, checkpoint_name, letter), cell in sorted(cells.items()):
        groups = np.asarray(cell["groups"])
        design = next(bundle["design"] for bundle in bundles
                      if bundle["task"] == task)
        policy = design.get("analysis") or {}
        seed = int(policy.get("seed", 47))
        n_boot = int(policy.get("bootstrap", 2000))
        text = grouped_interval(cell["text"], groups, seed, n_boot)
        gap = grouped_interval(cell["gap"], groups, seed, n_boot)
        gain = grouped_interval(
            cell["joint_gain_over_control"], groups, seed, n_boot)
        aggregates.append({
            "task": task, "checkpoint": checkpoint_name,
            "cue_letter": letter, "n_models": len(cell["models"]),
            "n_items": gap["n_items"], "ce_text": text["mean"],
            "gap": gap["mean"], "gap_ci95": gap["ci95"],
            "joint_gain_over_control": gain["mean"],
            "joint_gain_over_control_ci95": gain["ci95"],
        })
    return aggregates


def write_csv(path, rows):
    if not rows:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    serializable = []
    for row in rows:
        serializable.append({
            key: (json.dumps(value, separators=(",", ":"))
                  if isinstance(value, (list, tuple, dict)) else value)
            for key, value in row.items()})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(serializable[0]))
        writer.writeheader()
        writer.writerows(serializable)
    print(f"\nwrote {path}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runs", nargs="+", metavar="DIR",
        help="run directories; replaces the ten DEFAULT_RUN_DIRS")
    parser.add_argument(
        "--checkpoint", default="all",
        help="checkpoint name, 'primary', or 'all' (default: all)")
    parser.add_argument(
        "--csv", help="optional CSV export of the per-letter table")
    parser.add_argument(
        "--controls-csv", help="optional CSV export of nuisance controls")
    args = parser.parse_args(argv)

    runs = ([resolve_run(value) for value in args.runs]
            if args.runs else list(DEFAULT_RUN_DIRS))
    bundles = []
    errors = []
    for run in runs:
        try:
            bundles.append(load_run(Path(run)))
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            errors.append(str(exc))
    if errors:
        parser.error("\n  ".join(errors))
    metadata, controls, letters = flatten(bundles, args.checkpoint)
    aggregates = aggregate_letter_gaps(bundles, args.checkpoint)

    print("\nE47 CUE-LETTER AUDIT")
    print("=" * 132)
    print("\nAUDIT SOURCES")
    print_table(
        ["model", "task", "primary", "audit", "code drift"],
        [[row["model"], row["task"], row["primary"], row["audit"],
          row["changed_code"] or "none"] for row in metadata])

    print("\nFOLD-LOCAL CATEGORICAL NUISANCE CONTROLS")
    print_table(
        ["model", "task", "checkpoint", "control", "CE", "AUC",
         "text gain", "text 95CI", "state gain", "state 95CI",
         "joint gain", "joint 95CI"],
        [[row["model"], row["task"], row["checkpoint"], row["control"],
          row["ce"], row["auc"], row["text_gain"],
          ci_text(row["text_gain_ci95"]), row["state_gain"],
          ci_text(row["state_gain_ci95"]), row["joint_gain"],
          ci_text(row["joint_gain_ci95"])] for row in controls])

    print("\nPER-LETTER ASSIGNMENT AND OUTCOME BALANCE")
    print_table(
        ["model", "task", "checkpoint", "cue", "rows", "items",
         "class rows [0,1]", "class items [0,1]", "switch%", "status"],
        [[row["model"], row["task"], row["checkpoint"], row["cue_letter"],
          row["n_rows"], row["n_items"], row["class_rows"],
          row["class_items"], pct(row["positive_rate"]), row["status"]]
         for row in letters])

    print("\nPER-LETTER HELD-OUT READER PERFORMANCE")
    print_table(
        ["model", "task", "checkpoint", "cue", "CE_ctl", "CE_text",
         "CE_state", "CE_joint", "AUC_text", "AUC_state", "AUC_joint"],
        [[row["model"], row["task"], row["checkpoint"], row["cue_letter"],
          row["ce_control"], row["ce_text"], row["ce_state"],
          row["ce_joint"], row["auc_text"], row["auc_state"],
          row["auc_joint"]] for row in letters])

    print("\nPER-LETTER GAP SENSITIVITY")
    print_table(
        ["model", "task", "checkpoint", "cue", "G_bits", "rel_G%",
         "G_95CI", "G items", "joint-ctl", "joint-ctl 95CI",
         "gain items"],
        [[row["model"], row["task"], row["checkpoint"], row["cue_letter"],
          row["gap"], rel_pct(row["gap"], row["ce_text"]),
          ci_text(row["gap_ci95"]), row["gap_n_items"],
          row["joint_gain_over_control"],
          ci_text(row["joint_gain_over_control_ci95"]),
          row["joint_gain_over_control_n_items"]] for row in letters])

    print("\nPER-LETTER GAP SENSITIVITY AGGREGATED ACROSS MODELS")
    print_table(
        ["task", "checkpoint", "cue", "models", "model-items", "G_bits",
         "rel_G%", "G_95CI", "joint-ctl", "joint-ctl 95CI"],
        [[row["task"], row["checkpoint"], row["cue_letter"],
          row["n_models"], row["n_items"], row["gap"],
          rel_pct(row["gap"], row["ce_text"]), ci_text(row["gap_ci95"]),
          row["joint_gain_over_control"],
          ci_text(row["joint_gain_over_control_ci95"])]
         for row in aggregates])
    print("\n  Aggregates pool the fixed OOF losses over model-item groups. Each model's")
    print("  cue-letter control is reconstructed within its own folds before pooling;")
    print("  intervals bootstrap model-item groups and do not refit any reader.")

    print("\nINTERPRETATION")
    print("  target=1 means switch-to-cue; switch% is its prevalence within a cue letter.")
    print("  The categorical controls use only cue letter, or cue type x cue letter,")
    print("  with rates estimated from other OOF folds. Positive reader gain means")
    print("  lower held-out CE than that nuisance baseline.")
    print("  G_bits = CE_text - CE_joint. Letter rows stratify fixed grouped OOF")
    print("  predictions after fitting; they are sensitivity estimates, not separately")
    print("  fitted readers or independent confirmatory tests.")
    print("  Class-item counts need not sum because one item can contribute both classes.")

    if args.csv:
        write_csv(args.csv, letters)
    if args.controls_csv:
        write_csv(args.controls_csv, controls)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
