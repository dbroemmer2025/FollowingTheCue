#!/usr/bin/env python3
"""Plot the Luna-assisted output-aware RC reader comparison.

The direct-effect reader results use model-specific log-odds scales.  This
figure therefore emphasizes two comparable summaries: out-of-fold R^2 for the
reader families, and the residual increment as a share of the joint reader's
total improvement over the no-feature baseline.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "reports" / "de_readers" / "results"
DEFAULT_OUTPUT = ROOT / "figures" / "fig_e47_rc_output_access"

MODEL_NAMES = {
    "qwen2_5_7b": "Qwen2.5-7B",
    "qwen2_7b_it": "Qwen2.5-7B-Instruct",
    "gemma_2_9b_it": "Gemma-2-9B-IT",
    "mistral_nemo_instruct_2407": "Mistral-Nemo-Instruct",
    "olmo_2_1124_7b_instruct": "OLMo-2-7B-Instruct",
}
TASK_NAMES = {
    "movie_recommendations": "Movie recommendation",
    "temporal_sequences": "Temporal sequences",
}
COLORS = {
    "text_out": "#35618D",
    "state": "#D17A22",
    "joint_out": "#2E7D6E",
}


def load(path: Path) -> dict:
    with path.open() as handle:
        return json.load(handle)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_cell(path: Path, payload: dict) -> tuple[str, str]:
    """Recover model and task from the stable result filename."""
    stem = path.name.removesuffix("__output.json")
    for task in TASK_NAMES:
        suffix = f"_{task}"
        if stem.endswith(suffix):
            model = stem.removesuffix(suffix)
            return MODEL_NAMES.get(model, model.replace("_", " ")), TASK_NAMES[task]
    return str(payload.get("cell", stem)), ""


def rows(input_dir: Path) -> tuple[list[dict], list[Path]]:
    paths = sorted(input_dir.glob("*__output.json"))
    if len(paths) != 10:
        raise ValueError(f"Expected 10 *__output.json files in {input_dir}; found {len(paths)}")
    inputs = []
    result = []
    for path in paths:
        payload = load(path)
        target = payload.get("targets", {}).get("RC")
        if not target or not payload.get("output_access_control"):
            raise ValueError(f"Missing output-aware RC reader data in {path}")
        inputs.append(path)
        model, task = parse_cell(path, payload)
        families = target["families"]
        extra = target["state_beyond_text_and_output"]
        joint_gain = float(families["joint_out"]["gain_over_baseline"])
        delta = float(extra["delta"])
        ci = [float(x) for x in extra["ci95"]]
        result.append({
            "path": path,
            "model": model,
            "task": task,
            "r2_text_out": float(families["text_out"]["r2"]),
            "r2_state": float(families["state"]["r2"]),
            "r2_joint_out": float(families["joint_out"]["r2"]),
            "share": 100.0 * float(extra["share_of_joint_gain"]),
            "ci_share": [100.0 * ci[0] / joint_gain, 100.0 * ci[1] / joint_gain],
            "verdict": str(extra["verdict"]),
        })
    order = {"Movie recommendation": 0, "Temporal sequences": 1}
    result.sort(key=lambda row: (order.get(row["task"], 99), row["model"]))
    return result, inputs


def configure_style() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 7,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 160,
        "savefig.dpi": 300,
    })


def plot(data: list[dict], output: Path, inputs: list[Path]) -> None:
    configure_style()
    y = np.arange(len(data))[::-1]
    labels = [("Movie: " if row["task"] == "Movie recommendation" else "Temporal: ") + row["model"] for row in data]
    fig, (left, right) = plt.subplots(
        1, 2, figsize=(6.9, 3.9), sharey=True,
        gridspec_kw={"width_ratios": [1.12, 0.88], "wspace": 0.12},
    )

    # Panel A: output-aware reader fit. Lines emphasize the strict comparison.
    for yi, row in zip(y, data):
        left.plot(
            [row["r2_text_out"], row["r2_joint_out"]], [yi, yi],
            color="#B9C4CF", lw=1.4, zorder=1,
        )
    for key, label, marker in (
        ("r2_text_out", "Text + output + Luna", "o"),
        ("r2_state", "Residual state", "s"),
        ("r2_joint_out", "Text + output + Luna + state", "D"),
    ):
        left.scatter(
            [row[key] for row in data], y, s=31, marker=marker,
            color=COLORS[{"r2_text_out": "text_out", "r2_state": "state",
                          "r2_joint_out": "joint_out"}[key]], edgecolor="white",
            linewidth=0.55, label=label, zorder=3,
        )
    left.set_xlim(0, 0.76)
    left.set_xlabel("Held-out $R^2$ for relocation contrast")
    left.set_yticks(y, labels)
    left.grid(axis="x", color="#E1E5E9", lw=0.7)
    left.set_title("A. Relocation-contrast (RC) prediction", loc="left", fontweight="bold")

    # Panel B: conditional state contribution normalized by the joint gain.
    for yi, row in zip(y, data):
        low, high = row["ci_share"]
        color = COLORS["joint_out"] if row["verdict"] == "state adds beyond text+output" else "#A0A7AE"
        right.errorbar(
            row["share"], yi,
            xerr=np.array([[row["share"] - low], [high - row["share"]]]),
            fmt="o", ms=5.2, color=color, ecolor=color, capsize=2.5,
            elinewidth=1.25, zorder=3,
        )
    right.axvline(10, color="#777777", lw=1, ls="--", zorder=1)
    # Keep every bootstrap interval inside the panel.  OLMo temporal, for
    # example, has an upper bound above the original fixed 75% limit.
    max_ci_share = max(row["ci_share"][1] for row in data)
    right_limit = max(75.0, 5.0 * math.ceil((max_ci_share + 4.0) / 5.0))
    right.set_xlim(-8, right_limit)
    right.set_xlabel("Residual increment (% of joint gain)")
    right.grid(axis="x", color="#E1E5E9", lw=0.7)
    right.set_title("B. Residual increment", loc="left", fontweight="bold")
    right.tick_params(axis="y", left=False, labelleft=False)

    first_temporal = next((i for i, row in enumerate(data)
                           if row["task"] == "Temporal sequences"), None)
    if first_temporal is not None:
        boundary = (y[first_temporal - 1] + y[first_temporal]) / 2
        for axis in (left, right):
            axis.axhline(boundary, color="#BFC6CE", lw=0.8, zorder=0)

    handles, legend_labels = left.get_legend_handles_labels()
    fig.legend(
        handles, legend_labels, loc="lower center", ncol=3,
        frameon=False, handletextpad=0.35, columnspacing=1.1,
        bbox_to_anchor=(0.5, 0.018),
    )
    #fig.text(
    #    0.01, 0.002,
    #    "Each row is a model--task cell. Error bars are source-item bootstrap 95% intervals. "
    #    "RC is evaluated after factual output features are already available to the text side.",
    #    fontsize=6.8, color="#555555", ha="left",
    #)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.subplots_adjust(bottom=0.19)
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output.with_suffix(".png"), bbox_inches="tight")
    plt.close(fig)

    provenance = {
        "generator": Path(__file__).name,
        "description": (
            "Luna-assisted output-aware reader comparison for the relocation-contrast (RC) target. "
            "Panel A reports held-out R^2; panel B reports the residual increment as "
            "a share of joint gain with source-item bootstrap intervals."
        ),
        "inputs": [
            {"file": str(path.relative_to(ROOT)), "sha256": sha256(path)}
            for path in inputs
        ],
        "outputs": [
            str(output.with_suffix(".pdf").relative_to(ROOT)),
            str(output.with_suffix(".png").relative_to(ROOT)),
        ],
        "rows": [{key: str(value) if isinstance(value, Path) else value
                  for key, value in row.items()} for row in data],
    }
    with output.with_suffix(".prov.json").open("w") as handle:
        json.dump(provenance, handle, indent=2)
    print(f"wrote {output.with_suffix('.pdf')}")
    print(f"wrote {output.with_suffix('.png')}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    data, inputs = rows(args.input_dir)
    plot(data, args.output, inputs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
