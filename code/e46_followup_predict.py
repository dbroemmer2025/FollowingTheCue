#!/usr/bin/env python3
"""Predict later cue citation from cued q_end rows; CPU-only, nested item CV.

This forecasts reporting behavior. It does not measure whether a cue caused
the earlier answer, suppression, or a completed-trace verbalization gap.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import pickle

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from sklearn.pipeline import make_pipeline
from threadpoolctl import threadpool_limits

from e46_followup_gap import join
from e46_readers import VERSION, holm, loss_rows, matrix_reader, nested_oof


def collect(furesid, followups):
    rows, meta = join(furesid, followups)
    with open(followups, "rb") as f:
        fu = pickle.load(f)
    by = {(r["index"], r["arm"]): r for r in fu["rows"]}
    with open(furesid, "rb") as f:
        cache = pickle.load(f)
    selected = sorted([r for r in rows if r["anchor"] == "q_end"
                       and r["arm"] == "cued" and r["stratum"] != "other"],
                      key=lambda r: r["index"])
    if len(selected) < 20:
        raise ValueError("Need at least 20 q_end cued items")
    y = np.array([r["mentions_cue_turn2"] for r in selected], int)
    if min(np.bincount(y, minlength=2)) < 5:
        raise ValueError("Need at least five items per mention class")
    from e46_prompts_cue import episode_span
    # Identical source queries stay together even if their options are relabelled.
    group = []
    for r in selected:
        p = by[(r["index"], "cued")]["prompt1"]
        query = p.rsplit("\nQ:", 1)[-1].split("Options:", 1)[0].strip()
        group.append(hashlib.sha256(query.encode()).hexdigest())
    nuisance = [f"cue_{r['cue_letter']} answer_{r['answer']} "
                f"stratum_{r['stratum']} "
                f"episode_length_bin_{len(episode_span(by[(r['index'], 'cued')]['episode1'])) // 100} "
                f"demo_rotation_{r['index'] % 20}" for r in selected]
    X = {
        "state": cache["R"][[r["residual_row"] for r in selected]].astype(float),
        "text": [by[(r["index"], "cued")]["prompt2"] for r in selected],
        "cue_margin": np.array([r["cue_margin"] for r in selected]).reshape(-1, 1),
        "nuisance": nuisance,
    }
    return selected, y, np.array(group), X, meta


def candidates(channel):
    def lr(c):
        return LogisticRegression(C=c, max_iter=1000, random_state=46)
    if channel == "state":
        return [(f"pca{d}_C{c}", lambda d=d, c=c:
                 matrix_reader(lambda: lr(c), d))
                for d in (8, 16) for c in (0.01, 0.1, 1.0)]
    if channel in ("text", "nuisance"):
        kinds = ("word", "char") if channel == "text" else ("word",)
        return [(f"{kind}_C{c}", lambda kind=kind, c=c: make_pipeline(
            TfidfVectorizer(analyzer="char" if kind == "char" else "word",
                            ngram_range=(3, 5) if kind == "char" else (1, 2),
                            max_features=20000), lr(c)))
                for kind in kinds for c in (0.1, 1.0, 10.0)]
    return [(f"scalar_C{c}", lambda c=c: matrix_reader(lambda: lr(c)))
            for c in (0.01, 0.1, 1.0)]


def evaluate(X, y, groups, seed):
    return {k: nested_oof(x, y, groups, 2, candidates(k), seed=seed)
            for k, x in X.items()}


def paired_ci(values, groups, seed=46, n_boot=8000):
    # One draw per source item, not per CV repeat or token anchor.
    unique = np.unique(groups)
    means = np.array([np.mean(values[groups == g]) for g in unique])
    rng = np.random.default_rng(seed)
    draws = rng.choice(means, (n_boot, len(means)), replace=True).mean(1)
    return [float(v) for v in np.quantile(draws, [0.025, 0.975])]


def shuffled_labels(y, groups, rng):
    unique = np.unique(groups)
    # Each source has one label in this run. Refuse ambiguous grouped nulls.
    if any(len(np.unique(y[groups == g])) != 1 for g in unique):
        raise ValueError("Repeated source items have inconsistent labels; define a block null first")
    labels = rng.permutation([y[groups == g][0] for g in unique])
    by = dict(zip(unique, labels))
    return np.array([by[g] for g in groups])


def run(args):
    rows, y, groups, X, meta = collect(args.furesid, args.followups)
    print(f"q_end forecasting: {len(y)} items, {y.sum()} citations; nested CV", flush=True)
    result = evaluate(X, y, groups, args.seed)
    base = result["state"]["base"]
    loss = {k: loss_rows(y, v["P"]) for k, v in result.items()}
    loss["base"] = loss_rows(y, base)
    summary = {}
    for k in loss:
        p = result[k]["P"] if k != "base" else base
        delta = loss["base"] - loss[k]
        summary[k] = {"ce_bits": float(loss[k].mean()),
                      "auc": float(roc_auc_score(y, p[:, 1])),
                      "balanced_accuracy": float(balanced_accuracy_score(y, p.argmax(1))),
                      "gain_over_base_bits": float(delta.mean()),
                      "gain_ci": paired_ci(delta, groups),
                      "fold_selection": result[k]["details"] if k != "base" else []}
    comparisons = {"state_vs_base": loss["base"] - loss["state"],
                   "state_vs_text": loss["text"] - loss["state"]}
    tests = {k: {"gain_bits": float(v.mean()), "ci": paired_ci(v, groups)}
             for k, v in comparisons.items()}
    nulls = {k: [] for k in tests}
    rng = np.random.default_rng(args.seed + 1000)
    for i in range(args.n_perm):
        yp = shuffled_labels(y, groups, rng)
        # Refit the entire feature, model-selection, and calibration pipeline.
        perm = evaluate(X, yp, groups, args.seed)
        sl = loss_rows(yp, perm["state"]["P"])
        nulls["state_vs_base"].append(float((loss_rows(yp, perm["state"]["base"]) - sl).mean()))
        nulls["state_vs_text"].append(float((loss_rows(yp, perm["text"]["P"]) - sl).mean()))
        if (i + 1) % 10 == 0:
            print(f"  refitted null {i + 1}/{args.n_perm}", flush=True)
    if args.n_perm:
        for k, test in tests.items():
            test["permutation_p"] = (1 + sum(v >= test["gain_bits"] for v in nulls[k])) / (args.n_perm + 1)
        adjusted = holm([t["permutation_p"] for t in tests.values()])
        for test, p in zip(tests.values(), adjusted):
            test["holm_p"] = p
    report = {"analysis": getattr(args, "analysis_label", "early citation forecasting, exploratory existing data"),
              "evaluation_version": VERSION, "target": "mentions_cue_turn2",
              "anchor": "q_end", "arm": "cued", "n": len(y), "n_positive": int(y.sum()),
              "n_source_groups": len(np.unique(groups)), "seed": args.seed,
              "n_perm": args.n_perm, "join_audit": meta["join_audit"],
              "permutation_scope": "no-association label-shuffle null; state_vs_text is not a test of equal predictive performance under shared signal",
              "channels": summary, "comparisons": tests, "permutation_nulls": nulls,
              "ci_scope": "source-item bootstrap of fixed OOF losses; excludes refit uncertainty",
              "nuisance_scope": "diagnostic only; stratum uses counterfactual arm; rotation assumes current 20-item schedule",
              "interpretation": "Forecasting is not evidence of earlier causal knowledge or suppression."}
    print("\nreader          CE bits     AUC   balanced acc    gain/base bits")
    for k, v in summary.items():
        print(f"{k:<15} {v['ce_bits']:>7.3f}  {v['auc']:>6.3f}       {v['balanced_accuracy']:>6.3f}          {v['gain_over_base_bits']:>+7.3f}")
    for k, t in tests.items():
        print(k, t)
    print(report["interpretation"])
    output = Path(args.out_prefix)
    output.parent.mkdir(parents=True, exist_ok=True)
    with Path(str(output) + ".json").open("w") as f:
        json.dump(report, f, indent=2, allow_nan=False)
    fields = ["index", "source_group", "stratum", "target", "fold", "p_base", "loss_base_bits"]
    for k in result:
        fields += [f"p_{k}", f"loss_{k}_bits"]
    with Path(str(output) + "_oof.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for i, row in enumerate(rows):
            d = {"index": row["index"], "source_group": groups[i], "stratum": row["stratum"],
                 "target": int(y[i]), "fold": int(result["state"]["fold"][i]),
                 "p_base": base[i, 1], "loss_base_bits": loss["base"][i]}
            for k in result:
                d[f"p_{k}"] = result[k]["P"][i, 1]
                d[f"loss_{k}_bits"] = loss[k][i]
            writer.writerow(d)
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--furesid", required=True)
    ap.add_argument("--followups", required=True)
    ap.add_argument("--out-prefix", required=True)
    ap.add_argument("--n-perm", type=int, default=99)
    ap.add_argument("--seed", type=int, default=46)
    a = ap.parse_args()
    if a.n_perm < 0:
        ap.error("--n-perm must be nonnegative")
    for suffix in (".json", "_oof.csv"):
        if Path(a.out_prefix + suffix).exists():
            ap.error(f"Output exists: {a.out_prefix + suffix}; choose a new prefix")
    with threadpool_limits(limits=1):
        run(a)


if __name__ == "__main__":
    main()
