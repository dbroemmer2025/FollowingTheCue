#!/usr/bin/env python3
"""How large must n be before the state reader can resolve the effects we see?

    python code/e46_floor_vs_n.py --resid e46/qwen_fresh_01/residuals_all_layers.pkl \\
        --traces e46/qwen_fresh_01/traces.pkl --json reports/e46_floor_vs_n.json

At n=54 the measured empirical floor is AUC 0.85, while the unfitted logit lens
finds candidate signals at AUC 0.65-0.68. Every turn-1 null therefore sits in
the gap between what is plausibly present and what the reader can resolve. The
question this answers is how much data closes that gap -- BEFORE committing GPU
budget, not after.

HOW IT SIMULATES LARGER n, AND WHY THAT IS AN APPROXIMATION

Real residuals at one anchor number ~100 rows. To ask about n=400 we need a
generative model of the residual geometry, fitted here by probabilistic PCA on
ALL 1795 captured rows pooled across anchors and arms -- far better conditioned
than the 54 rows of the primary contrast, and the same model, layer and capture
throughout. Work happens in the k-dimensional PPCA coordinate space: the
synthetic data lives exactly in that subspace, so representing it in k
coordinates rather than 3584 is an isometry and preserves the p-greater-than-n
difficulty that makes the real problem hard.

Three caveats travel with every number here:

  * The covariance is estimated, not known. Pooling across anchors assumes they
    share a geometry; they are the same layer of the same model, but this is an
    assumption, not a measurement.
  * PPCA is Gaussian. Real residuals are not.
  * Planting happens along the empirical cue-letter discriminant, so the answer
    is "n for a signal shaped like one the model actually carries". A
    differently shaped signal will need a different n.

Treat the output as an order-of-magnitude planning guide -- is the answer 200
or 2000 -- and not as a power calculation.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import pickle
import sys

import numpy as np
from statistics import NormalDist

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def ppca(R, k, seed=0):
    """Mean, component scale and residual noise for a rank-k Gaussian model."""
    from sklearn.decomposition import PCA
    fit = PCA(n_components=k, random_state=seed).fit(R)
    spectrum = fit.explained_variance_
    # Isotropic noise for everything the k components do not span, which is
    # what keeps the synthetic data from being exactly rank k.
    left = max(0.0, (np.var(R, axis=0).sum() - spectrum.sum()) / max(1, R.shape[1] - k))
    return {"scale": np.sqrt(spectrum), "noise": math.sqrt(left), "k": k,
            "explained": float(fit.explained_variance_ratio_.sum()),
            "components": fit.components_}


def synth(model, n, rng):
    """Rows in PPCA coordinate space: k structured dims plus isotropic noise."""
    k = model["k"]
    return (rng.normal(size=(n, k)) * model["scale"]
            + rng.normal(size=(n, k)) * model["noise"])


def floor_at(n, model, direction_k, positive_rate, strengths, reps, n_null,
             ladder, seed):
    """Weakest planted effect resolved at this n, as an unbroken run."""
    import e46_gap as G
    from e46_gap import _state_reader, monotone_floor
    from p2_common import entropy_bits_miller_madow

    rng = np.random.default_rng(seed)
    normal = NormalDist()
    curve = []
    for target in strengths:
        rhos = []
        for rep in range(reps):
            X = synth(model, n, rng)
            z = (rng.random(n) < positive_rate).astype(int)
            if len(np.unique(z)) < 2:
                continue
            g = np.arange(n)
            _, H = entropy_bits_miller_madow(z, 2)
            v = direction_k / np.linalg.norm(direction_k)
            shift = math.sqrt(2) * normal.inv_cdf(target) * (X @ v).std()
            planted = X + shift * z[:, None] * v[None, :]
            rho = _state_reader(planted, z, g, 2, H, ladder,
                                split_seed=seed + 7 * rep)["rho"]
            null = [_state_reader(X, rng.permutation(z), g, 2, H, ladder,
                                  split_seed=seed + 7 * rep)["rho"]
                    for _ in range(n_null)]
            rhos.append((rho, float(np.quantile(null, 0.95)) if null else 0.0))
        if not rhos:
            continue
        hits = sum(r > t for r, t in rhos)
        curve.append({"planted_auc": target,
                      "rho_median": float(np.median([r for r, _ in rhos])),
                      "null_p95_median": float(np.median([t for _, t in rhos])),
                      "detected_frac": hits / len(rhos),
                      "detected": hits * 2 >= len(rhos)})
    floor, monotone = monotone_floor(curve, strengths)
    return {"n": n, "curve": curve, "floor_auc": floor, "monotone": monotone}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--resid", required=True)
    ap.add_argument("--traces", required=True)
    ap.add_argument("--anchor", default="prompt")
    ap.add_argument("--arm", default="cued")
    ap.add_argument("--k", type=int, default=64, help="PPCA rank")
    ap.add_argument("--sizes", type=int, nargs="+",
                    default=[54, 100, 200, 400, 800])
    ap.add_argument("--strengths", type=float, nargs="+",
                    default=[0.85, 0.75, 0.65, 0.60])
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--n-null", type=int, default=2)
    ap.add_argument("--ladder", action="store_true")
    ap.add_argument("--seed", type=int, default=46)
    ap.add_argument("--json", default=None)
    a = ap.parse_args(argv)

    from e46_gap import build_labels, select, discriminant_direction

    with open(a.resid, "rb") as f:
        blob = pickle.load(f)
    R = blob["R"].astype(np.float64)
    model = ppca(R - R.mean(axis=0), a.k, a.seed)

    # The planted direction is the one the model demonstrably uses for the cue
    # letter, expressed in PPCA coordinates. A design fact, never the target.
    labels, _ = build_labels(a.traces)
    sel, z, _ = select(labels, "flipper", "already_cue_wrong")
    want = {l["index"]: i for i, l in enumerate(sel)}
    take = [(i, want[r["index"]]) for i, r in enumerate(blob["rows"])
            if r["arm"] == a.arm and r["anchor"] == a.anchor and r["index"] in want]
    lut = {L: i for i, L in enumerate("ABCD")}
    direction = discriminant_direction(
        R[[t[0] for t in take]], [lut[sel[t[1]]["cue_letter"]] for t in take])
    if direction is None:
        raise SystemExit("Could not derive the empirical direction")
    direction_k = model["components"] @ direction
    alignment = float(np.linalg.norm(direction_k))
    positive_rate = float(np.mean(z))

    print(f"PPCA rank {a.k} on {R.shape[0]} pooled rows "
          f"(explains {model['explained']:.3f} of variance)")
    print(f"planted along the cue-letter discriminant; it retains "
          f"{alignment:.3f} of its norm in this subspace")
    print(f"positive rate {positive_rate:.3f}, {a.reps} reps per cell, "
          f"ladder={'on' if a.ladder else 'off'}\n")

    results = []
    print("     n   " + "".join(f"  AUC{s:.2f}" for s in a.strengths) + "   floor")
    for n in a.sizes:
        row = floor_at(n, model, direction_k, positive_rate, a.strengths,
                       a.reps, a.n_null, a.ladder, a.seed)
        results.append(row)
        cells = "".join(f"  {int(c['detected_frac'] * 100):4d}%" for c in row["curve"])
        flag = "" if row["monotone"] else "  (non-monotone)"
        print(f"  {n:4d}   {cells}    {row['floor_auc']}{flag}")

    report = {"ppca_rank": a.k, "ppca_explained": model["explained"],
              "pooled_rows": int(R.shape[0]), "anchor": a.anchor, "arm": a.arm,
              "direction_alignment": alignment, "positive_rate": positive_rate,
              "reps": a.reps, "ladder": bool(a.ladder), "seed": a.seed,
              "results": results,
              "scope": ("Simulation from a Gaussian PPCA fit of the real residual "
                        "geometry, planted along the empirical cue-letter "
                        "direction. An order-of-magnitude planning guide, not a "
                        "power calculation: the covariance is estimated, the "
                        "model is Gaussian, and a differently shaped signal "
                        "needs a different n.")}
    if a.json:
        Path(a.json).parent.mkdir(parents=True, exist_ok=True)
        with open(a.json, "w") as f:
            json.dump(report, f, indent=2, allow_nan=False, default=float)
        print(f"\nWrote {a.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
