#!/usr/bin/env python3
"""Conditional probing: what does internal state add BEYOND the written trace?

    python code/e46_conditional.py \\
        --traces e46/qwen_fresh_01/traces.pkl \\
        --resid  e46/qwen_fresh_01/residuals_all_layers.pkl \\
        --json   reports/e46_conditional.json

WHY THIS IS NOT WHAT e46_gap.py ALREADY DOES

`e46_gap.py` compares readers in ISOLATION: rho_state against rho_trace, each
fitted on its own inputs. That answers "is state more predictive than text",
which is not the question. Two isolated readers can differ because they see
different amounts of the same information, because one underfits, or because
their feature spaces have different geometry -- none of which is a gap.

The quantity that addresses "what does state add beyond text" is conditional:

    G = CE(text) - CE(text ++ state)          [bits, held out]

Both readers see the text. Only one also sees the residual. A positive G is
predictive information in the state that the specified text reader could not
extract from the trace, at this n. It is still BOUNDED-READER information --
not proof the trace lacks it in principle.

SUBSUMPTION IS ENFORCED, NOT ASSUMED

The joint reader's candidate grid includes `state_dim = 0`, i.e. the text-only
pipeline itself. Inner-fold selection can therefore always fall back to text,
so G cannot go meaningfully negative except through selection noise, and a
negative G is a diagnostic rather than a finding. Without this, concatenating a
3584-dimensional residual onto a 192-dimensional text block would let the state
swamp the joint PCA and produce a spuriously negative gap.

Blocks are reduced SEPARATELY and inside the CV folds: standardise then PCA
each block on training rows only, then concatenate the reduced blocks. Reducing
a concatenation would let the larger block dominate the components; reducing
outside the folds would leak.

TARGET

Defined by the intervention, never by a reader or by the model's own claim:
the paired contrast between items whose answer moved to the cue from a correct
uncued answer and items already on the cue. Cue presence, cue-letter identity
and self-reported influence are all excluded by construction.

Note that the cleaner `followed vs resisted` target is DEGENERATE on this run
(31 followed, 2 resisted) because adaptive cue assignment fills the flipper
stratum and starves its complement. Any successor experiment that wants that
target must balance assignment deliberately.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import pickle
import sys

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


class BlockReduce(BaseEstimator, TransformerMixin):
    """Standardise and PCA each block separately, then concatenate.

    `n_text` splits the input columns into the text block and the state block.
    `state_dim = 0` drops the state block entirely, which is what makes the
    joint reader a strict superset of the text reader.
    """

    def __init__(self, n_text, text_dim=16, state_dim=16):
        self.n_text = n_text
        self.text_dim = text_dim
        self.state_dim = state_dim

    def _fit_block(self, block, dim):
        if dim <= 0 or block.shape[1] == 0:
            return None
        scaler = StandardScaler().fit(block)
        k = max(1, min(dim, block.shape[0] - 1, block.shape[1]))
        return scaler, PCA(n_components=k, svd_solver="full").fit(scaler.transform(block))

    def fit(self, X, y=None):
        X = np.asarray(X, float)
        self.text_ = self._fit_block(X[:, :self.n_text], self.text_dim)
        self.state_ = self._fit_block(X[:, self.n_text:], self.state_dim)
        if self.text_ is None and self.state_ is None:
            raise ValueError("BlockReduce needs at least one block")
        return self

    def transform(self, X):
        X = np.asarray(X, float)
        parts = []
        for fitted, block in ((self.text_, X[:, :self.n_text]),
                              (self.state_, X[:, self.n_text:])):
            if fitted is not None:
                scaler, pca = fitted
                parts.append(pca.transform(scaler.transform(block)))
        return np.hstack(parts)


def candidates(n_text, state_dims, text_dims=(8, 16, 32), regs=(0.01, 0.1, 1.0)):
    out = []
    # With no text block, every text_dim yields the same pipeline; keeping the
    # full product would triple the grid and inflate selection noise for
    # nothing.
    if n_text == 0:
        text_dims = (text_dims[0],)
    for text_dim in text_dims:
        for state_dim in state_dims:
            for c in regs:
                name = f"text{text_dim}_state{state_dim}_C{c}"
                out.append((name, lambda t=text_dim, s=state_dim, c=c:
                            make_pipeline(BlockReduce(n_text, t, s),
                                          LogisticRegression(C=c, max_iter=1000,
                                                             random_state=46))))
    return out


def evaluate(X, y, groups, n_text, state_dims, seed):
    from e46_readers import loss_rows, nested_oof
    fit = nested_oof(X, y, groups, 2, candidates(n_text, state_dims), seed=seed)
    per_row = loss_rows(np.asarray(y, int), fit["P"])
    return {"ce_bits": float(per_row.mean()), "per_row": per_row,
            "selected": [d.get("candidate") for d in fit["details"]],
            "base_bits": float(loss_rows(np.asarray(y, int), fit["base"]).mean())}


def item_bootstrap(delta, groups, seed=46, n_boot=4000):
    """One draw per source item, on fixed held-out per-row losses."""
    groups = np.asarray(groups)
    unique = np.unique(groups)
    means = np.array([delta[groups == g].mean() for g in unique])
    rng = np.random.default_rng(seed)
    draws = rng.choice(means, (n_boot, len(means)), replace=True).mean(axis=1)
    return [float(v) for v in np.quantile(draws, [0.025, 0.975])]


def run(a):
    from e46_gap import (build_labels, select, ngram_feats, pre_onset,
                         query_block)

    labels, meta = build_labels(a.traces)
    sel, z, g = select(labels, "flipper", "already_cue_wrong")
    if len(sel) < 20:
        raise SystemExit(f"only {len(sel)} rows in the contrast")

    # TEXT: the model's own episode up to the cue onset, plus its query block.
    # Identical construction to e46_gap's rho_trace_prompt, so the text arm of
    # the conditional comparison is the one already reported there.
    trace = ngram_feats([pre_onset(l["cued_text"], l["onset"]) for l in sel])
    prompt = ngram_feats([query_block(l["cued_prompt"]) for l in sel])
    text = np.hstack([trace, prompt])
    n_text = text.shape[1]

    with open(a.resid, "rb") as f:
        blob = pickle.load(f)
    R, rows = blob["R"], blob["rows"]
    want = {l["index"]: i for i, l in enumerate(sel)}

    report = {"n": len(sel), "n_pos": int(z.sum()), "n_neg": int(len(z) - z.sum()),
              "model": meta.get("model"), "substrate": meta.get("substrate"),
              "text_features": int(n_text), "seed": a.seed, "anchors": {},
              "target": "flipper vs already_cue_wrong (intervention-defined)",
              "scope": ("G = CE(text) - CE(text++state) in bits, held out, "
                        "nested item-grouped CV. Bounded-reader predictive "
                        "information, not proof the trace lacks it in "
                        "principle. Subsumption is enforced by including a "
                        "state_dim=0 candidate in the joint grid.")}

    print(f"n={len(sel)} ({int(z.sum())}/{len(z) - int(z.sum())})  "
          f"text features={n_text}  target={report['target']}\n")
    print(f"  {'anchor':<12}{'CE base':>9}{'CE text':>9}{'CE state':>10}"
          f"{'CE joint':>10}{'G bits':>9}   {'G 95% CI':<20}")

    for anchor in a.anchors:
        take = [(i, want[r["index"]]) for i, r in enumerate(rows)
                if r["arm"] == a.arm and r["anchor"] == anchor and r["index"] in want]
        if len(take) < 20:
            continue
        idx = np.array([t[1] for t in take])
        state = R[[t[0] for t in take]].astype(float)
        zz, gg = z[idx], g[idx]
        if len(np.unique(zz)) < 2:
            continue
        joint_X = np.hstack([text[idx], state])

        text_only = evaluate(text[idx], zz, gg, n_text, (0,), a.seed)
        state_only = evaluate(state, zz, gg, 0, a.state_dims, a.seed)
        joint = evaluate(joint_X, zz, gg, n_text, (0,) + tuple(a.state_dims), a.seed)

        delta = text_only["per_row"] - joint["per_row"]
        lo, hi = item_bootstrap(delta, gg, a.seed)
        used_state = sum("state0" not in s for s in joint["selected"] if s)
        report["anchors"][anchor] = {
            "n": len(take), "ce_base_bits": text_only["base_bits"],
            "ce_text_bits": text_only["ce_bits"],
            "ce_state_bits": state_only["ce_bits"],
            "ce_joint_bits": joint["ce_bits"],
            "gap_bits": float(delta.mean()), "gap_ci": [lo, hi],
            "folds_using_state": used_state,
            "n_folds": len(joint["selected"]),
            "joint_selected": joint["selected"]}
        print(f"  {anchor:<12}{text_only['base_bits']:>9.3f}"
              f"{text_only['ce_bits']:>9.3f}{state_only['ce_bits']:>10.3f}"
              f"{joint['ce_bits']:>10.3f}{delta.mean():>+9.3f}"
              f"   [{lo:+.3f},{hi:+.3f}]"
              f"  state used in {used_state}/{len(joint['selected'])} folds")

    positives = [v for v in report["anchors"].values() if v["gap_ci"][0] > 0]
    report["anchors_with_ci_above_zero"] = len(positives)
    print(f"\n  anchors whose G interval excludes zero: "
          f"{len(positives)}/{len(report['anchors'])}")
    print("  G is bounded-reader information: a zero does not show the trace "
          "contains it,\n  only that this text reader extracted as much as the "
          "joint reader could.")
    if a.json:
        Path(a.json).parent.mkdir(parents=True, exist_ok=True)
        out = {k: v for k, v in report.items()}
        with open(a.json, "w") as f:
            json.dump(out, f, indent=2, allow_nan=False, default=float)
        print(f"\nWrote {a.json}")
    return report


def selftest():
    """Subsumption and recovery, on data where the answer is known."""
    from e46_readers import loss_rows
    ok = True

    def chk(name, cond):
        nonlocal ok
        ok = ok and bool(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}")

    rng = np.random.default_rng(0)
    n = 80
    y = np.array([1] * 40 + [0] * 40)
    g = np.arange(n)

    # The planted direction must be one a variance-ranked projection can FIND.
    # An earlier version of this control planted into 40 equal-variance
    # dimensions, where PCA has no reason to prefer the informative one, and it
    # failed -- testing the blindness characterised in e46_gap's sensitivity
    # surface rather than the conditional logic under test here.
    text = rng.normal(size=(n, 12))
    basis = rng.normal(size=(5, 40))
    state = rng.normal(size=(n, 5)) @ basis
    # Amplitude 15, not 6: a sweep showed this reader yields G = -0.031, +0.011
    # and +0.323 at amplitudes 2, 6 and 15. The control has to sit where the
    # conditional logic is what is being tested, not where the reader's own
    # marginal sensitivity dominates -- which is itself the finding measured by
    # e46_gap's sensitivity surface.
    state += 15.0 * np.outer(y - y.mean(), basis[0] / np.linalg.norm(basis[0]))
    joint = np.hstack([text, state])
    t = evaluate(text, y, g, 12, (0,), 46)
    j = evaluate(joint, y, g, 12, (0, 8, 16), 46)
    chk("state that carries the label produces a positive conditional gap",
        t["ce_bits"] - j["ce_bits"] > 0.2)

    # state is pure noise -> subsumption should keep G from going negative
    state2 = rng.normal(size=(n, 5)) @ rng.normal(size=(5, 40))
    text2 = rng.normal(size=(n, 12))
    text2[:, 0] += 3.0 * y
    t2 = evaluate(text2, y, g, 12, (0,), 46)
    j2 = evaluate(np.hstack([text2, state2]), y, g, 12, (0, 8, 16), 46)
    chk("uninformative state does not destroy a strong text signal "
        "(subsumption floor)", t2["ce_bits"] - j2["ce_bits"] > -0.15)

    b = BlockReduce(12, text_dim=4, state_dim=0).fit(np.hstack([text, state]))
    chk("state_dim=0 drops the state block entirely",
        b.transform(np.hstack([text, state])).shape[1] == 4)
    b2 = BlockReduce(12, text_dim=4, state_dim=5).fit(np.hstack([text, state]))
    chk("otherwise both reduced blocks are concatenated",
        b2.transform(np.hstack([text, state])).shape[1] == 9)
    return 0 if ok else 1


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--traces")
    p.add_argument("--resid")
    p.add_argument("--arm", default="cued")
    p.add_argument("--anchors", nargs="+",
                   default=["prompt", "frac25", "frac50", "frac75", "onset"])
    p.add_argument("--state-dims", type=int, nargs="+", default=[4, 8, 16, 32])
    p.add_argument("--seed", type=int, default=46)
    p.add_argument("--json")
    p.add_argument("--selftest", action="store_true")
    a = p.parse_args(argv)
    if a.selftest:
        return selftest()
    if not (a.traces and a.resid):
        p.error("--traces and --resid are required (or --selftest)")
    run(a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
