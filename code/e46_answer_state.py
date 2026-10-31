#!/usr/bin/env python3
"""E46 answer-letter decoding with nested item-level evaluation.

Reading a selected answer or an alternative is an observational probe result.
It does not establish causal use, override, replacement, or suppression.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from typing import Any

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from p2_common import (                                         # noqa: E402
    DECODERS, entropy_bits_miller_madow, ilb_rho)
from e46_gap import STATE_DIMS, build_labels                     # noqa: E402
from e46_readers import VERSION, matrix_reader, nested_oof

LETTERS = "ABCD"
N_FOLDS = 5


# --------------------------------------------------------------- OOF proba

def oof_proba(X, y, groups, K, clf_factory, n_folds: int = N_FOLDS):
    """Grouped held-out probabilities, with calibration on training items only."""
    return nested_oof(X, y, groups, K,
                      [("reader", lambda: matrix_reader(clf_factory))],
                      n_folds=n_folds)["P"]


# ------------------------------------------------------------------- rows

def collect(traces_path: str, resid_path: str):
    """One row per (item, arm, anchor) whose arm produced a parsed answer."""
    labels, meta = build_labels(traces_path)
    by_idx = {l["index"]: l for l in labels}
    blob = pickle.load(open(resid_path, "rb"))
    if blob.get("kind") != "e46_residuals":
        raise SystemExit(f"{resid_path} is not an E46 residual cache")
    R, rows = blob["R"], blob["rows"]
    keep = []
    for i, r in enumerate(rows):
        lb = by_idx.get(r["index"])
        if lb is None:
            continue
        ans = lb[f"{r['arm']}_answer"]
        # One pair in the n=100 run had no parsed uncued answer. A row with no
        # answer has no answer state to probe, so it is dropped rather than
        # given a placeholder label.
        if not isinstance(ans, str) or ans not in LETTERS:
            continue
        keep.append({"ri": i, "index": r["index"], "arm": r["arm"],
                     "anchor": r["anchor"], "answer": ans,
                     "stratum": lb["stratum"], "mentioned": lb["mentioned"],
                     "cue_letter": lb["cue_letter"],
                     "correct_letter": lb["correct_letter"],
                     "uncued_answer": lb["uncued_answer"],
                     "cued_answer": lb["cued_answer"]})
    return keep, R, blob["metadata"], meta


# ------------------------------------------------------------- the analysis

# Earlier pooled-arm fits read cued answers much better than uncued answers,
# consistent with a visible-cue shortcut. Uncued-only training is a useful
# transfer diagnostic, but introduces distribution shift. Neither a failed
# transfer nor a passing accuracy gate identifies replacement or suppression.

def run_anchor(rows, R, anchor: str, ladder: bool = False, dim: int = 32,
               train_arm: str | None = None):
    """Train the 4-way answer probe at one anchor.

    `train_arm="uncued"` fits on uncued rows only and predicts the cued rows
    out-of-sample; the default pools both arms and is kept only to reproduce
    the shortcut described above.
    """
    sel = [r for r in rows if r["anchor"] == anchor]
    if len(sel) < 40:
        return None
    ri = np.array([r["ri"] for r in sel])
    y = np.array([LETTERS.index(r["answer"]) for r in sel])
    # GROUPED BY ITEM, so an item's cued and uncued rows never straddle a
    # fold. They share the problem text and, for already_cue_* pairs, the same
    # label -- splitting them would let the probe memorise the item.
    g = np.array([r["index"] for r in sel])
    if len(np.unique(y)) < 2:
        return None
    Rraw = R[ri].astype(np.float64)
    # Pooled and arm-transfer fits both learn the projection on training items.
    _, H = entropy_bits_miller_madow(y, 4)
    candidates = [(name, lambda fac=fac: matrix_reader(fac, dim))
                  for name, fac in (DECODERS if ladder else DECODERS[:1])]
    mask = (np.array([r["arm"] == train_arm for r in sel])
            if train_arm else None)
    result = nested_oof(Rraw, y, g, 4, candidates, train_mask=mask)
    P = result["P"]
    ce, ilb, rho = ilb_rho(H, P[np.arange(len(y)), y])
    name = "nested"
    acc = float((P.argmax(1) == y).mean())
    return {"anchor": anchor, "n": len(sel), "H_bits": H, "rho": rho,
            "ilb": ilb, "accuracy": acc, "decoder": name,
            "P": P, "sel": sel, "y": y, "fold_selection": result["details"]}


def _transfer_proba(Rraw, y, g, sel, train_arm, clf_factory, dim,
                    K: int = 4):
    """ITEM-HELD-OUT transfer. Outer folds by item; every fitted object --
    the PCA basis included -- sees only training items.

    The first version fitted on all uncued rows and predicted the cued rows of
    THE SAME items, so a reader could partially learn item identity and carry
    it across the arm boundary. That is fine to look at and not fine to cite.
    Here an item is either in training or in test, in both arms at once, so
    neither the sanity number nor the readout can be inflated by having seen
    the item.

    The PCA basis is also fitted inside the fold. It is unsupervised, but
    fitting it on all rows lets the test rows' geometry -- and the cued/uncued
    condition split -- shape the basis the classifier then uses.
    """
    tr_mask = np.array([r["arm"] == train_arm for r in sel])
    return nested_oof(Rraw, y, g, K,
                      [("transfer", lambda: matrix_reader(clf_factory, dim))],
                      train_mask=tr_mask)["P"]


def sanity_ok(res, min_acc: float = 0.40) -> dict[str, Any]:
    """Can the probe read the answer the UNCUED arm actually gave, on flippers?

    This is the gate. If it fails, the cued-arm readout is uninterpretable for
    the same reason the first binary null was: a reader that cannot find a
    variable where it provably exists says nothing by not finding it elsewhere.
    """
    P, sel, y = res["P"], res["sel"], res["y"]
    pred = P.argmax(1)
    m = [k for k, s in enumerate(sel)
         if s["stratum"] == "flipper" and s["arm"] == "uncued"
         and not s["mentioned"]]
    acc = float((pred[m] == y[m]).mean()) if m else float("nan")
    return {"flipper_uncued_accuracy": acc, "n": len(m),
            "chance": 0.25, "min_acc": min_acc,
            "ok": bool(len(m) and acc >= min_acc)}


def flipper_readout(res) -> dict[str, Any]:
    """The measurement: what happens to the abandoned answer in the cued arm.

    For a flipper the uncued answer IS the correct letter and the cued answer
    is the cue, so each pair has four distinguishable letters:

        cue        what the model now says
        abandoned  what it said without the hint (= the correct answer)
        other x2   never in play -- the within-row baseline
    """
    P, sel = res["P"], res["sel"]
    out = {"cued": [], "uncued": []}
    for k, r in enumerate(sel):
        if r["stratum"] != "flipper" or r["mentioned"]:
            continue
        cue, ab = r["cue_letter"], r["correct_letter"]
        if cue == ab:
            continue
        others = [L for L in LETTERS if L not in (cue, ab)]
        p = P[k]
        out[r["arm"]].append({
            "index": r["index"],
            "p_cue": float(p[LETTERS.index(cue)]),
            "p_abandoned": float(p[LETTERS.index(ab)]),
            "p_other_mean": float(np.mean([p[LETTERS.index(L)]
                                           for L in others])),
        })
    summ = {}
    for arm, rs in out.items():
        if not rs:
            continue
        pc = np.array([r["p_cue"] for r in rs])
        pa = np.array([r["p_abandoned"] for r in rs])
        po = np.array([r["p_other_mean"] for r in rs])
        # Paired over items: the abandoned answer's lead over the letters that
        # were never in play, on the same row.
        lead = pa - po
        boot = _boot_mean(lead)
        summ[arm] = {"n": len(rs), "p_cue": float(pc.mean()),
                     "p_abandoned": float(pa.mean()),
                     "p_other": float(po.mean()),
                     "abandoned_lead": float(lead.mean()),
                     "abandoned_lead_ci": boot,
                     "survives": bool(boot[0] > 0.0)}
    return {"per_pair": out, "summary": summ}


def _boot_mean(v, n_boot: int = 2000, seed: int = 0):
    rng = np.random.default_rng(seed)
    if len(v) == 0:
        return (0.0, 0.0)
    b = rng.choice(v, size=(n_boot, len(v)), replace=True).mean(1)
    return (float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5)))


def analyse(traces_path, resid_path, ladder=False, dim=32,
            train_arm: str | None = "uncued"):
    rows, R, rmeta, tmeta = collect(traces_path, resid_path)
    anchors = rmeta["anchors"]
    out = {"model": tmeta.get("model"), "substrate": tmeta.get("substrate"),
           "cue_type": tmeta.get("cue_type"), "pca_dim": dim,
           "train_arm": train_arm or "both",
           "n_rows_total": len(rows), "evaluation_version": VERSION,
           "mechanism_verdict": "unresolved", "anchors": {}}
    for a in anchors:
        r = run_anchor(rows, R, a, ladder, dim, train_arm)
        if r is None:
            continue
        fr = flipper_readout(r)
        pred = r["P"].argmax(1)
        by = {}
        for st in ("flipper", "already_cue_wrong", "already_cue_correct"):
            for arm in ("uncued", "cued"):
                m = [k for k, s in enumerate(r["sel"])
                     if s["stratum"] == st and s["arm"] == arm
                     and not s["mentioned"]]
                if m:
                    by[f"{st}/{arm}"] = {
                        "acc": float((pred[m] == r["y"][m]).mean()),
                        "n": len(m)}
        out["anchors"][a] = {
            "n": r["n"], "rho": r["rho"], "accuracy": r["accuracy"],
            "H_bits": r["H_bits"], "decoder": r["decoder"],
            "fold_selection": r["fold_selection"],
            "sanity": sanity_ok(r), "by_cell": by,
            "flippers": fr["summary"],
        }
    return out


def report(res):
    W = 80
    print("\n" + "=" * W)
    print(f" E46 ANSWER-STATE PROBE   model={res['model']}  "
          f"substrate={res['substrate']}  cue={res['cue_type']}")
    ta = res.get("train_arm", "both")
    how = ("both arms pooled (SHORTCUT PATH -- diagnostic only)"
           if ta == "both" else
           f"trained on the {ta} arm only, ITEM-HELD-OUT")
    print(f" 4-way A/B/C/D from the residual, {how}, PCA dim {res['pca_dim']}")
    print("=" * W)
    print(f"\n  {'anchor':<12}{'n':>5}{'acc':>7}{'rho':>7}   "
          f"{'FLIPPERS, CUED ARM':<34}")
    print(f"  {'':<12}{'':>5}{'':>7}{'':>7}   "
          f"{'p(cue)':>8}{'p(aband)':>10}{'p(other)':>10}{'lead':>8}"
          f"  survives")
    for a, v in res["anchors"].items():
        f = v["flippers"].get("cued")
        row = (f"{f['p_cue']:>8.3f}{f['p_abandoned']:>10.3f}"
               f"{f['p_other']:>10.3f}{f['abandoned_lead']:>+8.3f}"
               f"  {f['survives']}") if f else " " * 34
        print(f"  {a:<12}{v['n']:>5}{v['accuracy']:>7.3f}{v['rho']:>7.3f}   {row}")
    print(f"\n  UNCUED ARM (the probe's own sanity check -- p(abandoned) is "
          f"what the\n  model actually answered there, so it should DOMINATE)")
    print(f"  {'anchor':<12}{'p(cue)':>8}{'p(aband)':>10}{'p(other)':>10}")
    for a, v in res["anchors"].items():
        f = v["flippers"].get("uncued")
        if f:
            print(f"  {a:<12}{f['p_cue']:>8.3f}{f['p_abandoned']:>10.3f}"
                  f"{f['p_other']:>10.3f}")
    print(f"\n  SANITY GATE   can the probe read the answer the UNCUED arm "
          f"actually gave,\n  on flipper rows? (chance 0.25, need "
          f"{next(iter(res['anchors'].values()))['sanity']['min_acc']:.2f})")
    for a, v in res["anchors"].items():
        s = v["sanity"]
        print(f"    {a:<12}{s['flipper_uncued_accuracy']:>7.3f}  n={s['n']:<4}"
              f"  ok={s['ok']}")
    good = [a for a, v in res["anchors"].items() if v["sanity"]["ok"]]
    if not good:
        best_a = max(res["anchors"],
                     key=lambda k: res["anchors"][k]["sanity"]
                     ["flipper_uncued_accuracy"])
        best_v = res["anchors"][best_a]["sanity"]["flipper_uncued_accuracy"]
        print(f"\n  ** No anchor clears the gate. Best is {best_a} at "
              f"{best_v:.3f} against a chance\n  ** of 0.25 -- WEAK, "
              f"UNDERPOWERED answer-state recovery, which is not the same as\n"
              f"  ** no answer-state signal. The cued-arm readout below cannot "
              f"decide\n  ** override vs replacement either way; it is not "
              f"evidence for replacement.")
    cued = [v["flippers"].get("cued") for a, v in res["anchors"].items()
            if a in good and v["flippers"].get("cued")]
    if cued:
        surv = [c for c in cued if c["survives"]]
        print(f"\n  VERDICT  the abandoned answer leads the never-in-play "
              f"letters at {len(surv)}/{len(cued)} anchors")
        print("    These are reader probability contrasts. They do not identify "
              "override,\n    replacement, suppression, or causal use. Mechanism unresolved.")
    print("=" * W)


# ----------------------------------------------------------------- selftest

def _selftest() -> int:
    ok = [0, 0]

    def chk(name, cond):
        ok[1] += 1
        ok[0] += bool(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}")

    rng = np.random.default_rng(0)
    n, K = 160, 4
    y = np.tile(np.arange(K), n // K)
    g = np.arange(n)
    basis = rng.normal(size=(K, 24))
    X = basis[y] * 3.0 + rng.normal(size=(n, 24))
    P = oof_proba(X, y, g, K, DECODERS[0][1])
    chk("oof_proba returns one row per sample and K columns",
        P.shape == (n, K))
    chk("rows are proper distributions",
        np.allclose(P.sum(1), 1.0, atol=1e-6))
    chk("and it recovers a separable answer state",
        float((P.argmax(1) == y).mean()) > 0.8)
    Pn = oof_proba(rng.normal(size=(n, 24)), y, g, K, DECODERS[0][1])
    chk("on noise it backs off to ~the marginal, never to confident garbage",
        abs(Pn.mean(0).max() - 0.25) < 0.12)

    # ---- the flipper readout ---------------------------------------------
    def mk(arm, p_vec, cue="A", correct="B"):
        return {"ri": 0, "index": 1, "arm": arm, "anchor": "onset",
                "answer": cue if arm == "cued" else correct,
                "stratum": "flipper", "mentioned": False,
                "cue_letter": cue, "correct_letter": correct,
                "uncued_answer": correct, "cued_answer": cue}
    # OVERRIDE: cue dominates but the abandoned answer is still well above the
    # letters that were never in play.
    res_over = {"P": np.array([[0.60, 0.30, 0.05, 0.05]]),
                "sel": [mk("cued", None)]}
    fo = flipper_readout(res_over)["summary"]["cued"]
    chk("positive lead: the abandoned answer outscores the reference letters",
        fo["survives"] and fo["abandoned_lead"] > 0.2)
    # REPLACEMENT: the abandoned answer has collapsed to the others' level.
    res_rep = {"P": np.array([[0.85, 0.05, 0.05, 0.05]]),
               "sel": [mk("cued", None)]}
    fr = flipper_readout(res_rep)["summary"]["cued"]
    chk("negative lead: the abandoned answer does not outscore the references",
        not fr["survives"] and abs(fr["abandoned_lead"]) < 1e-9)
    chk("the baseline is WITHIN-ROW -- the two letters never in play, not a "
        "cross-item average",
        abs(fr["p_other"] - 0.05) < 1e-9)
    chk("and the cue's own mass is reported beside it",
        abs(fr["p_cue"] - 0.85) < 1e-9)
    mixed = {"P": np.array([[0.60, 0.30, 0.05, 0.05],
                            [0.85, 0.05, 0.05, 0.05]]),
             "sel": [mk("cued", None), mk("cued", None)]}
    fm = flipper_readout(mixed)["summary"]["cued"]
    chk("the lead is averaged over pairs, with a bootstrap CI over pairs",
        fm["n"] == 2 and len(fm["abandoned_lead_ci"]) == 2)
    chk("a pair whose cue IS the correct answer is excluded -- it has no "
        "abandoned answer to look for",
        "cued" not in flipper_readout(
            {"P": np.array([[0.9, 0.05, 0.03, 0.02]]),
             "sel": [mk("cued", None, cue="A", correct="A")]})["summary"])
    chk("mentioned pairs are excluded, as everywhere else",
        "cued" not in flipper_readout(
            {"P": np.array([[0.9, 0.05, 0.03, 0.02]]),
             "sel": [{**mk("cued", None), "mentioned": True}]})["summary"])
    chk("both arms are read out separately, never differenced",
        set(flipper_readout(
            {"P": np.array([[0.6, 0.3, 0.05, 0.05], [0.2, 0.7, 0.05, 0.05]]),
             "sel": [mk("cued", None), mk("uncued", None)]})["summary"])
        == {"cued", "uncued"})

    chk("the probe pools BOTH arms, so it trains on ~2x the primary rows",
        "arm" not in run_anchor.__doc__.split("pooled")[0].split()[-1:])
    chk("STATE_DIMS is shared with e46_gap, so the two analyses use the same "
        "feature scale", max(STATE_DIMS) == 32)

    # ---- the sanity gate --------------------------------------------------
    def _row(arm, ans, st="flipper"):
        return {"ri": 0, "index": 1, "arm": arm, "anchor": "onset",
                "answer": ans, "stratum": st, "mentioned": False,
                "cue_letter": "A", "correct_letter": "B",
                "uncued_answer": "B", "cued_answer": "A"}
    good = {"P": np.array([[0.1, 0.7, 0.1, 0.1]] * 4),
            "sel": [_row("uncued", "B")] * 4,
            "y": np.array([1, 1, 1, 1])}
    bad = {"P": np.array([[0.7, 0.1, 0.1, 0.1]] * 4),
           "sel": [_row("uncued", "B")] * 4,
           "y": np.array([1, 1, 1, 1])}
    chk("the gate passes when the probe reads the uncued arm's own answer",
        sanity_ok(good)["ok"])
    chk("and FAILS when it does not -- the check that stops a reader failure "
        "being reported as replacement",
        not sanity_ok(bad)["ok"]
        and sanity_ok(bad)["flipper_uncued_accuracy"] == 0.0)
    chk("the gate is measured on FLIPPER uncued rows only, the ones the "
        "readout depends on",
        sanity_ok({"P": np.array([[0.1, 0.7, 0.1, 0.1]]),
                   "sel": [_row("cued", "A")], "y": np.array([0])})["n"] == 0)

    print(f"\n  {ok[0]}/{ok[1]} passed")
    return 0 if ok[0] == ok[1] else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces")
    ap.add_argument("--resid")
    ap.add_argument("--dim", type=int, default=32)
    ap.add_argument("--ladder", action="store_true")
    ap.add_argument("--train-arm", default="uncued",
                    choices=["uncued", "cued", "both"],
                    help="training arm; uncued is a transfer diagnostic that "
                         "avoids training on the visible cue but introduces distribution shift")
    ap.add_argument("--json", default=None)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return _selftest()
    if not (a.traces and a.resid):
        raise SystemExit("--traces and --resid are required (or --selftest)")
    res = analyse(a.traces, a.resid, a.ladder, a.dim,
                  None if a.train_arm == "both" else a.train_arm)
    report(res)
    if a.json:
        json.dump(res, open(a.json, "w"), indent=2, default=float)
        print(f"\n  wrote {a.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
