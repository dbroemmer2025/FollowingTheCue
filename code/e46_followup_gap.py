#!/usr/bin/env python3
"""Descriptive follow-up cue-vocabulary contrasts.

Reports paired shifts and citation-associated scores. These measurements do
not identify suppression or absence of early information. Use the separate
follow-up predictor for prospective citation prediction.
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

HEADLINE_ANCHOR = "q_end"


def cue_margin(row) -> float | None:
    """mean(cue-word logits) - mean(neutral-word logits) for one row."""
    c, n = row.get("cue_logits"), row.get("neutral_logits")
    if not c or not n:
        return None
    return float(np.mean(list(c.values())) - np.mean(list(n.values())))


def _boot(v, n_boot: int = 8000, seed: int = 0):
    rng = np.random.default_rng(seed)
    v = np.asarray(v, dtype=float)
    if len(v) < 2:
        return (float("nan"), float("nan"))
    b = rng.choice(v, size=(n_boot, len(v)), replace=True).mean(1)
    return (float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5)))


def join(furesid_path: str, followups_path: str):
    """Residual rows + the turn-2 mention label, joined on (index, arm).

    THE MENTION LABEL IS RECOMPUTED, not read. The capture ran before the
    follow-up pkl was scored, so `mentions_cue_turn2` is None on every residual
    row. Reading it would silently treat every row as a non-mention and put the
    cited rows into the silent group -- destroying the one contrast the lens
    adequacy check depends on.
    """
    from e46_followup import score
    from e46_cache import validate_followups
    fu = pickle.load(open(followups_path, "rb"))
    if fu.get("kind") != "e46_followups":
        raise SystemExit(f"{followups_path} is not an E46 follow-up cache")
    fr = [r for r in fu["rows"] if "text2" in r]
    score(fr, fu["metadata"].get("cue_type", "authority"))
    mention = {(r["index"], r["arm"]): bool(r["mentions_cue_turn2"])
               for r in fr}

    blob = pickle.load(open(furesid_path, "rb"))
    if blob.get("kind") != "e46_followup_residuals":
        raise SystemExit(
            f"{furesid_path} is not an E46 follow-up residual cache. Build it "
            f"with e46_capture_residuals.py --followups.")
    audit = validate_followups(fu, blob)
    audit["rows_without_finite_lens"] = 0
    rows = []
    for ri, r in enumerate(blob["rows"]):
        m = mention.get((r["index"], r["arm"]))
        if m is None:
            continue
        cm = cue_margin(r)
        if cm is None or not np.isfinite(cm):
            audit["rows_without_finite_lens"] += 1
            continue
        rows.append({**r, "mentions_cue_turn2": m, "cue_margin": cm,
                     "residual_row": ri})
    return rows, {**blob["metadata"], "join_audit": audit}


def lens_positive_control(rows) -> dict[str, Any] | None:
    """Within-citing-item score change from q_end to regex mention start."""
    q = {(r["index"], r["arm"]): r["cue_margin"] for r in rows
         if r["anchor"] == "q_end" and r["arm"] == "cued"}
    pairs = [(r["cue_margin"], q[(r["index"], r["arm"])]) for r in rows
             if r["anchor"] == "mention" and r["arm"] == "cued"
             and (r["index"], r["arm"]) in q]
    at_m = [m for m, qv in pairs]
    at_q = [qv for m, qv in pairs]
    if len(at_m) < 3 or len(at_q) < 3:
        return None
    lo, hi = _boot(np.array(at_m) - np.array(at_q))
    return {"mention_mean": float(np.mean(at_m)), "n_mention": len(at_m),
            "q_end_cued_mean": float(np.mean(at_q)), "n_q_end": len(at_q),
            "difference": float(np.mean(at_m) - np.mean(at_q)),
            "ci": [float(lo), float(hi)], "ok": bool(lo > 0),
            "design": "paired same citing items",
            "scope": "score change near regex mention start; not early-reader adequacy"}


def analyse(rows, metadata) -> dict[str, Any]:
    anchors = metadata["anchors"]
    out: dict[str, Any] = {
        "model": metadata.get("model"), "cue_type": metadata.get("cue_type"),
        "cue_vocab": metadata.get("cue_vocab"),
        "neutral_vocab": metadata.get("neutral_vocab"),
        "headline_anchor": HEADLINE_ANCHOR,
        "analysis_version": "descriptive_v2",
        "mechanism_verdict": "unresolved",
        "join_audit": metadata.get("join_audit"),
        "lens_positive_control": lens_positive_control(rows),
        "by_anchor": {}}
    for a in anchors:
        at = [r for r in rows if r["anchor"] == a]
        if len(at) < 8:
            continue
        cell = {}
        for st in ("flipper", "already_cue_wrong", "already_cue_correct",
                   "resister"):
            for arm in ("cued", "uncued"):
                v = [r["cue_margin"] for r in at
                     if r["stratum"] == st and r["arm"] == arm]
                if v:
                    cell[f"{st}/{arm}"] = {"n": len(v), "mean": float(np.mean(v)),
                                           "ci": _boot(v)}
        # ---- lens adequacy: does it see a cue the model is about to name? ---
        cued = [r for r in at if r["arm"] == "cued"]
        cite = [r["cue_margin"] for r in cued if r["mentions_cue_turn2"]]
        sil = [r["cue_margin"] for r in cued if not r["mentions_cue_turn2"]]
        lens = None
        if len(cite) >= 3 and len(sil) >= 3:
            rng = np.random.default_rng(1)
            d = [rng.choice(cite, len(cite), True).mean()
                 - rng.choice(sil, len(sil), True).mean() for _ in range(8000)]
            lo, hi = np.percentile(d, [2.5, 97.5])
            lens = {"cited_mean": float(np.mean(cite)), "n_cited": len(cite),
                    "silent_mean": float(np.mean(sil)), "n_silent": len(sil),
                    "difference": float(np.mean(cite) - np.mean(sil)),
                    "ci": [float(lo), float(hi)], "ok": bool(lo > 0)}
        # ---- the headline: SILENT flippers, paired against their own control -
        by_item = {}
        for r in at:
            if r["stratum"] != "flipper":
                continue
            by_item.setdefault(r["index"], {})[r["arm"]] = r
        paired = []
        for i, arms in by_item.items():
            if "cued" not in arms or "uncued" not in arms:
                continue
            if arms["cued"]["mentions_cue_turn2"]:
                continue          # restrict this descriptive contrast to silent cases
            paired.append(arms["cued"]["cue_margin"]
                          - arms["uncued"]["cue_margin"])
        supp = None
        if len(paired) >= 5:
            lo, hi = _boot(paired)
            supp = {"n": len(paired), "mean": float(np.mean(paired)),
                    "ci": [lo, hi], "elevated": bool(lo > 0)}
        out["by_anchor"][a] = {"cells": cell, "lens_adequacy": lens,
                               "silent_flipper_cue_shift": supp,
                               "silent_flipper_suppression": supp}
    out["legacy_fields"] = {"silent_flipper_suppression":
                             "Deprecated alias of silent_flipper_cue_shift; not suppression evidence",
                             "lens_adequacy": "Directional mean-separation diagnostic only"}
    return out


def report(res):
    W = 78
    print("\n" + "=" * W)
    print(" E46 FOLLOW-UP: descriptive cue-vocabulary scores")
    print(f" model={res.get('model')}  cue={res.get('cue_type')}")
    print(f" cue_margin = mean{res.get('cue_vocab')}\n"
          f"            - mean{res.get('neutral_vocab')}")
    print("=" * W)
    for a, blk in res["by_anchor"].items():
        star = "  <- HEADLINE" if a == res["headline_anchor"] else ""
        print(f"\n  --- {a} ---{star}")
        print(f"    {'stratum / arm':<30}{'n':>4}{'cue_margin':>12}{'95% CI':>20}")
        for k, c in blk["cells"].items():
            print(f"    {k:<30}{c['n']:>4}{c['mean']:>12.3f}"
                  f"{'[{:+.2f},{:+.2f}]'.format(*c['ci']):>20}")
        ln = blk["lens_adequacy"]
        if ln:
            print(f"    CITATION ASSOCIATION  cited {ln['cited_mean']:+.3f} "
                  f"(n={ln['n_cited']})  vs silent {ln['silent_mean']:+.3f} "
                  f"(n={ln['n_silent']})")
            print(f"                   difference {ln['difference']:+.3f} "
                  f"[{ln['ci'][0]:+.2f},{ln['ci'][1]:+.2f}]   ok={ln['ok']}")
        sp = blk["silent_flipper_suppression"]
        if sp:
            print(f"    SILENT FLIPPERS, paired cued-minus-uncued: "
                  f"{sp['mean']:+.3f} [{sp['ci'][0]:+.2f},{sp['ci'][1]:+.2f}]"
                  f"  n={sp['n']}  elevated={sp['elevated']}")
    h = res["by_anchor"].get(res["headline_anchor"], {})
    ln, sp = h.get("lens_adequacy"), h.get("silent_flipper_suppression")
    pc = res.get("lens_positive_control")
    print("\n" + "-" * W)
    if pc:
        print(f"  PAIRED POSITION CONTROL   at regex mention start: "
              f"{pc['mention_mean']:+.2f} (n={pc['n_mention']})")
        print(f"                          at q_end, cued: "
              f"{pc['q_end_cued_mean']:+.2f} (n={pc['n_q_end']})   "
              f"difference {pc['difference']:+.2f} "
              f"[{pc['ci'][0]:+.2f},{pc['ci'][1]:+.2f}]   ok={pc['ok']}")
    print("\n  MECHANISM UNRESOLVED. These are vocabulary-score contrasts.\n"
          "  A failed mean-separation test does not establish absent early information.\n"
          "  An elevated cue score does not distinguish suppression from prompt priming.\n"
          "  The paired mention-position control only tests a change near citation.")
    if res.get("join_audit"):
        print("  Join audit:", res["join_audit"])
    print("=" * W)


# ----------------------------------------------------------------- selftest

def _selftest() -> int:
    ok = [0, 0]

    def chk(name, cond):
        ok[1] += 1
        ok[0] += bool(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}")

    r = {"cue_logits": {"a": 3.0, "b": 1.0},
         "neutral_logits": {"x": 0.0, "y": 2.0}}
    chk("cue_margin is cue-mean minus neutral-mean, so a position that raises "
        "EVERY content word cancels", cue_margin(r) == 1.0)
    chk("a row without the lens is skipped, not scored as zero",
        cue_margin({"cue_logits": None, "neutral_logits": {}}) is None)

    def mk(i, arm, st, cm, cited=False, anchor="q_end"):
        return {"index": i, "arm": arm, "stratum": st, "anchor": anchor,
                "mentions_cue_turn2": cited,
                "cue_logits": {"p": cm}, "neutral_logits": {"m": 0.0},
                "cue_margin": cm}

    # Positive paired score shift for silent flippers.
    sup = []
    for i in range(12):
        sup += [mk(i, "cued", "flipper", 2.0), mk(i, "uncued", "flipper", 0.0)]
    for i in range(100, 105):                       # some rows DID cite
        sup += [mk(i, "cued", "flipper", 4.0, cited=True),
                mk(i, "uncued", "flipper", 0.0)]
    res = analyse(sup, {"anchors": ["q_end"]})
    blk = res["by_anchor"]["q_end"]
    chk("the lens passes adequacy when cited rows outrank silent ones",
        blk["lens_adequacy"]["ok"])
    chk("cited rows are excluded from the silent-flipper contrast",
        blk["silent_flipper_suppression"]["n"] == 12)
    chk("the paired score shift is detected when silent rows are elevated",
        blk["silent_flipper_suppression"]["elevated"]
        and abs(blk["silent_flipper_suppression"]["mean"] - 2.0) < 1e-9)

    # Zero paired score shift for silent flippers.
    flat = []
    for i in range(12):
        flat += [mk(i, "cued", "flipper", 0.0), mk(i, "uncued", "flipper", 0.0)]
    for i in range(100, 105):
        flat += [mk(i, "cued", "flipper", 4.0, cited=True),
                 mk(i, "uncued", "flipper", 0.0)]
    b2 = analyse(flat, {"anchors": ["q_end"]})["by_anchor"]["q_end"]
    chk("no score elevation when silent rows match their controls",
        not b2["silent_flipper_suppression"]["elevated"])
    chk("the cited-minus-silent score contrast remains positive",
        b2["lens_adequacy"]["ok"])

    # Equal citing and silent means fail this directional association check.
    blind = []
    for i in range(12):
        blind += [mk(i, "cued", "flipper", 1.0), mk(i, "uncued", "flipper", 0.0)]
    for i in range(100, 105):
        blind += [mk(i, "cued", "flipper", 1.0, cited=True),
                  mk(i, "uncued", "flipper", 0.0)]
    b3 = analyse(blind, {"anchors": ["q_end"]})["by_anchor"]["q_end"]
    chk("equal cited and silent scores fail mean separation even with a paired shift",
        not b3["lens_adequacy"]["ok"]
        and b3["silent_flipper_suppression"]["elevated"])

    # Ten cued rows, no uncued partner anywhere: enough rows to be analysed,
    # zero pairs to measure. The paired-shift cell must come back empty rather
    # than quietly comparing cued rows against nothing.
    unpaired = [mk(i, "cued", "flipper", 5.0) for i in range(10)]
    chk("pairing is WITHIN item -- unpaired cued rows contribute nothing",
        analyse(unpaired, {"anchors": ["q_end"]})["by_anchor"]["q_end"]
        ["silent_flipper_suppression"] is None)
    chk("and an anchor with too few rows is skipped entirely, not scored thin",
        "q_end" not in analyse(unpaired[:4],
                               {"anchors": ["q_end"]})["by_anchor"])
    chk("the headline anchor is q_end, before any justification exists",
        HEADLINE_ANCHOR == "q_end")

    # ---- the positive control separates two very different failures --------
    live = [mk(i, "cued", "flipper", 1.0) for i in range(10)]
    live += [mk(i, "cued", "flipper", 6.0, anchor="mention")
             for i in range(5)]
    pc = lens_positive_control(live)
    chk("the same-item score increases at the matched mention position",
        pc["ok"] and pc["difference"] > 4.0)
    dead = [mk(i, "cued", "flipper", 1.0) for i in range(10)]
    dead += [mk(i, "cued", "flipper", 1.0, anchor="mention")
             for i in range(5)]
    chk("and a vocabulary that reads nothing even there is caught",
        not lens_positive_control(dead)["ok"])
    chk("with no mention rows the control is skipped, not assumed",
        lens_positive_control([mk(i, "cued", "flipper", 1.0)
                               for i in range(10)]) is None)
    chk("every stratum is broken out by arm so the floor is visible",
        "flipper/uncued" in blk["cells"] and "flipper/cued" in blk["cells"])

    print(f"\n  {ok[0]}/{ok[1]} passed")
    return 0 if ok[0] == ok[1] else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--furesid", help="e46_furesid_*.pkl")
    ap.add_argument("--followups", help="e46_followup_*.pkl")
    ap.add_argument("--json", default=None)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return _selftest()
    if not (a.furesid and a.followups):
        raise SystemExit("--furesid and --followups are both required")
    rows, meta = join(a.furesid, a.followups)
    res = analyse(rows, meta)
    report(res)
    if a.json:
        json.dump(res, open(a.json, "w"), indent=2, default=float)
        print(f"\n  wrote {a.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
