#!/usr/bin/env python3
"""Re-score an E46 pretest pkl with the CURRENT scoring code -- no GPU.

The generation is fixed; mention, onset and the strata are not. When a scoring
rule changes (a mention pattern that fired on a content word, an onset that
missed the phrasing the model actually commits in), the honest check is to
re-read the traces already in hand rather than to spend another run.

    python e46_rescore_offline.py <traces.pkl> [--old <pretest.json>]
"""
from __future__ import annotations
import argparse, pickle, sys
from collections import Counter

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from e46_prompts_cue import (Item, answer_onset, episode_span, label_pair,
                             mentions_cue, extract_answer, STRATA)


def rebuild(rec) -> Item:
    d = dict(rec["item"]); d.pop("condition", None)
    return Item(**d)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("pkl"); ap.add_argument("--old", default=None)
    ap.add_argument("--show-mentions", action="store_true")
    a = ap.parse_args()
    blob = pickle.load(open(a.pkl, "rb"))
    rows = blob["raw_outputs"][blob["metadata"]["task_names"][0]]
    unc = {r["index"]: r for r in rows if r["condition"] == "uncued"}
    cue = {r["index"]: r for r in rows if r["condition"] == "cued"}
    ct = blob["metadata"]["cue_type"]

    counts, mentioned, onsets, moved = Counter(), [], [], 0
    for i, c in sorted(cue.items()):
        u = unc.get(i)
        if u is None: continue
        it = rebuild(c)
        lab = label_pair(it, u["text"], c["text"])
        counts[lab["stratum"]] += 1
        if lab["moved_to_cue"]:
            moved += 1
            if lab["mentioned"]:
                mentioned.append((i, c["text"]))
        if lab["onset_char"] is not None:
            onsets.append((i, lab["onset_char"],
                           lab["episode_chars_cued"]))

    print(f"\n  re-scored {len(cue)} pairs from {a.pkl.split('/')[-1]}\n")
    print("  stratum                 n")
    for s in STRATA:
        print(f"    {s:<22}{counts[s]:>4}")
    print(f"\n  moved-to-cue            {moved}")
    print(f"  mention | moved         {len(mentioned)}/{moved}"
          f"  ({len(mentioned)/max(moved,1):.3f})")
    if onsets:
        med = sorted(o for _, o, _ in onsets)[len(onsets) // 2]
        medl = sorted(l for _, _, l in onsets)[len(onsets) // 2]
        print(f"  onset found             {len(onsets)}/{len(cue)}"
              f"   median char {med} of {medl}"
              f"   ({med/max(medl,1):.2f} of the way in)")
    if a.show_mentions:
        for i, t in mentioned:
            print(f"\n  --- item {i} ---\n  {t.strip()[:400]}")
    if a.old:
        import json
        old = json.load(open(a.old))["stats"]["counts"]
        k = "all" if "all" in old[STRATA[0]] else "n"
        print("\n  stratum                 old   new")
        for s in STRATA:
            print(f"    {s:<22}{old[s][k]:>4}{counts[s]:>6}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
