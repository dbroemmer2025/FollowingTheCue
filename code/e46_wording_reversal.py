#!/usr/bin/env python3
"""Reproduce every table in ANALYSIS_E46_AVAILABILITY_ADDENDUM_2026-09-15.md.

Read-only. Reads the approved availability rows and recomputes the paired
wording analysis, so no number in the addendum has to be taken on trust.

    python code/e46_wording_reversal.py \
        --rows e46/qwen_fresh_01/qwen_fresh_01_availability_human_approved_rows.csv

The point of the analysis: wording is assigned per ITEM and both arms of an
item inherit it, so within a wording group the cued-vs-uncued contrast is
paired on the same items. McNemar tests each arm contrast, not the interaction
between wordings: different items received w0 and w1. These are exploratory,
unadjusted tests conditional on the reviewed binary labels and selected stratum.
"""
from __future__ import annotations

import argparse
import collections
import csv
import math
import sys

STRATUM = "flipper"


def wilson(k, n):
    if not n:
        return None
    z = 1.96
    centre = (k / n + z * z / (2 * n)) / (1 + z * z / n)
    radius = z * math.sqrt(k / n * (1 - k / n) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0., centre - radius), min(1., centre + radius)


def mcnemar_exact(b, c):
    """Two-sided exact McNemar on the discordant pairs only.

    With b+c discordant pairs and no effect, the split is Binomial(b+c, 1/2);
    the exact two-sided p doubles the smaller tail. Used rather than the
    chi-square form because the discordant counts here are 7 and 11.
    """
    if b + c == 0:
        return 1.0
    tail = sum(math.comb(b + c, k) for k in range(min(b, c) + 1)) / 2 ** (b + c)
    return min(1.0, 2 * tail)


def load(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def flag(row, field):
    value = (row.get(field) or "").strip()
    if value not in ("", "0", "1", "uncertain"):
        raise ValueError(f"Invalid {field}: {value}")
    if row.get("response_available", "").lower() != "true" or value not in ("0", "1"):
        return None
    return value == "1"


def rate(rows, field):
    values = [flag(r, field) for r in rows]
    n = sum(v is not None for v in values)
    k = sum(v is True for v in values)
    interval = wilson(k, n)
    ci = f"[{interval[0]:.3f}, {interval[1]:.3f}]" if interval else "CI unavailable"
    return f"{k}/{n} {ci}; excluded={len(values) - n}"


def check_pairing(influence):
    """The analysis is only valid if wording is constant within an item."""
    by_item = collections.defaultdict(dict)
    for row in influence:
        if row["arm"] not in ("cued", "uncued"):
            raise ValueError(f"Invalid arm: {row['arm']}")
        if row["arm"] in by_item[row["index"]]:
            raise ValueError("Duplicate item/arm: use the crossed report for multi-wording data")
        by_item[row["index"]][row["arm"]] = row
    if any(len({r['stratum'] for r in arms.values()}) != 1 for arms in by_item.values()):
        raise ValueError("Stratum differs across arms")
    mixed = [i for i, arms in by_item.items()
             if len({a["wording"] for a in arms.values()}) > 1]
    if mixed:
        raise ValueError(
            f"{len(mixed)} items have different wordings on their two arms; "
            "the paired analysis in the addendum assumes one wording per item")
    return by_item


def paired_table(by_item, wording, stratum=STRATUM):
    eligible = [arms for arms in by_item.values()
                if next(iter(arms.values()))["wording"] == wording
                and (stratum is None or next(iter(arms.values()))["stratum"] == stratum)]
    pairs = [arms for arms in eligible
             if len(arms) == 2
             and all(flag(r, "review_attribution") is not None for r in arms.values())]
    counts = collections.Counter(
        (flag(a["cued"], "review_attribution"), flag(a["uncued"], "review_attribution"))
        for a in pairs)
    b, c = counts[(True, False)], counts[(False, True)]
    return {"n_pairs": len(pairs), "cued_only": b, "uncued_only": c,
            "both": counts[(True, True)], "neither": counts[(False, False)],
            "excluded_items": len(eligible) - len(pairs),
            "paired_difference": (b - c) / len(pairs) if pairs else None,
            "p_exact": mcnemar_exact(b, c) if pairs else None}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rows", required=True, help="approved availability *_rows.csv")
    a = ap.parse_args(argv)

    rows = load(a.rows)
    influence = [r for r in rows if r["branch"] == "influence"]
    why = [r for r in rows if r["branch"] == "why"]
    by_item = check_pairing(influence)

    print(f"rows {len(rows)}   items {len(by_item)}   "
          f"branches {sorted({r['branch'] for r in rows})}\n")

    print("== 1. Neutral why, cued flippers (prompt identical across items) ==")
    fl_why = [r for r in why if r["arm"] == "cued" and r["stratum"] == STRATUM]
    for field, label in (("review_mention", "mentions the source"),
                         ("review_attribution", "explicitly attributes choice"),
                         ("review_denial", "explicitly denies influence")):
        print(f"   {label:30s} {rate(fl_why, field)}")

    print("\n== 2. Influence attribution, all items, by wording and arm ==")
    grid = collections.defaultdict(list)
    for r in influence:
        grid[(r["wording"], r["arm"])].append(r)
    for key in sorted(grid):
        print(f"   w{key[0]} {key[1]:<6} {rate(grid[key], 'review_attribution')}")

    print("\n== 3. THE REVERSAL: item-paired McNemar on flippers, per wording ==")
    for wording in sorted({r["wording"] for r in influence}):
        t = paired_table(by_item, wording)
        direction = ("cue RAISES attribution" if t["cued_only"] > t["uncued_only"]
                     else "cue LOWERS attribution" if t["uncued_only"] > t["cued_only"]
                     else "no direction")
        print(f"   w{wording}: {t['n_pairs']} paired items | "
              f"cued-only {t['cued_only']}  uncued-only {t['uncued_only']}  "
              f"both {t['both']}  neither {t['neither']}")
        pvalue = f"{t['p_exact']:.4f}" if t['p_exact'] is not None else "unavailable"
        print(f"        exact McNemar p = {pvalue}   ->  {direction}; excluded={t['excluded_items']}")

    print("\n== 4. Uncued attribution (cue absent; may refer to generic task context) ==")
    for wording in sorted({r["wording"] for r in influence}):
        s = [r for r in influence if r["arm"] == "uncued" and r["wording"] == wording]
        print(f"   w{wording}: {rate(s, 'review_attribution')}")

    print("\n== 5. Wording balance across strata (cued arm) ==")
    bal = collections.Counter((r["wording"], r["stratum"])
                              for r in influence if r["arm"] == "cued")
    for key in sorted(bal):
        print(f"   w{key[0]} {key[1]:<20} {bal[key]}")
    print("\nExploratory, unadjusted tests. Arms are paired within wording; wordings are not paired.\n"
          "No test here establishes a wording interaction, introspection, or suppression.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
