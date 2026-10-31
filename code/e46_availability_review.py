#!/usr/bin/env python3
"""Summarize reviewed availability data, queue audits, and export adjudications.

Offline companion: does not change the frozen availability implementation or
overwrite inputs. Edits are accepted only through an explicit adjudication CSV.
"""
import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
import random

from e46_availability import REVIEW_FIELDS, file_hash
from e46_llm_review import digest, utc_now


def read(path):
    with Path(path).open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if any(None in r or any(v is None for v in r.values()) for r in rows):
        raise ValueError("Malformed CSV")
    ids = [r["audit_id"] for r in rows]
    if len(set(ids)) != len(ids) or any(not i for i in ids):
        raise ValueError("Duplicate or empty audit IDs")
    return rows


def write(path, rows, fields=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def joined(rows_path, annotations_path):
    rows, annotations = read(rows_path), read(annotations_path)
    by = {r["audit_id"]: r for r in rows}
    for a in annotations:
        r = by.get(a["audit_id"])
        if r is None or r["content_sha256"] != a["content_sha256"]:
            raise ValueError(f"Annotation ID/fingerprint mismatch: {a['audit_id']}")
        for field in REVIEW_FIELDS:
            if a.get(field, "") not in ("", "0", "1", "uncertain"):
                raise ValueError(f"Invalid {field}")
    return rows, {r["audit_id"]: r for r in annotations}


def distribution(rows, field):
    counts = Counter()
    for r in rows:
        value = r.get(field, "")
        if value not in ("", "0", "1", "uncertain"):
            raise ValueError(f"Invalid {field}")
        if r["response_available"].lower() != "true":
            key = "missing_response"
        elif field in ("review_source_correct", "review_source_identity", "review_suggested_option", "review_quote") and r["arm"] == "uncued" and value == "":
            key = "not_applicable"
        else:
            key = {"": "unreviewed", "uncertain": "uncertain", "0": "negative", "1": "positive"}[value]
        counts[key] += 1
    return {k: counts[k] for k in ("positive", "negative", "uncertain", "unreviewed",
                                  "not_applicable", "missing_response")}


def summarize(rows, annotations=None):
    groups = defaultdict(list)
    provenance = Counter()
    for r in rows:
        r = dict(r)
        if annotations is not None:
            a = annotations.get(r["audit_id"], {})
            r.update({f: a.get(f, "") for f in REVIEW_FIELDS})
            method = ("adjudicated" if a.get("adjudicated_by") else a.get("approval_method", "unspecified"))
            provenance[method if a else "unreviewed"] += 1
        groups["/".join(r[k] for k in ("stratum", "arm", "branch", "wording"))].append(r)
    cells = {}
    for key, rr in sorted(groups.items()):
        cells[key] = {"n_rows": len(rr),
                      "response_coverage": sum(r["response_available"].lower() == "true" for r in rr),
                      "automatic_mentions": sum(r.get("auto_mention", "").lower() == "true" for r in rr),
                      "reviewed": {f: distribution(rr, f) for f in REVIEW_FIELDS}}
    return {"version": "e46_review_summary_v1", "n_rows": len(rows),
            "approval_methods": dict(provenance), "by_cell": cells}


def summary(args):
    if args.annotations:
        rows, ann = joined(args.rows, args.annotations)
    else:
        rows, ann = read(args.rows), None
    result = summarize(rows, ann)
    print("Response coverage is not source recall. AUTO is a lexical screen.")
    print("Reviewed fields: positive/binary-reviewed; ? uncertain, blank unreviewed, NA not applicable.")
    for key, cell in result["by_cell"].items():
        print(f"\n{key}  responses={cell['response_coverage']}/{cell['n_rows']}  AUTO mentions={cell['automatic_mentions']}")
        for field, c in cell["reviewed"].items():
            print(f"  {field:<24} {c['positive']}/{c['positive'] + c['negative']} "
                  f"?={c['uncertain']} blank={c['unreviewed']} NA={c['not_applicable']} missing={c['missing_response']}")
    print("Approval methods:", result["approval_methods"] if ann is not None else "not supplied")
    if args.json:
        with Path(args.json).open("x") as f:
            json.dump(result, f, indent=2)
            f.write("\n")
    return result


def audit(args):
    rows, ann = joined(args.rows, args.annotations)
    if args.random_n < 0 or args.random_n > len(ann):
        raise ValueError("--random-n must be between zero and the number of annotated rows")
    sampled = set(random.Random(args.seed).sample(sorted(ann), args.random_n))
    queued = []
    for r in rows:
        a = ann.get(r["audit_id"])
        if not a:
            continue
        reasons = []
        if r["arm"] == "cued" and (r["index"], r["branch"]) in {
                ("69", "influence"), ("84", "recall"), ("2", "recall"), ("22", "recall")}:
            reasons.append("previously_flagged_example")
        if r["arm"] == "cued" and r["branch"] == "recall" and a["review_source_correct"] == "1":
            reasons.append("claimed_full_recall")
        if r["arm"] == "cued" and r["branch"] == "why" and a["review_mention"] == "1":
            reasons.append("neutral_source_mention")
        if a["review_attribution"] == a["review_denial"] == "1":
            reasons.append("attribution_denial_conflict")
        if r["arm"] == "uncued" and r["branch"] == "influence" and a["review_attribution"] == "1":
            reasons.append("uncued_attribution")
        if a["review_source_correct"] == "1" and a["review_source_present"] != "1":
            reasons.append("identification_presence_disagreement")
        if r["audit_id"] in sampled:
            reasons.append("random_sample")
        if not reasons:
            continue
        queued.append({"audit_id": r["audit_id"], "content_sha256": r["content_sha256"],
                       "index": r["index"], "arm": r["arm"], "branch": r["branch"],
                       "wording": r["wording"], "selection_reasons": ";".join(reasons),
                       "random_sample": str(r["audit_id"] in sampled),
                       "baseline_annotation_sha256": digest(a),
                       "approval_method": a.get("approval_method", "unspecified"),
                       "conversation_before_response": a["conversation_before_response"],
                       "response_episode": a["response_episode"],
                       **{f"original_{f}": a[f] for f in REVIEW_FIELDS},
                       **{f: a[f] for f in REVIEW_FIELDS},
                       "decision": "", "adjudicator": "", "adjudication_note": ""})
    if not queued:
        raise ValueError("No rows selected")
    manifest = Path(args.out).with_suffix(".meta.json")
    if Path(args.out).exists() or manifest.exists():
        raise ValueError("Audit output exists; choose a new path")
    write(args.out, queued)
    with manifest.open("x") as f:
        json.dump({"version": "e46_adjudication_v1", "seed": args.seed, "random_n": args.random_n,
                   "rows_sha256": file_hash(args.rows), "annotations_sha256": file_hash(args.annotations),
                   "scope": "Targeted audit plus separately flagged uniform random sample; blank decisions are not approvals."}, f, indent=2)
    print(f"Queued {len(queued)} rows, including {len(sampled)} uniform random selections: {args.out}")


def apply(args):
    baseline = read(args.annotations)
    by = {r["audit_id"]: r for r in baseline}
    updates = {}
    for c in read(args.corrections):
        a = by.get(c["audit_id"])
        if a is None or c["content_sha256"] != a["content_sha256"] or c["baseline_annotation_sha256"] != digest(a):
            raise ValueError("Stale or unmatched correction; recreate the audit queue")
        if c["decision"] == "":
            continue
        if c["decision"] != "approve" or not c["adjudicator"].strip() or not c["adjudication_note"].strip():
            raise ValueError("Use decision=approve with adjudicator and note for each accepted row")
        for field in REVIEW_FIELDS:
            allowed = ("0", "1", "uncertain", "") if field == "review_source_correct" else ("0", "1", "uncertain")
            if c[field] not in allowed:
                raise ValueError(f"Invalid correction for {field}")
            if c[field] == "" and a[field] != "":
                raise ValueError("Do not change an applicable source-correct label to blank")
        updates[c["audit_id"]] = c
    if not updates:
        raise ValueError("No explicit adjudications to apply")
    exported, changes = [], []
    for a in baseline:
        c = updates.get(a["audit_id"])
        labels = {f: c[f] if c else a[f] for f in REVIEW_FIELDS}
        exported.append({k: a[k] for k in ("audit_id", "content_sha256", "conversation_before_response", "response_episode")}
                        | labels | {"reviewer": c["adjudicator"] if c else a["reviewer"],
                                    "notes": c["adjudication_note"] if c else a.get("notes", ""),
                                    "approval_method": "individual" if c else a.get("approval_method", "unspecified"),
                                    "prior_annotation_sha256": digest(a),
                                    "prior_approval_method": a.get("approval_method", "unspecified"),
                                    "adjudicated_by": c["adjudicator"] if c else "",
                                    "adjudicated_at": utc_now() if c else ""})
        for field in REVIEW_FIELDS:
            if a[field] != labels[field]:
                changes.append({"audit_id": a["audit_id"], "field": field, "old": a[field], "new": labels[field]})
    diff_path = Path(args.out).with_suffix(".changes.json")
    if Path(args.out).exists() or diff_path.exists():
        raise ValueError("Adjudication output exists; choose a new path")
    write(args.out, exported)
    with diff_path.open("x") as f:
        json.dump({"annotations_sha256": file_hash(args.annotations),
                   "corrections_sha256": file_hash(args.corrections),
                   "adjudicated_rows": len(updates), "changes": changes}, f, indent=2)
    print(f"Adjudicated {len(updates)} rows; changed {len(changes)} labels. Full annotation export: {args.out}")
    for c in changes:
        print(f"  {c['audit_id']} {c['field']}: {c['old']!r} -> {c['new']!r}")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(required=True)
    s = sub.add_parser("summary")
    s.add_argument("--rows", required=True)
    s.add_argument("--annotations")
    s.add_argument("--json")
    s.set_defaults(func=summary)
    s = sub.add_parser("audit")
    s.add_argument("--rows", required=True)
    s.add_argument("--annotations", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--random-n", type=int, default=30)
    s.add_argument("--seed", type=int, default=46)
    s.set_defaults(func=audit)
    s = sub.add_parser("apply")
    s.add_argument("--annotations", required=True)
    s.add_argument("--corrections", required=True)
    s.add_argument("--out", required=True)
    s.set_defaults(func=apply)
    args = p.parse_args(argv)
    try:
        args.func(args)
    except (ValueError, OSError, KeyError) as exc:
        p.exit(2, f"Availability review: {exc}\n")


if __name__ == "__main__":
    main()
