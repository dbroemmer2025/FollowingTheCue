#!/usr/bin/env python3
"""Read one E46 LLM-proposed review entry at a time, with evidence and rationale.

No model calls. Explicit approvals go to a separate CSV; proposals are unchanged.
Entry numbers are 1-based CSV rows, not E46 item IDs.
"""
import argparse
import csv
import json
from pathlib import Path
import sys
import textwrap

from e46_llm_review import FIELDS, atomic_csv, digest, read_csv, utc_now

APPROVAL_FIELDS = ("review_status", "approved_by", "approved_at", "proposal_sha256", "proposal_reviewer",
                   "approval_method", "proposal_labels_json")
OLD_APPROVAL_FIELDS = APPROVAL_FIELDS[:-1]
APPROVAL_METHODS = ("individual", "bulk", "corrected")


def metadata_path(source, explicit=None):
    if explicit is not None:
        return explicit
    manifest = source.with_suffix(".meta.json")
    candidates = [source.with_name(source.stem.removesuffix("_review") + "_rows.csv")]
    if manifest.exists():
        original = Path(json.loads(manifest.read_text())["input_path"])
        if original.name.endswith("_review.csv"):
            candidates.insert(0, original.with_name(original.name[:-len("_review.csv")] + "_rows.csv"))
    return next((p for p in candidates if p.exists()), None)


def load_metadata(path, rows):
    if path is None:
        return {}
    source = {r["audit_id"]: r for r in rows}
    result = {}
    with path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if not {"audit_id", "content_sha256", "index", "arm", "branch"}.issubset(reader.fieldnames or []):
            raise ValueError("--metadata must be availability_rows.csv with IDs, fingerprints, arm and branch")
        for r in reader:
            aid = r["audit_id"]
            if aid in result:
                raise ValueError(f"Duplicate metadata audit_id: {aid}")
            if aid in source and r["content_sha256"] != source[aid]["content_sha256"]:
                raise ValueError(f"Metadata fingerprint differs for {aid}")
            if r["arm"] not in ("cued", "uncued"):
                raise ValueError(f"Invalid metadata arm for {aid}")
            result[aid] = r
    return result


def approvable(row):
    if row.get("llm_status") != "proposed":
        raise ValueError("Only completed proposed rows can be approved; errors/pending rows are not labels")
    for f in FIELDS:
        if row[f] in ("0", "1", "uncertain"):
            continue
        if (f == "review_source_correct" and row[f] == ""
                and json_field(row, "llm_context_source_json").get("presence") == "absent"):
            continue
        raise ValueError(f"Cannot approve missing or invalid {f}")


def valid_label(row, field, value):
    if value in ("0", "1", "uncertain"):
        return True
    return (field == "review_source_correct" and value == ""
            and json_field(row, "llm_context_source_json").get("presence") == "absent")


def validate_labels(row, labels):
    for field in FIELDS:
        if not valid_label(row, field, labels.get(field, "")):
            raise ValueError(f"Cannot approve missing or invalid {field}")


def proposal_labels(row):
    return {field: row[field] for field in FIELDS}


def parse_label_input(value, field):
    value = value.strip().lower()
    if value in ("u", "?"):
        return "uncertain"
    if value in ("na", "n/a", "blank"):
        return ""
    if value in ("0", "1", "uncertain", ""):
        return value
    raise ValueError(f"{field}: use 0, 1, uncertain/u, or blank/na")


def has_positive_label(row):
    return any(row.get(f) == "1" for f in FIELDS)


def only_nonpositive_labels(row):
    """Rows with no positive bins; approvable() handles valid source_correct blanks."""
    return not has_positive_label(row)


def auto_approval_candidates(rows, approvals):
    eligible, skipped, already = [], 0, 0
    for row in rows:
        if not only_nonpositive_labels(row):
            continue
        if row["audit_id"] in approvals.rows:
            already += 1
            continue
        try:
            approvable(row)
        except ValueError:
            skipped += 1
        else:
            eligible.append(row)
    return eligible, skipped, already


class Approvals:
    """Atomic approved-only export, with stale-proposal and concurrent-edit checks."""

    def __init__(self, source, output, fields, rows):
        if source.resolve() == output.resolve():
            raise ValueError("Approved output must differ from the proposal CSV")
        if set(fields) & set(APPROVAL_FIELDS):
            raise ValueError("Open the original proposal CSV, not the approved-only export")
        self.source, self.output = source, output
        self.fields = fields + list(APPROVAL_FIELDS)
        self.source_rows = {r["audit_id"]: r for r in rows}
        self.order = {r["audit_id"]: i for i, r in enumerate(rows)}
        self.saved_bytes = output.read_bytes() if output.exists() else None
        self.rows = {}
        if self.saved_bytes is not None:
            out_fields, saved = read_csv(output)
            old_schema = fields + list(OLD_APPROVAL_FIELDS)
            legacy = out_fields == old_schema
            very_legacy = out_fields == old_schema[:-1]
            missing_method = out_fields == [f for f in self.fields
                                            if f != 'approval_method']
            if (out_fields != self.fields and not legacy and not very_legacy
                    and not missing_method):
                raise ValueError("Approved output schema differs; choose a new --approved-output")
            for r in saved:
                if very_legacy or missing_method:
                    r["approval_method"] = "individual"
                if legacy or very_legacy:
                    r["proposal_labels_json"] = ""
                original = self.source_rows.get(r["audit_id"])
                if original is None:
                    raise ValueError(f"Stale or modified approval for {r['audit_id']}; inspect before using this export")
                method = r["approval_method"]
                original_for_digest = dict(original)
                if method == "corrected":
                    try:
                        labels = json.loads(r["proposal_labels_json"])
                    except ValueError as exc:
                        raise ValueError(f"Invalid corrected-label provenance for {r['audit_id']}") from exc
                    if set(labels) != set(FIELDS) or any(labels[f] != original[f] for f in FIELDS):
                        raise ValueError(f"Corrected approval no longer matches proposal labels for {r['audit_id']}")
                    for f in FIELDS:
                        original_for_digest[f] = labels[f]
                if (r["proposal_sha256"] != digest(original_for_digest)
                        or method not in APPROVAL_METHODS
                        or r["review_status"] != "approved" or not r["approved_by"].strip()
                        or not r["approved_at"].strip() or r["reviewer"] != r["approved_by"]
                        or r["proposal_reviewer"] != original["reviewer"]
                        or any(r[k] != original[k] for k in fields
                               if k not in ("reviewer", *FIELDS))
                        or (method != "corrected"
                            and any(r[k] != original[k] for k in FIELDS))):
                    raise ValueError(f"Stale or modified approval for {r['audit_id']}; inspect before using this export")
                if method == "corrected":
                    validate_labels(original, r)
                else:
                    approvable(original)
                self.rows[r["audit_id"]] = r

    def save(self, row, reviewer, withdraw=False):
        self.save_many([row], reviewer, withdraw=withdraw, method="individual")

    def save_corrected(self, row, labels, reviewer):
        corrected = {**row, **labels}
        validate_labels(row, corrected)
        self.save_many([corrected], reviewer, method="corrected", source_rows=[row])

    def remaining(self, rows):
        eligible, skipped, approved = [], 0, 0
        for row in rows:
            if row["audit_id"] in self.rows:
                approved += 1
                continue
            try:
                approvable(row)
            except ValueError:
                skipped += 1
            else:
                eligible.append(row)
        return eligible, skipped, approved

    def save_many(self, rows, reviewer, withdraw=False, method="bulk", source_rows=None):
        if not reviewer.strip():
            raise ValueError("Use --reviewer YOUR_NAME to record explicit human approvals")
        if method not in APPROVAL_METHODS:
            raise ValueError("Unknown approval method")
        if source_rows is None:
            source_rows = rows
        if len(source_rows) != len(rows):
            raise ValueError("Corrected approval source mismatch")
        _, current = read_csv(self.source)
        current_by_id = {r["audit_id"]: r for r in current}
        source_by_id = {}
        for row in source_rows:
            if current_by_id.get(row["audit_id"]) != row:
                raise ValueError("This proposal changed on disk; reopen the viewer before approving it")
            source_by_id[row["audit_id"]] = row
        current_bytes = self.output.read_bytes() if self.output.exists() else None
        if current_bytes != self.saved_bytes:
            raise ValueError("Approved output changed in another process; reopen the viewer")
        updated = dict(self.rows)
        for row in rows:
            if withdraw:
                updated.pop(row["audit_id"], None)
            else:
                source = source_by_id[row["audit_id"]]
                if method == "corrected":
                    validate_labels(source, row)
                else:
                    approvable(source)
                if method == "bulk" and row["audit_id"] in updated:
                    continue
                updated[row["audit_id"]] = {**row, "reviewer": reviewer.strip(),
                    "review_status": "approved", "approved_by": reviewer.strip(), "approved_at": utc_now(),
                    "proposal_sha256": digest(source), "proposal_reviewer": source["reviewer"],
                    "approval_method": method,
                    "proposal_labels_json": (json.dumps(proposal_labels(source), sort_keys=True)
                                             if method == "corrected" else "")}
        self.output.parent.mkdir(parents=True, exist_ok=True)
        # Serialize writers and recheck after acquiring the lock. A leftover lock
        # after a crash is intentionally fail-closed, not silently overwritten.
        lock = self.output.with_suffix(self.output.suffix + ".lock")
        try:
            handle = lock.open("x")
        except FileExistsError as exc:
            raise ValueError(f"Approval file is locked: {lock}; check for another viewer") from exc
        try:
            with handle:
                if (self.output.read_bytes() if self.output.exists() else None) != self.saved_bytes:
                    raise ValueError("Approved output changed in another process; reopen the viewer")
                atomic_csv(self.output, self.fields, sorted(updated.values(), key=lambda r: self.order[r["audit_id"]]))
                self.saved_bytes = self.output.read_bytes()
                self.rows = updated
        finally:
            lock.unlink()


def block(title, text):
    print(f"\n{title}\n{'-' * len(title)}")
    for line in text.splitlines():
        print(textwrap.fill(line, width=100, replace_whitespace=False,
                            drop_whitespace=False) if line else "")


def json_field(row, key):
    value = row.get(key, "")
    try:
        return json.loads(value) if value else {}
    except ValueError:
        return {}

def check_number_and_zero(s: str) -> tuple[bool, bool]:
    """Checks if a string represents a number, and if so, whether it is zero.

    Returns: (is_number, is_zero)
    """
    try:
        val = float(s.strip())
        # Guard against NaN (not a number in floating point sense)
        if val != val:
            return False
        return val == 0.0
    except ValueError:
        return False


def edit_labels(row, approval=None):
    labels = {field: (approval or row)[field] for field in FIELDS}
    print("\nEdit labels. Enter keeps current value. Use 0, 1, u/uncertain, or na/blank.")
    for field in FIELDS:
        current = labels[field]
        shown = current if current != "" else "blank"
        try:
            raw = input(f"{field} [{shown}]: ")
        except (EOFError, KeyboardInterrupt):
            print("\nEdit cancelled.")
            return None
        if raw.strip() == "":
            continue
        try:
            labels[field] = parse_label_input(raw, field)
            validate_labels(row, labels)
        except ValueError as exc:
            print(f"Invalid edit: {exc}")
            return None
    return labels
    
def show(row, number, total, full_context=False, raw_generation=False, metadata=None, approval=None):
    print("\n" + "=" * 100)
    print(f"ENTRY {number}/{total} | audit_id {row['audit_id']}")
    print(f"Status: {row.get('llm_status') or 'unreviewed'} | Reviewer: {row.get('reviewer') or 'none'}")
    if metadata:
        cue_type = metadata.get('cue_type') or 'none'
        print(f"Arm: {metadata['arm'].upper()} | Cue: {cue_type} | "
              f"Condition: {metadata.get('condition') or '--'} | "
              f"Example: {metadata['index']} | Follow-up: {metadata['branch']}")
    else:
        print("Arm: UNKNOWN | Supply --metadata availability_rows.csv (not inferred from the prompt)")
    if approval:
        print(f"Human review: APPROVED by {approval['approved_by']} at {approval['approved_at']}")
        method = approval.get("approval_method")
        print("Approval method: " + ("BULK (not individually reviewed)"
              if method == "bulk" else method or "individual"))
    else:
        print("Human review: NOT APPROVED. Navigation alone does not approve labels.")
    print("Original LLM proposal is preserved; explicit approvals go to a separate CSV.")
    context = row["conversation_before_response"]
    if not full_context and "\nQ:" in context:
        context = "Q:" + context.rsplit("\nQ:", 1)[1]
        block("CURRENT QUESTION, ORIGINAL ANSWER, AND FOLLOW-UP (demonstrations hidden)", context)
    else:
        block("CONVERSATION BEFORE RESPONSE", context)
    block("RESPONSE BEING REVIEWED", row["response_episode"] or "[empty response]")
    if raw_generation:
        block("RAW GENERATION (later episodes are not scored)", row.get("raw_generation", ""))
    evidence = json_field(row, "llm_evidence_json")
    rationales = json_field(row, "llm_rationales_json")
    context_source = json_field(row, "llm_context_source_json")
    #if context_source:
    #    block("JUDGE'S CONTEXT READING (not a ground-truth label)",
    #          "\n".join(f"{k}: {v}" for k, v in context_source.items()))
    print("\n" + ("HUMAN-APPROVED RESPONSE LABELS" if approval else "PROPOSED RESPONSE LABELS") + "\n" + "-" * 100)
    label_row = approval or row
    for f in FIELDS:
        label = label_row[f]
        if label == "":
            label = ("not applicable (no source in context)" if f == "review_source_correct"
                     and row.get("llm_status") == "proposed" and context_source.get("presence") == "absent"
                     else "unreviewed")
        print(f"\n{f}: {label}")
        if not check_number_and_zero(label):
            for quote in evidence.get(f, []):
                print(textwrap.fill(f"Evidence: {quote}", width=150, initial_indent="  ", subsequent_indent="  "))
            if f in rationales:
                print(textwrap.fill("Reason: " + rationales[f], width=150, initial_indent="  ", subsequent_indent="  "))
    if row.get("llm_error"):
        block("ANNOTATION ERROR (not an uncertain semantic label)", row["llm_error"])
    #if row.get("notes"):
    #    block("NOTES", row["notes"])


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("csv", type=Path)
    p.add_argument("--entry", type=int, default=1, help="1-based original CSV row number")
    p.add_argument("--audit-id", help="select a specific immutable audit ID")
    p.add_argument("--only-uncertain", action="store_true")
    p.add_argument("--only-errors", action="store_true")
    p.add_argument("--full-context", action="store_true", help="also include demonstrations")
    p.add_argument("--raw-generation", action="store_true")
    p.add_argument("--once", action="store_true", help="print one entry and exit")
    p.add_argument("--metadata", type=Path, help="availability_rows.csv; otherwise locate via proposal manifest")
    p.add_argument("--reviewer", default="", help="your human reviewer name, required for approval/withdrawal")
    p.add_argument("--approved-output", type=Path, help="approved-only CSV; default <proposal_stem>_approved.csv")
    p.add_argument("--only-unapproved", action="store_true", help="skip rows already approved when opening")
    p.add_argument("--auto-approve-nonpositive", action="store_true",
                   help=("bulk-approve approvable rows with no positive review bins "
                         "(only 0/uncertain/valid NA), then show only rows with at least one 1"))
    args = p.parse_args(argv)
    try:
        fields, rows = read_csv(args.csv)
        metadata = load_metadata(metadata_path(args.csv, args.metadata), rows)
        approved_path = args.approved_output or args.csv.with_name(args.csv.stem + "_approved.csv")
        approvals = Approvals(args.csv, approved_path, fields, rows)
    except (OSError, ValueError, KeyError) as exc:
        p.error(str(exc))
    if args.entry < 1 or args.entry > len(rows):
        p.error("--entry is outside the CSV")
    if args.auto_approve_nonpositive:
        if not args.reviewer.strip():
            p.error("--auto-approve-nonpositive requires --reviewer YOUR_NAME")
        eligible, skipped, already = auto_approval_candidates(rows, approvals)
        if eligible:
            try:
                approvals.save_many(eligible, args.reviewer, method="bulk")
            except (OSError, ValueError) as exc:
                p.error(str(exc))
        print(f"Auto-approved {len(eligible)} non-positive rows; "
              f"{already} already approved, {skipped} pending/error/invalid rows skipped. "
              f"{len(approvals.rows)} approved rows saved to {approved_path}")
    selected = [(i, r) for i, r in enumerate(rows, 1)
                if (not args.only_uncertain or any(r[f] == "uncertain" for f in FIELDS))
                and (not args.only_errors or r.get("llm_status") == "error")]
    if args.auto_approve_nonpositive:
        selected = [(i, r) for i, r in selected if has_positive_label(r)]
    if args.only_unapproved:
        selected = [(i, r) for i, r in selected if r["audit_id"] not in approvals.rows]
    if not selected:
        if args.auto_approve_nonpositive:
            print("No positive-bin entries match these filters; manual review queue is empty.")
            return 0
        p.error("No entries match these filters")
    if args.audit_id:
        matches = [i for i, (_, r) in enumerate(selected) if r["audit_id"] == args.audit_id]
        if not matches:
            p.error("Audit ID not found in filtered entries")
        pos = matches[0]
    else:
        pos = next((i for i, (n, _) in enumerate(selected) if n >= args.entry), None)
        if pos is None:
            p.error("No matching entries at or after --entry")
    while True:
        number, row = selected[pos]
        show(row, number, len(rows), args.full_context, args.raw_generation,
             metadata.get(row["audit_id"]), approvals.rows.get(row["audit_id"]))
        if args.once or not sys.stdin.isatty():
            return 0
        try:
            command = input("\n[Enter/a] approve + next | e edit labels | A approve all remaining | "
                            "u withdraw approval | n skip/next | p previous | f full context | "
                            "g CSV row | q quit: ").strip()
        except (EOFError, KeyboardInterrupt):
            return 0
        if command == "A":
            eligible, skipped, already = approvals.remaining([r for _, r in selected[pos:]])
            print(f"Remaining selection (including current entry): {len(eligible)} to approve, "
                  f"{already} already approved, {skipped} pending/error/invalid rows skipped.")
            if not eligible:
                print("Nothing eligible to bulk-approve.")
                continue
            if not args.reviewer.strip():
                print("Not saved: use --reviewer YOUR_NAME to record explicit human approvals")
                continue
            try:
                confirm = input("Type APPROVE to accept these labels in bulk (not individual review): ")
            except (EOFError, KeyboardInterrupt):
                return 0
            if confirm.strip() != "APPROVE":
                print("Bulk approval cancelled; nothing changed.")
                continue
            try:
                approvals.save_many(eligible, args.reviewer)
            except (OSError, ValueError) as exc:
                print(f"Not saved: {exc}")
                continue
            print(f"Bulk-approved {len(eligible)} rows; {len(approvals.rows)} approved rows saved to {approved_path}")
            return 0
        command = command.lower()
        if command == "":
            command = "a"
        if command == "q":
            print(f"{len(approvals.rows)} approved rows in {approved_path}")
            return 0
        if command == "e":
            if not args.reviewer.strip():
                print("Not saved: use --reviewer YOUR_NAME to record corrected approvals")
                continue
            labels = edit_labels(row, approvals.rows.get(row["audit_id"]))
            if labels is None:
                continue
            try:
                approvals.save_corrected(row, labels, args.reviewer)
            except (OSError, ValueError) as exc:
                print(f"Not saved: {exc}")
                continue
            print(f"Corrected approval saved; {len(approvals.rows)} approved rows saved to {approved_path}")
            command = "n"
        if command in ("a", "u"):
            try:
                approvals.save(row, args.reviewer, withdraw=command == "u")
            except (OSError, ValueError) as exc:
                print(f"Not saved: {exc}")
                continue
            print(f"{'Approved' if command == 'a' else 'Approval withdrawn'}; "
                  f"{len(approvals.rows)} approved rows saved to {approved_path}")
            if command == "u":
                continue
            command = "n"
        if command == "n":
            if pos + 1 == len(selected):
                print("End of selected entries.")
                return 0
            pos += 1
        elif command == "p":
            pos = max(0, pos - 1)
        elif command == "f":
            args.full_context = not args.full_context
        elif command.startswith("g "):
            try:
                wanted = int(command[2:])
                pos = next(i for i, (n, _) in enumerate(selected) if n == wanted)
            except (ValueError, StopIteration):
                print("That CSV row is not in the selected entries.")


if __name__ == "__main__":
    raise SystemExit(main())
