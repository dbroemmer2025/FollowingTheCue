#!/usr/bin/env python3
"""Display an E46/E47 original trace and its follow-up responses by item ID.

Use --run-dir for the fixed workflow (why, recall, influence). Without it,
defaults to the legacy Qwen2.5-7B quota caches, including their archive location.
No tokenizer or GPU is needed. Only load trusted project-produced pickle files.
Example: ./show_with_why.sh 45 --run-dir ../e46/qwen_fresh_01 --arm both
Example: ./show_with_why.sh 3 --run-dir ../e47/qwen_run --condition disagree_sycophancy
"""
from __future__ import annotations

import argparse
from pathlib import Path
import pickle
import re
import sys

from inspect_traces import show_block
from e46_prompts_cue import extract_answer

ROOT = Path(__file__).resolve().parent.parent
class _Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, value):
        for stream in self.streams:
            stream.write(value)
        return len(value)

    def flush(self):
        for stream in self.streams:
            stream.flush()


def _start_text_output(path):
    if path is None:
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    sink = path.open("w", encoding="utf-8")
    original = sys.stdout
    sys.stdout = _Tee(original, sink)
    return original, sink


def _finish_text_output(handle, path):
    if handle is None:
        return
    original, sink = handle
    sys.stdout.flush()
    sys.stdout = original
    sink.close()
    print(f"Wrote text report: {path.resolve()}")


def _check_metadata(traces, followups):
    for field in ("hf", "model", "hf_revision", "replication_design_id", "seed",
                  "substrate", "cue_type", "prompt_regime", "cot_counterbalance"):
        a = traces.get("metadata", {}).get(field)
        b = followups.get("metadata", {}).get(field)
        if a is not None and b is not None and a != b:
            raise ValueError(f"Cache metadata mismatch for {field}: {a!r} != {b!r}")


def _verify_context(original, followup, index, arm):
    for first, second in (("prompt", "prompt1"), ("text", "episode1")):
        if not isinstance(original.get(first), str) or not isinstance(followup.get(second), str):
            raise ValueError(f"Cannot verify item {index}, arm {arm}: missing {first}/{second}")
        if original[first] != followup[second]:
            raise ValueError(f"Item {index}, arm {arm}: original {first} does not match "
                             f"follow-up {second}; check that the caches come from the same run")
    truth = original.get("item", {}).get("correct_letter")
    if truth and followup.get("correct_letter") and truth != followup["correct_letter"]:
        raise ValueError(f"Item {index}: ground truth differs between caches")


def _one(rows, index, arm, arm_field, source):
    matches = [(pos, r) for pos, r in enumerate(rows)
               if r.get("index") == index and r.get(arm_field) == arm]
    if len(matches) != 1:
        raise ValueError(f"{source}: expected one record for item {index}, "
                         f"arm {arm}; found {len(matches)}")
    return matches[0]


def matching_pair(traces, followups, index, arm, task="cue"):
    """Join stored IDs, not list offsets, and verify the replayed context."""
    if traces.get("kind") != "e46_pretest_traces":
        raise ValueError("Original cache is not an E46 pretest trace cache")
    if followups.get("kind") != "e46_followups":
        raise ValueError("Follow-up cache is not an E46 follow-up cache")
    _check_metadata(traces, followups)
    raw = traces.get("raw_outputs", {}).get(task)
    if raw is None:
        raise ValueError(f"Original cache has no task {task!r}")
    pos, original = _one(raw, index, arm, "condition", "Original cache")
    _, followup = _one(followups.get("rows", []), index, arm, "arm", "Follow-up cache")
    _verify_context(original, followup, index, arm)
    if not isinstance(followup.get("text2"), str):
        raise ValueError(f"No generated why trace for item {index}, arm {arm}")
    return pos, original, followup


def matching_followups(traces, followups, availability, index, arm, task="cue"):
    """Merge neutral and branched caches, checking every displayed continuation."""
    if traces.get("kind") != "e46_pretest_traces":
        raise ValueError("Original cache is not an E46 pretest trace cache")
    if task not in traces.get("raw_outputs", {}):
        raise ValueError(f"Original cache has no task {task!r}")
    pos, original = _one(traces["raw_outputs"][task], index, arm, "condition", "Original cache")
    branches = {}
    for blob, kind in ((followups, "e46_followups"),
                       (availability, "e46_availability_responses")):
        if blob is None:
            continue
        if blob.get("kind") != kind:
            raise ValueError(f"Expected {kind} cache; got {blob.get('kind')!r}")
        _check_metadata(traces, blob)
        seen = set()
        for row in blob.get("rows", []):
            if row.get("index") != index or row.get("arm") != arm:
                continue
            branch = "why" if kind == "e46_followups" else row.get("branch")
            if not isinstance(branch, str) or not branch:
                raise ValueError(f"Missing branch for item {index}, arm {arm}")
            if branch in seen:
                raise ValueError(f"Duplicate {branch} record for item {index}, arm {arm}")
            seen.add(branch)
            _verify_context(original, row, index, arm)
            if branch in branches:
                # Reused why rows must agree; never choose a newer-looking response.
                for field in ("prompt2", "text2", "ids2", "answer", "correct_letter", "stratum"):
                    if branches[branch].get(field) != row.get(field):
                        raise ValueError(f"Conflicting {branch} {field} for item {index}, arm {arm}")
            branches[branch] = row
    order = {"why": 0, "recall": 1, "influence": 2}
    return pos, original, sorted(branches.items(), key=lambda p: (order.get(p[0], 3), p[0]))


def _is_e47(blob):
    version = blob.get("version", "") if isinstance(blob, dict) else ""
    return isinstance(version, str) and version.startswith("e47_v")


def _check_e47_artifacts(traces, followups, design):
    if not _is_e47(traces) or not isinstance(traces.get("rows"), list):
        raise ValueError("Original cache is not an E47 trace artifact")
    expected = traces.get("design_id")
    for label, blob in (("follow-up", followups), ("design", design)):
        if blob is None:
            continue
        if blob.get("design_id") != expected:
            raise ValueError(f"E47 {label} artifact has a different design_id")
    if followups is not None and (
            not _is_e47(followups) or not isinstance(followups.get("rows"), list)):
        raise ValueError("Follow-up cache is not an E47 artifact")


def matching_e47_followups(traces, followups, design, index, condition):
    """Join an E47 trace to its neutral-why row using the stored parent ID."""
    _check_e47_artifacts(traces, followups, design)
    pos, original = _one(traces["rows"], index, condition, "condition", "Original cache")
    branches = []
    if followups is None:
        return pos, original, branches
    matches = [r for r in followups["rows"] if r.get("parent_id") == original.get("id")]
    if len(matches) > 1:
        raise ValueError(f"Follow-up cache: expected at most one parent match for item {index}, "
                         f"condition {condition}; found {len(matches)}")
    if not matches:
        # A row with the same visible coordinates but a different parent is a
        # provenance error, not an absent follow-up that can be ignored.
        coordinate = [r for r in followups["rows"]
                      if r.get("index") == index and r.get("condition") == condition]
        if coordinate:
            raise ValueError(f"Item {index}, condition {condition}: follow-up parent_id does "
                             "not match the selected initial trace")
        return pos, original, branches
    row = matches[0]
    for field in ("index", "source_id", "condition", "cue_letter"):
        if row.get(field) != original.get(field):
            raise ValueError(f"Item {index}, condition {condition}: E47 follow-up {field} "
                             "does not match its parent trace")
    history = original.get("prompt", "") + original.get("episode", "")
    if not isinstance(row.get("prompt"), str) or not row["prompt"].startswith(history):
        raise ValueError(f"Item {index}, condition {condition}: follow-up prompt does not "
                         "contain the exact saved initial prompt and episode")
    if not isinstance(row.get("text"), str):
        raise ValueError(f"No generated why trace for item {index}, condition {condition}")
    return pos, original, [("why", row)]


def cache_paths(args):
    """Resolve one explicit run; never guess the latest run from timestamps."""
    if args.run_dir is not None:
        if not args.run_dir.is_dir():
            raise ValueError(f"Run directory does not exist: {args.run_dir}")
        traces = args.traces or args.run_dir / "traces.pkl"
    elif args.traces is not None:
        traces = args.traces
    else:
        traces = ROOT / "e46/e46_traces_qwen2.5_7b_quota.pkl"
        if not traces.exists():
            traces = ROOT / "e46/archive" / traces.name
    directory = args.run_dir or traces.parent
    why = args.followups
    if why is None:
        candidates = [directory / "followups.pkl"]
        if args.run_dir is None and args.traces is None:
            candidates.append(directory / "e46_followup_qwen2.5_7b_quota.pkl")
        why = next((p for p in candidates if p.is_file()), None)
    availability = args.availability
    if availability is None:
        candidate = directory / "availability_responses.pkl"
        availability = candidate if candidate.is_file() else None
    design = directory / "design.json"
    design = design if design.is_file() else None
    return traces, why, availability, design


def _load(path):
    if path is None:
        return None
    if path.suffix == ".json":
        import json
        with path.open() as f:
            return json.load(f)
    with path.open("rb") as f:
        return pickle.load(f)


def answer_header(traces, original, followup, index, arm, task="cue"):
    """Compare both original arms with the cached ground-truth letter."""
    _, uncued = _one(traces["raw_outputs"][task], index, "uncued",
                     "condition", "Original cache")
    truth = original.get("item", {}).get("correct_letter")
    stored_truth = followup.get("correct_letter")
    if truth and stored_truth and truth != stored_truth:
        raise ValueError(f"Item {index}: ground truth differs between caches")
    truth = truth or stored_truth

    def describe(record):
        answer = extract_answer(record.get("text", ""))
        if answer is None:
            return "unparsed | accurate: unknown"
        accurate = "unknown" if not truth else ("yes" if answer == truth else "no")
        return f"{answer} | accurate: {accurate}"

    return (f"Ground truth: {truth or 'unknown'}\n"
            f"Selected arm ({arm}) answer: {describe(original)}\n"
            f"Original uncued answer: {describe(uncued)}")


def _e47_truth(design, original):
    if not isinstance(design, dict):
        return None
    matches = [item for item in design.get("items", [])
               if item.get("index") == original.get("index")]
    if original.get("source_id"):
        by_source = [item for item in matches
                     if item.get("source_id") == original.get("source_id")]
        if by_source:
            matches = by_source
    if len(matches) != 1:
        return None
    return matches[0].get("correct_letter")


def answer_header_e47(traces, design, original, index, condition):
    """Describe the selected E47 condition relative to its uncued baseline."""
    _, uncued = _one(traces["rows"], index, "uncued", "condition", "Original cache")
    truth = _e47_truth(design, original)

    def answer(record):
        return record.get("answer") or extract_answer(record.get("episode", record.get("text", "")))

    def describe(record):
        value = answer(record)
        if value is None:
            return "unparsed | accurate: unknown"
        accurate = "unknown" if not truth else ("yes" if value == truth else "no")
        return f"{value} | accurate: {accurate}"

    selected, baseline = answer(original), answer(uncued)
    cue = original.get("cue_letter") or "none"
    changed = "unknown" if selected is None or baseline is None else ("yes" if selected != baseline else "no")
    to_cue = "n/a" if cue == "none" else (
        "unknown" if selected is None else ("yes" if selected == cue and selected != baseline else "no"))
    return (f"Ground truth: {truth or 'unknown'}\n"
            f"Selected condition ({condition}) answer: {describe(original)}\n"
            f"Original uncued answer: {describe(uncued)}\n"
            f"Cue letter: {cue} | changed from uncued: {changed} | switched to cue: {to_cue}")


def _selected_e47_conditions(rows, index, arm, condition):
    available = []
    for row in rows:
        name = row.get("condition")
        if row.get("index") == index and name not in available:
            available.append(name)
    if not available:
        raise ValueError(f"Original cache has no E47 rows for item {index}")
    if condition:
        wanted = available if condition == "all" else [condition]
        missing = [name for name in wanted if name not in available]
        if missing:
            raise ValueError(
                f"Conditions absent for item {index}: {missing}; "
                f"available={available}")
        return tuple(wanted)
    if arm == "uncued":
        return ("uncued",)
    if arm == "cued":
        return tuple(name for name in available if name != "uncued")
    return tuple(available)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("index", type=int, help="stored E46/E47 item ID, not a cache row position")
    ap.add_argument("--arm", choices=("cued", "uncued", "both"), default="cued")
    ap.add_argument("--condition",
                    help=("E47 condition to show, or 'all'; validated against "
                          "the conditions actually frozen in the run"))
    ap.add_argument("--run-dir", type=Path, help="directory containing traces.pkl, followups.pkl, "
                    "and availability_responses.pkl; optional follow-up caches are auto-discovered")
    ap.add_argument("--traces", type=Path, help="explicit original cache (overrides --run-dir)")
    ap.add_argument("--followups", type=Path, help="explicit neutral why cache")
    ap.add_argument("--availability", type=Path, help="explicit branched follow-up response cache")
    ap.add_argument("--task", default="cue")
    ap.add_argument("--raw", action="store_true", help="show Python repr with whitespace escapes")
    ap.add_argument("--full-followup-prompts", action="store_true",
                    help="also show each follow-up's full saved input, including repeated history")
    ap.add_argument("--output-text", type=Path,
                    help="also write the complete displayed report to this UTF-8 text file")
    args = ap.parse_args(argv)
    try:
        paths = cache_paths(args)
        traces, followups, availability, design = map(_load, paths)
        displays = []
        family = "E47" if _is_e47(traces) else "E46"
        if family == "E47":
            if availability is not None:
                raise ValueError("E47 does not use E46 availability_responses.pkl; remove "
                                 "--availability or select the E46 run directory")
            for condition in _selected_e47_conditions(
                    traces["rows"], args.index, args.arm, args.condition):
                pos, original, branches = matching_e47_followups(
                    traces, followups, design, args.index, condition)
                answers = answer_header_e47(traces, design, original, args.index, condition)
                displays.append((condition, pos, original, branches, answers))
        else:
            if args.condition:
                raise ValueError("--condition applies only to E47 artifacts")
            arms = ("uncued", "cued") if args.arm == "both" else (args.arm,)
            for arm in arms:
                pos, original, branches = matching_followups(
                    traces, followups, availability, args.index, arm, args.task)
                context = branches[0][1] if branches else {}
                answers = answer_header(traces, original, context, args.index, arm, args.task)
                displays.append((arm, pos, original, branches, answers))
    except (OSError, ValueError, KeyError, EOFError, pickle.UnpicklingError) as exc:
        ap.error(str(exc))

    text_output = _start_text_output(args.output_text)
    for label, path in zip(("Original cache", "Why cache", "Availability cache", "Design"), paths):
        print(f"{label}: {path.resolve() if path else 'not supplied/found'}")
    for arm, pos, original, branches, answers in displays:
        dimension = "CONDITION" if family == "E47" else "ARM"
        print(f"\nITEM {args.index} | {dimension} {arm} | original cache row {pos}")
        if branches:
            if family == "E47":
                print("Verified: the why row names this trace as its parent and contains its exact saved history.")
            else:
                print("Verified: original prompt and trace match every displayed follow-up's saved turn 1.")
                print(f"Stratum: {branches[0][1].get('stratum', 'unknown')}")
        print(answers)
        show_block("original prompt (full)", original["prompt"], args.raw)
        show_block("original trace (full generation)", original["text"], args.raw)
        if family == "E46":
            print("\nFollow-up branches are independent continuations of the original answer, not consecutive turns.")
        else:
            print("\nE47 stores one neutral why continuation for each condition.")
        if not branches:
            print("No follow-up records for this item/arm in the loaded caches "
                  "(an unparsed original answer may have been skipped).")
        for branch, row in branches:
            hit_cap = row.get("hit_cap2", row.get("hit_cap", "not recorded"))
            print(f"\n--- {branch.upper()} | wording: {row.get('wording', 'not recorded')} "
                  f"| hit cap: {hit_cap} ---")
            prompt = row.get("prompt2", row.get("prompt"))
            question = row.get("question")
            if not isinstance(question, str) and branch == "why" and isinstance(prompt, str):
                # E46 legacy and E47 caches store the question inside the model input.
                matches = re.findall(r"Why did you say the answer was \([A-D]\)\?", prompt)
                question = matches[-1] if matches else None
            if isinstance(question, str):
                show_block(f"{branch} question", question, args.raw)
            else:
                print(f"{branch} question: not available separately; "
                      "use --full-followup-prompts to inspect the saved input")
            if args.full_followup_prompts:
                if isinstance(prompt, str):
                    show_block(f"{branch} prompt (full saved model input)", prompt, args.raw)
                else:
                    print(f"{branch} prompt: not stored in this cache; not reconstructed")
            response = row.get("text2", row.get("text"))
            if isinstance(response, str):
                show_block(f"{branch} response (full generation)", response, args.raw)
                if not response.strip():
                    print("Response is empty.")
            else:
                print(f"{branch} response: not generated/stored")
        if family == "E47":
            if "why" not in dict(branches):
                print("Follow-up not present: why")
        else:
            missing = [b for b in ("why", "recall", "influence") if b not in dict(branches)]
            if missing:
                print("Follow-ups not present: " + ", ".join(missing))
    _finish_text_output(text_output, args.output_text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
