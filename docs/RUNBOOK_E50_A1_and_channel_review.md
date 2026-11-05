# Runbook: E50 Amendment A1 and the channel-aware review

Read `docs/AMENDMENT_E50_A1.md` first. Order matters:

1. amend
2. generate the main block
3. freeze the rubric
4. review
5. audit

## 1. Record the A1 decisions (local, no GPU)

```bash
cd code
python e50_amendment_a1.py status
for run in qwen3_8b_movie_v2 qwen3_8b_temporal_v3 magistral_small_2509_movie_v4 \
           deepseek_r1_qwen7b_movie_v3 gpt_oss_20b_movie_v5 deepseek_r1_llama8b_movie_v4; do
  python e50_amendment_a1.py amend --run ../e50/$run || break
done
python e50_amendment_a1.py status
```

`amend` writes one binding `amendment_A1_<mode>.json` per run and mode. Each
file records the amendment document's hash, the original interim decision,
and the hashes of the progress file and the driver.

| Run / mode | A1 decision | Why |
|---|---|---|
| Qwen3-8B movie, thinking on | GO | |
| Qwen3-8B movie, thinking off | GO | |
| Qwen3-8B temporal, thinking on | GO | user-preference disclosure flagged not estimable (0 switches) |
| Magistral movie | GO | |
| DeepSeek-R1-Qwen-7B | NO-GO | incomplete forced reads |
| gpt-oss-20b | NO-GO | valid and forced rates |
| DeepSeek-R1-Llama-8B | NO-GO | validity |

`amend` refuses to run if main-block records already exist.

## 2. Generate the main blocks (Colab, vLLM)

```bash
status=0
for spec in "qwen3_8b_movie_v2 think nothink" "magistral_small_2509_movie_v4 think" \
            "qwen3_8b_temporal_v3 think"; do
  set -- $spec; run=$1; shift
  python e50_amendment_a1.py run-main --run ../e50/$run --modes "$@" \
    || { status=$?; echo "run-main FAILED for $run"; break; }
done
test "$status" -eq 0
```

`run-main` generates with each run's own frozen code, verified file by file
against its `design.json`. `qwen3_8b_movie_v2` uses the exact runner restored
from git commit `ae09a32` (`code/archive_e50/`). The command:

- appends main-block records to `progress.jsonl`;
- writes `assignment_<mode>_main.json` and `main_status_<mode>.json`;
- never touches the interim files.

It refuses to run if the A1 decision is NO-GO, or if the driver or the
amendment document changed after `amend`.

## 3. Freeze the rubric and export review items

```bash
python e50_channel_review.py freeze
python e50_channel_review.py export --runs ../e50/qwen3_8b_movie_v2 ../e50/magistral_small_2509_movie_v4 ../e50/qwen3_8b_temporal_v3
```

Read the rubric in `e50_channel_review.py` (`SYSTEM`, `USER_TEMPLATE`,
`REPAIR_TEMPLATE`, `SCHEMA`) before freezing. It explicitly excludes "the user" used only as the
asker, which is the false positive the lexical screen falls into.

The v1 20-item pilot produced 11 invalid labels. V2 reduced this to 3/20,
but one accepted label treated a rejection of option A as a cue mention
without any reference to the cue or its source. V3 explicitly distinguishes
option comparison from cue acknowledgment and is frozen separately under
`reports/e50_channel_review_v3`. Do not mix labels from v1, v2, and v3.
Re-run `freeze` and `export` with the v3 script even if an earlier export
exists. See `docs/AMENDMENT_E50_A1_REVIEW_V3.md` for provenance.

`export` creates one item per disagreement-trial response and channel:

- thinking on: reasoning trace and final response;
- thinking off: final response.

Both blocks are included and labelled.
The local review environment needs SciPy for exact binomial intervals and
paired McNemar tests; generation on Colab does not need the review script.

## 4. Review (local, API)

```bash
python e50_channel_review.py review --provider openai --model gpt-5.6-luna --reasoning-effort low --threads 8 --limit 40
python e50_channel_review.py revalidate
python e50_channel_review.py inspect-pilot
# Add --all to inspect accepted no-mention labels too.
# Only continue after checking positives, invalids, and a sample of negatives:
python e50_channel_review.py review --provider openai --model gpt-5.6-luna --reasoning-effort low --threads 8
python e50_channel_review.py report
```

The first call records the reviewer model; later calls with a different one
are refused. A positive label must quote evidence from the channel,
and the stance must agree with the mention label. On a validation failure,
the second call receives the original text, its first response, and the exact
validation errors. Both responses and their validation outcomes are saved in
`attempt_log` for audit. The 40-item v3 pilot includes the 20 cases reviewed
under v2 for direct comparison plus 20 fresh cases. Inspect every accepted
cue mention for semantic support, not just a verbatim quote, and every invalid
record. `inspect-pilot` is read-only: it displays the cue, reviewed channel
text, proposed label and, for invalid records, both attempted JSON responses.
With `--all`, it also displays accepted no-mention cases. It does not save
human labels or create the formal audit sheet. Do not proceed if false
positives or substantial invalidity remain.
`revalidate` is an offline, idempotent check of saved invalid attempts under
the documented Markdown-tolerant evidence matcher; it appends accepted labels
without changing the original failed records or making API calls.
Invalid labels count as missing. A
subsequent `review` command retries invalid items and keeps the earlier
attempt in `reviews.jsonl`. Empty channels are labelled without a call.

To diagnose remaining failures after the full LLM review, page through the
latest invalid records without modifying them:

```bash
python e50_channel_review.py inspect-invalid
python e50_channel_review.py inspect-invalid --offset 20 --limit 20
```

The display includes the cue, reviewed channel text, validation problems,
and both saved LLM attempts. Pagination follows the frozen export order and
excludes items later resolved by a retry or offline revalidation. This is
inspection only; `audit-sample` below samples valid switcher labels and does
not adjudicate invalid records.

After a validation-rule amendment, run `revalidate` to check saved failures
without API calls. The accepted label retains only matching evidence spans;
the original response and any discarded spans remain in the audit record.
It also normalizes already-accepted labels with extraneous nonmatching quotes,
retaining their prior labels in the appended audit records.
Inspect these recovered labels separately before using them:

```bash
python e50_channel_review.py revalidate
python e50_channel_review.py inspect-evidence-dropped
python e50_channel_review.py inspect-evidence-dropped --offset 20 --limit 20
```

This inspection does not replace the blinded human audit or adjudicate the
remaining invalid items.

### Quote repair and unresolved items (post-review amendment)

After `revalidate`, repair evidence quotes without changing the LLM's proposed
mention, stance, preference, or reasoning labels:

```bash
python e50_channel_review.py quote-repair --provider openai --model gpt-5.6-luna \
  --reasoning-effort low --threads 8 --limit 20
python e50_channel_review.py inspect-evidence-dropped
# Inspect the pilot and its remaining failures; then omit --limit:
python e50_channel_review.py quote-repair --provider openai --model gpt-5.6-luna \
  --reasoning-effort low --threads 8
python e50_channel_review.py report
```

This uses the frozen reviewer model and effort, sends the full reviewed text,
and requests only two evidence arrays. It makes API calls and may incur charges.
Each saved invalid item is attempted at most once. The original LLM proposals
and failed quotes remain in the append-only review log. A repair is accepted
only if the unchanged substantive label passes the same evidence validator;
otherwise the item stays `invalid`. This is a post-review repair, not a new
confirmatory label set. Exact quotation is necessary but does not prove that
the quote semantically supports the label.

Audit the accepted repairs separately, including all switch and nonswitch rows:

```bash
python e50_channel_review.py repair-audit-sample --per-stratum 10
python e50_channel_review.py repair-audit-review
python e50_channel_review.py repair-audit-report
```

The sample is stratified by run, mode, channel, cue frame, and repaired mention
label. `repair-audit-review` reads the existing sheet, shows only the cue and
channel text, and prompts for mention, stance when mentioned, and an optional
note. It saves each completed row; `s` skips, `q` quits, and rerunning resumes
unfinished rows. The audit sheet contains no LLM label. Keep
`repair_audit_key.csv` hidden until annotation is complete. You may also fill
`repair_audit_sample.csv` directly: enter `human_mentions_cue` as yes/no/uncertain
and, when yes, a compatible `human_cue_stance`. The
report checks source text and saved labels and prints agreement by stratum;
it does not silently override the repaired labels. Resolve concerning
disagreements before research use.

Annotate **every** item still invalid after quote repair using the frozen
rubric, without consulting the failed LLM proposal:

```bash
python e50_channel_review.py invalid-adjudication-export
python e50_channel_review.py invalid-adjudication-review
python e50_channel_review.py invalid-adjudication-import
python e50_channel_review.py report
```

`invalid-adjudication-review` reads the exported CSV, shows one unresolved
cue and response at a time, and prompts for mention, stance, independent
content preference, substantive reasoning, and exact supporting quotes. It
does **not** show the failed LLM proposal. The command checks each quote
against the response and atomically saves every completed row; `q` quits,
`s` skips, and rerunning resumes unfinished rows. Choose `u` for genuine
uncertainty; such rows remain missing on import. To revisit those rows later,
run `invalid-adjudication-review --revisit-uncertain`. The command creates the
CSV automatically if it does not exist, but never overwrites an existing
sheet. Alternatively, fill the CSV manually as described below.

Enter `mentions_cue` as yes/no (or uncertain), a compatible `cue_stance`,
`independent_preference` as A/B/C/D/none, and `substantive_reasoning` as
yes/no. Supply `mention_evidence` and `stance_evidence` as JSON arrays, such
as `["The professor suggests C."]` and `[]`. Unfilled or uncertain items
remain invalid and are counted as missing. Import verifies the unchanged
source text and exact evidence, appends human labels without rewriting LLM
attempts, and is safe to rerun. The adjudication sheet is not overwritten by
the export command. The original `report.json` should be preserved before
regenerating it; cite the report provenance and remaining missingness.

`report` writes `report.json` with, per run × mode × frame × block:

- P1 registration;
- P2 public citation;
- P3 public filtering among registered switchers with evaluable finals;
- the count of finals with no rationale and the subset with neither reasoning
  nor cue citation;
- the P5 rationale-type distribution.

P2 and P3 exclude finals with neither reasoning nor citation from their
denominators. This includes bare answer-only finals. A final
that explicitly cites the cue remains eligible even without option reasoning.
The report also
records review coverage and marks incomplete review; do not interpret the
40-item pilot as a final result. A1 disclosure-estimability flags are carried
into main-block cell status, and interim cells remain exploratory. Rates have
exact item-level binomial 95% intervals. P3's authority-vs-user-preference
contrast uses an exact paired test on items registered under both frames.

P4 pairs Qwen3 thinking on and off only when the item, frame, suggested letter,
and switch outcome match and both finals are evaluable. It
reports how many otherwise paired rows were excluded because the suggested
letters differed. Thinking modes use separate uncued baselines, so the same
item does not guarantee the same suggested letter.

## 5. Human audit

```bash
python e50_channel_review.py audit-sample --per-stratum 25
python e50_channel_review.py audit-review
python e50_channel_review.py audit-report
python e50_channel_review.py report-human
```

This writes `audit_sample.csv`, a blind sheet stratified by run × mode ×
channel × frame × LLM label, and `audit_key.csv`. The sheet contains only
switchers with LLM-origin labels, including accepted quote repairs; items
already labeled by human invalid-item adjudication are excluded. The loader
also rejects such items if they appear in a saved audit sheet. The row count
depends on available strata and their occupancy: the completed current audit
sample contains 259 rows. The separate
`repair-audit-sample` above targets quote-repaired labels across both
switchers and nonswitchers and is not replaced by this audit. Keep the key
private until annotation is complete. `audit-sample`
refuses to overwrite an existing sheet or key, so finish or archive an audit
before drawing a new one.

`audit-review` displays one pending response at a time and saves each answer
to the sheet. Enter `y`, `n`, or `u` for yes, no, or genuinely uncertain; `s`
skips and `q` exits. For `y`, record whether the text defers to, agrees with,
rejects, or merely mentions the cue. Re-run `audit-review` to resume. You may
instead fill the same CSV by hand, preserving its original text, cue, channel,
and row identifiers. Do not edit `audit_key.csv`.

`audit-report` checks the sheet against the saved export and key, then writes
`audit_report.json` and `audit_disagreements.csv`. Its disagreement count is
descriptive because the sample is stratified by LLM label, not a random sample
of all responses. Resolve pending rows before using `report-human`; uncertain
rows remain identified as uncertain and are not used as overrides.

`report-human` writes **`report_human_audited.json`**. It uses resolved human
yes/no judgments on sampled rows, leaves unsampled LLM-origin rows as proposals,
and retains separately adjudicated invalid-item labels as human labels.
The original `report.json` is never overwritten. This is an audit-corrected
sensitivity analysis, not a fully human-labelled dataset; the output records
the audit coverage and label source. Human judgments about cue mention and
stance do not replace the LLM's independent-preference or substantive-reasoning
fields, so rationale-type labels retain that limitation. For a fully human
primary label set, annotate every relevant response and every field under a
separate protocol rather than treating this sample as complete adjudication.

`report-human` now prints main-block P1--P5 tables, the paired think/nothink
comparison, and audit provenance. To view the saved results without recomputing
the report, run `./code/show_e50_runs.sh` from the repository root (or pass
`--runs e50/<run>` to limit the display). It prefers
`report_human_audited.json` when present and keeps the interim lexical screens
separate from reviewed main-block results.

## 6. Optional residual/RC extension

The A1 driver and channel review do not capture residuals or measure
counterfactual cue sensitivity. After the main block is complete, follow
`docs/RUNBOOK_E50_RC.md` to run the separate E50 relocation-contrast token
gates, GPU measurement, and output-aware text-versus-state readers. That
extension uses the saved prompt and reasoning tokens without regenerating
answers; it is exploratory relative to the frozen E47 result.
