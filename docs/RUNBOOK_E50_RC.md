# E50 relocation contrast (RC v5)

This is a separate, exploratory extension to E50 Amendment A1. A1 generates
reasoning/final-answer channels and reviews cue acknowledgment. It does **not**
capture residuals or measure RC. This pipeline uses the saved E50 prompt and
generated token IDs; it does not regenerate answers or call an API.

## Target and boundary

For each natural, valid disagreement response with a natural uncued comparator,
move only the assigned cue letter to an option that is neither the cue nor the
uncued answer. Choose that alternative by a fixed hash of frozen item, mode,
sample, frame, and cue metadata, not by the completed response. Hold the saved
generated token IDs fixed up to (but excluding) the final answer letter. Score
the *same evaluated model* under the original and relocated prompt. RC is the
change in log odds of the relocated letter versus the original cue letter:

`RC = [log P(relocated) - log P(cue)]_relocated - [log P(relocated) - log P(cue)]_original`.

The prompt-end residual is **not** the readout. The residual and the model's
four-option output distribution are captured at the validated pre-answer
boundary under the original prompt and saved reasoning. The analysis compares
text plus original output distribution with text plus original output plus
residual state, using source-item-grouped folds. Both sides are explicitly
given the original cue and relocated-letter identities as query features;
otherwise the reader would not know which counterfactual RC target to
predict. The state-only fit is a diagnostic and does not define the primary
text-versus-joint comparison. No Luna features are used.
The LLM-assisted channel review remains a separate disclosure endpoint.
The saved answers were sampled with vLLM, while these fixed-prefix forward
passes use Hugging Face Transformers with the pinned weights and saved token
IDs. Token identity is checked, but a sampled answer need not be the top
option under the replay; the two engines are not asserted to be bitwise
identical. Report this as a replay diagnostic, not a reconstruction of the
original sampling draw.

This is a fixed-rationale prompt-trace counterfactual diagnostic, **not** a
causal-mediation estimate or proof that the model represented its cue reliance
in a disentangled feature. Compare RC across authority and user-preference
frames and report all eligible rows; switchers are a secondary subgroup.

## Colab sequence

Use the current `code/e50_rc.py` before freezing. An RC directory frozen with
older code cannot be resumed under this version; the E50 A1 generations do
**not** need regeneration. Copy `code/e50_rc.py` and its imports (`e50_reasoning_pilot.py`,
`e46_pretest_cue.py`, `e47_capture.py`, `e47_de_readers.py`, `e47_readers.py`,
`e46_readers.py`, `e47_design.py`, `p2_common.py`, `models.py`) to the Colab code
directory. Keep the run's frozen `design.json`, `progress.jsonl`, A1 amendment,
main assignment, and `main_status_<mode>.json` together in its E50 directory.
Version 5 accepts a terminal inline `Answer: (X)` as well as a line-start
answer, including when the decoded token stream ends with a special token.
It also checks a one-token cue edit against the saved prompt IDs, even if a
full prompt re-encode adds or removes chat/BOS tokens. It still requires the
saved answer letter to match the exact single-token boundary. The v3 parser incorrectly excluded rows whose decoded
token stream had a trailing special token; do not use v3 reader estimates.
To reanalyze an existing RC run under changed code, use a new output
directory (for example, `--out "$RUN/rc_think_main_v5"`) for **every** command
below. Do not overwrite or resume an older RC directory. The A1 generations can be
reused, but `freeze`, `tokens`, `measure`, `readers`, and `report` must be rerun.
Run each mode separately after its A1 main block is complete. Example:

```bash
RUN=/content/drive/MyDrive/verbalization_ceiling/e50/qwen3_8b_movie_v2
python code/e50_rc.py freeze --run "$RUN" --mode think
python code/e50_rc.py tokens --run "$RUN" --mode think
```

Inspect `rc_think_main/token_gates.json`. The gate requires: one-token cue
relocation with the same prompt token count; a single-token answer letter at
the saved final-answer boundary; context-preserving single-token encodings of
all four options; and an exact match between the saved answer token ID and
the original prompt's contextual option token ID. Failure counts are
exclusions, not grounds to relax the gate after seeing RC. Stop if coverage
is too low to support an interpretable comparison. Because generation is
sampled, the gate does **not** require the answer to be the greedy top option
on replay; that requirement would exclude valid sampled answers.

For Magistral, v4 rejected every row because re-encoding the whole decoded
chat prompt changed its token count. Version 5 finds a candidate token edit
using the loaded tokenizer but applies it to the **saved** prompt IDs, then
requires the edited IDs to decode to the exact relocated prompt. The
Transformers Mistral-regex warning is separate: do not change tokenizer flags
mid-analysis merely to silence it, because the saved vLLM token IDs must stay
the reference. Check the new token-gate counts before loading the 24B model:

Magistral's `Mistral3Config` is not accepted by `AutoModelForCausalLM`.
Copy the updated `code/e46_pretest_cue.py` to Colab before `measure`; its
loader selects `Mistral3ForConditionalGeneration` for this checkpoint. This
does not change `e50_rc.py` or its frozen code hash. If `tokens` reports
`{'ok': 398}`, keep the same RC output directory that passed the gate and run
`measure` there; do not refreeze or repeat the token gate. For example, if
the successful gate is in `rc_think_main`:

```bash
RUN=/content/drive/MyDrive/verbalization_ceiling/e50/magistral_small_2509_movie_v4
python code/e50_rc.py measure --run "$RUN" --mode think --out "$RUN/rc_think_main" --limit 5
```

The tokenizer-regex warning is not the model-loading failure. Use an 80 GB
GPU for this BF16 checkpoint.

```bash
RUN=/content/drive/MyDrive/verbalization_ceiling/e50/magistral_small_2509_movie_v4
OUT="$RUN/rc_think_main_v5"
python code/e50_rc.py freeze --run "$RUN" --mode think --out "$OUT"
python code/e50_rc.py tokens --run "$RUN" --mode think --out "$OUT"
```

If the gate still rejects most rows, stop and inspect the reason counts. Do
not run `measure` on an empty gate. If it passes, use `--out "$OUT"` on each
remaining `measure`, `readers`, and `report` command. No A1 generation or
review needs repeating.

On a CUDA GPU, smoke-test five rows, inspect `measure.jsonl`, then resume the
same command without a limit:

```bash
python code/e50_rc.py measure --run "$RUN" --mode think --limit 5
python code/e50_rc.py measure --run "$RUN" --mode think
python code/e50_rc.py readers --run "$RUN" --mode think --quick
python code/e50_rc.py readers --run "$RUN" --mode think
python code/e50_rc.py report --run "$RUN" --mode think
```

`measure` is resumable by row; do not change `progress.jsonl` or the RC code
after `freeze`. Complete all modes/main blocks first if they share a progress
file. The measurement records whether the sampled answer was also the top
option, but does not condition RC or reader eligibility on that diagnostic.
For thinking-off, replace `--mode think` with `--mode nothink` in every
command. An interim-only diagnostic uses `--block interim`, but it is not a
main-block result. Separate runs/models/modes require separate freezes and
outputs. The eligible main runs/modes are:

```text
e50/qwen3_8b_movie_v2                 think
e50/qwen3_8b_movie_v2                 nothink
e50/qwen3_8b_temporal_v3              think
e50/magistral_small_2509_movie_v4    think
```

Use the same `freeze`, `tokens`, `measure`, `readers`, `report` sequence above
for each line, changing `RUN` and `--mode`. Start with Qwen3-8B movie
thinking-on; its five-row `measure --limit 5` run is the runtime and token
coverage smoke test before committing to all four.

The default `readers` includes the order-aware Transformer and can be slow on
long reasoning traces. `--quick` is a diagnostic without that rung; it writes
separate files and must not be reported as the full-reader result. Full readers
require at least 30 valid source items. The model forward passes need a GPU;
the Transformer reader also benefits from one. `freeze`, `tokens`, and
`report` do not need a GPU. A100 80 GB is appropriate for 24B Magistral;
8B models may fit on smaller GPUs depending on sequence length. Runtime is
dominated by two full-prefix forwards per row and the nested Transformer fits;
benchmark the five-row smoke on the actual device before projecting total time.

### Missing reader arrays after measurement

If `readers` raises `FileNotFoundError` for a `states/*.npy` or
`sequences/*.npy` file after `measure` reached its full count, do not refreeze
or delete `measure.jsonl`. Copy `code/e50_rc_artifacts.py` alongside the
**unchanged frozen** `e50_rc.py`, then audit the same output directory:

```bash
python code/e50_rc_artifacts.py audit --run "$RUN" --mode think --out "$RUN/rc_think_main"
```

If `repair_ids` is nonzero, run `repair` on the GPU. It recreates only missing
arrays from the saved original prompt and token IDs, checking the frozen code,
answer boundary, layer, and saved original option probabilities. It does not
alter the measurement log or RC values:

```bash
python code/e50_rc_artifacts.py repair --run "$RUN" --mode think --out "$RUN/rc_think_main"
python code/e50_rc.py readers --run "$RUN" --mode think --out "$RUN/rc_think_main"
python code/e50_rc.py report --run "$RUN" --mode think --out "$RUN/rc_think_main"
```

If the audit reports unmeasured rows, resume `e50_rc.py measure` instead.
Check that `repair_ids` becomes zero before fitting readers. A large number
of missing files may indicate an incomplete Drive transfer or mount problem;
verify the actual Colab output directory before running the GPU repair.

## Outputs and interpretation

The `rc_<mode>_<block>` directory contains `config.json`,
`token_gates.json`, `measure.jsonl`, per-row residual and embedding projections,
`readers.json`, `readers_oof.csv`, and `report.json`. Verify the token gate,
saved-answer-token match, and sampled-answer-top-option diagnostic counts
before interpreting results. The reader
comparison uses mean squared error for continuous RC, not AUC; a positive
`delta_text_minus_joint` indicates lower held-out error when residual state is
added. `report.json` marks subgroups with fewer than 30 rows as descriptive
low-occupancy results; a single-row subgroup has no bootstrap interval. The
report's 10% share gate is a descriptive sensitivity criterion,
not a preregistered E47 confirmation claim. Do not treat inability to pass a
token gate as evidence of zero cue reliance.

To inspect saved results without loading the model, use:

```bash
./code/show_e50_rc.sh e50/qwen3_8b_movie_v2
./code/show_e50_rc.sh e50/qwen3_8b_movie_v2 --mode nothink
./code/show_e50_rc.sh e50/qwen3_8b_movie_v2 --all-rows
./code/show_e50_rc.sh e50/qwen3_8b_movie_v2 --id ROW_ID_OR_UNIQUE_PREFIX
```

Pass the `rc_<mode>_<block>` directory directly if it has a nonstandard name.
The viewer displays token coverage, RC by cue frame and original switch status,
replay argmax changes, held-out reader losses, per-letter slices, and optional
row-level detail. It reads saved artifacts only; it does not refit readers.
