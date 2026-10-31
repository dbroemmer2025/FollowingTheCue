# Runbook: Fair-Reader Comparison on the Direct-Effect Targets

**Code:** `code/e47_de_readers.py` (3 selftest groups; end-to-end tested locally with a fake LLM)
**Preregistration:** `reports/de_readers/PREREGISTRATION.json`, frozen before any LLM call. The LLM model is recorded on the first `llm` run.
**Outputs:** `reports/de_readers/{targets/, llm/, results/, report.json}`

## Question

Can residual state predict how much the cue line matters, with the rationale held fixed, **beyond** what the strongest text-side reader infers from the same prompt and rationale?

## Targets

Rows are the 4,438 direct-effect rows that pass every gate.

| | target | type | population | cells |
|---|---|---|---|---|
| primary | `dP` = P(cue \| cue line) − P(cue \| filler), same rationale | continuous | all rows | 10 |
| primary | `RC` relocation contrast (token-validated) | continuous | all rows | 10 |
| primary | `decisive` | binary | switchers | 4 that pass occupancy (≥ 20 positive and ≥ 20 negative items): OLMo ×2, Qwen2.5-7B base ×2 |
| secondary | `D` (log-odds) | continuous | all rows | 10 |
| secondary | `reloc_swap` | binary | switchers | the same 4 |

The other six cells have only 3–16 decisive items, so they are skipped and reported as such. They are not forced through a model.

## Reader families

Each family is a candidate grid with nested, item-grouped selection; the design's seed and 5 folds are used throughout.

| family | inputs |
|---|---|
| `heuristic` | rationale-endorsement features (endorses cue / other / none; endorsement rule) |
| `llm` | features from a **new preregistered prompt**. The LLM gets the prompt and the rationale cut before the answer letter, and returns the answer distribution under the original, neutral (filler) and relocated cue lines, with the response held fixed. The six label-free features include predicted dP, predicted relocation contrast and predicted argmax changes. |
| `ngram` | TF-IDF on prompt + rationale |
| `text_plus` | ngram + heuristic + LLM features, plus the E6-style Transformer rung in a final GPU run |
| `state` | residual at the frozen layer (Colab, shards) |
| `joint_plus` | text_plus + state (nests text_plus) |

**Key quantity:** Δ = L(text_plus) − L(joint_plus). This is in squared error for continuous targets and bits for binary ones, with an item-bootstrap 95% CI. The final verdict requires `--transformer`: the text-only and joint candidate grids then both include the same E6-style Transformer branch, while the joint grid adds its state-augmented counterpart.

**Decision rule** (per cell × target): *state adds beyond text* if the CI lower bound > 0 **and** Δ ≥ 10% of the joint_plus gain over baseline. Otherwise, *not shown*. Verdicts are reported per cell, with no pooled claim beyond counts.

## Commands

### Local (done)

```bash
cd code
python e47_de_readers.py --selftest
python e47_de_readers.py prereg      # written 2026-10-24
python e47_de_readers.py targets     # reports/de_readers/targets/*.csv + occupancy.json
```

### Local: LLM reader

Each call returns three distributions. I checked one real message by hand (OLMo temporal row `031f…`). The provider, model, reasoning effort, prompt template, and response schema are frozen when the first LLM call begins; later calls reject a mismatch. Pilot first:

```bash
python e47_de_readers.py llm --provider openai --model gpt-5.6-luna --reasoning-effort low --threads 8 --limit 20
```

`--limit 20` means **20 calls total across all cells**, not 20 per cell. Structured completions default to a bounded 512 tokens; increase `--max-tokens` only after inspecting a genuine truncation. Check `reports/de_readers/llm/*.jsonl` (all `ok`; sensible distributions), then run the full set. That's about 4,400 calls, more output tokens per call than A1, and it resumes:

```bash
python e47_de_readers.py llm --provider openai --model gpt-5.6-luna --reasoning-effort low --threads 8
```

If a reader run reports one or more missing LLM rows, repair only that cell.
Existing `ok` records are reused, so this normally makes calls only for the
missing IDs:

```bash
python e47_de_readers.py llm --provider openai --model gpt-5.6-luna \
  --reasoning-effort low --threads 8 \
  --cells qwen2_7b_it_temporal_sequences
```

`--cells` takes the canonical cell name shown by a reader error, not the E47
run-directory name (for example, omit `_thinking_v10`).

Optionally, run the linear text-side families locally first. This is a smoke test only: it cannot issue the final reader verdict because it omits the Transformer rung.

```bash
python e47_de_readers.py readers --run ../e47/<cell> --no-state
```

### Colab: state and joint families

Copy up:

- `code/e47_de_readers.py`. It imports `e47_readers`, `e46_readers`, `e47_prequential`, `e47_direct_effect`, `e47_text_baselines`, `show_e47_gap_summary`, `show_e47_text_baselines` and `run_dirs`, which are already there.
- `reports/de_readers/PREREGISTRATION.json`
- `reports/de_readers/targets/`
- `reports/de_readers/llm/`
- the shards (already on Drive).

Run the primary targets first. Use `--transformer` for the confirmatory reader comparison; omitting it writes a clearly labeled linear-only diagnostic.

```bash
cd code
for cell in qwen2_5_7b_movie_recommendations_thinking_v10 qwen2_7b_it_movie_recommendations_thinking_v10 \
  gemma_2_9b_it_movie_recommendations_thinking_v10 mistral_nemo_instruct_2407_movie_recommendations_thinking_v10 \
  olmo_2_1124_7b_instruct_movie_recommendations_thinking_v10 qwen2_5_7b_temporal_sequences_thinking_v10 \
  qwen2_7b_it_temporal_sequences_thinking_v10 gemma_2_9b_it_temporal_sequences_thinking_v10 \
  mistral_nemo_instruct_2407_temporal_sequences_thinking_v10 olmo_2_1124_7b_instruct_temporal_sequences_thinking_v10
do
  python e47_de_readers.py readers --run ../e47/$cell --targets dP RC decisive --transformer
done
```

Then copy back `reports/de_readers/results/*.json` and, locally:

```bash
python code/e47_de_readers.py report
```

The report prints, per target, each family's out-of-fold R² (continuous) or AUC (binary), plus Δ, its CI and the verdict. It also writes `results/<cell>.oof.csv`, containing each held-out item, its fold, target, baseline prediction/loss, and each reader family's prediction/loss. The full (state) run takes precedence over a `--no-state` file for the same cell.

## Runtime

The linear grid has 21 joint candidates. With `--transformer`, the text grid adds the prespecified Transformer text candidate and the joint grid adds its state-augmented counterpart; nested fitting makes this GPU work, not a practical CPU job. Run one cell first and record wall time before dispatching the remaining cells. The linear-only diagnostic may be useful for debugging but does not meet the final reader-adequacy contract.

## Note on a latent bug found during testing

`e47_readers.nested_oof_regression` with `strata=None` fits the outer model on raw y and then adds the training mean again, shifting every out-of-fold prediction by mean(y). The inner model selection is unaffected.

**All existing callers pass `strata`** (the experimental condition), so no published or previously reported result is affected. `e47_de_readers.py` passes a constant stratum to avoid the bug, and a selftest checks that a +5 shift of the target leaves the loss unchanged.

The frozen file is not edited. Consider fixing it, with a code-drift note, before any future caller uses `strata=None`.


## Post hoc output-access control (Amendment 1, 2026-10-24)

The direct-effect targets share a term with the model's own factual next-token distribution, which a residual probe at the pre-answer position reads almost directly. This run lets both sides see that distribution. On Colab, using the same files as before plus `reports/direct_effect/<cell>.jsonl`, run per cell:

```bash
python e47_de_readers.py readers --run ../e47/$cell --transformer --targets dP RC decisive \
       --families state output text_out joint_out
```

This writes `results/<cell>__output.json` and does not touch the preregistered `<cell>.json`. In this extension, `text_out` and `joint_out` include distinct output-aware Transformer candidates: both receive the same fold-standardized heuristic, LLM, and factual-output scalar features; only the joint candidate additionally receives residual state. Copy the files back and run `python e47_de_readers.py report`; the report adds an OUTPUT-ACCESS CONTROL table with Δ_out, its verdict, and a `Transformer+output` check. A positive Δ_out is evidence of bounded-reader information about local cue-line sensitivity beyond the text and factual output distribution; it is not a mediation or necessity result.

Runtime is about the same as the original run: `joint_out` dominates, as `joint_plus` did.


## Post hoc no-LLM ablation (Amendment 2, 2026-10-30)

Question: does the RC residual increment beyond text + output depend on the
GPT-5.6-Luna features? Arm A refits the output-access comparison with the LLM
features removed from **both** sides (`text_out_nollm`, `joint_out_nollm`;
identical candidate grids, including the output-aware Transformer rung, and
auxiliary features = heuristic + factual output). Arm B is the existing
Amendment 1 run. No generation, capture, replay, or API call is needed. The
LLM cache is still read for row validation, but its features never enter the
new families (a selftest checks that scrambling them leaves the fits unchanged).

Amendment 2 in `reports/de_readers/AMENDMENTS.md` fixes the rule and the
prespecified reading before the run.

### Colab

Copy up `code/e47_de_readers.py` (only this file changed). Everything else
the Amendment 1 run used is already on Drive. Then:

```bash
cd code
python e47_de_readers.py --selftest
CELLS=(
  qwen2_5_7b_movie_recommendations_thinking_v10 qwen2_7b_it_movie_recommendations_thinking_v10
  gemma_2_9b_it_movie_recommendations_thinking_v10 mistral_nemo_instruct_2407_movie_recommendations_thinking_v10
  olmo_2_1124_7b_instruct_movie_recommendations_thinking_v10 qwen2_5_7b_temporal_sequences_thinking_v10
  qwen2_7b_it_temporal_sequences_thinking_v10 gemma_2_9b_it_temporal_sequences_thinking_v10
  mistral_nemo_instruct_2407_temporal_sequences_thinking_v10 olmo_2_1124_7b_instruct_temporal_sequences_thinking_v10
)
status=0
for cell in "${CELLS[@]}"; do
  python e47_de_readers.py readers --run "../e47/$cell" --transformer \
    --targets RC dP decisive --families text_out_nollm joint_out_nollm \
    || { status=$?; echo "ablation FAILED for $cell"; break; }
done
test "$status" -eq 0
```

To save GPU time, run `--targets RC` first; RC is the primary target. Runtime
is roughly that of the Amendment 1 run, without the `state` and `output`
families.

### Local

Copy back `reports/de_readers/results/*__nollm.json` and `*__nollm.oof.csv`,
then:

```bash
python code/e47_de_readers.py report
```

`report` prints the existing tables unchanged and then a NO-LLM ABLATION
table per target, with the following columns:

- out-of-fold R² for both new families;
- Arm A Δ, its CI and the pass rule;
- Arm B Δ and pass;
- the paired difference A − B;
- the LLM contribution to the text side;
- the LLM contribution to the joint side (positive means lower loss with Luna);
- the descriptive class: passes_both / passes_with_llm_only /
  passes_without_llm_only / passes_neither.

Keep Amendment 1 as the existing primary comparison regardless of ablation
results. A pass/fail difference alone does not establish dependence on Luna:
interpret the paired A-minus-B interval alongside Luna's loss reduction for
both text and joint readers. A-minus-B equals Luna's help to text minus its
help to joint. Intervals concern fixed OOF predictions, not repeated training
or model-selection uncertainty.

An RC-only run can be followed by a run for `dP decisive`: compatible results
are merged, retaining RC. Re-running RC replaces only that target. Merging
refuses mismatched code/provenance; archive the earlier artifacts explicitly
before replacing an incompatible run. Download both JSON and CSV together:
the reporter prefers the adjacent CSV over the saved absolute Colab path.

It writes `reports/de_readers/nollm_ablation.json`. It refuses to run if:

- the arms differ in rows, source-item IDs, targets, baseline losses, original-output
  file hashes, or outer folds; duplicate IDs, split source items and nonfinite
  numeric values are also rejected;
- arm B does not reproduce the reported Amendment 1 deltas and verdicts exactly (checked locally: 24 of 24 cell-targets reproduce, RC 9/10);
- the ablation was run without `--transformer`.
