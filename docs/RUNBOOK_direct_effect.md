# Runbook: Direct Effect of the Cue Beyond the Rationale

**Code:** `code/e47_direct_effect.py` (3 selftest groups pass; end-to-end tested on a tiny model)
**Preregistration:** `reports/direct_effect/PREREGISTRATION.json`, frozen 2026-10-23T18:36Z, before any measurement
**Outputs:** `reports/direct_effect/`

## What it measures

This is a **prompt–trace counterfactual diagnostic**: forward passes only, no new generation.

For each of the 4,454 Table II rows (disagreement trials, `initial/pre_answer`):

1. **Fix a written rationale prefix r.** These are the saved token ids up to the pre-answer boundary, i.e. just before the answer letter.
2. **Vary only the cue line in the prompt.**
3. **Read the model's next-token probability for each option letter.** These are full-vocabulary probabilities, using E47's context-preserving single-token check.

| prompt X | cue line becomes |
|---|---|
| `cue` | unchanged (the saved prompt ids) |
| `filler` (**primary** neutralization) | *There are four answer options listed above.* |
| `removed` | deleted. Identical to the saved uncued prompt; verified for all 4,598 disagreement prompts |
| `source` | same source, no suggestion (source placebo) |
| `relocated` | same cue sentence, pointing to another wrong option y |

| rationale r | |
|---|---|
| `cued` | the cued response's own prefix |
| `uncued` | the uncued response's prefix for the same item |

That gives 5 × 2 = 10 forward passes per row.

**Primary quantity:**

  D = logit P(cue | X_cue, r_cued) − logit P(cue | X_filler, r_cued)   (nats)

In words: given exactly the same written rationale, does keeping rather than neutralizing the cue line still change the model's imminent preference for the cue option?

**Secondary quantities:**

- **D_removed and D_source:** D computed with the other two neutralizations.
- **Relocation shift:** does the answer follow the cue's *letter* when the cue points elsewhere, with the rationale fixed?
- **Rationale carry:** cued vs uncued rationale under the same prompt (the prompt–trace swap).
- **D under the uncued rationale.**
- **Flip rate:** how often the 4-way argmax changes between the cue and filler prompts.
- **Log-odds of the cue option vs the uncued answer**, for each condition.

**Groupings:** every cell is broken down by cue family, by switch / no switch, and by reviewed initial acknowledgment. The key group is `switch_silent`: rows where the model switched to the cue and did not acknowledge it.

**Gates.** Failing rows are excluded and counted, never kept silently:

- the cued prompt must re-tokenize to the saved ids;
- the (cue, cued) cell must reproduce the model's actual answer;
- every option letter must be a context-preserving single token.

**Scope sentence for the paper.** This is a controlled counterfactual dependence test on one manipulated factor. It does not decompose the model's full reasons, and it is not causal mediation: a cue-conditioned rationale under a neutralized prompt is a cross-world, possibly off-distribution input.

## Commands

### 0. Local checks (already passing)

```bash
cd code
python e47_direct_effect.py --selftest   # 3 groups
python e47_direct_effect.py plan         # 4,454 rows; cue line once per row; removed == uncued prompt
```

### 1. Copy to Colab

Use the same Drive layout as before (repo root = the folder holding `code/` and `e47/`).

- **Code:** `code/e47_direct_effect.py`. It imports `e47_capture.py`, `e47_design.py`, `e46_prompts_cue.py`, `e46_pretest_cue.py` and `run_dirs.py`, which are already there.
- **Per cell:** `e47/<cell>/design.json`, `analysis.json`, `oof_losses.csv`, `traces.pkl`. Shards are **not** needed.
- **Preregistration:** `reports/direct_effect/PREREGISTRATION.json`. If it's missing, the first `measure` call writes it. If it's present and differs from the code, the run refuses to start.

### 2. Pilot one cell (GPU)

```bash
cd code
python e47_direct_effect.py measure --run ../e47/gemma_2_9b_it_movie_recommendations_thinking_v10 --limit 20
python e47_direct_effect.py analyze
```

Check the `kept` column and any `gate failures` line. `answer_reproduced` failures should be rare: the cued prompt + cued rationale is exactly what the model generated, so the argmax should be its answer. If more than a few percent fail, stop and send me the output before running the rest.

### 3. All ten cells (resumable: rerunning skips finished rows)

```bash
for cell in \
  qwen2_5_7b_movie_recommendations_thinking_v10 \
  qwen2_7b_it_movie_recommendations_thinking_v10 \
  gemma_2_9b_it_movie_recommendations_thinking_v10 \
  mistral_nemo_instruct_2407_movie_recommendations_thinking_v10 \
  olmo_2_1124_7b_instruct_movie_recommendations_thinking_v10 \
  qwen2_5_7b_temporal_sequences_thinking_v10 \
  qwen2_7b_it_temporal_sequences_thinking_v10 \
  gemma_2_9b_it_temporal_sequences_thinking_v10 \
  mistral_nemo_instruct_2407_temporal_sequences_thinking_v10 \
  olmo_2_1124_7b_instruct_temporal_sequences_thinking_v10
do
  python e47_direct_effect.py measure --run ../e47/$cell
done
```

### 3b. Token gate for the relocation contrast (Colab; tokenizer only, seconds per cell)

The filler and source-placebo prompts are **not token-position matched**: they are shorter than the cue sentence, so the fixed rationale starts at a different absolute position. The relocation arm changes only the letter. This stage verifies that per row (equal length, exactly one differing token that encodes the letter) and records the filler/source position offsets.

```bash
for cell in <the ten cells as above>; do
  python e47_direct_effect.py tokens --run ../e47/$cell
done
```

Copy back `reports/direct_effect/<cell>.tokens.json` (ten files). `analyze` then restricts the relocation contrast to gate-passing rows; without the files it omits the contrast rather than treating it as validated. The summary viewer automatically regenerates when a token-gate file is newer than its summary.

### 4. Copy back and analyze (locally or on Colab)

Copy back `reports/direct_effect/<cell>.jsonl` (ten files). Then:

```bash
python code/e47_direct_effect.py analyze
```

- Acknowledgment groupings read each cell's `review_approved.csv`. It's local; on Colab, copy it up too, or skip those groups.
- Writes `direct_effect_summary.json` and `direct_effect_rows.csv` (one row per kept item, for later reader comparisons).

## Runtime

There are 10 full forward passes per row, with no KV-cache reuse because the prompt changes. That's about 44,500 passes across the ten cells, each over roughly 600–1,200 tokens. Expect about **10–25 min per cell on an A100** (2–3× longer on L4/T4) plus model loading, or about 2–4 h in total. Calibrate from the pilot's `rows/s` line: total ≈ 4,454 ÷ rate.

## How to read the results

| pattern (cued rationale fixed) | meaning |
|---|---|
| D ≈ 0 or decisive = 0 for a switcher | **Cue-consistent under the filler counterfactual**: with the cue line replaced, the frozen cued rationale still yields the cue option. This does **not** show that the cue acted through the rationale, that the rationale mediated the effect, or that the model rewrote its reasoning because of the cue. |
| D > 0 (CI excludes 0) | The cue line changes the model's next-option preference with the rationale fixed. D is a broad local sensitivity and is large for non-switchers too, so it is not specific to the mechanism behind an observed switch. Use the decisive rate as the behavioral endpoint. |
| large relocation contrast / reloc swap | The answer tracks the cue's *letter* when only the letter changes and the reasoning is fixed. This is the template-matched robustness result (Amendment 2); quote it only for rows that pass the token gate. |
| D ≈ D_source | The suggested option drives the effect, not the mention of a source alone: removing only the suggestion (source placebo) removes about as much as removing the whole line. |
| family difference | Report only the cell-stratified, source-item paired bootstrap (`family_paired`). Pooled rates are descriptive. |

Look first at **OLMo temporal**. The error viewer showed rationales that conclude one thing while the model writes another, so expect its largest D.

## After this

As the feedback recommends, D can become a secondary target for the same readers: heuristic, LLM (with a new preregistered prompt), state and joint. Hold off until the measurements are in; `direct_effect_rows.csv` is the input for that step.
