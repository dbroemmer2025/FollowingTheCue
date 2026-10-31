# Amendments to the fair-reader preregistration

`PREREGISTRATION.json` is unchanged. The preregistered verdicts (Δ = L(text_plus) − L(joint_plus)) are reported exactly as run.

## Amendment 1 — 2026-10-24, after the results (post hoc; output-access control)

**Motivation.** Every direct-effect target contains a term from the factual run: the model's own next-token distribution with the cue line present, at the pre-answer position.

- dP contains P(cue | cue line).
- RC contains log P(y|X_x) − log P(x|X_x).
- Decisive requires the cue to be the factual argmax.

The residual state at the second-to-last block, at that same position, encodes those logits almost directly through the unembedding. A state advantage over a text-only reader can therefore reflect the probe reading the model's own output, not information about cue reliance that the text lacks.

**Addition** (no change to targets, folds, rule or existing results):

- `output`: features of the factual next-token distribution only, namely log P(x), log P(y), the max over the other options, log P(x) − log P(y), and log(1 − P(x)).
- `text_out` = text_plus + output; `joint_out` = joint_plus + output.  The
  post-hoc Transformer rungs are output-aware: both receive the identical
  fold-standardized heuristic, LLM, and factual-output scalar features; only
  the joint rung additionally receives residual state.
- Δ_out = L(text_out) − L(joint_out), judged by the same rule ("state adds beyond text+output").
- Run with `--families output text_out joint_out`; results go to `<cell>__output.json` and are merged by `report`.

**Interpretation.**

- The preregistered comparison answers: *does state beat a text-only overseer?*
- Δ_out answers: *does state carry information about cue-line dependence beyond the text AND the model's own output distribution at that position?*
- Only the second supports a bounded-readout claim about local cue-line
  sensitivity beyond the rationale and the factual output distribution.  It is
  still not a mediation estimate or evidence that this state is necessary for
  the answer.

## Amendment 2 — 2026-10-30, before any ablation run (post hoc; no-LLM ablation)

Written before the ablation was run. Reader code at writing:
`code/e47_de_readers.py` SHA-256 `f93b6226d08b984ba085738cc36a533458ff61cff854bdf9c447cb70abc26c69`.

Review revision (2026-10-30): interpretation rules below were corrected
to retain Amendment 1 as primary and avoid inferring an interaction from
pass/fail differences. Three of ten cell fits had already completed by the
time this revision was saved; it must not be described as fully pre-run.
Reporting now checks source grouping,
finite values, baselines and original-output file hashes; it resolves downloaded
OOF files locally and preserves earlier targets on compatible subsequent runs.
Reviewed reader SHA-256:
`d4f57a93dc9e6baaaedc4e1281c82ce97dcf5ace4eec42f2f5d975c463fefb03`.

**Motivation.** In Amendment 1 both reader families receive features derived
from GPT-5.6-Luna's verbal estimates of the evaluated model's answer
distributions (`llm_features`). The RC targets themselves come only from the
evaluated models' forward passes (recomputed for all 4,438 rows). The question
is whether the reported residual increment depends on including the LLM
features in the comparison.

**Addition** (no change to targets, rows, folds, seed, candidate grids, rule,
or any existing result):

- `text_out_nollm`: identical candidates to `text_out`, including the
  output-aware Transformer rung, with auxiliary features = heuristic +
  factual output only (no LLM features).
- `joint_out_nollm`: identical candidates to `joint_out`, same auxiliary
  features, plus residual state.
- Arm A: Δ_A = L(text_out_nollm) − L(joint_out_nollm). Arm B: the existing
  Amendment 1 Δ_out = L(text_out) − L(joint_out), read from the saved
  `<cell>__output.oof.csv` and required to reproduce the reported values.
- Per cell and target: Δ_A with its item-bootstrap CI and the same frozen rule
  (CI lower bound > 0 and Δ ≥ 10% of the arm's joint gain); the paired
  difference Δ_A − Δ_B; and the LLM contribution to the text side,
  L(text_out_nollm) − L(text_out). All intervals are item bootstraps of fixed
  out-of-fold losses (2,000 draws, seed 47). The report refuses to compare
  arms unless rows, targets and outer folds are identical.
- Targets: **RC is primary**; dP and decisive are reported descriptively
  only (they are not the primary robustness target here).
- Run with `--transformer --targets RC dP decisive --families text_out_nollm
  joint_out_nollm`; results go to `<cell>__nollm.json` / `.oof.csv`, and
  `report` writes `reports/de_readers/nollm_ablation.json`.

**Prespecified reading** (per cell, RC):

- Descriptive classes: *passes_both*, *passes_with_llm_only*,
  *passes_without_llm_only*, and *passes_neither*.
- Amendment 1 remains the existing primary result regardless of the ablation
  outcome. Arm A is a post hoc sensitivity analysis, not an outcome-dependent
  replacement for the primary result.
- A difference in passing status is not evidence that the increments differ.
  Interpret the paired interval for Delta_A minus Delta_B, and report reader
  selections when the outcomes differ. An interval spanning zero is unresolved,
  not evidence of equivalence.
- Report Luna's contribution to both text and joint prediction with paired
  intervals. Delta_A minus Delta_B equals its help to text minus its help to
  joint. Passing only without Luna does not by itself show that Luna helps
  the text side or closes the increment.
- These bootstrap intervals condition on the fitted OOF predictions and do
  not capture retraining or candidate-selection variability.
- Pass counts are ten unadjusted per-cell tests per arm; no pooled claim beyond
  counts.

**Reporting clarification (2026-10-31, after all ten ablation fits).** The
paper displays the original Luna-assisted output-aware comparison as primary,
consistent with Amendment 1. The Luna-free fit is a post hoc sensitivity
check, not a second preregistered primary analysis. Both arms pass in the same
nine cells; the paper reports their paired differences and the two cells where
Luna measurably improves the text reader. This clarification does not change
targets, folds, models, thresholds, or stored results.
