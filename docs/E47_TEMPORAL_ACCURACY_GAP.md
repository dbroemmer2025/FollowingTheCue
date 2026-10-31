# Temporal-sequences accuracy versus gap

The existing `e47_accuracy_gap.py` analysis is sufficient for the descriptive
question: do correct individual responses have larger or smaller pre-answer
gaps than incorrect responses? It joins BBH correctness to fixed grouped OOF
losses. Readers still predict switch-to-cue, not correctness or the correct
option. No GPU, API calls, generation, or decoder refitting is needed.

## Show and export

From the repository root:

```bash
python code/show_accuracy_gap.py --task temporal_sequences \
  --csv reports/e47_temporal_accuracy_gap.csv \
  --figure-prefix figures/e47_temporal_accuracy_gap \
  --latex-table tables/e47_temporal_accuracy_gap.tex
```

This uses the five temporal runs in `code/run_dirs.py`. Override them with
`--runs RUN_A RUN_B ...`. The existing `show_accuracy_gap.sh` wrapper forwards
all these flags. The script verifies recorded input hashes before reporting,
checks comparison compatibility, and produces PDF/PNG figures plus a provenance
JSON listing the exact source artifacts. The CSV includes losses, relative
gaps, switch rates, counts, and gap intervals by cue family.

The optional draft is `sections/temporal_accuracy_gap.tex`. It references the
generated table and PDF but is not included in the manuscript.

## Regenerate stale analyses

If traces, design, or primary analysis changed, regenerate each affected run:

```bash
python code/e47_accuracy_gap.py --run e47/RUN_NAME \
  --checkpoint initial/pre_answer --bootstrap 2000
```

Then rerun the show/export command. The reporter selects the newest matching
artifact and fails on stale inputs rather than silently accepting them.

## Interpretation

- Positive correct-minus-incorrect means correct responses have larger gaps.
- The current five models have mixed signs: Qwen base and Mistral-Nemo positive;
  Qwen-Instruct, Gemma and OLMo negative. Both outcome subsets have positive
  pooled gaps in all five models.
- Accuracy here covers only eligible disagreement responses. It is not uncued
  task accuracy. Counts are responses, but estimates weight source items equally
  within each subset. The same source can occur in both correctness subsets.
- The 95% intervals bootstrap source items while holding OOF readers fixed.
  They do not capture training variability and are not multiplicity-adjusted.
- Do not infer a causal effect of gap on accuracy. Correctness and cue following
  are coupled post-intervention outcomes. A useful further analysis would stratify
  jointly by cue correctness, uncued correctness, and switching, reporting sparse
  or empty cells explicitly. The current artifacts do not isolate those effects.
- Absolute and relative contrasts may differ because the text-loss denominator
  changes. Inspect channel losses and switch rates alongside the gap.

Avoid fitting a correlation line through five model averages: the more useful
evidence is the within-model correct-minus-incorrect contrast and its cue-family
sensitivity. Claims about internally knowing the correct answer require a
correct-option target, separate from this switch-target diagnostic.
