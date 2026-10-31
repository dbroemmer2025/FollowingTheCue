# Runbook: echo-aware disclosure reporting

Adds the exact cue-line echo share to the reviewed acknowledgment rates, so the
primary figures can show how much of each acknowledgment bar is verbatim
copying rather than disclosure.

Nothing here regenerates model outputs. It re-derives one analysis JSON per
cell from responses and approved reviews that already exist, then rebuilds the
figures from it.

## Why this is needed

`e47_design.cue_echo_answer` identifies responses whose whole generated episode
is a copy of a cue-bearing line of their own prompt. The broad acknowledgment
endpoint scores such a response as acknowledging the cue. Copying the prompt is
not a report of influence, so each rate has an echo-inclusive and an echo-free
reading.

All 51 echoes in the study are in **Qwen2.5-7B movie recommendation**, where
they are 38 of the 41 initial acknowledgments among switchers. The figure for
that cell currently reports 12.3% initial acknowledgment where the echo-free
rate is 0.9%.

## Precondition

The per-sample capture shards must be present. `what-causes-the-switch` reads
the uncued option distribution from them:

```bash
ls e47/<run>/capture/*.pkl | wc -l      # must be non-zero
```

If this is zero the command fails with `No such file or directory:
.../capture/<id>.pkl`. Restore the shards for the run before continuing.

`--allow-code-drift` is **not** required. `what-causes-the-switch` is already
in the `check_code=False` list in `e47_experiment.main`, so reporting-only code
changes do not block it.

## Step 1 — re-run the analysis for all ten cells

From `code/`:

```bash
cd code
for R in \
  qwen2_5_7b_movie_recommendations_thinking_v10 \
  qwen2_5_7b_temporal_sequences_thinking_v10 \
  qwen2_7b_it_movie_recommendations_thinking_v10 \
  qwen2_7b_it_temporal_sequences_thinking_v10 \
  gemma_2_9b_it_movie_recommendations_thinking_v10 \
  gemma_2_9b_it_temporal_sequences_thinking_v10 \
  mistral_nemo_instruct_2407_movie_recommendations_thinking_v10 \
  mistral_nemo_instruct_2407_temporal_sequences_thinking_v10 \
  olmo_2_1124_7b_instruct_movie_recommendations_thinking_v10 \
  olmo_2_1124_7b_instruct_temporal_sequences_thinking_v10 ; do
    echo "== $R"
    python3 e47_experiment.py what-causes-the-switch \
      --run ../e47/$R \
      --annotations ../e47/$R/review_approved.csv || echo "FAILED: $R"
done
```

The output is a new content-addressed `what_causes_switch_<hash>.json` inside
each run directory. Existing files are **not** replaced, so this is additive
and reversible: delete the new files to revert.

## Step 2 — verify the new field is present

```bash
cd code
python3 - <<'PY'
import json, glob, sys
sys.path.insert(0, '.')
from run_dirs import DEFAULT_RUN_DIRS
from pathlib import Path
ok = True
for run in DEFAULT_RUN_DIRS:
    newest = max(Path(run).glob('what_causes_switch_*.json'),
                 key=lambda p: p.stat().st_mtime)
    d = json.load(open(newest))['disclosure_plan']['all_switchers_label_rates']
    e = d['initial_cue_acknowledgment']
    has = 'positive_cue_echo' in e
    ok &= has
    print(f"{Path(run).name[:46]:48} echo field: {has}  "
          f"ack={e.get('positive')} echo={e.get('positive_cue_echo')} "
          f"free={e.get('positive_echo_free')}")
print('\nALL CELLS UPDATED' if ok else '\nSOME CELLS MISSING THE FIELD')
PY
```

Expected: `positive_cue_echo` is 38 for `qwen2_5_7b_movie_recommendations` and
0 everywhere else.

## Step 3 — rebuild the figures

```bash
cd figures
python3 make_e47_figures.py
```

`causes_path()` selects the newest `what_causes_switch_*.json` by mtime, so no
path edits are needed. Panel A of `fig_e47_primary.pdf` and
`fig_e47_primary_ten_cell_average.pdf` will gain a white hatched overlay on the
echo portion of the acknowledgment bars, with an `exact cue-line echo` legend
entry. Cells with no echoes are unchanged.

If a cell's JSON lacks the field, its bar is drawn with no overlay rather than
an implied zero.

## Step 4 — cross-check against the independent count

`code/e47_echo_sensitivity.py` derives the same quantities from traces and
approved reviews without touching the analysis JSONs, so it is an independent
check on Step 1:

```bash
cd code
python3 e47_echo_sensitivity.py --selftest
python3 e47_echo_sensitivity.py --e47-root ../e47 --out ../reports
python3 show_authority_echo_metrics.py
```

Reference values to match:

| quantity | value |
|---|---|
| total exact echoes, all cells | 51 |
| cells containing echoes | Qwen2.5-7B movie only |
| Qwen movie, authority acks / echoes (all trials) | 58 / 51 (87.9%) |
| Qwen movie, disagreement acks / echoes | 42 / 38 (90.5%) |
| Qwen movie, initial ack among switchers | 41 of 334; echo-free 3 |
| Qwen movie panel A, inclusive vs echo-free | 12.3% vs 0.9% |
| ten-cell mean panel A, inclusive vs echo-free | 6.3% vs 5.2% |
| pooled authority initial rate, inclusive vs echo-free | 7.9% vs 6.8% |

The switcher denominators above come from an independent reimplementation of
the switch rule and may differ by one row from `report.json`; the acknowledgment
and echo counts do not depend on it.

## Step 5 — paper numbers that depend on this

If any value changes, these need updating:

- Table 4 row `Movie / Qwen2.5-7B`, initial authority 42/234.
- The echo decomposition paragraph in `sections/cue_acknowledgment_determinants.tex`.
- The residual-disclosure paragraph in the same file, which reports that an
  echo-free refit of the Qwen movie prompt-end AUC is infeasible.
- Figure captions in `main_journal_v2.tex` that quote panel A rates.

## Not covered here

Re-running the prompt-end disclosure AUC with echoes excluded
(`e47_mention_analysis.py captures --exclude-echoes`) needs the capture shards
*and* enough non-echo positives. Qwen2.5-7B movie has 4, against the frozen
stratum minimum of 20, so that refit cannot be fitted rather than merely being
weaker.
