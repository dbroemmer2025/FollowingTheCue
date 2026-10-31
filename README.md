# Following the Cue, Omitting the Influence

Research code for *Following the Cue, Omitting the Influence: Behavioral and
Internal Evidence from Language-Model Rationales*. The study asks whether a
model acknowledges an external suggestion that changes its answer, and whether
its pre-answer state contains information about sensitivity to that suggestion
that the evaluated text-side readers do not recover.

## Study at a glance

The E47-v10 experiment pairs an uncued response with four cued responses to
the same item. The cues cross two source frames (expert authority and user
preference) with whether the suggested answer agrees or conflicts with the
uncued answer. The released study evaluates five open-weight models on the
BIG-Bench Hard `movie_recommendation` and `temporal_sequences` tasks. It also
asks a neutral follow-up question about each answer.

The main analyses are:

1. **Behavior:** Does a conflicting cue change the selected option to the
   suggested one? A neutral-insertion control checks whether the effect is
   specific to an answer suggestion rather than just adding a sentence.
2. **Acknowledgment:** Does the initial response or the neutral *why*
   follow-up recognize the cue? Reported acknowledgment uses reviewed labels,
   not an automatic string match alone.
3. **Internal sensitivity:** With the generated rationale held fixed, how does
   the evaluated model's next-answer distribution respond when the cue's
   suggested letter is relocated? The relocation contrast (RC) is measured
   from each evaluated model, then predicted using text/output, residual-state,
   and combined readers.

These tests establish neither intentional concealment nor a complete account
of the model's internal causal mechanism. The reader result is relative to the
evaluated reader families; it does not show that no stronger text-only reader
could recover RC. An auxiliary LLM supplies *features* to one text-side reader;
it does not supply the evaluated models' next-token probabilities or the RC
target. See the manuscript for the estimands, controls, and limitations.

## Repository map

| Location | Purpose |
| --- | --- |
| `code/e47_experiment.py` | Freeze, preview, generate, capture, analyze, and report an E47 run |
| `code/e47_neutral_insertion.py` | Answer-free neutral control and uncued replay |
| `code/e47_direct_effect.py` | Fixed-rationale prompt replay and RC measurement |
| `code/e47_de_readers.py` | Held-out text/output/state reader comparisons for RC and related targets |
| `code/e46_llm_review.py` | LLM annotation proposals; see the review guide for the supported workflow |
| `docs/PROMPTS.md` | Readable prompt templates and frozen-design conventions |
| `docs/GUIDE_E47_REVIEW.md` | Export and adjudication of reviewed labels |
| `docs/RUNBOOK_direct_effect.md` | Counterfactual measurement protocol |
| `docs/RUNBOOK_de_readers.md` | Reader protocol, amendments, and analysis commands |
| `docs/RUNBOOK_E47_NEUTRAL_INSERTION.md` | Neutral control and matched-session replay |

The `docs/PROMPTS.md` file is the starting point for understanding
the study; the runbooks specify the operational steps.

## Environment

Use Python with a compatible PyTorch/Transformers installation. The general
dependency list is in `code/requirements.txt`:

```bash
python -m venv code/.venv
source code/.venv/bin/activate
python -m pip install -r code/requirements.txt
python code/e47_experiment.py --help
```

Model generation, activation capture, fixed-rationale forward passes, and
Transformer readers require a suitable GPU; reporting and many audits run on
CPU. The experimental designs pin model revisions and record runtime
provenance. Exact replication should use the frozen versions and seeds recorded
with each run, rather than assuming that the latest library or model revision
is equivalent. The optional LLM annotation and reader-feature stages require
their respective API credentials and may incur usage charges.

## Reproduction path

1. Read [`docs/PROMPTS.md`](docs/PROMPTS.md) and the manuscript to identify the
   prompt family, model checkpoints, task pools, and analysis populations.
2. Use `code/e47_experiment.py freeze` to create a **new** design, then
   `preview` to inspect it. Follow the version-specific instructions in
   [`docs/GUIDE_E47.md`](docs/GUIDE_E47.md); do not substitute a current BBH
   download for a released frozen item pool without verifying source IDs.
3. Run the frozen E47 stages and review-export workflow. Use
   [`docs/GUIDE_E47_REVIEW.md`](docs/GUIDE_E47_REVIEW.md) before treating proposed
   labels as adjudicated data.
4. Run the neutral-insertion control and fixed-rationale replay using their
   linked runbooks. Run the RC reader comparison only after its measured
   targets and required capture artifacts are available.
5. Inspect per-cell reports and provenance before pooling cells or making
   cross-model claims. Post hoc extensions are identified as such in their
   runbooks and manuscript.

For an existing run with its artifacts present, these commands inspect the
frozen design and the experiment CLI without regenerating responses:

```bash
python code/e47_experiment.py preview --run e47/RUN_NAME --example 0
python code/e47_experiment.py --help
```

`RUN_NAME` is a placeholder for an actual frozen run directory.

## Data and provenance

Large model outputs, residual captures, and derived reader results are not
assumed to be part of this source repository. This repository excludes
pickle files, generated figure PDFs, and several report directories.
Any public data release should preserve the run's design ID,
checkpoint revision, exclusion rules, and provenance files. Do not upload API
keys or private review worksheets.

## Citation and license

Please cite the accompanying paper when using this work.
The code uses an MIT licensed as noted in the `LICENSE` file.
