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

## E50 reasoning-channel extension

E50 is a separate follow-up, not an additional E47-v10 cell. It reuses frozen
E47 source items and the cue-assignment rule in new, model-specific designs,
then samples responses with vLLM. Assignments are frozen from each E50 run's
own uncued responses. Thinking and final-answer text are stored
separately so cue acknowledgment can be assessed in each channel. Where the
model supports both modes, thinking-on and thinking-off runs can be compared.
The first 40 items are an interim feasibility block; the remaining 200-item
main block runs only after the frozen gate passes. Natural answers, forced
reads, caps, and uncued repeat consistency are reported separately. A lexical
mention screen is not a reviewed acknowledgment label.

The E50 Amendment A1 workflow records the interim decision before main-block
generation, then freezes a channel-aware review rubric, exports responses,
reviews them, and audits the labels. Its endpoints distinguish cue mention in
the thinking channel from citation in the final response. See
[`docs/RUNBOOK_E50_A1_and_channel_review.md`](docs/RUNBOOK_E50_A1_and_channel_review.md)
and the [A1 protocol](docs/AMENDMENT_E50_A1.md). Existing frozen runs must use
their original runner and code hashes; the v5 runner fixes a DeepSeek-Llama
output-decoding issue for **new** designs, not earlier frozen runs.

**E50 RC** is an extension after an E50 main block is
complete. It holds the saved generated token prefix fixed, relocates only the
suggested cue letter, and measures the change in the evaluated model's
pre-answer option log odds. It then compares held-out readers using text plus
the original output distribution against readers also given the original
residual state. It reuses E50 generations but requires additional token gates,
GPU forward passes, and reader fits; E50 generation and channel review alone
do not produce RC results. This is a fixed-prefix counterfactual diagnostic,
not a causal-mediation estimate. See the [E50 RC runbook](docs/RUNBOOK_E50_RC.md).

## Repository map

| Location | Purpose |
| --- | --- |
| `main_journal_v3.tex` | Current manuscript source |
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
| `code/e50_reasoning_run.py`, `code/e50_reasoning_run_v4.py`, `code/e50_reasoning_run_v5.py` | Version-specific E50 generation runners; use the runner frozen with each design |
| `code/e50_amendment_a1.py`, `code/e50_channel_review.py` | E50 main-block amendment and channel-aware review |
| `code/e50_rc.py` | E50 RC token gates, measurement, readers, and report |
| `docs/RUNBOOK_E50_reasoning_run_v3.md`, `docs/RUNBOOK_E50_reasoning_run_v4.md`, `docs/RUNBOOK_E50_reasoning_run_v5.md` | Generation instructions by runner version |
| `docs/RUNBOOK_E50_A1_and_channel_review.md`, `docs/RUNBOOK_E50_RC.md` | E50 review and RC workflows |

The manuscript and `docs/PROMPTS.md` are the starting points for understanding
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
E50 generation additionally uses vLLM (install it in the GPU environment and
follow the model-specific runbook); its channel review uses SciPy. E50 RC uses
Hugging Face Transformers for fixed-prefix forward passes, even though the
saved answers were generated with vLLM. These engines are not assumed to
produce bitwise-identical answer distributions.

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

For E50, start with the version-specific generation runbook above. The
following illustrates the sequence for a **new** v5 design, assuming the E47
source run is available; it is not a command to resume an older design:

```bash
python code/e50_reasoning_run_v5.py freeze --out e50/NEW_RUN \
  --source-run e47/SOURCE_RUN --model deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
  --revision auto
python code/e50_reasoning_run_v5.py inspect --out e50/NEW_RUN
python code/e50_reasoning_run_v5.py run --out e50/NEW_RUN --stop-after-interim
python code/e50_reasoning_run_v5.py report --out e50/NEW_RUN
```

Continue only on a GO decision, following the A1 runbook for the applicable
design. To inspect saved E50 results without regenerating responses, run
`./code/show_e50_runs.sh --runs e50/RUN_NAME` or
`./code/show_e50_trace.sh --run e50/RUN_NAME --list`.

After an eligible E50 main block, the separate RC sequence is:

```bash
python code/e50_rc.py freeze --run e50/RUN_NAME --mode think
python code/e50_rc.py tokens --run e50/RUN_NAME --mode think
python code/e50_rc.py measure --run e50/RUN_NAME --mode think --limit 5
python code/e50_rc.py measure --run e50/RUN_NAME --mode think
python code/e50_rc.py readers --run e50/RUN_NAME --mode think
python code/e50_rc.py report --run e50/RUN_NAME --mode think
./code/show_e50_rc.sh e50/RUN_NAME
```

Inspect token-gate coverage before `measure`; the five-row measurement is a
runtime smoke test. Use `--mode nothink` only for a run that froze that mode.
The [RC runbook](docs/RUNBOOK_E50_RC.md) covers exclusions, output directories,
reader requirements, and recovery of missing arrays.

## Data and provenance

Large model outputs, residual captures, and derived reader results are not
assumed to be part of this source repository. The local `.gitignore` excludes
`e47/`, `e50/`, pickles, generated figure PDFs, and several report directories by
default. Reproducing a published numerical result requires the corresponding
frozen `design.json`,
trace/capture artifacts, reviewed labels, and analysis outputs; code alone is
not enough. Compiling the manuscript also requires its generated figure assets.
Any public data release should preserve the run's design ID,
checkpoint revision, exclusion rules, and provenance files. Do not upload API
keys or private review worksheets.

## Citation and license

Please cite the accompanying paper when using this work. This checkout does
not yet include a `LICENSE` file; reuse permissions are not granted by the
README itself. Add an explicit license before publishing the repository if
reuse is intended.
