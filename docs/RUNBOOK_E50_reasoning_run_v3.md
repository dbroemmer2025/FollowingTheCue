# E50 v3 reasoning-model pilots

E50 v3 uses the same 240-item E47-v10 source design for each task, but freezes
new sampled generations and a model-specific chat-template profile. It does not
reuse E47 responses. The first 40 source items form a development/interim block;
the other 200 are generated only when the interim feasibility gate is GO.

The three supported models are:

| Model | Frozen modes | Prompt handling |
|---|---|---|
| `Qwen/Qwen3-8B` | `think`, optionally `nothink` | Qwen3 `enable_thinking` switch |
| `Qwen/Qwen3-4B-Thinking-2507` | `think` only | Native thinking template; do not pass `enable_thinking` |
| `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B` | `think` only | Native chat template with a forced `<think>` prefix |

The Qwen3-8B and Qwen3-4B thinking settings are temperature 0.6, top-p 0.95,
top-k 20, min-p 0. DeepSeek uses temperature 0.6 and top-p 0.95 without a
top-k cutoff; its model card warns about greedy repetition.
The default thinking cap is 6,144 generated tokens and the vLLM context is
8,192 tokens. These, the revision, available modes, gate thresholds, prompt
source, and code hashes are frozen in `design.json`.

## Colab setup

Use a GPU runtime, initially an L4 or A100. Install vLLM in that runtime, then
restart if Colab requests it. Run the commands below from the `code` directory:

```python
%cd /content/drive/MyDrive/verbalization_ceiling/code
!pip install -q vllm
```

The current `code/e50_reasoning_run.py` and all its hash-bound dependencies
must be present before freezing. Do not replace one after freeze and then
resume under new code. Existing v2 reports remain readable, but v2 generation
cannot be resumed with the changed code hash.

## A. Qwen3-8B thinking on, temporal sequences

This is the closest follow-up to the movie pilot: keep the model and thinking
regime, change the task. Freeze **thinking only** so the run does not spend GPU
time on the thinking-off control.

```bash
python e50_reasoning_run.py freeze \
  --out ../e50/qwen3_8b_temporal_v3 \
  --source-run ../e47/qwen2_5_7b_temporal_sequences_thinking_v10 \
  --model Qwen/Qwen3-8B --modes think --revision auto
python e50_reasoning_run.py inspect --out ../e50/qwen3_8b_temporal_v3
python e50_reasoning_run.py run --out ../e50/qwen3_8b_temporal_v3 --stop-after-interim
python e50_reasoning_run.py report --out ../e50/qwen3_8b_temporal_v3
```

## B. DeepSeek-R1-Distill-Qwen-7B thinking, movie recommendations

This is a separate model pilot, not a matched thinking-on/off comparison.
The model-card-recommended sampled decoding tests whether the earlier greedy
run's unfinished/repetitive answers were specific to greedy decoding.

```bash
python e50_reasoning_run.py freeze \
  --out ../e50/deepseek_r1_qwen7b_movie_v3 \
  --source-run ../e47/qwen2_5_7b_movie_recommendations_thinking_v10 \
  --model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B --revision auto
python e50_reasoning_run.py inspect --out ../e50/deepseek_r1_qwen7b_movie_v3
python e50_reasoning_run.py run --out ../e50/deepseek_r1_qwen7b_movie_v3 --stop-after-interim
python e50_reasoning_run.py report --out ../e50/deepseek_r1_qwen7b_movie_v3
```

## C. Qwen3-4B-Thinking-2507, movie recommendations

This model only supports thinking. Its card warns that it often thinks for
longer than earlier Qwen3 variants, so inspect cap and forced-read rates before
continuing. The model is smaller, but a long trace can still make it slow.

```bash
python e50_reasoning_run.py freeze \
  --out ../e50/qwen3_4b_thinking_movie_v3 \
  --source-run ../e47/qwen2_5_7b_movie_recommendations_thinking_v10 \
  --model Qwen/Qwen3-4B-Thinking-2507 --revision auto
python e50_reasoning_run.py inspect --out ../e50/qwen3_4b_thinking_movie_v3
python e50_reasoning_run.py run --out ../e50/qwen3_4b_thinking_movie_v3 --stop-after-interim
python e50_reasoning_run.py report --out ../e50/qwen3_4b_thinking_movie_v3
```

For notebook cells, prefix each shell command with `!` after `%cd`.

## Inspect the interim decision

For each run, inspect `interim_think.json` and `run_report.json`. `inspect`
prints the exact chat-rendered prompts and option-letter token IDs before any
GPU generation. All four option letters must be single tokens; otherwise do
not run the model under this protocol. Check that the DeepSeek prompt contains
one `<think>` opening, and that the Qwen3-4B prompt uses its native template.

The interim gate checks natural answer rate, forced reads, both switch/resist
classes overall and per cue family, and at least 90% nonempty reasoning
channels. A `NO-GO` is binding for that design. Do not edit the gate or its
saved decision after seeing the data. Report the interim as development data;
do not merge it with the 200-item main block as a confirmatory estimate.

If and only if `interim_think.json` says `GO`, resume that same directory:

```bash
python e50_reasoning_run.py run --out ../e50/<run_name>
python e50_reasoning_run.py report --out ../e50/<run_name>
python e50_reasoning_run.py export-review --out ../e50/<run_name>
```

`progress.jsonl` is append-only and resumable; `run_report.json` summarizes
outcomes, and `review_export.csv` contains separate final and thinking rows for
human/LLM acknowledgment review. Lexical mentions in `run_report.json` are
screens, not adjudicated disclosure labels. The run does not capture residuals
or run a neutral-insertion arm. A paper claim about internal state or cue-caused
individual switches needs additional, separately validated analyses.

If a model has excessive cap failures, use a **new output directory** and
freeze a larger `--think-max-new-tokens` with a larger `--max-model-len`, after
checking available GPU memory. Those settings change the design and must not
be edited into an existing `design.json`.
