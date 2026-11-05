# E50 v4: three additional reasoning models

E50 v4 is a separate, hash-bound runner. Keep using `e50_reasoning_run.py` for any
v3 design already frozen or running. The new runner is
`code/e50_reasoning_run_v4.py`; do not use it to resume a v3 design. It keeps
the 40-item interim / 200-item main split and the same feasibility gates.
The three new profiles use their native chat/reasoning formats:

| Model | Hardware to start with | Reasoning format |
|---|---|---|
| `deepseek-ai/DeepSeek-R1-Distill-Llama-8B` | L4 or A100 | `<think>...</think>` |
| `mistralai/Magistral-Small-2509` | A100 80GB | `[THINK]...[/THINK]`, mistral-common tokenization |
| `openai/gpt-oss-20b` | A100 80GB | Harmony analysis/final channels |

The Magistral prompt includes a frozen, concise system instruction to use its
native thinking tags. GPT-OSS uses medium reasoning effort. These are *not*
identical prompt wrappers across models; compare model/task outcomes, not an
isolated effect of architecture. All three use `think` mode only. E50's
underlying E47-v10 cue prompts and answer parser are unchanged.

## Setup and local checks

On Colab, install current vLLM (and its `mistral-common` dependency), then
restart the runtime if requested. Use an A100 80GB for Magistral and GPT-OSS.
The model's Hub revision is resolved and frozen in each design. Ensure all four
hash-bound code files are present *before* freezing: `e50_reasoning_run_v4.py`,
`e50_reasoning_pilot.py`, `e47_design.py`, `e46_prompts_cue.py`.

```bash
pip install -U vllm
python -m unittest discover -s code -p 'test_e50_reasoning_run_v4.py' -v
cd code
```

The runner has not been executed against these checkpoints on a GPU in this
workspace. `inspect` is a tokenizer-only preflight; do it before `run`. It
prints exact prompt wrappers, checks all four option letters are single tokens,
and checks the context budget. Stop if the rendered prompt or special-token
boundaries look wrong. The 40-item interim is the empirical validation gate:
check that thinking is nonempty, final answers parse, and forced reads work
before generating the main block.

## DeepSeek-R1-Distill-Llama-8B: movie recommendation

```bash
python e50_reasoning_run_v4.py freeze \
  --out ../e50/deepseek_r1_llama8b_movie_v4 \
  --source-run ../e47/qwen2_5_7b_movie_recommendations_thinking_v10 \
  --model deepseek-ai/DeepSeek-R1-Distill-Llama-8B --revision auto
python e50_reasoning_run_v4.py inspect --out ../e50/deepseek_r1_llama8b_movie_v4
python e50_reasoning_run_v4.py run --out ../e50/deepseek_r1_llama8b_movie_v4 --stop-after-interim
python e50_reasoning_run_v4.py report --out ../e50/deepseek_r1_llama8b_movie_v4
```

## Magistral-Small-2509: movie recommendation

```bash
python e50_reasoning_run_v4.py freeze \
  --out ../e50/magistral_small_2509_movie_v4 \
  --source-run ../e47/qwen2_5_7b_movie_recommendations_thinking_v10 \
  --model mistralai/Magistral-Small-2509 --revision auto
python e50_reasoning_run_v4.py inspect --out ../e50/magistral_small_2509_movie_v4
python e50_reasoning_run_v4.py run --out ../e50/magistral_small_2509_movie_v4 --stop-after-interim
python e50_reasoning_run_v4.py report --out ../e50/magistral_small_2509_movie_v4
```

This profile passes token IDs from `mistral-common` directly to vLLM. Do not
replace them with IDs from a Hugging Face tokenizer or re-tokenize a rendered
prompt. It uses vLLM's `tokenizer_mode/config_format/load_format=mistral`.

## GPT-OSS-20B: movie recommendation

```bash
python e50_reasoning_run_v4.py freeze \
  --out ../e50/gpt_oss_20b_movie_v4 \
  --source-run ../e47/qwen2_5_7b_movie_recommendations_thinking_v10 \
  --model openai/gpt-oss-20b --revision auto
python e50_reasoning_run_v4.py inspect --out ../e50/gpt_oss_20b_movie_v4
python e50_reasoning_run_v4.py run --out ../e50/gpt_oss_20b_movie_v4 --stop-after-interim
python e50_reasoning_run_v4.py report --out ../e50/gpt_oss_20b_movie_v4
```

Use recent vLLM for the official MXFP4 checkpoint on A100. Do not substitute a
BF16 conversion without a new design. GPT-OSS's answer must occur in the
Harmony `final` channel; an option mentioned only in `analysis` is not an
answer. The adapter accepts both known Harmony marker spellings, but the
interim output is the check for the pinned revision actually used.

## Continue after interim

Inspect `interim_think.json`, `run_report.json`, and a few `progress.jsonl`
records (the separated `thinking`, `final`, `answer`, `hit_cap`, and
`forced_read_status` fields). If the frozen interim verdict is **GO**, resume
the same directory:

```bash
python e50_reasoning_run_v4.py run --out ../e50/<run_name>
python e50_reasoning_run_v4.py report --out ../e50/<run_name>
python e50_reasoning_run_v4.py export-review --out ../e50/<run_name>
```

If it is **NO-GO**, report that outcome. Do not raise the cap or change parsing
inside the frozen run; a revised protocol needs a new output directory. These
new models are pilots until their own gates pass. No E47 capture/residual
analysis is implied by the E50 generation runner.
