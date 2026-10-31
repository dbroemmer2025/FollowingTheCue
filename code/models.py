"""Model registry for the verbalization-ceiling experiments.

Maps a CLI key to its HuggingFace checkpoint, dtype, and the per-model run
parameters (examples per task and the generation budget). Imported by every
experiment script so a single --model flag selects a model consistently. All
models load via the HuggingFace transformers API.

HOW n_examples AND cot_tokens WERE SET
--------------------------------------
Both come from a measurement, not from intuition. e0e_pilot_budget.py generates
30 examples per model-task under a deliberately generous cap and records
time-to-completion, where "completed" means the trace reached the end of a
reasoning episode -- either the answer form the few-shot prefix demonstrates, or
the model starting the next problem. Results: pilot_budget/.

cot_tokens = ceil(1.25 x worst-task p99), rounded up to a tidy value. It is a
SAFETY CAP, not a target length: E1 stops generation at the end of the episode
(e1_sequential_bandwidth.STOP_STRINGS), so the cap binds only when a trace fails
to terminate.

  Why a stop criterion matters. Without one, generation always runs the full
  budget: the model solves the assigned problem, then invents and solves further
  problems until the budget is exhausted. Against the pilot's measured episode
  lengths, the assigned problem accounted for a mean of 29% of cached trace
  tokens under the previous settings -- and 3-9% for the reasoning-distilled
  models at cot_tokens=1024, i.e. 12-29 episodes per trace.

n_examples = 500 for every model. Uniform by design: a single value removes the
budget-confound argument from cross-model comparisons entirely. 500 also keeps
the decoder's training set adequate now that traces are one episode: at a median
arithmetic episode of ~35 tokens, ~half of which survive the content filter,
n=500 yields ~8.7k content tokens against MAX_DECODER_TOKENS=12000. Under the
previous run-to-budget regime each example contributed ~500 content tokens, so
the 12k cap discarded most of them and n barely mattered; with single-episode
traces n binds directly.

WHAT THE PILOT ALSO SHOWED (report these, do not tune them away)
---------------------------------------------------------------
* The reasoning-distilled models are NOT long-CoT on these prompts. DeepSeek-R1
  distills and Qwen3-14B complete in 35-87 tokens median, indistinguishable from
  Llama and Mistral. These are raw few-shot completion prompts with no chat
  template, so the models follow the demonstrated 3-step pattern rather than
  entering extended-thinking mode. Their previous cot_tokens=1024 was ~25x the
  measured need.

* No model is length-censored. Every task's p99 sits well below its pilot cap,
  so the completing mass is fully observed and a larger cap would not recover
  the non-completions.

* Non-completion is a TERMINATION failure, not a length problem, and it
  concentrates in deduction -- the worst-completing task for 10 of 11 models.
  Worst cases: gpt2-small 13%, gpt2-medium 43%, gpt2-xl 70%. Those traces run to
  the cap because the model never closes the episode.

Per-model provenance (worst-task p99 -> cap, and the minimum completion rate
across the three tasks) is recorded in the "pilot" field of each entry.
"""

# Examples per task. Uniform across models on purpose -- see the module
# docstring. Change it here rather than per entry so it cannot drift.
N_EXAMPLES = 500

# ── Extended-thinking condition ──────────────────────────────────────────────
# A second elicitation regime for the reasoning-distilled models only: chat
# template, zero-shot, native extended reasoning. See
# dissertation-plan/07-thinking-mode-condition.md.
#
# n_examples is 150, not 500, and that is not an oversight. MAX_DECODER_TOKENS
# subsamples at random across the pooled content tokens, so n binds only when
# traces are SHORT. At ~1500 tokens per trace with ~half surviving the content
# filter, n=150 gives ~112k content tokens against the 12k cap -- 9x saturated,
# with all 150 groups represented. The same arithmetic that forced n=500 for
# single-episode traces forces a much smaller n here.
#
# cot_tokens starts as a placeholder: run e0e_pilot_budget.py --think and set it
# from the measurement, exactly as for the imitative condition. THINK_PILOT_DONE
# guards against running the main experiment on a guessed budget.
# MEASURED. e0e --think, 30 traces per model-task, cap 4096 (never binding:
# completion and </think> closure were both 100% for all four models).
#
# n_examples = 250, raised from the 150 placeholder for two measured reasons:
#   1. The shortest think block in the grid is deepseek-7b arithmetic at p50=62
#      tokens. At n=150 that cell supplies only ~4.6k content tokens against the
#      12k decoder cap; at n=250 it supplies ~7.8k, comparable to the imitative
#      arithmetic cells. Ten of the twelve cells saturate at either value -- it
#      is the two short arithmetic cells that bind.
#   2. E19 pairs on min(n_imitative, n_thinking), so the thinking n caps the
#      paired sample outright. 150 -> 250 narrows the Delta-gap CI by ~23%, and
#      that CI is the headline statistic of the whole condition.
#
# Going further to 500 would narrow it by 45% instead, but roughly doubles
# generation for the 32B, whose replies run to a p50 of 573 tokens. 250 is the
# knee.
THINK_N_EXAMPLES = 250
THINK_PILOT_DONE = True

MODELS = {
    "gpt2-small": {
        "label": "gpt2-small", "hf": "gpt2", "dtype": "float32",
        "n_examples": N_EXAMPLES, "cot_tokens": 512,
        "pilot": {"worst_p99": 366, "worst_task": "arithmetic",
                  "min_completion": 0.13, "min_completion_task": "deduction"}},

    "gpt2-medium": {
        "label": "gpt2-medium", "hf": "gpt2-medium", "dtype": "float32",
        "n_examples": N_EXAMPLES, "cot_tokens": 384,
        "pilot": {"worst_p99": 304, "worst_task": "deduction",
                  "min_completion": 0.43, "min_completion_task": "deduction"}},

    "gpt2-large": {
        "label": "gpt2-large", "hf": "gpt2-large", "dtype": "float32",
        "n_examples": N_EXAMPLES, "cot_tokens": 256,
        "pilot": {"worst_p99": 192, "worst_task": "deduction",
                  "min_completion": 0.77, "min_completion_task": "deduction"}},

    "gpt2-xl": {
        "label": "gpt2-xl", "hf": "gpt2-xl", "dtype": "float32",
        "n_examples": N_EXAMPLES, "cot_tokens": 768,
        "pilot": {"worst_p99": 489, "worst_task": "deduction",
                  "min_completion": 0.70, "min_completion_task": "deduction"}},

    "gpt-neo-1.3b": {
        "label": "gpt-neo-1.3b", "hf": "EleutherAI/gpt-neo-1.3B", "dtype": "float32",
        "n_examples": N_EXAMPLES, "cot_tokens": 256,
        "pilot": {"worst_p99": 176, "worst_task": "deduction",
                  "min_completion": 0.90, "min_completion_task": "deduction"}},

    "llama-3.2-3b": {
        "label": "llama-3.2-3b", "hf": "meta-llama/Llama-3.2-3B", "dtype": "float16",
        "n_examples": N_EXAMPLES, "cot_tokens": 128,
        "pilot": {"worst_p99": 96, "worst_task": "counting",
                  "min_completion": 0.97, "min_completion_task": "deduction"}},

    # Provisional entries for the post-GPT-2 comparison grid. Run
    # e0e_pilot_budget.py on these models before treating their L_max values as
    # paper measurements.
    "gemma-2-2b": {
        "label": "gemma-2-2b", "hf": "google/gemma-2-2b", "dtype": "bfloat16",
        "n_examples": N_EXAMPLES, "cot_tokens": 768,
        "pilot": {"status": "unmeasured", "worst_p99": None, "worst_task": None,
                  "min_completion": None, "min_completion_task": None}},

    "gemma-2-9b": {
        "label": "gemma-2-9b", "hf": "google/gemma-2-9b", "dtype": "bfloat16",
        "n_examples": N_EXAMPLES, "cot_tokens": 768,
        "pilot": {"status": "unmeasured", "worst_p99": None, "worst_task": None,
                  "min_completion": None, "min_completion_task": None}},

    # E47 model-replication checkpoint. At 9B in bfloat16 this is near the
    # practical upper end for an L4; use device_map=auto and pilot generation
    # before committing to a full run. Hugging Face access requires accepting
    # Google's Gemma license.
    "gemma-2-9b-it": {
        "label": "gemma-2-9b-it", "hf": "google/gemma-2-9b-it",
        "dtype": "bfloat16",
        "n_examples": 200, "cot_tokens": 768,
        "pilot": {"status": "unmeasured", "worst_p99": None,
                  "worst_task": None, "min_completion": None,
                  "min_completion_task": None}},

    "gemma-4-12b-it": {
        "label": "gemma-4-12b-it", "hf": "google/gemma-4-12B-it",
        "dtype": "bfloat16",
        "n_examples": N_EXAMPLES, "cot_tokens": 768,
        "pilot": {"status": "unmeasured", "worst_p99": None, "worst_task": None,
                  "min_completion": None, "min_completion_task": None}},

    "gemma-4-31b-it": {
        "label": "gemma-4-31b-it", "hf": "google/gemma-4-31B-it",
        "dtype": "bfloat16",
        "n_examples": N_EXAMPLES, "cot_tokens": 768,
        "pilot": {"status": "unmeasured", "worst_p99": None, "worst_task": None,
                  "min_completion": None, "min_completion_task": None}},

    "mistral-nemo-base-2407": {
        "label": "mistral-nemo-base-2407",
        "hf": "mistralai/Mistral-Nemo-Base-2407", "dtype": "bfloat16",
        "n_examples": 200, "cot_tokens": 768,
        "pilot": {"status": "unmeasured", "worst_p99": None, "worst_task": None,
                  "min_completion": None, "min_completion_task": None}},

    # E47 instruction-tuned replication of the registered Mistral Nemo base
    # checkpoint. Keep the frozen E47 prompt mode fixed across model comparisons;
    # using its chat template is a separate prompting condition.
    "mistral-nemo-instruct-2407": {
        "label": "mistral-nemo-instruct-2407",
        "hf": "mistralai/Mistral-Nemo-Instruct-2407", "dtype": "bfloat16",
        "n_examples": 200, "cot_tokens": 768,
        "pilot": {"status": "unmeasured", "worst_p99": None,
                  "worst_task": None, "min_completion": None,
                  "min_completion_task": None}},

    "olmo-2-1124-7b-instruct": {
        "label": "olmo-2-1124-7b-instruct",
        "hf": "allenai/OLMo-2-1124-7B-Instruct", "dtype": "bfloat16",
        "n_examples": 200, "cot_tokens": 768,
        "pilot": {"status": "unmeasured", "worst_p99": None, "worst_task": None,
                  "min_completion": None, "min_completion_task": None}},

    "qwen2.5-7b": {
        "label": "qwen2.5-7b", "hf": "Qwen/Qwen2.5-7B", "dtype": "bfloat16",
        "n_examples": 200, "cot_tokens": 768,
        "pilot": {"status": "unmeasured", "worst_p99": None, "worst_task": None,
                  "min_completion": None, "min_completion_task": None}},

    # Direct E47 base-versus-post-training comparison at identical Qwen2.5
    # architecture and scale. Preserve the frozen E47 prompt mode for the
    # confirmatory contrast; a separate chat-template cell is a different
    # prompting condition, not a replacement for this cell.
    "qwen2.5-7b-instruct": {
        "label": "qwen2.5-7b-instruct",
        "hf": "Qwen/Qwen2.5-7B-Instruct", "dtype": "bfloat16",
        "n_examples": 200, "cot_tokens": 768,
        "pilot": {"status": "unmeasured", "worst_p99": None,
                  "worst_task": None, "min_completion": None,
                  "min_completion_task": None}},

    # Widely used cross-family instruction-tuned replication. The checkpoint
    # fits an L4 in bfloat16, but access is gated by the Llama 3.1 license.
    "llama-3.1-8b-instruct": {
        "label": "llama-3.1-8b-instruct",
        "hf": "meta-llama/Llama-3.1-8B-Instruct", "dtype": "bfloat16",
        "n_examples": 200, "cot_tokens": 768,
        "pilot": {"status": "unmeasured", "worst_p99": None,
                  "worst_task": None, "min_completion": None,
                  "min_completion_task": None}},

    "mistral-7b-v0.1": {
        "label": "mistral-7b-v0.1", "hf": "mistralai/Mistral-7B-v0.1", "dtype": "float16",
        "n_examples": N_EXAMPLES, "cot_tokens": 192,
        "pilot": {"worst_p99": 142, "worst_task": "counting",
                  "min_completion": 0.87, "min_completion_task": "deduction"}},

    "deepseek-r1-distill-qwen-7b": {
        "label": "deepseek-r1-distill-qwen-7b",
        "hf": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B", "dtype": "float16",
        "n_examples": N_EXAMPLES, "cot_tokens": 128,
        "pilot": {"worst_p99": 87, "worst_task": "counting",
                  "min_completion": 0.97, "min_completion_task": "deduction"},
        "think": {"n_examples": THINK_N_EXAMPLES, "cot_tokens": 1536,
                  "enable_thinking": True,
                  "pilot": {"worst_p99": 1085, "worst_task": "counting",
                            "min_completion": 1.00, "closed_frac": 1.00,
                            "think_p50": {"arithmetic": 62, "deduction": 350,
                                          "counting": 274}}}},

    "deepseek-r1-distill-llama-8b": {
        "label": "deepseek-r1-distill-llama-8b",
        "hf": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        "dtype": "bfloat16",
        "n_examples": 200, "cot_tokens": 768,
        "pilot": {"status": "unmeasured", "worst_p99": None,
                  "worst_task": None, "min_completion": None,
                  "min_completion_task": None},
        "think": {"n_examples": THINK_N_EXAMPLES, "cot_tokens": 1536,
                  "enable_thinking": True,
                  "pilot": {"status": "unmeasured", "worst_p99": None,
                            "worst_task": None, "min_completion": None,
                            "closed_frac": None}}},

    # The long tail here is real: deduction p50=38 but p90=582, and a third of
    # its completions were detected by the model starting the next problem
    # rather than by the demonstrated answer form. Hence a much larger cap than
    # its 7B and 32B siblings despite a similar median.
    "deepseek-r1-distill-qwen-14b": {
        "label": "deepseek-r1-distill-qwen-14b",
        "hf": "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B", "dtype": "float16",
        "n_examples": N_EXAMPLES, "cot_tokens": 1536,
        "pilot": {"worst_p99": 863, "worst_task": "deduction",
                  "min_completion": 0.97, "min_completion_task": "deduction"},
        "think": {"n_examples": THINK_N_EXAMPLES, "cot_tokens": 1536,
                  "enable_thinking": True,
                  "pilot": {"worst_p99": 1048, "worst_task": "deduction",
                            "min_completion": 1.00, "closed_frac": 1.00,
                            "think_p50": {"arithmetic": 321, "deduction": 435,
                                          "counting": 335}}}},

    # ~64 GB of fp16 weights: needs an 80 GB GPU or quantisation.
    "deepseek-r1-distill-qwen-32b": {
        "label": "deepseek-r1-distill-qwen-32b",
        "hf": "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B", "dtype": "float16",
        "n_examples": N_EXAMPLES, "cot_tokens": 128,
        "pilot": {"worst_p99": 87, "worst_task": "counting",
                  "min_completion": 0.83, "min_completion_task": "deduction"},
        "think": {"n_examples": THINK_N_EXAMPLES, "cot_tokens": 1536,
                  "enable_thinking": True,
                  "pilot": {"worst_p99": 973, "worst_task": "deduction",
                            "min_completion": 1.00, "closed_frac": 1.00,
                            "think_p50": {"arithmetic": 74, "deduction": 468,
                                          "counting": 315}}}},

    # The only model with a real enable_thinking toggle, so the only one where
    # thinking on/off is measurable at identical weights -- the tightest
    # manipulation in the E19 design. Measured thinking-off: all three tasks
    # complete 100% at p50 41-48 tokens, i.e. indistinguishable in length from
    # the imitative condition, which is the sanity check that "off" is off.
    # cot_tokens=192 there rather than 1024: the pilot's reasoning-model grid
    # starts at 1024, but worst-task p99 is only 128.
    "qwen3-14b": {
        "label": "qwen3-14b", "hf": "Qwen/Qwen3-14B", "dtype": "float16",
        "n_examples": N_EXAMPLES, "cot_tokens": 1024,
        "pilot": {"worst_p99": 638, "worst_task": "deduction",
                  "min_completion": 1.00, "min_completion_task": "arithmetic"},
        "think": {"n_examples": THINK_N_EXAMPLES, "cot_tokens": 1024,
                  "enable_thinking": True,
                  "pilot": {"worst_p99": 777, "worst_task": "arithmetic",
                            "min_completion": 1.00, "closed_frac": 1.00,
                            "think_p50": {"arithmetic": 272, "deduction": 289,
                                          "counting": 186}}},
        # Thinking-OFF, measured separately. Its own cot_tokens because the
        # pilot's reasoning-model grid starts at 1024 while the worst-task p99
        # here is only 128. A 1024 cap would be harmless (completion is 100% and
        # generation stops at EOS), but the paper should report what was measured.
        #
        # The sanity check that "off" is really off: p50 of 41-48 tokens across
        # the three tasks, i.e. indistinguishable in length from the imitative
        # condition (35/43/84), and 0% of traces produced a think block of their
        # own -- the template injects an already-closed empty one.
        # cot_tokens is 1024, NOT the measured 192, and that is deliberate.
        #
        # think_off exists to be compared against think, which runs at 1024.
        # Setting think_off to its measured need would make the two conditions
        # differ in generation budget AS WELL AS in enable_thinking, putting a
        # confound inside the one experiment designed to isolate that variable.
        #
        # The extra budget costs nothing, and MEASURED e0n confirms it costs
        # literally nothing: on the cached traces post% = 0.0 and at_cap% = 0
        # for all three tasks, with med_cached == med_epis at 40/40/60 tokens.
        # Generation stops on the stop string, not the budget, so the traces are
        # byte-identical at 192 or 1024. There is not even filler to remove.
        #
        # 192 is retained below as measured_cot_tokens: it is the honest answer
        # to "how long does this condition actually need", and it is what a
        # budget table in the paper should report.
        "think_off": {"n_examples": THINK_N_EXAMPLES, "cot_tokens": 1024,
                      "measured_cot_tokens": 192,
                      "enable_thinking": False,
                      "pilot": {"worst_p99": 128, "worst_task": "counting",
                                "min_completion": 1.00,
                                "p50": {"arithmetic": 41, "deduction": 41,
                                        "counting": 48}}}},
}


# Only these models have a thinking condition. The base models have no chat
# template and no extended-reasoning mode, so they stay in the imitative
# condition; asking for --regime thinking on them should fail loudly rather
# than silently fall back.
THINKING_MODELS = [k for k, v in MODELS.items() if 'think' in v]


def think_cfg(key):
    """Config for the extended-thinking condition, or a clear error."""
    c = MODELS.get(key)
    if c is None:
        raise KeyError(f"{key!r} not in models.py. Available: {sorted(MODELS)}")
    if 'think' not in c:
        raise ValueError(
            f"{key!r} has no extended-thinking condition. That regime applies "
            f"only to the reasoning-distilled models ({THINKING_MODELS}); the "
            f"base models have no chat template or thinking mode.")
    if not THINK_PILOT_DONE:
        raise RuntimeError(
            f"the thinking-condition budget for {key!r} has not been measured. "
            f"cot_tokens is still the {c['think']['cot_tokens']} placeholder, "
            f"which would reproduce exactly the cap-determines-trace-length "
            f"problem the pilot exists to remove.\n"
            f"    run:  python e0e_pilot_budget.py --think --models {key}\n"
            f"    then set models.py THINK_PILOT_DONE = True")
    if (c.get('think', {}).get('pilot') or {}).get('status') == 'unmeasured':
        raise RuntimeError(
            f"the thinking-condition budget for {key!r} has not been measured. "
            f"Run: python e0e_pilot_budget.py --think --models {key}")
    return c['think']


def resolve_torch_dtype(name):
    """Resolve a registry dtype string to a torch dtype.

    Imported lazily so lightweight metadata consumers can import models.py
    without importing torch.
    """
    import torch

    table = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    key = str(name).lower()
    if key not in table:
        raise ValueError(
            f"unsupported dtype {name!r}; use float32, float16, or bfloat16")
    return table[key]
