#!/usr/bin/env python3
# =============================================================================
# e2_prompts_think — chat-templated, zero-shot prompts for the EXTENDED-THINKING
#                    condition (see dissertation-plan/07-thinking-mode-condition.md)
#
# WHY A SECOND PROMPT MODULE
#   The main grid uses raw few-shot COMPLETION prompts. Those demonstrations are
#   exactly what suppress extended thinking: a model shown three worked
#   three-step solutions imitates them, which is why the reasoning-distilled
#   models currently produce 35-87 token episodes -- indistinguishable from
#   Llama-3.2-3B. To observe the regime these models were trained for, the
#   demonstrations have to go and the chat template has to be applied.
#
#   The PROBLEM INSTANCES are unchanged. Both conditions draw from the same
#   generators in e2_prompts with the same seeds, so example i is the same
#   problem in both regimes and the comparison can be paired example-by-example.
#   Only the elicitation changes.
#
# THE MANIPULATION IS DELIBERATELY A BUNDLE
#   template + zero-shot + thinking mode all move together, because that bundle
#   IS the regime. No claim is made about the chat template in isolation. Trace
#   length -- the part that could confound the channel argument -- is controlled
#   analytically in E19's budget-matched sweep, not by decomposing the prompt.
#
# USAGE
#   from transformers import AutoTokenizer
#   from e2_prompts_think import build_think_prompts, supports_thinking_toggle
#   tok = AutoTokenizer.from_pretrained(hf)
#   prompts = build_think_prompts(tok, n=150)            # {task: [str, ...]}
#   prompts_off = build_think_prompts(tok, n=150, enable_thinking=False)
# =============================================================================

from __future__ import annotations

from e2_prompts import (generate_arithmetic, generate_deduction,
                        generate_counting, N_EXAMPLES)

# The instruction carries the output contract. Without an explicit answer form,
# completion detection has nothing to key on beyond EOS, and a model that
# reasons correctly but trails off is scored the same as one that never solved
# the problem.
INSTR = {
    'arithmetic': ("Solve the arithmetic problem. End your reply with a final "
                   "line of the form 'Answer: N'."),
    'deduction':  ("Answer the reasoning problem. End your reply with a final "
                   "line of the form 'Answer: Yes' or 'Answer: No'."),
    'counting':   ("Count how many times the target letter appears. End your "
                   "reply with a final line of the form 'Answer: N'."),
}


def problem_body(task, ex):
    """The problem statement alone, with no demonstrations and no answer cue.

    Rebuilt from the generator's structured fields rather than by stripping a
    suffix off `plain_prompt`, so a change to either prompt format cannot
    silently corrupt the other condition.
    """
    if task == 'arithmetic':
        return ex['expr']
    if task == 'deduction':
        X, Y, Z = ex['X'], ex['Y'], ex['Z']
        return (f"Premise 1: All {X} are {Y}. Premise 2: All {Y} are {Z}. "
                f"Question: Are all {X} {Z}?")
    if task == 'counting':
        return f"[{' '.join(ex['seq'])}] Target: {ex['target']}"
    raise ValueError(task)


def supports_thinking_toggle(tok):
    """Does this tokenizer's chat template actually honour `enable_thinking`?

    Templates are Jinja, and Jinja ignores undefined variables -- so passing
    enable_thinking to a template that does not implement it succeeds silently
    and changes nothing. Rather than string-matching the model name (which
    breaks on forks, local paths and renames), render both ways and compare.
    A real toggle produces different text; anything else does not.

    Qwen3 gates thinking this way, which makes it the single most valuable model
    in this experiment: thinking on vs off at identical weights, identical
    template and identical problems is a far tighter manipulation than the
    cross-condition contrast.
    """
    msgs = [{"role": "user", "content": "x"}]
    try:
        on = tok.apply_chat_template(msgs, tokenize=False,
                                     add_generation_prompt=True,
                                     enable_thinking=True)
        off = tok.apply_chat_template(msgs, tokenize=False,
                                      add_generation_prompt=True,
                                      enable_thinking=False)
    except Exception:
        return False
    return on != off


def has_chat_template(tok):
    return getattr(tok, 'chat_template', None) is not None


def think_prompt(tok, task, ex, enable_thinking=True):
    """One rendered chat prompt for one example."""
    if not has_chat_template(tok):
        raise RuntimeError(
            f"{getattr(tok, 'name_or_path', tok)!r} has no chat template, so it "
            f"cannot be run in the thinking condition. The extended-thinking "
            f"experiment applies only to the reasoning-distilled models; base "
            f"models stay in the imitative condition.")
    content = f"{INSTR[task]}\n\n{problem_body(task, ex)}"
    kw = {}
    if supports_thinking_toggle(tok):
        kw['enable_thinking'] = enable_thinking
    elif not enable_thinking:
        raise RuntimeError(
            f"enable_thinking=False was requested but {getattr(tok,'name_or_path',tok)!r}"
            f" has no thinking toggle in its chat template. Silently returning a "
            f"thinking-on prompt would make the off condition a duplicate of the "
            f"on condition -- refusing.")
    return tok.apply_chat_template([{"role": "user", "content": content}],
                                   tokenize=False, add_generation_prompt=True,
                                   **kw)


def build_think_examples(n=N_EXAMPLES, seq_len=6):
    """Same generators, same seeds as the imitative condition."""
    return {'arithmetic': generate_arithmetic(n),
            'deduction':  generate_deduction(n),
            'counting':   generate_counting(n, seq_len=seq_len)}


def build_think_prompts(tok, n=N_EXAMPLES, seq_len=6, enable_thinking=True):
    ex = build_think_examples(n, seq_len)
    return {t: [think_prompt(tok, t, e, enable_thinking) for e in ex[t]]
            for t in ex}


def prompt_provenance(tok, enable_thinking=True):
    """Record what was actually rendered.

    Chat templates change between transformers releases and between checkpoint
    revisions. A silent template change is indistinguishable from a model effect
    in the results, so the rendered string and the template itself are stored
    alongside every cache.
    """
    import hashlib
    tmpl = getattr(tok, 'chat_template', None) or ''
    ex = build_think_examples(1)
    sample = {t: think_prompt(tok, t, ex[t][0], enable_thinking) for t in ex}
    return {
        'condition': 'thinking',
        'enable_thinking': enable_thinking,
        'thinking_toggle_supported': supports_thinking_toggle(tok),
        'chat_template_sha1': hashlib.sha1(tmpl.encode()).hexdigest(),
        'rendered_example': sample,
        'instructions': dict(INSTR),
    }
