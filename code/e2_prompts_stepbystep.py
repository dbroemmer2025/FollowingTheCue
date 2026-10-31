#!/usr/bin/env python3
"""Zero-shot step-by-step prompts for the reasoning tasks.

This is a third elicitation regime, separate from:

* imitative: few-shot completion prompts with worked examples;
* think: chat-template prompts for native extended-thinking models.

The stepbystep regime poses only the problem instance and asks for standard
chain-of-thought style reasoning in a publication-friendly way.  It keeps the
same answer contracts as the other regimes so completion and accuracy parsers
can reuse their existing answer patterns.
"""
from __future__ import annotations

from e2_prompts import (N_EXAMPLES, generate_arithmetic,
                        generate_balanced_deduction, generate_counting,
                        generate_deduction)


INSTR = {
    "arithmetic": (
        "Think through the arithmetic problem step by step. End your response "
        "with a final line of the form \"Answer: N\"."
    ),
    "counting": (
        "Count how many times the target letter appears. Think through the "
        "counting problem step by step. End your response "
        "with a final line of the form \"Answer: N\"."
    ),
    "balanced_deduction": (
        "Think through the reasoning problem step by step. End your response "
        "with a final line of the form \"Answer: Yes\" or \"Answer: No\"."
    ),
    "deduction": (
        "Think through the reasoning problem step by step. End your response "
        "with a final line of the form \"Answer: Yes\" or \"Answer: No\"."
    ),
}


def problem_body(task, ex):
    """The problem statement alone, with no demonstrations."""
    if task == "arithmetic":
        return f"Problem: {ex['expr']}"
    if task == "counting":
        return f"Sequence: [{' '.join(ex['seq'])}] Target: {ex['target']}"
    if task == "balanced_deduction":
        p2 = (f"All {ex['Y']} are {ex['Z']}." if ex["valid_chain"]
              else f"All {ex['Z']} are {ex['Y']}.")
        return (f"Premise 1: All {ex['X']} are {ex['Y']}. Premise 2: {p2} "
                f"Question: Are all {ex['X']} {ex['Z']}?")
    if task == "deduction":
        return (f"Premise 1: All {ex['X']} are {ex['Y']}. "
                f"Premise 2: All {ex['Y']} are {ex['Z']}. "
                f"Question: Are all {ex['X']} {ex['Z']}?")
    raise ValueError(task)


def prompt(task, ex):
    return f"{INSTR[task]}\n\n{problem_body(task, ex)}"


def build_stepbystep_examples(n=N_EXAMPLES, seq_len=6, tasks=None):
    want = tuple(tasks) if tasks else ("arithmetic", "counting", "balanced_deduction")
    out = {}
    for task in want:
        if task == "arithmetic":
            out[task] = generate_arithmetic(n)
        elif task == "counting":
            out[task] = generate_counting(n, seq_len=seq_len)
        elif task == "balanced_deduction":
            out[task] = generate_balanced_deduction(n)
        elif task == "deduction":
            out[task] = generate_deduction(n)
        else:
            raise ValueError(task)
    return out


def build_stepbystep_prompts(n=N_EXAMPLES, seq_len=6, tasks=None):
    examples = build_stepbystep_examples(n=n, seq_len=seq_len, tasks=tasks)
    return {task: [prompt(task, ex) for ex in exs]
            for task, exs in examples.items()}


def prompt_provenance(tasks=None):
    return {
        "condition": "stepbystep",
        "template": "problem-only zero-shot completion prompt",
        "instructions": {t: INSTR[t] for t in (tasks or INSTR)},
    }
