#!/usr/bin/env python3
# =============================================================================
# e2_prompts — side-effect-free extraction of E2's dataset generators
#
# e2_discrete_recoverability.py builds its prompt sets at module import (and
# carries argparse/model state), so it cannot be imported by other scripts.
# This module holds the generators VERBATIM (same seeds, same prefixes) so that
# E7/E8/E10 reconstruct byte-identical prompts for the cached examples.
# If E2's generators change, re-copy them here.
# =============================================================================

from __future__ import annotations
import itertools
import numpy as np

N_EXAMPLES = 500

ARITH_PREFIX = """Solve step by step.
Problem: ((3+7)*2)-4
Step 1: 3+7=10
Step 2: 10*2=20
Step 3: 20-4=16
Answer: 16

Problem: ((8-3)+5)*2
Step 1: 8-3=5
Step 2: 5+5=10
Step 3: 10*2=20
Answer: 20

Problem: ((2+4)*3)-7
Step 1: 2+4=6
Step 2: 6*3=18
Step 3: 18-7=11
Answer: 11

Problem: """


def generate_arithmetic(n=N_EXAMPLES, seed=42):
    """Byte-identical to the generators in e1_sequential_bandwidth.py and
    E2_Discrete_Recoverability_HF.py (AST-verified; see _devtool_compare_impls.py).
    Emits the full field set both scripts use, so either can consume this."""
    rng = np.random.default_rng(seed); ops = ['+', '-', '*']; examples = []; att = 0
    while len(examples) < n and att < n * 20:
        att += 1; a, b, c, d = rng.integers(1, 9, 4); op1, op2, op3 = rng.choice(ops, 3)
        if op1 == '*' and op2 == '*': continue
        v1 = int(eval(f"{a}{op1}{b}")); v2 = int(eval(f"{v1}{op2}{c}")); v3 = int(eval(f"{v2}{op3}{d}"))
        if abs(v3) > 99: continue
        expr = f"(({a}{op1}{b}){op2}{c}){op3}{d}"
        examples.append({'expr': expr, 'answer': v3,
                         'cot_prompt': ARITH_PREFIX + expr + "\nStep 1:",
                         'plain_prompt': f"{expr}="})
    return examples


ENTITIES = ["wugs", "blickets", "daxes", "feps", "kiki", "mav", "zorp", "tove",
            "blorb", "snerg", "plim", "veck", "quob", "nilt", "yorp", "grix",
            "tald", "prunt", "sliv", "bront", "creel", "phog", "zant", "klev",
            "moop", "dreff", "spave", "tunk", "wrell", "flob"]
DEDUCT_PREFIX = """Answer the reasoning problem step by step.
Premise 1: All snurbs are blickets. Premise 2: All blickets are glurps. Question: Are all snurbs glurps?
Step 1: Premise 1 tells us snurbs → blickets.
Step 2: Premise 2 tells us blickets → glurps.
Step 3: By transitivity, snurbs → glurps. Yes.

Premise 1: All feps are daxes. Premise 2: All daxes are wugs. Question: Are all feps wugs?
Step 1: Premise 1 tells us feps → daxes.
Step 2: Premise 2 tells us daxes → wugs.
Step 3: By transitivity, feps → wugs. Yes.

Premise 1: All veck are moop. Premise 2: All moop are tunk. Question: Are all veck tunk?
Step 1: """


# ── balanced deduction (opt-in; see generate_deduction) ──────────────────────
# The original generator emits a valid syllogism every time, so answer=="Yes"
# for all 500 examples. Measured consequences: H(answer) = 0, so the
# problem-defined target carries no information; and accuracy is unmeasurable,
# because a model that replies "Yes" unconditionally scores 100% (e0n found
# 82-100% across every model, which says nothing).
#
# The negative case flips the DIRECTION of premise 2 rather than introducing an
# unrelated fourth entity: X→Y and Z→Y share a superset but imply nothing about
# X and Z. Same three entities, same surface form, so the model has to track
# direction rather than spot an odd word. The demonstrations carry one negative
# so the "No" form is shown.
#
# ── FORMAT 2: THE VERDICT MOVED OFF "Step 3:" AND ONTO ITS OWN "Answer:" LINE ─
#
# Format 1 ended each demonstration with the verdict inline -- "Step 3: By
# transitivity, snurbs → glurps. Yes." -- which gave the trace no terminator.
# Measured consequence in the first pilot: 73-97% of traces produced a scorable
# verdict, worst model 73%, against 96% for arithmetic and counting. Those two
# tasks close on "Answer: <number>" on a line of its own; this one closed on
# prose, and a model that keeps writing prose has no signal to stop, so it runs
# on into the next problem's header and the episode is lost.
#
# The label distribution was NOT the problem -- the loss was measured to be
# label-independent (worst H loss 0.0215 bits against a population 0.9871,
# smallest Fisher p 0.427, and the direction of the drop reverses across
# models). Termination was the only obstacle, so this fixes termination and
# changes nothing else: same entities, same premises, same negative
# construction, same 50/50 draw.
#
# Step 3 now states the relation and STOPS. The verdict appears exactly once,
# on the following line, in exactly one place a parser has to look. That is the
# counting task's shape, transposed to a binary target.
#
# THIS IS A BREAKING PROMPT CHANGE. Every trace captured under format 1 -- the
# six 30-example pilots in pilot_budget/balanced_deduction -- is scored under a
# different pattern and must not be pooled with format 2 traces.
# `BALANCED_DEDUCTION_FORMAT` is stamped so the two can never be confused, and
# e30 detects the generation from the traces themselves.
BALANCED_DEDUCTION_FORMAT = 2

DEDUCT_PREFIX_BALANCED = """Answer the reasoning problem step by step, then state the verdict on a line beginning "Answer:".
Premise 1: All snurbs are blickets. Premise 2: All blickets are glurps. Question: Are all snurbs glurps?
Step 1: Premise 1 tells us snurbs → blickets.
Step 2: Premise 2 tells us blickets → glurps.
Step 3: By transitivity, snurbs → glurps.
Answer: Yes

Premise 1: All feps are daxes. Premise 2: All wugs are daxes. Question: Are all feps wugs?
Step 1: Premise 1 tells us feps → daxes.
Step 2: Premise 2 tells us wugs → daxes, not daxes → wugs.
Step 3: No chain from feps to wugs.
Answer: No

Premise 1: All veck are moop. Premise 2: All moop are tunk. Question: Are all veck tunk?
Step 1: """


def generate_balanced_deduction(n=N_EXAMPLES, seed=42):
    """The balanced task, under its own name.

    A SEPARATE TASK, NOT A FLAG ON THE OLD ONE.

    `generate_deduction(balanced=True)` produced the same prompts but returned
    them under the task key `deduction`, so its traces would have been cached
    beside -- and indistinguishable from -- the withdrawn all-"Yes" task's. Two
    tasks that share a name and a cache directory is the failure mode this
    project has already hit twice with world-state levels and e2w variants.

    `balanced_deduction` is therefore its own key everywhere: its own prompts,
    its own e1 cache, its own budget entry, its own row in every table. Nothing
    from the withdrawn task can be mistaken for it and nothing already on disk
    is touched.
    """
    return generate_deduction(n, seed=seed, balanced=True)


def generate_deduction(n=N_EXAMPLES, seed=42, balanced=False):
    """Transitive-chain deduction.

    balanced=False (DEFAULT, and do not change it lightly) reproduces the
    original all-"Yes" generator exactly. E1 and E2 import this module, so
    flipping the default would change every prompt and silently invalidate every
    cached trace in the project against caches whose metadata records no
    generator version.

    balanced=True gives a ~50/50 Yes/No split for future runs. It is not
    backward compatible by construction, so a run using it needs its own E1
    cache; that is the point, not a limitation.
    """
    rng = np.random.default_rng(seed); triples = list(itertools.permutations(ENTITIES, 3))
    rng.shuffle(triples); examples = []
    if not balanced:
        for X, Y, Z in triples[:n]:
            p1, p2 = f"All {X} are {Y}.", f"All {Y} are {Z}."; q = f"Are all {X} {Z}?"
            prompt = f"Premise 1: {p1} Premise 2: {p2} Question: {q}"
            cot = DEDUCT_PREFIX.replace("Are all veck tunk?\nStep 1: ", f"{q}\nStep 1: "
                  ).replace("All veck are moop. Premise 2: All moop are tunk.", f"{p1} Premise 2: {p2}")
            examples.append({'X': X, 'Y': Y, 'Z': Z, 'answer': "Yes", 'cot_prompt': cot,
                             'plain_prompt': prompt + " Answer:"})
        return examples

    # Drawn from the same shuffled triple stream, so a balanced run and an
    # unbalanced run at the same seed cover the same entity triples and differ
    # only in premise 2's direction.
    valid = rng.random(n) < 0.5
    stem_old = ("Premise 1: All veck are moop. Premise 2: All moop are tunk. "
                "Question: Are all veck tunk?\nStep 1: ")
    for i, (X, Y, Z) in enumerate(triples[:n]):
        if valid[i]:
            p2, ans = f"All {Y} are {Z}.", "Yes"
        else:
            p2, ans = f"All {Z} are {Y}.", "No"
        p1, q = f"All {X} are {Y}.", f"Are all {X} {Z}?"
        stem_new = f"Premise 1: {p1} Premise 2: {p2} Question: {q}\nStep 1: "
        examples.append({
            'X': X, 'Y': Y, 'Z': Z, 'answer': ans, 'valid_chain': bool(valid[i]),
            'cot_prompt': DEDUCT_PREFIX_BALANCED.replace(stem_old, stem_new),
            'plain_prompt': f"Premise 1: {p1} Premise 2: {p2} Question: {q} Answer:"})
    return examples


COUNT_LETTERS = ["a", "b", "c"]
COUNT_PREFIX = """Count how many times the target letter appears.
Sequence: [a b c a b a] Target: a
Step 1: Position 1: a — match (count=1)
Step 2: Position 2: b — no match
Step 3: Position 3: c — no match
Step 4: Position 4: a — match (count=2)
Step 5: Position 5: b — no match
Step 6: Position 6: a — match (count=3)
Answer: 3

Sequence: [b c b a c b] Target: b
Step 1: Position 1: b — match (count=1)
Step 2: Position 2: c — no match
Step 3: Position 3: b — match (count=2)
Step 4: Position 4: a — no match
Step 5: Position 5: c — no match
Step 6: Position 6: b — match (count=3)
Answer: 3

Sequence: """


def generate_counting(n=N_EXAMPLES, seq_len=6, seed=42):
    rng = np.random.default_rng(seed); examples = []
    for _ in range(n):
        seq = list(rng.choice(COUNT_LETTERS, size=seq_len)); tgt = rng.choice(COUNT_LETTERS)
        s = " ".join(seq); prompt = f"[{s}] Target: {tgt}"
        examples.append({'seq': seq, 'target': tgt, 'count': seq.count(tgt),
                         'answer': str(seq.count(tgt)),
                         'plain_prompt': prompt + "\nAnswer:",
                         'cot_prompt': COUNT_PREFIX + prompt + "\nStep 1:"})
    return examples


ALL_TASKS = ('arithmetic', 'deduction', 'counting', 'balanced_deduction')


def build_examples(n=N_EXAMPLES, seq_len=6, tasks=None):
    """Task sets, keyed by task name.

    `n` MUST match the value used when the traces were generated. E1 takes it
    from CLI/config (500 for most models, 250 for the reasoning-distilled runs),
    so callers that consume cached traces have to pass the same number — the
    default here is only the common case, not a guarantee.

    `tasks=None` returns the ORIGINAL THREE and nothing else. That default is
    load-bearing: arithmetic and counting are fully captured across the grid,
    and a caller that silently gained a fourth task would either regenerate
    them or index a dict whose shape it did not expect. Ask for
    `balanced_deduction` explicitly.
    """
    want = tuple(tasks) if tasks else ('arithmetic', 'deduction', 'counting')
    unknown = [t for t in want if t not in ALL_TASKS]
    if unknown:
        raise ValueError(f"unknown task(s) {unknown}; known: {list(ALL_TASKS)}")
    out = {}
    for t in want:
        if t == 'arithmetic':
            out[t] = generate_arithmetic(n)
        elif t == 'deduction':
            out[t] = generate_deduction(n)
        elif t == 'counting':
            out[t] = generate_counting(n, seq_len=seq_len)
        elif t == 'balanced_deduction':
            out[t] = generate_balanced_deduction(n)
    return out


def build_prompts(n=N_EXAMPLES, seq_len=6, tasks=None):
    ex = build_examples(n, seq_len, tasks=tasks)
    return {task: [e['cot_prompt'] for e in ex[task]] for task in ex}
