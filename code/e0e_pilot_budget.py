#!/usr/bin/env python3
# =============================================================================
# e0e — pilot run to SET the generation budget from measured completion length
#
# WHY
#   Every model in the current grid runs to its cap (69-100% of budget used,
#   mostly >90%), so the cap is a binding constraint rather than a safety net:
#   the cached traces are the first N tokens of a reasoning episode, cut
#   mid-thought. Choosing budgets by intuition ("GPT-2 rambles, give it less")
#   bakes that truncation in and makes T_bar -- which multiplies straight into
#   the realized-channel figure -- an artifact of the cap.
#
#   This script measures instead. For each model/task it generates a small
#   sample with a GENEROUS cap and a stop criterion on the answer marker, then
#   reports:
#     * completion rate      -- fraction that emitted an answer at all
#     * length percentiles   -- tokens-to-completion among those that finished
#     * a recommended cap    -- ceil(p99) with headroom, rounded
#   and emits ready-to-paste models.py entries.
#
#   Report the completion rate in the paper. A budget chosen from measured
#   completion is defensible; a budget chosen to make a headroom claim work is
#   not, and this is the difference between them.
#
# NOTE the stop criterion is the SAME answer marker E2/E4 parse ("Answer:"),
#   so "completed" here means exactly "parseable by the downstream analysis".
#
# USAGE
#   python e0e_pilot_budget.py --models gpt2-small llama-3.2-3b
#   python e0e_pilot_budget.py                       # all of models.py
#   python e0e_pilot_budget.py --n 40 --cap 512 --reasoning-cap 2048
# =============================================================================

from __future__ import annotations
import argparse, json, math, os, re

# ── per-task completion markers ──────────────────────────────────────────────
# A single "Answer:" regex is WRONG here. Only the arithmetic and counting
# few-shot prefixes demonstrate an "Answer:" line; DEDUCT_PREFIX ends its
# demonstrations with "Step 3: By transitivity, X -> Z. Yes." and never shows
# the word "Answer" at all. Scoring deduction against /Answer\s*:/ reports 0%
# completion for every model at every cap -- a measurement artifact that would
# have made the whole pilot uninterpretable for that task.
#
# Two signals count as "episode finished", whichever comes first:
#   ANSWER  -- the answer form the prefix actually demonstrates
#   NEXT    -- the model has started a NEW problem, which is the unambiguous
#              end of the current one for a few-shot completion prompt
# NEXT is what makes this robust to a model that ignores the demonstrated
# format (a live risk for the reasoning-distilled models, which are trained to
# emit their own structure and are being fed a raw completion prompt here).
DONE_RE = {
    'arithmetic': [('ANSWER', re.compile(
                       r"(?:\**\s*(?:Final\s+)?Answer\s*\**\s*:\s*\**\s*-?\d+"
                       r"|\\boxed\s*\{\s*-?\d+\s*\})", re.I)),
                   ('NEXT',   re.compile(r"\n\s*Problem\s*:"))],
    'counting':   [('ANSWER', re.compile(
                       r"(?:\**\s*(?:Final\s+)?Answer\s*\**\s*:\s*\**\s*-?\d+"
                       r"|\\boxed\s*\{\s*-?\d+\s*\})", re.I)),
                   ('NEXT',   re.compile(r"\n\s*Sequence\s*:"))],
    'deduction':  [('ANSWER', re.compile(r"Step\s*3\s*:[^\n]*?\b(?:Yes|No)\b")),
                   ('NEXT',   re.compile(r"\n\s*Premise\s*1\s*:"))],
    # SAME SHAPE AS arithmetic/counting, AND THAT IS THE ENTIRE POINT.
    #
    # This pattern used to hunt for a Yes/No within 160 characters of any
    # "Step N:", because format 1 of the prompt ended each demonstration with
    # the verdict inline ("Step 3: By transitivity, snurbs -> glurps. Yes.")
    # and models scattered it across layouts -- next line, "Step 4:", inside a
    # think block. Widening the regex raised the measured rate but never fixed
    # the cause: the task had no terminator, so a model with more to say simply
    # kept saying it and ran into the next problem's header.
    #
    # Format 2 gives it one. Each demonstration now closes with the verdict
    # alone on an "Answer:" line, exactly as arithmetic and counting close on
    # "Answer: <number>" -- the two tasks that complete at 96%. So the pattern
    # is theirs, transposed to a binary target, and needs none of the
    # window-and-lookahead machinery it replaced.
    #
    # `\**` tolerates "**Answer:** Yes", and the optional "Final" tolerates
    # standard step-by-step prompts that ask for "Final Answer: Yes". The R1
    # distills often format their final line in Markdown regardless of the
    # demonstrated form.
    #
    # `deduction` keeps the strict pattern. It is withdrawn, its published
    # numbers were computed under it, and changing it would silently move
    # figures nobody is going to re-derive.
    'balanced_deduction': [
        ('ANSWER', re.compile(
            r"\**\s*(?:Final\s+)?Answer\s*\**\s*:\s*\**\s*(?:Yes|No)\b",
            re.I)),
        ('NEXT',   re.compile(r"\n\s*Premise\s*1\s*:"))],
    # E45 cargo-drive. The answer is a PLACE NAME, so the numeric and Yes/No
    # patterns above cannot be reused. No NEXT marker: the few-shot variant
    # separates demonstrations with the `Streets:` header, and a model restating
    # the street map mid-drive would look like the next problem starting.
    'cargodrive': [
        ('ANSWER', re.compile(r"\**\s*(?:Final\s+)?Answer\s*\**\s*:\s*\**\s*"
                              r"[A-Za-z][A-Za-z_-]*", re.I))],
    # E46 cue paradigm: a lettered option, `Answer: (B)` or `Answer: B`.
    'cue': [
        ('ANSWER', re.compile(r"\**\s*(?:Final\s+)?Answer\s*\**\s*:\s*\**\s*"
                              r"\(?\s*[A-D]\s*\)?", re.I)),
        ('NEXT',   re.compile(r"\n\s*Problem\s*:"))],
}

STOP_STRINGS = {'arithmetic': ['\nProblem:'], 'counting': ['\nSequence:'],
                'deduction': ['\nPremise 1:'],
                'balanced_deduction': ['\nPremise 1:'],
                'cargodrive': ['\nStreets:'],
                'cue': ['\nProblem:']}


# ── extended-thinking condition ──────────────────────────────────────────────
# Different regime, different completion signal. Under a chat template the model
# ends its turn with EOS, so EOS is the primary terminator and needs no regex.
# The secondary signal catches a model that emits the answer but keeps going.
#
# A trace that never closes </think> is a NON-COMPLETION, not a long think: the
# model is still ruminating when the budget runs out. That distinction is
# reported separately because it is the diagnostic that says whether the model
# entered thinking mode and came back out of it.
THINK_ANSWER_RE = re.compile(r"</think>.*?Answer\s*:\s*\S+", re.S | re.I)
THINK_OPEN_RE = re.compile(r"<think>")
THINK_CLOSE_RE = re.compile(r"</think>")
# fallback for a model that reasons without emitting think tags at all
BARE_ANSWER_RE = re.compile(r"Answer\s*:\s*\S+", re.I)

# ── answer VALUE extraction ─────────────────────────────────────────────────
# The regexes above locate a completion; they do not read the value off it.
# THINK_ANSWER_RE ends in \S+ precisely so it can match "**Final Answer:**",
# where the number arrives several lines later inside \boxed{}. Anything scoring
# CORRECTNESS needs the value, and needs every form a model actually uses.
#
# The form that matters: reasoning-distilled models are trained to emit
# \boxed{...} for mathematics, and revert to it under the chat template even
# when the few-shot prefix demonstrates "Answer: N". Measured on the thinking
# pilot, DeepSeek-R1-Distill-7B writes \boxed on 28 of 30 arithmetic traces and
# "Answer: N" on 1 -- so a scorer that knows only the demonstrated form reports
# 5% parse for a model that is answering, correctly, almost every time.
BOXED_RE = re.compile(r"\\boxed\s*\{\s*(-?\d+)\s*\}")
ANSWER_NUM_RE = re.compile(
    r"\**\s*(?:Final\s+)?Answer\s*\**\s*:\s*\**\s*(-?\d+)", re.I)
YESNO_RE = re.compile(r"\b(Yes|No)\b")
# Format 2's committed form. Preferred over the bare YESNO_RE for the same
# reason ANSWER_NUM_RE is preferred over a loose integer: a trace mentions
# "yes" and "no" while working, and only the Answer line is a commitment.
YESNO_ANSWER_RE = re.compile(
    r"\**\s*(?:Final\s+)?Answer\s*\**\s*:\s*\**\s*(Yes|No)\b", re.I)


def extract_answer(task, text):
    """The value the model committed to, or None. One definition, used by the
    pilot and by the accuracy table.

    Two rules beyond "find a number":

    AFTER </think> ONLY, when a think block closed. Everything inside it is
    working, and working contains intermediate values -- scoring against those
    would credit a model for a number it went on to revise.

    LAST match, not first. A model that writes a candidate and then corrects
    itself has committed to the correction.
    """
    span = text
    m = THINK_CLOSE_RE.search(text)
    if m:
        span = text[m.end():]
    elif THINK_OPEN_RE.search(text):
        # Opened a block and never closed it: it is still thinking, and nothing
        # in there is a commitment.
        return None
    if task == 'balanced_deduction':
        # Format 2 commits on an "Answer:" line, so read that and nothing else.
        # The bare-Yes/No fallback stays for two cases: a format-1 trace being
        # re-scored offline by e30, and a model that answers in prose without
        # the header. Both are worse evidence than an Answer line, which is why
        # they are the fallback rather than the rule -- a trace says "yes" and
        # "no" many times while working, and only the header marks a
        # commitment.
        hits = YESNO_ANSWER_RE.findall(span) or YESNO_RE.findall(span)
        return hits[-1] if hits else None
    if task == 'deduction':
        # Withdrawn, and left exactly as its published numbers were computed.
        hits = YESNO_RE.findall(span)
        return hits[-1] if hits else None
    for rx in (ANSWER_NUM_RE, BOXED_RE):
        hits = rx.findall(span)
        if hits:
            return hits[-1]
    return None


def _tok_index_at_char(ids, tok, end_char):
    """Token index covering `end_char` in the decoded text (1-based count)."""
    acc = 0
    for i, t in enumerate(ids):
        acc += len(tok.decode([t]))
        if acc >= end_char:
            return i + 1
    return len(ids)


class GeneratedCompletionStop:
    """Stop a single-example generation once the generated suffix completes."""

    def __init__(self, tok, prompt_len, task, allow_next):
        self.tok = tok
        self.prompt_len = int(prompt_len)
        self.task = task
        self.allow_next = bool(allow_next)

    def __call__(self, input_ids, scores, **kwargs):
        import torch
        suffix = input_ids[0, self.prompt_len:].tolist()
        if not suffix:
            return torch.zeros(input_ids.shape[0], device=input_ids.device,
                               dtype=torch.bool)
        text = self.tok.decode(suffix, skip_special_tokens=False)
        end_c, _ = first_completion(self.task, text, allow_next=self.allow_next)
        matched = end_c is not None
        return torch.full((input_ids.shape[0],), matched,
                          device=input_ids.device, dtype=torch.bool)


def probe_think_tags(tok):
    """How do <think>/</think> tokenize, and do they SURVIVE decoding?

    This is the single most important preflight for the thinking condition.
    In Qwen3 and the R1 distills these tags are registered SPECIAL tokens, so
    decoding with skip_special_tokens=True deletes them. Detection would then
    report 0% think-block closure on traces that closed perfectly well, and --
    worse -- the BARE fallback would start firing on ruminating traces, because
    THINK_OPEN_RE finds nothing to suppress it. Both failures look like a model
    result rather than a decoding bug.

    Returns a dict and prints a verdict.
    """
    info = {}
    for tag in ('<think>', '</think>'):
        ids = tok.encode(tag, add_special_tokens=False)
        special = set(getattr(tok, 'all_special_ids', []) or [])
        info[tag] = {
            'ids': ids,
            'n_tokens': len(ids),
            'is_special': any(i in special for i in ids),
            'survives_skip_true': tag in tok.decode(ids, skip_special_tokens=True),
            'survives_skip_false': tag in tok.decode(ids, skip_special_tokens=False),
        }
    print("  think-tag probe:")
    for tag, d in info.items():
        flag = '' if d['survives_skip_false'] else '   *** NOT RECOVERABLE ***'
        print(f"    {tag:<10} ids={d['ids']}  n_tok={d['n_tokens']}  "
              f"special={d['is_special']}  "
              f"survives(skip=True)={d['survives_skip_true']}  "
              f"survives(skip=False)={d['survives_skip_false']}{flag}")
    stripped = [t for t, d in info.items() if not d['survives_skip_true']]
    if stripped:
        print(f"    -> {stripped} are stripped by skip_special_tokens=True. This "
              f"script detects\n"
              f"       on the skip_special_tokens=False decoding, so detection is "
              f"unaffected --\n"
              f"       but any downstream code that decodes with skip=True will "
              f"silently see no tags.")
    dead = [t for t, d in info.items() if not d['survives_skip_false']]
    if dead:
        raise RuntimeError(
            f"{dead} do not round-trip through this tokenizer at all, so the "
            f"thinking condition cannot be measured. Check the model/tokenizer "
            f"pairing before spending compute.")
    return info


def preview_imitative(key, task, records, n, max_chars=700):
    """Show the traces that did NOT produce an answer, in preference to the rest.

    WHY OUTCOME-ORDERED AND NOT FIRST-N

    `--preview` used to be wired only into the thinking branch, so an imitative
    run printed nothing at all and looked like a no-op. And showing examples
    0..N-1 is the wrong sample anyway: what a pilot needs explaining is the
    tail. On balanced_deduction five traces per model terminated on the NEXT
    marker -- they reached the following problem's header without emitting a
    Step 3 Yes/No -- and those five are the entire question of whether the task
    is usable. Printing the first eight successes answers nothing.

    Order: no marker at all (burned the cap), then NEXT (ran on into the next
    problem), then ANSWER. Within each group, first encountered.
    """
    rank = {None: 0, 'NEXT': 1, 'ANSWER': 2}
    ordered = sorted(enumerate(records),
                     key=lambda kv: (rank.get(kv[1].get('marker'), 0), kv[0]))
    bar = "\u2500" * 78
    counts = {}
    for r in records:
        counts[r.get('marker')] = counts.get(r.get('marker'), 0) + 1
    print(f"\n{bar}\n PREVIEW  {key} / {task}   markers: {counts}\n{bar}")
    print(" ANSWER = emitted a Step 3 Yes/No. NEXT = reached the next problem's")
    print(" header first, so no answer was produced. None = burned the cap.")
    for i, r in ordered[:n]:
        mk = r.get('marker') or 'NONE'
        print(f"\n  [{i}] marker={mk}  generated={r['n_generated']} tok  "
              f"hit_cap={r['hit_cap']}  episode_len={r['episode_len']}")
        stem = r['prompt'].rstrip().splitlines()[-2:]
        for ln in stem:
            print(f"    prompt| {ln}")
        body = r['text'][:max_chars]
        for ln in body.splitlines()[:14]:
            print(f"      gen| {ln}")
        if len(r['text']) > max_chars:
            print(f"      gen| ... ({len(r['text']) - max_chars} more chars)")


def preview_trace(key, task, i, prompt, ids, tok, raw, clean, end_c, which,
                  cap, max_chars=400):
    """Dump one trace with the segmentation the downstream analysis will use.

    The point is to make the provenance of the CoT visible BEFORE committing
    compute: which prompt produced it, where the think block opens and closes,
    how many tokens fall on each side, and exactly which positions E19 would
    keep under the pre-</think> specification.
    """
    bar = "\u2500" * 78
    print(f"\n{bar}\n PREVIEW  {key} / {task} / example {i}\n{bar}")

    ptoks = tok.encode(prompt, add_special_tokens=False)
    print(f" PROMPT  ({len(ptoks)} tokens)  -- tail:")
    tail = prompt[-max_chars:]
    for ln in tail.splitlines()[-12:]:
        print(f"   | {ln}")

    # token index where the think block closes, by the same rule E19 uses
    close_tok = None
    acc = ''
    for j, t in enumerate(ids):
        acc += tok.decode([int(t)], skip_special_tokens=False)
        if '</think>' in acc:
            close_tok = j
            break
    self_opened = '<think>' in raw
    tmpl_opened = prompt.rstrip().endswith('<think>')
    # Qwen3 with enable_thinking=False injects an already-CLOSED empty block into
    # the prompt. Nothing the model writes is think content, so labelling the
    # whole reply "think block" and reporting "still reasoning when generation
    # stopped" is doubly wrong.
    tmpl_disabled = '</think>' in prompt

    print(f"\n GENERATED  ({len(ids)} tokens, cap {cap})")
    if tmpl_disabled:
        print("   thinking      : DISABLED by the chat template (the prompt "
              "contains an\n"
              "                   already-closed empty <think></think> block), so "
              "everything\n"
              "                   below is the answer, not think content.")
        print(f"\n   +-- reply ({len(raw)} chars, first {max_chars}):")
        for ln in raw[:max_chars].splitlines()[:12]:
            print(f"   |  {ln}")
        print("   +--")
        print(f"\n COMPLETION: marker={which}  "
              f"{'(episode closed)' if which else '(NOT completed -- ran to cap)'}")
        print(" E19 uses the WHOLE reply for this condition -- there is no think "
              "block to\n   restrict to, and the reasoning is the visible answer.")
        print(bar)
        return
    if tmpl_opened:
        print(f"   <think> opened : yes, BY THE CHAT TEMPLATE (the prompt ends with"
              f" '<think>',\n"
              f"                    so generation starts inside the block and the"
              f" model never\n"
              f"                    emits the opening tag itself)")
    else:
        print(f"   <think> opened : {self_opened}  (emitted by the model)")
    if close_tok is not None:
        print(f"   </think> closed: yes, at token {close_tok} "
              f"({close_tok} think tokens, {len(ids)-close_tok} after)")
    else:
        print(f"   </think> closed: NO -- still reasoning when generation stopped")

    think_txt = raw.split('</think>')[0].replace('<think>', '').strip()
    after_txt = raw.split('</think>', 1)[1].strip() if close_tok is not None else ''
    print(f"\n   +-- think block ({len(think_txt)} chars, first {max_chars}):")
    for ln in think_txt[:max_chars].splitlines()[:10]:
        print(f"   |  {ln}")
    if close_tok is not None:
        print(f"   +-- after </think> ({len(after_txt)} chars):")
        for ln in after_txt[:200].splitlines()[:6]:
            print(f"   |  {ln}")
    print("   +--")

    print(f"\n COMPLETION: marker={which}  "
          f"{'(episode closed)' if which else '(NOT completed -- ran to cap)'}")
    if close_tok is not None:
        print(f" E19 PRIMARY SPEC would KEEP tokens 0..{close_tok-1} "
              f"({close_tok} positions) and DROP the {len(ids)-close_tok} tokens "
              f"after </think>,")
        print(f"   because the answer is usually stated verbatim there and would "
              f"close the gap for free.")
    else:
        print(" E19 PRIMARY SPEC would REJECT this trace (no closing tag).")
    print(bar)


def think_completion(text, hit_eos):
    """(end_char, marker) for the thinking condition, or (None, None).

    Priority: EOS (the model ended its turn) > answer-after-</think> > bare
    answer. Returns the END of the trace for EOS since the whole turn is the
    episode.
    """
    m = THINK_ANSWER_RE.search(text)
    if m:
        return m.end(), 'ANSWER'
    if hit_eos:
        return len(text), 'EOS'
    # BARE is a fallback for a model that reasons WITHOUT think tags at all. It
    # must not fire inside an unclosed <think> block: a ruminating model that
    # writes "Answer: 16 ... but wait, let me reconsider" has not finished, and
    # scoring it complete would inflate the completion rate with exactly the
    # traces this diagnostic exists to identify.
    if not THINK_OPEN_RE.search(text):
        m = BARE_ANSWER_RE.search(text)
        if m:
            return m.end(), 'BARE'
    return None, None


def think_stats(text, prompt=None):
    """Think/answer split, accounting for a chat template that PRE-OPENS <think>.

    DeepSeek-R1's template ends the assistant turn opener with a literal
    `<think>`, so generation begins INSIDE the block and the model never emits
    the opening tag itself. Testing only the generated text therefore reports
    think_opened=0% on traces that are entirely think content -- a false alarm
    that looks exactly like "the model is not entering thinking mode".

    The block is open at step 0 if the prompt ends with the opening tag; that is
    template_opened, and it counts as opened.
    """
    self_opened = bool(THINK_OPEN_RE.search(text))
    template_opened = bool(prompt is not None
                           and prompt.rstrip().endswith('<think>'))
    # Qwen3 with enable_thinking=False injects an ALREADY-CLOSED empty block
    # (`<think>\n\n</think>`) into the prompt. Nothing about the think block is
    # then measurable in the generated text, and treating think_closed=False as
    # a failure would flag every trace of a condition that has no think block by
    # design.
    template_disabled = bool(prompt is not None and '</think>' in prompt)
    mc = THINK_CLOSE_RE.search(text)
    return {'think_opened': self_opened or template_opened,
            'think_self_opened': self_opened,
            'think_template_opened': template_opened,
            'think_disabled_by_template': template_disabled,
            'think_closed': bool(mc),
            'think_chars': mc.start() if mc else (len(text) if
                                                  (self_opened or template_opened)
                                                  else 0),
            'answer_chars': len(text) - mc.end() if mc else 0}


def first_completion(task, text, allow_next=True):
    """(end_char, which_marker) of the earliest completion signal, or (None, None)."""
    hits = [(m.end(), tag) for tag, rx in DONE_RE[task]
            if allow_next or tag != 'NEXT'
            for m in [rx.search(text)] if m]
    return min(hits) if hits else (None, None)
# models whose traces are extended-thinking; they get the larger pilot cap
REASONING_HINTS = ('deepseek', 'qwen3', 'r1')


from p2_common import load_tokenizer   # 3-tier loader, see p2_common


def is_reasoning(key: str) -> bool:
    k = key.lower()
    return any(h in k for h in REASONING_HINTS)


def pct(xs, q):
    if not xs:
        return None
    xs = sorted(xs)
    i = min(len(xs) - 1, max(0, int(math.ceil(q / 100 * len(xs))) - 1))
    return xs[i]


def round_cap(n, reasoning):
    """Round a raw token count up to a tidy cap with headroom."""
    if n is None:
        return None
    target = int(math.ceil(n * 1.25))          # 25% headroom over p99
    grid = ([128, 192, 256, 384, 512, 768, 1024, 1536, 2048] if not reasoning
            else [1024, 1536, 2048, 3072, 4096, 6144, 8192, 12288, 16384])
    for g in grid:
        if target <= g:
            return g
    # Falling off the end means p99 exceeded the largest grid value. Clamping here
    # would silently reproduce the truncation this script exists to detect, so
    # return the raw target and let the caller see an off-grid number.
    print(f"  !! p99-derived target {target} exceeds the grid (max {grid[-1]}). "
          f"Returning {target} unrounded — the measurement itself may be censored; "
          f"check the completion rate before using this.")
    return target


def next_cap_after(n, reasoning):
    """Next tidy pilot cap above a censored cap."""
    grid = ([128, 192, 256, 384, 512, 768, 1024, 1536, 2048] if not reasoning
            else [1024, 1536, 2048, 3072, 4096, 6144, 8192, 12288, 16384])
    for g in grid:
        if g > n:
            return g
    return int(math.ceil(n * 1.5))


def _show_prompts(args, keys):
    """Render the prompts and exit. No model weights, no GPU, no generation.

    The cheapest possible check that the thinking condition is actually asking
    what you think it is: the chat template is applied by the TOKENIZER, so this
    shows the real rendered string -- role markers, generation prompt, any
    template-injected think scaffold -- without loading a single weight.

    Run this before every new model. A chat template that silently changed
    between transformers releases is indistinguishable from a model effect once
    the numbers are in.
    """
    from p2_common import load_tokenizer
    from models import MODELS
    bar = "\u2500" * 78
    for key in keys:
        cfg = MODELS[key]
        print(f"\n{bar}\n {key}   ({cfg['hf']})\n{bar}")
        tok = load_tokenizer(cfg['hf'])

        print(f"  REGIME          : {args.regime.upper()}")
        if args.regime == 'imitative':
            print("                    few-shot completion prompts")
        elif args.regime == 'stepbystep':
            print("                    zero-shot problem-only step-by-step prompts")
        if args.think:
            from e2_prompts_think import (build_think_prompts, INSTR,
                                          supports_thinking_toggle,
                                          has_chat_template, prompt_provenance)
            print(f"  source          : e2_prompts_think.build_think_prompts()")
            print(f"  chat template   : {has_chat_template(tok)}")
            print(f"  thinking toggle : {supports_thinking_toggle(tok)}"
                  + ("" if supports_thinking_toggle(tok)
                     else "   (enable_thinking is a no-op here)"))
            try:
                probe_think_tags(tok)
            except Exception as e:
                print(f"  !! tag probe failed: {e}")
            want = not args.no_enable_thinking
            prompts = build_think_prompts(tok, n=2, enable_thinking=want)
            prov = prompt_provenance(tok, enable_thinking=want)
            print(f"  template sha1   : {prov['chat_template_sha1']}")
        elif args.regime == 'stepbystep':
            from e2_prompts_stepbystep import (INSTR, build_stepbystep_prompts,
                                               prompt_provenance)
            prov = prompt_provenance(tasks=args.tasks)
            print(f"  source          : e2_prompts_stepbystep.build_stepbystep_prompts()")
            print(f"  template        : {prov['template']}")
            prompts = build_stepbystep_prompts(n=2, tasks=args.tasks)
        else:
            from e2_prompts import build_prompts
            print(f"  source          : e2_prompts.build_prompts()  "
                  f"(few-shot completion, no chat template)")
            prompts = build_prompts(n=2, tasks=args.tasks)

        for task in args.tasks:
            pr = prompts[task][0]
            ids = tok.encode(pr, add_special_tokens=False)
            print(f"\n  --- {task}  ({len(ids)} prompt tokens) ---")
            if args.think or args.regime == 'stepbystep':
                print(f"      instruction: {INSTR[task]}")
            print(f"      rendered prompt (repr, so template markers are visible):")
            body = pr if len(pr) < 900 else pr[:300] + "\n   ...[middle elided]...\n" + pr[-560:]
            for ln in body.splitlines():
                print(f"      | {ln}")
            print(f"      LAST 3 TOKENS the model sees before generating: "
                  f"{[tok.decode([i], skip_special_tokens=False) for i in ids[-3:]]!r}")

    print(f"\n{bar}")
    print(" Nothing was generated and no weights were loaded.")
    print(" Check that the prompt ends where generation should begin, that the")
    print(" instruction is present, and that the think scaffold (if any) matches")
    print(" what you expect. Then re-run with --preview 2 to see real traces.")
    print(bar)


def _prompt_formats():
    """Prompt-format generation per task, for the trace stamp.

    Only tasks whose prompts have been REVISED after traces existed need an
    entry. The three original generators have never changed and carry no
    version, so absence here means "the one and only format".
    """
    try:
        from e2_prompts import BALANCED_DEDUCTION_FORMAT
    except ImportError:                                        # noqa: BLE001
        return {}
    return {'balanced_deduction': BALANCED_DEDUCTION_FORMAT}


def _save_traces(args, key, cfg, all_records, per_task, tok):
    """Write the generated traces in the E1 cache LAYOUT so inspect_traces.py can
    read them -- but never under an E1 cache NAME.

    inspect_traces.py expects {'raw_outputs': {task: [{'ids': [...]}, ...]}} and
    aligns record i with example i from the shared generators, which holds here
    because the pilot draws the first `--n` examples from those same
    deterministic generators.

    The filename is deliberately NOT e1_2_<slug>_bw_outputs.pkl. A pilot cache
    carrying a real cache's name is one stray copy away from being picked up by
    E2 as though it were the full generation run; a distinct stem plus
    metadata['kind'] makes that impossible to do by accident.

    Records also carry the rendered PROMPT and the decoded text. Non-imitative
    regimes do not use the few-shot generators' `cot_prompt`, so reconstructing
    prompts later from the default generators would display the wrong thing.
    """
    import pickle, datetime
    regime = args.regime
    slug = cfg['label'].replace('-', '_')
    out_dir = os.path.dirname(os.path.abspath(args.out)) or '.'
    path = os.path.join(out_dir, f'e0e_traces_{slug}_{regime}.pkl')

    blob = {
        'kind': 'e0e_pilot_traces',          # NOT an E1 generation cache
        'metadata': {
            'kind': 'e0e_pilot_traces',
            'model': cfg['label'], 'hf': cfg['hf'], 'regime': regime,
            'enable_thinking': (not args.no_enable_thinking) if args.think else None,
            'n_per_task': args.n,
            # Prompt-format generation, stamped so traces from two formats can
            # never be pooled or re-scored under each other's pattern. Absent in
            # the six format-1 pilots already on disk, which is how e30 tells
            # them apart -- see BALANCED_DEDUCTION_FORMAT in e2_prompts.
            'prompt_format': _prompt_formats(),
            'pilot_cap': {t: per_task[t]['pilot_cap'] for t in per_task},
            'sampler': {'top_k': 0, 'top_p': 1.0, 'temperature': 1.0},
            'created': datetime.datetime.now().isoformat(timespec='seconds'),
            'note': 'pilot traces for inspection only -- NOT the E1 generation '
                    'cache. Do not feed these to E2.',
        },
        'raw_outputs': all_records,
        'summary': dict(per_task),
    }
    with open(path, 'wb') as f:
        pickle.dump(blob, f)
    n = sum(len(v) for v in all_records.values())
    print(f"\n  wrote {os.path.basename(path)}  "
          f"({n} traces, {os.path.getsize(path)/1e6:.1f} MB)")
    print(f"    inspect:  python inspect_traces.py --pkl {path} \\")
    print(f"                --model-label {cfg['label']} --hf {cfg['hf']} "
          f"--task {args.tasks[0]} --n 5")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--models', nargs='*', default=None, help='models.py keys')
    ap.add_argument('--tasks', nargs='+',
                    choices=['arithmetic', 'deduction', 'counting',
                             'balanced_deduction'],
                    default=['arithmetic', 'deduction', 'counting'],
                    help='tasks to pilot; default is the original three')
    ap.add_argument('--n', type=int, default=30, help='examples per model-task')
    ap.add_argument('--cap', type=int, default=512, help='pilot cap, base models')
    ap.add_argument('--reasoning-cap', type=int, default=2048)
    ap.add_argument('--n-examples', type=int, default=500,
                    help='n_examples to write into the emitted models.py entries')
    ap.add_argument('--out', default='pilot_budget.json')
    ap.add_argument('--device', default=None)
    ap.add_argument('--dtype', choices=('float32', 'float16', 'bfloat16'),
                    default=None,
                    help='override models.py dtype for all selected models')
    ap.add_argument('--no-save-traces', action='store_true',
                    help='skip writing the trace pkl (saved by default so the '
                         'generated text can be inspected with inspect_traces.py)')
    ap.add_argument('--regime', choices=('imitative', 'thinking', 'thinking-off',
                                         'stepbystep'), default=None,
                    help='elicitation regime to pilot. Preferred over legacy '
                         '--think; use --regime stepbystep for the zero-shot '
                         'step-by-step prompts.')
    ap.add_argument('--think', action='store_true',
                    help='extended-thinking condition: chat template, zero-shot, '
                         'native reasoning mode. Reasoning-distilled models only. '
                         'Legacy alias for --regime thinking.')
    ap.add_argument('--show-prompts', action='store_true',
                    help='render and print the prompts, then exit. No model load, '
                         'no generation, zero GPU -- verifies prompt construction '
                         'before any compute is spent.')
    ap.add_argument('--preview', type=int, default=0, metavar='N',
                    help='dump N traces per model-task in full. Traces are '
                         'ALSO written to e0e_traces_<slug>_<regime>.pkl by '
                         'default (see --no-save-traces), so a question about '
                         'one need not cost a re-run. Imitative: '
                         'ordered by OUTCOME -- cap-burning first, then '
                         'NEXT-terminated (no answer), then answered -- '
                         'because the failures are the sample worth reading. '
                         'Under --think: the first N, with the think/answer '
                         'split.')
    ap.add_argument('--preview-only', action='store_true',
                    help='with --preview, stop after the previews instead of '
                         'running the whole pilot')
    ap.add_argument('--no-enable-thinking', action='store_true',
                    help='with --think, request thinking OFF (Qwen3 only). Gives '
                         'an on/off contrast at identical weights.')
    args = ap.parse_args()

    legacy_regime = ('thinking-off' if (args.think and args.no_enable_thinking)
                     else 'thinking' if args.think else 'imitative')
    if args.regime is None:
        args.regime = legacy_regime
    elif args.think and args.regime not in ('thinking', 'thinking-off'):
        raise SystemExit('--think cannot be combined with '
                         f'--regime {args.regime!r}')
    elif args.no_enable_thinking and args.regime == 'thinking':
        raise SystemExit('--no-enable-thinking is equivalent to '
                         '--regime thinking-off; use that spelling.')

    if args.regime == 'thinking-off':
        args.think = True
        args.no_enable_thinking = True
    elif args.regime == 'thinking':
        args.think = True
        args.no_enable_thinking = False
    elif args.no_enable_thinking:
        raise SystemExit('--no-enable-thinking applies only to thinking-off')
    else:
        args.think = False
        args.no_enable_thinking = False

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteriaList
    from models import MODELS
    from e2_prompts import build_prompts

    keys = args.models or list(MODELS)
    if args.think:
        from models import THINKING_MODELS
        bad = [k for k in keys if k not in THINKING_MODELS]
        if bad:
            raise SystemExit(
                f"--think requested for models with no thinking condition: {bad}\n"
                f"  the base models have no chat template and no extended-reasoning "
                f"mode; only {THINKING_MODELS} do.")
    if args.show_prompts:
        _show_prompts(args, keys)
        return

    dev = args.device or ('cuda' if torch.cuda.is_available() else
                          'mps' if torch.backends.mps.is_available() else 'cpu')
    # imitative prompts are model-independent; thinking prompts need the
    # tokenizer's chat template, so they are built per model inside the loop
    # tasks=args.tasks, NOT the default. build_prompts() with no task list
    # returns only the original three, so `--tasks balanced_deduction` produced
    # a KeyError at the first indexing of prompts[task] rather than anything
    # that named the cause.
    if args.think:
        prompts = None
    elif args.regime == 'stepbystep':
        from e2_prompts_stepbystep import build_stepbystep_prompts
        prompts = build_stepbystep_prompts(n=args.n, tasks=args.tasks)
    else:
        prompts = build_prompts(n=args.n, tasks=args.tasks)
    # ── regime banner ────────────────────────────────────────────────────────
    # Printed unconditionally and BEFORE any work. The regime used to be
    # announced only inside the thinking branch, so an imitative run said
    # nothing at all -- and a --think flag that failed to reach the script
    # (a stale notebook cell, a forgotten re-run) was indistinguishable from a
    # correct imitative run. Now the prompt source is always stated.
    _regime = ('THINKING-OFF (chat template, thinking explicitly disabled)'
               if args.regime == 'thinking-off' else
               'THINKING (chat template, zero-shot)' if args.regime == 'thinking' else
               'STEPBYSTEP (zero-shot problem-only step-by-step prompts)'
               if args.regime == 'stepbystep' else
               'IMITATIVE (few-shot completion prompts)')
    _source = ('e2_prompts_think.build_think_prompts()' if args.think else
               'e2_prompts_stepbystep.build_stepbystep_prompts()'
               if args.regime == 'stepbystep' else
               'e2_prompts.build_prompts()')
    print("=" * 78)
    print(f" REGIME : {_regime}")
    print(f" PROMPTS: {_source}")
    if args.regime == 'imitative':
        print(" NOTE   : if you meant another elicitation condition, --regime did")
        print("          NOT select it. Check the command line above.")
    print("=" * 78)

    results, recommended = {}, {}

    for key in keys:
        cfg = MODELS[key]
        reasoning = is_reasoning(key)
        cap = args.reasoning_cap if reasoning else args.cap
        from models import resolve_torch_dtype
        dtype_name = args.dtype or cfg['dtype']
        dtype = resolve_torch_dtype(dtype_name)
        print(f"\n=== {key}  ({cfg['hf']}, {dtype_name}, pilot cap {cap}) ===")
        tok = load_tokenizer(cfg['hf'])
        model = AutoModelForCausalLM.from_pretrained(
            cfg['hf'], torch_dtype=dtype, trust_remote_code=True).to(dev).eval()
        torch.manual_seed(42)

        if args.think:
            from e2_prompts_think import (build_think_prompts,
                                          supports_thinking_toggle)
            want = not args.no_enable_thinking
            if args.no_enable_thinking and not supports_thinking_toggle(tok):
                raise SystemExit(
                    f"{key}: --no-enable-thinking requested but this chat template "
                    f"has no thinking toggle, so the 'off' run would be identical "
                    f"to the 'on' run. Only Qwen3 gates thinking this way.")
            prompts = build_think_prompts(tok, n=args.n, enable_thinking=want)
            print(f"  regime: thinking (enable_thinking={want}, "
                  f"toggle_supported={supports_thinking_toggle(tok)})")
            probe_think_tags(tok)
            print(f"  prompt source: e2_prompts_think.build_think_prompts() -> "
                  f"chat template of {cfg['hf']}")
            print(f"  first rendered prompt, tail:")
            for _ln in prompts[args.tasks[0]][0][-220:].splitlines()[-6:]:
                print(f"    | {_ln}")

        per_task, all_records = {}, {}
        for task in args.tasks:
            lens, completed, marker_counts, think_diag = [], 0, {}, {}
            hit_cap_n = 0
            think_lens = []
            task_records = []
            t_task = __import__('time').time()
            print(f"  {task} ", end='', flush=True)
            for ei, p in enumerate(prompts[task][:args.n]):
                inp = tok(p, return_tensors='pt').to(dev)
                sl = inp.input_ids.shape[1]
                kw = dict(max_new_tokens=cap, do_sample=True, temperature=1.0,
                          top_k=0, top_p=1.0, pad_token_id=tok.eos_token_id)
                if args.think:
                    # EOS is the terminator under a chat template; no stop
                    # strings, and eos_token_id left at the model default so
                    # end-of-turn halts generation.
                    out = model.generate(**inp, **kw)
                    ids = out[0, sl:].tolist()
                    # Detect on the UNSTRIPPED decoding: <think>/</think> are
                    # special tokens in these models, so skip_special_tokens=True
                    # deletes them and every trace would look like it never
                    # opened a think block. See probe_think_tags().
                    text = tok.decode(ids, skip_special_tokens=False)
                    clean = tok.decode(ids, skip_special_tokens=True)
                    hit_eos = (len(ids) < cap)
                    hit_cap_n += int(len(ids) >= cap)
                    end_c, which = think_completion(text, hit_eos)
                    ts = think_stats(text, prompt=p)
                    if ei < args.preview:
                        preview_trace(key, task, ei, p, ids, tok, text, clean,
                                      end_c, which, cap)
                    for k_, v_ in ts.items():
                        if isinstance(v_, bool):
                            think_diag[k_] = think_diag.get(k_, 0) + int(v_)
                    # Token index where </think> closes. E19's primary
                    # specification uses ONLY these positions, so this -- not the
                    # length to 'Answer:' -- is the trace length that determines
                    # the channel in the thinking condition.
                    think_tok = None
                    if ts['think_closed']:
                        _m = THINK_CLOSE_RE.search(text)
                        think_tok = _tok_index_at_char(ids, tok, _m.start())
                        think_lens.append(think_tok)
                    ep_len = None
                    if end_c is not None:
                        completed += 1
                        marker_counts[which] = marker_counts.get(which, 0) + 1
                        ep_len = _tok_index_at_char(ids, tok, end_c)
                        lens.append(ep_len)
                        print('.', end='', flush=True)
                    else:
                        print('X', end='', flush=True)
                    task_records.append({
                        'ids': ids, 'prompt': p, 'text': text, 'text_clean': clean,
                        'completed': end_c is not None, 'marker': which,
                        'episode_len': ep_len, 'think_tokens': think_tok,
                        'n_generated': len(ids),
                        'hit_cap': len(ids) >= cap, **ts})
                    continue
                if args.regime == 'imitative':
                    try:
                        # Few-shot completion prompts can end by starting the
                        # next demonstration problem. Zero-shot step-by-step
                        # prompts cannot use that marker: "Problem:" or
                        # "Sequence:" may simply be the model restating the
                        # single problem it was asked to solve.
                        out = model.generate(**inp,
                                             stop_strings=STOP_STRINGS[task],
                                             tokenizer=tok, **kw)
                    except Exception:
                        # older transformers (no stop_strings), or a slow tokenizer
                        # that StopStringCriteria cannot use. Falls back to running
                        # to the cap; completion is still detected by parsing.
                        out = model.generate(**inp, **kw)
                else:
                    if args.regime == 'stepbystep':
                        kw['stopping_criteria'] = StoppingCriteriaList([
                            GeneratedCompletionStop(
                                tok, sl, task, allow_next=False)
                        ])
                    out = model.generate(**inp, **kw)
                ids = out[0, sl:].tolist()
                hit_cap_n += int(len(ids) >= cap)
                text = tok.decode(ids)
                end_c, which = first_completion(
                    task, text, allow_next=(args.regime == 'imitative'))
                if end_c is not None:
                    completed += 1
                    marker_counts[which] = marker_counts.get(which, 0) + 1
                    # tokens up to and including the answer marker
                    pieces = [tok.decode([t]) for t in ids]
                    ce = 0
                    for i, piece in enumerate(pieces):
                        ce += len(piece)
                        if ce >= end_c:
                            lens.append(i + 1)
                            break
                    else:
                        lens.append(len(ids))
                    print('.', end='', flush=True)
                else:
                    # never emitted the answer marker -> burned the whole cap.
                    # These dominate wall-clock: each costs `cap` tokens.
                    print('X', end='', flush=True)
                task_records.append({
                    'ids': ids, 'prompt': p, 'text': text,
                    'completed': end_c is not None, 'marker': which,
                    'episode_len': lens[-1] if (end_c is not None and lens) else None,
                    'n_generated': len(ids), 'hit_cap': len(ids) >= cap})
            print(f"  ({__import__('time').time() - t_task:.0f}s)")
            if args.preview:
                preview_imitative(key, task, task_records, args.preview)
            if args.preview and args.preview_only:
                print("\n--preview-only: stopping before the full pilot. Re-run "
                      "without it once the traces look right.")
                return
            rate = completed / max(args.n, 1)
            censored = rate < 0.95
            rec = {'n': args.n, 'completion_rate': rate, 'markers': marker_counts,
                   'censored': censored,
                   'regime': args.regime,
                   'p50': pct(lens, 50), 'p90': pct(lens, 90),
                   'p99': pct(lens, 99), 'max': max(lens) if lens else None,
                   'pilot_cap': cap,
                   'hit_cap_frac': hit_cap_n / max(args.n, 1),
                   'noncompletion_frac': 1.0 - rate}
            if args.think:
                rec['think'] = {
                    'opened_frac': think_diag.get('think_opened', 0) / max(args.n, 1),
                    'self_opened_frac': think_diag.get('think_self_opened', 0) / max(args.n, 1),
                    'template_opened_frac': think_diag.get('think_template_opened', 0) / max(args.n, 1),
                    'disabled_by_template_frac': think_diag.get('think_disabled_by_template', 0) / max(args.n, 1),
                    'closed_frac': think_diag.get('think_closed', 0) / max(args.n, 1),
                    # length of the think block alone -- the quantity E19's
                    # primary specification actually operates on
                    'think_p50': pct(think_lens, 50), 'think_p90': pct(think_lens, 90),
                    'think_p99': pct(think_lens, 99),
                    'think_max': max(think_lens) if think_lens else None}
                tk = rec['think']
                print(f"  {task:<11} think block: p50={tk['think_p50']} "
                      f"p90={tk['think_p90']} p99={tk['think_p99']}  "
                      f"(vs {rec['p50']} to the answer marker)")
                print(f"              -> E19's primary spec uses the THINK BLOCK only, "
                      f"so that is the trace length that sets the channel.")
            per_task[task] = rec
            all_records[task] = task_records
            print(f"  {task:<11} completed {completed}/{args.n} ({rate:5.1%})  "
                  f"p50={rec['p50']}  p90={rec['p90']}  p99={rec['p99']}  "
                  f"max={rec['max']}  markers={marker_counts}")
            if censored:
                print(f"  !! {1-rate:.0%} did not reach a completion marker "
                      f"within the {cap}-token pilot cap. p99={rec['p99']} is a "
                      f"LOWER BOUND on the true p99 -- the distribution is "
                      f"right-censored. Re-run this model with a larger cap "
                      f"before trusting the recommendation.")
        if not args.no_save_traces:
            _save_traces(args, key, cfg, all_records, per_task, tok)

        results[key] = per_task

        p99s = [v['p99'] for v in per_task.values() if v['p99']]
        worst = max(p99s) if p99s else None
        cap_rec = round_cap(worst, reasoning) or cap
        any_censored = any(v['censored'] for v in per_task.values())
        if any_censored and cap_rec <= cap:
            cap_rec = next_cap_after(cap, reasoning)
        low = min((v['completion_rate'] for v in per_task.values()), default=0)
        recommended[key] = {'cot_tokens': cap_rec, 'n_examples': args.n_examples,
                            'worst_p99': worst, 'min_completion_rate': low,
                            'censored': any_censored}
        print(f"  -> worst-task p99 = {worst}; recommended cot_tokens = {cap_rec}")
        if any_censored:
            print("     recommendation is a next pilot cap, not a final budget, "
                  "because at least one task was censored.")
        if low < 0.8:
            print(f"  !! completion rate as low as {low:.0%} — this model often "
                  "failed to emit an answer within this pilot cap. Inspect the "
                  "saved traces before deciding whether this is non-completion "
                  "or a budget that is too small.")
        del model
        if dev == 'cuda':
            torch.cuda.empty_cache()

    json.dump({'per_task': results, 'recommended': recommended},
              open(args.out, 'w'), indent=2)
    print(f"\nwrote {args.out}")

    print("\n" + "=" * 78)
    print(" SUGGESTED models.py ENTRIES (paste over the existing dict values)")
    print("=" * 78)
    for key, rec in recommended.items():
        c = MODELS[key]
        print(f'    "{key}": {{"label": "{c["label"]}", "hf": "{c["hf"]}", '
              f'"dtype": "{c["dtype"]}", "n_examples": {rec["n_examples"]}, '
              f'"cot_tokens": {rec["cot_tokens"]}}},')
    print()
    print("  n_examples is uniform by design: a single value removes the")
    print("  budget-confound argument from the paper entirely. cot_tokens differs")
    print("  by model because it is set from MEASURED completion length -- state")
    print("  that in the methods, together with the completion rates above.")
    rates = {k: v['min_completion_rate'] for k, v in recommended.items()}
    if rates:
        worst_k = min(rates, key=rates.get)
        print(f"\n  lowest completion rate: {worst_k} at {rates[worst_k]:.0%}")


if __name__ == '__main__':
    main()
