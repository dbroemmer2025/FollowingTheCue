#!/usr/bin/env python3
# =============================================================================
# inspect_traces.py — spot-check generated CoT traces against ground truth.
#
# Diagnoses accuracy disconnects (e.g. Qwen/DeepSeek arithmetic ~0% with
# parse_rate=1.0): decodes the cached generated traces and shows, per example,
#   * the problem and the true answer,
#   * the decoded model trace (truncated unless --full),
#   * what several candidate parsers extract (final "Answer: N", the last "=N"),
#   * whether the trace even reached an "Answer:" span (truncation check),
# then prints summary stats over ALL examples so you can tell whether the issue
# is TRUNCATION (few traces reach "Answer:") or a PARSE-TARGET bug (the parser
# grabs an intermediate "=v" instead of the final answer).
#
# Reads e1_2_<slug>_bw_outputs.pkl (the E1 generation cache; gitignored, so run
# this where that file lives). Prompts/answers are rebuilt from the E2 generators
# (same seed/order), so example i aligns with raw_outputs[task][i].
#
# Example:
#   python inspect_traces.py --drive-dir . --model qwen3-14b \
#          --task arithmetic --n 10
#   python inspect_traces.py --drive-dir . --model gpt2-small \
#          --task arithmetic --regime stepbystep --full --n 3
# =============================================================================
import os, sys, re, pickle, argparse, importlib.util


def show_block(label, text, raw=False, indent='      ', width=100):
    """Print a prompt or trace as the model actually wrote it.

    These were printed with `!r`, which renders every newline as a literal
    \n and puts a 600-character trace on one unreadable line. The content of
    a trace is the thing being inspected, so it has to look like text: the
    label on its own line, then the body indented, with real line breaks and
    long lines wrapped.

    `raw=True` restores the repr for the cases where the WHITESPACE is the
    question -- a stray carriage return, a double space, a trailing newline
    that shifts tokenization. Those are real debugging needs, so the old
    behaviour is kept behind a flag rather than deleted.
    """
    if text is None:
        text = ''
    if raw:
        print(f"\n{indent}{label.upper()}: {text!r}")
        return
    print(f"\n{indent}{label.upper()}:")
    body = indent + '  '
    avail = max(20, width - len(body))
    for line in str(text).split('\n'):
        if not line:
            print(body.rstrip())
            continue
        while len(line) > avail:
            cut = line.rfind(' ', 0, avail)
            cut = cut if cut > 0 else avail
            print(body + line[:cut])
            line = line[cut:].lstrip()
        print(body + line)


def _load_prompt_examples(code_dir, task, n, regime='imitative', blob=None):
    """Return the examples aligned with the cache, embedded ones first.

    A cache may CARRY its own examples under `blob['examples'][task]`. When it
    does, use them: alignment is then a property of the file rather than of
    re-running a generator at the same seed and version, which is the one
    assumption in this script that has silently broken before. E45 caches always
    embed, because their episodes come from a frozen pack and there is no
    generator to re-run.

    The inspector used to import the full E2 runner only to read its
    `task_examples` global. That made a display utility depend on E2's CLI
    initialization. `e2_prompts.py` is the actual source of the examples, and it
    supports opt-in tasks such as balanced_deduction directly.
    """
    embedded = None
    if isinstance(blob, dict):
        embedded = (blob.get('examples') or {}).get(task)
    if embedded:
        return embedded

    if regime == 'stepbystep':
        path = os.path.join(code_dir, 'e2_prompts_stepbystep.py')
        spec = importlib.util.spec_from_file_location('e2prompts_stepbystep', path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        examples = mod.build_stepbystep_examples(n=n, tasks=[task])[task]
        prompts = mod.build_stepbystep_prompts(n=n, tasks=[task])[task]
        for ex, prompt in zip(examples, prompts):
            ex['cot_prompt'] = prompt
        return examples

    path = os.path.join(code_dir, 'e2_prompts.py')
    spec = importlib.util.spec_from_file_location('e2prompts', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.build_examples(n=n, tasks=[task])[task]


def _load_model_registry(code_dir):
    """Import models.py without relying on the caller's cwd."""
    path = os.path.join(code_dir, 'models.py')
    spec = importlib.util.spec_from_file_location('models_registry', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.MODELS


def _resolve_model_args(args, parser):
    """Fill model_label and hf from models.py when possible.

    Preferred use is `--model <models.py key>`. The older
    `--model-label ... --hf ...` form remains available for ad hoc/local caches.
    """
    models = None

    def registry():
        nonlocal models
        if models is None:
            models = _load_model_registry(args.code_dir)
        return models

    if args.model:
        table = registry()
        if args.model not in table:
            parser.error(f"--model {args.model!r} not in models.py. "
                         f"Have: {sorted(table)}")
        cfg = table[args.model]
        args.model_label = args.model_label or cfg["label"]
        args.hf = args.hf or cfg["hf"]
        return args

    if args.model_label and not args.hf:
        table = registry()
        cfg = table.get(args.model_label)
        if cfg is None:
            cfg = next((v for v in table.values()
                        if v.get("label") == args.model_label), None)
        if cfg is not None:
            args.hf = cfg["hf"]

    if not args.model_label or not args.hf:
        parser.error("provide --model <models.py key>, or provide both "
                     "--model-label and --hf for an ad hoc cache")

    return args


def _normalize_regime(regime):
    """Accept user-facing regime names and legacy filename suffixes."""
    if regime is None:
        return 'imitative'
    aliases = {
        '': 'imitative',
        'imitative': 'imitative',
        'think': 'think',
        'thinking': 'think',
        '_think': 'think',
        'thinkoff': 'thinkoff',
        'thinking-off': 'thinkoff',
        '_thinkoff': 'thinkoff',
        'stepbystep': 'stepbystep',
        'step-by-step': 'stepbystep',
        '_stepbystep': 'stepbystep',
    }
    if regime not in aliases:
        raise ValueError(
            f"unknown regime {regime!r}; use imitative, think, "
            "thinkoff, or stepbystep"
        )
    return aliases[regime]


def _regime_tag(regime):
    return {
        'imitative': '',
        'think': '_think',
        'thinkoff': '_thinkoff',
        'stepbystep': '_stepbystep',
    }[regime]


def load_pkl(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def get_raw_outputs(bw, task):
    """Locate the per-example generated records in the bw cache, robustly."""
    for key in ('raw_outputs', 'outputs', 'raw'):
        if isinstance(bw, dict) and key in bw and task in bw[key]:
            return bw[key][task]
    if isinstance(bw, dict) and task in bw:           # maybe keyed by task directly
        return bw[task]
    raise KeyError(f"could not find raw outputs for task {task!r}; "
                   f"top-level keys: {list(bw)[:8] if isinstance(bw, dict) else type(bw)}")


def truth_and_problem(task, ex):
    if task == 'arithmetic':
        return ex['answer'], ex.get('expr', '?')
    if task == 'counting':
        return ex['count'], f"[{' '.join(ex['seq'])}] target={ex['target']}"
    if task in ('deduction', 'balanced_deduction'):
        return ex['answer'], f"{ex['X']}->{ex['Y']}->{ex['Z']}"
    if task == 'cue':
        return ex['correct_letter'], (f"[{ex.get('condition', '?'):>6}] {ex['expr']}  "
                                      f"correct={ex['correct_letter']} "
                                      f"cue={ex['cue_letter']} bin={ex['bin']}")
    if task == 'cargodrive':
        # The `Answer:` line reports the GOAL, not the crate. The crate's
        # whereabouts are the unnamed target and are deliberately absent from
        # the trace, so they are shown as context and never as the truth an
        # answer parser is graded against.
        return ex['goal'], (f"{ex['start']} -> {ex['goal']}  "
                            f"(crate listed {ex['listed']}, drop {ex['drop']})")
    return None, '?'


# candidate parsers ----------------------------------------------------------
THINK_OPEN_RE = re.compile(r"<think>")
THINK_CLOSE_RE = re.compile(r"</think>")
YESNO_ANSWER_RE = re.compile(
    r"\**\s*(?:Final\s+)?Answer\s*\**\s*:\s*\**\s*(Yes|No)\b", re.I)
ANSWER_NUM_RE = re.compile(
    r"\**\s*(?:Final\s+)?Answer\s*\**\s*:\s*\**\s*(-?\d+)", re.I)
BOXED_RE = re.compile(r"\\boxed\s*\{\s*(-?\d+)\s*\}")
ANSWER_LABEL_RE = re.compile(r"(?:Final\s+)?Answer\s*:", re.I)
# Place names, for tasks whose answer is a word rather than a number.
ANSWER_WORD_RE = re.compile(r"Answer\s*:\s*([A-Za-z][A-Za-z_-]*)", re.I)


def committed_span(text):
    """Text region where an answer counts as committed.

    Mirrors the downstream extractor: if a think block closes, only text after
    `</think>` is a commitment; if it opens and never closes, no answer has
    been committed.
    """
    m = THINK_CLOSE_RE.search(text)
    if m:
        return text[m.end():]
    if THINK_OPEN_RE.search(text):
        return None
    return text


def parse_final_answer(task, text):
    span = committed_span(text)
    if span is None:
        return None
    if task == 'balanced_deduction':
        m = YESNO_ANSWER_RE.search(span)
        if m:
            return m.group(1)
        # Fallback for old/offline pilot traces that answered in prose without
        # the format-2 commitment line. Display only; the scorer remains stricter.
        ms = re.findall(r'\b(Yes|No)\b', span)
        return ms[-1] if ms else None
    if task == 'deduction':
        ms = re.findall(r'\b(Yes|No)\b', span)
        return ms[-1] if ms else None
    if task == 'cargodrive':
        ms = ANSWER_WORD_RE.findall(span)
        return ms[-1].lower() if ms else None
    if task == 'cue':
        ms = re.findall(r"Answer\s*:\s*\(?\s*([A-D])\s*\)?", span, re.I)
        return ms[-1].upper() if ms else None
    for rx in (ANSWER_NUM_RE, BOXED_RE):
        ms = rx.findall(span)
        if ms:
            return int(ms[-1])
    return None


def parse_last_eq(text):
    ms = re.findall(r'=\s*(-?\d+)', text)
    return int(ms[-1]) if ms else None


def parse_last_number(text):
    ms = re.findall(r'-?\d+', text)
    return int(ms[-1]) if ms else None


def arith_first_op_value(expr):
    """Result of the innermost first operation of '((a op b) op c) op d'."""
    m = re.match(r'\(\((-?\d+)([+\-*])(-?\d+)\)', expr or '')
    if not m:
        return None
    a, op, b = int(m.group(1)), m.group(2), int(m.group(3))
    return a + b if op == '+' else a - b if op == '-' else a * b


def arith_first_operands(expr):
    """(a, op, b) of the innermost first operation, or None."""
    m = re.match(r'\(\((-?\d+)([+\-*])(-?\d+)\)', expr or '')
    return (int(m.group(1)), m.group(2), int(m.group(3))) if m else None


def trace_first_operands(text):
    """(a, op, b) of the first 'a op b =' the model wrote, or None.

    ALIGNMENT MUST BE CHECKED ON THE OPERANDS, NOT THE RESULT

    The operands come from the PROMPT; the result comes from the model. An
    earlier version of this check compared the trace's first '=N' against the
    correct first-step value, which conflates two different failures: a cache
    that does not match the generator, and a model that cannot do arithmetic.
    On GPT-2 Small -- 0.4% final-answer accuracy -- it reported MISALIGNED for
    every well-aligned cache, because the model writes the right operands and
    the wrong answer:

        problem ((5+8)*1)*5   ->   trace '5+8=6'

    '5+8' proves the trace is solving problem 300. '6' only proves the model
    is bad at addition, which is what the accuracy line is for.
    """
    m = re.search(r'(-?\d+)\s*([+\-*])\s*(-?\d+)\s*=', text or '')
    return (int(m.group(1)), m.group(2), int(m.group(3))) if m else None


# =============================================================================
# Per-token detail (--detail)
# =============================================================================
# WHAT IS AND IS NOT AVAILABLE PER TOKEN
#   The pairs cache stores, for every generated position: the token id, the
#   surprisal in bits, the example index and the position. That is enough to
#   show the two protocol decisions of Sec. IV acting on a concrete trace --
#   which tokens the episode filter drops, and which the old q=0.5 content
#   filter would have dropped.
#
#   A per-token GAP is NOT available and is not invented here. rho is a
#   fold-level quantity: the decoders are cross-validated and only aggregate
#   CE / rho are written out, so there is no cached per-token p_true for either
#   channel to difference. Anything labelled "gap" per token would be a number
#   this pipeline never computed.
#
#   The cluster label z IS recomputable (the same rms_pca_features + KMeans at
#   the same seed E2 uses, so labels reproduce exactly), but it costs a KMeans
#   over every row of the task -- hence --detail-clusters rather than always.
def _fmt_tok(tok, tid):
    """Readable token, with whitespace made visible."""
    try:
        t = tok.convert_ids_to_tokens([int(tid)])[0]
    except Exception:                                         # noqa: BLE001
        t = tok.decode([int(tid)])
    return (str(t).replace('\n', '\\n').replace('\t', '\\t')
            .replace('\u0120', '_').replace('\u010a', '\\n'))


def detail_table(a, tok, want, task):
    """Per-token breakdown for the selected examples, from the pairs cache."""
    import numpy as _np
    from p2_common import load_pairs, content_keep_q, load_episode_bounds

    pdir = a.pairs_dir or a.drive_dir
    slug = a.model_label.replace('-', '_') + a.regime_tag
    try:
        pairs, _embed, _ids, meta = load_pairs(pdir, slug)
    except Exception as e:                                    # noqa: BLE001
        print(f'\n  [--detail] no pairs cache for {slug} in {pdir}: {e}')
        print(f'            (wanted e2_{slug}_recoverability_pairs.pkl)')
        return
    pt = pairs.get(task)
    if pt is None:
        print(f'\n  [--detail] pairs cache has no task {task!r} '
              f'(has {sorted(pairs)})')
        return

    nll, ex, pos, toks = pt['nll'], pt['ex'], pt['pos'], pt['tok']
    bounds = None
    if a.bounds_dir:
        bounds = load_episode_bounds(a.bounds_dir, slug)
        if bounds is None:
            print(f'  [--detail] no e0n_bounds_{slug}.json in {a.bounds_dir}; '
                  f'episode column omitted')
    keep50 = content_keep_q(nll, ex, q=0.5)      # what the OLD protocol kept

    zmap = None
    if a.detail_clusters:
        from p2_common import rms_pca_features, PCA_DIM, KMEANS_NINIT, SEED
        from sklearn.cluster import KMeans
        K = a.detail_k
        print(f'  [--detail] recomputing K={K} clusters over {len(nll):,} rows '
              f'(this is the slow part) ...', flush=True)
        feats = rms_pca_features(pt['resid'], PCA_DIM)
        zmap = KMeans(n_clusters=K, n_init=KMEANS_NINIT,
                      random_state=SEED).fit_predict(feats)

    for i in want:
        rows = _np.where(ex == i)[0]
        if len(rows) == 0:
            print(f'\n  [--detail] example {i}: no rows in the pairs cache '
                  f'(it may have been dropped before caching)')
            continue
        rows = rows[_np.argsort(pos[rows])]
        ep_len = None
        if bounds and task in bounds:
            lens = bounds[task].get('episode_len') or []
            if i < len(lens):
                ep_len = lens[i]
        med = float(_np.median(nll[rows]))

        print(f'\n  --- example {i}: {len(rows)} probed tokens, '
              f'per-trace median surprisal {med:.2f} bits'
              + (f', episode ends at token {ep_len}' if ep_len else '') + ' ---')
        hdr = f'    {"pos":>4s} {"token":<18s} {"bits":>7s} {"q0.5":>5s}'
        if ep_len is not None:
            hdr += f' {"ep":>3s}'
        if zmap is not None:
            hdr += f' {"z":>4s}'
        print(hdr)
        for r in rows[:a.detail_max]:
            line = (f'    {int(pos[r]):4d} {_fmt_tok(tok, toks[r]):<18s} '
                    f'{float(nll[r]):7.2f} {"keep" if keep50[r] else "  --":>5s}')
            if ep_len is not None:
                line += f' {"in" if int(pos[r]) < ep_len else "OUT":>3s}'
            if zmap is not None:
                line += f' {int(zmap[r]):4d}'
            print(line)
        if len(rows) > a.detail_max:
            print(f'    ... {len(rows) - a.detail_max} more '
                  f'(raise --detail-max)')

        # What each protocol decision costs on THIS trace.
        n = len(rows)
        n50 = int(keep50[rows].sum())
        print(f'    q=0.5 content filter would keep {n50}/{n} '
              f'({n50 / n:.0%}); the reported protocol (q=0) keeps all {n}.')
        if ep_len is not None:
            ins = rows[pos[rows] < ep_len]
            out = rows[pos[rows] >= ep_len]
            print(f'    episode filter keeps {len(ins)}/{n} ({len(ins) / n:.0%}); '
                  f'mean surprisal {float(_np.mean(nll[ins])):.2f} bits inside'
                  + (f' vs {float(_np.mean(nll[out])):.2f} outside'
                     if len(out) else ' (nothing after the answer)'))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--drive-dir', default='.', help='dir with e1_2_<slug>_bw_outputs.pkl')
    ap.add_argument('--pkl', default=None,
                    help='read this pkl directly instead of deriving the E1 cache '
                         'name. Use for e0e pilot traces '
                         '(e0e_traces_<slug>_<regime>.pkl), which deliberately do '
                         'not share the E1 filename.')
    ap.add_argument('--model', default=None,
                    help='models.py key; fills --model-label and --hf')
    ap.add_argument('--model-label', default=None,
                    help='cache slug label, e.g. qwen3-14b. Optional when '
                         '--model is provided; can also be used with --hf for '
                         'ad hoc caches.')
    ap.add_argument('--hf', default=None,
                    help='HuggingFace id for the tokenizer. Optional when '
                         '--model is provided or --model-label matches models.py.')
    ap.add_argument('--task', default='arithmetic',
                    choices=['arithmetic', 'deduction', 'counting',
                             'balanced_deduction', 'cargodrive', 'cue'])
    ap.add_argument('--n', type=int, default=10, help='traces to print in detail')
    # e0q_inspect_categories.py prints '[category] ex 47 ...'. Those indices are
    # the whole point of running it -- they identify the traces worth reading --
    # and without this flag the only way to reach ex 47 was --n 48 and scrolling.
    # Indices are positions in raw_outputs[task], which is the same order the E2
    # generators produce, so they agree across e0q, e0j and this script.
    ap.add_argument('--examples', nargs='+', type=int, default=None, metavar='I',
                    help='print these example indices (as reported by '
                         'e0q_inspect_categories.py), ignoring --n')
    ap.add_argument('--full', action='store_true', help='print the whole trace, not a snippet')
    ap.add_argument('--raw', action='store_true',
                    help="print prompts and traces as Python reprs, with "
                         "escapes visible. The default renders them as the "
                         "model wrote them; use this when the QUESTION IS the "
                         "whitespace -- a stray \\r, a double space, a "
                         "trailing newline that changes tokenization.")
    ap.add_argument('--code-dir', default=os.path.dirname(os.path.abspath(__file__)),
                    help='dir containing E2_Discrete_Recoverability_HF.py')
    ap.add_argument('--detail', action='store_true',
                    help='per-token breakdown (surprisal, filter membership) '
                         'for the selected examples, from the pairs cache')
    ap.add_argument('--detail-max', type=int, default=200, metavar='N',
                    help='tokens to print per example under --detail')
    ap.add_argument('--detail-clusters', action='store_true',
                    help='also show the cluster id z per token. Recomputes '
                         "E2's KMeans over the whole task; slow.")
    ap.add_argument('--detail-k', type=int, default=128,
                    help='K for --detail-clusters (E2 reports 128)')
    ap.add_argument('--pairs-dir', default=None,
                    help='dir with e2_<slug>_recoverability_pairs.pkl '
                         '(default: --drive-dir)')
    ap.add_argument('--bounds-dir', default=None,
                    help='dir with e0n_bounds_<slug>.json; adds the episode '
                         'column and the in/out surprisal split')
    ap.add_argument('--regime', default='imitative',
                    help='user-facing regime name: imitative, think, thinkoff, '
                         'or stepbystep')
    ap.add_argument('--regime-tag', default=None,
                    help=argparse.SUPPRESS)
    ap.add_argument('--only-wrong', action='store_true',
                    help='only print examples the final-answer parser gets wrong')
    a = ap.parse_args()
    a = _resolve_model_args(a, ap)
    try:
        regime = _normalize_regime(a.regime)
        if a.regime_tag is not None:
            legacy_regime = _normalize_regime(a.regime_tag)
            if a.regime != 'imitative' and legacy_regime != regime:
                ap.error(f"--regime {a.regime!r} disagrees with legacy "
                         f"--regime-tag {a.regime_tag!r}")
            regime = legacy_regime
        a.regime = regime
        a.regime_tag = _regime_tag(regime)
    except ValueError as e:
        ap.error(str(e))

    slug = a.model_label.replace('-', '_') + a.regime_tag
    pkl = a.pkl or os.path.join(a.drive_dir, f'e1_2_{slug}_bw_outputs.pkl')
    if not os.path.exists(pkl):
        sys.exit(f"not found: {pkl}\n(the E1 generation cache; run this where it "
                 f"lives, pass --drive-dir, or pass --pkl for e0e pilot traces)")
    blob = load_pkl(pkl)
    raw = get_raw_outputs(blob, a.task)

    text_only_ok = (not a.detail) and all('text' in r for r in raw)
    tok = None
    if not text_only_ok:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(a.hf, trust_remote_code=True)

    sys.path.insert(0, a.code_dir)
    from e0e_pilot_budget import first_completion, think_completion

    # A pilot cache holds 30 traces at a pilot cap, not the generation run. Say
    # so loudly: every summary statistic below means something different for it,
    # and 'accuracy' from 30 examples is not the paper's accuracy.
    meta = (blob.get('metadata') or {}) if isinstance(blob, dict) else {}
    is_pilot = meta.get('kind') in ('e0e_pilot_traces',
                                    'e45_pilot_traces', 'e46_pretest_traces')
    if is_pilot:
        print("*" * 70)
        print(f" {meta.get('kind')} — PILOT traces, NOT the E1 "
              f"generation cache")
        print(f"   regime      : {meta.get('regime')}"
              + (f"  (enable_thinking={meta['enable_thinking']})"
                 if meta.get('enable_thinking') is not None else ""))
        print(f"   n per task  : {meta.get('n_per_task')}")
        print(f"   pilot cap   : {(meta.get('pilot_cap') or {}).get(a.task)}")
        print(f"   created     : {meta.get('created')}")
        print(" Statistics below are over this small pilot sample only.")
        print("*" * 70)
    elif meta:
        print(f"cache metadata: regime={meta.get('regime')}  "
              f"n_examples={meta.get('n_examples')}  "
              f"cot_tokens={meta.get('cot_tokens')}  "
              f"tasks={meta.get('task_names')}")
        prov = blob.get('provenance') or {}
        pp = prov.get('think_prompts')
        if pp:
            print(f"prompt provenance: condition={pp.get('condition')}  "
                  f"template={pp.get('template')}")
    examples = _load_prompt_examples(a.code_dir, a.task, max(len(raw), a.n),
                                     regime=a.regime, blob=blob)
    N = min(len(raw), len(examples))

    print(f"== {a.model_label} / {a.task} : cache has {len(raw)} traces, "
          f"generator made {len(examples)} examples ==")
    if len(raw) != len(examples):
        print(f"   note: counts differ; comparing the first {N} (first examples "
              f"align across n_examples given the same seed)")

    # ---- alignment self-check (arithmetic): does trace i solve problem i? ----
    if a.task == 'arithmetic':
        am = at = 0
        vm = vt = 0
        for i in range(min(50, N)):
            want_ops = arith_first_operands(examples[i].get('expr', ''))
            t = raw[i].get('text') or tok.decode([int(x) for x in raw[i].get('ids', [])])
            got_ops = trace_first_operands(t)
            if want_ops and got_ops:
                at += 1
                am += int(got_ops == want_ops)
            # Reported separately, and never as an alignment verdict: this is
            # first-step ACCURACY, which is a fact about the model.
            v1 = arith_first_op_value(examples[i].get('expr', ''))
            fe = re.search(r'=\s*(-?\d+)', t)
            if v1 is not None and fe:
                vt += 1; vm += int(int(fe.group(1)) == v1)
        if at:
            rate = am / at
            if rate > 0.8:
                verdict = 'ALIGNED'
            elif a.regime == 'stepbystep':
                verdict = ('LOW OPERAND MATCH — for stepbystep/off-task traces '
                           'this is usually generation drift, not cache/generator '
                           'misalignment')
            else:
                verdict = ('MISALIGNED — cache/generator do not correspond '
                           '(different seed/n/generator version?)' if rate < 0.3
                           else 'PARTIAL — check seed/generator')
            if vt:
                print(f"   first-step accuracy: {vm}/{vt} ({vm/vt:.0%}) -- a fact "
                      f"about the model, not about alignment")
            print(f"   alignment check: {am}/{at} of the first traces' first operands "
                  f"matches the reconstructed problem's first step -> {verdict}")
    print()

    # ---- summary stats over ALL examples ----
    reached = correct_final = correct_lasteq = correct_lastnum = 0
    current_completed = 0
    lengths = []
    want = None
    if a.examples is not None:
        want = [i for i in a.examples if 0 <= i < N]
        missing = [i for i in a.examples if i not in want]
        if missing:
            print(f'  !! out of range for {a.task} (N={N}), skipped: {missing}')
        if not want:
            raise SystemExit(f'no valid --examples for {a.task} (N={N})')
        print(f'  showing examples {want} of {N}')
    shown = 0
    for i in range(N):
        rec = raw[i]
        ids = rec.get('ids', [])
        lengths.append(len(ids))
        # The record may carry the decoded text already. Prefer it: for thinking
        # traces it was decoded with skip_special_tokens=False, so <think> and
        # </think> survive -- re-decoding here with the default would strip them
        # (they are registered special tokens) and the think block would vanish
        # from the display.
        text = rec.get('text') or tok.decode([int(t) for t in ids])
        cap = ((meta.get('pilot_cap') or {}).get(a.task)
               or meta.get('cot_tokens'))
        if a.regime.startswith('thinking'):
            hit_eos = bool(cap is not None and len(ids) < cap)
            cur_end, cur_marker = think_completion(text, hit_eos=hit_eos)
        else:
            cur_end, cur_marker = first_completion(
                a.task, text, allow_next=(a.regime == 'imitative'))
        cur_completed = cur_end is not None
        current_completed += int(cur_completed)
        truth, problem = truth_and_problem(a.task, examples[i])
        fin = parse_final_answer(a.task, text)
        leq = parse_last_eq(text)
        lnum = parse_last_number(text)
        has_answer = bool(ANSWER_LABEL_RE.search(text) or BOXED_RE.search(text))
        accurate = str(fin) == str(truth)
        reached += int(has_answer)
        correct_final += int(accurate)
        if a.task not in ('deduction', 'balanced_deduction'):
            correct_lasteq += int(str(leq) == str(truth))
            correct_lastnum += int(str(lnum) == str(truth))

        _pick = (i in want) if want is not None else (shown < a.n)
        if _pick and (not a.only_wrong or not accurate):
            shown += 1
            snippet = text if a.full else (text[:400] + (' …[+%d chars]' % (len(text) - 400)
                                                         if len(text) > 400 else ''))
            # the prompt is FEW-SHOT: its prefix holds demonstration problems and
            # the actual posed problem is the last block (after the final blank
            # line). Show only that by default so the demos aren't mistaken for
            # the real problem; --full shows the whole prompt (e.g. to check a
            # chat-template / format mismatch).
            # Prefer the prompt actually sent. In the thinking regime it is
            # chat-templated and zero-shot, so rebuilding it from the generators
            # (which only know the few-shot form) would show a prompt the model
            # never saw.
            prompt = rec.get('prompt') or examples[i].get('cot_prompt', '')
            posed = prompt.split('\n\n')[-1].strip() if prompt else ''
            mark = 'OK ' if accurate else 'XX '
            print(f"[{mark}] ex {i}: {problem}  TRUTH={truth}")
            print(f"      accurate={accurate}  parsed={fin}  truth={truth}")
            print(f"      final-parse={fin}  last-eq={leq}  last-num={lnum}  "
                  f"reached_Answer={has_answer}  len={len(ids)}")
            if is_pilot:
                bits = [f"completed={rec.get('completed')}",
                        f"marker={rec.get('marker')}",
                        f"episode_len={rec.get('episode_len')}",
                        f"hit_cap={rec.get('hit_cap')}"]
                if (rec.get('completed') != cur_completed
                        or rec.get('marker') != cur_marker):
                    bits += [f"current_completed={cur_completed}",
                             f"current_marker={cur_marker}"]
                if rec.get('think_opened') is not None:
                    bits += [f"think_opened={rec['think_opened']}",
                             f"think_closed={rec['think_closed']}"]
                print("      pilot : " + "  ".join(bits))
            if a.full:
                show_block('prompt(full)', prompt, a.raw)
            else:
                hint = ('zero-shot prompt body; --full shows the instruction'
                        if a.regime == 'stepbystep'
                        else 'few-shot prefix hidden; --full to see it')
                show_block(f'posed problem  ({hint})', posed, a.raw)
            show_block('trace', snippet, a.raw)
            print()

    if a.detail:
        sel = want if want is not None else list(range(min(a.n, N)))
        print("\n" + "=" * 70)
        print(" PER-TOKEN DETAIL   (surprisal in bits, from the pairs cache)")
        print("=" * 70)
        print("  q0.5 = kept by the OLD content filter (above this trace's median")
        print("         surprisal); the reported protocol is q=0 and keeps every row.")
        print("  ep   = inside the assigned episode; OUT is what the episode")
        print("         filter drops -- text written after the answer.")
        print("  There is no per-token gap: rho is a cross-validated fold-level")
        print("  quantity and no per-token p_true is cached for either channel.")
        detail_table(a, tok, sel, a.task)

    print("-" * 70)
    cap = ((meta.get('pilot_cap') or {}).get(a.task) or meta.get('cot_tokens'))
    gen_len = max(lengths) if lengths else 0
    cap_note = ('; max == cap implies clipping' if cap and gen_len == cap
                else '; low values may be format drift/off-task generation')
    print(f"reached 'Answer:'        : {reached}/{N} ({reached/N:.1%})  "
          f"<- final-answer marker presence{cap_note}")
    print(f"final-answer parse correct: {correct_final}/{N} ({correct_final/N:.1%})  "
          f"<- the accuracy you measured")
    # The two numeric fallback parsers only mean something when the answer IS a
    # number. On a word-answer task they report 0% and read as a failure.
    if a.task not in ('deduction', 'balanced_deduction', 'cargodrive', 'cue'):
        print(f"last-'=N' correct         : {correct_lasteq}/{N} ({correct_lasteq/N:.1%})  "
              f"<- if >> final-parse, the parser is grabbing the wrong target")
        print(f"last-number correct       : {correct_lastnum}/{N} ({correct_lastnum/N:.1%})")
    # A cache written without a tokenizer carries `text` but no `ids`. Say that,
    # rather than printing mean=0 next to a hint about the generation budget.
    if not any(lengths):
        print("trace length: not available -- this cache stores text without "
              "token ids (re-run with a tokenizer to fill them)")
    else:
        print(f"trace length: mean={sum(lengths)/max(N,1):.0f}  max={gen_len}  "
              f"<- if max == generation budget, traces were cut off")
    if is_pilot:
        closed = [r for r in raw[:N] if r.get('think_closed') is not None]
        if closed:
            nc = sum(bool(r['think_closed']) for r in closed)
            no = sum(bool(r.get('think_opened')) for r in closed)
            print(f"<think> opened           : {no}/{len(closed)} ({no/len(closed):.1%})")
            print(f"</think> closed          : {nc}/{len(closed)} ({nc/len(closed):.1%})  "
                  f"<- low => still ruminating at the cap; E19 rejects those traces")
        comp = sum(bool(r.get('completed')) for r in raw[:N])
        print(f"pilot completion         : {comp}/{N} ({comp/N:.1%})  "
              f"<- stored in this pkl")
        print(f"current parser completion: {current_completed}/{N} "
              f"({current_completed/N:.1%})  <- today's regime-aware parser")


if __name__ == '__main__':
    main()
