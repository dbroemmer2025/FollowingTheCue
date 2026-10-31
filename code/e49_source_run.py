#!/usr/bin/env python3
"""E49 GPU stages: generate, capture MLP neurons, export review, patch.

Stages (run in order; each resumes and refuses to mix designs):

  generate   Greedy generation of every frozen E49 prompt (8 cells x items).
  capture    One forward pass per response (E49 rows + the two replayed E47
             transfer cells). Saves, at four positions, the input to every
             selected layer's mlp.down_proj (= the MLP neurons, post-gate)
             and the residual stream. float16 memmaps + index.json.
  export     Review worksheet in the E47 schema for e46_llm_review.py.
  patch      Regenerate evaluation items with a neuron family shifted by its
             discovery-estimated category difference from the cue onward,
             plus matched controls and a regenerated 'none' arm.

Positions (absolute token index in prompt+generation):
  cue_last              last token overlapping the cue sentence
  prompt_end            last prompt token
  rationale_first_stop  first generated token that closes a sentence/line
  pre_answer            E47's pre_answer anchor (token before the answer letter)

Usage (from code/):
  python e49_source_run.py generate --run ../e49/gemma_2_9b_it_movie --batch-size 8
  python e49_source_run.py capture  --run ../e49/gemma_2_9b_it_movie --layer-stride 2
  python e49_source_run.py export   --run ../e49/gemma_2_9b_it_movie
  python e49_source_run.py patch    --run ../e49/gemma_2_9b_it_movie \
         --families ../e49/gemma_2_9b_it_movie/analysis/families.json
  python e49_source_run.py --selftest          # tiny random Gemma-2, CPU
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pickle
import re
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

POSITIONS = ('cue_last', 'prompt_end', 'rationale_first_stop', 'pre_answer')
STOP_RE = re.compile(r'(?:Answer:|So the answer is)\s*\([A-D]\)[.!]?\s*$', re.I)
SENTENCE_RE = re.compile(r'\S[.!?:](?:\s|$)|\S\n')


def digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()


def atomic_pickle(path, obj):
    tmp = Path(str(path) + '.tmp')
    tmp.write_bytes(pickle.dumps(obj))
    tmp.replace(path)


def load_design(run):
    d = json.loads((Path(run) / 'design.json').read_text())
    if digest_design(d) != d['design_id']:
        raise ValueError('E49 design.json changed after freezing')
    return d


def digest_design(d):
    return hashlib.sha256(json.dumps({k: v for k, v in d.items() if k != 'design_id'},
                                     sort_keys=True).encode()).hexdigest()


# --------------------------------------------------------------------------
# Pure helpers (tested without a GPU)
# --------------------------------------------------------------------------

def is_stop(text):
    return bool(STOP_RE.search(text) or '\nQ:' in text or '\nProblem:' in text)


def cue_token_span(offsets, prompt, cue_text):
    """First and last prompt-token indices overlapping the cue sentence."""
    start = prompt.index(cue_text)
    end = start + len(cue_text)
    hit = [i for i, (a, b) in enumerate(offsets)
           if b > a and a < end and b > start]
    if not hit:
        raise ValueError('cue sentence maps to no token')
    return hit[0], hit[-1]


def first_stop_count(tok, ids):
    """Smallest k such that decode(ids[:k]) closes a sentence or line."""
    for k in range(1, len(ids) + 1):
        if SENTENCE_RE.search(tok.decode(ids[:k], skip_special_tokens=True)):
            return k
    return None


def trim_generated(ids, stop_len, eos_ids, pad_id, cap):
    """Mirror E47 single-stream output: through the stop token or EOS."""
    out = []
    for i, t in enumerate(ids):
        if stop_len is not None and i >= stop_len:
            break
        if t == pad_id and t not in eos_ids and i > 0 and stop_len is not None:
            break
        out.append(int(t))
        if t in eos_ids:
            break
    return out[:cap]


def make_row(d, spec, text, ids, prompt_ids, cap, extra=None):
    from e47_design import answer_parse
    from e46_prompts_cue import episode_span
    ep = episode_span(text)
    answer, compliant, parse_kind = answer_parse(ep)
    row = {'id': spec['id'], 'design_id': d['design_id'], 'index': spec['index'],
           'source_id': spec['source_id'], 'cell': spec['cell'],
           'speaker': spec.get('speaker'), 'expertise': spec.get('expertise'),
           'confidence': spec.get('confidence'), 'frame': spec.get('frame'),
           'split': spec.get('split'), 'condition': spec['cell'],
           'cue_letter': spec['cue_letter'], 'cue_text': spec['cue_text'],
           'baseline_answer': spec['baseline_answer'], 'prompt': spec['prompt'],
           'text': text, 'episode': ep, 'answer': answer,
           'format_compliant': compliant, 'answer_parse_kind': parse_kind,
           'ids': list(map(int, ids)), 'prompt_ids': list(map(int, prompt_ids)),
           'prompt_add_special_tokens': True, 'hit_cap': len(ids) >= cap,
           **(extra or {})}
    row['content_sha256'] = digest(row)
    return row


def switch_to_cue(row):
    from e47_design import scored_answer
    a = scored_answer(row)
    if a not in tuple('ABCD') or row.get('hit_cap'):
        return None
    return int(a == row['cue_letter'] and a != row['baseline_answer'])


# --------------------------------------------------------------------------
# Model helpers
# --------------------------------------------------------------------------

def load(d, device_map='auto'):
    from e46_pretest_cue import load_model
    tok, model = load_model(d['hf'], d['dtype'], device_map, d['revision'])
    return tok, model


def mlp_modules(model):
    from e46_layer_capture import transformer_blocks
    out = []
    for i, block in enumerate(transformer_blocks(model)):
        down = getattr(getattr(block, 'mlp', None), 'down_proj', None)
        if down is None:
            raise ValueError(f'block {i} has no mlp.down_proj')
        out.append(down)
    return out


class _Batch:
    """Per-row first-stop detection inside a batched greedy generate."""

    def __init__(self, tok, n_prompt, batch):
        self.tok, self.n, self.stop = tok, n_prompt, [None] * batch

    def __call__(self, input_ids, scores, **kwargs):
        import torch
        done = []
        for b in range(input_ids.shape[0]):
            if self.stop[b] is None:
                gen = input_ids[b, self.n:].tolist()
                if is_stop(self.tok.decode(gen, skip_special_tokens=True)):
                    self.stop[b] = len(gen)
            done.append(self.stop[b] is not None)
        return torch.tensor(done, device=input_ids.device)


def generate_batch(tok, model, prompts, cap, hooks=None):
    """Greedy, left-padded batch. Returns [(ids, prompt_ids)].

    ``hooks`` is an optional context manager factory taking (pad_lens, n_max).
    """
    import contextlib
    import torch
    from transformers import StoppingCriteriaList
    tok.padding_side = 'left'
    enc = tok(prompts, return_tensors='pt', padding=True, add_special_tokens=True)
    dev = model.get_input_embeddings().weight.device
    enc = {k: enc[k].to(dev) for k in ('input_ids', 'attention_mask')}
    n_max = enc['input_ids'].shape[1]
    pad_lens = (enc['attention_mask'] == 0).sum(1).tolist()
    crit = _Batch(tok, n_max, len(prompts))
    ctx = hooks(pad_lens, n_max) if hooks else contextlib.nullcontext()
    with ctx, torch.inference_mode():
        out = model.generate(**enc, do_sample=False, max_new_tokens=cap,
                             pad_token_id=tok.pad_token_id,
                             stopping_criteria=StoppingCriteriaList([crit]))
    eos = model.generation_config.eos_token_id
    eos_ids = set(eos if isinstance(eos, (list, tuple)) else [eos]) - {None}
    results = []
    for b, prompt in enumerate(prompts):
        prompt_ids = enc['input_ids'][b, pad_lens[b]:].tolist()
        gen = out[b, n_max:].tolist()
        ids = trim_generated(gen, crit.stop[b], eos_ids, tok.pad_token_id, cap)
        results.append((ids, prompt_ids))
    return results


# --------------------------------------------------------------------------
# Stage: generate
# --------------------------------------------------------------------------

def stage_generate(run, batch_size=8, limit=None, tok=None, model=None,
                   save_every=16):
    run = Path(run)
    d = load_design(run)
    path = run / 'generations.pkl'
    blob = (pickle.loads(path.read_bytes()) if path.exists() else
            {'design_id': d['design_id'], 'rows': [], 'complete': False,
             'batch_size': batch_size})
    if blob['design_id'] != d['design_id']:
        raise ValueError('generations.pkl belongs to another design')
    if blob.get('batch_size') != batch_size:
        raise ValueError(f"resume with --batch-size {blob.get('batch_size')}; "
                         'batch composition changes bf16 numerics')
    done = {r['id'] for r in blob['rows']}
    todo = [r for r in d['rows'] if r['id'] not in done]
    if limit:
        todo = todo[:limit]
    if tok is None:
        tok, model = load(d)
    # Sort by prompt length so batches pad little; order is deterministic.
    todo.sort(key=lambda r: (len(r['prompt']), r['id']))
    t0, since = time.time(), 0
    for start in range(0, len(todo), batch_size):
        chunk = todo[start:start + batch_size]
        if batch_size == 1:
            from e47_experiment import generate_one
            res = generate_one(tok, model, chunk[0]['prompt'], d['max_new_tokens'])
            outs = [(res['ids'], res['prompt_ids'])]
        else:
            outs = generate_batch(tok, model, [r['prompt'] for r in chunk],
                                  d['max_new_tokens'])
        for spec, (ids, prompt_ids) in zip(chunk, outs):
            check = tok(spec['prompt'], add_special_tokens=True)['input_ids']
            if check != prompt_ids:
                raise ValueError('left-padded prompt ids differ from unpadded encoding')
            text = tok.decode(ids, skip_special_tokens=True)
            blob['rows'].append(make_row(d, spec, text, ids, prompt_ids,
                                         d['max_new_tokens']))
        since += len(chunk)
        if since >= save_every:
            atomic_pickle(path, blob)
            since = 0
        rate = (start + len(chunk)) / max(time.time() - t0, 1e-9)
        print(f'  generated {len(blob["rows"])}/{len(d["rows"])} '
              f'({rate:.2f}/s)', flush=True)
    blob['complete'] = len(blob['rows']) == len(d['rows'])
    atomic_pickle(path, blob)
    return blob


# --------------------------------------------------------------------------
# Stage: capture
# --------------------------------------------------------------------------

def transfer_rows(run, d):
    """Replay E47's original authority and user responses for these items."""
    from e49_source_design import e47_cue_lines
    e47 = (HERE.parent / 'e47' / d['e47_run'])
    traces = pickle.loads((e47 / 'traces.pkl').read_bytes())
    by_id = {r['id']: r for r in traces['rows']}
    out = []
    for it in d['items']:
        for cell in ('e47_authority', 'e47_user'):
            r = dict(by_id[it['e47_ids'][cell]])
            r.update(cell=cell, split=it['split'], frame=None,
                     speaker='third' if cell == 'e47_authority' else 'user',
                     expertise='expert' if cell == 'e47_authority' else None,
                     confidence='certain' if cell == 'e47_authority' else 'tentative',
                     baseline_answer=it['baseline_answer'],
                     cue_text=e47_cue_lines(it['cue_letter'])[cell])
            out.append(r)
    return out


def positions_for(tok, row):
    """Absolute positions of the four sites; None where undefined."""
    from e47_capture import checkpoints
    enc = tok(row['prompt'], add_special_tokens=True, return_offsets_mapping=True)
    if enc['input_ids'] != row['prompt_ids']:
        raise ValueError(f"{row['id']}: prompt ids differ from generation")
    n = len(row['prompt_ids'])
    first, last = cue_token_span(enc['offset_mapping'], row['prompt'], row['cue_text'])
    pos = {'cue_first': first, 'cue_last': last, 'prompt_end': n - 1}
    k = first_stop_count(tok, row['ids'])
    pos['rationale_first_stop'] = None if k is None else n + k - 1
    try:
        points, _ = checkpoints(row, tok)
        pos['pre_answer'] = (n + points['pre_answer'] - 1
                             if 'pre_answer' in points else None)
    except ValueError:
        pos['pre_answer'] = None
    return pos


def stage_capture(run, layer_stride=2, layers=None, tok=None, model=None,
                  limit=None):
    import torch
    run = Path(run)
    d = load_design(run)
    gens = pickle.loads((run / 'generations.pkl').read_bytes())
    if not gens.get('complete') and limit is None:
        raise ValueError('finish generate first')
    rows = gens['rows'] + transfer_rows(run, d)
    if limit:
        rows = rows[:limit]
    if tok is None:
        tok, model = load(d)
    mods = mlp_modules(model)
    n_layers = len(mods)
    if layers is None:
        layers = sorted(set(list(range(0, n_layers, layer_stride)) + [n_layers - 1]))
    d_mlp = mods[0].in_features
    d_model = model.get_input_embeddings().weight.shape[1]
    cap_dir = run / 'capture'
    cap_dir.mkdir(exist_ok=True)
    index_path = cap_dir / 'index.json'
    shape_mlp = (len(rows), len(POSITIONS), len(layers), d_mlp)
    shape_res = (len(rows), len(POSITIONS), len(layers), d_model)
    gb = (np.prod(shape_mlp) + np.prod(shape_res)) * 2 / 1e9
    if index_path.exists():
        index = json.loads(index_path.read_text())
        if (index['design_id'] != d['design_id'] or index['layers'] != layers
                or index['n_rows'] != len(rows)):
            raise ValueError('existing capture differs (design/layers/rows)')
        mode = 'r+'
    else:
        index = {'design_id': d['design_id'], 'layers': layers,
                 'positions': list(POSITIONS), 'n_rows': len(rows),
                 'd_mlp': int(d_mlp), 'd_model': int(d_model),
                 'n_model_layers': n_layers,
                 'site': 'input to mlp.down_proj (post-gate MLP neurons); '
                         'residual = output_hidden_states[layer + 1]',
                 'rows': [], 'done': []}
        mode = 'w+'
    print(f'capture: {len(rows)} rows x {len(POSITIONS)} positions x '
          f'{len(layers)} layers; d_mlp {d_mlp}, d_model {d_model}; '
          f'{gb:.1f} GB on disk', flush=True)
    mlp = np.lib.format.open_memmap(cap_dir / 'mlp.npy', mode=mode,
                                    dtype=np.float16, shape=shape_mlp)
    res = np.lib.format.open_memmap(cap_dir / 'resid.npy', mode=mode,
                                    dtype=np.float16, shape=shape_res)
    if mode == 'w+':
        mlp[:] = np.nan
        res[:] = np.nan
        index['rows'] = [None] * len(rows)
    done = set(index['done'])
    grabbed = {}

    def hook(li):
        def fn(module, args):
            grabbed[li] = args[0][0].detach()
        return fn

    handles = [mods[L].register_forward_pre_hook(hook(i))
               for i, L in enumerate(layers)]
    dev = model.get_input_embeddings().weight.device
    try:
        for r_i, row in enumerate(rows):
            if r_i in done:
                continue
            pos = positions_for(tok, row)
            wanted = [pos[p] for p in POSITIONS]
            last = max(p for p in wanted if p is not None)
            full = torch.tensor([row['prompt_ids'] + row['ids']], device=dev)[:, :last + 1]
            grabbed.clear()
            with torch.inference_mode():
                out = model(input_ids=full, attention_mask=torch.ones_like(full),
                            output_hidden_states=True, use_cache=False)
            for p_i, p in enumerate(wanted):
                if p is None:
                    continue
                for l_i, L in enumerate(layers):
                    mlp[r_i, p_i, l_i] = grabbed[l_i][p].float().cpu().numpy().astype(np.float16)
                    res[r_i, p_i, l_i] = (out.hidden_states[L + 1][0, p]
                                          .float().cpu().numpy().astype(np.float16))
            index['rows'][r_i] = {
                'id': row['id'], 'cell': row['cell'], 'index': row['index'],
                'source_id': row['source_id'], 'split': row.get('split'),
                'frame': row.get('frame'), 'speaker': row.get('speaker'),
                'expertise': row.get('expertise'),
                'confidence': row.get('confidence'),
                'cue_letter': row['cue_letter'],
                'switch_to_cue': switch_to_cue(row), 'positions': pos,
                'content_sha256': row['content_sha256']}
            index['done'].append(r_i)
            if len(index['done']) % 25 == 0:
                mlp.flush(); res.flush()
                index_path.write_text(json.dumps(index))
                print(f'  captured {len(index["done"])}/{len(rows)}', flush=True)
    finally:
        for h in handles:
            h.remove()
        mlp.flush(); res.flush()
        index_path.write_text(json.dumps(index))
    index['complete'] = len(index['done']) == len(rows)
    index_path.write_text(json.dumps(index))
    return index


# --------------------------------------------------------------------------
# Stage: export review worksheet
# --------------------------------------------------------------------------

REVIEW_FIELDS = ('review_mention', 'review_source_present',
                 'review_source_correct', 'review_attribution',
                 'review_denial', 'review_answer_changed')


def stage_export(run, source='generations.pkl', output=None):
    run = Path(run)
    blob = pickle.loads((run / source).read_bytes())
    rows = blob['rows']
    output = Path(output) if output else run / (Path(source).stem + '_review.csv')
    if output.exists():
        raise ValueError(f'{output} exists; worksheets are immutable')
    with output.open('x', newline='') as fh:
        w = None
        for r in rows:
            rec = {'audit_id': r['id'], 'content_sha256': r['content_sha256'],
                   'index': r['index'], 'arm': r.get('arm', 'cued'),
                   'condition': r['cell'],
                   'cue_type': f"{r.get('speaker')}_{r.get('expertise')}",
                   'branch': 'initial',
                   'conversation_before_response': r['prompt'],
                   'response_episode': r['episode'], 'raw_generation': r['text'],
                   **{k: '' for k in REVIEW_FIELDS}, 'reviewer': '', 'notes': ''}
            if w is None:
                w = csv.DictWriter(fh, fieldnames=list(rec))
                w.writeheader()
            w.writerow(rec)
    print(f'wrote {output} ({len(rows)} rows)')
    return output


# --------------------------------------------------------------------------
# Stage: patch
# --------------------------------------------------------------------------

class ShiftHooks:
    """Add a per-neuron shift to selected MLP neurons from each row's cue on.

    Positions are tracked across the prefill call and every cached decoding
    step; with left padding each row's start is pad_len + cue_first.
    """

    def __init__(self, model, plan, starts):
        import torch
        self.model, self.plan = model, plan     # plan: {layer: (idx, delta)}
        self.starts = starts                     # absolute start per row (padded)
        self.offset = 0
        self.cur = None
        self.handles = []
        self.torch = torch

    def __enter__(self):
        torch = self.torch
        mods = mlp_modules(self.model)
        base = self.model.get_decoder() if hasattr(self.model, 'get_decoder') else self.model

        def pre(module, args, kwargs):
            x = kwargs.get('input_ids')
            if x is None and args:
                x = args[0]
            if x is None:
                x = kwargs.get('inputs_embeds')
            L = x.shape[1]
            self.cur = torch.arange(self.offset, self.offset + L)
            self.offset += L

        self.handles.append(base.register_forward_pre_hook(pre, with_kwargs=True))
        for layer, (idx, delta) in self.plan.items():
            mod = mods[int(layer)]
            dev = mod.weight.device
            idx_t = torch.as_tensor(idx, device=dev, dtype=torch.long)
            delta_t = torch.as_tensor(delta, device=dev, dtype=mod.weight.dtype)
            starts = torch.as_tensor(self.starts, device=dev)

            def fn(module, args, idx_t=idx_t, delta_t=delta_t, starts=starts):
                x = args[0]
                cur = self.cur.to(x.device)
                mask = (cur[None, :] >= starts[:, None]).to(x.dtype)   # B x L
                x = x.clone()
                x[..., idx_t] = x[..., idx_t] + mask[..., None] * delta_t
                return (x,) + tuple(args[1:])

            self.handles.append(mod.register_forward_pre_hook(fn))
        return self

    def __exit__(self, *exc):
        for h in self.handles:
            h.remove()
        return False


def stage_patch(run, families, cell='third_expert_certain', arms=None,
                batch_size=8, dose=1.0, site='from_cue', limit=None,
                tok=None, model=None):
    run = Path(run)
    d = load_design(run)
    fam = json.loads(Path(families).read_text())
    if fam['design_id'] != d['design_id']:
        raise ValueError('families.json is from another design')
    arms = arms or ['none'] + list(fam['arms'])
    specs = [r for r in d['rows'] if r['split'] == 'evaluation' and r['cell'] == cell]
    if limit:
        specs = specs[:limit]
    if tok is None:
        tok, model = load(d)
    out_path = run / f'patch_{cell}_dose{dose:g}_{site}.pkl'
    blob = (pickle.loads(out_path.read_bytes()) if out_path.exists() else
            {'design_id': d['design_id'], 'families_sha256':
             hashlib.sha256(Path(families).read_bytes()).hexdigest(),
             'cell': cell, 'dose': dose, 'site': site, 'batch_size': batch_size,
             'rows': []})
    if blob['families_sha256'] != hashlib.sha256(Path(families).read_bytes()).hexdigest():
        raise ValueError('families.json changed since this patch run began')
    done = {(r['arm'], r['spec_id']) for r in blob['rows']}
    for arm in arms:
        plan = {} if arm == 'none' else {
            int(L): (v['neurons'], [dose * x for x in v['delta']])
            for L, v in fam['arms'][arm]['plan'].items()}
        todo = [s for s in specs if (arm, s['id']) not in done]
        for start in range(0, len(todo), batch_size):
            chunk = todo[start:start + batch_size]
            cue_first = []
            for s in chunk:
                enc = tok(s['prompt'], add_special_tokens=True, return_offsets_mapping=True)
                first, last = cue_token_span(enc['offset_mapping'], s['prompt'], s['cue_text'])
                cue_first.append(first if site == 'from_cue' else last + 1)

            def hooks(pad_lens, n_max, cue_first=cue_first, plan=plan):
                starts = [p + c for p, c in zip(pad_lens, cue_first)]
                return ShiftHooks(model, plan, starts)

            outs = generate_batch(tok, model, [s['prompt'] for s in chunk],
                                  d['max_new_tokens'], hooks=hooks if plan else None)
            for s, (ids, prompt_ids) in zip(chunk, outs):
                text = tok.decode(ids, skip_special_tokens=True)
                spec = {**s, 'id': digest([s['id'], arm, dose, site])}
                row = make_row(d, spec, text, ids, prompt_ids, d['max_new_tokens'],
                               {'arm': arm, 'spec_id': s['id'], 'dose': dose,
                                'site': site})
                blob['rows'].append(row)
            atomic_pickle(out_path, blob)
            print(f'  {arm}: {min(start + batch_size, len(todo))}/{len(todo)}', flush=True)
    blob['complete'] = True
    atomic_pickle(out_path, blob)
    return out_path


# --------------------------------------------------------------------------
# Selftest: tiny random Gemma-2 and a character tokenizer, CPU only
# --------------------------------------------------------------------------

def _tiny():
    import torch
    from tokenizers import Tokenizer, models, pre_tokenizers, decoders
    from transformers import Gemma2Config, Gemma2ForCausalLM, PreTrainedTokenizerFast
    chars = [chr(c) for c in range(32, 127)] + ['\n']
    vocab = {'<pad>': 0, '<bos>': 1, '<eos>': 2, '<unk>': 3}
    for ch in chars:
        vocab[ch] = len(vocab)
    tk = Tokenizer(models.WordLevel(vocab, unk_token='<unk>'))
    tk.pre_tokenizer = pre_tokenizers.Split('', 'isolated')
    tk.decoder = decoders.Fuse()
    from tokenizers.processors import TemplateProcessing
    tk.post_processor = TemplateProcessing(single='<bos> $A', special_tokens=[('<bos>', 1)])
    tok = PreTrainedTokenizerFast(tokenizer_object=tk, bos_token='<bos>',
                                  eos_token='<eos>', pad_token='<pad>',
                                  unk_token='<unk>')
    torch.manual_seed(0)
    cfg = Gemma2Config(vocab_size=len(vocab), hidden_size=32, intermediate_size=48,
                       num_hidden_layers=3, num_attention_heads=2,
                       num_key_value_heads=1, head_dim=16, pad_token_id=0,
                       bos_token_id=1, eos_token_id=2, sliding_window=64,
                       attn_implementation='eager')
    model = Gemma2ForCausalLM(cfg).eval()
    return tok, model


def selftest():
    n = 0
    # 1. Pure helpers.
    assert is_stop('blah\nAnswer: (B)') and not is_stop('Answer: (B) and more')
    assert trim_generated([5, 6, 7, 0, 0], 3, {2}, 0, 10) == [5, 6, 7]
    assert trim_generated([5, 2, 9], None, {2}, 0, 10) == [5, 2]
    offs = [(0, 0), (0, 2), (2, 5), (5, 9), (9, 12)]
    assert cue_token_span(offs, 'abcdefghijkl', 'efgh') == (2, 3)
    n += 1
    try:
        import torch  # noqa: F401
        tok, model = _tiny()
    except ImportError:
        print(f'e49_source_run selftest: {n}/1 passed (torch absent; model tests skipped)')
        return
    import torch
    # 2. Offsets map the cue sentence to exactly its characters.
    prompt = 'Q: x\n\nI am a person, and I think (A).\n\nThink.\nResponse:'
    cue = 'I am a person, and I think (A).'
    enc = tok(prompt, add_special_tokens=True, return_offsets_mapping=True)
    a, b = cue_token_span(enc['offset_mapping'], prompt, cue)
    assert tok.decode(enc['input_ids'][a:b + 1]) == cue, tok.decode(enc['input_ids'][a:b + 1])
    n += 1
    # 3. Batched left-padded generation equals single-row generation (fp32).
    prompts = [prompt, 'Short prompt.\nResponse:']
    batch = generate_batch(tok, model, prompts, 12)
    for p, (ids, pids) in zip(prompts, batch):
        single = generate_batch(tok, model, [p], 12)[0]
        assert single == (ids, pids), (single, ids)
    n += 1
    # 4. The pre-hook captures exactly the down_proj input at a position.
    mods = mlp_modules(model)
    seen = {}
    h = mods[1].register_forward_pre_hook(lambda m, a: seen.__setitem__('x', a[0][0].clone()))
    full = torch.tensor([enc['input_ids']])
    with torch.inference_mode():
        out = model(input_ids=full, output_hidden_states=True, use_cache=False)
    h.remove()
    assert seen['x'].shape == (full.shape[1], model.config.intermediate_size)
    # The captured tensor is what down_proj consumes: feeding it back through
    # down_proj reproduces the block's MLP output computed by a forward hook.
    got = {}
    h2 = mods[1].register_forward_hook(lambda m, a, o: got.__setitem__('y', o[0].clone()))
    with torch.inference_mode():
        model(input_ids=full, use_cache=False)
    h2.remove()
    with torch.inference_mode():
        y = mods[1](seen['x'])
    assert torch.allclose(y, got['y'], atol=1e-6)
    n += 1
    # 5. Zero-shift hooks are a no-op; a shift changes the logits only from
    #    the start position onward, in prefill and through cached decoding.
    starts = [a]
    plan0 = {1: ([0, 5], [0.0, 0.0])}
    with ShiftHooks(model, plan0, starts), torch.inference_mode():
        z0 = model(input_ids=full, use_cache=False).logits
    with torch.inference_mode():
        ref = model(input_ids=full, use_cache=False).logits
    assert torch.allclose(z0, ref)
    plan = {1: ([0, 5, 7], [3.0, -2.0, 1.5])}
    with ShiftHooks(model, plan, starts), torch.inference_mode():
        z1 = model(input_ids=full, use_cache=False).logits
    diff = (z1 - ref).abs().amax(-1)[0]
    assert diff[:a].max() < 1e-6 and diff[a:].max() > 1e-4
    n += 1
    # 6. Shifted generation with cache equals shifted generation computed by
    #    full re-forward at each step (hook position tracking is correct).
    def hooks(pad_lens, n_max):
        return ShiftHooks(model, plan, [pl + a for pl in pad_lens])
    (ids_h, _), = generate_batch(tok, model, [prompt], 8, hooks=hooks)
    seq = list(enc['input_ids'])
    for _ in range(len(ids_h)):
        with ShiftHooks(model, plan, [a]), torch.inference_mode():
            nxt = int(model(input_ids=torch.tensor([seq]), use_cache=False)
                      .logits[0, -1].argmax())
        seq.append(nxt)
        if nxt == 2:
            break
    assert seq[len(enc['input_ids']):][:len(ids_h)] == ids_h, (seq, ids_h)
    n += 1
    # 7. Left padding shifts each row's start; batched patched = single patched.
    (single_ids, _), = generate_batch(tok, model, [prompt], 8, hooks=hooks)
    both = generate_batch(tok, model, [prompt, 'Short prompt.\nResponse:'], 8,
                          hooks=lambda pl, nm: ShiftHooks(model, plan, [pl[0] + a, pl[1] + 10**6]))
    assert both[0][0] == single_ids
    n += 1
    print(f'e49_source_run selftest: {n}/7 passed')


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('stage', nargs='?', choices=('generate', 'capture', 'export', 'patch'))
    p.add_argument('--run')
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--limit', type=int)
    p.add_argument('--layer-stride', type=int, default=2)
    p.add_argument('--layers', type=int, nargs='+')
    p.add_argument('--source', default='generations.pkl')
    p.add_argument('--families')
    p.add_argument('--cell', default='third_expert_certain')
    p.add_argument('--arms', nargs='+')
    p.add_argument('--dose', type=float, default=1.0)
    p.add_argument('--site', choices=('from_cue', 'after_cue'), default='from_cue')
    p.add_argument('--selftest', action='store_true')
    a = p.parse_args(argv)
    if a.selftest:
        selftest()
        return 0
    if not a.stage or not a.run:
        p.error('stage and --run are required')
    if a.stage == 'generate':
        stage_generate(a.run, a.batch_size, a.limit)
    elif a.stage == 'capture':
        stage_capture(a.run, a.layer_stride, a.layers, limit=a.limit)
    elif a.stage == 'export':
        stage_export(a.run, a.source)
    elif a.stage == 'patch':
        if not a.families:
            p.error('--families is required for patch')
        stage_patch(a.run, a.families, a.cell, a.arms, a.batch_size, a.dose,
                    a.site, a.limit)
    return 0


if __name__ == '__main__':
    sys.exit(main())
