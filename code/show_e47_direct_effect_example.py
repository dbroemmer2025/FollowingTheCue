#!/usr/bin/env python3
"""Inspect individual direct-effect rows: the exact prompt variants, the fixed
rationales, and the per-condition answer probabilities and log-odds.

List rows (worst/most interesting first) in one cell:
  python code/show_e47_direct_effect_example.py --cell olmo_2_1124_7b_instruct_temporal_sequences
  python code/show_e47_direct_effect_example.py --cell <cell> --filter decisive --family user
  python code/show_e47_direct_effect_example.py --cell <cell> --filter reloc-follows --sort RC

Show one row in full:
  python code/show_e47_direct_effect_example.py --cell <cell> --filter decisive --show 1
  python code/show_e47_direct_effect_example.py --cell <cell> --show-id 723e4f56
  ... --full-prompt filler     # print the complete prompt for one variant

Filters: all, switch, no-switch, silent (switch & not acknowledged), acknowledged,
decisive, not-decisive (switchers whose answer the rationale alone produces),
reloc-follows.  Sort: dP (default), D, -D, RC, -RC.  ``reloc-follows`` is the
strongest concrete relocation example: the answer argmax is the factual cue
letter under $X_x$ and the relocated letter under $X_y$.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import pickle
import sys
import textwrap
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
csv.field_size_limit(10 ** 9)

DE = ROOT / 'reports' / 'direct_effect'
W = 100


def logit(p):
    p = min(max(p, 1e-12), 1 - 1e-12)
    return math.log(p / (1 - p))


def load(cell):
    import e47_direct_effect as E
    recs = {json.loads(l)['id']: json.loads(l) for l in (DE / f'{cell}.jsonl').read_text().splitlines()}
    tpath = DE / f'{cell}.tokens.json'
    token_rows = json.loads(tpath.read_text())['rows'] if tpath.exists() else {}
    run = ROOT / 'e47' / f'{cell}_thinking_v10'
    ack = E.load_ack(run)
    correct = ground_truth(run)
    metadata = item_metadata(run)
    rows = []
    for r in recs.values():
        ok = all(r['gates'].get(g) for g in ('retokenize', 'cued_rationale', 'answer_reproduced', 'options_ok'))
        d = E.derive(r) if ok else {}
        token = token_rows.get(r['id'])
        if ok:
            E.apply_relocation_gate(d, token)
        rows.append({**r, **d, 'kept': ok, 'ack': ack.get(r['id']),
                     'correct': correct.get(r['id']), 'reloc_token': token,
                     **metadata.get(r['id'], {})})
    return rows


def ground_truth(run):
    """Row id -> the item's correct option letter (design.json items)."""
    design = json.loads((Path(run) / 'design.json').read_text())
    by_index = {int(it['index']): it.get('correct_letter') for it in design['items']}
    traces = pickle.loads((Path(run) / 'traces.pkl').read_bytes())['rows']
    return {t['id']: by_index.get(int(t['index'])) for t in traces}


def item_metadata(run):
    """Response ID -> viewer arguments shared with show_with_why.sh."""
    traces = pickle.loads((Path(run) / 'traces.pkl').read_bytes())['rows']
    return {
        trace['id']: {
            'item_index': int(trace['index']),
            'condition': trace.get('condition'),
        }
        for trace in traces
    }


def select(rows, flt, family):
    keep = [r for r in rows if r['kept']]
    if family != 'all':
        keep = [r for r in keep if r['cue_type'] == ('sycophancy' if family == 'user' else 'authority')]
    f = {'all': lambda r: True, 'switch': lambda r: r['z'] == 1, 'no-switch': lambda r: r['z'] == 0,
         'silent': lambda r: r['z'] == 1 and r['ack'] == 0,
         'acknowledged': lambda r: r['z'] == 1 and r['ack'] == 1,
         'decisive': lambda r: r.get('decisive') == 1,
         'not-decisive': lambda r: r['z'] == 1 and r.get('decisive') == 0,
         'reloc-follows': lambda r: r.get('reloc_follows') == 1}[flt]
    return [r for r in keep if f(r)]


def rationale_text(row):
    from e47_design import answer_parse_span
    ep = row['episode']
    span = answer_parse_span(ep)[3]
    return ep[:span[0]] if span else ep


def wrap(text, indent=''):
    for line in text.splitlines():
        print(textwrap.fill(line, W, initial_indent=indent, subsequent_indent=indent) if line else '')


def show(cell, r, full_prompt=None):
    import e47_direct_effect as E
    run = ROOT / 'e47' / f'{cell}_thinking_v10'
    specs = {s['id']: s for s in E.plan_cell(run)[0]}
    s = specs[r['id']]
    traces = {t['id']: t for t in pickle.loads((run / 'traces.pkl').read_bytes())['rows']}
    x, u, y, fin = r['cue'], r['uncued_answer'], r['relocated'], r['final_answer']
    fam = 'user preference' if r['cue_type'] == 'sycophancy' else 'authority'
    print('=' * W)
    print(f"{cell}   item {r.get('item_index', '--')} / {r.get('condition', '--')}   "
          f"row {r['id'][:16]}   {fam} cue")
    gt = r.get('correct')
    tag = lambda letter: ' (correct)' if letter == gt else ''
    print(f"ground truth {gt}   cue letter {x}{tag(x)}   uncued answer {u}{tag(u)}   "
          f"model's cued answer {fin}{tag(fin)}   relocated letter {y}{tag(y)}")
    print(f"switch-to-cue Z={r['z']}   acknowledged={r['ack']}   "
          f"cue points to the {'correct' if x == gt else 'wrong'} answer; the switch made the model "
          f"{'correct' if fin == gt else 'wrong'} (uncued it was {'correct' if u == gt else 'wrong'})")
    if r.get('item_index') is not None and r.get('condition'):
        print(f"show_with_why: ./show_with_why.sh {r['item_index']} --run-dir "
              f"../e47/{cell}_thinking_v10 --condition {r['condition']}")
    print('=' * W)

    hint = E.cue_line(r['cue_type'], x)
    shared = s['prompts']['cue'].replace(hint, '<<< CUE LINE >>>')
    print('PROMPT (identical in every variant except the marked line)')
    print('-' * W)
    wrap(shared)
    print('-' * W)
    print('THE MARKED LINE IN EACH PROMPT VARIANT')
    variants = {'cue': hint, 'filler': E.FILLER, 'removed': '(line and its blank line deleted '
                '= the saved uncued prompt)', 'source': E.SOURCE_PLACEBO[r['cue_type']],
                'relocated': E.cue_line(r['cue_type'], y)}
    for k, v in variants.items():
        print(f'  {k:10s} {v}')
    if full_prompt:
        print('-' * W + f'\nFULL PROMPT: {full_prompt}')
        wrap(s['prompts'][full_prompt])

    print('-' * W)
    print('FIXED RATIONALE A: the cued response (as scored: generated tokens up to the answer letter)')
    wrap(rationale_text(traces[r['id']]), '  ')
    print(f"  [the model then wrote: ({fin}); ground truth ({gt})]")
    print('-' * W)
    print('FIXED RATIONALE B: the uncued response for the same item')
    if r['gates'].get('uncued_rationale'):
        wrap(rationale_text(traces[s['uncued_id']]), '  ')
        print(f"  [the uncued response wrote: ({u}); ground truth ({gt})]")
    else:
        print('  (no pre-answer boundary; not scored)')

    print('-' * W)
    print('NEXT-TOKEN PROBABILITY AT THE ANSWER POSITION (full vocabulary)')
    print(f'  ground truth={gt or "--"}   factual cue={x}   cued response={fin}   '
          f'relocated cue={y}')
    print('  D comparison: factual cue vs filler.   RC comparison: factual cue vs relocated cue.')
    print('  Cue delta is factual-cue logit minus this row; RC is shown on its relocated-cue row.')
    hdr = (f"{'prompt':10s} {'rationale':9s}  {'P(A)':>6s} {'P(B)':>6s} {'P(C)':>6s} {'P(D)':>6s}  "
           f"{'argmax':6s} {'logit P(cue)':>12s} {'log P(cue)/P(unc)':>17s} {'P(reloc)':>8s} {'P(gt)':>6s} "
           f"{'cue d(nat)':>10s} {'RC(nat)':>8s}")
    print(hdr)
    nats = lambda value: '--' if value is None else f'{value:+.2f}'
    for rat in ('cued', 'uncued'):
        for pr in ('cue', 'filler', 'removed', 'source', 'relocated'):
            sc = r['scores'].get(f'{pr}|{rat}')
            if not sc:
                continue
            if sc.get('status') != 'ok':
                print(f"{pr:10s} {rat:9s}  ({sc.get('status')})")
                continue
            P = sc['p']
            am = max(P, key=P.get)
            lo = math.log(max(P[x], 1e-12)) - math.log(max(P[u], 1e-12))
            mark = {
                ('cue', 'cued'): ' <- D, RC factual',
                ('filler', 'cued'): ' <- D comparison',
                ('relocated', 'cued'): ' <- RC comparison',
            }.get((pr, rat), '')
            cue_delta = {
                ('filler', 'cued'): r.get('D'),
                ('removed', 'cued'): r.get('D_removed'),
                ('source', 'cued'): r.get('D_source'),
            }.get((pr, rat))
            rc_value = r.get('reloc_contrast') if (pr, rat) == ('relocated', 'cued') else None
            pgt = f'{P[gt]:6.3f}' if gt in P else '    --'
            print(f"{pr:10s} {rat:9s}  " + ' '.join(f'{P[c]:6.3f}' for c in 'ABCD')
                  + f"  {am + ('*' if am == gt else ''):6s} {logit(P[x]):+12.2f} {lo:+17.2f} "
                  f"{P[y]:8.3f} {pgt} {nats(cue_delta):>10s} {nats(rc_value):>8s}{mark}")
    print('  argmax* = the ground-truth answer')
    print('-' * W)
    f = lambda v, d=2: '--' if v is None else f'{v:+.{d}f}'
    print(f"dP = {f(r.get('dP'), 3)}   decisive = {r.get('decisive')}   flip = {r.get('flip')}   "
          f"relocation shift = {f(r.get('reloc_shift_valid'))}   reloc follows = {r.get('reloc_follows')}")
    tk = r.get('reloc_token')
    gate = ('not run' if tk is None else
            f"{'pass' if tk['reloc_ok'] else 'FAIL'} (len {tk['len_cue']} vs {tk['len_relocated']}, "
            f"{tk['n_diff']} differing token(s); filler offset {tk['offset_filler']:+d} tokens)")
    print(f"reloc swap = {r.get('reloc_swap')}   token gate: {gate}")
    print(f"carry (cue prompt, cued vs uncued rationale) = {f(r.get('carry_cue'))}   "
          f"D | uncued rationale = {f(r.get('D_uncued_rationale'))}")
    if r.get('decisive') == 1:
        print('\nReading: with this exact rationale, the cue option wins only when the cue line is present.')
    elif r['z'] == 1:
        print('\nReading: cue-consistent under the filler counterfactual -- with the cue line replaced, this'
              ' rationale\nstill yields the cue option. (This does not show the cue acted through the rationale.)')


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--cell', required=True)
    p.add_argument('--filter', default='switch',
                   choices=('all', 'switch', 'no-switch', 'silent', 'acknowledged', 'decisive',
                            'not-decisive', 'reloc-follows'))
    p.add_argument('--family', choices=('all', 'authority', 'user'), default='all')
    p.add_argument('--sort', choices=('dP', 'D', '-D', 'RC', '-RC'), default='dP')
    p.add_argument('--n', type=int, default=20)
    p.add_argument('--show', type=int)
    p.add_argument('--show-id')
    p.add_argument('--full-prompt', choices=('cue', 'filler', 'removed', 'source', 'relocated'))
    a = p.parse_args(argv)
    rows = load(a.cell)
    if a.show_id:
        hits = [r for r in rows if r['id'].startswith(a.show_id)]
        if len(hits) != 1:
            p.error(f'--show-id matched {len(hits)} rows')
        show(a.cell, hits[0], a.full_prompt)
        return 0
    sel = select(rows, a.filter, a.family)
    key = {'dP': lambda r: -(r.get('dP') or 0), 'D': lambda r: -(r.get('D') or 0),
           '-D': lambda r: (r.get('D') or 0),
           'RC': lambda r: -(r.get('reloc_contrast') if r.get('reloc_contrast') is not None
                              else float('-inf')),
           '-RC': lambda r: (r.get('reloc_contrast') if r.get('reloc_contrast') is not None
                              else float('inf'))}[a.sort]
    sel.sort(key=key)
    if a.show:
        if not 1 <= a.show <= len(sel):
            p.error(f'--show must be 1..{len(sel)}')
        show(a.cell, sel[a.show - 1], a.full_prompt)
        return 0
    excluded = sum(not r['kept'] for r in rows)
    print(f"{a.cell}: {len(sel)} rows ({a.filter}, family={a.family}, sorted by {a.sort}); "
          f"{excluded} excluded by gates\n")
    print(f"{'#':>3s} {'item':>4s} {'id':8s} {'fam':4s} {'gt':2s} {'x':1s} {'u':1s} {'fin':3s} {'Z':1s} {'ack':3s} "
          f"{'P(x)|cue':>8s} {'P(x)|fill':>9s} {'D':>6s} {'dP':>6s} {'RC':>6s} {'dec':3s} {'reloc':5s}")
    for i, r in enumerate(sel[:a.n], 1):
        pc = r['scores']['cue|cued']['p'][r['cue']]
        pf = r['scores']['filler|cued']['p'][r['cue']]
        print(f"{i:3d} {str(r.get('item_index', '--')):>4s} {r['id'][:8]} "
              f"{('user' if r['cue_type'] == 'sycophancy' else 'auth'):4s} "
              f"{str(r['correct']):2s} {r['cue']} {r['uncued_answer']} {r['final_answer']:3s} {r['z']} {str(r['ack']):3s} "
              f"{pc:8.3f} {pf:9.3f} {r['D']:+6.2f} {r['dP']:+6.3f} "
              f"{(r['reloc_contrast'] if r['reloc_contrast'] is not None else float('nan')):+6.2f} "
              f"{str(r['decisive']):3s} "
              f"{str(r['reloc_follows']):5s}")
    if len(sel) > a.n:
        print(f'... {len(sel) - a.n} more (use --n)')
    print(f'\nfull example: --cell {a.cell} --filter {a.filter} --show N   '
          '(add --full-prompt filler to print a whole variant)')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
