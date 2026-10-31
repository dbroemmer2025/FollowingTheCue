#!/usr/bin/env python3
"""Summary of the direct-effect (prompt-trace counterfactual) results.

Reads reports/direct_effect/direct_effect_summary.json and direct_effect_rows.csv
(run `e47_direct_effect.py analyze` first; this script re-runs it if they are
missing or older than the measurement files).

Tables:
  KEY RESULTS    compact view of D, decisive cue dependence, and the validated
                 relocation contrast (RC)
  PRIMARY       D = logit P(cue | cue prompt, cued rationale)
                  - logit P(cue | filler prompt, cued rationale), nats,
                item mean [95% CI], by cue family / switch / silence
  BEHAVIOURAL   dP, decisive rate (cue option is argmax only with the cue line),
                relocation-follows rate (Amendment 1)
  CONTROLS      D_removed, D_source (placebo), relocation shift, rationale carry
  RELOCATION    template-matched relocation contrast (Amendment 2) and the
                token gate status; position offset of the filler prompt
  CUE FAMILY    authority minus user preference, cell-stratified source-item
                paired bootstrap (plus pooled descriptive rates)
  ACK           acknowledged vs silent switchers (cells with >= 10 acknowledged)
  COHERENCE     decisive rate vs residual-state gain left after the LLM reader
                (A1 nested), Spearman across 10 cells -- hypothesis-generating,
                not independent confirmation

  python code/show_e47_direct_effect.py
  python code/show_e47_direct_effect.py --csv reports/direct_effect/show.csv --latex
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))

from show_e47_gap_summary import print_table  # noqa: E402
from show_e47_text_baselines import labels  # noqa: E402

DE = ROOT / 'reports' / 'direct_effect'
NESTED = ROOT / 'reports' / 'text_baselines' / 'nested'


def ensure_summary():
    summ, rows = DE / 'direct_effect_summary.json', DE / 'direct_effect_rows.csv'
    # The token-gate artifacts determine whether relocation contrasts are
    # reportable, so a newly written gate must invalidate an older summary.
    inputs = [*DE.glob('*.jsonl'), *DE.glob('*.tokens.json'),
              ROOT / 'code' / 'e47_direct_effect.py']
    newest = max((p.stat().st_mtime for p in inputs if p.exists()), default=0)
    if not summ.exists() or not rows.exists() or summ.stat().st_mtime < newest:
        import contextlib
        import io
        import e47_direct_effect as E
        with contextlib.redirect_stdout(io.StringIO()):
            E.analyze()
    return json.loads(summ.read_text()), list(csv.DictReader(rows.open()))


def ci(s, digits=2):
    if not s or not s.get('n_rows'):
        return '--'
    return f"{s['mean']:+.{digits}f} [{s['ci95'][0]:+.{digits}f},{s['ci95'][1]:+.{digits}f}]"


def rate(g, k):
    r, n = g.get(f'{k}_rate'), g.get(f'{k}_n')
    return '--' if r is None else f'{r:.3f} (n={n})'


def num(v):
    return None if v in ('', None, 'None') else float(v)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--csv', help='write the per-cell table to this path')
    p.add_argument('--latex', action='store_true', help='print LaTeX rows')
    a = p.parse_args(argv)
    summary, rows = ensure_summary()
    cells = summary['cells']
    lab = {c: labels(c) for c in cells}

    print('DIRECT EFFECT OF THE CUE BEYOND THE RATIONALE  (prompt-trace counterfactual diagnostic)')
    print('D = logit P(cue | cue prompt, cued rationale) - logit P(cue | filler prompt, cued rationale); nats\n')
    print('KEY RESULTS')
    print_table(['model', 'task', 'kept', 'D: cue vs filler', 'decisive among switchers',
                 'RC: moved letter', 'reloc swap among switchers'],
                [[*lab[c], f"{v['n_kept']}/{v['n_measured']}",
                  ci(v['groups']['all']['D']),
                  rate(v['groups']['switch'], 'decisive'),
                  ci(v['groups']['all'].get('reloc_contrast')),
                  rate(v['groups']['switch'], 'reloc_swap')]
                 for c, v in cells.items()])
    print('\n  D: removing the cue changes cue-option preference with the rationale fixed.')
    print('  decisive: removing the cue changes the four-option argmax, among actual cue switchers.')
    print('  RC: changing only the cue letter shifts relative preference toward that letter;')
    print('  reloc swap: the four-option argmax follows that moved letter, among actual switchers.\n')

    print('PRIMARY (item mean [95% CI])')
    print_table(['model', 'task', 'kept', 'all', 'authority', 'user', 'switch & silent', 'no switch'],
                [[*lab[c], f"{v['n_kept']}/{v['n_measured']}", ci(v['groups']['all']['D']),
                  ci(v['groups']['authority']['D']), ci(v['groups']['user']['D']),
                  ci(v['groups']['switch_silent']['D']), ci(v['groups']['no_switch']['D'])]
                 for c, v in cells.items()])
    fails = {c: v['gate_failures'] for c, v in cells.items() if v['gate_failures']}
    if fails:
        print('\n  gate exclusions: ' + '; '.join(f'{labels(c)[0]} {labels(c)[1]} {f}' for c, f in fails.items()))

    print('\nBEHAVIOURAL COMPANIONS (Amendment 1)')
    print_table(['model', 'task', 'dP all', 'dP switch&silent', 'flip', 'decisive (switchers)',
                 'reloc follows'],
                [[*lab[c], ci(v['groups']['all']['dP']), ci(v['groups']['switch_silent']['dP']),
                  f"{v['groups']['all']['flip_rate']:.3f}", rate(v['groups']['switch'], 'decisive'),
                  rate(v['groups']['all'], 'reloc_follows')] for c, v in cells.items()])
    print('\n  dP = P(cue|cue prompt) - P(cue|filler); decisive = cue option is the 4-way argmax only')
    print('  with the cue line (rationale fixed); reloc follows = argmax moves to the relocated letter')
    print('  (rows where that letter is not the model\'s own answer). Non-decisive switchers are')
    print('  "cue-consistent under the filler counterfactual" -- this does not show that the cue acted')
    print('  through the rationale (no mediation or rationalization claim). D is a broad local')
    print('  sensitivity: it is large for non-switchers too.')
    print('  The filler is NOT token-position matched (shorter than the cue sentence); the relocation')
    print('  contrast below is the template-matched robustness result.')

    print('\nRELOCATION CONTRAST (template-matched; cued rationale fixed)')
    print_table(['model', 'task', 'token gate', 'RC all', 'RC switch & silent', 'reloc swap (switchers)',
                 'filler offset (tokens)'],
                [[*lab[c], v.get('reloc_token_gate', 'not run'), ci(v['groups']['all'].get('reloc_contrast')),
                  ci(v['groups']['switch_silent'].get('reloc_contrast')),
                  rate(v['groups']['switch'], 'reloc_swap'),
                  ('--' if v.get('mean_filler_offset_tokens') is None
                   else f"{v['mean_filler_offset_tokens']:+.1f}")]
                 for c, v in cells.items()])
    print('\n  RC = [log P(y|X_y) - log P(x|X_y)] - [log P(y|X_x) - log P(x|X_x)], X_y = the cue line with only')
    print('  the letter changed to y. reloc swap = argmax x under X_x and y under X_y. Until the token gate')
    print('  is run (`e47_direct_effect.py tokens --run ...`, Colab), RC and reloc swap are omitted.')

    print('\nCONTROLS AND SECONDARY (all rows)')
    print_table(['model', 'task', 'D', 'D_removed', 'D_source (placebo)', 'reloc shift',
                 'carry (cue prompt)', 'D | uncued rationale'],
                [[*lab[c], ci(g['D']), ci(g['D_removed']), ci(g['D_source']),
                  ci(g['reloc_shift_valid']), ci(g['carry_cue']), ci(g['D_uncued_rationale'])]
                 for c, v in cells.items() for g in [v['groups']['all']]])
    print('\n  D_source ~ D: the effect is the suggested answer, not the mention of a source.')
    print('  carry = logit P(cue|cue prompt, cued rationale) - logit P(cue|cue prompt, uncued rationale):')
    print('  cross-world and descriptive only (not mediation).')

    fam = summary.get('family_paired') or {}
    if fam:
        print('\nCUE FAMILY: authority minus user preference (cell-stratified, source-item paired bootstrap)')
        print_table(['quantity', 'auth - user', '95% CI', 'pairs'],
                    [[k, f"{v['pooled_mean_auth_minus_user']:+.3f}",
                      f"[{v['ci95'][0]:+.3f},{v['ci95'][1]:+.3f}]", v['n_pairs']] for k, v in fam.items()])
        print('\n  per-cell paired means (auth - user):')
        print_table(['model', 'task'] + list(fam),
                    [[*lab[c]] + [('--' if c not in fam[k]['per_cell'] else
                                   f"{fam[k]['per_cell'][c]['mean']:+.3f}") for k in fam] for c in cells])
        print('\n  decisive_both_switch uses items where both cue families switched.')
    print('\nCUE FAMILY (pooled switchers, descriptive)')
    fam_rows = []
    for fam, name in (('authority', 'authority'), ('sycophancy', 'user preference')):
        sub = [r for r in rows if r['cue_type'] == fam and r['z'] == '1']
        dec = [int(r['decisive']) for r in sub if r['decisive'] not in ('', 'None')]
        fam_rows.append([name, len(sub), f'{np.mean(dec):.3f} ({sum(dec)})',
                         f"{np.mean([num(r['dP']) for r in sub]):.3f}",
                         f"{np.median([num(r['D']) for r in sub]):.2f}"])
    print_table(['family', 'switchers', 'decisive', 'mean dP', 'median D'], fam_rows)

    print('\nACKNOWLEDGED vs SILENT SWITCHERS (cells with >= 10 acknowledged switchers)')
    ack_rows = []
    for c in cells:
        A = [r for r in rows if r['cell'] == c and r['z'] == '1' and r['ack'] == '1']
        S = [r for r in rows if r['cell'] == c and r['z'] == '1' and r['ack'] == '0']
        if len(A) >= 10:
            m = lambda L, k: np.mean([num(r[k]) for r in L if num(r[k]) is not None])
            ack_rows.append([*lab[c], len(A), f"{m(A, 'decisive'):.3f}", f"{m(A, 'D'):+.2f}",
                             len(S), f"{m(S, 'decisive'):.3f}", f"{m(S, 'D'):+.2f}"])
    print_table(['model', 'task', 'ack n', 'ack decisive', 'ack D', 'silent n', 'silent decisive',
                 'silent D'], ack_rows)

    xs, ys, cc = [], [], []
    for c, v in cells.items():
        path = NESTED / f'{c}__llm.json'
        if path.exists():
            xs.append(v['groups']['switch']['decisive_rate'])
            ys.append(json.loads(path.read_text())['G_conditional_on_feature']['bits'])
            cc.append(c)
    if len(xs) >= 3:
        from scipy.stats import spearmanr
        print('\nCOHERENCE CHECK: decisive rate vs residual-state gain left after the LLM reader (A1 nested)')
        print_table(['model', 'task', 'decisive (switchers)', 'G | LLM (bits)'],
                    [[*lab[c], f'{x:.3f}', f'{y:+.3f}'] for c, x, y in zip(cc, xs, ys)])
        sp = spearmanr(xs, ys)
        print(f'\n  Spearman rho = {sp.statistic:.2f} (p = {sp.pvalue:.1g}) across {len(xs)} cells. Hypothesis-')
        print('  generating only: ten points, shared outcome construction, OLMo cells influential.')

    if a.latex:
        print('\n% model & task & D [CI] & dP & decisive & reloc follows \\\\')
        for c, v in cells.items():
            g = v['groups']
            print(f"{lab[c][0]} & {lab[c][1]} & {ci(g['all']['D'])} & {g['all']['dP']['mean']:.3f} & "
                  f"{g['switch']['decisive_rate']:.2f} & {g['all']['reloc_follows_rate']:.2f} \\\\")
    if a.csv:
        out = Path(a.csv)
        keys = ['model', 'task', 'n_kept', 'D', 'D_lo', 'D_hi', 'dP', 'decisive_switch',
                'reloc_follows', 'D_source', 'D_removed']
        with out.open('w', newline='') as fh:
            w = csv.writer(fh)
            w.writerow(keys)
            for c, v in cells.items():
                g = v['groups']['all']
                w.writerow([*lab[c], v['n_kept'], g['D']['mean'], *g['D']['ci95'], g['dP']['mean'],
                            v['groups']['switch']['decisive_rate'], g['reloc_follows_rate'],
                            g['D_source']['mean'], g['D_removed']['mean']])
        print(f'\nwrote {out}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
