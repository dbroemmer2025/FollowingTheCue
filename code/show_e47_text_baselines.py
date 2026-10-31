#!/usr/bin/env python3
"""Compare the A1 text readers with the frozen State-Over-Text numbers.

For every cell in DEFAULT_RUN_DIRS (or --runs), at the frozen primary
checkpoint (initial/pre_answer), this prints:

  CHANNEL LOSSES   prior, frozen n-gram text, heuristic, LLM (calibrated),
                   LLM on the redacted rationale, state, joint
  ADVANTAGE/GAIN   frozen A_ST = L_text - L_state and G_SOT = L_text - L_joint,
                   next to the same quantities with the n-gram text reader
                   replaced by the heuristic or the LLM reader:
                       A^X = L_X - L_state      G^X = L_X - L_joint
                   Positive = the residual-state reader is still ahead;
                   negative = the text-side reader X is better.
                   R = (L_prior - L_X) / (L_prior - L_joint) and the
                   preregistered class (approx / partial / much_less).
  NESTED           G on the cheap linear grid, and G conditional on the
                   heuristic or the LLM feature (state added on top of it).
  AUC              discrimination of the switch-to-cue target per reader.

Inputs: e47/<run>/analysis.json (frozen), reports/text_baselines/
text_baselines.json (from `e47_text_baselines.py report`), and
reports/text_baselines/nested/<cell>__{heuristic,llm}.json (Colab).

    python code/show_e47_text_baselines.py
    python code/show_e47_text_baselines.py --csv reports/text_baselines/compare.csv
    python code/show_e47_text_baselines.py --latex
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))

from show_e47_gap_summary import fmt, print_table, resolve_run  # noqa: E402

TB = ROOT / 'reports' / 'text_baselines'
MODEL = {'qwen2_5_7b': 'Qwen2.5-7B', 'qwen2_7b_it': 'Qwen2.5-7B-Inst.',
         'gemma_2_9b_it': 'Gemma-2-9B-IT',
         'mistral_nemo_instruct_2407': 'Mistral-Nemo-Inst.',
         'olmo_2_1124_7b_instruct': 'OLMo-2-7B-Inst.'}
TASK = {'movie_recommendations': 'movie', 'temporal_sequences': 'temporal'}


def cell_name(run):
    return Path(run).name.replace('_thinking_v10', '')


def labels(cell):
    for task_key, task in TASK.items():
        if cell.endswith('_' + task_key):
            m = cell[:-len(task_key) - 1]
            return MODEL.get(m, m), task
    return cell, '--'


def ci(pair, digits=3):
    if not pair:
        return '--'
    return f'[{pair[0]:+.{digits}f},{pair[1]:+.{digits}f}]'


def signed(value):
    return '--' if value is None else f'{value:+.3f}'


def load(run, tb, nested_dir):
    run = resolve_run(run)
    cell = cell_name(run)
    analysis = json.loads((run / 'analysis.json').read_text())
    key = analysis['primary']
    frozen = analysis['checkpoints'][key]
    c = (tb.get('cells') or {}).get(cell)
    if c is None:
        raise ValueError(f'{cell}: not in text_baselines.json; run '
                         '`e47_text_baselines.py report --model <MODEL>`')
    ce = c['ce']
    # The A1 rows reproduce the frozen losses exactly; refuse silent drift.
    for ch in ('text', 'state', 'joint'):
        if abs(ce[ch] - frozen[f'ce_{ch}']) > 1e-6:
            raise ValueError(f'{cell}: A1 {ch} loss {ce[ch]} != frozen {frozen[f"ce_{ch}"]}')
    model, task = labels(cell)
    row = {'cell': cell, 'model': model, 'task': task, 'checkpoint': key,
           'n_rows': c['n_rows'], 'n_items': c['n_items'],
           'ce_prior': ce['prior'], 'ce_text': ce['text'],
           'ce_heuristic': ce.get('heuristic'), 'ce_llm': ce.get('llm_full'),
           'ce_llm_redacted': ce.get('llm_redacted'),
           'ce_state': ce['state'], 'ce_joint': ce['joint'],
           'A_ST': ce['text'] - ce['state'], 'G_SOT': ce['text'] - ce['joint'],
           'G_SOT_ci': (frozen.get('gap') or {}).get('ci95')}
    for tag, k in (('heur', 'heuristic'), ('llm', 'llm_full'), ('llmred', 'llm_redacted')):
        v = (c.get('vs_joint') or {}).get(k)
        if ce.get(k) is None or v is None:
            row.update({f'A_{tag}': None, f'G_{tag}': None, f'G_{tag}_ci': None,
                        f'R_{tag}': None, f'class_{tag}': None})
            continue
        row.update({f'A_{tag}': ce[k] - ce['state'], f'G_{tag}': v['Delta_bits'],
                    f'G_{tag}_ci': v['Delta_ci95'], f'R_{tag}': v['R_share_of_joint_gain'],
                    f'class_{tag}': v['class']})
    for k, a in (c.get('auc') or {}).items():
        row[f'auc_{k}'] = a
    for feat in ('heuristic', 'llm'):
        path = nested_dir / f'{cell}__{feat}.json'
        if path.exists():
            n = json.loads(path.read_text())
            row['G_cheap'] = n['G_cheap_grid']['bits']
            row['G_cheap_ci'] = n['G_cheap_grid']['ci95']
            row[f'G_given_{feat}'] = n['G_conditional_on_feature']['bits']
            row[f'G_given_{feat}_ci'] = n['G_conditional_on_feature']['ci95']
        else:
            row[f'G_given_{feat}'] = row[f'G_given_{feat}_ci'] = None
    row.setdefault('G_cheap', None)
    row.setdefault('G_cheap_ci', None)
    return row


def mean(rows, key):
    vals = [r[key] for r in rows if r.get(key) is not None]
    return sum(vals) / len(vals) if len(vals) == len(rows) else None


def latex(rows):
    print('\n% LaTeX rows: model & task & L_prior & L_text & L_LLM & L_state & L_joint '
          '& G_SOT & G^LLM & G|LLM \\\\')
    for r in rows:
        cells = [r['model'], r['task']] + [
            fmt(r[k]) for k in ('ce_prior', 'ce_text', 'ce_llm', 'ce_state', 'ce_joint')] + [
            signed(r['G_SOT']), signed(r['G_llm']), signed(r.get('G_given_llm'))]
        print(' & '.join(cells) + r' \\')


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--runs', nargs='+', metavar='DIR')
    p.add_argument('--baselines', default=str(TB / 'text_baselines.json'))
    p.add_argument('--nested-dir', default=str(TB / 'nested'))
    p.add_argument('--csv', metavar='PATH')
    p.add_argument('--latex', action='store_true')
    a = p.parse_args(argv)
    from run_dirs import DEFAULT_RUN_DIRS
    runs = a.runs or DEFAULT_RUN_DIRS
    tb = json.loads(Path(a.baselines).read_text())
    rows = [load(r, tb, Path(a.nested_dir)) for r in runs]
    avg = {'model': 'MEAN (equal weight)', 'task': ''}
    for k in rows[0]:
        if k.startswith(('ce_', 'A_', 'G_', 'R_', 'auc_')) and not k.endswith('_ci'):
            avg[k] = mean(rows, k)

    print(f"A1 TEXT READERS vs FROZEN STATE-OVER-TEXT  (reader model: {tb.get('model')}; "
          f"checkpoint {rows[0]['checkpoint']}; held-out CE bits, lower is better)")
    print('\nCHANNEL LOSSES')
    cols = ['ce_prior', 'ce_text', 'ce_heuristic', 'ce_llm', 'ce_llm_redacted', 'ce_state', 'ce_joint']
    print_table(['model', 'task', 'n', 'prior', 'text(ngram)', 'heuristic', 'LLM',
                 'LLM redacted', 'state', 'joint'],
                [[r['model'], r['task'], r['n_rows']] + [r[k] for k in cols] for r in rows]
                + [[avg['model'], '', ''] + [avg.get(k) for k in cols]])

    print('\nADVANTAGE / GAIN: frozen n-gram text vs the new text readers')
    print_table(['model', 'task', 'A_ST', 'G_SOT', 'G_SOT 95CI',
                 'A^heur', 'G^heur', 'A^LLM', 'G^LLM', 'G^LLM 95CI', 'R_LLM', 'class'],
                [[r['model'], r['task'], signed(r['A_ST']), signed(r['G_SOT']), ci(r['G_SOT_ci']),
                  signed(r['A_heur']), signed(r['G_heur']), signed(r['A_llm']),
                  signed(r['G_llm']), ci(r['G_llm_ci']), fmt(r['R_llm'], 2), r['class_llm'] or '--']
                 for r in rows]
                + [[avg['model'], '', signed(avg['A_ST']), signed(avg['G_SOT']), '',
                    signed(avg.get('A_heur')), signed(avg.get('G_heur')), signed(avg.get('A_llm')),
                    signed(avg.get('G_llm')), '', fmt(avg.get('R_llm'), 2), '']])
    print('\n  A_ST  = L_text - L_state     G_SOT = L_text - L_joint   (frozen n-gram text reader)')
    print('  A^X   = L_X - L_state        G^X   = L_X - L_joint      (X = heuristic or LLM)')
    print('  Positive: residual state still ahead of that text reader. Negative: the text')
    print('  reader is better. R = share of the joint gain over prior recovered by the LLM.')
    print('  class: preregistered rule (approx if R >= 0.80 or the G^LLM CI includes 0).')

    if any(r.get('G_given_llm') is not None or r.get('G_given_heuristic') is not None for r in rows):
        print('\nNESTED: what state adds on top of each text-side feature (cheap linear grid)')
        def share(r):
            return (None if r.get('G_given_llm') is None or not r.get('G_cheap')
                    else f"{max(0.0, 100 * r['G_given_llm'] / r['G_cheap']):.0f}%")
        print_table(['model', 'task', 'G cheap', '95CI', 'G | heuristic', '95CI',
                     'G | LLM', '95CI', 'share left'],
                    [[r['model'], r['task'], signed(r['G_cheap']), ci(r['G_cheap_ci']),
                      signed(r.get('G_given_heuristic')), ci(r.get('G_given_heuristic_ci')),
                      signed(r.get('G_given_llm')), ci(r.get('G_given_llm_ci')), share(r) or '--']
                     for r in rows]
                    + [[avg['model'], '', signed(avg.get('G_cheap')), '',
                        signed(avg.get('G_given_heuristic')), '', signed(avg.get('G_given_llm')),
                        '', '']])
        print('\n  G | X = (L_text+X - L_joint+X): state added to the text reader once X is a')
        print('  feature of both families. Cheap grid is weaker than the frozen grid; compare')
        print('  G | X with G cheap, not with G_SOT.')

    print('\nAUC (switch-to-cue)')
    akeys = ['auc_text', 'auc_heuristic', 'auc_llm_full', 'auc_llm_redacted', 'auc_state', 'auc_joint']
    print_table(['model', 'task', 'text(ngram)', 'heuristic', 'LLM', 'LLM redacted', 'state', 'joint'],
                [[r['model'], r['task']] + [r.get(k) for k in akeys] for r in rows])

    verdicts = tb.get('verdicts') or {}
    if verdicts:
        print('\nPREREGISTERED VERDICTS')
        for k, v in verdicts.items():
            print(f"  {k:13s} {v['counts']}  majority={v['majority']}  mean R={v['mean_R']:.2f}")

    if a.latex:
        latex(rows)
    if a.csv:
        out = Path(a.csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        flat = [{k: (json.dumps(v) if isinstance(v, list) else v) for k, v in r.items()}
                for r in rows]
        keys = list(dict.fromkeys(k for r in flat for k in r))
        with out.open('w', newline='') as fh:
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader()
            w.writerows(flat)
        print(f'\nwrote {out}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
