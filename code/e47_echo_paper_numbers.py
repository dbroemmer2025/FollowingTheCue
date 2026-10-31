#!/usr/bin/env python3
"""Paper-facing numbers and displays under the corrected echo rule (v2).

Companion to `e47_echo_rescore.py` (schema e47_echo_rescore_v1). Computes
frozen (v1) and corrected (v2) answer-based quantities, checks the frozen
summary and headline counts, and writes v2 display inputs:

  reports/e47_echo_v2_paper_numbers.json     every number, v1, v2, and
                                              answer-independent framing
  reports/summary_csv_echo_v2.csv            figure input (switcher counts v2)
  reports/e47_primary_rate_audit_echo_v2.json  figure intervals, same schema
  tables/tab_e47_switch_to_cue.tex           regenerated under v2
  tables/tab_e47_switch_to_cue_v1.tex        frozen-scoring sensitivity table

Answer-based outcomes exclude echoes under v2. The paper's primary framing
contrast is `reviewed_all_disagreement` and uses every reviewed disagreement
response, including capped, unparsed, and echoed responses; it does not
require an answer. The legacy `determinants` field retains the answer-valid
population for historical frozen-result checks.

Reads frozen artifacts only. CPU only.
"""
from __future__ import annotations

import argparse
import ast
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

from e47_design import read, review_cue_acknowledgment, scored_answer
from e47_echo_rescore import SCORERS, answer_v2, is_echo, mcnemar_exact, target_with
from e47_primary_rate_audit import interval
from run_dirs import DEFAULT_RUN_DIRS

ROOT = Path(__file__).resolve().parents[1]
FAMILIES = ('authority', 'sycophancy')
LABELS = {
    'qwen2.5-7b': 'Qwen2.5-7B', 'qwen2.5-7b-instruct': 'Qwen2.5-7B-Instruct',
    'gemma-2-9b-it': 'Gemma-2-9B-IT', 'mistral-nemo-instruct-2407': 'Mistral-Nemo-Instruct',
    'olmo-2-1124-7b-instruct': 'OLMo-2-1124-7B-Instruct',
}
TASKS = {'movie_recommendation': 'Movie recommendation',
         'temporal_sequences': 'Temporal sequences'}
FROZEN_SWITCH_BY_ARM = {
    'qwen2_5_7b_movie_recommendations_thinking_v10': ((194, 234), (139, 230)),
    'qwen2_5_7b_temporal_sequences_thinking_v10': ((158, 190), (138, 204)),
    'qwen2_7b_it_movie_recommendations_thinking_v10': ((107, 240), (48, 240)),
    'qwen2_7b_it_temporal_sequences_thinking_v10': ((93, 240), (25, 239)),
    'gemma_2_9b_it_movie_recommendations_thinking_v10': ((142, 230), (124, 223)),
    'gemma_2_9b_it_temporal_sequences_thinking_v10': ((125, 232), (87, 231)),
    'mistral_nemo_instruct_2407_movie_recommendations_thinking_v10': ((157, 192), (169, 198)),
    'mistral_nemo_instruct_2407_temporal_sequences_thinking_v10': ((109, 224), (131, 231)),
    'olmo_2_1124_7b_instruct_movie_recommendations_thinking_v10': ((219, 240), (216, 240)),
    'olmo_2_1124_7b_instruct_temporal_sequences_thinking_v10': ((189, 237), (164, 237)),
}


def wilson(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def load(run: Path) -> dict:
    traces = read(run / 'traces.pkl')['rows']
    followups = read(run / 'followups.pkl')['rows']
    with (run / 'review_approved.csv').open(newline='') as stream:
        approved = {r['audit_id']: r for r in csv.DictReader(stream)
                    if r['review_status'] == 'approved'}
    return {'traces': traces, 'by': {(r['index'], r['condition']): r for r in traces},
            'why': {(r['index'], r['condition']): r for r in followups},
            'approved': approved, 'echo': {r['id'] for r in traces if is_echo(r)}}


def ack_of(cell, row_id):
    label = review_cue_acknowledgment(cell['approved'].get(row_id))
    return None if label is None else int(label)


def cell_numbers(cell: dict, task: str, name: str) -> dict:
    score = SCORERS[name]
    fam = {f: {'switchers': 0, 'eligible': 0, 'initial_positive': 0, 'initial_reviewed': 0,
               'why_positive': 0, 'why_reviewed': 0} for f in FAMILIES}
    events, stage_pairs, det = [], [], []
    for row in cell['traces']:
        if not row['condition'].startswith('disagree_'):
            continue
        f = row['condition'].split('_', 1)[1]
        z = target_with(score, cell['by'][(row['index'], 'uncued')], row)
        if z is None:
            continue
        src = row['source_id']
        fam[f]['eligible'] += 1
        events.append((task, src, 'switch', z))
        ini = ack_of(cell, row['id'])
        follow = cell['why'].get((row['index'], row['condition']))
        why = ack_of(cell, follow['id']) if follow else None
        det.append({'family': f, 'switch': z, 'initial': ini, 'why': why,
                    'source_id': src})
        if z != 1:
            continue
        fam[f]['switchers'] += 1
        if ini is not None:
            fam[f]['initial_reviewed'] += 1
            fam[f]['initial_positive'] += ini
            events.append((task, src, 'initial_ack', bool(ini)))
        if why is not None:
            fam[f]['why_reviewed'] += 1
            fam[f]['why_positive'] += why
            events.append((task, src, 'why_ack', bool(why)))
        if ini is not None and why is not None:
            stage_pairs.append((ini, why))
    return {'families': fam, 'events': events, 'stage_pairs': stage_pairs, 'det': det}


def determinant_records(run: Path, echo_ids: set, name: str) -> list:
    """The paper's disagreement-trial acknowledgment population (204/2,156 etc.),
    loaded exactly as e47_label_audit / e47_disclosure_determinants do; v2 drops
    cue-echo rows."""
    import e47_disclosure_determinants as D
    from e47_label_audit import rows_artifact
    cell = D.load_cell(run, rows_file=rows_artifact(run))
    out = []
    for r in cell['records']:
        if name == 'v2' and r['id'] in echo_ids:
            continue
        out.append({'family': 'authority' if r['cue_type'] == 'authority' else 'sycophancy',
                    'switch': r['switch'], 'initial': r['initial'], 'why': r['why'],
                    'source_id': r['source_id'], 'cell': run.name})
    return out


def all_reviewed_disagreement(cell: dict, run: Path) -> list[dict]:
    """Answer-independent framing population, one response per arm and item."""
    out = []
    for row in cell['traces']:
        if not row['condition'].startswith('disagree_'):
            continue
        follow = cell['why'].get((row['index'], row['condition']))
        out.append({'family': row['condition'].split('_', 1)[1],
                    'initial': ack_of(cell, row['id']),
                    'why': ack_of(cell, follow['id']) if follow else None,
                    'source_id': row['source_id'], 'cell': run.name})
    return out


def parsed_counts(cell: dict, name: str) -> dict:
    score = SCORERS[name]
    rows = cell['traces']
    return {'initial_parsed': sum(score(r) in tuple('ABCD') for r in rows),
            'disagree_parsed': sum(score(r) in tuple('ABCD') for r in rows
                                   if r['condition'].startswith('disagree_'))}


def agreement_pairs(cell: dict, exclude_echo: bool) -> list:
    out = []
    for index in {r['index'] for r in cell['traces']}:
        a = cell['by'].get((index, 'agree_authority'))
        d = cell['by'].get((index, 'disagree_authority'))
        if a is None or d is None:
            continue
        la, ld = ack_of(cell, a['id']), ack_of(cell, d['id'])
        if la is None or ld is None:
            continue
        if exclude_echo and (a['id'] in cell['echo'] or d['id'] in cell['echo']):
            continue
        out.append((a['source_id'], la, ld))
    return out


def frame_pairs(records: list[dict], key: str) -> dict:
    """Matched reviewed-label contrast; answer validity is not required."""
    pairs = defaultdict(dict)
    for row in records:
        pairs[(row['cell'], row['source_id'])][row['family']] = row
    complete = [pair for pair in pairs.values() if len(pair) == 2]
    labels = [(pair['authority'][key], pair['sycophancy'][key]) for pair in complete
              if pair['authority'][key] is not None and pair['sycophancy'][key] is not None]
    a = sum(x == 1 and y == 0 for x, y in labels)
    u = sum(x == 0 and y == 1 for x, y in labels)
    return {'pairs': len(labels), 'auth_only': a, 'user_only': u,
            'both_zero': sum(x == 0 and y == 0 for x, y in labels),
            'both_one': sum(x == 1 and y == 1 for x, y in labels),
            'mcnemar_p': mcnemar_exact(a, u)}


def switch_table(rows: list, caption_note: str) -> str:
    lines = []
    for r in rows:
        cells = []
        for f in FAMILIES:
            k, n = r['families'][f]['switchers'], r['families'][f]['eligible']
            lo, hi = wilson(k, n)
            cells += [f"{k} / {n} ({100 * k / n:.1f}\\%)", f"[{100 * lo:.1f},{100 * hi:.1f}]"]
        lines.append(f"{LABELS[r['model']]} & {TASKS[r['task']]} & " + ' & '.join(cells) + r' \\')
    return '\n'.join([
        '% Generated by code/e47_echo_paper_numbers.py; do not edit by hand.',
        r'\begin{table*}[t]', r'\centering',
        r'\caption{Switch-to-cue rates on conflicting trials by model, task, and cue',
        r'frame. The denominator is the number of valid paired uncued--cued outcomes for',
        r'that condition. Brackets give Wilson 95\% intervals; each cue-frame arm has at',
        r'most one outcome per source item.' + caption_note + '}',
        r'\label{tab:e47-switch-to-cue}', r'\scriptsize',
        r'\begin{tabular}{llrrrr}', r'\toprule',
        r'Model & Task & Authority cue & 95\% CI & User-preference cue & 95\% CI \\',
        r'\midrule', *lines, r'\bottomrule', r'\end{tabular}', r'\end{table*}', ''])


ORDER = ('qwen2.5-7b', 'qwen2.5-7b-instruct', 'gemma-2-9b-it',
         'mistral-nemo-instruct-2407', 'olmo-2-1124-7b-instruct')


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--summary-csv', type=Path, default=ROOT / 'reports' / 'summary_csv.csv')
    parser.add_argument('--draws', type=int, default=5000)
    args = parser.parse_args(argv)
    summary = {Path(r['run']).name: r for r in csv.DictReader(args.summary_csv.open())}
    out = {'schema': 'e47_echo_v2_paper_numbers_v1', 'post_hoc': True,
           'rule': 'v2: cue-echo responses invalid; v1: frozen scored_answer', 'v1': {}, 'v2': {}}
    per = {n: [] for n in SCORERS}
    reviewed_all = []
    for run in DEFAULT_RUN_DIRS:
        cell = load(run)
        reviewed_all.extend(all_reviewed_disagreement(cell, run))
        s = summary[run.name]
        task = s['task']
        for name in SCORERS:
            c = cell_numbers(cell, task, name)
            c['det'] = determinant_records(run, cell['echo'], name)
            c['det_ack'] = determinant_records(run, cell['echo'], 'v1')
            c.update(model=s['model'], task=task, run=run.name, run_path=s['run'],
                     parsed=parsed_counts(cell, name),
                     agree=agreement_pairs(cell, exclude_echo=False),
                     agree_echoes_excluded=agreement_pairs(cell, exclude_echo=True))
            per[name].append(c)
        # v1 must reproduce the frozen summary exactly.
        frozen = ast.literal_eval(s['switcher_ack_by_cue'])
        v1 = per['v1'][-1]['families']
        for f in FAMILIES:
            for key in ('switchers', 'initial_positive', 'initial_reviewed',
                        'why_positive', 'why_reviewed'):
                if frozen[f][key] != v1[f][key]:
                    raise ValueError(f'{run.name}/{f}/{key}: v1 {v1[f][key]} != frozen {frozen[f][key]}')
        arm_counts = tuple((v1[f]['switchers'], v1[f]['eligible']) for f in FAMILIES)
        if arm_counts != FROZEN_SWITCH_BY_ARM[run.name]:
            raise ValueError(f'{run.name}: v1 Table II counts {arm_counts} != '
                             f'{FROZEN_SWITCH_BY_ARM[run.name]}')
        for key, actual in (('switch_to_cue', sum(v1[f]['switchers'] for f in FAMILIES)),
                            ('switch_eligible', sum(v1[f]['eligible'] for f in FAMILIES)),
                            ('initial_parsed', per['v1'][-1]['parsed']['initial_parsed'])):
            if int(s[key]) != actual:
                raise ValueError(f'{run.name}/{key}: v1 {actual} != frozen {s[key]}')
    for name, cells in per.items():
        events = [e for c in cells for e in c['events']]
        pooled = {m: interval(events, m, draws=args.draws)
                  for m in ('switch', 'initial_ack', 'why_ack')}
        def rate(c, k, n):
            K = sum(c['families'][f][k] for f in FAMILIES)
            N = sum(c['families'][f][n] for f in FAMILIES)
            return K, N
        cell_switch = [rate(c, 'switchers', 'eligible') for c in cells]
        cell_ack = [rate(c, 'initial_positive', 'initial_reviewed') for c in cells]
        arms = [(c['families'][f]['switchers'], c['families'][f]['eligible'])
                for c in cells for f in FAMILIES]
        stage = [p for c in cells for p in c['stage_pairs']]
        det = [d for c in cells for d in c['det']]
        det_ack = [d for c in cells for d in c['det_ack']]
        def fam_rate(records, f, key, cond=None):
            rows = [d for d in records if d['family'] == f and d[key] is not None
                    and (cond is None or d['switch'] == cond)]
            return sum(d[key] for d in rows), len(rows)
        initial_pairs = frame_pairs(det_ack, 'initial')
        hi = 0.02429259716610854  # audited user-preference false-negative upper bound
        wc_auth = round(initial_pairs['auth_only'] * (1 - hi))
        wc_user = round(hi * initial_pairs['both_zero'])
        def agree_summary(key):
            p = [x for c in cells for x in c[key]]
            ev = ([(c['task'], s, 'agree', bool(a)) for c in cells for s, a, _ in c[key]] +
                  [(c['task'], s, 'disagree', bool(d)) for c in cells for s, _, d in c[key]])
            b = sum(a == 0 and d == 1 for _, a, d in p)
            cc = sum(a == 1 and d == 0 for _, a, d in p)
            return {'pairs': len(p), 'agree_ack': sum(a for _, a, _ in p),
                    'disagree_ack': sum(d for _, _, d in p),
                    'agree_ci95': interval(ev, 'agree', draws=args.draws)['ci95'],
                    'disagree_ci95': interval(ev, 'disagree', draws=args.draws)['ci95'],
                    'both': sum(a == 1 and d == 1 for _, a, d in p),
                    'neither': sum(a == 0 and d == 0 for _, a, d in p),
                    'disagree_only': b, 'agree_only': cc, 'mcnemar_p': mcnemar_exact(b, cc)}
        gemma = next(c for c in cells if c['run'].startswith('gemma_2_9b_it_movie'))
        out[name] = {
            'pooled': pooled,
            'cell_switch': {c['run']: list(x) for c, x in zip(cells, cell_switch)},
            'cell_switch_range': [min(k / n for k, n in cell_switch), max(k / n for k, n in cell_switch)],
            'cell_switch_median': statistics.median(k / n for k, n in cell_switch),
            'arm_switch_range': [min(k / n for k, n in arms), max(k / n for k, n in arms)],
            'cell_initial_ack': {c['run']: list(x) for c, x in zip(cells, cell_ack)},
            'cell_initial_ack_range': [min(k / n for k, n in cell_ack), max(k / n for k, n in cell_ack)],
            'cell_initial_ack_median': statistics.median(k / n for k, n in cell_ack),
            'both_stages': {
                'reviewed_both': len(stage),
                'neither': sum(a == 0 and b == 0 for a, b in stage),
                'initial_ack_repeat': [sum(a == 1 and b == 1 for a, b in stage),
                                       sum(a == 1 for a, b in stage)],
                'silent_then_ack': [sum(a == 0 and b == 1 for a, b in stage),
                                    sum(a == 0 for a, b in stage)],
                'gemma_movie_neither': [sum(a == 0 and b == 0 for a, b in gemma['stage_pairs']),
                                        len(gemma['stage_pairs'])],
            },
            'determinants': {
                'authority_initial': fam_rate(det_ack, 'authority', 'initial'),
                'user_initial': fam_rate(det_ack, 'sycophancy', 'initial'),
                'authority_why': fam_rate(det_ack, 'authority', 'why'),
                'user_why': fam_rate(det_ack, 'sycophancy', 'why'),
                'authority_initial_switchers': fam_rate(det, 'authority', 'initial', 1),
                'authority_initial_resisters': fam_rate(det, 'authority', 'initial', 0),
                'user_why_switchers': fam_rate(det, 'sycophancy', 'why', 1),
                'user_why_resisters': fam_rate(det, 'sycophancy', 'why', 0),
                'initial_frame_pairs': initial_pairs,
                'why_frame_pairs': frame_pairs(det_ack, 'why'),
                'label_audit_worst_case': {'auth_only': wc_auth, 'user_only': wc_user,
                                           'p': mcnemar_exact(wc_auth, wc_user)},
            },
            'agreement': agree_summary('agree'),
            'agreement_echoes_excluded': agree_summary('agree_echoes_excluded'),
            'parsed': {k: sum(c['parsed'][k] for c in cells) for k in ('initial_parsed', 'disagree_parsed')},
            'base_vs_instruct': {c['run']: c_rate for c, c_rate in
                                 ((c, cell_switch[i][0] / cell_switch[i][1]) for i, c in enumerate(cells))
                                 if c['model'] in ('qwen2.5-7b', 'qwen2.5-7b-instruct')},
        }
        out[name]['cells_detail'] = {c['run']: c['families'] for c in cells}
    # Frozen-paper checks for v1.
    v1 = out['v1']
    checks = {
        'switch': ((v1['pooled']['switch']['numerator'], v1['pooled']['switch']['denominator']), (2734, 4532)),
        'initial_ack': ((v1['pooled']['initial_ack']['numerator'], v1['pooled']['initial_ack']['denominator']), (174, 2653)),
        'why_ack': ((v1['pooled']['why_ack']['numerator'], v1['pooled']['why_ack']['denominator']), (275, 2619)),
        'initial_frame_pairs': ((v1['determinants']['initial_frame_pairs']['pairs'],
                                 v1['determinants']['initial_frame_pairs']['auth_only'],
                                 v1['determinants']['initial_frame_pairs']['user_only']), (2131, 198, 0)),
        'why_frame_pairs': ((v1['determinants']['why_frame_pairs']['auth_only'],
                             v1['determinants']['why_frame_pairs']['user_only']), (360, 44)),
        'agreement': ((v1['agreement']['pairs'], v1['agreement']['agree_ack'],
                       v1['agreement']['disagree_ack'], v1['agreement']['disagree_only'],
                       v1['agreement']['agree_only']), (2235, 141, 207, 124, 58)),
        'parsed': ((v1['parsed']['initial_parsed'], v1['parsed']['disagree_parsed']), (11391, 4533)),
        'worst_case': ((v1['determinants']['label_audit_worst_case']['auth_only'],
                        v1['determinants']['label_audit_worst_case']['user_only']), (193, 47)),
    }
    bad = {k: v for k, v in checks.items() if v[0] != v[1]}
    if bad:
        raise ValueError(f'v1 does not reproduce the paper: {bad}')
    def reviewed_rate(family, key):
        labels = [r[key] for r in reviewed_all
                  if r['family'] == family and r[key] is not None]
        return [sum(labels), len(labels)]
    out['reviewed_all_disagreement'] = {
        'population': 'all reviewed disagreement responses; no answer-validity filter',
        'authority_initial': reviewed_rate('authority', 'initial'),
        'user_initial': reviewed_rate('sycophancy', 'initial'),
        'authority_why': reviewed_rate('authority', 'why'),
        'user_why': reviewed_rate('sycophancy', 'why'),
        'initial_frame_pairs': frame_pairs(reviewed_all, 'initial'),
        'why_frame_pairs': frame_pairs(reviewed_all, 'why'),
    }
    (ROOT / 'reports' / 'e47_echo_v2_paper_numbers.json').write_text(json.dumps(out, indent=2) + '\n')

    # Display inputs.
    def ordered(cells):
        return sorted(cells, key=lambda c: (0 if 'movie' in c['task'] else 1, ORDER.index(c['model'])))
    tab_order = lambda cells: sorted(cells, key=lambda c: (ORDER.index(c['model']), c['task']))
    (ROOT / 'tables' / 'tab_e47_switch_to_cue.tex').write_text(switch_table(
        tab_order(per['v2']), ' Cue-echo responses (whole-response copies of the cue'
        ' sentence) are invalid, as are unparsed responses.'))
    (ROOT / 'tables' / 'tab_e47_switch_to_cue_v1.tex').write_text(switch_table(
        tab_order(per['v1']), ' Frozen scoring, which counts cue-echo responses as'
        ' choosing the cue letter (sensitivity check).'))
    rows = list(csv.DictReader(args.summary_csv.open()))
    by_run = {c['run']: c for c in per['v2']}
    for r in rows:
        fam = by_run[Path(r['run']).name]['families']
        frozen = ast.literal_eval(r['switcher_ack_by_cue'])
        for f in FAMILIES:
            frozen[f].update({k: fam[f][k] for k in ('switchers', 'initial_positive',
                              'initial_reviewed', 'why_positive', 'why_reviewed')})
        r['switcher_ack_by_cue'] = repr(frozen)
        r['switch_to_cue'] = sum(fam[f]['switchers'] for f in FAMILIES)
        r['switch_eligible'] = sum(fam[f]['eligible'] for f in FAMILIES)
        r['switcher_ack_source'] = 'e47_echo_paper_numbers.py v2 (cue echoes invalid)'
    with (ROOT / 'reports' / 'summary_csv_echo_v2.csv').open('w', newline='') as stream:
        w = csv.DictWriter(stream, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    audit = {'scope': 'fixed ten-cell model grid; resample source items within task; '
                      'v2 echo rule', 'bootstrap_draws': args.draws, 'seed': 47,
             'cells': [{'run_name': c['run'], 'model': c['model'], 'task': c['task'],
                        **{m: interval(c['events'], m, draws=args.draws)
                           for m in ('switch', 'initial_ack', 'why_ack')}}
                       for c in per['v2']],
             'pooled': out['v2']['pooled']}
    (ROOT / 'reports' / 'e47_primary_rate_audit_echo_v2.json').write_text(json.dumps(audit, indent=2) + '\n')
    for name in SCORERS:
        r = out[name]
        p = r['pooled']
        print(f"[{name}] switch {p['switch']['numerator']}/{p['switch']['denominator']} "
              f"{100 * p['switch']['rate']:.1f}% {[round(100 * x, 1) for x in p['switch']['ci95']]}; "
              f"cells {100 * r['cell_switch_range'][0]:.1f}-{100 * r['cell_switch_range'][1]:.1f} "
              f"median {100 * r['cell_switch_median']:.1f}; arms "
              f"{100 * r['arm_switch_range'][0]:.1f}-{100 * r['arm_switch_range'][1]:.1f}")
        print(f"     init ack {p['initial_ack']['numerator']}/{p['initial_ack']['denominator']} "
              f"{[round(100 * x, 1) for x in p['initial_ack']['ci95']]}; why "
              f"{p['why_ack']['numerator']}/{p['why_ack']['denominator']} "
              f"{[round(100 * x, 1) for x in p['why_ack']['ci95']]}; cell init ack range "
              f"{100 * r['cell_initial_ack_range'][0]:.1f}-{100 * r['cell_initial_ack_range'][1]:.1f} "
              f"median {100 * r['cell_initial_ack_median']:.2f}")
        print(f"     both stages {r['both_stages']}")
        print(f"     determinants {json.dumps(r['determinants'])}")
        print(f"     agreement {r['agreement']}")
        print(f"     parsed {r['parsed']}; base/instruct {r['base_vs_instruct']}")
    print(f"[reviewed_all_disagreement] {json.dumps(out['reviewed_all_disagreement'])}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
