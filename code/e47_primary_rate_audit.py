#!/usr/bin/env python3
"""Reconcile E47 headline counts and bootstrap source-item rate intervals."""
from __future__ import annotations

import argparse
import ast
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from e47_design import read, review_cue_acknowledgment, target


ROOT = Path(__file__).resolve().parents[1]
METRICS = ('switch', 'initial_ack', 'why_ack')


def interval(events: list[tuple], metric: str, seed: int = 47,
             draws: int = 5000) -> dict:
    by_task = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    for task, source, name, positive in events:
        if name != metric:
            continue
        pair = by_task[task][source]
        pair[0] += int(positive)
        pair[1] += 1
    strata = [np.asarray(list(items.values()), dtype=int)
              for items in by_task.values()]
    numerator = sum(int(a[:, 0].sum()) for a in strata)
    denominator = sum(int(a[:, 1].sum()) for a in strata)
    if not denominator:
        raise ValueError(f'No denominator for {metric}')
    rng = np.random.default_rng(seed)
    totals = np.zeros((draws, 2), dtype=float)
    for values in strata:
        picks = rng.integers(0, len(values), size=(draws, len(values)))
        totals += values[picks].sum(axis=1)
    rates = totals[:, 0] / totals[:, 1]
    low, high = np.quantile(rates, [.025, .975])
    if numerator == 0:
        # The ordinary percentile bootstrap is [0,0] for an all-zero sample.
        # Use the exact-binomial upper bound on unique source clusters instead.
        n_clusters = sum(len(values) for values in strata)
        high = 1 - (.025) ** (1 / n_clusters)
    return {'numerator': numerator, 'denominator': denominator,
            'rate': numerator / denominator, 'ci95': [float(low), float(high)],
            'ci_method': ('source-item bootstrap; zero-count upper bound over '
                          'unique source items' if numerator == 0 else
                          'source-item bootstrap')}


def cell_events(run: Path, task: str) -> list[tuple]:
    traces = read(run / 'traces.pkl')['rows']
    followups = read(run / 'followups.pkl')['rows']
    uncued = {row['index']: row for row in traces if row['condition'] == 'uncued'}
    why = {(row['index'], row['condition']): row for row in followups}
    with (run / 'review_approved.csv').open(newline='') as stream:
        approved = {row['audit_id']: row for row in csv.DictReader(stream)
                    if row['review_status'] == 'approved'}
    events = []
    for row in traces:
        if not row['condition'].startswith('disagree_'):
            continue
        z = target(uncued[row['index']], row)
        if z is None:
            continue
        source = row['source_id']
        events.append((task, source, 'switch', z))
        if z != 1:
            continue
        ack = review_cue_acknowledgment(approved.get(row['id']))
        if ack is not None:
            events.append((task, source, 'initial_ack', ack == '1'))
        followup = why.get((row['index'], row['condition']))
        ack = review_cue_acknowledgment(approved.get(followup['id'])) if followup else None
        if ack is not None:
            events.append((task, source, 'why_ack', ack == '1'))
    return events


def matched_authority_events(run: Path, task: str) -> list[tuple]:
    traces = read(run / 'traces.pkl')['rows']
    by = {(row['index'], row['condition']): row for row in traces}
    with (run / 'review_approved.csv').open(newline='') as stream:
        approved = {row['audit_id']: row for row in csv.DictReader(stream)
                    if row['review_status'] == 'approved'}
    events = []
    for index in {row['index'] for row in traces}:
        agree = by.get((index, 'agree_authority'))
        disagree = by.get((index, 'disagree_authority'))
        if agree is None or disagree is None:
            continue
        a = review_cue_acknowledgment(approved.get(agree['id']))
        d = review_cue_acknowledgment(approved.get(disagree['id']))
        if a is None or d is None:
            continue
        source = agree['source_id']
        if source != disagree['source_id']:
            raise ValueError(f'{run.name}/{index}: mismatched source IDs')
        events.extend(((task, source, 'agree_authority', a == '1'),
                       (task, source, 'disagree_authority', d == '1')))
    return events


def audit(summary_csv: Path, draws: int) -> dict:
    with summary_csv.open(newline='') as stream:
        cells = list(csv.DictReader(stream))
    if len(cells) != 10:
        raise ValueError(f'Expected ten paper cells; found {len(cells)}')
    all_events = []
    matched_events = []
    per_cell = []
    for row in cells:
        events = cell_events(Path(row['run']), row['task'])
        by = {metric: interval(events, metric, draws=draws) for metric in METRICS}
        expected = {'switch': (int(row['switch_to_cue']), int(row['switch_eligible']))}
        cue = ast.literal_eval(row['switcher_ack_by_cue'])
        for metric, stage in (('initial_ack', 'initial'), ('why_ack', 'why')):
            expected[metric] = (sum(v[f'{stage}_positive'] for v in cue.values()),
                                sum(v[f'{stage}_reviewed'] for v in cue.values()))
        for metric, counts in expected.items():
            if (by[metric]['numerator'], by[metric]['denominator']) != counts:
                raise ValueError(f"{row['run_name']}/{metric}: raw counts disagree "
                                 f"with summary {counts}")
        per_cell.append({'run_name': row['run_name'], 'model': row['model'],
                         'task': row['task'], **by})
        all_events.extend(events)
        matched_events.extend(matched_authority_events(Path(row['run']), row['task']))
    pooled = {metric: interval(all_events, metric, draws=draws)
              for metric in METRICS}
    matched_authority = {metric: interval(matched_events, metric, draws=draws)
                         for metric in ('agree_authority', 'disagree_authority')}
    if [(matched_authority[m]['numerator'], matched_authority[m]['denominator'])
            for m in ('agree_authority', 'disagree_authority')] != [
                (141, 2235), (207, 2235)]:
        raise ValueError('Matched authority counts disagree with Table III')
    return {'scope': 'fixed ten-cell model grid; resample source items within task',
            'bootstrap_draws': draws, 'seed': 47,
            'cells': per_cell, 'pooled': pooled,
            'matched_authority': matched_authority}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--summary-csv', type=Path,
                        default=ROOT / 'reports/e47_gap_summary.csv')
    parser.add_argument('--out', type=Path,
                        default=ROOT / 'reports/e47_primary_rate_audit.json')
    parser.add_argument('--draws', type=int, default=5000)
    args = parser.parse_args()
    if args.draws < 100:
        parser.error('--draws must be at least 100')
    result = audit(args.summary_csv, args.draws)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + '\n')
    for name, value in result['pooled'].items():
        print(f"{name}: {value['numerator']}/{value['denominator']} "
              f"= {100*value['rate']:.1f}% "
              f"[{100*value['ci95'][0]:.1f}, {100*value['ci95'][1]:.1f}]%")
    print(f'wrote {args.out}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
