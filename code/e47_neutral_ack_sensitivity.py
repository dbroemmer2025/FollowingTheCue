#!/usr/bin/env python3
"""Audit initial cue acknowledgment by paired neutral assigned-letter outcome.

These are descriptive strata, not identified individual cue-caused switches.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

from e47_design import read, review_cue_acknowledgment
from e47_neutral_insertion import PAPER_TASKS
from e47_primary_rate_audit import interval
from make_tab_e47_neutral_insertion import ROOT, TABLE_ORDER, matched_rows


def audit(runs_root: Path, sidecar_name: str, draws: int) -> dict:
    events = defaultdict(list)
    counts = defaultdict(lambda: {'switchers': 0, 'reviewed': 0, 'acknowledged': 0})
    for model in TABLE_ORDER:
        for task in PAPER_TASKS:
            run = runs_root / f'{model}_{task}_thinking_v10'
            traces = read(run / 'traces.pkl')['rows']
            by_key = {(int(row['index']), row['condition']): row for row in traces}
            if len(by_key) != len(traces):
                raise ValueError(f'{run}: duplicate trace index/condition')
            with (run / 'review_approved.csv').open(newline='') as stream:
                approved = {row['audit_id']: row for row in csv.DictReader(stream)
                            if row['review_status'] == 'approved'}
            for row in matched_rows(run, sidecar_name=sidecar_name):
                if row['frame'] not in ('authority', 'sycophancy'):
                    raise ValueError(f'{run}: unexpected frame {row["frame"]}')
                if int(row['cue_to_assigned']) != 1:
                    continue
                trace = by_key[(int(row['index']), f'disagree_{row["frame"]}')]
                if trace['source_id'] != row['source_id']:
                    raise ValueError(f'{run}: source mismatch at index {row["index"]}')
                stratum = 'both' if int(row['neutral_to_assigned']) else 'cue_only'
                keys = (('all', stratum), (row['frame'], stratum))
                for key in keys:
                    counts[key]['switchers'] += 1
                ack = review_cue_acknowledgment(approved.get(trace['id']))
                if ack is None:
                    continue
                for key in keys:
                    counts[key]['reviewed'] += 1
                    counts[key]['acknowledged'] += int(ack == '1')
                    events[key].append((task, row['source_id'], 'ack', ack == '1'))
    result = {}
    for (frame, stratum), count in sorted(counts.items()):
        summary = interval(events[(frame, stratum)], 'ack', draws=draws)
        if (summary['numerator'], summary['denominator']) != (
                count['acknowledged'], count['reviewed']):
            raise ValueError(f'Count mismatch: {frame}/{stratum}')
        result[f'{frame}/{stratum}'] = {**count, **summary}
    return {
        'definition': 'Valid cued answer selects assigned letter; cue_only means '
                      'paired neutral answer does not, both means it does.',
        'interpretation': 'Descriptive paired outcome strata, not identified '
                          'individual causal effects.',
        'sidecar_name': sidecar_name,
        'draws': draws,
        'strata': result,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs-root', type=Path, default=ROOT / 'e47')
    parser.add_argument('--sidecar-name', default='neutral_insertion_same_session')
    parser.add_argument('--draws', type=int, default=5000)
    parser.add_argument('--out', type=Path,
                        default=ROOT / 'reports' / 'e47_neutral_ack_sensitivity.json')
    args = parser.parse_args(argv)
    result = audit(args.runs_root, args.sidecar_name, args.draws)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + '\n')
    for key, value in result['strata'].items():
        print(f'{key}: {value["acknowledged"]}/{value["reviewed"]} '
              f'({100 * value["rate"]:.1f}%), valid switchers {value["switchers"]}')
    print(args.out)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
