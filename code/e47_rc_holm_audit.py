#!/usr/bin/env python3
"""Post hoc Holm check of RC loss increments across paper cells."""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

from scipy.stats import ttest_1samp


ROOT = Path(__file__).resolve().parents[1]


def audit(directory: Path, arm: str = 'output') -> list[dict]:
    suffix = '__nollm' if arm == 'nollm' else '__output'
    text_family = 'text_out_nollm' if arm == 'nollm' else 'text_out'
    joint_family = 'joint_out_nollm' if arm == 'nollm' else 'joint_out'
    gain_key = ('state_beyond_text_and_output_nollm' if arm == 'nollm'
                else 'state_beyond_text_and_output')
    results = []
    for path in sorted(directory.glob(f'*{suffix}.json')):
        payload = json.loads(path.read_text())
        target = payload.get('targets', {}).get('RC', {})
        if gain_key not in target:
            raise ValueError(f'{path.name}: missing RC result for {arm}')
        oof_path = path.with_name(path.name.replace('.json', '.oof.csv'))
        by_item = defaultdict(list)
        with oof_path.open(newline='') as stream:
            for row in csv.DictReader(stream):
                if row['target'] == 'RC':
                    by_item[row['source_id']].append(
                        float(row[f'{text_family}_loss']) -
                        float(row[f'{joint_family}_loss']))
        means = [sum(values) / len(values) for values in by_item.values()]
        if len(means) < 2:
            raise ValueError(f'{path.name}: fewer than two source items')
        delta = sum(means) / len(means)
        reported = target[gain_key]
        if abs(delta - reported['delta']) > 1e-8:
            raise ValueError(f'{path.name}: OOF loss delta disagrees with JSON')
        p = float(ttest_1samp(means, 0, alternative='greater').pvalue)
        results.append({'cell': path.name.removesuffix(f'{suffix}.json'),
                        'n_items': len(means), 'delta': delta,
                        'p_one_sided': p,
                        'share_of_joint_gain': reported['share_of_joint_gain']})
    if len(results) != 10:
        raise ValueError(f'Expected ten RC output-access cells; found {len(results)}')
    ordered = sorted(results, key=lambda row: row['p_one_sided'])
    running = 0.0
    for rank, row in enumerate(ordered):
        running = max(running, min(1.0, (len(ordered) - rank) * row['p_one_sided']))
        row['p_holm'] = running
        row['positive_after_holm'] = running < .05
        row['meets_share_gate'] = row['share_of_joint_gain'] >= .10
    return sorted(results, key=lambda row: row['cell'])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', type=Path,
                        default=ROOT / 'reports/de_readers/results')
    parser.add_argument('--out', type=Path,
                        default=ROOT / 'reports/e47_rc_holm_audit.csv')
    parser.add_argument('--arm', choices=['nollm', 'output'], default='output')
    args = parser.parse_args()
    rows = audit(args.input_dir, args.arm)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Positive after Holm: {sum(r['positive_after_holm'] for r in rows)}/10; "
          f"meet 10% share gate: {sum(r['meets_share_gate'] for r in rows)}/10")
    print(f'wrote {args.out}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
