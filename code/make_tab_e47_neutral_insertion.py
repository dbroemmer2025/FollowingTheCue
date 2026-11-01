#!/usr/bin/env python3
"""Generate tables/tab_e47_neutral_insertion.tex from the sidecar rows.csv files.

Also prints descriptive supporting numbers: the share of neutral departures
that land on the assigned letter, the paired cue/neutral outcome counts
(cue-only, both, neutral-only, neither) by frame, and the prompt-length
difference. Paired counts are descriptive comparisons with one neutral sentence;
they are not individual causal attributions.
Reads only <run>/neutral_insertion/rows.csv for the ten paper cells.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
from pathlib import Path

from e47_echo_rescore import echo_keys, neutral_row_is_echo
from e47_neutral_insertion import PAPER_MODELS, PAPER_TASKS, pooled_cluster_ci

ROOT = Path(__file__).resolve().parents[1]
LABELS = {
    'qwen2_5_7b': 'Qwen2.5-7B', 'qwen2_7b_it': 'Qwen2.5-7B-Instruct',
    'gemma_2_9b_it': 'Gemma-2-9B-IT',
    'mistral_nemo_instruct_2407': 'Mistral-Nemo-Instruct',
    'olmo_2_1124_7b_instruct': 'OLMo-2-1124-7B-Instruct',
}
TABLE_ORDER = ('qwen2_5_7b', 'qwen2_7b_it', 'gemma_2_9b_it',
               'mistral_nemo_instruct_2407', 'olmo_2_1124_7b_instruct')
TASK_LABEL = {'movie_recommendations': 'Movie', 'temporal_sequences': 'Temporal'}
CAPTION = r"""\caption{Post hoc neutral-insertion control on disagreement trials with valid
uncued, cued, and neutral answers. Cue and neutral columns give the percentage
selecting the preassigned alternative letter; the neutral prompt does not
contain that letter. The paired difference is in percentage points, with a
95\% source-item-bootstrap interval that keeps the two cue frames together.
Neutral-away is the percentage whose neutral answer differs from the stored
uncued answer, on the same matched trials, with Wilson 95\% intervals for
individual cells. These Wilson intervals are descriptive because each source
item can contribute two cue-frame outcomes. The pooled row is descriptive for this fixed model grid;
both pooled intervals resample source items within task while preserving
repeated model and frame outcomes.}"""


def matched_rows(run: Path, scoring: str = 'v2') -> list[dict]:
    """Matched rows; under v2 (default) cue-echo cued responses are invalid."""
    keys = echo_keys(run) if scoring == 'v2' else set()
    with (run / 'neutral_insertion' / 'rows.csv').open(newline='') as stream:
        return [r for r in csv.DictReader(stream)
                if r['cue_to_assigned'] and r['neutral_to_assigned']
                and not neutral_row_is_echo(r, keys)]


def item_ci(rows: list[dict], draws: int, seed: int) -> list[float]:
    clusters = {}
    for r in rows:
        d = int(r['cue_to_assigned']) - int(r['neutral_to_assigned'])
        total, n = clusters.get(r['source_id'], (0, 0))
        clusters[r['source_id']] = (total + d, n + 1)
    values = list(clusters.values())
    rng = random.Random(seed)
    est = []
    for _ in range(draws):
        s = n = 0
        for _ in values:
            a, b = values[rng.randrange(len(values))]
            s += a
            n += b
        est.append(s / n)
    est.sort()
    return [est[int(.025 * draws)], est[int(.975 * draws)]]


def wilson_ci(k: int, n: int, z: float = 1.959963984540054) -> list[float]:
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    radius = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return [center - radius, center + radius]


def pooled_away_ci(records: list[tuple[str, dict]], draws: int, seed: int) -> list[float]:
    by_source = {}
    for task, row in records:
        key = (task, row['source_id'])
        total, count = by_source.get(key, (0, 0))
        by_source[key] = (total + int(row['neutral_away']), count + 1)
    by_task = {}
    for (task, _), value in by_source.items():
        by_task.setdefault(task, []).append(value)
    rng = random.Random(seed)
    estimates = []
    for _ in range(draws):
        total = count = 0
        for clusters in by_task.values():
            for _ in clusters:
                sampled_total, sampled_count = clusters[rng.randrange(len(clusters))]
                total += sampled_total
                count += sampled_count
        estimates.append(total / count)
    estimates.sort()
    return [estimates[int(.025 * draws)], estimates[int(.975 * draws)]]


def rates(rows: list[dict]) -> dict:
    n = len(rows)
    cue = sum(int(r['cue_to_assigned']) for r in rows)
    neu = sum(int(r['neutral_to_assigned']) for r in rows)
    away = sum(int(r['neutral_away']) for r in rows)
    return {'n': n, 'cue': cue, 'neutral': neu, 'neutral_away': away,
            'excess': (cue - neu) / n}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs-root', type=Path, default=ROOT / 'e47')
    parser.add_argument('--out', type=Path,
                        default=ROOT / 'tables' / 'tab_e47_neutral_insertion.tex')
    parser.add_argument('--numbers', type=Path,
                        default=ROOT / 'reports' / 'e47_neutral_insertion_paper_numbers.json')
    parser.add_argument('--draws', type=int, default=2000)
    parser.add_argument('--scoring', choices=('v1', 'v2'), default='v2',
                        help='v2 (default): cue-echo cued responses invalid; '
                             'v1: frozen scored_answer (sensitivity)')
    parser.add_argument('--seed', type=int, default=47)
    args = parser.parse_args(argv)

    lines, records, all_rows, cells = [], [], [], {}
    for model in TABLE_ORDER:
        for task in PAPER_TASKS:
            run = args.runs_root / f'{model}_{task}_thinking_v10'
            rows = matched_rows(run, args.scoring)
            r = rates(rows)
            lo, hi = item_ci(rows, args.draws, args.seed)
            away_lo, away_hi = wilson_ci(r['neutral_away'], r['n'])
            cells[run.name] = {**r, 'ci95': [lo, hi],
                               'neutral_away_ci95': [away_lo, away_hi]}
            all_rows += rows
            records += [(task, row) for row in rows]
            lines.append(
                f"{LABELS[model]} & {TASK_LABEL[task]} & {r['n']} & "
                f"{100 * r['cue'] / r['n']:.1f}\\% & {100 * r['neutral'] / r['n']:.1f}\\% & "
                f"{100 * r['excess']:+.1f} [{100 * lo:.1f},{100 * hi:.1f}] & "
                f"{100 * r['neutral_away'] / r['n']:.1f} "
                f"[{100 * away_lo:.1f},{100 * away_hi:.1f}] \\\\")
    assert set(PAPER_MODELS) == set(TABLE_ORDER)
    pooled = rates(all_rows)
    plo, phi = pooled_cluster_ci(records, seed=args.seed, draws=args.draws)
    away_plo, away_phi = pooled_away_ci(records, args.draws, args.seed)
    n = pooled['n']
    body = '\n'.join([
        '% Generated by code/make_tab_e47_neutral_insertion.py; do not edit by hand.',
        r'\begin{table*}[t]', r'\centering', CAPTION,
        r'\label{tab:e47-neutral-insertion}', r'\scriptsize',
        r'\begin{tabular}{llrrrrr}', r'\toprule',
        r'Model & Task & Matched $n$ & Cue to letter & Neutral to letter & '
        r'Difference [95\% CI] & Neutral away [95\% CI] \\', r'\midrule', *lines, r'\midrule',
        f"Fixed-grid pooled & Both tasks & {n:,} & {100 * pooled['cue'] / n:.1f}\\% & "
        f"{100 * pooled['neutral'] / n:.1f}\\% & {100 * pooled['excess']:+.1f} "
        f"[{100 * plo:.1f},{100 * phi:.1f}] & {100 * pooled['neutral_away'] / n:.1f} "
        f"[{100 * away_plo:.1f},{100 * away_phi:.1f}] \\\\",
        r'\bottomrule', r'\end{tabular}', r'\end{table*}', ''])
    args.out.write_text(body)

    def paired(rows):
        counts = {'cue_only': 0, 'both': 0, 'neutral_only': 0, 'neither': 0}
        for r in rows:
            key = {(1, 0): 'cue_only', (1, 1): 'both', (0, 1): 'neutral_only',
                   (0, 0): 'neither'}[(int(r['cue_to_assigned']), int(r['neutral_to_assigned']))]
            counts[key] += 1
        return counts

    by_frame = {frame: {**rates([r for r in all_rows if r['frame'] == frame]),
                        'paired_counts': paired([r for r in all_rows if r['frame'] == frame])}
                for frame in ('authority', 'sycophancy')}
    tokens = sorted({int(r['neutral_prompt_tokens']) - int(r['cue_prompt_tokens'])
                     for r in all_rows})
    numbers = {
        'post_hoc': True, 'descriptive': True, 'scoring': args.scoring, 'matched_n': n,
        'pooled': {**pooled, 'ci95': [plo, phi],
                   'neutral_away_ci95': [away_plo, away_phi],
                   'paired_counts': paired(all_rows)},
        'neutral_departures_to_letter_share': pooled['neutral'] / pooled['neutral_away'],
        'by_frame': by_frame, 'prompt_token_delta_range': [tokens[0], tokens[-1]],
        'cells': cells, 'bootstrap': {'draws': args.draws, 'seed': args.seed},
    }
    args.numbers.write_text(json.dumps(numbers, indent=2) + '\n')
    print(f"wrote {args.out}")
    print(f"pooled: n={n:,} cue={pooled['cue']} neutral={pooled['neutral']} "
          f"excess={100 * pooled['excess']:+.1f} [{100 * plo:.1f},{100 * phi:.1f}] pp")
    print(f"neutral departures landing on assigned letter: {pooled['neutral']}/"
          f"{pooled['neutral_away']} = {100 * numbers['neutral_departures_to_letter_share']:.1f}% "
          f"(1/3 = 33.3%)")
    pc = numbers['pooled']['paired_counts']
    print(f"paired counts: cue_only={pc['cue_only']} both={pc['both']} "
          f"neutral_only={pc['neutral_only']} neither={pc['neither']}")
    print(f"neutral minus cue prompt tokens: {tokens[0]} to {tokens[-1]}")
    print(f"wrote {args.numbers}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
