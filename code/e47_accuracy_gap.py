#!/usr/bin/env python3
"""Post-outcome BBH correctness versus E47 verbalization-gap diagnostic.

This analysis never refits a reader.  It joins the frozen initial responses and
benchmark labels to the fixed grouped out-of-fold losses written by
``e47_analysis.py``, then stratifies the prespecified disagreement population
by whether the model's final answer is benchmark-correct.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from e47_design import (atomic, completed, digest, file_hash, load_design,
                        scored_answer)
from e47_readers import interval

SCHEMA = 'e47_accuracy_gap_v1'


def _item_mean(values, groups):
    values = np.asarray(values, float)
    groups = np.asarray(groups)
    unique = np.unique(groups)
    return float(np.mean([values[groups == group].mean()
                          for group in unique]))


def _subset(rows, seed, bootstrap):
    if not rows:
        return {'n_rows': 0, 'n_items': 0,
                'status': 'empty_correctness_stratum'}
    groups = np.asarray([row['source_id'] for row in rows])
    fields = ('text_loss_bits', 'state_loss_bits', 'joint_loss_bits',
              'gap_bits')
    result = {
        'status': 'descriptive_fixed_oof_subset',
        'n_rows': len(rows),
        'n_items': len(set(groups)),
        'switch_rate': interval(
            [row['target'] for row in rows], groups, seed, bootstrap),
    }
    for field in fields:
        result[field.replace('_loss_bits', '_ce_bits')] = _item_mean(
            [row[field] for row in rows], groups)
    result['gap'] = interval(
        [row['gap_bits'] for row in rows], groups, seed + 1, bootstrap)
    text_ce = result['text_ce_bits']
    result['relative_gap'] = (result['gap']['mean'] / text_ce
                              if text_ce else None)
    result['text_minus_state_bits'] = (
        result['text_ce_bits'] - result['state_ce_bits'])
    return result


def _difference(rows, seed, bootstrap):
    correct = [row for row in rows if row['answer_correct']]
    incorrect = [row for row in rows if not row['answer_correct']]
    if not correct or not incorrect:
        return {'status': 'inconclusive_missing_correctness_class',
                'mean': None, 'ci95': None,
                'n_correct_rows': len(correct),
                'n_incorrect_rows': len(incorrect)}
    point = ( _item_mean([row['gap_bits'] for row in correct],
                         [row['source_id'] for row in correct])
             - _item_mean([row['gap_bits'] for row in incorrect],
                          [row['source_id'] for row in incorrect]))
    by_group = {}
    for row in rows:
        by_group.setdefault(row['source_id'], []).append(row)
    group_names = np.asarray(sorted(by_group))
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(bootstrap):
        sampled = rng.choice(group_names, len(group_names), replace=True)
        c_values, i_values = [], []
        for group in sampled:
            selected = by_group[group]
            c = [row['gap_bits'] for row in selected
                 if row['answer_correct']]
            i = [row['gap_bits'] for row in selected
                 if not row['answer_correct']]
            if c:
                c_values.append(float(np.mean(c)))
            if i:
                i_values.append(float(np.mean(i)))
        if c_values and i_values:
            draws.append(float(np.mean(c_values) - np.mean(i_values)))
    if not draws:
        return {'status': 'inconclusive_bootstrap_occupancy',
                'mean': point, 'ci95': None,
                'n_correct_rows': len(correct),
                'n_incorrect_rows': len(incorrect)}
    return {
        'status': 'descriptive_post_outcome_contrast',
        'mean': point,
        'ci95': np.quantile(draws, [.025, .975]).tolist(),
        'n_bootstrap_valid': len(draws),
        'n_correct_rows': len(correct),
        'n_incorrect_rows': len(incorrect),
        'interpretation': ('Positive means a larger fixed-OOF text-minus-joint '
                           'gap on benchmark-correct responses.'),
    }


def summarize(rows, seed=47, bootstrap=2000):
    cue_types = sorted({row['cue_type'] for row in rows})
    selections = [('all', rows)] + [
        (cue_type, [row for row in rows if row['cue_type'] == cue_type])
        for cue_type in cue_types]
    output = {}
    for offset, (name, selected) in enumerate(selections):
        groups = np.asarray([row['source_id'] for row in selected])
        accuracy = (interval([row['answer_correct'] for row in selected],
                             groups, seed + 100 * offset, bootstrap)
                    if selected else None)
        output[name] = {
            'n_rows': len(selected),
            'n_items': len(set(groups)),
            'accuracy': accuracy,
            'all': _subset(selected, seed + 100 * offset + 1, bootstrap),
            'correct': _subset(
                [row for row in selected if row['answer_correct']],
                seed + 100 * offset + 2, bootstrap),
            'incorrect': _subset(
                [row for row in selected if not row['answer_correct']],
                seed + 100 * offset + 3, bootstrap),
            'correct_minus_incorrect_gap': _difference(
                selected, seed + 100 * offset + 4, bootstrap),
        }
    return output


def analyze(run, output=None, checkpoint='initial/pre_answer',
            bootstrap=2000):
    if int(bootstrap) < 100:
        raise ValueError('bootstrap must be at least 100')
    run = Path(run)
    design = load_design(run, check_code=False)
    traces = completed(run, 'traces.pkl', design)
    analysis_path = run / 'analysis.json'
    if not analysis_path.exists():
        raise ValueError('Run analyze first; analysis.json is missing')
    analysis = json.loads(analysis_path.read_text())
    if not analysis.get('complete'):
        raise ValueError('analysis.json is incomplete')
    export_name = analysis.get('oof_losses_file', 'oof_losses.csv')
    export_path = run / export_name
    if not export_path.exists():
        raise ValueError(f'Saved OOF predictions are missing: {export_path}')
    export_sha = file_hash(export_path)
    if (analysis.get('oof_losses_sha256')
            and analysis['oof_losses_sha256'] != export_sha):
        raise ValueError('OOF prediction hash differs from analysis.json')

    trace_by_id = {row['id']: row for row in traces['rows']}
    rows = []
    with export_path.open(newline='') as handle:
        for raw in csv.DictReader(handle):
            if (raw['checkpoint'] != checkpoint
                    or not raw['condition'].startswith('disagree_')):
                continue
            trace = trace_by_id.get(raw['id'])
            if trace is None:
                raise ValueError(f'OOF row has no initial trace: {raw["id"]}')
            answer = scored_answer(trace)
            if answer not in tuple('ABCD') or trace.get('hit_cap'):
                continue
            item = design['items'][int(trace['index'])]
            rows.append({
                'id': raw['id'], 'source_id': raw['source_id'],
                'condition': raw['condition'],
                'cue_type': raw.get('cue_type') or (
                    'sycophancy' if raw['condition'].endswith('_sycophancy')
                    else 'authority'),
                'target': int(raw['target']),
                'answer': answer,
                'correct_letter': item['correct_letter'],
                'answer_correct': int(answer == item['correct_letter']),
                **{field: float(raw[field]) for field in (
                    'text_loss_bits', 'state_loss_bits', 'joint_loss_bits',
                    'gap_bits')},
            })
    if not rows:
        raise ValueError(f'No eligible disagreement rows at {checkpoint}')

    inputs = {
        'design.json': file_hash(run / 'design.json'),
        'traces.pkl': file_hash(run / 'traces.pkl'),
        'analysis.json': file_hash(analysis_path),
        export_name: export_sha,
        'script': file_hash(Path(__file__)),
    }
    signature = digest({
        'schema': SCHEMA, 'design_id': design['design_id'],
        'checkpoint': checkpoint, 'bootstrap': bootstrap, 'inputs': inputs})
    destination = (Path(output) if output else
                   run / f'accuracy_gap_{signature[:12]}.json')
    result = {
        'schema': SCHEMA, 'signature': signature,
        'design_id': design['design_id'], 'checkpoint': checkpoint,
        'population': 'parsed_uncapped_disagreement_initial_responses',
        'target': analysis.get('target', 'switch_to_cue'),
        'gap_definition': 'text_loss_bits - joint_loss_bits',
        'correctness_definition': (
            'scored initial cued answer equals frozen BBH correct_letter'),
        'reader_fit': ('Frozen full-population grouped OOF predictions; no '
                       'reader is refit or selected by correctness.'),
        'post_outcome_warning': (
            'Correctness is a post-treatment outcome strongly coupled to cue '
            'following. This is descriptive moderation, not a causal effect '
            'of verbalization gap on accuracy.'),
        'bootstrap': {'draws': bootstrap, 'unit': 'source item'},
        'inputs': inputs,
        'by_cue_type': summarize(rows, design['analysis']['seed'], bootstrap),
    }
    if destination.exists():
        existing = json.loads(destination.read_text())
        if existing.get('signature') != signature:
            raise ValueError(f'Existing output has another signature: {destination}')
        return existing
    atomic(destination, result)
    print('wrote', destination)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True)
    parser.add_argument('--output')
    parser.add_argument('--checkpoint', default='initial/pre_answer')
    parser.add_argument('--bootstrap', type=int, default=2000)
    args = parser.parse_args(argv)
    if args.bootstrap < 100:
        parser.error('--bootstrap must be at least 100')
    analyze(args.run, args.output, args.checkpoint, args.bootstrap)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
