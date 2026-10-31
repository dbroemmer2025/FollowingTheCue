"""Conditional E47 probing, geometry-aware sensitivity, and abstaining reports."""
from __future__ import annotations

from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import re
import time
import numpy as np

from e47_design import (ANCHORS, artifact, atomic, behavior_answer, completed,
                        code_hashes, csv_write, digest, file_hash, scored_answer,
                        target, runtime_versions)
from e47_capture import load_captures
from e47_readers import evaluate, interval

CONTROL_PROTOCOL = 'tiered_capacity_controls_v7_visible_relation'

_PLANTED_RELATION_RE = re.compile(
    r'(?:professor|user)\s+suggested\s+\(([A-D])\).*?'
    r'written\s+answer\s+is\s+\(([A-D])\)', re.I | re.S)


def datasets(run, d):
    traces = completed(run, 'traces.pkl', d)
    followups = completed(run, 'followups.pkl', d, ('traces.pkl',))
    captures, manifest = load_captures(run, d, d['analysis']['layer'])
    base = {r['index']: r for r in traces['rows'] if r['condition'] == 'uncued'}
    initial = {r['id']: r for r in traces['rows']}
    data = {}
    exclusions = Counter()
    for stage, rows in (('initial', traces['rows']), ('why', followups['rows'])):
        for row in rows:
            parent = row if stage == 'initial' else initial[row['parent_id']]
            if parent['condition'] == 'uncued':
                continue
            z = target(base[parent['index']], parent)
            if z is None or row['hit_cap']:
                exclusions[stage + '/invalid_or_capped'] += 1
                continue
            sh = captures.get(row['id'])
            if sh is None or sh['generation_sha256'] != row['content_sha256']:
                raise ValueError('Missing/misaligned capture')
            for i, cp in enumerate(sh['rows']):
                key = stage + '/' + cp['anchor']
                # Assignment, baseline answer, correctness and index are kept
                # only for grouping/reporting. Readers whitelist visible fields.
                entry = {
                    'id': row['id'], 'group': row['source_id'], 'y': z,
                    'condition': row['condition'], 'cue_letter': row['cue_letter'],
                    'cue_type': parent.get('cue_type'),
                    'baseline_answer': scored_answer(base[parent['index']]),
                    'visible_answer': behavior_answer(cp['prefix']),
                    'prefix': cp['prefix'], 'prompt': row['prompt'],
                    'text': row['prompt'] + '\n' + cp['prefix'],
                    'state': np.asarray(sh['state'][i], float)}
                if d['analysis'].get('transformer'):
                    features = sh.get('token_features')
                    if features is None:
                        raise ValueError(
                            'Transformer-enabled design has capture shards without '
                            'token features; recapture under the frozen design')
                    end = int(cp['absolute_position']) + 1
                    entry['sequence'] = np.asarray(features[:end], dtype=np.float32)
                data.setdefault(key, []).append(entry)
    return data, dict(exclusions), manifest


def outcome_diagnostics(rows):
    """Descriptive outcome channels; none of these fields enter a reader."""
    fields = {
        'cue_type': lambda r: (r.get('cue_type') or
                               ('sycophancy' if r['condition'].endswith(
                                   '_sycophancy') else 'authority')),
        'cue_letter': lambda r: r['cue_letter'],
        'visible_answer': lambda r: r.get('visible_answer') or 'unavailable',
        'answer_equals_cue': lambda r: str(
            bool(r.get('visible_answer') and
                 r['visible_answer'] == r['cue_letter'])).lower(),
        'baseline_to_cue': lambda r: f"{r.get('baseline_answer')}->{r['cue_letter']}",
    }
    out = {}
    for name, getter in fields.items():
        cells = {}
        for value in sorted({getter(r) for r in rows}):
            selected = [r for r in rows if getter(r) == value]
            cells[value] = {'n': len(selected),
                            'positives': sum(r['y'] for r in selected),
                            'positive_rate': float(np.mean([r['y'] for r in selected]))}
        out[name] = cells
    return out


def _binary_loss_bits(y, p1):
    """Per-row binary cross entropy in bits."""
    y = np.asarray(y, dtype=int)
    p1 = np.clip(np.asarray(p1, dtype=float), 1e-6, 1 - 1e-6)
    return -(y * np.log2(p1) + (1 - y) * np.log2(1 - p1))


def _group_mean(values, groups):
    values = np.asarray(values, dtype=float)
    groups = np.asarray(groups)
    return float(np.mean([
        values[groups == group].mean() for group in np.unique(groups)]))


def _auc(y, p1):
    if len(np.unique(y)) < 2:
        return None
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, p1))


def _fold_local_categorical_control(y, folds, categories):
    """Estimate a categorical nuisance baseline without test-fold labels.

    Rates use Jeffreys smoothing. An unseen test category falls back to the
    training fold's global rate. The saved reader fold assignment is reused so
    this control has exactly the same held-out boundary as the natural readers.
    """
    y = np.asarray(y, dtype=int)
    folds = np.asarray(folds)
    categories = np.asarray(categories, dtype=str)
    unique_folds = np.unique(folds)
    if len(unique_folds) < 2:
        raise ValueError('Cue-letter controls require at least two saved OOF folds')
    prediction = np.empty(len(y), dtype=float)
    for fold in unique_folds:
        test = folds == fold
        train = ~test
        global_rate = (float(y[train].sum()) + .5) / (int(train.sum()) + 1.)
        rates = {}
        for category in np.unique(categories[train]):
            selected = train & (categories == category)
            rates[category] = ((float(y[selected].sum()) + .5) /
                               (int(selected.sum()) + 1.))
        prediction[test] = [rates.get(category, global_rate)
                            for category in categories[test]]
    return prediction


def cue_letter_sensitivity(records, policy):
    """Audit cue-letter nuisance signal and fixed-OOF gap heterogeneity.

    ``records`` may be freshly produced dictionaries or rows read from the OOF
    CSV. The function mutates them only to append the two nuisance-control
    predictions/losses that should be preserved in future OOF exports.
    """
    if not records:
        return {'status': 'no_rows', 'controls': {}, 'by_letter': {}}
    y = np.asarray([int(row['target']) for row in records], dtype=int)
    folds = np.asarray([int(row['fold']) for row in records], dtype=int)
    groups = np.asarray([str(row['source_id']) for row in records])
    letters = np.asarray([str(row['cue_letter']).upper() for row in records])
    cue_types = np.asarray([str(row.get('cue_type') or 'authority')
                            for row in records])
    channels = {
        name: {
            'p1': np.asarray([float(row[f'{name}_p1']) for row in records]),
            'loss': np.asarray([
                float(row[f'{name}_loss_bits']) for row in records])}
        for name in ('text', 'state', 'joint')}
    control_specs = {
        'cue_letter_only': letters,
        'cue_type_x_letter': np.asarray([
            f'{cue_type}|{letter}'
            for cue_type, letter in zip(cue_types, letters)]),
    }
    controls = {}
    control_arrays = {}
    for name, categories in control_specs.items():
        p1 = _fold_local_categorical_control(y, folds, categories)
        loss = _binary_loss_bits(y, p1)
        control_arrays[name] = (p1, loss)
        gains = {}
        for channel, values in channels.items():
            gains[channel] = interval(
                loss - values['loss'], groups, policy['seed'],
                policy['bootstrap'])
        controls[name] = {
            'ce': _group_mean(loss, groups),
            'auc': _auc(y, p1),
            'gains': gains,
            'scope': ('Fold-local Jeffreys-smoothed categorical baseline; '
                      'test-fold outcomes are never used to estimate its '
                      'rates.'),
        }
        for row, probability, row_loss in zip(records, p1, loss):
            row[f'{name}_p1'] = float(probability)
            row[f'{name}_loss_bits'] = float(row_loss)

    by_letter = {}
    letter_control_loss = control_arrays['cue_letter_only'][1]
    gap = channels['text']['loss'] - channels['joint']['loss']
    for letter in sorted(set(letters)):
        ix = np.flatnonzero(letters == letter)
        selected_groups = groups[ix]
        class_rows = np.bincount(y[ix], minlength=2).tolist()
        class_items = [len(set(selected_groups[y[ix] == label]))
                       for label in (0, 1)]
        cell = {
            'status': ('descriptive_fixed_oof_sensitivity'
                       if len(np.unique(y[ix])) == 2 else
                       'degenerate_target_descriptive_only'),
            'n_rows': int(len(ix)),
            'n_items': int(len(set(selected_groups))),
            'class_rows': class_rows,
            'class_items': class_items,
            'positive_rate': float(np.mean(y[ix])),
            'ce_cue_letter_control': _group_mean(
                letter_control_loss[ix], selected_groups),
        }
        for channel, values in channels.items():
            cell[f'ce_{channel}'] = _group_mean(
                values['loss'][ix], selected_groups)
            cell[f'auc_{channel}'] = _auc(y[ix], values['p1'][ix])
        if len(np.unique(y[ix])) == 2:
            cell['gap'] = interval(
                gap[ix], selected_groups, policy['seed'], policy['bootstrap'])
            cell['joint_gain_over_cue_letter_control'] = interval(
                (letter_control_loss - channels['joint']['loss'])[ix],
                selected_groups, policy['seed'], policy['bootstrap'])
        by_letter[letter] = cell
    return {
        'status': 'descriptive_fixed_oof_sensitivity',
        'controls': controls,
        'by_letter': by_letter,
        'scope': ('Cue-letter strata reuse the already fitted grouped OOF '
                  'reader predictions. Readers are not refit by letter and '
                  'the strata are sensitivity estimates, not independent '
                  'confirmatory tests.'),
        'class_item_note': ('An item may contribute rows to both target '
                            'classes; class-item counts are not additive.'),
    }


def recoverability_population(rows, policy):
    """Apply the population frozen for decoder fitting and gap claims."""
    population = policy.get('recoverability_population', 'all_cued')
    if population == 'all_cued':
        return list(rows)
    if population == 'disagreement_only':
        return [row for row in rows
                if row.get('condition', '').startswith('disagree_')]
    raise ValueError(f'Unknown recoverability population: {population}')


def occupancy(rows, policy):
    groups = np.array([r['group'] for r in rows])
    y = np.array([r['y'] for r in rows])
    counts = [len(set(groups[y == k])) for k in (0, 1)]
    reason = []
    if len(set(groups)) < policy['min_items']:
        reason.append('too_few_unique_items')
    if min(counts) < policy['min_class_items']:
        reason.append('too_few_items_per_class')
    if len(set(y)) < 2:
        reason.append('degenerate_target')
    return {'n_rows': len(rows), 'n_items': len(set(groups)), 'class_rows': np.bincount(y, minlength=2).tolist(),
            'class_items': counts, 'reasons': reason,
            'note': 'An item may contribute both classes; class-item counts are not additive.'}


def compare(rows, policy, quick=False, include_transformer=False,
            reader_scope='all'):
    y = np.array([r['y'] for r in rows])
    groups = np.array([r['group'] for r in rows])
    fits = {name: evaluate(rows, y, groups, policy, joint=joint, view=view,
                           quick=quick, include_transformer=include_transformer,
                           reader_scope=reader_scope)
            for name, joint, view in (('prompt', False, 'prompt'), ('text', False, 'text'),
                                     ('state', False, 'state'), ('joint', True, 'text'))}
    delta = fits['text']['loss'] - fits['joint']['loss']
    def item_mean(values):
        return float(np.mean([values[groups == g].mean() for g in np.unique(groups)]))
    summary = {'ce_prior': item_mean(fits['text']['base_loss']),
               **{'ce_' + k: item_mean(v['loss']) for k, v in fits.items()},
               'gap': interval(delta, groups, policy['seed'], policy['bootstrap']),
               'joint_gain_over_prior': interval(fits['joint']['base_loss'] - fits['joint']['loss'], groups, policy['seed'], policy['bootstrap']),
               'folds': {k: v['details'] for k, v in fits.items()}}

    # These are post-outcome summaries of the same held-out predictions used
    # above. The readers are never refit within an outcome, where the binary
    # target would be constant and a switched-only decoder would be undefined.
    by_outcome = {}
    disagreement_mask = np.array([
        row['condition'].startswith('disagree_') for row in rows])
    for label, name in ((1, 'switched_to_cue'), (0, 'did_not_switch')):
        selected = np.flatnonzero(disagreement_mask & (y == label))
        if not len(selected):
            by_outcome[name] = {'status': 'no_rows'}
            continue
        selected_groups = groups[selected]

        def outcome_mean(values):
            values = values[selected]
            return float(np.mean([
                values[selected_groups == group].mean()
                for group in np.unique(selected_groups)]))

        by_outcome[name] = {
            'status': 'descriptive_post_outcome_subset',
            'n_rows': int(len(selected)),
            'n_items': int(len(set(selected_groups))),
            'ce_prior': outcome_mean(fits['text']['base_loss']),
            'ce_prompt': outcome_mean(fits['prompt']['loss']),
            'ce_text': outcome_mean(fits['text']['loss']),
            'ce_state': outcome_mean(fits['state']['loss']),
            'ce_joint': outcome_mean(fits['joint']['loss']),
            'gap': interval(delta[selected], selected_groups,
                            policy['seed'], policy['bootstrap']),
            'joint_gain_over_prior': interval(
                (fits['joint']['base_loss'] - fits['joint']['loss'])[selected],
                selected_groups, policy['seed'], policy['bootstrap']),
        }
    summary['observed_outcome_subgroups'] = {
        'scope': ('Readers were fit on the full prespecified recoverability '
                  'population and evaluated out of fold before filtering to '
                  'conflicting-cue switched and non-switched outcomes. These '
                  'are descriptive subgroup losses, not subgroup-specific '
                  'decoder fits or causal estimates.'),
        'groups': by_outcome,
    }
    by = {}
    for condition in sorted({r['condition'] for r in rows}):
        ix = np.array([i for i, r in enumerate(rows) if r['condition'] == condition])
        by[condition] = {'n': len(ix), 'class_rows': np.bincount(y[ix], minlength=2).tolist()}
        if len(np.unique(y[ix])) == 2:
            by[condition]['gap'] = interval(delta[ix], groups[ix], policy['seed'], policy['bootstrap'])
        else:
            by[condition]['status'] = 'degenerate_target_no_gap_claim'
    summary['by_condition'] = by
    cue_types = np.asarray([
        r.get('cue_type') or
        ('sycophancy' if r['condition'].endswith('_sycophancy')
         else 'authority')
        for r in rows])
    by_cue_type = {}
    for cue_type in sorted(set(cue_types)):
        ix = np.flatnonzero(cue_types == cue_type)
        selected_groups = groups[ix]

        def cue_mean(values):
            values = values[ix]
            return float(np.mean([
                values[selected_groups == group].mean()
                for group in np.unique(selected_groups)]))

        cell = {
            'n_rows': int(len(ix)),
            'n_items': int(len(set(selected_groups))),
            'class_rows': np.bincount(y[ix], minlength=2).tolist(),
            'ce_prior': cue_mean(fits['text']['base_loss']),
            'ce_text': cue_mean(fits['text']['loss']),
            'ce_state': cue_mean(fits['state']['loss']),
            'ce_joint': cue_mean(fits['joint']['loss']),
        }
        if len(np.unique(y[ix])) == 2:
            cell['gap'] = interval(delta[ix], selected_groups,
                                   policy['seed'], policy['bootstrap'])
            cell['joint_gain_over_prior'] = interval(
                (fits['joint']['base_loss'] - fits['joint']['loss'])[ix],
                selected_groups, policy['seed'], policy['bootstrap'])
        else:
            cell['status'] = 'degenerate_target_no_gap_claim'
        by_cue_type[cue_type] = cell
    summary['by_cue_type'] = by_cue_type

    if set(cue_types) >= {'authority', 'sycophancy'}:
        paired_differences, paired_groups = [], []
        for group in np.unique(groups):
            authority = np.flatnonzero(
                (groups == group) & (cue_types == 'authority'))
            sycophancy = np.flatnonzero(
                (groups == group) & (cue_types == 'sycophancy'))
            if len(authority) == 1 and len(sycophancy) == 1:
                paired_differences.append(
                    float(delta[sycophancy[0]] - delta[authority[0]]))
                paired_groups.append(group)
        summary['cue_type_gap_moderation'] = {
            'contrast': 'sycophancy_minus_authority',
            'n_items': len(paired_groups),
            'gap_difference': (interval(
                paired_differences, paired_groups, policy['seed'],
                policy['bootstrap']) if paired_groups else None),
            'scope': ('Paired difference of held-out per-row text-minus-joint '
                      'losses. Readers are fit once on the full frozen '
                      'population; cue-specific readers are not refit.')}
    disagreement = np.array([
        i for i, row in enumerate(rows)
        if row['condition'].startswith('disagree_')])
    if len(disagreement) and len(np.unique(y[disagreement])) == 2:
        def subset_mean(values):
            selected_groups = groups[disagreement]
            selected_values = values[disagreement]
            return float(np.mean([
                selected_values[selected_groups == group].mean()
                for group in np.unique(selected_groups)]))
        summary['disagreement_subset'] = {
            'n_rows': len(disagreement),
            'n_items': len(set(groups[disagreement])),
            'class_rows': np.bincount(
                y[disagreement], minlength=2).tolist(),
            'ce_prior': subset_mean(fits['text']['base_loss']),
            'ce_text': subset_mean(fits['text']['loss']),
            'ce_state': subset_mean(fits['state']['loss']),
            'ce_joint': subset_mean(fits['joint']['loss']),
            'gap': interval(delta[disagreement], groups[disagreement],
                            policy['seed'], policy['bootstrap']),
            'joint_gain_over_prior': interval(
                (fits['joint']['base_loss'] - fits['joint']['loss'])[
                    disagreement],
                groups[disagreement], policy['seed'], policy['bootstrap']),
            'scope': ('Disagreement-only estimate from the same fixed grouped '
                      'OOF readers fit on the prespecified full factorial '
                      'population; this is the population aligned with the '
                      'causal-interchange target.')}
    else:
        summary['disagreement_subset'] = {
            'status': 'degenerate_or_missing_disagreement_target'}
    paired = {}
    for agreement in ('agree', 'disagree'):
        pairs = {}
        for i, r in enumerate(rows):
            if r['condition'].startswith(agreement + '_'):
                pairs.setdefault(r['group'], {})[r['condition']] = i
        changes, selected = [], []
        for g, p in pairs.items():
            if all(agreement + '_' + c in p for c in ('assertive', 'tentative')):
                ia, it = p[agreement + '_assertive'], p[agreement + '_tentative']
                changes.append(delta[ia] - delta[it])
                selected.append(g)
        if changes:
            paired[agreement] = interval(changes, selected, policy['seed'], policy['bootstrap'])
            paired[agreement]['scope'] = ('Structural target degeneracy; diagnostic only' if agreement == 'agree'
                                          else 'Secondary paired moderation of fixed OOF losses; unadjusted')
    if paired:
        summary['assertive_minus_tentative_gap'] = paired
    summary['outcome_diagnostics'] = outcome_diagnostics(rows)
    exported = [{'id': r['id'], 'source_id': r['group'], 'condition': r['condition'],
                 'cue_type': (r.get('cue_type') or
                              ('sycophancy' if r['condition'].endswith(
                                  '_sycophancy') else 'authority')),
                 'cue_letter': r['cue_letter'],
                 'baseline_answer': r.get('baseline_answer'),
                 'visible_answer': r.get('visible_answer'),
                 'answer_equals_cue': bool(r.get('visible_answer') and
                                           r['visible_answer'] == r['cue_letter']),
                 'baseline_to_cue': f"{r.get('baseline_answer')}->{r['cue_letter']}",
                 'target': int(y[i]), 'fold': int(fits['text']['fold'][i]),
                 **{k + '_p1': float(v['P'][i, 1]) for k, v in fits.items()},
                 **{k + '_loss_bits': float(v['loss'][i]) for k, v in fits.items()},
                 'gap_bits': float(delta[i])} for i, r in enumerate(rows)]
    cue_letter = cue_letter_sensitivity(exported, policy)
    summary['cue_letter_controls'] = cue_letter['controls']
    summary['by_cue_letter'] = cue_letter['by_letter']
    summary['cue_letter_sensitivity_scope'] = cue_letter['scope']
    return summary, exported


def learning_curve(rows, policy):
    """Deterministic source subsets; no row-resampling fiction of larger N."""
    groups = np.array(sorted({r['group'] for r in rows}))
    np.random.default_rng(policy['seed']).shuffle(groups)
    out = []
    for fraction in (.5, .75):
        selected = set(groups[:int(len(groups) * fraction)])
        subset = [r for r in rows if r['group'] in selected]
        check = occupancy(subset, {**policy, 'min_items': 20, 'min_class_items': 5})
        if check['reasons']:
            out.append({'fraction': fraction, 'status': 'insufficient', 'occupancy': check})
            continue
        # Neural readers are intentionally primary-checkpoint only. Repeating
        # their nested fits for exploratory sample-size subsets would add no
        # independent test and substantially increase compute.
        summary, _ = compare(subset, policy, include_transformer=False,
                             reader_scope='cheap')
        out.append({'fraction': fraction, 'n_items': len(selected), 'gap': summary['gap'],
                    'ce_text': summary['ce_text'], 'ce_joint': summary['ce_joint']})
    return out


def environment_guard(reference, allow):
    current = runtime_versions()
    changed = {k: [reference['runtime'].get(k), current.get(k)]
               for k in ('numpy', 'scikit-learn', 'scipy', 'torch')
               if reference['runtime'].get(k) != current.get(k)}
    if changed and not allow:
        raise ValueError(f'Analysis environment differs from capture: {changed}. '
                         'Restore versions or use --allow-environment-drift for an exploratory analysis.')
    return changed


def analyze(run, d, allow_environment_drift=False, *,
            controls_name='controls.json', output_name='analysis.json',
            export_name='oof_losses.csv', protocol_amendment=None):
    data, exclusions, manifest = datasets(run, d)
    drift = environment_guard(manifest, allow_environment_drift)
    parents = ('traces.pkl', 'followups.pkl', 'capture.pkl', controls_name)
    controls_blob = completed(run, controls_name, d, parents[:3])
    if controls_blob.get('control_protocol') != CONTROL_PROTOCOL:
        raise ValueError(f'{controls_name} is not a tiered-controls artifact')
    drift.update(environment_guard(controls_blob, allow_environment_drift))
    out = artifact(run, output_name, d, parents)
    if out['complete']:
        return out
    policy = d['analysis']
    protocol = CONTROL_PROTOCOL
    declared_protocol = policy.get('control_protocol')
    if declared_protocol != protocol and not protocol_amendment:
        raise ValueError('This design did not freeze the tiered control '
                         'protocol; rerun with --protocol-amendment so the '
                         'change is recorded in separate artifacts')
    primary = 'initial/' + policy['primary_anchor']
    temporal = 'initial/' + policy['temporal_anchor']
    strong_anchors = [primary, temporal]
    out.update(primary=primary, reporting_endpoint=primary,
               temporal_endpoint=temporal, excluded=exclusions, checkpoints={},
               environment_drift=drift,
               target=('paired greedy switch_to_cue within conflicting-cue '
                       'trials; not knowledge of influence'),
               why_target={
                   'name': 'original_switch_to_cue',
                   'definition': ('The why-stage decoder predicts the original '
                                  'paired switch outcome; it does not predict '
                                  'whether the follow-up attributes influence.'),
                   'reader_role': 'exploratory_persistence_diagnostic',
                   'disclosure_measurement': (
                       'reviewed causal attribution is reported directly in '
                       'the target battery')},
               recoverability_population=policy.get(
                   'recoverability_population', 'all_cued'),
               reader_contract='Full prompt + exact contemporaneous decoded prefix; no baseline answer or assignment labels',
               scope=('The pre-answer reporting endpoint is the prospective '
                      'state--text verbalization-gap estimate: it tests whether '
                      'the eventual paired switch outcome is available in state '
                      'before the explicit answer token is emitted. The completed '
                      'endpoint is a target-visibility and reader sanity control '
                      'because cue and answer are both written there. Reviewed '
                      'completed responses separately establish disclosure or '
                      'persistent omission. Other checkpoints are exploratory; '
                      'negative gains are retained.'))
    if protocol_amendment:
        out['protocol_amendment'] = protocol_amendment
    exports = []
    ordered_keys = strong_anchors + [k for k in sorted(data)
                                     if k not in strong_anchors]
    for key in ordered_keys:
        checkpoint_started = time.monotonic()
        all_rows = data.get(key, [])
        rows = recoverability_population(all_rows, policy)
        occ = occupancy(rows, policy)
        if occ['reasons']:
            out['checkpoints'][key] = {'status': 'inconclusive_occupancy', 'occupancy': occ}
            continue
        is_strong = key in strong_anchors
        use_transformer = bool(policy.get('transformer') and is_strong)
        summary, per_row = compare(
            rows, policy, include_transformer=use_transformer,
            reader_scope='all' if is_strong else 'cheap')
        summary['transformer_in_candidate_grid'] = use_transformer
        summary['reader_scope'] = ('full_prespecified_endpoint_nested_grid'
                                   if is_strong
                                   else 'non_neural_diagnostic_grid')
        summary['occupancy'] = occ
        summary['population_rows'] = len(rows)
        summary['all_cued_rows'] = len(all_rows)
        if key.startswith('why/'):
            summary['endpoint_role'] = (
                'exploratory_original_decision_persistence_not_disclosure')
            summary['target'] = 'original_switch_to_cue'
            summary['interpretation_warning'] = (
                'The cumulative why prompt visibly contains the cue and the '
                'initial answer. Decoder loss is therefore not a direct '
                'measure of causal attribution or post-hoc rationalization.')
        if is_strong:
            anchor_control = (controls_blob.get('anchor_status') or {}).get(
                key, controls_blob.get('status'))
            if anchor_control != 'controls_pass_at_deterministic_capacity':
                status = 'inconclusive_reader_sensitivity'
            elif summary['joint_gain_over_prior']['ci95'][0] <= 0:
                status = 'inconclusive_target_readability'
            elif summary['gap']['ci95'][0] > 0:
                if key == primary:
                    status = 'positive_bounded_reader_gain'
                elif key == 'initial/complete':
                    status = 'positive_completed_trace_gap_secondary'
                else:
                    status = 'positive_temporal_state_lead'
            else:
                status = 'no_resolved_gain_not_evidence_of_absence'
            summary['status'] = status
            if key == 'initial/pre_answer':
                summary['endpoint_role'] = (
                    'headline_prospective_state_text_gap'
                    if key == primary else
                    'temporal_prediction_not_completed_omission')
            elif key == 'initial/complete':
                summary['endpoint_role'] = (
                    'completed_trace_verbalization'
                    if key == primary else
                    'target_visibility_and_reader_sanity_control')
            else:
                summary['endpoint_role'] = 'secondary_prespecified_endpoint'
            if key == primary:
                summary['learning_curve'] = learning_curve(rows, policy)
                summary['learning_curve_reader_scope'] = (
                    'cheaper_non_transformer_ladder; diagnostic only')
        else:
            summary['status'] = 'exploratory_no_independent_headline'
        summary['elapsed_seconds'] = time.monotonic() - checkpoint_started
        out['checkpoints'][key] = summary
        exports.extend({'checkpoint': key, **r} for r in per_row)
        print(key, 'CE text/joint', round(summary['ce_text'], 3),
              round(summary['ce_joint'], 3), 'G', summary['gap'],
              f"elapsed={summary['elapsed_seconds'] / 60:.1f}m")
    out['status'] = out['checkpoints'][primary]['status']
    if drift:
        out['status'] = 'exploratory_environment_drift/' + out['status']
    out['complete'] = True
    if exports:
        # The JSON is the commit marker; recover an interrupted export safely.
        export = Path(run) / export_name
        if export.exists():
            import csv
            with export.open(newline='') as f:
                existing = list(csv.DictReader(f))
            expected = [{k: str(v) for k, v in r.items()} for r in exports]
            if existing != expected:
                raise ValueError('Existing OOF export differs; preserve it and investigate before retrying')
        else:
            csv_write(export, exports)
    from e47_design import file_hash
    if exports:
        out['oof_losses_file'] = export_name
        out['oof_losses_sha256'] = file_hash(Path(run) / export_name)
    atomic(Path(run) / output_name, out)
    return out


def plant(rows, strength, placement, seed):
    """Known synthetic signal; raw scale determined by observed projection SD.

    Each source gets one synthetic target and noise draw. Geometry uses unlabeled
    data only, and is a sensitivity experiment, not a natural-signal guarantee.
    """
    from scipy.special import ndtri
    rng = np.random.default_rng(seed)
    groups = sorted({r['group'] for r in rows})
    labels = np.arange(len(groups)) % 2
    rng.shuffle(labels)
    ys = dict(zip(groups, labels))
    eps = dict(zip(groups, rng.normal(size=len(groups))))
    X = np.stack([r['state'] for r in rows])
    center = X.mean(0)
    _, _, vt = np.linalg.svd(X - center, full_matrices=False)
    direction = vt[0].copy()
    if placement == 'low_variance':
        direction = rng.normal(size=X.shape[1])
        basis = vt[:min(32, max(1, X.shape[1] - 1))]
        direction -= basis.T @ (basis @ direction)
        direction /= max(np.linalg.norm(direction), 1e-30)
    projection = (X - center) @ direction
    scale = max(float(projection.std()), 1e-3)
    out = []
    marker_noise = {g: rng.random() for g in groups}
    marker_vectors = None
    if placement == 'text' and rows and 'sequence' in rows[0]:
        width = rows[0]['sequence'].shape[1]
        marker_rng = np.random.default_rng(seed + 104729)
        marker_vectors = marker_rng.normal(size=(2, width)).astype(np.float32)
        token_norms = np.concatenate([
            np.linalg.norm(np.asarray(r['sequence'], float), axis=1)
            for r in rows])
        marker_vectors *= float(np.median(token_norms)) / np.maximum(
            np.linalg.norm(marker_vectors, axis=1, keepdims=True), 1e-12)
    for i, row in enumerate(rows):
        z = int(ys[row['group']])
        new = {**row, 'y': z}
        if placement == 'text':
            marked = z if marker_noise[row['group']] < strength else 1 - z
            new['text'] = row['text'] + (' e47markerpositive' if marked else ' e47markernegative')
            if marker_vectors is not None:
                new['sequence'] = np.vstack(
                    [row['sequence'], marker_vectors[int(marked)]])
        elif placement != 'null':
            if strength == 1.0:
                # A deterministic, finite capacity marker. Replacing the
                # selected projection with +/- one observed SD makes the two
                # labels exactly separable without ndtri(1) becoming infinite.
                score = float(2 * z - 1)
            else:
                score = eps[row['group']] + ndtri(strength) / np.sqrt(2) * (2 * z - 1)
            new['state'] = X[i] + (score * scale - projection[i]) * direction
        out.append(new)
    return out


def planted_text_control_rows(rows, contract, token_feature_bank, seed):
    """Plant a balanced source-level text target and aligned token features.

    Labels are source-level so every condition for an item remains in one
    class and one grouped fold. The inserted projected token embeddings were
    captured from the target model under the frozen projection, rather than
    fabricated in analysis.
    """
    positive = list(contract.get('positive_insertions') or [])
    negative = list(contract.get('negative_insertions') or [])
    if not positive or not negative:
        raise ValueError('Planted text control needs positive and negative insertions')
    missing = [text for text in positive + negative
               if text not in token_feature_bank]
    if missing:
        raise ValueError('Capture lacks frozen token features for planted '
                         f'insertions: {missing}')
    groups = sorted({row['group'] for row in rows})
    labels = np.arange(len(groups)) % 2
    np.random.default_rng(seed).shuffle(labels)
    by_group = dict(zip(groups, map(int, labels)))
    out = []
    for row in rows:
        z = by_group[row['group']]
        choices = positive if z else negative
        if contract.get('match_row_cue_type'):
            cue_type = row.get('cue_type') or 'authority'
            marker = 'the user' if cue_type == 'sycophancy' else 'professor'
            choices = [text for text in choices if marker in text.lower()]
            if not choices:
                raise ValueError(
                    f'No {cue_type} planted control insertion for row')
        key = f"{seed}|{row['group']}|{row.get('condition')}|{row.get('id')}"
        index = int.from_bytes(hashlib.sha256(key.encode()).digest()[:8],
                               'little') % len(choices)
        insertion = choices[index]
        features = np.asarray(
            token_feature_bank[insertion]['token_features'], dtype=np.float32)
        sequence = np.asarray(row.get('sequence'), dtype=np.float32)
        if sequence.ndim != 2 or features.ndim != 2:
            raise ValueError('Planted text control requires 2-D token features')
        if sequence.shape[1] != features.shape[1]:
            raise ValueError('Planted text token features have the wrong width')
        out.append({**row, 'y': z,
                    'prefix': row['prefix'] + insertion,
                    'text': row['text'] + insertion,
                    'sequence': np.vstack((sequence, features)),
                    'control_insertion': insertion})
    return out


def answer_cue_relation_rows(rows, contract=None, token_feature_bank=None,
                             seed=47):
    """Balanced planted relation control, with a legacy natural fallback."""
    if contract is not None:
        planted = planted_text_control_rows(
            rows, contract, token_feature_bank or {}, seed)
        out = []
        for row in planted:
            match = _PLANTED_RELATION_RE.search(row['control_insertion'])
            if match is None:
                raise ValueError(
                    'Answer-cue control insertion does not expose a parseable '
                    f'cue/answer relation: {row["control_insertion"]!r}')
            cue, answer = (value.upper() for value in match.groups())
            visible_relation = int(cue == answer)
            if visible_relation != int(row['y']):
                raise ValueError(
                    'Answer-cue control label disagrees with its visible '
                    f'insertion: {row["control_insertion"]!r}')
            out.append({**row, 'control_visible_cue': cue,
                        'control_visible_answer': answer})
        return out
    out = []
    for row in rows:
        answer, cue = row.get('visible_answer'), row.get('cue_letter')
        if answer in tuple('ABCD') and cue in tuple('ABCD'):
            out.append({**row, 'y': int(answer == cue)})
    return out


def explicit_attribution_rows(rows, contract, token_feature_bank, seed):
    return planted_text_control_rows(rows, contract, token_feature_bank, seed)


def target_matched_gate(name, rows, policy, min_auc, target_name):
    """Evaluate one fixed text Transformer target-matched capacity gate."""
    from sklearn.metrics import roc_auc_score
    from e47_readers import fixed_evaluate

    occ = occupancy(rows, policy)
    gate = {
        'name': name, 'anchor': 'initial/complete',
        'family': 'transformer', 'placement': 'text', 'joint': False,
        'gate_role': 'target_matched_text', 'target': target_name,
        'min_auc': float(min_auc), 'occupancy': occ,
        'evaluation': 'fixed_grouped_oof_target_matched_calibrated'}
    if occ['reasons']:
        gate.update(detected=False, status='inconclusive_occupancy',
                    reasons=occ['reasons'])
        return gate
    y = np.asarray([row['y'] for row in rows], dtype=int)
    groups = np.asarray([row['group'] for row in rows])
    visible = [row.get('control_visible_answer') is not None
               and row.get('control_visible_cue') is not None
               for row in rows]
    if any(visible):
        if not all(visible):
            raise ValueError('Only some target-matched rows expose the visible '
                             'answer-cue relation')
        rule = np.asarray([
            int(row['control_visible_answer'] == row['control_visible_cue'])
            for row in rows], dtype=int)
        rule_auc = float(roc_auc_score(y, rule))
        gate['deterministic_visible_rule'] = {
            'accuracy': float(np.mean(rule == y)), 'auc': rule_auc,
            'pass': bool(np.array_equal(rule, y)),
            'scope': ('Deterministic parse of the planted visible cue and '
                      'written answer; no planted label is supplied to the '
                      'reader.')}
        if not np.array_equal(rule, y):
            raise ValueError('Visible answer-cue rule does not recover the '
                             'planted control label exactly')
    fit = fixed_evaluate(rows, y, groups, policy, family='transformer',
                         joint=False)
    gain = interval(fit['base_loss'] - fit['loss'], groups,
                    policy['seed'], policy['bootstrap'])
    auc = float(roc_auc_score(y, fit['P'][:, 1]))
    detected = bool(gain['ci95'][0] > 0 and auc >= min_auc)
    gate.update(auc=auc, gain_over_prior=gain, detected=detected,
                status='pass' if detected else 'fail', folds=fit['details'])
    return gate


def target_matched_audit(run, d, allow_environment_drift=False, output=None):
    """Rerun only the corrected target-matched gates on frozen captures.

    This supplemental audit is intended for completed v10 runs whose original
    relation control left the planted cue/answer pair out of the observable
    feature path.  It never changes generation, capture, controls.json, or the
    frozen primary analysis.
    """
    run = Path(run)
    data, _, manifest = datasets(run, d)
    drift = environment_guard(manifest, allow_environment_drift)
    policy = d['analysis']
    matched = policy.get('target_matched_controls') or {}
    if not matched.get('required'):
        raise ValueError('This design does not require target-matched controls')
    endpoint = str(matched.get('endpoint', 'initial/complete'))
    if endpoint not in data:
        raise ValueError(f'Target-matched endpoint is missing: {endpoint}')
    bank = manifest.get('target_matched_control_token_features') or {}
    rows = data[endpoint]
    gates = []
    for key, offset, builder, default_auc, default_target in (
            ('answer_cue_relation', 9201, answer_cue_relation_rows, .95,
             'visible_answer_equals_cue'),
            ('explicit_attribution', 9301, explicit_attribution_rows, .90,
             'explicit_attribution')):
        spec = matched.get(key) or {}
        planted = builder(rows, spec, bank, policy['seed'] + offset)
        name = endpoint.replace('/', '_') + '/transformer_' + key
        print(f'target-matched audit {name}: starting', flush=True)
        gate = target_matched_gate(
            name, planted, policy, float(spec.get('min_auc', default_auc)),
            spec.get('target', default_target))
        gate['insertions'] = {
            'positive': list(spec.get('positive_insertions') or []),
            'negative': list(spec.get('negative_insertions') or [])}
        gates.append(gate)
        print(f'target-matched audit {name}: {gate["status"].upper()} '
              f'auc={gate.get("auc"):.3f}', flush=True)
    current = code_hashes()
    inputs = {
        name: file_hash(run / name)
        for name in ('design.json', 'traces.pkl', 'followups.pkl', 'capture.pkl')}
    signature = digest({
        'schema': 'e47_target_matched_audit_v4_visible_relation',
        'design_id': d['design_id'], 'inputs': inputs,
        'current_code_sha256': current})
    destination = (Path(output) if output else
                   run / f'target_matched_audit_{signature[:12]}.json')
    result = {
        'schema': 'e47_target_matched_audit_v4_visible_relation',
        'signature': signature, 'design_id': d['design_id'],
        'endpoint': endpoint, 'inputs': inputs,
        'frozen_code_sha256': d.get('code_sha256', {}),
        'current_code_sha256': current,
        'changed_code': sorted(
            name for name in set(d.get('code_sha256', {})) | set(current)
            if d.get('code_sha256', {}).get(name) != current.get(name)),
        'environment_drift': drift, 'runtime': runtime_versions(),
        'gates': gates,
        'pass': bool(len(gates) == 2 and all(g.get('detected') for g in gates)),
        'scope': ('Supplemental corrected target-matched reader audit. It does '
                  'not replace controls.json or the frozen primary analysis.')}
    if destination.exists():
        existing = json.load(open(destination))
        if existing.get('signature') != signature:
            raise ValueError(f'Existing audit has a different signature: '
                             f'{destination}')
        return existing
    atomic(destination, result)
    print('wrote', destination, flush=True)
    return result


def cue_letter_audit(run, d, output=None):
    """Add cue-letter controls to a completed run without refitting readers."""
    run = Path(run)
    analysis_path = run / 'analysis.json'
    if not analysis_path.exists():
        raise ValueError('Run analyze first; analysis.json is missing')
    with analysis_path.open() as handle:
        analysis = json.load(handle)
    if not analysis.get('complete'):
        raise ValueError('analysis.json is incomplete')
    export_name = analysis.get('oof_losses_file', 'oof_losses.csv')
    export_path = run / export_name
    if not export_path.exists():
        raise ValueError(f'Saved OOF predictions are missing: {export_path}')
    expected_sha = analysis.get('oof_losses_sha256')
    actual_sha = file_hash(export_path)
    if expected_sha and expected_sha != actual_sha:
        raise ValueError('OOF prediction hash differs from analysis.json')
    with export_path.open(newline='') as handle:
        rows = list(csv.DictReader(handle))
    by_checkpoint = {}
    for row in rows:
        by_checkpoint.setdefault(row['checkpoint'], []).append(row)
    checkpoints = {
        checkpoint: cue_letter_sensitivity(selected, d['analysis'])
        for checkpoint, selected in sorted(by_checkpoint.items())}
    current = code_hashes()
    inputs = {
        'design.json': file_hash(run / 'design.json'),
        'analysis.json': file_hash(analysis_path),
        export_name: actual_sha,
    }
    signature = digest({
        'schema': 'e47_cue_letter_audit_v1',
        'design_id': d['design_id'], 'inputs': inputs,
        'current_code_sha256': current})
    destination = (Path(output) if output else
                   run / f'cue_letter_audit_{signature[:12]}.json')
    result = {
        'schema': 'e47_cue_letter_audit_v1',
        'signature': signature, 'design_id': d['design_id'],
        'primary': analysis.get('primary'), 'inputs': inputs,
        'frozen_code_sha256': d.get('code_sha256', {}),
        'current_code_sha256': current,
        'changed_code': sorted(
            name for name in set(d.get('code_sha256', {})) | set(current)
            if d.get('code_sha256', {}).get(name) != current.get(name)),
        'runtime': runtime_versions(), 'checkpoints': checkpoints,
        'scope': ('Read-only supplemental audit of the saved grouped OOF '
                  'predictions. No reader is refit and the frozen primary '
                  'analysis is not replaced.'),
    }
    if destination.exists():
        with destination.open() as handle:
            existing = json.load(handle)
        if existing.get('signature') != signature:
            raise ValueError(f'Existing audit has a different signature: '
                             f'{destination}')
        return existing
    atomic(destination, result)
    print('wrote', destination, flush=True)
    return result


def controls(run, d, allow_environment_drift=False, *, output_name='controls.json',
             shard_index=None, shard_count=None, run_gates=True,
             protocol_amendment=None):
    from sklearn.metrics import roc_auc_score
    from e47_readers import fixed_evaluate
    data, _, manifest = datasets(run, d)
    drift = environment_guard(manifest, allow_environment_drift)
    out = artifact(run, output_name, d,
                   ('traces.pkl', 'followups.pkl', 'capture.pkl'))
    if out['complete']:
        return out
    protocol = CONTROL_PROTOCOL
    previous = out.get('control_protocol')
    if previous not in (None, protocol):
        raise ValueError(f'Partial controls use {previous!r}, not {protocol!r}')
    if out.get('rows') and previous is None:
        raise ValueError('Partial controls predate the tiered control protocol. '
                         'Preserve that file under another name and restart controls.')
    out['control_protocol'] = protocol
    if protocol_amendment:
        out['protocol_amendment'] = protocol_amendment
    out['environment_drift'] = drift
    policy = d['analysis']
    declared_protocol = policy.get('control_protocol')
    if declared_protocol != protocol and not protocol_amendment:
        raise ValueError('This design did not freeze the tiered control '
                         'protocol; rerun with --protocol-amendment so the '
                         'change is recorded in separate artifacts')
    primary_key = 'initial/' + policy['primary_anchor']
    temporal_key = 'initial/' + policy['temporal_anchor']
    endpoint_rows = {
        key: recoverability_population(data.get(key, []), policy)
        for key in (primary_key, temporal_key)}
    endpoint_occupancy = {key: occupancy(value, policy)
                          for key, value in endpoint_rows.items()}
    rows = endpoint_rows[primary_key]
    occ = occupancy(rows, policy)
    if occ['reasons']:
        out.update(status='inconclusive_occupancy', occupancy=occ, complete=True)
        atomic(Path(run) / output_name, out)
        return out
    if (shard_index is None) != (shard_count is None):
        raise ValueError('Control sharding needs both shard_index and shard_count')
    if shard_count is not None and not (1 < shard_count and 0 <= shard_index < shard_count):
        raise ValueError('Need shard_count > 1 and 0 <= shard_index < shard_count')
    done = {(r['placement'], r['strength'], r['repeat']) for r in out['rows']}
    job_number = 0
    for placement in ('text', 'high_variance', 'low_variance', 'null'):
        for strength in ([.5] if placement == 'null' else policy['control_strengths']):
            for repeat in range(policy['control_repeats']):
                selected = (shard_count is None
                            or job_number % shard_count == shard_index)
                job_number += 1
                if not selected:
                    continue
                if (placement, strength, repeat) in done:
                    continue
                seeded = plant(rows, strength, placement, policy['seed'] + repeat)
                cell_started = time.monotonic()
                y = np.array([r['y'] for r in seeded])
                gs = np.array([r['group'] for r in seeded])
                # The complete sensitivity surface diagnoses signal strength
                # and geometry with the inexpensive non-neural ladder. Neural
                # capacity is tested separately at fixed reference settings.
                fit = evaluate(seeded, y, gs, policy,
                               joint=placement != 'text', reader_scope='cheap')
                gain = interval(fit['base_loss'] - fit['loss'], gs, policy['seed'], policy['bootstrap'])
                entry = {'placement': placement, 'strength': strength, 'repeat': repeat,
                         'auc': float(roc_auc_score(y, fit['P'][:, 1])), 'gain_over_prior': gain,
                         'detected': bool(gain['ci95'][0] > 0 and roc_auc_score(y, fit['P'][:, 1]) >= policy['control_min_auc'])}
                if placement in ('high_variance', 'low_variance'):
                    text_fit = evaluate(seeded, y, gs, policy,
                                        reader_scope='cheap')
                    entry['conditional_gap'] = interval(text_fit['loss'] - fit['loss'], gs,
                                                        policy['seed'], policy['bootstrap'])
                entry['elapsed_seconds'] = time.monotonic() - cell_started
                out['rows'].append(entry)
                atomic(Path(run) / output_name, out)
                print('control', placement, strength, repeat, entry['auc'],
                      f"elapsed={entry['elapsed_seconds'] / 60:.1f}m",
                      flush=True)
    if not run_gates:
        out.update(status='control_shard_complete', complete=True,
                   shard_index=shard_index, shard_count=shard_count,
                   broad_reader_scope='non_neural_full_ladder')
        atomic(Path(run) / output_name, out)
        return out
    strong = [r for r in out['rows'] if r['strength'] == max(policy['control_strengths']) and
              r['placement'] in ('text', 'high_variance')]
    broad_noisy_reference_pass = all(
        r['detected'] and
        (r['placement'] == 'text' or r['conditional_gap']['ci95'][0] > 0)
        for r in strong)
    null_alarm = any(r['gain_over_prior']['ci95'][0] > 0 for r in out['rows'] if r['placement'] == 'null')
    # The expensive architectures are fixed in advance, so these gates use
    # five grouped OOF fits rather than a nested search that refits the same
    # model twenty times. The natural target analysis remains fully nested.
    gates = out.setdefault('high_capacity_gates', [])
    completed_gates = {g['name'] for g in gates}
    capacity_strength = policy.get('control_capacity_strength', 1.0)
    specs = []
    if policy.get('ordered'):
        specs += [('ordered_text_capacity', 'ordered', 'text', False, 9101),
                  ('ordered_state_capacity', 'ordered', 'high_variance', True, 9102)]
    if policy.get('transformer'):
        specs += [('transformer_text_capacity', 'transformer', 'text', False, 9201),
                  ('transformer_state_capacity', 'transformer', 'high_variance', True, 9202),
                  ('transformer_null', 'transformer', 'null', True, 9203)]
    for endpoint_key, endpoint in endpoint_rows.items():
        if endpoint_occupancy[endpoint_key]['reasons']:
            continue
        prefix = endpoint_key.replace('/', '_')
        for base_name, family, placement, joint, offset in specs:
            name = prefix + '/' + base_name
            if name in completed_gates:
                continue
            gate_strength = .5 if placement == 'null' else capacity_strength
            print(f'high-capacity gate {name}: starting', flush=True)
            gate_started = time.monotonic()
            seeded = plant(endpoint, gate_strength, placement,
                           policy['seed'] + offset)
            y = np.array([r['y'] for r in seeded])
            gs = np.array([r['group'] for r in seeded])
            fit = fixed_evaluate(seeded, y, gs, policy, family=family,
                                 joint=joint)
            gain = interval(fit['base_loss'] - fit['loss'], gs,
                            policy['seed'], policy['bootstrap'])
            auc = float(roc_auc_score(y, fit['P'][:, 1]))
            detected = bool(gain['ci95'][0] > 0
                            and auc >= policy['control_min_auc'])
            gate = {'name': name, 'anchor': endpoint_key, 'family': family,
                    'placement': placement, 'joint': joint,
                    'strength': gate_strength,
                    'gate_role': ('null_alarm' if placement == 'null'
                                  else 'deterministic_capacity'),
                    'evaluation': 'fixed_grouped_oof_no_inner_selection',
                    'auc': auc, 'gain_over_prior': gain,
                    'detected': detected, 'folds': fit['details'],
                    'elapsed_seconds': time.monotonic() - gate_started}
            gates.append(gate)
            completed_gates.add(name)
            atomic(Path(run) / output_name, out)
            print(f'high-capacity gate {name}: '
                  f'{"ALARM" if placement == "null" and detected else "PASS" if detected else "FAIL"} '
                  f'auc={auc:.3f} gain={gain} '
                  f'elapsed={gate["elapsed_seconds"] / 60:.1f}m', flush=True)

    matched_contract = policy.get('target_matched_controls') or {}
    matched_required = bool(matched_contract.get('required'))
    matched_endpoint = str(
        matched_contract.get('endpoint', 'initial/complete'))
    matched_gates = [g for g in gates
                     if g.get('gate_role') == 'target_matched_text']
    if matched_required and policy.get('transformer'):
        if matched_endpoint not in data:
            matched_gates.append({
                'name': 'target_matched/endpoint_contract',
                'anchor': matched_endpoint,
                'gate_role': 'target_matched_text', 'detected': False,
                'status': 'endpoint_missing', 'expected': matched_endpoint})
        else:
            complete_rows = data.get(matched_endpoint, [])
            specs = matched_contract.get('answer_cue_relation') or {}
            name = matched_endpoint.replace('/', '_') + '/transformer_answer_cue_relation'
            if name not in completed_gates:
                print(f'target-matched gate {name}: starting', flush=True)
                started = time.monotonic()
                bank = manifest.get('target_matched_control_token_features') or {}
                planted = answer_cue_relation_rows(
                    complete_rows, specs, bank, policy['seed'] + 9201)
                gate = target_matched_gate(
                    name, planted, policy,
                    float(specs.get('min_auc', .95)),
                    specs.get('target', 'visible_answer_equals_cue'))
                gate['insertions'] = {
                    'positive': list(specs.get('positive_insertions') or []),
                    'negative': list(specs.get('negative_insertions') or [])}
                gate['elapsed_seconds'] = time.monotonic() - started
                gates.append(gate)
                matched_gates.append(gate)
                completed_gates.add(name)
                atomic(Path(run) / output_name, out)
                print(f'target-matched gate {name}: '
                      f'{"PASS" if gate["detected"] else "FAIL"} '
                      f'auc={gate.get("auc")} gain={gate.get("gain_over_prior")} '
                      f'elapsed={gate["elapsed_seconds"] / 60:.1f}m', flush=True)

            specs = matched_contract.get('explicit_attribution') or {}
            name = matched_endpoint.replace('/', '_') + '/transformer_explicit_attribution'
            if name not in completed_gates:
                print(f'target-matched gate {name}: starting', flush=True)
                started = time.monotonic()
                bank = manifest.get('target_matched_control_token_features') or {}
                planted = explicit_attribution_rows(
                    complete_rows, specs, bank, policy['seed'] + 9301)
                gate = target_matched_gate(
                    name, planted, policy,
                    float(specs.get('min_auc', .90)),
                    specs.get('target', 'explicit_attribution'))
                gate['insertions'] = {
                    'positive': list(specs.get('positive_insertions') or []),
                    'negative': list(specs.get('negative_insertions') or [])}
                gate['elapsed_seconds'] = time.monotonic() - started
                gates.append(gate)
                matched_gates.append(gate)
                completed_gates.add(name)
                atomic(Path(run) / output_name, out)
                print(f'target-matched gate {name}: '
                      f'{"PASS" if gate["detected"] else "FAIL"} '
                      f'auc={gate.get("auc")} gain={gate.get("gain_over_prior")} '
                      f'elapsed={gate["elapsed_seconds"] / 60:.1f}m', flush=True)
    elif matched_required:
        matched_gates = [{
            'name': 'target_matched/transformer_required',
            'anchor': matched_endpoint, 'gate_role': 'target_matched_text',
            'detected': False, 'status': 'transformer_not_frozen'}]

    # Re-read from the persisted list when resuming after one matched gate.
    persisted_matched = [g for g in gates
                         if g.get('gate_role') == 'target_matched_text']
    if persisted_matched:
        matched_gates = persisted_matched
    matched_pass = (not matched_required or
                    (len(matched_gates) == 2
                     and all(g.get('detected') for g in matched_gates)))

    anchor_status = {}
    for endpoint_key in endpoint_rows:
        selected = [g for g in gates if g.get('anchor') == endpoint_key]
        positives = [g for g in selected
                     if g.get('gate_role') == 'deterministic_capacity']
        endpoint_alarm = any(g['detected'] for g in selected
                             if g['placement'] == 'null')
        passed = (not endpoint_occupancy[endpoint_key]['reasons'] and
                  all(g['detected'] for g in positives) and
                  not endpoint_alarm and not null_alarm and
                  (endpoint_key != matched_endpoint or matched_pass))
        anchor_status[endpoint_key] = (
            'controls_pass_at_deterministic_capacity' if passed
            else 'inconclusive_reader_sensitivity')
    positive_gates = [g for g in gates if g.get('gate_role') == 'deterministic_capacity']
    neural_null_alarm = any(g['detected'] for g in gates if g['placement'] == 'null')
    capacity_gates_pass = all(g['detected'] for g in positive_gates)
    null_alarm = null_alarm or neural_null_alarm
    transformer_gate = next((g for g in gates
                             if g.get('anchor') == primary_key and
                             g['name'].endswith('/transformer_text_capacity')), None)
    out.update(status=('controls_pass_at_deterministic_capacity'
                       if all(v == 'controls_pass_at_deterministic_capacity'
                              for v in anchor_status.values())
                       else 'inconclusive_reader_sensitivity'),
               complete=True, occupancy=occ,
               recoverability_population=policy.get(
                   'recoverability_population', 'all_cued'),
               endpoint_occupancy=endpoint_occupancy,
               anchor_status=anchor_status,
               transformer_text_control=transformer_gate,
               target_matched_controls={
                   'required': matched_required,
                   'endpoint': matched_endpoint,
                   'pass': matched_pass,
                   'gates': matched_gates},
               target_matched_controls_pass=matched_pass,
               capacity_gates_pass=capacity_gates_pass,
               broad_noisy_reference_pass=broad_noisy_reference_pass,
               broad_noisy_reference_role='diagnostic_non_vetoing',
               null_alarm=null_alarm,
               broad_reader_scope='non_neural_full_ladder',
               high_capacity_evaluation='fixed grouped five-fold OOF; no inner candidate search',
               scope=('Deterministic capacity controls test whether each frozen '
                      'expensive reader can recover an exactly planted signal. '
                      'For movie recommendation, completed-trace target-matched '
                      'answer--cue and explicit-attribution Transformer controls '
                      'veto the completed visibility-control endpoint, but are '
                      'not borrowed as validation of a pre-answer headline. The '
                      'noisy broad strength/geometry '
                      'surface is diagnostic and non-vetoing. Null alarms still '
                      'veto. Passing does not prove power for natural signals. '
                      'Low-variance failures remain visible.'))
    atomic(Path(run) / output_name, out)
    return out


def merge_control_shards(run, d, shard_count,
                         allow_environment_drift=False, *,
                         output_name='controls.json',
                         shard_prefix='controls_shard',
                         protocol_amendment=None):
    """Merge independent broad-grid shards, then run serial neural gates."""
    parents = ('traces.pkl', 'followups.pkl', 'capture.pkl')
    pieces = []
    runtimes = []
    for index in range(shard_count):
        name = f'{shard_prefix}_{index:02d}_of_{shard_count:02d}.json'
        part = completed(run, name, d, parents)
        if part.get('control_protocol') != CONTROL_PROTOCOL:
            raise ValueError(f'{name} has the wrong control protocol')
        pieces.extend(part['rows'])
        runtimes.append(part.get('runtime'))
    keys = [(r['placement'], r['strength'], r['repeat']) for r in pieces]
    if len(keys) != len(set(keys)):
        raise ValueError('Control shards overlap')
    expected = 3 * len(d['analysis']['control_strengths']) * d['analysis']['control_repeats']
    expected += d['analysis']['control_repeats']
    if len(keys) != expected:
        raise ValueError(f'Control shards contain {len(keys)} broad cells; expected {expected}')
    if any(runtime != runtimes[0] for runtime in runtimes[1:]):
        raise ValueError('Control shards used different runtimes')
    destination = Path(run) / output_name
    if destination.exists():
        existing = artifact(run, output_name, d, parents)
        if existing.get('rows') and existing.get('control_protocol') != CONTROL_PROTOCOL:
            raise ValueError('Existing controls.json is from the old protocol; preserve and move it before merging')
    else:
        merged = artifact(run, output_name, d, parents)
        merged.update(control_protocol=CONTROL_PROTOCOL,
                      rows=sorted(pieces, key=lambda r: (
                          ('text', 'high_variance', 'low_variance', 'null').index(r['placement']),
                          r['strength'], r['repeat'])),
                      merged_shards=shard_count,
                      shard_runtimes=runtimes, complete=False)
        if protocol_amendment:
            merged['protocol_amendment'] = protocol_amendment
        atomic(destination, merged)
    return controls(run, d, allow_environment_drift,
                    output_name=output_name,
                    protocol_amendment=protocol_amendment)
