"""Post-freeze E47 Transformer robustness analysis for legacy capture caches.

This never mutates the frozen design or primary artifacts. It reconstructs the
label-independent token-feature sequence from stored token IDs and the pinned
model's input embedding table, releases the language model, and writes a
content-addressed secondary result under the current analysis code.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from e47_analysis import (compare, datasets, environment_guard, interval,
                          occupancy, plant, recoverability_population)
from e47_capture import embedding_projector, load_captures
from e47_design import (atomic, code_hashes, completed, csv_write, digest,
                        file_hash, read, runtime_versions)
from e47_readers import fixed_evaluate


def default_config(seed):
    return {
        'architecture': 'e47_transformer_v3_lexical_skip',
        'input': 'target_model_input_embeddings_gaussian_projection',
        'input_dim': 64, 'projection_seed': int(seed + 4700),
        'd_model': 128, 'layers': 2, 'heads': 4, 'epochs': 60,
        'batch_size': 8, 'attention_budget': 24_000_000,
        'patience': 8, 'state_dim': 32, 'lexical_features': 4096}


def prepare_primary(run, d, model):
    """Build primary rows with projected model-token features on CPU."""
    import torch
    from tqdm.auto import tqdm

    data, exclusions, manifest = datasets(run, d)
    key = 'initial/' + d['analysis']['primary_anchor']
    rows = recoverability_population(data.get(key, []), d['analysis'])
    traces = completed(run, 'traces.pkl', d)
    generated = {r['id']: r for r in traces['rows']}
    captures, _ = load_captures(run, d, d['analysis']['layer'])
    config = default_config(d['analysis']['seed'])
    projector = embedding_projector(
        model, config['input_dim'], config['projection_seed'])
    embedding = model.get_input_embeddings()

    prepared = []
    for row in tqdm(rows, desc='E47 Transformer token features', unit='response',
                    dynamic_ncols=True):
        generation = generated[row['id']]
        shard = captures[row['id']]
        checkpoint = next((cp for cp in shard['rows']
                           if cp['anchor'] == d['analysis']['primary_anchor']), None)
        if checkpoint is None:
            raise ValueError(f"Missing primary checkpoint for {row['id']}")
        count = int(checkpoint['token_count'])
        ids = generation['prompt_ids'] + generation['ids'][:count]
        if len(ids) != int(checkpoint['absolute_position']) + 1:
            raise ValueError('Stored token IDs and checkpoint position disagree')
        token_ids = torch.tensor(ids, device=embedding.weight.device)
        with torch.inference_mode():
            sequence = (embedding(token_ids).float() @ projector).cpu().numpy()
        prepared.append({**row, 'sequence': sequence.astype(np.float32)})
    return prepared, exclusions, manifest, config


def score(run, d, rows, exclusions, manifest, config,
          allow_environment_drift=False, output=None):
    from sklearn.metrics import roc_auc_score

    drift = environment_guard(manifest, allow_environment_drift)
    policy = {**d['analysis'], 'transformer': True,
              'transformer_config': config}
    occ = occupancy(rows, policy)
    if occ['reasons']:
        summary, per_row = None, []
        status = 'inconclusive_occupancy'
        gate = None
    else:
        print('Transformer robustness: fitting natural text and joint ladders',
              flush=True)
        summary, per_row = compare(
            rows, policy, include_transformer=True)
        strength = policy.get('control_capacity_strength', 1.0)
        planted = plant(rows, strength, 'text', policy['seed'] + 7919)
        y = np.asarray([r['y'] for r in planted])
        groups = np.asarray([r['group'] for r in planted])
        print('Transformer robustness: fitting planted text-capacity gate',
              flush=True)
        fit = fixed_evaluate(
            planted, y, groups, policy, family='transformer', joint=False)
        gain = interval(fit['base_loss'] - fit['loss'], groups,
                        policy['seed'], policy['bootstrap'])
        auc = float(roc_auc_score(y, fit['P'][:, 1]))
        gate = {'placement': 'text', 'strength': strength,
                'gate_role': 'deterministic_capacity',
                'candidate_scope': 'fixed_text_transformer_grouped_calibrated',
                'evaluation_version': fit['evaluation_version'],
                'auc': auc, 'gain_over_prior': gain,
                'detected': bool(gain['ci95'][0] > 0
                                 and auc >= policy['control_min_auc']),
                'folds': fit['details']}
        if not gate['detected']:
            status = 'inconclusive_transformer_sensitivity'
        elif summary['joint_gain_over_prior']['ci95'][0] <= 0:
            status = 'inconclusive_target_readability'
        elif summary['gap']['ci95'][0] > 0:
            status = 'positive_bounded_reader_gain_secondary_robustness'
        else:
            status = 'no_resolved_gain_not_evidence_of_absence'
        summary['occupancy'] = occ
        summary['status'] = status

    parents = {name: file_hash(Path(run) / name) for name in
               ('design.json', 'traces.pkl', 'followups.pkl', 'capture.pkl')}
    environment = runtime_versions()
    signature_payload = {
        'schema': 'e47_transformer_robustness_v1',
        'design_id': d['design_id'], 'parents': parents,
        'transformer_config': config, 'current_code_sha256': code_hashes(),
        'environment': environment}
    signature = digest(signature_payload)
    destination = (Path(output) if output else
                   Path(run) / f'transformer_robustness_{signature[:12]}.json')
    if destination.exists():
        existing = read(destination)
        if existing.get('signature') != signature:
            raise ValueError(
                f'Refusing to overwrite a different robustness result: {destination}')
        print('Transformer robustness already exists:', destination)
        return existing
    result = {
        **signature_payload, 'signature': signature,
        'frozen_code_sha256': d.get('code_sha256', {}),
        'environment_drift': drift,
        'primary': 'initial/' + d['analysis']['primary_anchor'],
        'occupancy': occ, 'excluded': exclusions, 'transformer_control': gate,
        'analysis': summary, 'status': status,
        'scope': ('Post-freeze secondary robustness analysis under current code. '
                  'Generation, residual capture, frozen controls, and primary '
                  'analysis are unchanged. This is not retroactive confirmation.')}
    if drift:
        result['status'] = 'exploratory_environment_drift/' + status

    if per_row:
        csv_path = destination.with_suffix('.oof.csv')
        csv_write(csv_path, per_row)
        result['oof_csv'] = str(csv_path)
        result['oof_sha256'] = file_hash(csv_path)
    atomic(destination, result)
    print('Transformer robustness:', result['status'])
    print(destination)
    return result
