"""Checkpoint-only E47 capture and prespecified, held-out subspace patching."""
from __future__ import annotations

from pathlib import Path
import numpy as np

from e47_design import (answer_content_redaction, answer_parse_span, artifact,
                        atomic, completed, digest, file_hash, read,
                        scored_answer, target, verify_row)
from e46_layer_capture import checkpoint_states, transformer_blocks, layer_metadata
from e46_capture_residuals import final_norm


# Fixed token offsets BACK FROM the answer token, in addition to the fractional
# anchors. The fractional ones scale with trace length, so frac75 -> pre_answer
# is a ~52-token hole (medians over 1000 traces) in which rho goes from 0.001 to
# 0.659 with nothing measured inside it. Two readings fit that interval
# identically and they are the whole question:
#
#   RAMP   rho climbs across the 52 tokens -> the answer accrues while writing.
#   STEP   rho is flat, then jumps in the last 2 -> nothing accrues and the
#          residual is merely loading the token it is about to emit.
#
# Offsets from the answer, not further fractions, because the quantity of
# interest is distance to emission and that is what has to be held constant
# across traces of different lengths.
LEAD_OFFSETS = (40, 20, 10, 5)


def checkpoints(row, tok, why=False, lead_offsets=LEAD_OFFSETS):
    ids = row['ids']
    if tok.decode(ids, skip_special_tokens=True) != row['text']:
        raise ValueError('Stored token IDs do not reproduce generation text')
    ep = row['episode'].rstrip()
    prefixes = [tok.decode(ids[:i], skip_special_tokens=True) for i in range(len(ids) + 1)]
    ends = [i for i, s in enumerate(prefixes) if s.rstrip() == ep]
    # Earliest endpoint avoids sampling EOS/post-completion tokens as "final".
    end = min(ends) if ends and ep else None
    points = {'prompt': 0}
    if why:
        if end is not None:
            for frac in (25, 50, 75):
                points[f'frac{frac}'] = max(1, int(end * frac / 100))
            if not row['hit_cap']:
                points['complete'] = end
        return points, {
            'episode_end_resolved': end is not None,
            'why_temporal_anchors': [name for name in points
                                     if name.startswith('frac')]}
    if end is not None:
        for frac in (25, 50, 75):
            points[f'frac{frac}'] = max(1, int(end * frac / 100))
        if scored_answer(row) is not None and not row['hit_cap']:
            points['complete'] = end
    # Anchor immediately BEFORE the token carrying the last explicit answer
    # letter, not after it, nor before an earlier mention of the option title.
    answer_index = None
    matched_letter, _, parse_kind, answer_span = answer_parse_span(ep)
    if answer_span is not None and matched_letter == row.get('answer'):
        char = answer_span[0]
        for j in range(len(ids)):
            if len(prefixes[j]) <= char < len(prefixes[j + 1]):
                answer_index = j
                points['pre_answer'] = j
                break
    leads = {}
    if answer_index is not None:
        taken = set(points.values())
        for off in sorted(set(int(o) for o in lead_offsets), reverse=True):
            k = answer_index - off
            # k >= 1 keeps at least one generated token in the prefix, matching
            # the floor the fractional anchors already use. A duplicate count is
            # skipped rather than added: it would put the same position twice in
            # the positions list that drives checkpoint_states, costing a
            # forward slice for a column identical to one already present.
            if k >= 1 and k not in taken:
                leads[f'lead{off}'] = k
                taken.add(k)
        points.update(leads)
    return points, {'episode_end_resolved': end is not None,
                    'answer_token_index': answer_index,
                    'answer_parse_kind': parse_kind,
                    'answer_char_span': answer_span,
                    'lead_offsets_requested': sorted(
                        set(int(o) for o in lead_offsets), reverse=True),
                    'lead_anchors_placed': sorted(leads)}


def option_distribution(tok, model, full_ids, pos):
    """Actual output probabilities under a strict context-preserving encoding.

    Refuse multi-token/retokenized options. No max over spelling variants and
    no claim that a four-option renormalization is calibrated confidence.
    """
    import torch
    prefix_ids = full_ids[0, :pos + 1].tolist()
    prefix = tok.decode(prefix_ids, skip_special_tokens=False)
    option_ids = {}
    for c in 'ABCD':
        ids = tok.encode(prefix + c, add_special_tokens=False)
        if ids[:-1] != prefix_ids or len(ids) != len(prefix_ids) + 1:
            return {'status': 'unsupported_context_preserving_single_token_options'}
        option_ids[c] = ids[-1]
    if len(set(option_ids.values())) != 4:
        return {'status': 'nonunique_option_tokens'}
    with torch.inference_mode():
        logits = model(input_ids=full_ids[:, :pos + 1],
                       attention_mask=torch.ones_like(full_ids[:, :pos + 1]),
                       use_cache=False).logits[0, -1].float()
        probs = logits.softmax(0)[list(option_ids.values())].cpu().numpy()
    mass = float(probs.sum())
    return {'status': 'ok', 'option_token_ids': option_ids, 'valid_option_mass': mass,
            'probabilities': dict(zip('ABCD', map(float, probs))),
            'conditional': dict(zip('ABCD', map(float, probs / max(mass, 1e-30)))),
            'scope': 'actual next-token output at this prefix; not pre-cue confidence or causal influence'}


def answer_content_token_mask(row, tok, token_count, prompt_tokens, options):
    """Map generated-text redaction spans onto stored token-feature rows."""
    ids = row['ids'][:token_count]
    prefixes = [tok.decode(ids[:i], skip_special_tokens=True)
                for i in range(len(ids) + 1)]
    if any(len(prefixes[i]) > len(prefixes[i + 1])
           for i in range(len(prefixes) - 1)):
        raise ValueError('Tokenizer decoding is non-monotonic; cannot align redaction')
    _, audit = answer_content_redaction(prefixes[-1], options)
    mask = np.zeros(prompt_tokens + token_count, dtype=bool)
    for token_index in range(token_count):
        start, end = len(prefixes[token_index]), len(prefixes[token_index + 1])
        if any(start < span_end and end > span_start
               for span_start, span_end in audit['spans']):
            mask[prompt_tokens + token_index] = True
    return mask, audit


def capture(run, d, tok, model):
    import torch
    from tqdm.auto import tqdm
    traces = completed(run, 'traces.pkl', d)
    why = completed(run, 'followups.pkl', d, ('traces.pkl',))
    manifest = artifact(run, 'capture.pkl', d, ('traces.pkl', 'followups.pkl'))
    if manifest['complete']:
        return manifest
    norm = final_norm(model)
    info = layer_metadata(model, norm)
    if d['analysis']['layer'] > info['n_layers']:
        raise ValueError('Frozen layer exceeds number of model blocks')
    manifest['layer_metadata'] = info
    transformer_cfg = d['analysis'].get('transformer_config')
    projector = None
    if d['analysis'].get('transformer'):
        if not transformer_cfg:
            raise ValueError('Transformer enabled without frozen transformer_config')
        projector = embedding_projector(
            model, int(transformer_cfg['input_dim']),
            int(transformer_cfg['projection_seed']))
        manifest['token_feature_projection'] = {
            'method': 'gaussian_random_projection_of_input_embeddings',
            'input_dim': int(transformer_cfg['input_dim']),
            'seed': int(transformer_cfg['projection_seed'])}
        matched = d['analysis'].get('target_matched_controls') or {}
        insertion_specs = [matched.get('answer_cue_relation') or {},
                           matched.get('explicit_attribution') or {}]
        insertions = ([] if not matched.get('required') else [
            text for spec in insertion_specs
            for text in (list(spec.get('positive_insertions') or [])
                         + list(spec.get('negative_insertions') or []))])
        insertions = list(dict.fromkeys(insertions))
        if matched.get('required') and not insertions:
            raise ValueError('Frozen target-matched controls have no insertions')
        if insertions:
            bank = {}
            embedding = model.get_input_embeddings()
            device = embedding.weight.device
            with torch.inference_mode():
                for insertion in insertions:
                    ids = tok.encode(insertion, add_special_tokens=False)
                    if not ids:
                        raise ValueError('Target-matched control insertion tokenized empty')
                    token_ids = torch.tensor(ids, device=device)
                    features = (embedding(token_ids).float() @ projector).cpu()
                    bank[insertion] = {
                        'token_ids': list(map(int, ids)),
                        'token_features': features.numpy().astype(np.float16)}
            manifest['target_matched_control_token_features'] = bank
    items = {int(item['index']): item for item in d['items']}
    done = {r['id'] for r in manifest['rows']}
    for entry in manifest['rows']:
        if file_hash(Path(run) / entry['file']) != entry['sha256']:
            raise ValueError('Changed capture shard')
    total = len(traces['rows']) + len(why['rows'])
    with tqdm(total=total, initial=len(done), desc='E47 capture', unit='response',
              dynamic_ncols=True, leave=True) as bar:
        for stage, blob in (('initial', traces), ('why', why)):
            for row in blob['rows']:
                if row['id'] in done:
                    continue
                verify_row(row)
                if tok(row['prompt'], add_special_tokens=row.get('prompt_add_special_tokens', True))['input_ids'] != row['prompt_ids']:
                    raise ValueError('Tokenizer prompt encoding differs from generation')
                points, audit = checkpoints(row, tok, stage == 'why')
                n = len(row['prompt_ids'])
                positions = [n + count - 1 for count in points.values()]
                device = model.get_input_embeddings().weight.device
                full = torch.tensor([row['prompt_ids'] + row['ids']], device=device)
                _, layers = checkpoint_states(model, full[:, :max(positions) + 1], positions,
                                              all_layers=True, norm=norm)
                token_features = None
                if projector is not None:
                    with torch.inference_mode():
                        token_features = (
                            model.get_input_embeddings()(full[:, :max(positions) + 1])
                            .squeeze(0).float() @ projector).cpu().numpy().astype(np.float16)
                rows = []
                for anchor, count in points.items():
                    prefix = tok.decode(row['ids'][:count], skip_special_tokens=True)
                    rows.append({'id': row['id'], 'stage': stage, 'anchor': anchor,
                                 'prefix': prefix, 'token_count': count,
                                 'absolute_position': n + count - 1})
                probs = option_distribution(tok, model, full, n + points['pre_answer'] - 1) if 'pre_answer' in points else {'status': 'no_validated_pre_answer_anchor'}
                payload = {'design_id': d['design_id'], 'generation_sha256': row['content_sha256'],
                           'rows': rows, 'R_layers': layers, 'audit': audit,
                           'decision_distribution': probs}
                if token_features is not None:
                    payload['token_features'] = token_features
                    payload['token_feature_projection'] = manifest['token_feature_projection']
                    mask, redaction_audit = answer_content_token_mask(
                        row, tok, token_features.shape[0] - n, n,
                        items[int(row['index'])].get('options') or {})
                    payload['answer_content_mask'] = mask
                    payload['answer_content_redaction_audit'] = redaction_audit
                name = 'capture/' + row['id'] + '.pkl'
                atomic(Path(run) / name, payload, True)
                manifest['rows'].append({'id': row['id'], 'stage': stage, 'file': name,
                                         'sha256': file_hash(Path(run) / name)})
                done.add(row['id'])
                atomic(Path(run) / 'capture.pkl', manifest, True)
                bar.update(1)
                bar.set_postfix({'shards': len(manifest['rows'])}, refresh=False)
    manifest['complete'] = True
    atomic(Path(run) / 'capture.pkl', manifest, True)
    return manifest


def load_captures(run, d, layer=None, ids=None, layers=None):
    if layer is not None and layers is not None:
        raise ValueError('Request either one layer or a layer set, not both')
    manifest = completed(run, 'capture.pkl', d, ('traces.pkl', 'followups.pkl'))
    wanted = None if ids is None else set(ids)
    out = {}
    for entry in manifest['rows']:
        if wanted is not None and entry['id'] not in wanted:
            continue
        path = Path(run) / entry['file']
        if file_hash(path) != entry['sha256']:
            raise ValueError(f'Changed capture shard: {path}')
        shard = read(path)
        if shard['design_id'] != d['design_id'] or entry['id'] in out:
            raise ValueError('Invalid/duplicate capture shard')
        if layer is not None:
            shard['state'] = np.array(shard.pop('R_layers')[:, layer, :], copy=True)
        elif layers is not None:
            stack = shard.pop('R_layers')
            invalid = [index for index in layers
                       if index < 0 or index >= stack.shape[1]]
            if invalid:
                raise ValueError(f'Requested layer indices outside capture stack: {invalid}')
            shard['states_by_layer'] = {
                int(index): np.array(stack[:, index, :], copy=True)
                for index in layers}
        out[entry['id']] = shard
    if wanted is not None and set(out) != wanted:
        missing = sorted(wanted - set(out))
        raise ValueError(f'Missing requested capture shards: {missing[:3]}')
    return out, manifest


def embedding_projector(model, width, seed):
    """Frozen label-independent projection of the target model's token table."""
    import torch
    hidden = int(model.get_input_embeddings().weight.shape[1])
    rng = np.random.default_rng(seed)
    matrix = rng.normal(size=(hidden, width)).astype(np.float32)
    matrix /= np.sqrt(width)
    return torch.from_numpy(matrix).to(
        model.get_input_embeddings().weight.device)


def patch_vector(model, ids, pos, layer, delta, option_ids):
    import torch
    block = transformer_blocks(model)[layer - 1]
    def hook(module, args, output):
        h = output[0] if isinstance(output, tuple) else output
        changed = h.clone()
        changed[0, pos] += torch.as_tensor(delta, dtype=h.dtype, device=h.device)
        return (changed,) + output[1:] if isinstance(output, tuple) else changed
    handle = block.register_forward_hook(hook)
    try:
        with torch.inference_mode():
            logits = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False).logits[0, -1].float()
            p = logits.softmax(0)[option_ids].cpu().numpy()
        return p
    finally:
        handle.remove()


def patch_vectors(model, ids, pos, layer, deltas, option_ids, batch_size=6):
    """Evaluate several same-prefix patches in bounded batches."""
    import torch
    block = transformer_blocks(model)[layer - 1]
    outputs = []
    for start in range(0, len(deltas), batch_size):
        chunk = np.asarray(deltas[start:start + batch_size])
        repeated = ids.repeat(len(chunk), 1)

        def hook(module, args, output):
            hidden = output[0] if isinstance(output, tuple) else output
            changed = hidden.clone()
            changed[:, pos] += torch.as_tensor(
                chunk, dtype=hidden.dtype, device=hidden.device)
            return ((changed,) + output[1:]
                    if isinstance(output, tuple) else changed)

        handle = block.register_forward_hook(hook)
        try:
            with torch.inference_mode():
                logits = model(
                    input_ids=repeated,
                    attention_mask=torch.ones_like(repeated),
                    use_cache=False).logits[:, -1].float()
                outputs.extend(logits.softmax(1)[:, option_ids].cpu().numpy())
        finally:
            handle.remove()
    return outputs


def causal_interchange_v1(run, d, tok, model):
    """Held-out target-subspace interventions for the E47 causal claim.

    The target is cue following among disagreement trials only.  Every fold
    learns its direction and reference scores without the held-out source
    items.  Counterfactual and erasure interventions are compared with a
    same-target donor, orthogonal and isotropic norm-matched perturbations,
    and a sham replay at the same layer and position.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler
    from e46_readers import splits
    from e47_readers import interval

    cfg = d.get('analysis', {}).get('causal_interchange')
    if not cfg or cfg.get('version') != 'e47_causal_interchange_v1':
        raise ValueError('This design does not freeze causal_interchange_v1; '
                         'freeze a new E47-v3 run')
    traces = completed(run, 'traces.pkl', d)
    captures, _ = load_captures(run, d, int(cfg['layer']))
    parents = ('traces.pkl', 'followups.pkl', 'capture.pkl')
    result = artifact(run, 'causal_interchange.json', d, parents)
    if result['complete']:
        return result

    baselines = {row['index']: row for row in traces['rows']
                 if row['condition'] == 'uncued'}

    def state(row):
        shard = captures[row['id']]
        matches = [i for i, cp in enumerate(shard['rows'])
                   if cp['anchor'] == cfg['anchor']]
        return None if len(matches) != 1 else np.asarray(
            shard['state'][matches[0]], dtype=float)

    eligible = []
    for row in traces['rows']:
        if not row['condition'].startswith('disagree_') or row['hit_cap']:
            continue
        baseline = baselines.get(row['index'])
        z = target(baseline, row) if baseline is not None else None
        vector = state(row)
        if z is not None and vector is not None:
            eligible.append((row, int(z), vector))
    if not eligible:
        result.update(status='no_eligible_disagreement_rows', complete=True,
                      scope='No intervention was run.')
        atomic(Path(run) / 'causal_interchange.json', result)
        return result

    X = np.stack([entry[2] for entry in eligible])
    y = np.asarray([entry[1] for entry in eligible], dtype=int)
    groups = np.asarray([entry[0]['source_id'] for entry in eligible])
    class_items = {str(k): len(set(groups[y == k])) for k in (0, 1)}
    if (min(class_items.values()) < d['analysis']['min_class_items']
            or len(set(groups)) < d['analysis']['min_items']):
        result.update(status='inconclusive_class_occupancy',
                      target=cfg['target'], class_items=class_items,
                      eligible_rows=len(eligible), complete=True)
        atomic(Path(run) / 'causal_interchange.json', result)
        return result

    source_order = sorted(set(groups), key=lambda value: digest(
        [d['seed'], 'causal_interchange', value]))
    limit = int(cfg.get('max_source_items', 0))
    selected = set(source_order if limit == 0 else source_order[:limit])
    predictions = np.full(len(y), np.nan)
    skipped = {}
    done = {row['id'] for row in result['rows']}
    from tqdm.auto import tqdm
    selected_rows = sum(group in selected for group in groups)
    progress = tqdm(total=selected_rows, initial=len(done),
                    desc='E47 causal interchange', unit='response',
                    dynamic_ncols=True, leave=True)

    def unit_random(seed, dimension):
        rng = np.random.default_rng(seed)
        value = rng.normal(size=dimension)
        return value / max(np.linalg.norm(value), 1e-30)

    for fold, (train, test) in enumerate(
            splits(groups, int(cfg['folds']), d['seed'])):
        if len(np.unique(y[train])) != 2:
            continue
        scaler = StandardScaler().fit(X[train])
        clf = LogisticRegression(
            C=0.1, max_iter=1500, solver='liblinear',
            random_state=d['seed']).fit(scaler.transform(X[train]), y[train])
        predictions[test] = clf.predict_proba(scaler.transform(X[test]))[:, 1]
        direction = clf.coef_[0] / np.maximum(scaler.scale_, 1e-30)
        direction /= max(np.linalg.norm(direction), 1e-30)
        scores = X[train] @ direction
        class_scores = {k: scores[y[train] == k] for k in (0, 1)}
        class_indices = {k: train[y[train] == k] for k in (0, 1)}
        references = {k: float(class_scores[k].mean()) for k in (0, 1)}
        pooled_reference = float(scores.mean())
        result.setdefault('folds', {})[str(fold)] = {
            'train_sources': sorted(set(groups[train])),
            'test_sources': sorted(set(groups[test])),
            'class_reference_scores': references,
            'pooled_reference_score': pooled_reference,
            'direction_norm': float(np.linalg.norm(direction))}

        for i in test:
            row, label, vector = eligible[i]
            if row['source_id'] not in selected:
                continue
            if row['id'] in done:
                continue
            shard = captures[row['id']]
            distribution = shard['decision_distribution']
            if distribution.get('status') != 'ok':
                reason = distribution.get('status', 'invalid_distribution')
                skipped[reason] = skipped.get(reason, 0) + 1
                progress.update(1)
                continue
            cp = next(cp for cp in shard['rows']
                      if cp['anchor'] == cfg['anchor'])
            import torch
            ids = torch.tensor(
                [row['prompt_ids'] + row['ids'][:cp['token_count']]],
                device=model.get_input_embeddings().weight.device)
            pos = ids.shape[1] - 1
            option_ids = [distribution['option_token_ids'][letter]
                          for letter in 'ABCD']
            baseline_p = patch_vector(
                model, ids, pos, int(cfg['layer']),
                np.zeros_like(direction), option_ids)
            captured_p = np.asarray(
                [distribution['probabilities'][letter] for letter in 'ABCD'])
            if not np.allclose(baseline_p, captured_p, atol=.003, rtol=.03):
                raise ValueError('Causal replay differs from captured distribution')

            score = float(vector @ direction)
            counterfactual_label = 1 - label
            counterfactual = direction * (
                references[counterfactual_label] - score)
            erasure = direction * (pooled_reference - score)
            pool = class_indices[label]
            donor_offset = int(digest(
                [row['id'], fold, 'same_target'])[:16], 16) % len(pool)
            same_score = float(X[pool[donor_offset]] @ direction)
            same_target = direction * (same_score - score)
            seed = int(digest([row['id'], fold, 'causal_controls'])[:16], 16)
            random_direction = unit_random(seed, direction.size)
            orthogonal = random_direction - direction * np.dot(
                random_direction, direction)
            orthogonal /= max(np.linalg.norm(orthogonal), 1e-30)
            norm = float(np.linalg.norm(counterfactual))
            interventions = {
                'counterfactual': counterfactual,
                'erasure': erasure,
                'same_target': same_target,
                'orthogonal_norm_matched': orthogonal * norm,
                'random_norm_matched': random_direction * norm,
                'sham': np.zeros_like(direction)}

            cue_index = 'ABCD'.index(row['cue_letter'])
            def cue_log_odds(probability):
                cue = max(float(probability[cue_index]), 1e-30)
                other = max(float(np.delete(probability, cue_index).sum()),
                            1e-30)
                return float(np.log(cue) - np.log(other))

            baseline_odds = cue_log_odds(baseline_p)
            sign = 1.0 if counterfactual_label == 1 else -1.0
            requests = []
            for alpha in cfg['alphas']:
                for control, delta in interventions.items():
                    requests.append((float(alpha), control,
                                     float(alpha) * delta, delta))
            probabilities = patch_vectors(
                model, ids, pos, int(cfg['layer']),
                [request[2] for request in requests], option_ids,
                batch_size=int(cfg['batch_size']))
            values = []
            for (alpha, control, applied, original), probability in zip(
                    requests, probabilities):
                raw_odds = cue_log_odds(probability) - baseline_odds
                raw_probability = float(
                    probability[cue_index] - baseline_p[cue_index])
                values.append({
                    'alpha': float(alpha), 'control': control,
                    'cue_log_odds_delta': raw_odds,
                    'cue_probability_delta': raw_probability,
                    'signed_cue_log_odds_delta': sign * raw_odds,
                    'signed_cue_probability_delta': sign * raw_probability,
                    'option_argmax': 'ABCD'[int(np.argmax(probability))],
                    'probabilities': probability.tolist(),
                    'delta_norm': float(np.linalg.norm(original)),
                    'applied_delta_norm': float(np.linalg.norm(applied))})
            result['rows'].append({
                'id': row['id'], 'source_id': row['source_id'],
                'condition': row['condition'], 'fold': fold,
                'target': label,
                'counterfactual_target': counterfactual_label,
                'cue_letter': row['cue_letter'],
                'baseline_answer': scored_answer(row),
                'baseline_option_argmax': 'ABCD'[int(np.argmax(baseline_p))],
                'baseline_probabilities': baseline_p.tolist(),
                'state_score': score, 'interventions': values})
            atomic(Path(run) / 'causal_interchange.json', result)
            progress.update(1)
            progress.set_postfix({'saved': len(result['rows'])}, refresh=False)

    progress.close()

    auc = (float(roc_auc_score(y, predictions))
           if np.isfinite(predictions).all() else None)
    summaries = {}
    contrasts = {}
    for alpha in cfg['alphas']:
        for control in cfg['interventions']:
            values, source_ids = [], []
            for row in result['rows']:
                value = next(v for v in row['interventions']
                             if v['alpha'] == float(alpha)
                             and v['control'] == control)
                values.append(value['signed_cue_log_odds_delta'])
                source_ids.append(row['source_id'])
            if values:
                summaries[f'{alpha}/{control}'] = interval(
                    values, source_ids, d['seed'], d['analysis']['bootstrap'])
        for reference in ('same_target', 'orthogonal_norm_matched'):
            values, source_ids = [], []
            for row in result['rows']:
                by_control = {
                    value['control']: value
                    for value in row['interventions']
                    if value['alpha'] == float(alpha)}
                values.append(
                    by_control['counterfactual']['signed_cue_log_odds_delta']
                    - by_control[reference]['signed_cue_log_odds_delta'])
                source_ids.append(row['source_id'])
            if values:
                contrasts[f'{alpha}/counterfactual_minus_{reference}'] = interval(
                    values, source_ids, d['seed'], d['analysis']['bootstrap'])
    class_effects = {}
    primary_alpha = float(cfg['primary_alpha'])
    for label, name in ((0, 'rescue_nonfollowers'), (1, 'reverse_followers')):
        values, source_ids = [], []
        for row in result['rows']:
            if row['target'] != label:
                continue
            value = next(v for v in row['interventions']
                         if v['alpha'] == primary_alpha
                         and v['control'] == 'counterfactual')
            values.append(value['signed_cue_log_odds_delta'])
            source_ids.append(row['source_id'])
        if values:
            class_effects[name] = interval(
                values, source_ids, d['seed'], d['analysis']['bootstrap'])

    primary = {
        name: contrasts.get(f'{primary_alpha}/{name}')
        for name in cfg['primary_contrasts']}
    readable = auc is not None and auc >= d['analysis']['control_min_auc']
    controlled = (all(value and value['ci95'][0] > 0
                      for value in primary.values())
                  and all(value['ci95'][0] > 0
                          for value in class_effects.values())
                  and len(class_effects) == 2)
    status = ('inconclusive_direction_readability' if not readable else
              'positive_controlled_local_causal_sensitivity' if controlled else
              'no_resolved_controlled_causal_effect')
    result.update(
        complete=True, status=status, target=cfg['target'],
        checkpoint=f"initial/{cfg['anchor']}", layer=int(cfg['layer']),
        direction_oof_auc=auc, class_items=class_items,
        summaries=summaries, contrasts=contrasts,
        class_effects=class_effects,
        primary_contrasts=primary,
        coverage={'eligible_rows': len(eligible),
                  'eligible_source_items': len(set(groups)),
                  'selected_source_items': len(selected),
                  'intervened_rows': len(result['rows']),
                  'skipped': skipped},
        scope=('Held-out local pre-answer target-subspace intervention. A '
               'positive status establishes controlled causal sensitivity at '
               'this site, not whole-trajectory mediation or awareness. It '
               'supports a causal verbalization-gap claim only when the same '
               'target/checkpoint also has a positive conditional text-to-joint gap.'))
    atomic(Path(run) / 'causal_interchange.json', result)
    return result


def causal_interchange_v2(run, d, tok, model):
    """Relation-matched interchange of within-item cue-induced state changes.

    A cue-following label is relational: following can mean A->B, C->A, and
    so on.  V1 pooled those geometrically different transitions into one
    predictive direction.  V2 instead matches held-out recipients to training
    donors with the same ordered abandoned->cue answer relation.  It replaces
    the recipient's within-item cued-minus-uncued delta with an opposite-class
    donor delta.  Rescue and reversal therefore use different donor maps and
    are never assumed to be negatives of one global direction.
    """
    from e46_readers import splits
    from e47_readers import interval

    cfg = d.get('analysis', {}).get('causal_interchange')
    if not cfg or cfg.get('version') != 'e47_causal_interchange_v2':
        raise ValueError('This design does not freeze causal_interchange_v2')
    traces = completed(run, 'traces.pkl', d)
    captures, _ = load_captures(run, d, int(cfg['layer']))
    parents = ('traces.pkl', 'followups.pkl', 'capture.pkl')
    result = artifact(run, 'causal_interchange.json', d, parents)
    if result['complete']:
        return result

    baselines = {row['index']: row for row in traces['rows']
                 if row['condition'] == 'uncued'}

    def state(row):
        if row is None:
            return None
        shard = captures.get(row['id'])
        if shard is None:
            return None
        matches = [i for i, cp in enumerate(shard['rows'])
                   if cp['anchor'] == cfg['anchor']]
        return None if len(matches) != 1 else np.asarray(
            shard['state'][matches[0]], dtype=float)

    def distribution(row):
        if row is None:
            return None
        shard = captures.get(row['id'])
        value = {} if shard is None else shard.get('decision_distribution', {})
        return value if value.get('status') == 'ok' else None

    def relation_margin(dist, cue, abandoned):
        values = dist['probabilities']
        return float(np.log(max(float(values[cue]), 1e-30))
                     - np.log(max(float(values[abandoned]), 1e-30)))

    eligible = []
    excluded = {}
    for row in traces['rows']:
        if not row['condition'].startswith('disagree_') or row['hit_cap']:
            continue
        baseline = baselines.get(row['index'])
        abandoned = scored_answer(baseline)
        cue = row.get('cue_letter')
        label = target(baseline, row) if baseline is not None else None
        cued_state, uncued_state = state(row), state(baseline)
        cued_dist, uncued_dist = distribution(row), distribution(baseline)
        reason = None
        if (label is None or abandoned not in tuple('ABCD')
                or cue not in tuple('ABCD')):
            reason = 'invalid_behavioral_pair'
        elif abandoned == cue:
            reason = 'not_a_disagreement_relation'
        elif cued_state is None or uncued_state is None:
            reason = 'missing_paired_pre_answer_state'
        elif cued_dist is None or uncued_dist is None:
            reason = 'missing_paired_decision_distribution'
        if reason:
            excluded[reason] = excluded.get(reason, 0) + 1
            continue
        eligible.append({
            'row': row, 'baseline': baseline, 'label': int(label),
            'state': cued_state, 'uncued_state': uncued_state,
            'cue_delta': cued_state - uncued_state,
            'cue': cue, 'abandoned': abandoned,
            'cue_type': row.get('cue_type', 'authority'),
            'relation': (f'{row.get("cue_type", "authority")}:'
                         f'{abandoned}->{cue}'),
            'uncued_margin': relation_margin(
                uncued_dist, cue, abandoned),
            'cued_distribution': cued_dist,
        })

    if not eligible:
        result.update(status='no_eligible_disagreement_rows', complete=True,
                      exclusions=excluded, scope='No intervention was run.')
        atomic(Path(run) / 'causal_interchange.json', result)
        return result

    y = np.asarray([entry['label'] for entry in eligible], dtype=int)
    groups = np.asarray([entry['row']['source_id'] for entry in eligible])
    class_items = {str(k): len(set(groups[y == k])) for k in (0, 1)}
    relation_items = {}
    for entry in eligible:
        cell = relation_items.setdefault(entry['relation'], {'0': 0, '1': 0})
        cell[str(entry['label'])] += 1
    if (min(class_items.values()) < d['analysis']['min_class_items']
            or len(set(groups)) < d['analysis']['min_items']):
        result.update(status='inconclusive_class_occupancy',
                      target=cfg['target'], class_items=class_items,
                      relation_items=relation_items,
                      eligible_rows=len(eligible), exclusions=excluded,
                      complete=True)
        atomic(Path(run) / 'causal_interchange.json', result)
        return result

    source_order = sorted(set(groups), key=lambda value: digest(
        [d['seed'], 'causal_interchange_v2', value]))
    limit = int(cfg.get('max_source_items', 0))
    selected = set(source_order if limit == 0 else source_order[:limit])
    done = {row['id'] for row in result['rows']}
    skipped = {}
    minimum = int(cfg.get('minimum_exact_relation_donors_per_class', 1))

    def choose(candidates, recipient, purpose, fold):
        if len(candidates) < minimum:
            return None
        # Matching on the uncued cue-vs-abandoned margin avoids selecting an
        # opposite-class donor merely because its prior preference was easier.
        return min(candidates, key=lambda candidate: (
            abs(candidate['uncued_margin'] - recipient['uncued_margin']),
            digest([recipient['row']['id'], candidate['row']['id'],
                    purpose, fold])))

    from tqdm.auto import tqdm
    selected_rows = sum(group in selected for group in groups)
    progress = tqdm(total=selected_rows, initial=len(done),
                    desc='E47 causal interchange v2', unit='response',
                    dynamic_ncols=True, leave=True)

    for fold, (train, test) in enumerate(
            splits(groups, int(cfg['folds']), d['seed'])):
        train_entries = [eligible[i] for i in train]
        for i in test:
            recipient = eligible[i]
            row = recipient['row']
            if row['source_id'] not in selected or row['id'] in done:
                continue
            opposite = [entry for entry in train_entries
                        if entry['relation'] == recipient['relation']
                        and entry['label'] == 1 - recipient['label']]
            same = [entry for entry in train_entries
                    if entry['relation'] == recipient['relation']
                    and entry['label'] == recipient['label']
                    and entry['row']['source_id'] != row['source_id']]
            relation_train = [entry for entry in train_entries
                              if entry['relation'] == recipient['relation']]
            by_true_label = {
                label: sorted(
                    [entry for entry in relation_train
                     if entry['label'] == label],
                    key=lambda entry: digest([
                        d['seed'], fold, recipient['relation'],
                        'balanced_shuffle_pool', entry['row']['id']]))
                for label in (0, 1)}
            balanced_n = min(map(len, by_true_label.values()))
            balanced_relation_train = (
                by_true_label[0][:balanced_n]
                + by_true_label[1][:balanced_n])
            shuffled_labels = [entry['label']
                               for entry in balanced_relation_train]
            shuffle_rng = np.random.default_rng(int(digest([
                d['seed'], fold, recipient['relation'],
                'label_shuffled_donors'])[:16], 16))
            shuffle_rng.shuffle(shuffled_labels)
            shuffled = [entry for entry, pseudo_label in zip(
                balanced_relation_train, shuffled_labels)
                        if pseudo_label == 1 - recipient['label']]
            counterfactual_donor = choose(
                opposite, recipient, 'counterfactual', fold)
            same_target_donor = choose(same, recipient, 'same_target', fold)
            shuffled_donor = choose(
                shuffled, recipient, 'label_shuffled', fold)
            if counterfactual_donor is None:
                reason = 'no_exact_relation_opposite_class_donor'
                skipped[reason] = skipped.get(reason, 0) + 1
                progress.update(1)
                continue
            if same_target_donor is None:
                reason = 'no_exact_relation_same_class_donor'
                skipped[reason] = skipped.get(reason, 0) + 1
                progress.update(1)
                continue
            if shuffled_donor is None:
                reason = 'no_exact_relation_label_shuffled_donor'
                skipped[reason] = skipped.get(reason, 0) + 1
                progress.update(1)
                continue

            shard = captures[row['id']]
            cp = next(cp for cp in shard['rows']
                      if cp['anchor'] == cfg['anchor'])
            import torch
            ids = torch.tensor(
                [row['prompt_ids'] + row['ids'][:cp['token_count']]],
                device=model.get_input_embeddings().weight.device)
            pos = ids.shape[1] - 1
            dist = recipient['cued_distribution']
            option_ids = [dist['option_token_ids'][letter]
                          for letter in 'ABCD']
            baseline_p = patch_vector(
                model, ids, pos, int(cfg['layer']),
                np.zeros_like(recipient['state']), option_ids)
            captured_p = np.asarray(
                [dist['probabilities'][letter] for letter in 'ABCD'])
            if not np.allclose(baseline_p, captured_p, atol=.003, rtol=.03):
                raise ValueError(
                    'Causal replay differs from captured distribution')

            paired_delta = (counterfactual_donor['cue_delta']
                            - recipient['cue_delta'])
            full_state = counterfactual_donor['state'] - recipient['state']
            same_delta = (same_target_donor['cue_delta']
                          - recipient['cue_delta'])
            shuffled_delta = (shuffled_donor['cue_delta']
                              - recipient['cue_delta'])
            seed = int(digest(
                [row['id'], fold, 'causal_interchange_v2_controls'])[:16], 16)
            rng = np.random.default_rng(seed)
            random_direction = rng.normal(size=paired_delta.size)
            random_direction /= max(np.linalg.norm(random_direction), 1e-30)
            unit = paired_delta / max(np.linalg.norm(paired_delta), 1e-30)
            orthogonal = random_direction - unit * float(random_direction @ unit)
            orthogonal /= max(np.linalg.norm(orthogonal), 1e-30)
            norm = float(np.linalg.norm(paired_delta))
            interventions = {
                'paired_delta_counterfactual': paired_delta,
                'full_state_counterfactual': full_state,
                'same_target_paired_delta': same_delta,
                'label_shuffled_paired_delta': shuffled_delta,
                'orthogonal_norm_matched': orthogonal * norm,
                'random_norm_matched': random_direction * norm,
                'sham': np.zeros_like(paired_delta),
            }

            cue_index = 'ABCD'.index(recipient['cue'])
            abandoned_index = 'ABCD'.index(recipient['abandoned'])

            def margin(probability):
                return float(
                    np.log(max(float(probability[cue_index]), 1e-30))
                    - np.log(max(float(probability[abandoned_index]), 1e-30)))

            baseline_margin = margin(baseline_p)
            sign = 1.0 if recipient['label'] == 0 else -1.0
            requests = []
            for alpha in cfg['alphas']:
                for control, delta in interventions.items():
                    requests.append((float(alpha), control,
                                     float(alpha) * delta, delta))
            probabilities = patch_vectors(
                model, ids, pos, int(cfg['layer']),
                [request[2] for request in requests], option_ids,
                batch_size=int(cfg['batch_size']))
            values = []
            counterfactual_letter = (
                recipient['cue'] if recipient['label'] == 0
                else recipient['abandoned'])
            for (alpha, control, applied, original), probability in zip(
                    requests, probabilities):
                raw_margin = margin(probability) - baseline_margin
                option_argmax = 'ABCD'[int(np.argmax(probability))]
                values.append({
                    'alpha': alpha, 'control': control,
                    'cue_vs_abandoned_log_odds_delta': raw_margin,
                    'signed_cue_vs_abandoned_log_odds_delta': sign * raw_margin,
                    # Compatibility aliases for report-only tools written for v1.
                    'cue_log_odds_delta': raw_margin,
                    'signed_cue_log_odds_delta': sign * raw_margin,
                    'cue_probability_delta': float(
                        probability[cue_index] - baseline_p[cue_index]),
                    'signed_cue_probability_delta': sign * float(
                        probability[cue_index] - baseline_p[cue_index]),
                    'option_argmax': option_argmax,
                    'counterfactual_argmax': option_argmax == counterfactual_letter,
                    'changed_to_counterfactual_argmax': (
                        option_argmax == counterfactual_letter
                        and 'ABCD'[int(np.argmax(baseline_p))]
                        != counterfactual_letter),
                    'probabilities': probability.tolist(),
                    'delta_norm': float(np.linalg.norm(original)),
                    'applied_delta_norm': float(np.linalg.norm(applied)),
                })
            result['rows'].append({
                'id': row['id'], 'source_id': row['source_id'],
                'condition': row['condition'], 'fold': fold,
                'target': recipient['label'],
                'counterfactual_target': 1 - recipient['label'],
                'cue_letter': recipient['cue'],
                'abandoned_letter': recipient['abandoned'],
                'relation': recipient['relation'],
                'baseline_answer': scored_answer(row),
                'baseline_option_argmax': 'ABCD'[int(np.argmax(baseline_p))],
                'baseline_probabilities': baseline_p.tolist(),
                'uncued_cue_vs_abandoned_log_odds': recipient['uncued_margin'],
                'counterfactual_donor_id': counterfactual_donor['row']['id'],
                'counterfactual_donor_target': counterfactual_donor['label'],
                'same_target_donor_id': same_target_donor['row']['id'],
                'same_target_donor_target': same_target_donor['label'],
                'label_shuffled_donor_id': shuffled_donor['row']['id'],
                'label_shuffled_donor_true_target': shuffled_donor['label'],
                'interventions': values,
            })
            atomic(Path(run) / 'causal_interchange.json', result)
            done.add(row['id'])
            progress.update(1)
            progress.set_postfix({'saved': len(result['rows'])}, refresh=False)
    progress.close()

    summaries = {}
    contrasts = {}
    outcome = 'signed_cue_vs_abandoned_log_odds_delta'
    for alpha in cfg['alphas']:
        for control in cfg['interventions']:
            values, source_ids = [], []
            for row in result['rows']:
                value = next(v for v in row['interventions']
                             if v['alpha'] == float(alpha)
                             and v['control'] == control)
                values.append(value[outcome])
                source_ids.append(row['source_id'])
            if values:
                summaries[f'{alpha}/{control}'] = interval(
                    values, source_ids, d['seed'], d['analysis']['bootstrap'])
        for reference in ('same_target_paired_delta',
                          'label_shuffled_paired_delta',
                          'orthogonal_norm_matched'):
            values, source_ids = [], []
            for row in result['rows']:
                by_control = {value['control']: value
                              for value in row['interventions']
                              if value['alpha'] == float(alpha)}
                values.append(
                    by_control['paired_delta_counterfactual'][outcome]
                    - by_control[reference][outcome])
                source_ids.append(row['source_id'])
            if values:
                key = (f'{alpha}/paired_delta_counterfactual_minus_'
                       f'{reference}')
                contrasts[key] = interval(
                    values, source_ids, d['seed'], d['analysis']['bootstrap'])

    primary_alpha = float(cfg['primary_alpha'])
    class_effects = {}
    argmax_effects = {}
    for label, name in ((0, 'rescue_nonfollowers'), (1, 'reverse_followers')):
        values, source_ids, argmax, changed = [], [], [], []
        for row in result['rows']:
            if row['target'] != label:
                continue
            value = next(v for v in row['interventions']
                         if v['alpha'] == primary_alpha
                         and v['control'] == 'paired_delta_counterfactual')
            values.append(value[outcome])
            source_ids.append(row['source_id'])
            argmax.append(int(value['counterfactual_argmax']))
            changed.append(int(value['changed_to_counterfactual_argmax']))
        if values:
            class_effects[name] = interval(
                values, source_ids, d['seed'], d['analysis']['bootstrap'])
            argmax_effects[name] = {
                'n_items': len(values),
                'counterfactual_argmax': int(sum(argmax)),
                'counterfactual_argmax_rate': float(np.mean(argmax)),
                'changed_to_counterfactual_argmax': int(sum(changed)),
                'changed_to_counterfactual_argmax_rate': float(np.mean(changed)),
            }

    primary = {
        name: contrasts.get(f'{primary_alpha}/{name}')
        for name in cfg['primary_contrasts']}
    matched_class_items = {
        str(label): sum(row['target'] == label for row in result['rows'])
        for label in (0, 1)}
    matched_minimum = int(cfg.get(
        'minimum_intervened_items_per_class',
        d['analysis']['min_class_items']))
    matched_occupancy = min(matched_class_items.values()) >= matched_minimum
    class_resolved = (matched_occupancy and len(class_effects) == 2
                      and all(value['ci95'][0] > 0
                              for value in class_effects.values()))
    contrasts_resolved = all(
        value and value['ci95'][0] > 0 for value in primary.values())
    argmax_resolved = (len(argmax_effects) == 2
                       and all(value['changed_to_counterfactual_argmax'] > 0
                               for value in argmax_effects.values()))
    controlled = class_resolved and contrasts_resolved and argmax_resolved
    status = ('inconclusive_relation_matched_coverage'
              if not matched_occupancy or len(class_effects) < 2 else
              'positive_controlled_local_causal_sensitivity'
              if controlled else
              'no_resolved_controlled_causal_effect')
    result.update(
        complete=True, status=status, target=cfg['target'],
        checkpoint=f"initial/{cfg['anchor']}", layer=int(cfg['layer']),
        protocol='relation_matched_paired_cue_delta_interchange',
        direction_oof_auc=None, class_items=class_items,
        matched_class_items=matched_class_items,
        relation_items=relation_items, summaries=summaries,
        contrasts=contrasts, class_effects=class_effects,
        argmax_effects=argmax_effects, primary_contrasts=primary,
        adequacy={'matched_class_occupancy_pass': matched_occupancy,
                  'minimum_intervened_items_per_class': matched_minimum,
                  'class_effects_resolved': class_resolved,
                  'controlled_contrasts_resolved': contrasts_resolved,
                  'argmax_changes_in_both_directions': argmax_resolved},
        coverage={'eligible_rows': len(eligible),
                  'eligible_source_items': len(set(groups)),
                  'selected_source_items': len(selected),
                  'intervened_rows': len(result['rows']),
                  'intervened_relations': sorted(set(
                      row['relation'] for row in result['rows'])),
                  'skipped': skipped, 'excluded': excluded},
        scope=('Held-out local pre-answer interchange of relation-matched, '
               'within-item cue-induced residual changes. A positive status '
               'requires controlled log-odds effects and actual greedy '
               'counterfactual answer changes in both rescue and reversal. '
               'It is not whole-trajectory mediation or awareness.'))
    atomic(Path(run) / 'causal_interchange.json', result)
    return result


def causal_interchange(run, d, tok, model):
    """Dispatch the causal protocol frozen into the design."""
    version = (d.get('analysis', {}).get('causal_interchange') or {}).get(
        'version')
    if version == 'e47_causal_interchange_v1':
        return causal_interchange_v1(run, d, tok, model)
    if version == 'e47_causal_interchange_v2':
        return causal_interchange_v2(run, d, tok, model)
    raise ValueError(f'Unsupported frozen causal-interchange protocol: {version!r}')


def correct_location_anchor(d):
    """Return the frozen pre-answer anchor across legacy and current designs."""
    analysis = d.get('analysis', {})
    if analysis.get('primary_anchor') == 'pre_answer':
        return 'pre_answer'
    if analysis.get('temporal_anchor') == 'pre_answer':
        return 'pre_answer'
    raise ValueError(
        'Correct-location causal test requires pre_answer as a frozen primary '
        'or temporal anchor')


def correct_location_causal(run, d, tok, model, output=None):
    """Exploratory held-out patch toward the ground-truth option class.

    The multinomial direction is learned only from uncued training items.
    Interventions are applied only to held-out conflicting-cue rows that
    switched to an incorrect cue. This separates ground-truth location state
    from cue wording and from the switch label used by causal_interchange().
    """
    import torch
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, roc_auc_score
    from sklearn.preprocessing import StandardScaler, label_binarize
    from e46_readers import splits
    from e47_readers import interval
    from e47_design import code_hashes, runtime_versions

    run = Path(run)
    traces = completed(run, 'traces.pkl', d)
    layer = int(d['analysis']['layer'])
    anchor = correct_location_anchor(d)
    captures, _ = load_captures(run, d, layer)
    items = {int(item['index']): item for item in d['items']}
    baselines = {int(row['index']): row for row in traces['rows']
                 if row['condition'] == 'uncued'}

    def state(row):
        shard = captures.get(row['id'])
        if shard is None:
            return None
        matches = [i for i, cp in enumerate(shard['rows'])
                   if cp['anchor'] == anchor]
        return (None if len(matches) != 1 else
                np.asarray(shard['state'][matches[0]], dtype=float))

    uncued = []
    for row in baselines.values():
        vector = state(row)
        correct = items[int(row['index'])].get('correct_letter')
        if vector is not None and correct in tuple('ABCD'):
            uncued.append((row, 'ABCD'.index(correct), vector))
    switched = []
    for row in traces['rows']:
        if (not row['condition'].startswith('disagree_') or row['hit_cap']
                or target(baselines.get(int(row['index'])), row) != 1):
            continue
        correct = items[int(row['index'])].get('correct_letter')
        vector = state(row)
        if (correct in tuple('ABCD') and row['cue_letter'] != correct
                and vector is not None):
            switched.append((row, 'ABCD'.index(correct), vector))
    if not uncued or not switched:
        raise ValueError('No uncued training rows or incorrect switched rows '
                         'with aligned pre-answer captures')

    inputs = {name: file_hash(run / name)
              for name in ('design.json', 'traces.pkl', 'capture.pkl')}
    signature = digest({
        'version': 'e47_correct_location_causal_v2',
        'design_id': d['design_id'], 'inputs': inputs, 'layer': layer,
        'anchor': anchor, 'code': code_hashes()})
    destination = (Path(output) if output else
                   run / f'correct_location_causal_{signature[:12]}.json')
    if destination.exists():
        existing = read(destination)
        if existing.get('signature') != signature:
            raise ValueError('Existing correct-location causal output has a '
                             'different signature')
        if existing.get('complete'):
            print('correct-location causal result already exists:', destination)
            return existing

    X = np.stack([entry[2] for entry in uncued])
    y = np.asarray([entry[1] for entry in uncued], dtype=int)
    groups = np.asarray([entry[0]['source_id'] for entry in uncued])
    fold_for_source = {}
    folds = splits(groups, int(d['analysis'].get('folds', 5)), d['seed'])
    predictions = np.full((len(y), 4), np.nan)
    result = existing if destination.exists() else {
        'version': 'e47_correct_location_causal_v2',
        'signature': signature, 'design_id': d['design_id'],
        'inputs': inputs, 'runtime': runtime_versions(), 'complete': False,
        'target': 'ground_truth_option_direction_from_uncued_state',
        'checkpoint': f'initial/{anchor}', 'layer': layer, 'rows': [],
        'scope': ('Exploratory held-out local sensitivity. The direction is '
                  'learned only from uncued training items and patched only '
                  'on held-out incorrect switched cue trials. Positive effects '
                  'do not establish whole-trajectory mediation or awareness. '
                  'Because the uncued model usually plans the correct option, '
                  'the direction need not isolate a symbolic location from an '
                  'answer plan or option-letter representation.')}
    done = {row['id'] for row in result['rows']}
    atomic(destination, result)

    option_rows = {}
    for row, correct, vector in switched:
        option_rows.setdefault(row['source_id'], []).append(
            (row, correct, vector))
    rng_seed = lambda *parts: int(digest(parts)[:16], 16)
    alphas = (.25, .5, 1.0)
    controls = ('toward_correct', 'orthogonal_norm_matched',
                'random_norm_matched', 'sham')

    for fold, (train, test) in enumerate(folds):
        scaler = StandardScaler().fit(X[train])
        clf = LogisticRegression(
            C=0.1, max_iter=2000, solver='lbfgs', random_state=d['seed'])
        clf.fit(scaler.transform(X[train]), y[train])
        predictions[test] = clf.predict_proba(scaler.transform(X[test]))
        raw_coef = clf.coef_ / np.maximum(scaler.scale_[None, :], 1e-30)
        train_sources = set(groups[train])
        test_sources = set(groups[test])
        for source in test_sources:
            fold_for_source[source] = fold
            if source not in option_rows:
                continue
            for row, correct_index, vector in option_rows[source]:
                if row['id'] in done:
                    continue
                cue_index = 'ABCD'.index(row['cue_letter'])
                direction = raw_coef[correct_index] - raw_coef[cue_index]
                direction /= max(np.linalg.norm(direction), 1e-30)
                scores = X[train] @ direction
                correct_scores = scores[y[train] == correct_index]
                cue_scores = scores[y[train] == cue_index]
                if not len(correct_scores) or not len(cue_scores):
                    continue
                target_score = float(correct_scores.mean())
                current_score = float(vector @ direction)
                delta = direction * (target_score - current_score)
                norm = float(np.linalg.norm(delta))
                seed = rng_seed(row['id'], fold, 'correct_location_controls')
                random = np.random.default_rng(seed).normal(size=direction.size)
                random /= max(np.linalg.norm(random), 1e-30)
                orthogonal = random - direction * float(random @ direction)
                orthogonal /= max(np.linalg.norm(orthogonal), 1e-30)
                perturbations = {
                    'toward_correct': delta,
                    'orthogonal_norm_matched': orthogonal * norm,
                    'random_norm_matched': random * norm,
                    'sham': np.zeros_like(delta)}

                shard = captures[row['id']]
                distribution = shard.get('decision_distribution', {})
                if distribution.get('status') != 'ok':
                    continue
                cp = next(cp for cp in shard['rows'] if cp['anchor'] == anchor)
                ids = torch.tensor(
                    [row['prompt_ids'] + row['ids'][:cp['token_count']]],
                    device=model.get_input_embeddings().weight.device)
                pos = ids.shape[1] - 1
                option_ids = [distribution['option_token_ids'][letter]
                              for letter in 'ABCD']
                baseline = patch_vector(
                    model, ids, pos, layer, np.zeros_like(delta), option_ids)

                def margin(probability):
                    return float(np.log(max(float(probability[correct_index]), 1e-30))
                                 - np.log(max(float(probability[cue_index]), 1e-30)))

                requests = [(alpha, control, alpha * value)
                            for alpha in alphas
                            for control, value in perturbations.items()]
                probabilities = patch_vectors(
                    model, ids, pos, layer,
                    [value for _, _, value in requests], option_ids,
                    batch_size=int(
                        d['analysis'].get('causal_interchange', {}).get(
                            'batch_size', 4)))
                values = []
                for (alpha, control, applied), probability in zip(
                        requests, probabilities):
                    values.append({
                        'alpha': alpha, 'control': control,
                        'correct_vs_cue_margin_delta': (
                            margin(probability) - margin(baseline)),
                        'correct_probability_delta': float(
                            probability[correct_index]
                            - baseline[correct_index]),
                        'option_argmax': 'ABCD'[int(np.argmax(probability))],
                        'applied_delta_norm': float(np.linalg.norm(applied)),
                        'probabilities': probability.tolist()})
                result['rows'].append({
                    'id': row['id'], 'source_id': source,
                    'condition': row['condition'], 'fold': fold,
                    'correct_letter': 'ABCD'[correct_index],
                    'cue_letter': row['cue_letter'],
                    'baseline_answer': scored_answer(row),
                    'baseline_argmax': 'ABCD'[int(np.argmax(baseline))],
                    'baseline_probabilities': baseline.tolist(),
                    'direction_train_sources': sorted(train_sources),
                    'interventions': values})
                done.add(row['id'])
                if len(result['rows']) % 10 == 0:
                    atomic(destination, result)

    valid = np.isfinite(predictions).all(1)
    direction_accuracy = (float(accuracy_score(
        y[valid], np.argmax(predictions[valid], axis=1))) if valid.any()
                          else None)
    try:
        direction_auc = float(roc_auc_score(
            label_binarize(y[valid], classes=np.arange(4)),
            predictions[valid], average='macro', multi_class='ovr'))
    except ValueError:
        direction_auc = None
    summaries, contrasts = {}, {}
    probability_summaries, probability_contrasts = {}, {}
    for alpha in alphas:
        for control in controls:
            selected = [next(value for value in row['interventions']
                             if value['alpha'] == alpha
                             and value['control'] == control)
                        for row in result['rows']]
            values = [value['correct_vs_cue_margin_delta']
                      for value in selected]
            if selected:
                groups_ = [row['source_id'] for row in result['rows']]
                summaries[f'{alpha}/{control}'] = interval(
                    values, groups_, d['seed'], d['analysis']['bootstrap'])
                probability_summaries[f'{alpha}/{control}'] = {
                    'correct_probability_delta': interval(
                        [value['correct_probability_delta'] for value in selected],
                        groups_, d['seed'], d['analysis']['bootstrap']),
                    'cue_probability_delta': interval(
                        [float(value['probabilities'][
                            'ABCD'.index(row['cue_letter'])]
                         - row['baseline_probabilities'][
                            'ABCD'.index(row['cue_letter'])])
                         for row, value in zip(result['rows'], selected)],
                        groups_, d['seed'], d['analysis']['bootstrap']),
                    'correct_argmax_rate': interval(
                        [float(value['option_argmax'] == row['correct_letter'])
                         for row, value in zip(result['rows'], selected)],
                        groups_, d['seed'], d['analysis']['bootstrap']),
                    'cue_argmax_rate': interval(
                        [float(value['option_argmax'] == row['cue_letter'])
                         for row, value in zip(result['rows'], selected)],
                        groups_, d['seed'], d['analysis']['bootstrap'])}
        for reference in ('orthogonal_norm_matched', 'random_norm_matched'):
            values, correct_probability_values, cue_probability_values = [], [], []
            for row in result['rows']:
                by = {value['control']: value for value in row['interventions']
                      if value['alpha'] == alpha}
                values.append(
                    by['toward_correct']['correct_vs_cue_margin_delta']
                    - by[reference]['correct_vs_cue_margin_delta'])
                correct_probability_values.append(
                    by['toward_correct']['correct_probability_delta']
                    - by[reference]['correct_probability_delta'])
                cue_index = 'ABCD'.index(row['cue_letter'])
                cue_probability_values.append(
                    float(by['toward_correct']['probabilities'][cue_index]
                          - by[reference]['probabilities'][cue_index]))
            if values:
                key = f'{alpha}/toward_correct_minus_{reference}'
                groups_ = [row['source_id'] for row in result['rows']]
                contrasts[key] = interval(
                    values, groups_, d['seed'], d['analysis']['bootstrap'])
                probability_contrasts[key] = {
                    'correct_probability_delta': interval(
                        correct_probability_values, groups_, d['seed'],
                        d['analysis']['bootstrap']),
                    'cue_probability_delta': interval(
                        cue_probability_values, groups_, d['seed'],
                        d['analysis']['bootstrap'])}
    primary = contrasts.get('1.0/toward_correct_minus_orthogonal_norm_matched')
    primary_probability = probability_contrasts.get(
        '1.0/toward_correct_minus_orthogonal_norm_matched')
    readable = direction_auc is not None and direction_auc >= .70
    status = ('inconclusive_direction_readability' if not readable else
              'positive_controlled_ground_truth_option_sensitivity'
              if primary and primary['ci95'][0] > 0 else
              'no_resolved_controlled_ground_truth_option_effect')
    result.update(
        complete=True, status=status,
        direction_oof_accuracy=direction_accuracy,
        direction_oof_macro_auc=direction_auc,
        class_items={str(k): len(set(groups[y == k])) for k in range(4)},
        coverage={'uncued_direction_rows': len(uncued),
                  'uncued_direction_items': len(set(groups)),
                  'eligible_switched_rows': len(switched),
                  'eligible_switched_items': len(set(
                      row['source_id'] for row, _, _ in switched)),
                  'intervened_rows': len(result['rows']),
                  'intervened_items': len(set(
                      row['source_id'] for row in result['rows']))},
        baseline_distribution={
            'correct_probability': interval(
                [row['baseline_probabilities']['ABCD'.index(row['correct_letter'])]
                 for row in result['rows']],
                [row['source_id'] for row in result['rows']], d['seed'],
                d['analysis']['bootstrap']),
            'cue_probability': interval(
                [row['baseline_probabilities']['ABCD'.index(row['cue_letter'])]
                 for row in result['rows']],
                [row['source_id'] for row in result['rows']], d['seed'],
                d['analysis']['bootstrap']),
            'correct_argmax_rate': interval(
                [float(row['baseline_argmax'] == row['correct_letter'])
                 for row in result['rows']],
                [row['source_id'] for row in result['rows']], d['seed'],
                d['analysis']['bootstrap']),
            'cue_argmax_rate': interval(
                [float(row['baseline_argmax'] == row['cue_letter'])
                 for row in result['rows']],
                [row['source_id'] for row in result['rows']], d['seed'],
                d['analysis']['bootstrap'])},
        summaries=summaries, contrasts=contrasts, primary_contrast=primary,
        probability_summaries=probability_summaries,
        probability_contrasts=probability_contrasts,
        primary_probability_contrast=primary_probability)
    atomic(destination, result)
    print('correct-location causal result:', destination)
    return result


def patch(run, d, tok, model):
    """Cross-fitted predictive direction + within-item donor projection.

    A diagnostic of local answer-distribution sensitivity, not necessity,
    whole-trajectory mediation, or proof of awareness of cue influence.
    """
    import torch
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from e46_readers import splits
    from e47_readers import interval
    traces = completed(run, 'traces.pkl', d)
    from e47_design import conditions
    if d.get('version') not in ('e47_v1', 'e47_v2', 'e47_v3', 'e47_v4'):
        result = artifact(run, 'patch.json', d,
                          ('traces.pkl', 'followups.pkl', 'capture.pkl'))
        result.update(
            status='not_applicable_no_tentative_assertiveness_pair', complete=True,
            scope=('The legacy donor-projection diagnostic requires paired '
                   'tentative and assertive states for the same item and cue. '
                   'V5 and later use causal-interchange instead.'))
        atomic(Path(run) / 'patch.json', result)
        return result
    captures, _ = load_captures(run, d, d['analysis']['layer'])
    result = artifact(run, 'patch.json', d, ('traces.pkl', 'followups.pkl', 'capture.pkl'))
    if result['complete']:
        return result
    layer = d['analysis']['layer']
    baselines = {r['index']: r for r in traces['rows'] if r['condition'] == 'uncued'}
    def vector(r):
        sh = captures[r['id']]
        i = next((i for i, cp in enumerate(sh['rows']) if cp['anchor'] == 'pre_answer'), None)
        return None if i is None else np.asarray(sh['state'][i], float)
    eligible = [r for r in traces['rows'] if r['condition'] != 'uncued'
                and target(baselines[r['index']], r) is not None and vector(r) is not None]
    y = np.array([target(baselines[r['index']], r) for r in eligible])
    groups = np.array([r['source_id'] for r in eligible])
    class_items = [len(set(groups[y == k])) for k in (0, 1)]
    if (min(class_items) < d['analysis']['min_class_items']
            or len(set(groups)) < d['analysis']['min_items']):
        result.update(status='inconclusive_class_occupancy', class_items=class_items, complete=True)
        atomic(Path(run) / 'patch.json', result)
        return result
    X = np.stack([vector(r) for r in eligible])
    by = {(r['index'], r['condition']): r for r in traces['rows']}
    selected_items = set(sorted(set(groups), key=lambda s: digest([d['seed'], s]))[:d['analysis']['patch_items']])
    done = {r['id'] for r in result['rows']}
    predictions = np.full(len(y), np.nan)
    skipped = {}
    for fold, (train, test) in enumerate(splits(groups, 5, d['seed'])):
        if len(np.unique(y[train])) < 2:
            continue
        scaler = StandardScaler().fit(X[train])
        clf = LogisticRegression(C=0.1, max_iter=1500, solver='liblinear', random_state=d['seed']).fit(scaler.transform(X[train]), y[train])
        predictions[test] = clf.predict_proba(scaler.transform(X[test]))[:, 1]
        direction = clf.coef_[0] / scaler.scale_
        direction /= max(np.linalg.norm(direction), 1e-30)
        result.setdefault('directions', {})[str(fold)] = {
            'raw_unit_direction': direction.tolist(),
            'train_sources': sorted(set(groups[train])), 'test_sources': sorted(set(groups[test]))}
        for i in test:
            r = eligible[i]
            if r['source_id'] not in selected_items or r['id'] in done:
                continue
            sh = captures[r['id']]
            dist = sh['decision_distribution']
            donor_condition = r['condition'].replace('assertive', 'tentative') if 'assertive' in r['condition'] else r['condition'].replace('tentative', 'assertive')
            donor = by.get((r['index'], donor_condition))
            dv = vector(donor) if donor is not None and target(baselines[donor['index']], donor) is not None else None
            if dist['status'] != 'ok' or dv is None:
                reason = dist['status'] if dist['status'] != 'ok' else 'missing_valid_donor'
                skipped[reason] = skipped.get(reason, 0) + 1
                continue
            cp = next(cp for cp in sh['rows'] if cp['anchor'] == 'pre_answer')
            ids = torch.tensor([r['prompt_ids'] + r['ids'][:cp['token_count']]],
                               device=model.get_input_embeddings().weight.device)
            pos = ids.shape[1] - 1
            option_ids = [dist['option_token_ids'][c] for c in 'ABCD']
            baseline = patch_vector(model, ids, pos, layer, np.zeros_like(direction), option_ids)
            old = np.array([dist['probabilities'][c] for c in 'ABCD'])
            if not np.allclose(baseline, old, atol=0.003, rtol=0.03):
                raise ValueError('Patch replay differs from captured unpatched distribution')
            shift = direction * np.dot(dv - X[i], direction)
            rng = np.random.default_rng(int(r['id'][:8], 16))
            random_direction = rng.normal(size=direction.shape)
            random_direction /= np.linalg.norm(random_direction)
            matched_noise = random_direction * np.linalg.norm(shift)
            values = []
            def cue_log_odds(p):
                cue_index = 'ABCD'.index(r['cue_letter'])
                return float(np.log(max(float(p[cue_index]), 1e-30)) -
                             np.log(max(float(np.delete(p, cue_index).sum()), 1e-30)))
            for alpha in d['analysis']['patch_alpha']:
                for control, delta in (('targeted', shift), ('reverse', -shift),
                                       ('random_norm_matched', matched_noise), ('sham', np.zeros_like(shift))):
                    p = patch_vector(model, ids, pos, layer, alpha * delta, option_ids)
                    values.append({'alpha': alpha, 'control': control,
                                   'cue_probability_delta': float(p['ABCD'.index(r['cue_letter'])] - baseline['ABCD'.index(r['cue_letter'])]),
                                   'cue_log_odds_delta': cue_log_odds(p) - cue_log_odds(baseline),
                                   'probabilities': p.tolist(), 'valid_option_mass': float(p.sum())})
            result['rows'].append({'id': r['id'], 'source_id': r['source_id'], 'condition': r['condition'],
                                   'fold': fold, 'direction_train_sources': sorted(set(groups[train])),
                                   'delta_norm': float(np.linalg.norm(shift)), 'interventions': values})
            atomic(Path(run) / 'patch.json', result)
    summaries = {}
    for condition in sorted({r['condition'] for r in result['rows']}):
        for alpha in d['analysis']['patch_alpha']:
            for control in ('targeted', 'reverse', 'random_norm_matched', 'sham'):
                values, odds, gs = [], [], []
                for row in result['rows']:
                    if row['condition'] != condition:
                        continue
                    for v in row['interventions']:
                        if v['alpha'] == alpha and v['control'] == control:
                            values.append(v['cue_probability_delta'])
                            odds.append(v['cue_log_odds_delta'])
                            gs.append(row['source_id'])
                if values:
                    summaries[f'{condition}/{alpha}/{control}'] = {
                        'probability_delta': interval(values, gs), 'log_odds_delta': interval(odds, gs)}
    from sklearn.metrics import roc_auc_score
    result.update(status='diagnostic_only' if result['rows'] else 'no_eligible_patches',
                  summaries=summaries, complete=True,
                  direction_oof_auc=float(roc_auc_score(y, predictions)) if np.isfinite(predictions).all() else None,
                  coverage={'eligible_rows': len(eligible), 'selected_source_items': len(selected_items),
                            'patched_rows': len(result['rows']), 'skipped_selected': skipped},
                  scope='Cross-fitted predictive direction; donor is other assertiveness of same item/cue. Local pre-answer sufficiency diagnostic, not necessity or mediation. Signed effects may cancel across donor directions.')
    atomic(Path(run) / 'patch.json', result)
    return result
