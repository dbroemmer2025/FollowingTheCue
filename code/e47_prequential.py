#!/usr/bin/env python3
"""Paper D: prequential (online) codelength for the E47 verbalization gap.

The paper's G_SOT is a held-out log-loss difference between the best member of
a text reader family and the best member of a nested text+state family. A
reviewer can always ask whether the gap reflects the residual stream or simply
the larger joint family. Prequential MDL (Rissanen 1984; Blier & Ollivier 2018;
Voita & Titov 2020) answers that by *charging for capacity*: each block of
items is encoded with a model trained only on the preceding blocks, so a family
that needs many examples to become useful pays for them in bits.

For each model--task cell this script reports, on the frozen population and
the frozen prespecified checkpoint:

  * codelength (bits) of the switch target under a uniform code, an adaptive
    Laplace (label-only) code, and the text, state and joint reader families;
  * the MDL gap  M_SOT = L_text - L_joint  (and the non-nested M_A = L_text -
    L_state as a diagnostic), total and per row;
  * a capacity curve: the joint-family codelength when the residual block may
    use at most d principal components, d in the ladder, so the gap can be read
    as a function of state capacity rather than as one number;
  * per-row prequential posteriors, which ``deficiency.py --oof`` turns into
    the cost-curve deficiency under capacity-accounted readers.

Family codes. Each fixed candidate gives a prequential code. A family is coded
by the block-level Bayes mixture over its candidates with a uniform prior,
which is a valid code whose length is within log2(K) bits of the best
candidate *in hindsight* -- so selection is paid for, not hidden. The
follow-the-leader code (pick the candidate with the smallest codelength on past
blocks) is also reported. The joint candidate set literally contains every
text candidate, as in the frozen design; nesting therefore bounds
L_mix(joint) <= min_text L_k + log2(K_joint), not L_mix(joint) <= L_mix(text).

Validity. Item order is a seeded permutation of item identifiers and never
consults labels; every item's rows stay in one block; all preprocessing
(TF-IDF vocabulary, scaler, PCA, calibration) is fit on the past blocks only.
The first block is encoded with the uniform code for every family, so it
cancels in every difference. Codelengths depend on the order; the script
averages over --orders permutations and reports their spread.

Reader capacity is still family-relative. MDL makes the comparison
capacity-*accounted*, not capacity-*free*: the conclusion is "with this
ladder, paying for what the readers learn", not a statement about all readers.

Requires the per-sample capture shards (Colab/Drive), like the settling curve.

Usage (from code/):
  python e47_prequential.py --run ../e47/<cell> --out ../reports/prequential
  python e47_prequential.py --run ../e47/<cell> --layers 0 7 14 21 27
  python e47_prequential.py --run ../e47/<cell> --quick       # small ladder
  python e47_prequential.py --selftest                        # no shards
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

VERSION = 'e47_prequential_v1'
DEFAULT_FRACTIONS = (1 / 32, 1 / 16, 1 / 8, 1 / 4, 1 / 2, 3 / 4, 1.0)
TEXT_C = (0.01, 0.1, 1.0, 10.0, 100.0)
STATE_DIMS = (1, 2, 4, 8, 16, 32, 64, -1)        # -1 = full standardized state
STATE_C = (0.1, 1.0)
JOINT_SCALES = (0.1, 1.0)


# --------------------------------------------------------------------------
# Candidate ladder
# --------------------------------------------------------------------------

def ladder(quick=False):
    """Return {family: [(name, meta, factory)]}. Joint contains all text."""
    from e47_readers import BlockReader, StateReader
    text_c = (0.1, 1.0, 10.0) if quick else TEXT_C
    dims = (2, 8, 32) if quick else STATE_DIMS
    state_c = (1.0,) if quick else STATE_C
    scales = (1.0,) if quick else JOINT_SCALES
    text = [(f'text_C{c}', {'state_dim': 0, 'C': c},
             lambda c=c: BlockReader('word', c, 0)) for c in text_c]
    state = [(f'state{d}_C{c}', {'state_dim': d, 'C': c},
              lambda d=d, c=c: StateReader(c=c, state_dim=d))
             for d in dims for c in state_c]
    joint = list(text) + [
        (f'joint_C{c}_state{d}_scale{s}', {'state_dim': d, 'C': c, 'scale': s},
         lambda d=d, c=c, s=s: BlockReader('word', c, d, s))
        for d in dims for c in state_c for s in scales]
    return {'text': text, 'state': state, 'joint': joint}


def capacity_rank(d):
    return math.inf if d == -1 else d


# --------------------------------------------------------------------------
# Prequential engine
# --------------------------------------------------------------------------

def block_plan(groups, fractions, seed):
    """Label-independent item order and block boundaries (row index arrays)."""
    groups = np.asarray(groups)
    items = np.unique(groups)
    order = np.random.default_rng(seed).permutation(items)
    n = len(order)
    cuts = sorted({min(n, max(1, int(math.ceil(f * n)))) for f in fractions})
    if cuts[-1] != n:
        cuts.append(n)
    rank = {g: i for i, g in enumerate(order)}
    item_rank = np.asarray([rank[g] for g in groups])
    blocks, start = [], 0
    for cut in cuts:
        blocks.append(np.flatnonzero((item_rank >= start) & (item_rank < cut)))
        start = cut
    return [b for b in blocks if len(b)]


def predict_block(factory, rows, y, groups, train, test, calibrate, seed):
    """Fit on past rows only; return calibrated p(z=1) for the test block."""
    from e46_readers import (calibrate as apply_cal, choose_calibration,
                             fit_predict, prior, splits)
    base = prior(y[train], 2)
    if len(np.unique(y[train])) < 2:
        return np.full(len(test), base[1])
    raw = fit_predict(factory, rows, y, train, test, 2)
    if not calibrate or len(np.unique(groups[train])) < 3:
        return raw[:, 1]
    inner = splits(groups[train], 3, seed)
    pi = np.zeros((len(train), 2))
    ib = np.zeros((len(train), 2))
    for it, iv in inner:
        pi[iv] = fit_predict(factory, rows, y, train[it], train[iv], 2)
        ib[iv] = prior(y[train[it]], 2)
    _, temp, mix = choose_calibration(pi, y[train], ib)
    return apply_cal(raw, temp, mix, np.tile(base, (len(test), 1)))[:, 1]


def bits(y, p1):
    p1 = np.clip(np.asarray(p1, float), 1e-12, 1 - 1e-12)
    return -(y * np.log2(p1) + (1 - y) * np.log2(1 - p1))


def prequential_family(rows, y, groups, blocks, candidates, *, calibrate,
                       seed, jobs=1):
    """Per-candidate per-row bits and p1 (first block: uniform, p1 = .5)."""
    n = len(y)
    names = [c[0] for c in candidates]
    P = np.full((len(candidates), n), 0.5)
    for b in range(1, len(blocks)):
        train = np.concatenate(blocks[:b])
        test = blocks[b]

        def run(k):
            return predict_block(candidates[k][2], rows, y, groups, train,
                                 test, calibrate, seed + 7919 * b)
        if jobs == 1:
            outs = [run(k) for k in range(len(candidates))]
        else:
            from joblib import Parallel, delayed
            outs = Parallel(n_jobs=jobs)(delayed(run)(k)
                                         for k in range(len(candidates)))
        for k, p in enumerate(outs):
            P[k, test] = p
    B = np.stack([bits(y, P[k]) for k in range(len(candidates))])
    return names, P, B


def mixture_code(B, blocks):
    """Block-level Bayes mixture (uniform prior) and follow-the-leader codes.

    Returns total bits for each, plus per-row mixture predictive weights used
    to export a row-level p1 (weights depend only on past blocks).
    """
    K = B.shape[0]
    block_bits = np.stack([B[:, blk].sum(1) for blk in blocks], 1)   # K x nb
    past = np.zeros(K)
    mix_total = 0.0
    ftl_total = 0.0
    weights = []
    for j in range(block_bits.shape[1]):
        logw = -past * math.log(2) - math.log(K)
        logw -= np.logaddexp.reduce(logw)
        w = np.exp(logw)
        weights.append(w)
        # -log2 sum_k w_k 2^{-L_k(block)}
        mix_total += -np.logaddexp.reduce(logw - block_bits[:, j] * math.log(2)) / math.log(2)
        ftl_total += block_bits[int(np.argmin(past)), j]   # ties -> first (simplest)
        past += block_bits[:, j]
    best = float(past.min())
    return {'mixture_bits': float(mix_total), 'ftl_bits': float(ftl_total),
            'best_in_hindsight_bits': best,
            'best_in_hindsight': int(np.argmin(past)),
            'mixture_regret_bound_bits': math.log2(K),
            'block_weights': weights}


def row_mixture_p1(P, blocks, weights):
    p = np.full(P.shape[1], 0.5)
    for blk, w in zip(blocks, weights):
        p[blk] = w @ P[:, blk]
    return p


def laplace_code(y, blocks):
    from e46_readers import prior
    p = np.full(len(y), 0.5)
    for b in range(1, len(blocks)):
        train = np.concatenate(blocks[:b])
        p[blocks[b]] = prior(y[train], 2)[1]
    return p


def run_cell(rows, y, groups, *, fractions=DEFAULT_FRACTIONS, orders=3,
             seed=47, calibrate=True, quick=False, jobs=1, families=None):
    fams = ladder(quick) if families is None else families
    y, groups = np.asarray(y, int), np.asarray(groups)
    per_order = []
    export = None
    for o in range(orders):
        blocks = block_plan(groups, fractions, seed + 1000 * o)
        first = blocks[0]
        coded = np.concatenate(blocks[1:])
        res = {'order': o, 'n_blocks': len(blocks),
               'first_block_rows': int(len(first)),
               'uniform_bits': float(len(y)),
               'laplace_bits': float(bits(y, laplace_code(y, blocks)).sum())}
        cand, rowp = {}, {'prior': laplace_code(y, blocks)}
        for fam, candidates in fams.items():
            names, P, B = prequential_family(
                rows, y, groups, blocks, candidates, calibrate=calibrate,
                seed=seed + 1000 * o, jobs=jobs)
            mix = mixture_code(B, blocks)
            res[fam] = {k: v for k, v in mix.items() if k != 'block_weights'}
            res[fam]['best_in_hindsight'] = names[mix['best_in_hindsight']]
            cand[fam] = {nm: float(B[k].sum()) for k, nm in enumerate(names)}
            rowp[fam] = row_mixture_p1(P, blocks, mix['block_weights'])
            # Capacity curve: mixture over joint candidates with state_dim <= d.
            if fam == 'joint':
                curve = {}
                dims = sorted({m['state_dim'] for _, m, _ in candidates
                               if m['state_dim'] != 0}, key=capacity_rank)
                for d in dims:
                    keep = [k for k, (_, m, _) in enumerate(candidates)
                            if capacity_rank(m['state_dim']) <= capacity_rank(d)]
                    curve[str(d)] = mixture_code(B[keep], blocks)['mixture_bits']
                res['joint_capacity_curve'] = curve
        res['candidates'] = cand
        res['M_SOT_bits'] = res['text']['mixture_bits'] - res['joint']['mixture_bits']
        res['M_A_bits'] = res['text']['mixture_bits'] - res['state']['mixture_bits']
        res['coded_rows'] = int(len(coded))
        per_order.append(res)
        if o == 0:
            export = {'blocks': blocks, 'rowp': rowp}
    return per_order, export


def summarize(per_order, n_rows):
    def stat(get):
        v = np.array([get(r) for r in per_order], float)
        return {'mean': float(v.mean()), 'sd': float(v.std(ddof=1)) if len(v) > 1 else 0.0,
                'min': float(v.min()), 'max': float(v.max())}
    coded = per_order[0]['coded_rows']
    out = {'n_rows': n_rows, 'coded_rows_order0': coded,
           'uniform_bits': float(n_rows)}
    out['laplace_bits'] = stat(lambda r: r['laplace_bits'])
    for fam in ('text', 'state', 'joint'):
        out[f'{fam}_mixture_bits'] = stat(lambda r: r[fam]['mixture_bits'])
        out[f'{fam}_ftl_bits'] = stat(lambda r: r[fam]['ftl_bits'])
    out['M_SOT_bits'] = stat(lambda r: r['M_SOT_bits'])
    out['M_SOT_bits_per_coded_row'] = stat(lambda r: r['M_SOT_bits'] / r['coded_rows'])
    out['M_A_bits'] = stat(lambda r: r['M_A_bits'])
    out['text_minus_laplace_bits'] = stat(
        lambda r: r['laplace_bits'] - r['text']['mixture_bits'])
    out['compression_vs_uniform'] = {
        fam: float(n_rows / out[f'{fam}_mixture_bits']['mean'])
        for fam in ('text', 'state', 'joint')}
    if 'joint_capacity_curve' in per_order[0]:
        dims = list(per_order[0]['joint_capacity_curve'])
        out['capacity_curve_M_SOT_bits'] = {
            d: stat(lambda r, d=d: r['text']['mixture_bits']
                    - r['joint_capacity_curve'][d]) for d in dims}
    return out


# --------------------------------------------------------------------------
# E47 loading (Colab; needs capture shards)
# --------------------------------------------------------------------------

def resolve_layers(spec, run, design_layer):
    """Per-model capture-site indices. Index 0 is the block input; index k is
    block k's output, so the frozen 'second_to_last' rule gives n_layers - 1."""
    import pickle
    meta = pickle.loads((Path(run) / 'capture.pkl').read_bytes())['layer_metadata']
    n = int(meta['n_layers'])
    width = int(meta.get('n_sites', len(meta.get('layer_sites', [])) or n + 2))
    out = []
    for item in spec:
        item = str(item)
        if item == 'frozen':
            out.append(int(design_layer))
        elif item == 'auto':
            out += [0, round(.25 * n), round(.50 * n), round(.75 * n),
                    int(design_layer)]
        else:
            out.append(int(item))
    out = list(dict.fromkeys(out))
    if min(out) < 0 or max(out) >= width:
        raise ValueError(f'{Path(run).name}: layers must be in [0, {width - 1}]')
    return out


def load_cell(run, layers, checkpoint=None, include_sequence=False):
    """Rows for one checkpoint on the frozen population, all requested layers.

    Mirrors e47_analysis.datasets for the 'initial' stage and then checks the
    resulting ids and targets against the frozen oof_losses.csv, so the
    population cannot silently drift.  ``include_sequence`` exposes the
    frozen random-projected input-embedding prefix used by E47's Transformer
    reader; it is opt-in so existing lightweight callers do not load it.
    """
    from e47_design import completed, load_design, scored_answer, target
    from e47_capture import load_captures
    from e47_analysis import recoverability_population
    run = Path(run)
    d = load_design(run, check_code=False)
    analysis = json.loads((run / 'analysis.json').read_text())
    key = checkpoint or analysis['primary']
    stage, anchor = key.split('/')
    if stage != 'initial':
        raise ValueError('Prequential codes are prespecified for initial/* only')
    traces = completed(run, 'traces.pkl', d)
    layers = [int(x) for x in layers]
    captures, _ = load_captures(run, d, layers=layers)
    base = {r['index']: r for r in traces['rows'] if r['condition'] == 'uncued'}
    rows = []
    for row in traces['rows']:
        if row['condition'] == 'uncued':
            continue
        z = target(base[row['index']], row)
        if z is None or row['hit_cap']:
            continue
        sh = captures.get(row['id'])
        if sh is None or sh['generation_sha256'] != row['content_sha256']:
            raise ValueError('Missing/misaligned capture')
        hits = [i for i, cp in enumerate(sh['rows']) if cp['anchor'] == anchor]
        if not hits:
            continue
        i = hits[0]
        cp = sh['rows'][i]
        entry = {'id': row['id'], 'group': row['source_id'], 'y': z,
                 'condition': row['condition'], 'cue_letter': row['cue_letter'],
                 'prefix': cp['prefix'], 'prompt': row['prompt'],
                 'text': row['prompt'] + '\n' + cp['prefix'],
                 'states': {L: np.asarray(sh['states_by_layer'][L][i], float)
                            for L in layers}}
        if include_sequence:
            features = sh.get('token_features')
            if features is None:
                raise ValueError(
                    f'{run.name}: Transformer comparison requires token_features; '
                    'the frozen capture shards do not contain them')
            end = int(cp['absolute_position']) + 1
            entry['sequence'] = np.asarray(features[:end], dtype=np.float32)
        rows.append(entry)
    rows = recoverability_population(rows, d['analysis'])
    frozen = {r['id']: int(r['target']) for r in csv.DictReader(
        (run / analysis.get('oof_losses_file', 'oof_losses.csv')).open())
        if r['checkpoint'] == key}
    got = {r['id']: r['y'] for r in rows}
    if got != frozen:
        raise ValueError(f'{run.name}/{key}: population differs from frozen '
                         f'oof_losses.csv ({len(got)} vs {len(frozen)} rows)')
    return rows, d, key


# --------------------------------------------------------------------------
# Selftests (synthetic; no shards)
# --------------------------------------------------------------------------

def _synthetic(n_items=120, signal='state', seed=0, dim=16):
    rng = np.random.default_rng(seed)
    words = ['alpha', 'beta', 'gamma', 'delta', 'eps', 'zeta', 'eta', 'theta']
    rows, y, g = [], [], []
    for it in range(n_items):
        for rep in range(2):
            z = int(rng.integers(0, 2))
            state = rng.normal(size=dim)
            text = ' '.join(rng.choice(words, 12))
            if signal == 'state':
                state[0] += 2.5 * (2 * z - 1)
            elif signal == 'text':
                text += ' marker' if z else ' other'
            rows.append({'text': text, 'state': state, 'group': f'i{it}'})
            y.append(z)
            g.append(f'i{it}')
    return rows, np.array(y), np.array(g)


def selftest():
    from e47_readers import BlockReader, StateReader
    small = {
        'text': [('text_C1', {'state_dim': 0, 'C': 1.0},
                  lambda: BlockReader('word', 1.0, 0))],
        'state': [('state4_C1', {'state_dim': 4, 'C': 1.0},
                   lambda: StateReader(c=1.0, state_dim=4))],
        'joint': [('text_C1', {'state_dim': 0, 'C': 1.0},
                   lambda: BlockReader('word', 1.0, 0)),
                  ('joint_C1_state4', {'state_dim': 4, 'C': 1.0, 'scale': 1.0},
                   lambda: BlockReader('word', 1.0, 4, 1.0))]}
    n = 0
    # 1. Blocks partition rows and keep items whole; order ignores labels.
    rows, y, g = _synthetic()
    blocks = block_plan(g, DEFAULT_FRACTIONS, 3)
    allrows = np.sort(np.concatenate(blocks))
    assert np.array_equal(allrows, np.arange(len(y)))
    for b in blocks:
        for other in blocks:
            if other is not b:
                assert not set(g[b]) & set(g[other])
    assert all(np.array_equal(a, b) for a, b in zip(
        blocks, block_plan(g, DEFAULT_FRACTIONS, 3)))
    n += 1
    # 2. Mixture code within log2 K of the best candidate and never better
    #    than the best by more than zero.
    rng = np.random.default_rng(1)
    B = rng.exponential(1.0, (4, len(y)))
    m = mixture_code(B, blocks)
    assert m['best_in_hindsight_bits'] - 1e-9 <= m['mixture_bits'] \
        <= m['best_in_hindsight_bits'] + 2 + 1e-9
    n += 1
    # 3. State-signal data: joint and state codes beat text; M_SOT > 0.
    per, exp = run_cell(rows, y, g, orders=2, calibrate=True, families=small)
    s = summarize(per, len(y))
    assert s['M_SOT_bits']['mean'] > 20, s['M_SOT_bits']
    assert s['M_A_bits']['mean'] > 20
    n += 1
    # 4. Text-signal data: adding state cannot buy much; M_SOT within the
    #    mixture regret (1 bit for K=2) plus small estimation noise.
    rows_t, y_t, g_t = _synthetic(signal='text', seed=2)
    per_t, _ = run_cell(rows_t, y_t, g_t, orders=2, calibrate=True, families=small)
    st = summarize(per_t, len(y_t))
    assert st['M_SOT_bits']['mean'] < 8, st['M_SOT_bits']
    assert st['text_minus_laplace_bits']['mean'] > 20
    n += 1
    # 5. Null data: nothing beats the Laplace code by much.
    rows_n, y_n, g_n = _synthetic(signal='none', seed=4)
    per_n, _ = run_cell(rows_n, y_n, g_n, orders=1, calibrate=True, families=small)
    sn = summarize(per_n, len(y_n))
    assert sn['joint_mixture_bits']['mean'] > sn['laplace_bits']['mean'] - 8
    n += 1
    # 6. First block is uniform for every family (p1 = .5) and cancels.
    first = exp['blocks'][0]
    for fam in ('text', 'state', 'joint'):
        assert np.allclose(exp['rowp'][fam][first], 0.5)
    n += 1
    # 7. Determinism.
    per2, _ = run_cell(rows, y, g, orders=2, calibrate=True, families=small)
    assert abs(per2[0]['M_SOT_bits'] - per[0]['M_SOT_bits']) < 1e-9
    n += 1
    # 8. The full ladder nests text inside joint.
    full = ladder()
    tnames = {nm for nm, _, _ in full['text']}
    assert tnames <= {nm for nm, _, _ in full['joint']}
    n += 1
    print(f'e47_prequential selftest: {n}/8 passed')


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def summarize_dir(path):
    """Cross-cell table from *.prequential.json, compared with frozen G_SOT."""
    path = Path(path)
    print(f"{'cell':48s} {'L':>3s} {'n':>4s} {'laplace':>8s} {'text':>8s} "
          f"{'joint':>8s} {'M_SOT':>8s} {'±sd':>5s} {'/row':>6s} {'G_SOT':>6s}")
    rows = []
    for f in sorted(path.glob('*.prequential.json')):
        p = json.loads(f.read_text())
        run = HERE.parent / 'e47' / p['run']
        a = json.loads((run / 'analysis.json').read_text())
        g = a['checkpoints'][p['checkpoint']]['gap']['mean']
        for L, block in p['layers'].items():
            s = block['summary']
            row = [p['run'].replace('_thinking_v10', ''), L, p['n_rows'],
                   s['laplace_bits']['mean'], s['text_mixture_bits']['mean'],
                   s['joint_mixture_bits']['mean'], s['M_SOT_bits']['mean'],
                   s['M_SOT_bits']['sd'], s['M_SOT_bits_per_coded_row']['mean'], g]
            rows.append(row)
            print(f'{row[0]:48s} {L:>3s} {row[2]:4d} {row[3]:8.1f} {row[4]:8.1f} '
                  f'{row[5]:8.1f} {row[6]:8.1f} {row[7]:5.1f} {row[8]:6.3f} {row[9]:6.3f}')
    with (path / 'prequential_summary.csv').open('w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['cell', 'layer', 'n_rows', 'laplace_bits', 'text_bits',
                    'joint_bits', 'M_SOT_bits', 'M_SOT_sd', 'M_SOT_per_row',
                    'G_SOT_frozen'])
        w.writerows(rows)
    print(f"\nM_SOT/row is per coded row (first block excluded); G_SOT is the "
          f"frozen item-averaged log-loss gap. Compare signs and ordering, "
          f"not magnitudes: a prequential code includes the cost of learning.")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--summarize', metavar='DIR',
                   help='print/write the cross-cell table and exit')
    p.add_argument('--run')
    p.add_argument('--checkpoint', default=None, help='default: frozen primary')
    p.add_argument('--layers', nargs='+', default=['frozen'],
                   help="capture-site indices, 'frozen' (design.json "
                        "analysis.layer, the default), or 'auto' (embedding "
                        "input, 25%%, 50%%, 75%% depth, frozen; same rule as "
                        "e47_residual_trajectory.py)")
    p.add_argument('--orders', type=int, default=3)
    p.add_argument('--fractions', nargs='+', type=float, default=list(DEFAULT_FRACTIONS))
    p.add_argument('--no-calibrate', dest='calibrate', action='store_false')
    p.add_argument('--quick', action='store_true', help='reduced ladder (smoke test)')
    p.add_argument('--jobs', type=int, default=1)
    p.add_argument('--seed', type=int, default=47)
    p.add_argument('--out', default=str(HERE.parent / 'reports' / 'prequential'))
    p.add_argument('--selftest', action='store_true')
    a = p.parse_args(argv)
    if a.selftest:
        selftest()
        return 0
    if a.summarize:
        summarize_dir(a.summarize)
        return 0
    if not a.run:
        p.error('--run is required')
    run = Path(a.run).resolve()
    design_layer = json.loads((run / 'design.json').read_text())['analysis']['layer']
    layers = resolve_layers(a.layers, run, design_layer)
    print(f'{run.name}: layers {layers} (frozen {design_layer})', flush=True)
    t0 = time.time()
    rows, d, key = load_cell(run, layers, a.checkpoint)
    y = np.array([r['y'] for r in rows])
    groups = np.array([r['group'] for r in rows])
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cell = run.name.replace('_thinking_v10', '')
    oof = run / 'oof_losses.csv'
    payload = {'version': VERSION, 'run': run.name, 'checkpoint': key,
               'design_id': d['design_id'], 'frozen_layer': design_layer,
               'population': d['analysis'].get('recoverability_population'),
               'oof_losses_sha256': hashlib.sha256(oof.read_bytes()).hexdigest(),
               'code_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               'orders': a.orders, 'fractions': a.fractions,
               'calibrate': a.calibrate, 'quick': a.quick, 'seed': a.seed,
               'n_rows': int(len(y)), 'n_items': int(len(set(groups))),
               'class_rows': np.bincount(y, minlength=2).tolist(),
               'layers': {}}
    for L in layers:
        lrows = [{**r, 'state': r['states'][L]} for r in rows]
        per, exp = run_cell(lrows, y, groups, fractions=a.fractions,
                            orders=a.orders, seed=a.seed, calibrate=a.calibrate,
                            quick=a.quick, jobs=a.jobs)
        payload['layers'][str(L)] = {'summary': summarize(per, len(y)),
                                     'per_order': per,
                                     'frozen_layer': L == design_layer}
        coded = np.sort(np.concatenate(exp['blocks'][1:]))
        path = out / f'{cell}__L{L}.rows.csv'
        with path.open('w', newline='') as fh:
            w = csv.writer(fh)
            w.writerow(['id', 'source_id', 'target', 'condition', 'prior_p1',
                        'text_p1', 'state_p1', 'joint_p1'])
            for i in coded:
                w.writerow([rows[i]['id'], rows[i]['group'], int(y[i]),
                            rows[i]['condition'], exp['rowp']['prior'][i],
                            exp['rowp']['text'][i], exp['rowp']['state'][i],
                            exp['rowp']['joint'][i]])
        s = payload['layers'][str(L)]['summary']
        print(f"L{L:>3}: text {s['text_mixture_bits']['mean']:.1f}  "
              f"state {s['state_mixture_bits']['mean']:.1f}  "
              f"joint {s['joint_mixture_bits']['mean']:.1f}  "
              f"laplace {s['laplace_bits']['mean']:.1f} bits | "
              f"M_SOT {s['M_SOT_bits']['mean']:.1f} ± {s['M_SOT_bits']['sd']:.1f} "
              f"({s['M_SOT_bits_per_coded_row']['mean']:.3f}/row)", flush=True)
    payload['elapsed_seconds'] = time.time() - t0
    name = out / f'{cell}.prequential.json'
    name.write_text(json.dumps(payload, indent=1, default=float))
    print(f'wrote {name}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
