#!/usr/bin/env python3
"""E47 determinants of cue acknowledgment, re-estimated on the existing runs.

Revises Section V-H ("What Determines Whether the Model Acknowledges the
Cue?") in three ways. All three run on data we already have.

1. PRE-TREATMENT OVERRIDE MODERATOR. Whether the model answered correctly
   *without* the cue is fixed before treatment, so it is a legitimate
   moderator on the full randomized disagreement population (unlike
   switching, which is post-treatment). On Gemma movie, authority
   acknowledgment is 59% when the uncued answer was correct vs 37% when it
   was wrong; among switchers, overriding a correct answer ("flipper") is
   acknowledged 83% of the time vs 41% for wrong-to-cue. The script reports
   this per cell and pools it with a Mantel-Haenszel odds ratio stratified
   by cell.

2. A LOWER-DIMENSIONAL PROSPECTIVE STACKED READER. The paper's
   joint reader (13 pre-treatment features + 32 unsupervised PCs of one
   layer, C=1, n~138) can score below the pre-treatment reader (Gemma movie:
   -0.122). That is estimator dilution, not evidence that the state hurts.
   Here the state enters as a nested out-of-fold probability (stacking) with
   C tuned in an inner grouped CV, across several layers. This reduces
   estimator dilution but does not guarantee nonnegative held-out AUC gain.
   The paper-style joint estimate is reported alongside for comparison.

3. MENTION-CENSORED CHECKPOINTS. At every checkpoint after the prompt,
   responses whose prefix already contains acknowledgment language are
   dropped, so a later-checkpoint AUC measures anticipation, not readout of
   text already written (the paper's lead-5 AUC 0.839 is not censored).

Populations (per cell x cue family):
  all_disagree   every reviewed, valid disagreement trial (primary, per V-H g)
  switchers      switch-to-cue trials only (secondary; post-treatment)
Targets:
  initial        reviewed acknowledgment in the initial response
  why            reviewed acknowledgment in the neutral why response
  why_new        why acknowledgment among initially silent responses

The acknowledgment label is e47_design.review_cue_acknowledgment (mention OR
source_present OR source_correct), the same definition as the paper's.

Usage:
  python e47_disclosure_determinants.py                      # DEFAULT_RUN_DIRS, strata only
  python e47_disclosure_determinants.py --runs ../e47/gemma_2_9b_it_movie_recommendations_thinking_v10 \\
      --readers --layers 16,24,31,41 --out ../reports/disclosure_determinants
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import warnings
from pathlib import Path

import numpy as np

from e46_readers import prior, splits
from e47_design import review_cue_acknowledgment
from e47_what_causes_switch import META_NAMES, PRE_NAMES
from run_dirs import DEFAULT_RUN_DIRS

SCHEMA = 'e47_disclosure_determinants_v1'
ROOT = Path(__file__).resolve().parents[1]
ANCHORS = ('prompt', 'frac25', 'frac50', 'frac75', 'lead40', 'lead20',
           'lead10', 'lead5', 'pre_answer', 'complete')
# Liberal lexicons for censoring only (not for labels): any hint that the
# source has been named in the prefix removes the row at that checkpoint.
CENSOR_RE = {
    'authority': re.compile(r'stanford|professor|expert', re.I),
    'sycophancy': re.compile(
        r"\byou(?:'re| think| said| suggest| mention| believe)|"
        r"\byour (?:answer|suggestion|guess|view|opinion)|\bthe user\b", re.I),
}
STRATA = ('flipper', 'wrong_to_cue', 'resister', 'other')
C_GRID = (0.01, 0.1, 1.0)


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------

def resolve_run(value):
    path = Path(value).expanduser()
    return path.resolve() if path.exists() else (ROOT / path).resolve()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def resolved_rows_artifact(run, design, review_path, explicit=None):
    """Resolve rows through its signed JSON companion, never file mtime."""
    if explicit is not None:
        path = Path(explicit).expanduser().resolve()
        if not path.exists():
            raise ValueError(f'explicit rows file does not exist: {path}')
        return path
    annotation_hash = file_hash(review_path)
    matches = []
    for manifest in run.glob('what_causes_switch*.json'):
        payload = json.loads(manifest.read_text())
        if (payload.get('design_id') != design.get('design_id')
                or payload.get('inputs', {}).get('annotations') != annotation_hash):
            continue
        rows = manifest.with_suffix('.rows.csv')
        if rows.exists():
            matches.append(rows)
    if len(matches) != 1:
        raise ValueError(
            f'{run}: expected one what-causes-switch rows artifact bound to '
            f'the current design and annotations, found {len(matches)}; '
            'use --rows-file for a single-run analysis')
    return matches[0]


def read_csv(path):
    with Path(path).open(newline='') as handle:
        return list(csv.DictReader(handle))


def num(value):
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(out) else out


def stratum(row):
    unc_correct = row['uncued_answer'] == row['correct_letter']
    followed = row['cued_answer'] == row['cue_letter']
    if followed and unc_correct:
        return 'flipper'
    if followed and row['uncued_answer'] != row['cue_letter']:
        return 'wrong_to_cue'
    if not followed and unc_correct and row['cued_answer'] == row['correct_letter']:
        return 'resister'
    return 'other'


def load_cell(run, rows_file=None):
    design = json.loads((run / 'design.json').read_text())
    review_path = run / 'review_approved.csv'
    from e47_experiment import approved_annotations
    review = approved_annotations(run, design, review_path)
    rows_path = resolved_rows_artifact(
        run, design, review_path, explicit=rows_file)
    records = []
    for r in read_csv(rows_path):
        if not r['condition'].startswith('disagree_'):
            continue
        initial = review_cue_acknowledgment(review.get(r['id']))
        why = review_cue_acknowledgment(review.get(r.get('why_id') or ''))
        rec = {
            'id': r['id'], 'source_id': r['source_id'],
            'cue_type': r['cue_type'] or r['condition'].split('_', 1)[1],
            'cue_letter': r['cue_letter'], 'uncued_answer': r['uncued_answer'],
            'cued_answer': r['cued_answer'], 'correct_letter': r['correct_letter'],
            'switch': int(float(r['switch_to_cue'])),
            'initial': None if initial is None else int(initial),
            'why': None if why is None else int(why),
            'pre': [num(r.get(n)) for n in PRE_NAMES + META_NAMES],
        }
        rec['uncued_correct'] = int(rec['uncued_answer'] == rec['correct_letter'])
        rec['stratum'] = stratum(rec)
        rec['why_new'] = (rec['why'] if rec['initial'] == 0 else None)
        records.append(rec)
    return {
        'run': run, 'rows_file': rows_path.name, 'design': design,
        'model': design.get('model', run.name), 'task': design.get('task', '--'),
        'records': records,
    }


def resolve_layers(cell, specification):
    if specification != 'auto':
        return [int(value) for value in specification.split(',') if value]
    from e47_design import read
    manifest = read(cell['run'] / 'capture.pkl')
    sites = manifest.get('layer_metadata', {}).get('layer_sites') or []
    if not sites:
        raise ValueError('capture manifest has no layer-site metadata')
    maximum = len(sites) - 1
    frozen = int(cell['design'].get('analysis', {}).get('layer', maximum))
    return sorted(set([round(maximum * fraction) for fraction in (.25, .5, .75)]
                      + [min(maximum, frozen)]))


# --------------------------------------------------------------------------
# 1. strata and the pre-treatment override moderator
# --------------------------------------------------------------------------

def rate(rows, key):
    rows = [r for r in rows if r[key] is not None]
    k = sum(r[key] for r in rows)
    return {'k': k, 'n': len(rows), 'rate': k / len(rows) if rows else None}


def fisher_2x2(a, b, c, d):
    from scipy.stats import fisher_exact
    if min(a + b, c + d, a + c, b + d) == 0:
        return None, None
    odds, p = fisher_exact([[a, b], [c, d]])
    return float(odds), float(p)


def two_by_two(rows, key, exposed):
    """(a, b, c, d) = exposed pos/neg, unexposed pos/neg, over labeled rows."""
    rows = [r for r in rows if r[key] is not None]
    a = sum(1 for r in rows if exposed(r) and r[key])
    b = sum(1 for r in rows if exposed(r) and not r[key])
    c = sum(1 for r in rows if not exposed(r) and r[key])
    d = sum(1 for r in rows if not exposed(r) and not r[key])
    return a, b, c, d


def mantel_haenszel(tables):
    """Pooled OR and CMH chi-square (1 df, continuity-corrected) over strata."""
    from scipy.stats import chi2
    num_or = den_or = 0.0
    sum_a = sum_e = sum_v = 0.0
    used = 0
    for a, b, c, d in tables:
        n = a + b + c + d
        if n < 2 or (a + b) == 0 or (c + d) == 0 or (a + c) == 0 or (b + d) == 0:
            continue
        used += 1
        num_or += a * d / n
        den_or += b * c / n
        sum_a += a
        sum_e += (a + b) * (a + c) / n
        sum_v += (a + b) * (c + d) * (a + c) * (b + d) / (n * n * (n - 1))
    if not used or sum_v == 0:
        return {'strata_used': used, 'odds_ratio': None, 'p': None}
    stat = (abs(sum_a - sum_e) - 0.5) ** 2 / sum_v
    return {'strata_used': used,
            'odds_ratio': (num_or / den_or) if den_or else math.inf,
            'cmh_chi2': stat, 'p': float(chi2.sf(stat, 1))}


def strata_report(cells):
    out, pooled_tables = [], {}
    for cell in cells:
        for cue_type in sorted({r['cue_type'] for r in cell['records']}):
            rows = [r for r in cell['records'] if r['cue_type'] == cue_type]
            for target in ('initial', 'why'):
                entry = {'model': cell['model'], 'task': cell['task'],
                         'run': cell['run'].name, 'cue_type': cue_type,
                         'target': target,
                         'all': rate(rows, target),
                         'uncued_correct': rate([r for r in rows if r['uncued_correct']], target),
                         'uncued_wrong': rate([r for r in rows if not r['uncued_correct']], target)}
                for s in STRATA:
                    entry[s] = rate([r for r in rows if r['stratum'] == s], target)
                # primary (pre-treatment) contrast on all disagreement trials
                t = two_by_two(rows, target, lambda r: r['uncued_correct'] == 1)
                entry['uncued_correct_or'], entry['uncued_correct_p'] = fisher_2x2(*t)
                pooled_tables.setdefault((cue_type, target, 'uncued_correct'), []).append(t)
                # secondary (post-treatment) contrast among switchers
                sw = [r for r in rows if r['stratum'] in ('flipper', 'wrong_to_cue')]
                t = two_by_two(sw, target, lambda r: r['stratum'] == 'flipper')
                entry['flipper_vs_wrong_to_cue_or'], entry['flipper_vs_wrong_to_cue_p'] = fisher_2x2(*t)
                pooled_tables.setdefault((cue_type, target, 'flipper_vs_wrong_to_cue'), []).append(t)
                out.append(entry)
    pooled = [{'cue_type': ct, 'target': tg, 'contrast': name, **mantel_haenszel(tabs)}
              for (ct, tg, name), tabs in sorted(pooled_tables.items())]
    return out, pooled


# --------------------------------------------------------------------------
# 2-3. prospective readers on censored checkpoints
# --------------------------------------------------------------------------

def _scaled(train, test):
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler().fit(train)
    return scaler.transform(train), scaler.transform(test)


def _state_model(c, k):
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    return make_pipeline(StandardScaler(), PCA(k, random_state=0),
                         LogisticRegression(C=c, max_iter=3000))


def _state_scores(S, y, groups, train, test, seed, k):
    """Inner-CV-tuned state model with probability-scale meta-features."""
    from sklearn.metrics import roc_auc_score
    inner = list(splits(groups[train], 5, seed + 101))
    best_c, best_auc, best_oof = C_GRID[0], -1.0, None
    for c in C_GRID:
        oof = np.zeros(len(train))
        for itr, ite in inner:
            if len(np.unique(y[train][itr])) < 2:
                oof[ite] = float(np.mean(y[train][itr]))
                continue
            kk = max(1, min(k, len(itr) - 1))
            oof[ite] = _state_model(c, kk).fit(S[train][itr], y[train][itr]) \
                .predict_proba(S[train][ite])[:, 1]
        auc = roc_auc_score(y[train], oof) if len(np.unique(y[train])) == 2 else 0.5
        if auc > best_auc:
            best_c, best_auc, best_oof = c, auc, oof
    kk = max(1, min(k, len(train) - 1))
    test_probability = _state_model(best_c, kk).fit(
        S[train], y[train]).predict_proba(S[test])[:, 1]
    return best_oof, test_probability, best_c


def _paper_joint(P, S, y, train, test, seed):
    """The paper's estimator: pre + 32 unsupervised PCs, one LR at C=1."""
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    a, b = _scaled(P[train], P[test])
    sa, sb = _scaled(S[train], S[test])
    k = max(1, min(32, len(train) - 1, S.shape[1]))
    pca = PCA(k, svd_solver='randomized', random_state=seed).fit(sa)
    clf = LogisticRegression(C=1., max_iter=2000, solver='liblinear', random_state=seed)
    clf.fit(np.hstack([a, pca.transform(sa)]), y[train])
    return clf.predict_proba(np.hstack([b, pca.transform(sb)]))[:, 1]


def oof_readers(P, S, y, groups, seeds, k):
    """Seed-averaged OOF probabilities for pre / state / stacked / paper_joint."""
    from sklearn.linear_model import LogisticRegression
    out = {name: np.zeros(len(y)) for name in ('pre', 'state', 'stacked', 'paper_joint')}
    chosen_c = []
    for seed in seeds:
        for train, test in splits(groups, 5, seed):
            if len(np.unique(y[train])) < 2:
                for name in out:
                    out[name][test] += prior(y[train], 2)[1] / len(seeds)
                continue
            a, b = _scaled(P[train], P[test])
            pre = LogisticRegression(C=1., max_iter=2000, solver='liblinear',
                                     random_state=seed).fit(a, y[train])
            out['pre'][test] += pre.predict_proba(b)[:, 1] / len(seeds)
            oof, test_probability, c = _state_scores(
                S, y, groups, train, test, seed, k)
            chosen_c.append(c)
            out['state'][test] += test_probability / len(seeds)
            stack = LogisticRegression(C=1., max_iter=2000, solver='liblinear',
                                       random_state=seed)
            stack.fit(np.column_stack([a, oof]), y[train])
            out['stacked'][test] += stack.predict_proba(
                np.column_stack([b, test_probability]))[:, 1] / len(seeds)
            out['paper_joint'][test] += _paper_joint(P, S, y, train, test, seed) / len(seeds)
    return out, chosen_c


def bootstrap_diff(y, p1, p0, groups, n_boot, seed):
    """Item-cluster bootstrap CI of AUC(p1) - AUC(p0)."""
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(seed)
    unique = np.unique(groups)
    index = {g: np.flatnonzero(groups == g) for g in unique}
    diffs = []
    for _ in range(n_boot):
        take = np.concatenate([index[g] for g in rng.choice(unique, len(unique))])
        if len(np.unique(y[take])) < 2:
            continue
        diffs.append(roc_auc_score(y[take], p1[take]) - roc_auc_score(y[take], p0[take]))
    if not diffs:
        return None
    return [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))]


def readers_report(cell, layers, anchors, seeds, k, n_boot, min_pos):
    from sklearn.metrics import roc_auc_score
    from e47_capture import load_captures
    records = [r for r in cell['records'] if all(v is not None for v in r['pre'])]
    shards, _ = load_captures(cell['run'], cell['design'], layers=layers,
                              ids=[r['id'] for r in records])
    out = []
    populations = {
        'all_disagree': lambda r: True,
        'switchers': lambda r: r['switch'] == 1,
    }
    for cue_type in sorted({r['cue_type'] for r in records}):
        censor = CENSOR_RE.get(cue_type)
        for pop_name, keep in populations.items():
            for target in ('initial', 'why', 'why_new'):
                base = [r for r in records if r['cue_type'] == cue_type and keep(r)
                        and r[target] is not None]
                for anchor in anchors:
                    rows = []
                    for r in base:
                        shard = shards[r['id']]
                        i = next((j for j, cp in enumerate(shard['rows'])
                                  if cp['anchor'] == anchor), None)
                        if i is None:
                            continue
                        # censor only the initial target: the why response
                        # always follows every initial checkpoint
                        if (target == 'initial' and anchor != 'prompt' and censor
                                and censor.search(shard['rows'][i]['prefix'])):
                            continue
                        rows.append((r, i))
                    y = np.array([r[target] for r, _ in rows], int)
                    if y.sum() < min_pos or len(y) - y.sum() < min_pos:
                        out.append({'model': cell['model'], 'task': cell['task'],
                                    'cue_type': cue_type, 'population': pop_name,
                                    'target': target, 'anchor': anchor,
                                    'n': int(len(y)), 'pos': int(y.sum()),
                                    'status': 'too_few_labels'})
                        continue
                    P = np.array([r['pre'] for r, _ in rows], float)
                    groups = np.array([r['source_id'] for r, _ in rows])
                    for layer in layers:
                        S = np.stack([shards[r['id']]['states_by_layer'][layer][i]
                                      for r, i in rows]).astype(np.float32)
                        probs, cs = oof_readers(P, S, y, groups, seeds, k)
                        auc = {name: float(roc_auc_score(y, p)) for name, p in probs.items()}
                        out.append({
                            'model': cell['model'], 'task': cell['task'],
                            'cue_type': cue_type, 'population': pop_name,
                            'target': target, 'anchor': anchor, 'layer': layer,
                            'n': int(len(y)), 'pos': int(y.sum()), 'status': 'fit',
                            **{f'auc_{name}': v for name, v in auc.items()},
                            'increment_stacked': auc['stacked'] - auc['pre'],
                            'increment_stacked_ci95': bootstrap_diff(
                                y, probs['stacked'], probs['pre'], groups, n_boot, 47),
                            'increment_paper_joint': auc['paper_joint'] - auc['pre'],
                            'state_C_mode': max(set(cs), key=cs.count) if cs else None,
                            'censored': target == 'initial' and anchor != 'prompt'})
    return out


# --------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------

def pct(x):
    return '--' if x is None else f'{100 * x:.1f}%'


def cell_str(entry):
    return f"{entry['k']}/{entry['n']} ({pct(entry['rate'])})"


def print_strata(strata, pooled):
    print('\nACKNOWLEDGMENT BY PRE-TREATMENT CORRECTNESS AND STRATUM '
          '(all valid disagreement trials)')
    header = ('model', 'task', 'cue', 'target', 'all', 'uncued correct',
              'uncued wrong', 'OR (p)', 'flipper', 'wrong_to_cue', 'resister',
              'flip/w2c OR (p)')
    rows = []
    for e in strata:
        rows.append((e['model'], e['task'], e['cue_type'], e['target'],
                     cell_str(e['all']), cell_str(e['uncued_correct']),
                     cell_str(e['uncued_wrong']),
                     _or(e['uncued_correct_or'], e['uncued_correct_p']),
                     cell_str(e['flipper']), cell_str(e['wrong_to_cue']),
                     cell_str(e['resister']),
                     _or(e['flipper_vs_wrong_to_cue_or'], e['flipper_vs_wrong_to_cue_p'])))
    widths = [max(len(str(x)) for x in col) for col in zip(header, *rows)]
    for line in [header] + [tuple('-' * w for w in widths)] + rows:
        print('  ' + '  '.join(str(x).ljust(w) for x, w in zip(line, widths)))
    print('  Zero-acknowledgment cells are retained. OR = Fisher odds ratio.')
    print('  "uncued correct" is fixed before treatment (clean moderator); the')
    print('  flipper vs wrong_to_cue contrast conditions on switching (post-treatment).')
    print('\nPOOLED OVER CELLS (Mantel-Haenszel, cells as strata)')
    for p in pooled:
        print(f"  {p['cue_type']:<10} {p['target']:<8} {p['contrast']:<24} "
              f"cells={p['strata_used']}  OR={_f(p['odds_ratio'])}  p={_f(p['p'], 4)}")


def _f(x, d=3):
    return '--' if x is None else f'{x:.{d}f}'


def _or(odds, p):
    return '--' if odds is None else f'{odds:.2f} ({p:.3g})'


def print_readers(readers):
    fit = [r for r in readers if r['status'] == 'fit']
    if not fit:
        print('\nno reader fits (too few labels everywhere)')
        return
    print('\nPROSPECTIVE READERS (best layer per row; initial-target checkpoints censored)')
    best = {}
    for r in fit:
        key = (r['model'], r['task'], r['cue_type'], r['population'], r['target'], r['anchor'])
        if key not in best or r['auc_stacked'] > best[key]['auc_stacked']:
            best[key] = r
    header = ('model', 'task', 'cue', 'population', 'target', 'anchor', 'n', 'pos',
              'layer', 'pre', 'state', 'stacked', 'incr [95% CI]', 'paper joint incr')
    rows = []
    for key in sorted(best, key=lambda k: (k[:5], ANCHORS.index(k[5]))):
        r = best[key]
        ci = r['increment_stacked_ci95']
        rows.append((*key, r['n'], r['pos'], r['layer'], _f(r['auc_pre']),
                     _f(r['auc_state']), _f(r['auc_stacked']),
                     f"{r['increment_stacked']:+.3f} "
                     + (f'[{ci[0]:+.3f}, {ci[1]:+.3f}]' if ci else '[--]'),
                     f"{r['increment_paper_joint']:+.3f}"))
    widths = [max(len(str(x)) for x in col) for col in zip(header, *rows)]
    for line in [header] + [tuple('-' * w for w in widths)] + rows:
        print('  ' + '  '.join(str(x).ljust(w) for x, w in zip(line, widths)))
    print('  Picking the best layer per row is optimistic; the per-layer table is in')
    print('  the readers CSV. Treat a layer chosen here as a hypothesis for a frozen')
    print('  confirmation run, not as the confirmatory estimate.')


def write_csv(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with Path(path).open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: (json.dumps(v) if isinstance(v, (dict, list)) else v)
                             for k, v in r.items()})


def flatten_strata(strata):
    flat = []
    for e in strata:
        row = {k: v for k, v in e.items() if not isinstance(v, dict)}
        for k, v in e.items():
            if isinstance(v, dict):
                row.update({f'{k}_k': v['k'], f'{k}_n': v['n'], f'{k}_rate': v['rate']})
        flat.append(row)
    return flat


def main(argv=None):
    warnings.filterwarnings('ignore')
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--runs', nargs='+', metavar='DIR',
                    help='run directories; default run_dirs.DEFAULT_RUN_DIRS')
    ap.add_argument('--out', type=Path, help='directory for JSON/CSV outputs')
    ap.add_argument('--readers', action='store_true',
                    help='also fit the prospective readers (needs capture shards)')
    ap.add_argument('--layers', default='auto',
                    help='comma-separated capture layer-site indices, or auto '
                         '(25%%, 50%%, 75%% and the run\'s frozen primary layer)')
    ap.add_argument('--rows-file', type=Path,
                    help='explicit what-causes-switch rows CSV; valid only with one --runs entry')
    ap.add_argument('--anchors', default='prompt,frac25,frac50,frac75,lead40,lead20,lead10,lead5')
    ap.add_argument('--seeds', type=int, default=3, help='outer CV repetitions')
    ap.add_argument('--pca', type=int, default=32)
    ap.add_argument('--bootstrap', type=int, default=500)
    ap.add_argument('--min-pos', type=int, default=10,
                    help='minimum positives and negatives to fit a reader')
    a = ap.parse_args(argv)

    runs = [resolve_run(r) for r in a.runs] if a.runs else list(DEFAULT_RUN_DIRS)
    if a.rows_file and len(runs) != 1:
        ap.error('--rows-file requires exactly one --runs directory')
    cells, skipped = [], []
    for run in runs:
        try:
            cells.append(load_cell(run, a.rows_file))
        except (OSError, ValueError, KeyError) as exc:
            skipped.append(f'{run}: {exc}')
    if not cells:
        ap.error('no usable runs:\n  ' + '\n  '.join(skipped))

    strata, pooled = strata_report(cells)
    print('\nE47 DETERMINANTS OF CUE ACKNOWLEDGMENT')
    print('=' * 120)
    print_strata(strata, pooled)

    readers = []
    if a.readers:
        anchors = [x for x in a.anchors.split(',') if x]
        for cell in cells:
            print(f"\nfitting readers: {cell['model']} / {cell['task']} ...", flush=True)
            try:
                layers = resolve_layers(cell, a.layers)
                print(f'  resolved layer sites: {layers}', flush=True)
                readers += readers_report(cell, layers, anchors, list(range(a.seeds)),
                                          a.pca, a.bootstrap, a.min_pos)
            except (OSError, ValueError) as exc:
                skipped.append(f"{cell['run']}: readers: {exc}")
        print_readers(readers)

    if skipped:
        print('\nskipped:')
        for s in skipped:
            print(f'  {s}')
    if a.out:
        a.out.mkdir(parents=True, exist_ok=True)
        (a.out / 'disclosure_determinants.json').write_text(json.dumps({
            'schema': SCHEMA, 'runs': [str(c['run']) for c in cells],
            'rows_files': {c['run'].name: c['rows_file'] for c in cells},
            'strata': strata, 'pooled': pooled, 'readers': readers,
            'skipped': skipped}, indent=2, default=str))
        write_csv(a.out / 'disclosure_determinants.strata.csv', flatten_strata(strata))
        write_csv(a.out / 'disclosure_determinants.readers.csv', readers)
        print(f'\nwrote {a.out}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
