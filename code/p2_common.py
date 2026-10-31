#!/usr/bin/env python3
# =============================================================================
# p2_common — shared estimator core for the Paper 2 (observer hierarchy) suite
#
# Everything in the "ESTIMATOR CORE" section is copied VERBATIM from
# e2_discrete_recoverability.py / E6_prompt_augmented_decoder.py so that every
# Paper 2 number is produced by the *identical* certified-lower-bound machinery
# used in Paper 1 (calibrated decoders -> CE upper-bounds H(Z|.), so
# Î_LB = H(Z) − CE ≤ I(Z;.) always).  Do not "improve" these functions here;
# if E2 changes, re-copy.
#
# The Paper 2 scripts (E7–E10) import from this module instead of re-copying,
# because Paper 2 adds several new observer classes and copy-drift across five
# files is how estimator bugs happen.
#
# CONTENTS
#   1. Estimator core (verbatim): entropy, calibration, decoder_oof,
#      plugin_token_decoder_oof, ilb_rho, bootstrap_rho, rms_pca_features,
#      DECODERS ladder, channel_best, guard_target.
#   2. Trace parsing + per-position targets (verbatim from E2): needed by the
#      LLM-monitor observer (E8), which elicits *named* reasoning values.
#   3. E2 cache access: load_pairs()/rebuild_cluster_labels() reproduce E2's
#      content filter and (deterministic) KMeans so cluster labels are
#      IDENTICAL to the ones behind Paper 1's tables.
#   4. Small utilities shared by E7–E10 (JSON/pickle io, fano_floor).
# =============================================================================

from __future__ import annotations
import os, re, json, pickle, warnings
import numpy as np

warnings.filterwarnings('ignore')

from sklearn.cluster import KMeans
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

# ── E2-matching constants (keep in sync with e2_discrete_recoverability.py) ──
SEED               = 42
K_SWEEP            = [16, 32, 64, 128]
PCA_DIM            = 512
EMB_PCA_DIM        = 64
WINDOW             = 2
N_FOLDS            = 5
N_BOOT             = 300
KMEANS_NINIT       = 4
MAX_DECODER_TOKENS = 12000
SMOOTH_ALPHA       = 1.0
POS_CONTROL_MIN    = 0.30
MIN_TARGET_TOKENS  = 1500
K_ARITH_BINS       = 12
CALIBRATE          = True
CAL_TEMPS          = (0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0)
CAL_LAMBDAS        = (0.0, 0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0)
LOGREG_C           = 0.1
LN2                = np.log(2.0)

rng_global = np.random.default_rng(SEED)

TASKS = ('arithmetic', 'deduction', 'counting')

# Deduction entity vocabulary — verbatim from E2 (needed to parse/elicit the
# "current entity" reasoning value).
ENTITIES = ["wugs", "blickets", "daxes", "feps", "kiki", "mav", "zorp", "tove",
            "blorb", "snerg", "plim", "veck", "quob", "nilt", "yorp", "grix",
            "tald", "prunt", "sliv", "bront", "creel", "phog", "zant", "klev",
            "moop", "dreff", "spave", "tunk", "wrell", "flob"]
ENTITY_INDEX = {w: i for i, w in enumerate(ENTITIES)}


# =============================================================================
# 1. ESTIMATOR CORE — copied verbatim from E2/E6 (do not edit; re-copy on change)
# =============================================================================
# =============================================================================
# TOKENIZER LOADING
# =============================================================================
def assert_no_local_shadow(hf):
    """Fail loudly if a local file/dir shadows the HuggingFace repo id.

    from_pretrained() resolves a local path BEFORE querying the Hub, so a
    directory or symlink named e.g. 'gpt2-medium' in the working directory
    silently hijacks the load. The failure surfaces far downstream and is
    thoroughly misleading -- "Unrecognized model in gpt2-medium. Should have a
    `model_type` key in its config.json" -- because transformers is faithfully
    reporting that YOUR FOLDER is not a model checkpoint.

    Nothing in this codebase needs such a directory: every cache and result
    path is built from the models.py KEY (model_label / MODEL_KEY), never from
    the HF id. So a name collision here is always accidental.
    """
    import os
    if not os.path.exists(hf):
        return
    kind = ('symlink -> ' + os.path.realpath(hf)) if os.path.islink(hf) else \
           ('directory' if os.path.isdir(hf) else 'file')
    raise RuntimeError(
        f"local path {hf!r} shadows the HuggingFace repo id of the same name.\n"
        f"  found : ./{hf}  ({kind})\n"
        f"  cwd   : {os.getcwd()}\n"
        f"transformers loads the local path instead of the Hub checkpoint, "
        f"which fails later with a confusing config/tokenizer error.\n"
        f"Nothing here requires that folder -- cache and result paths come from "
        f"the models.py key, not the HF id. Remove or rename it:\n"
        f"    rm ./{hf}          # if it is a symlink\n"
        f"    mv ./{hf} ./{hf}_data   # if it holds your results")


# Substrings that mark a TOKENIZER-BACKEND failure, i.e. one the use_fast=False
# fallback below could plausibly fix. Anything else -- auth, network, a typo in
# the repo id -- must propagate unchanged rather than be retried.
#
# This list was referenced before it existed, so every non-recoverable failure
# raised `NameError: _TOK_RECOVERABLE is not defined` from inside the handler
# and buried the real cause. A gated-repo 401 on meta-llama surfaced as a
# NameError, which is the least useful possible message.
_TOK_RECOVERABLE = (
    'tokenizer', 'convert_slow', 'sentencepiece', 'tiktoken', 'protobuf',
    'use_fast', 'backend', 'conversion',
)


def load_tokenizer(hf, **kwargs):
    """AutoTokenizer.from_pretrained with two fallbacks, most faithful first.

      1. the model's own tokenizer
      2. use_fast=False              (no-op where slow tokenizers were removed;
                                      retained for older transformers installs)

    Anything past that raises with the three things actually worth checking.
    It deliberately does NOT substitute a different repo's tokenizer: silently
    swapping tokenizers in an information-theoretic experiment would corrupt
    every downstream token count.

    Any kwargs are forwarded. Errors that are not tokenizer-backend failures
    (auth, network, typo in the repo id) propagate unchanged.
    """
    from transformers import AutoTokenizer
    assert_no_local_shadow(hf)

    def _try(name, **kw):
        # trust_remote_code defaults on, but a caller that passes it explicitly
        # (E2 exposes --no-trust-remote-code) wins. Merging into one dict rather
        # than passing it positionally avoids a duplicate-keyword TypeError.
        opts = {'trust_remote_code': True, **kwargs, **kw}
        return AutoTokenizer.from_pretrained(name, **opts)

    try:
        return _try(hf)
    except Exception as e1:
        if not any(k in str(e1).lower() for k in _TOK_RECOVERABLE):
            raise
        try:
            tok = _try(hf, use_fast=False)
            print(f"  ! {hf}: no fast tokenizer available; using the slow one.")
            return tok
        except Exception:
            pass
        raise RuntimeError(
            f"could not load a tokenizer for {hf!r}.\n"
            f"  underlying error: {e1}\n"
            f"Check, in order:\n"
            f"  1. a local path shadowing the repo id   ->  ls -la ./{hf}\n"
            f"  2. a partial or corrupt download        ->  "
            f"rm -rf ~/.cache/huggingface/hub/models--{hf.replace('/', '--')}\n"
            f"  3. a missing conversion backend         ->  pip install tiktoken "
            f"sentencepiece") from e1


def entropy_bits_miller_madow(labels, K):
    """Plug-in entropy (bits) and Miller–Madow-corrected entropy (bits)."""
    counts = np.bincount(labels, minlength=K).astype(np.float64)
    N = counts.sum()
    p = counts[counts > 0] / N
    H_raw = float(-(p * np.log2(p)).sum())
    m_obs = int((counts > 0).sum())
    H_mm = H_raw + (m_obs - 1) / (2.0 * N) / LN2
    return H_raw, H_mm


def _inner_fit_val(tr, z, groups, n=4):
    """Grouped fit/val split of an outer training fold (local indices into `tr`)."""
    g = groups[tr]
    nu = len(np.unique(g))
    if nu < 2:
        idx = np.arange(len(tr))
        return idx, idx                       # degenerate: calibrate on train itself
    f, v = next(GroupKFold(n_splits=min(n, nu)).split(tr, z[tr], g))
    return f, v


def _apply_cal(P, T, lam, marg):
    """Temperature-scale P then mix with the marginal."""
    logP = np.log(np.clip(P, 1e-12, 1.0))
    pT = np.exp(logP / T)
    pT /= pT.sum(axis=1, keepdims=True)
    return (1.0 - lam) * pT + lam * marg[None, :]


def _grid_calibrate(P_val, z_val, marg):
    """Pick (T, λ) minimising validation CE (λ=1 ⇒ marginal ⇒ CE=H, never vacuous)."""
    best, best_ce = (1.0, 1.0), np.inf
    idx = np.arange(len(z_val))
    for T in CAL_TEMPS:
        for lam in CAL_LAMBDAS:
            pc = _apply_cal(P_val, T, lam, marg)
            ce = float(np.mean(-np.log2(np.clip(pc[idx, z_val], 1e-12, 1.0))))
            if ce < best_ce:
                best_ce, best = ce, (T, lam)
    return best


def _marg(z_fit, K):
    m = np.bincount(z_fit, minlength=K).astype(np.float64) + 0.5
    return m / m.sum()


def plugin_token_decoder_oof(tok, z, groups, K, n_folds=N_FOLDS, alpha=SMOOTH_ALPHA,
                             calibrate=CALIBRATE):
    """Single-token channel: calibrated plug-in decoder (grouped CV by example)."""
    def cond_table(tok_a, z_a):
        cond = {}
        for c, zz in zip(tok_a, z_a):
            v = cond.get(c)
            if v is None:
                v = np.zeros(K, dtype=np.float64); cond[c] = v
            v[zz] += 1.0
        return cond

    def P_for(tok_a, cond, marg):
        P = np.empty((len(tok_a), K), dtype=np.float64)
        for r, c in enumerate(tok_a):
            v = cond.get(int(c))
            P[r] = marg if v is None else (v + alpha * marg) / (v.sum() + alpha)
        return P

    marg = _marg(z, K)
    p_true = np.zeros(len(z), dtype=np.float64)
    for tr, te in GroupKFold(n_splits=n_folds).split(tok, z, groups):
        f, v = _inner_fit_val(tr, z, groups)
        cond = cond_table(tok[tr][f], z[tr][f])
        if calibrate:
            P_val = P_for(tok[tr][v], cond, marg)
            T, lam = _grid_calibrate(P_val, z[tr][v], marg)
        else:
            T, lam = 1.0, 0.0
        P_te = _apply_cal(P_for(tok[te], cond, marg), T, lam, marg)
        p_true[te] = P_te[np.arange(len(te)), z[te]]
    return np.clip(p_true, 1e-12, 1.0)


def _full_proba(clf, X, K):
    """Map a classifier's predict_proba onto the full K-class index, renormalised."""
    proba = clf.predict_proba(X)
    full = np.full((X.shape[0], K), 1e-9, dtype=np.float64)
    for j, c in enumerate(clf.classes_):
        full[:, int(c)] = proba[:, j]
    return full / full.sum(axis=1, keepdims=True)


def _oof_one_fold(X, z, groups, K, clf_factory, marg, calibrate, fold_marg, tr, te):
    """One outer fold, as a pure function.

    Depends only on its arguments and writes nothing shared, which is what makes
    the folds safe to run concurrently. Every step is deterministic:
      _inner_fit_val   GroupKFold, no RNG
      StandardScaler   closed form
      _grid_calibrate  exhaustive grid over CAL_TEMPS x CAL_LAMBDAS
      clf_factory()    LogisticRegression(lbfgs) is deterministic; MLPClassifier
                       carries random_state=SEED, so it no longer depends on the
                       global numpy RNG and therefore no longer on fold ORDER.
    That last point matters: before the seeding fix, parallelising this would
    have silently changed the MLP results, because each fold drew its init from
    a shared stream whose state depended on how many folds had run before it."""
    f, v = _inner_fit_val(tr, z, groups)
    m = _marg(z[tr][f], K) if fold_marg else marg
    Xtr, ztr = X[tr], z[tr]
    sc = StandardScaler().fit(Xtr[f])
    clf = clf_factory()
    clf.fit(sc.transform(Xtr[f]), ztr[f])
    if calibrate:
        P_val = _full_proba(clf, sc.transform(Xtr[v]), K)
        T, lam = _grid_calibrate(P_val, ztr[v], m)
    else:
        T, lam = 1.0, 0.0
    P_te = _apply_cal(_full_proba(clf, sc.transform(X[te]), K), T, lam, m)
    return te, P_te[np.arange(len(te)), z[te]]


def resolve_n_jobs(n_jobs=None, cap=N_FOLDS):
    """How many folds to fit at once.

    None  -> read E2_N_JOBS, else auto (min(cap, cpu_count)) when joblib is
             importable, else 1.
    1     -> serial, the historical code path. Use it for the equivalence check.
    """
    if n_jobs is None:
        env = os.environ.get('E2_N_JOBS', '').strip()
        n_jobs = int(env) if env else 0
    if n_jobs == 1:
        return 1
    if n_jobs and n_jobs > 1:
        return n_jobs
    try:
        import joblib  # noqa: F401
    except ImportError:
        return 1
    return max(1, min(cap, os.cpu_count() or 1))


def decoder_oof(X, z, groups, K, clf_factory, n_folds=N_FOLDS, calibrate=CALIBRATE,
                fold_marg=False, n_jobs=None):
    """Grouped-CV OOF P̂(Z=z_i | features), temperature + marginal back-off
    calibrated on an inner val split (so CE ≤ H(Z) always).

    fold_marg=False reproduces E2/Paper 1 exactly (marginal computed on ALL
    labels — a small test-fold leak into the null/back-off prior).
    fold_marg=True computes the marginal from the training fold only (the
    strict Ethayarajh-recipe null); use for the V-information protocol (E18).

    The outer folds are fitted concurrently when joblib is available. Each fold
    writes only its own test slice of p_true, so the result is EXACTLY equal to
    the serial version, not approximately -- see _oof_one_fold. Set E2_N_JOBS=1
    (or n_jobs=1) to force the serial path and verify that.

    This is where E2's Stage B time goes: ~465 decoder fits per model, of which
    roughly 180 are a 128-class multinomial logistic on 512-d state features."""
    marg = _marg(z, K)
    p_true = np.zeros(len(z), dtype=np.float64)
    splits = list(GroupKFold(n_splits=n_folds).split(X, z, groups))
    jobs = resolve_n_jobs(n_jobs)
    args = (X, z, groups, K, clf_factory, marg, calibrate, fold_marg)
    if jobs == 1:
        out = [_oof_one_fold(*args, tr, te) for tr, te in splits]
    else:
        from joblib import Parallel, delayed
        # loky memory-maps large arrays instead of pickling a copy per worker,
        # and clamps each worker's BLAS thread pool so 5 folds x N BLAS threads
        # does not oversubscribe the box.
        out = Parallel(n_jobs=jobs)(
            delayed(_oof_one_fold)(*args, tr, te) for tr, te in splits)
    for te, p in out:
        p_true[te] = p
    return np.clip(p_true, 1e-12, 1.0)


def calibrated_oof_from_proba(P, z, groups, K, n_folds=N_FOLDS, fold_marg=False):
    """Certified bound for an observer that PRODUCES ITS OWN distribution P̂(Z|·)
    (e.g. the prompted LLM monitor, E8): no decoder is trained, but the same
    inner-split temperature + marginal back-off is applied per outer fold so the
    result is comparable to decoder_oof (and Î_LB stays a valid, non-vacuous
    lower bound regardless of monitor miscalibration).  P: (N, K) row-stochastic.
    This is the in-context PVI estimator in the sense of Lu et al.; fold_marg
    as in decoder_oof."""
    marg = _marg(z, K)
    p_true = np.zeros(len(z), dtype=np.float64)
    for tr, te in GroupKFold(n_splits=n_folds).split(P, z, groups):
        f, v = _inner_fit_val(tr, z, groups)
        if fold_marg:
            marg = _marg(z[tr][f], K)
        T, lam = _grid_calibrate(P[tr][v], z[tr][v], marg)
        P_te = _apply_cal(P[te], T, lam, marg)
        p_true[te] = P_te[np.arange(len(te)), z[te]]
    return np.clip(p_true, 1e-12, 1.0)


def ilb_rho(H_bits, p_true):
    ce = float(np.mean(-np.log2(np.clip(p_true, 1e-12, 1.0))))
    ilb = max(0.0, H_bits - ce)
    rho = ilb / H_bits if H_bits > 0 else 0.0
    return ce, ilb, rho


def bootstrap_rho(z, p_true, groups, K, n_boot=N_BOOT, seed=SEED):
    """Bootstrap over examples (CIs over examples, not tokens)."""
    rng = np.random.default_rng(seed)
    ex_ids = np.unique(groups)
    by_ex = {e: np.where(groups == e)[0] for e in ex_ids}
    rhos = np.empty(n_boot)
    for b in range(n_boot):
        samp = rng.choice(ex_ids, size=len(ex_ids), replace=True)
        idx = np.concatenate([by_ex[e] for e in samp])
        _, H_mm = entropy_bits_miller_madow(z[idx], K)
        _, _, rho = ilb_rho(H_mm, p_true[idx])
        rhos[b] = rho
    return float(np.percentile(rhos, 2.5)), float(np.percentile(rhos, 97.5))


def rms_pca_features(resid, dim):
    """Center + RMS-normalise then top-`dim` PCA projection (verbatim from E2)."""
    X = resid.astype(np.float64)
    X = X - X.mean(axis=0)
    rms = float(np.sqrt((X ** 2).mean())) + 1e-8
    X = X / rms
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    dim = min(dim, Vt.shape[0])
    return X @ Vt[:dim].T


# random_state=SEED: MLPClassifier draws its initial weights and its
# early-stopping validation split from the global numpy RNG when this is left
# unset, so an MLP-selected cell was reproducible only to about +/-0.01. Measured
# on a re-run of the GPT-2 grid: logistic-selected cells reproduced to 0.0004,
# MLP-selected cells drifted up to 0.0088 -- enough to move a borderline
# CI-separation call, and enough to make "did this code change move the number?"
# unanswerable. LogisticRegression is deterministic here (lbfgs, no subsampling)
# and needs no seed.
DECODERS = [
    ('logistic',    lambda: LogisticRegression(max_iter=1000, C=LOGREG_C)),
    ('mlp_128',     lambda: MLPClassifier(hidden_layer_sizes=(128,),
                                          max_iter=120, early_stopping=True,
                                          random_state=SEED)),
    ('mlp_256_128', lambda: MLPClassifier(hidden_layer_sizes=(256, 128),
                                          max_iter=120, early_stopping=True,
                                          random_state=SEED)),
]


# =============================================================================
# NESTED FEATURE-DIMENSION SELECTION
# =============================================================================
# WHY THIS EXISTS
#
# PCA_DIM=512 is right for the reasoning tasks -- 12,000 rows over 512 dims is
# ~23 rows/dim. It is badly wrong for the world-state query probes, where 2,317
# rows over 512 dims is 4.5 rows/dim and the state decoder overfits hard.
# Measured on ds-r1-qwen-7b L3, rho_state against the modal answer:
#
#     dim     32     48     64    128    256    512   1024
#     rho  0.796  0.830  0.832  0.781  0.741  0.589  0.284
#
# At 512 the state channel reads 0.589; at 64 it reads 0.832. Every negative
# verbalization gap in that run was this artifact, not a property of the model.
#
# The dimension therefore has to be chosen -- and choosing it by looking at the
# table above would be selecting on the outcome, which inflates whichever
# channel gets tuned. It is chosen here by INNER cross-validation on the outer
# training fold only. The test rows of each outer fold never influence either
# the dimension or the projection.
#
# Two properties that matter for the comparison being fair:
#   * every channel gets the same treatment and the same grid, so the state
#     channel cannot be tuned while the text channels are left at a handicap;
#   * the PCA basis is fit per outer training fold rather than once over all
#     rows, removing a mild train/test leak that the fixed-dim path has.
# The cost is |dims| x n_inner extra fits per outer fold.

# `None` means NO truncation -- the raw features, regularised only by the
# decoder's own penalty. It must stay in the grid. PCA is unsupervised, so when
# the target lives in low-variance directions truncation destroys it: on a
# synthetic with the signal in 5 low-variance dims among 900 noise dims, every
# truncated option scored ~0 while the untruncated reader scored 0.585. Without
# an untruncated candidate the selector would have confidently returned zero.
DIM_GRID = (16, 32, 64, 128, 256, 512, None)


def _export_module_path():
    """Put this module's directory on PYTHONPATH for child processes.

    joblib's loky backend starts fresh interpreters and pickles worker
    functions BY REFERENCE, so each child must `import p2_common`. When the
    caller reached this module through a runtime sys.path insert -- which is
    exactly what the Colab runners do via runpy.run_path -- some loky versions
    do not propagate that path, and the run dies with

        BrokenProcessPool: A task has failed to un-serialize
        ModuleNotFoundError: No module named 'p2_common'

    naming neither the real cause nor the fix. Child processes inherit
    os.environ, so exporting the directory here is enough. Idempotent, and it
    only ever adds the directory the caller is already running from.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    cur = os.environ.get("PYTHONPATH", "")
    if here not in cur.split(os.pathsep):
        os.environ["PYTHONPATH"] = (here + os.pathsep + cur) if cur else here


def _fit_pca(Xtr, dim):
    """PCA basis from TRAINING rows only. Returns (project_fn, dim_used).

    dim=None means no projection at all.
    """
    if dim is None:
        return (lambda X: X), int(Xtr.shape[1])
    mu = Xtr.mean(axis=0)
    Xc = Xtr - mu
    rms = float(np.sqrt((Xc ** 2).mean())) + 1e-8
    _, _, Vt = np.linalg.svd(Xc / rms, full_matrices=False)
    d = min(dim, Vt.shape[0])
    B = Vt[:d].T

    def project(X):
        return ((X - mu) / rms) @ B
    return project, d


def _nested_one_fold(X, z, groups, K, clf_factory, marg, calibrate, fold_marg,
                     dims, n_inner, tr, te):
    """One outer fold: pick the dimension inside `tr`, then score `te`."""
    zt, Xt, gt = z[tr], X[tr], groups[tr]
    nu = len(np.unique(gt))
    inner = list(GroupKFold(n_splits=min(n_inner, max(2, nu))).split(Xt, zt, gt))

    scores = {}
    for d in dims:
        ces = []
        for itr, ite in inner:
            if len(np.unique(zt[itr])) < 2:
                continue
            proj, _ = _fit_pca(Xt[itr], d)
            sc = StandardScaler().fit(proj(Xt[itr]))
            clf = clf_factory()
            clf.fit(sc.transform(proj(Xt[itr])), zt[itr])
            P = _full_proba(clf, sc.transform(proj(Xt[ite])), K)
            p = np.clip(P[np.arange(len(ite)), zt[ite]], 1e-12, 1.0)
            ces.append(float(np.mean(-np.log2(p))))
        if ces:
            scores[d] = float(np.mean(ces))
    best = min(scores, key=scores.get) if scores else dims[0]

    # Refit at the chosen dimension on the whole outer-training fold, then apply
    # the SAME calibration path as _oof_one_fold so the two are comparable.
    f, v = _inner_fit_val(tr, z, groups)
    m = _marg(zt[f], K) if fold_marg else marg
    proj, used = _fit_pca(Xt[f], best)
    sc = StandardScaler().fit(proj(Xt[f]))
    clf = clf_factory()
    clf.fit(sc.transform(proj(Xt[f])), zt[f])
    if calibrate:
        T, lam = _grid_calibrate(_full_proba(clf, sc.transform(proj(Xt[v])), K),
                                 zt[v], m)
    else:
        T, lam = 1.0, 0.0
    P_te = _apply_cal(_full_proba(clf, sc.transform(proj(X[te])), K), T, lam, m)
    return te, P_te[np.arange(len(te)), z[te]], int(used)


def nested_dim_oof(X, z, groups, K, clf_factory=None, dims=DIM_GRID,
                   n_folds=N_FOLDS, n_inner=4, calibrate=CALIBRATE,
                   fold_marg=False, n_jobs=None):
    """OOF p_true with the feature dimension chosen inside each training fold.

    Returns (p_true, chosen_dims_per_outer_fold).
    """
    if clf_factory is None:
        clf_factory = DECODERS[0][1]
    dims = tuple(d for d in dims if d is None or d <= X.shape[1]) or (None,)
    marg = _marg(z, K)
    p_true = np.zeros(len(z), dtype=np.float64)
    splits = list(GroupKFold(n_splits=n_folds).split(X, z, groups))
    args = (X, z, groups, K, clf_factory, marg, calibrate, fold_marg, dims,
            n_inner)
    jobs = resolve_n_jobs(n_jobs)
    out = None
    if jobs > 1:
        try:
            from joblib import Parallel, delayed
            _export_module_path()
            out = Parallel(n_jobs=jobs)(
                delayed(_nested_one_fold)(*args, tr, te) for tr, te in splits)
        except Exception as e:                                    # noqa: BLE001
            # Never let a backend problem end a run that has already paid for
            # its GPU capture. Serial is ~4x slower and gives bit-identical
            # results (verified), so falling back is free apart from time.
            print(f"    nested_dim_oof: parallel path failed "
                  f"({type(e).__name__}: {str(e)[:80]}); running serially")
            out = None
    if out is None:
        out = [_nested_one_fold(*args, tr, te) for tr, te in splits]
    chosen = []
    for te, p, d in out:
        p_true[te] = p
        chosen.append(d)
    return np.clip(p_true, 1e-12, 1.0), chosen


def channel_best_nested(features, z_s, g_s, K, H_mm, dims=DIM_GRID):
    """channel_best's signature, with nested dimension selection.

    Only the logistic decoder is offered. The MLP ladder is a second capacity
    knob, and sweeping two capacity knobs against the same inner folds is how a
    selection procedure starts fitting its own validation set.
    """
    p, chosen = nested_dim_oof(features, z_s, g_s, K, dims=dims)
    ce, ilb, rho = ilb_rho(H_mm, p)
    return (p, ilb, rho, ce, f"logistic@d={'/'.join(map(str, chosen))}"), chosen


def channel_best(features, z_s, g_s, K, H_mm, run_ladder=True):
    """Decode Z from `features`; return best (tightest Î_LB) decoder + ladder."""
    p = decoder_oof(features, z_s, g_s, K, DECODERS[0][1])
    ce, ilb, rho = ilb_rho(H_mm, p)
    best = (p, ilb, rho, ce, 'logistic')
    ladder = {'logistic': ilb}
    if run_ladder:
        for name, fac in DECODERS[1:]:
            pc = decoder_oof(features, z_s, g_s, K, fac)
            cec, ilbc, rhoc = ilb_rho(H_mm, pc)
            ladder[name] = ilbc
            if ilbc > best[1]:
                best = (pc, ilbc, rhoc, cec, name)
    return best, ladder


# =============================================================================
# REGIME-INVARIANT TARGET  (cross-condition comparisons only)
# =============================================================================
# Both of the paper's existing targets are defined FROM THE TRACE: the cluster
# target is KMeans over residual states, and the per-position target is parsed
# out of trace events. That is fine within a condition, but it makes cross-REGIME
# comparison invalid -- change the elicitation and both Z and H(Z) change, so
# rho = I_LB / H(Z) has a different denominator in each condition and the two
# numbers are not commensurable. Differencing them would be the easiest thing in
# the paper for a reviewer to attack.
#
# This target is fixed by the PROBLEM, straight out of the example generator, so
# it is identical in every condition by construction.
#
#   arithmetic  final value, quantile-binned            K <= K_ARITH_BINS
#   counting    final count                             K = #distinct counts
#   deduction   CONCLUSION ENTITY, not the answer       K = #distinct entities
#
# The deduction case is a trap worth stating explicitly: generate_deduction sets
# answer="Yes" for every example, because every generated syllogism is valid. An
# answer-level target there has H(Z) = 0 and yields a meaningless rho. The
# conclusion entity is the informative problem-defined quantity.
#
# One label per EXAMPLE (not per position); apply it to every position of that
# example's trace and group by example, exactly as the existing machinery does.


# =============================================================================
# EPISODE BOUNDS — restricting analysis to the assigned problem
# =============================================================================
# E1 stops generation on the NEXT-PROBLEM marker only, but first_completion()
# (e0e, e0f, E4) ends an episode at the answer form OR the next-problem marker,
# whichever comes first. So every cached trace carries the text the model wrote
# AFTER answering, and e0n measured that at 8-87% of tokens depending on the
# model's cot_tokens cap. Those positions are probe positions in E2, so the
# trace decoder is partly reading text with no bearing on the assigned problem,
# and per-example row counts are wildly unequal, which unbalances GroupKFold.
#
# The fix is a row mask, not a regeneration: the pairs cache already holds a
# residual at every position. e0n --emit-bounds writes the episode length per
# example ONCE, with the canonical definition, and these helpers apply it.
# Nothing here changes a default -- callers opt in -- so existing numbers stay
# reproducible and both columns can be reported.

def load_episode_bounds(bounds_dir, slug):
    """{task: {'episode_len': [...], 'completed': [...]}} or None if absent.

    Returns None rather than raising so a caller can degrade to unfiltered
    behaviour with a warning; a missing bounds file is a "you have not run e0n
    yet" condition, not a corrupt-state condition.
    """
    p = os.path.join(bounds_dir, f'e0n_bounds_{slug}.json')
    if not os.path.exists(p):
        return None
    with open(p) as fh:
        return (json.load(fh) or {}).get('tasks')


def episode_row_mask(bounds, task, ex_idx, pos_idx, require_completed=False):
    """Boolean mask over pair rows: keep positions inside the episode.

    ex_idx / pos_idx are the per-row example id and within-trace position that
    the pairs cache already stores, so this is pure indexing.

    require_completed additionally drops every row of any example where no
    completion signal was found. That is a DIFFERENT correction: filler is extra
    data and dropping it is uncontroversial, whereas an unfinished trace is
    missing data and dropping it changes n. Report it as a sensitivity, not as
    the default.

    Conditioning on a CORRECT answer is deliberately not offered here. That
    selects a non-random subset -- only problems the model happened to solve --
    which plausibly biases the gap, and it belongs in an explicit analysis
    rather than behind a mask helper.
    """
    if not bounds or task not in bounds:
        return np.ones(len(ex_idx), dtype=bool)
    b = bounds[task]
    ends = np.asarray(b['episode_len'], dtype=np.int64)
    ex = np.asarray(ex_idx, dtype=np.int64)
    pos = np.asarray(pos_idx, dtype=np.int64)
    if ex.size and int(ex.max()) >= ends.size:
        raise ValueError(
            f'episode bounds cover {ends.size} examples but the pairs cache '
            f'references example {int(ex.max())} for {task!r}. The bounds were '
            f'built from a different E1 cache (n_examples mismatch); re-run e0n '
            f'--emit-bounds against the cache these pairs came from.')
    keep = pos < ends[ex]
    if require_completed:
        done = np.asarray(b['completed'], dtype=bool)
        keep &= done[ex]
    return keep


def problem_defined_target(task, examples):
    """(z, K, H_bits, meta) from generator fields. One label per example.

    `examples` is the list returned by e2_prompts.generate_*; only structured
    fields are read, never the prompt text, so the target is unaffected by the
    prompt format under test.
    """
    if task == 'arithmetic':
        raw = np.array([int(e['answer']) for e in examples], dtype=np.int64)
        uniq = np.unique(raw)
        K = int(min(K_ARITH_BINS, len(uniq)))
        # quantile bins over the observed values, matching discretize_position_values
        ranks = np.argsort(np.argsort(raw, kind='stable'), kind='stable')
        z = (ranks * K // len(raw)).astype(np.int64)
        kind = 'final_value_quantile_bins'
        class_values = [sorted(set(int(v) for v, c in zip(raw, z) if c == k))
                        for k in range(K)]

    elif task == 'counting':
        raw = np.array([int(e['count']) for e in examples], dtype=np.int64)
        uniq, z = np.unique(raw, return_inverse=True)
        K = int(len(uniq)); kind = 'final_count'
        class_values = [[int(u)] for u in uniq]

    elif task == 'deduction':
        answers = {str(e['answer']) for e in examples}
        if len(answers) < 2:
            # expected: every generated syllogism is valid, so answer is constant
            pass
        raw = [e['Z'] for e in examples]
        uniq = sorted(set(raw))
        idx = {u: i for i, u in enumerate(uniq)}
        z = np.array([idx[r] for r in raw], dtype=np.int64)
        K = len(uniq); kind = 'conclusion_entity'
        class_values = [[u] for u in uniq]
        if len(answers) < 2:
            kind += ' (answer is constant across examples; an answer-level ' \
                    'target would have H(Z)=0)'

    else:
        raise ValueError(task)

    if K < 2:
        raise ValueError(
            f"problem-defined target for {task!r} is degenerate: K={K}, so "
            f"H(Z)=0 and rho is undefined. Check the example generator.")

    H = entropy_bits_miller_madow(z, K)
    H = float(H[0]) if isinstance(H, tuple) else float(H)
    meta = {'kind': kind, 'K': K, 'n_examples': len(examples),
            'H_bits': H, 'regime_invariant': True,
            'class_values': class_values}
    return z, K, H, meta


def expand_example_target(z_by_example, ex_array):
    """Broadcast a per-example label to per-token positions.

    ex_array is the per-token example index the caches already carry, so this
    is the only step needed to use a per-example Z with the existing
    per-token decoders and the existing GroupKFold-by-example grouping.
    """
    z_by_example = np.asarray(z_by_example)
    ex_array = np.asarray(ex_array)
    bad = ex_array[(ex_array < 0) | (ex_array >= len(z_by_example))]
    if len(bad):
        raise IndexError(
            f"{len(bad)} token positions reference example ids outside "
            f"0..{len(z_by_example)-1} (e.g. {bad[:5].tolist()}). The target was "
            f"built from a different number of examples than the cache holds.")
    return z_by_example[ex_array]


def guard_target(entry, target_name):
    """Positive-control guard (verbatim semantics from E2)."""
    rs = entry.get('rho_state', 0.0)
    entry['valid'] = bool(rs >= POS_CONTROL_MIN)
    if not entry['valid']:
        entry['invalid_reason'] = (
            f"positive control failed: rho_state={rs:.3f} < {POS_CONTROL_MIN}.")
    return entry


# =============================================================================
# 2. TRACE PARSING + PER-POSITION TARGETS — verbatim from E2
# =============================================================================
def parse_trace_events(task, text):
    ev = []
    if task == 'counting':
        for m in re.finditer(r"count\s*=\s*(\d+)", text):
            ev.append((m.end(1), int(m.group(1))))
    elif task == 'arithmetic':
        for m in re.finditer(r"=\s*(-?\d+)", text):
            ev.append((m.end(1), int(m.group(1))))
    elif task in ('deduction', 'balanced_deduction'):
        # SAME RULE FOR BOTH, AND THAT IS THE POINT.
        #
        # The per-position target is the entity most recently named, not the
        # Yes/No verdict. `balanced_deduction` draws from the same ENTITIES
        # pool and writes the same "X -> Y" chain, so the event stream is
        # identical in kind; only premise 2's direction and the final line
        # differ, and neither is an entity event.
        #
        # This branch was missing, and `else: raise ValueError(task)` meant E2
        # died on the position target AFTER the whole E1 capture was already
        # on disk. That is the failure mode the e29 readiness probes exist to
        # catch, and it went uncaught because the probes covered the prompt,
        # pilot and episode-filter lookups but not this one.
        for m in re.finditer(r"[A-Za-z]+", text):
            w = m.group(0)
            if w in ENTITY_INDEX:
                ev.append((m.end(0), ENTITY_INDEX[w]))
    else:
        raise ValueError(task)
    ev.sort(key=lambda t: t[0])
    return ev


def ensure_e2_task_sets(e2, tasks, verbose=True, force=False):
    """Make an imported E2 module's prompt/example sets cover `tasks`.

    WHY EVERY DOWNSTREAM SCRIPT NEEDS THIS.

    E4, E5 and E6 import E2 as a module to reuse its prompt strings and parsed
    examples. E2 builds `cot_prompts` and `task_examples` in its MODULE BODY
    from a hardcoded default task list; the `--tasks` selector lives in
    `configure_from_cli()`, which only E2's own `main()` calls. So an imported
    E2 always holds exactly the three original tasks no matter what the
    importing script was invoked with.

    The symptom is a bare `KeyError: '<task>'` at the point of use -- in E6
    that is after the model has loaded and the episode filter has already
    printed, so the failure looks like a data problem rather than a wiring one.

    Rebuild through E2's own `_build_task_sets` rather than calling the
    generators here: the prompt text must be byte-identical to what E1 wrote
    into the pairs cache, and a single definition is the only way to guarantee
    that.
    """
    have = getattr(e2, 'cot_prompts', {}) or {}
    missing = list(tasks) if force else [t for t in tasks if t not in have]
    if not missing:
        return e2
    built_p, built_e = e2._build_task_sets(missing, e2.N_EXAMPLES)
    e2.cot_prompts = {**have, **built_p}
    e2.task_examples = {**(getattr(e2, 'task_examples', {}) or {}), **built_e}
    if verbose:
        print(f"  built E2 prompt/example sets for {missing} "
              f"(n_examples={e2.N_EXAMPLES}; E2's module body covers only its "
              f"default tasks)")
    return e2


def check_prompt_coverage(prompts, ex_array, task):
    """The prompt list must span every example index present in the pairs.

    A short list does not raise on its own -- it silently indexes the wrong
    prompt for every example past its end, attaching one example's query block
    to another's residuals and corrupting the prompt channel without a word.
    """
    need = int(max(ex_array)) + 1 if len(ex_array) else 0
    if len(prompts) < need:
        raise SystemExit(
            f"{task}: {len(prompts)} prompts but the pairs cache references "
            f"example {need - 1}. E2's N_EXAMPLES does not match the E1 "
            f"capture; rebuild with the same --n-examples.")


def discretize_position_values(task, raw):
    """Map raw per-token reasoning values (None = undefined) to contiguous classes.
    Returns (z, K, meta, valid_mask, class_values) where class_values maps class
    id -> representative raw value(s) (needed by E8 to elicit named values)."""
    arr = np.array([np.nan if v is None else float(v) for v in raw], dtype=np.float64)
    valid = ~np.isnan(arr)
    z = np.full(len(arr), -1, dtype=np.int64)
    if valid.sum() == 0:
        return z, 0, {'kind': 'none', 'K': 0}, valid, []
    if task == 'arithmetic':
        # Bin by VALUE, not by rank -- see the long note on the same fix in
        # E2_Discrete_Recoverability_HF._discretize_position_values. Rank binning
        # split tied values by stable-sort order, i.e. by example index, putting
        # ~45% of the arithmetic label outside the arithmetic. These two
        # implementations must stay identical; _devtool_compare_impls.py checks.
        vv = arr[valid]
        K_want = int(min(K_ARITH_BINS, len(np.unique(vv))))
        edges = np.quantile(vv, np.linspace(0.0, 1.0, K_want + 1)[1:-1])
        zb = np.searchsorted(edges, vv, side='right').astype(np.int64)
        remap = {b: i for i, b in enumerate(sorted(set(zb.tolist())))}
        zc = np.array([remap[b] for b in zb.tolist()], dtype=np.int64)
        K = len(remap)
        z[valid] = zc
        # class -> sorted list of raw values falling in that quantile bin
        class_values = [sorted(set(int(v) for v, c in zip(vv, zc) if c == k))
                        for k in range(K)]
        kind = 'running_result_quantile_bins'
    else:
        uniq, inv = np.unique(arr[valid], return_inverse=True)
        z[valid] = inv
        K = int(len(uniq))
        class_values = [[int(u)] for u in uniq]
        kind = 'running_count' if task == 'counting' else 'current_entity'
    meta = {'kind': kind, 'K': K, 'n_valid': int(valid.sum()), 'n_total': int(len(arr))}
    return z, K, meta, valid, class_values


def build_position_labels(task, ids_by_ex_task, ex_array, pos_array, tokenizer):
    """Per-token reasoning-state target (verbatim logic from E2), plus the decoded
    per-example trace text and per-token char offsets (reused by E7/E8 to align
    observer-model features / monitor queries to subject-token positions).

    Returns (z, K, meta, valid, event_mask, class_values, text_by_ex, char_end_by_ex).
    """
    vals_cache, events_cache = {}, {}
    text_by_ex, char_end_by_ex = {}, {}

    def compute(e):
        if e in vals_cache:
            return
        ids = ids_by_ex_task[e]
        T = len(ids)
        if T == 0:
            vals_cache[e] = []; events_cache[e] = set()
            text_by_ex[e] = ''; char_end_by_ex[e] = np.zeros(0, dtype=np.int64)
            return
        pieces = [tokenizer.decode([int(t)]) for t in ids]
        text = ''
        char_end = np.empty(T, dtype=np.int64)
        for j, p in enumerate(pieces):
            text += p
            char_end[j] = len(text)
        vals = [None] * T
        events = set()
        for cpos, val in parse_trace_events(task, text):
            ti = int(np.searchsorted(char_end, cpos))
            if ti >= T:
                ti = T - 1
            events.add(ti)
            for k in range(ti, T):
                vals[k] = val
        vals_cache[e] = vals; events_cache[e] = events
        text_by_ex[e] = text; char_end_by_ex[e] = char_end

    raw = []
    event_mask = np.zeros(len(ex_array), dtype=bool)
    for i in range(len(ex_array)):
        e = int(ex_array[i]); p = int(pos_array[i])
        compute(e)
        v = vals_cache[e]
        raw.append(v[p] if 0 <= p < len(v) else None)
        if p in events_cache.get(e, ()):
            event_mask[i] = True
    z, K, meta, valid, class_values = discretize_position_values(task, raw)
    return z, K, meta, valid, event_mask, class_values, text_by_ex, char_end_by_ex


# =============================================================================
# 3. E2 CACHE ACCESS
# =============================================================================
def load_pkl(path):
    if os.path.exists(path):
        with open(path, 'rb') as f:
            return pickle.load(f)
    return None


def save_pkl(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)


def to_python(obj):
    if isinstance(obj, dict):
        return {k: to_python(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_python(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


def save_json(obj, path):
    with open(path, 'w') as f:
        json.dump(to_python(obj), f, indent=2)
    print(f"  wrote {path}")


def pairs_paths(drive_dir, slug):
    return {
        'pairs':   f"{drive_dir}/e2_{slug}_recoverability_pairs.pkl",
        'e2_json': f"{drive_dir}/e2_{slug}_discrete_recoverability.json",
        'e6_json': f"{drive_dir}/e6_{slug}_repr_prompt_augmented.json",
        'bw':      f"{drive_dir}/e1_2_{slug}_bandwidth.pkl",
    }


def load_pairs(drive_dir, slug):
    """Load the matched (r,c) pairs cache produced by E2's Stage A."""
    path = pairs_paths(drive_dir, slug)['pairs']
    cached = load_pkl(path)
    if cached is None:
        raise FileNotFoundError(
            f"{path} not found — run e2_discrete_recoverability.py for this "
            "model first (Stage A caches the matched pairs).")
    return cached['pairs'], cached['embed'], cached['ids_by_ex'], cached['metadata']


def content_keep_q(nll, ex, q=0.5, strict=True):
    """Per-trace quantile content filter.

    q=0.5, strict=True is E2's protocol filter and reproduces the published
    numbers exactly (np.quantile(x, 0.5) is np.median(x)). q=0 disables
    selection entirely and keeps every row.

    WHY q IS EXPOSED (e0r_content_filter_sweep.py measured this)
      The median cut is a convention, never a result, and it is not neutral. It
      selects on surprisal, which is defined against the trace context that the
      TRACE decoder's features are built from -- so it preferentially keeps rows
      where the trace channel is handicapped. Measured on gpt2-small arithmetic
      the relative gap runs 0.107 at q=0 against 0.156 at q=0.5, and the
      capability contrast against deepseek-7b shrinks 37% when the filter is
      removed. Four of five cells tested were insensitive; the fifth was not.

      q=0 needs no justification and roughly doubles n, which lifts several
      cells over the 50*K power line.
    """
    nll = np.asarray(nll, dtype=np.float64)
    ex = np.asarray(ex)
    if q <= 0.0:
        return np.ones(len(nll), dtype=bool)
    keep = np.zeros(len(nll), dtype=bool)
    for e in np.unique(ex):
        m = ex == e
        thr = np.quantile(nll[m], q)
        keep[m] = nll[m] > thr if strict else nll[m] >= thr
    return keep


def content_filter(pairs_t, q=0.5, strict=True):
    """E2's content-position filter: surprisal above per-trace median.

    Defaults are unchanged, so every existing caller keeps its behaviour.
    """
    return content_keep_q(pairs_t['nll'], pairs_t['ex'], q=q, strict=strict)


def rebuild_cluster_labels(pairs_t, reported_K):
    """Reproduce E2's cluster target Z on the kept content tokens.  KMeans with
    fixed seed/n_init on the identical feature matrix is deterministic, so these
    labels match the ones behind Paper 1's tables.  Returns (keep, feats, z, km)."""
    keep = content_filter(pairs_t)
    feats = rms_pca_features(pairs_t['resid'][keep], PCA_DIM)
    km = KMeans(n_clusters=reported_K, n_init=KMEANS_NINIT, random_state=SEED)
    z = km.fit_predict(feats).astype(np.int64)
    return keep, feats, z, km


def reported_K_for(e2_json, task, n_content):
    """Prefer E2's recorded reported_K; fall back to the protocol rule."""
    if e2_json:
        t = e2_json.get('results', e2_json).get(task)
        if t and 'reported_K' in t:
            return int(t['reported_K'])
    return max([k for k in K_SWEEP if n_content >= 50 * k] + [K_SWEEP[0]])


# =============================================================================
# 4. FANO FLOOR (Theorem C empirical companion; used by E10)
# =============================================================================
def _h2(p):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return float(-p * np.log2(p) - (1 - p) * np.log2(1 - p))


def fano_floor(H_cond_bits, M):
    """Smallest error probability consistent with Fano's inequality for an
    M-ary variable with conditional entropy H(B|C) = H_cond_bits:
        H(B|C) ≤ h2(Pe) + Pe·log2(M−1).
    Solved numerically.  NOTE (paper framing): H_cond here is the *class-
    restricted* conditional entropy H_V(B|C), estimated by the best calibrated
    decoder's held-out CE.  The floor therefore binds monitors in (roughly) the
    same observer class V — it is a semi-empirical prediction to be validated,
    not an information-theoretic bound on arbitrary monitors.  Theorem A says
    no such bound exists for unbounded observers."""
    if H_cond_bits <= 0:
        return 0.0
    if M < 2:
        return 0.0
    lo, hi = 0.0, 1.0 - 1.0 / M
    def rhs(pe):
        return _h2(pe) + pe * (np.log2(M - 1) if M > 2 else 0.0)
    if rhs(hi) < H_cond_bits:          # even max-error can't produce this H: clamp
        return hi
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if rhs(mid) < H_cond_bits:
            lo = mid
        else:
            hi = mid
    return float(lo)


# =============================================================================
# Self-test:  python p2_common.py
# =============================================================================
if __name__ == '__main__':
    rng = np.random.default_rng(0)
    N, K = 4000, 8
    z = rng.integers(0, K, N)
    groups = np.repeat(np.arange(N // 8), 8)
    # informative features -> rho should be well above 0; shuffled -> ~0
    X = np.eye(K)[z] + 0.5 * rng.normal(size=(N, K))
    _, H = entropy_bits_miller_madow(z, K)
    p = decoder_oof(X, z, groups, K, DECODERS[0][1])
    ce, ilb, rho = ilb_rho(H, p)
    p_sh = decoder_oof(X, rng.permutation(z), groups, K, DECODERS[0][1])
    _, _, rho_sh = ilb_rho(H, p_sh)
    # monitor-style path: a noisy externally-produced distribution
    P = np.full((N, K), 0.05); P[np.arange(N), z] = 1.0
    P /= P.sum(1, keepdims=True)
    flip = rng.random(N) < 0.3
    P[flip] = np.roll(P[flip], 1, axis=1)
    pm = calibrated_oof_from_proba(P, z, groups, K)
    _, _, rho_m = ilb_rho(H, pm)
    print(f"self-test: H={H:.2f}b rho_feat={rho:.3f} (expect >0.5) "
          f"rho_shuffle={rho_sh:.3f} (expect ~0) rho_monitor={rho_m:.3f} (expect ~0.4-0.8)")
    print(f"fano_floor(H_cond=1.5, M=7) = {fano_floor(1.5, 7):.3f} (expect ~0.2-0.4)")
    assert rho > 0.3 and rho_sh < 0.05
    print("p2_common self-test OK")
