"""E46-only nested evaluation. No fitted object sees outer test items."""
from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

VERSION = "e46_nested_v1"


class SafePCA(BaseEstimator, TransformerMixin):
    def __init__(self, n_components=16):
        self.n_components = n_components

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        if not np.isfinite(X).all():
            raise ValueError("PCA input contains non-finite values")
        self.constant_ = float(np.var(X, axis=0).sum()) <= 1e-12
        self.pca_ = None
        if not self.constant_:
            self.pca_ = PCA(n_components=max(1, min(self.n_components,
                            X.shape[0] - 1, X.shape[1])),
                            svd_solver="full").fit(X)
        return self

    def transform(self, X):
        if self.constant_:
            return np.zeros((len(X), 1), dtype=float)
        return self.pca_.transform(X)


def matrix_reader(factory, dim=None):
    steps = [StandardScaler()]
    if dim is not None:
        steps.append(SafePCA(dim))
    return make_pipeline(*steps, factory())


def take(X, indices):
    return X[indices] if hasattr(X, "shape") else [X[i] for i in indices]


def splits(groups, n_folds=5, seed=46):
    """Randomize group ordering without consulting targets; reusable under nulls."""
    groups = np.asarray(groups)
    unique, inv = np.unique(groups, return_inverse=True)
    if len(unique) < 2:
        raise ValueError("Need at least two independent groups")
    remap = np.random.default_rng(seed).permutation(len(unique))[inv]
    return list(GroupKFold(min(n_folds, len(unique))).split(remap, groups=remap))


def prior(y, K):
    counts = np.bincount(y, minlength=K).astype(float) + 1.0
    return counts / counts.sum()


def loss_rows(y, P):
    return -np.log2(np.clip(P[np.arange(len(y)), y], 1e-12, 1))


def calibrate(P, temperature, mixture, base):
    logp = np.log(np.clip(P, 1e-12, 1)) / temperature
    logp -= logp.max(axis=1, keepdims=True)
    q = np.exp(logp)
    q /= q.sum(axis=1, keepdims=True)
    return (1 - mixture) * q + mixture * base


def choose_calibration(P, y, bases):
    best = None
    for temp in (0.5, 1.0, 2.0, 4.0, 8.0):
        for mix in (0.0, 0.25, 0.5, 0.75, 1.0):
            ce = float(loss_rows(y, calibrate(P, temp, mix, bases)).mean())
            if best is None or ce < best[0]:
                best = ce, temp, mix
    return best


def fit_predict(factory, X, y, train, test, K, return_estimator=False):
    if len(np.unique(y[train])) < 2:
        result = np.tile(prior(y[train], K), (len(test), 1))
        return (result, None) if return_estimator else result
    clf = factory()
    clf.fit(take(X, train), y[train])
    p = np.zeros((len(test), K))
    p[:, np.asarray(clf.classes_, dtype=int)] = clf.predict_proba(take(X, test))
    return (p, clf) if return_estimator else p


def nested_oof(X, y, groups, K, candidates, *, seed=46, n_folds=5,
               train_mask=None, outer=None):
    """Choose pipeline and calibration on inner OOF rows, refit on outer train.

    candidates: (name, zero-argument pipeline factory) pairs. train_mask permits
    arm-transfer: all fits and calibration see only the designated training arm,
    while the outer test predicts both arms of unseen items.
    """
    y, groups = np.asarray(y, dtype=int), np.asarray(groups)
    mask = np.ones(len(y), bool) if train_mask is None else np.asarray(train_mask)
    outer = splits(groups, n_folds, seed) if outer is None else outer
    P, bases = np.zeros((len(y), K)), np.zeros((len(y), K))
    fold_ids = np.full(len(y), -1, dtype=int)
    details = []
    for fold, (train, test) in enumerate(outer):
        train = train[mask[train]]
        if not len(train):
            raise ValueError("Outer fold has no eligible training rows")
        base = prior(y[train], K)
        bases[test] = base
        fold_ids[test] = fold
        if len(np.unique(groups[train])) < 3 or len(np.unique(y[train])) < 2:
            P[test] = base
            details.append({"fold": fold, "candidate": "base_only"})
            continue
        inner = splits(groups[train], 3, seed + fold + 1)
        inner_bases = np.zeros((len(train), K))
        for it, iv in inner:
            inner_bases[iv] = prior(y[train[it]], K)
        best = None
        for name, factory in candidates:
            pi = np.zeros((len(train), K))
            for it, iv in inner:
                pi[iv] = fit_predict(factory, X, y, train[it], train[iv], K)
            ce, temp, mix = choose_calibration(pi, y[train], inner_bases)
            if best is None or ce < best[0]:
                best = ce, name, factory, temp, mix
        ce, name, factory, temp, mix = best
        raw, fitted = fit_predict(factory, X, y, train, test, K,
                                  return_estimator=True)
        P[test] = calibrate(raw, temp, mix, base)
        detail = {"fold": fold, "candidate": name, "temperature": temp,
                  "prior_mixture": mix, "inner_ce_bits": ce,
                  "train_groups": np.unique(groups[train]).tolist(),
                  "test_groups": np.unique(groups[test]).tolist()}
        if fitted is not None and isinstance(getattr(fitted, 'fit_stats', None), dict):
            detail['fit_stats'] = fitted.fit_stats
        details.append(detail)
    if np.any(fold_ids < 0):
        raise ValueError("Outer folds did not cover every row")
    return {"P": P, "base": bases, "fold": fold_ids, "details": details,
            "evaluation_version": VERSION}


def holm(pvalues):
    """Monotone Holm adjusted p-values in original order."""
    p = np.asarray(pvalues, float)
    order = np.argsort(p)
    out = np.empty(len(p))
    out[order] = np.minimum(1, np.maximum.accumulate(
        (len(p) - np.arange(len(p))) * p[order]))
    return out.tolist()
