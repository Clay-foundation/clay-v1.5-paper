"""Shared evaluation helpers: spatial blocks, label budgets, fixed heads, block bootstrap.

Fixed choices (do not tune on test): StandardScaler (train only) + LogisticRegression(C=1, lbfgs,
max_iter=2000) for classification; RidgeCV(alphas=logspace(-3,3,13)) with inner CV on TRAIN only
for regression. 5 label seeds. 1000 block-bootstrap resamples, RNG seed 1729.
"""
from __future__ import annotations
import hashlib
import os
import numpy as np
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.metrics import f1_score, r2_score, accuracy_score, balanced_accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

SEEDS = (0, 1, 2, 3, 4)
NBOOT = int(os.environ.get("CLAY_NBOOT", "1000"))  # smoke test only (experiments/repro_smoke.sh) lowers this; reported numbers use 1000


def hblock(lat, lon, km=100.0, salt="claypaper-v2"):
    """Deterministic ~km grid block id from lat/lon (equal-angle approx, lon scaled by cos(lat))."""
    lat = np.asarray(lat, float); lon = np.asarray(lon, float)
    dy = km / 111.0
    iy = np.floor(lat / dy).astype(int)
    dx = km / (111.0 * np.clip(np.cos(np.radians((iy + 0.5) * dy)), 0.05, 1))
    ix = np.floor(lon / dx).astype(int)
    return np.array([f"{salt}|{a}|{b}" for a, b in zip(iy, ix)])


def block_split(blocks, frac=(0.7, 0.15, 0.15)):
    """Hash each block to train/val/test deterministically."""
    u = np.array([int(hashlib.sha256(b.encode()).hexdigest()[:8], 16) % 100 for b in blocks])
    return np.where(u < frac[0] * 100, "train", np.where(u < (frac[0] + frac[1]) * 100, "val", "test"))


def label_budget(ids, y, k, seed, salt="claypaper-label-v2"):
    """Nested per-class prefixes ranked by sha256(salt|seed|id). Returns index array."""
    ids = np.asarray(ids).astype(str); y = np.asarray(y)
    out = []
    for c in np.unique(y):
        idx = np.where(y == c)[0]
        key = [hashlib.sha256(f"{salt}|{seed}|{ids[i]}".encode()).hexdigest() for i in idx]
        out.extend(idx[np.argsort(key)][:k])
    return np.array(sorted(out))


def n_budget(ids, n, seed, salt="claypaper-n-v2"):
    ids = np.asarray(ids).astype(str)
    key = [hashlib.sha256(f"{salt}|{seed}|{i}".encode()).hexdigest() for i in ids]
    return np.argsort(key)[:n]


def clf_head():
    return make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000, tol=1e-4))


def reg_head():
    return make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 3, 13)))


def clf_metrics(y, p):
    return {"macro_f1": f1_score(y, p, average="macro"), "acc": accuracy_score(y, p),
            "bal_acc": balanced_accuracy_score(y, p)}


def block_bootstrap(metric_fn, preds_by_seed, y, blocks, n=None, seed=1729):
    """preds_by_seed: list of prediction arrays (one per seed) on the same test rows.
    Returns point (mean over seeds), lo, hi (2.5/97.5 pct over block resamples of the seed-mean)."""
    n = NBOOT if n is None else n
    rng = np.random.default_rng(seed)
    ub = np.unique(blocks); rows = {b: np.where(blocks == b)[0] for b in ub}
    point = float(np.mean([metric_fn(y, p) for p in preds_by_seed]))
    stats = []
    for _ in range(n):
        idx = np.concatenate([rows[b] for b in rng.choice(ub, len(ub), replace=True)])
        stats.append(np.mean([metric_fn(y[idx], p[idx]) for p in preds_by_seed]))
    return point, float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))


def paired_block_bootstrap(metric_fn, preds_a, preds_b, y, blocks, n=None, seed=1729):
    """CI on mean-over-seeds(metric(A)) - mean-over-seeds(metric(B)) with shared block resamples."""
    n = NBOOT if n is None else n
    rng = np.random.default_rng(seed)
    ub = np.unique(blocks); rows = {b: np.where(blocks == b)[0] for b in ub}
    f = lambda idx: np.mean([metric_fn(y[idx], p[idx]) for p in preds_a]) - np.mean([metric_fn(y[idx], p[idx]) for p in preds_b])
    point = float(f(np.arange(len(y))))
    stats = [f(np.concatenate([rows[b] for b in rng.choice(ub, len(ub), replace=True)])) for _ in range(n)]
    return point, float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))
