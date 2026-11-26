#!/usr/bin/env python3
"""Deterministic, leak-free public-data benchmark of Linear Probes & Linear SVMs
against Centroid prototypes, k-NN, Decision-Tree Ensembles (RF, HistGBDT), MLPs,
and 2D UMAP projections on frozen Clay v1.5 1,024-D embeddings.

Datasets & Spatial Separation Guarantees (100% public):
  1. Spatially blocked EuroSAT MS (N = 25,648 Sentinel-2 chips; 18,232 train across
     256 LAEA 100 km blocks, 3,176 test across 49 held-out blocks with >= 2 km gap).
  2. Public Copernicus Sentinel-2 L2A 1.28 km scenes (randomly drawn Texas scenes, summer 2024; chip list and sha256 of every array in results/s2_nlcd_320_manifest.csv) + USGS
     NLCD fractional impervious (2024) and tree canopy cover (2023) co-occurring labels (N = 223 scenes;
     5-fold GroupKFold over single-linkage spatial clusters separated by > 30 km).
  3. AlphaEarth Foundations 15-task evaluation synthesis (from verified E1 runs:
     Linear probe vs. 3-NN across 15 classification/regression/change tasks).

Inputs are built by `experiments/build_lesson_inputs.py` (s2-nlcd and eurosat subcommands) and are shared with
run_clay_v15_experiments.py (same chip root). Run:
    python experiments/run_probe_and_lsvm_benchmark.py [--latlon-order training|documented] [--eurosat-feat-dir DIR] [--out-name NAME]
Environment overrides: CLAY_V15_CKPT, CLAY_MODEL_REPO, CLAY_S2_NLCD_ROOT, CLAY_EUROSAT_FEAT_DIR.

Zero-leakage guarantee:
  Every normalization (L2 row norm, StandardScaler), manifold reducer (UMAP), and
  hyperparameter selector (inner CV) is fitted strictly inside the training split.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
import warnings
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["VECLIB_MAXIMUM_THREADS"] = "2"
os.environ["NUMBA_NUM_THREADS"] = "2"
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.metrics import average_precision_score, f1_score, r2_score, roc_auc_score
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, LinearSVR
import torch
import umap
import yaml

WORK_ROOT = Path(__file__).resolve().parents[1]
MODEL_REPO = Path(os.environ.get("CLAY_MODEL_REPO", str(WORK_ROOT / "experiments/suite/common")))
CKPT_PATH = Path(
    os.environ.get("CLAY_V15_CKPT", str(WORK_ROOT / "checkpoints/v1.5/clay-v1.5.ckpt"))
)
EUROSAT_FEAT_DIR = Path(os.environ.get("CLAY_EUROSAT_FEAT_DIR", str(WORK_ROOT / "data/eurosat/features")))
S2_ROOT = Path(os.environ.get("CLAY_S2_NLCD_ROOT", str(WORK_ROOT / "data/s2_nlcd_320")))  # same root as run_clay_v15_experiments.py
LATLON_ORDER = "training"  # set from --latlon-order in main(); P4 location slots
E1_AEF_JSON = WORK_ROOT / "experiments/results/e1_label_efficiency_aef.json"
OUT_PATH = WORK_ROOT / "experiments/results/probe_and_lsvm_results.json"

REQUIRED_CKPT_SHA256 = "21432069250b9b3f9a65ffd0071c5ad56b793247285ab0604edf7f531d4798d0"
CLAY_S2_BANDS_12_IDX = [1, 2, 3, 4, 5, 6, 7, 8, 10, 11]
CLAY_S2_BAND_NAMES = [
    "blue", "green", "red", "rededge1", "rededge2",
    "rededge3", "nir", "nir08", "swir16", "swir22",
]
SEED = 42


class L2RowNormalizer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        norms = np.clip(np.linalg.norm(X, axis=1, keepdims=True), 1e-12, None)
        return X / norms


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def paired_block_bootstrap_ci(y_true, pred_a, pred_b, blocks, metric_fn, n_boot: int = 1000, seed: int = SEED) -> dict:
    """Paired block-bootstrap interval of metric(a) - metric(b): both predictions are scored on the same resampled blocks."""
    rng = np.random.default_rng(seed)
    uniq_b = np.unique(blocks)
    idx_by_b = [np.where(blocks == b)[0] for b in uniq_b]
    diffs = []
    for _ in range(n_boot):
        draw = rng.integers(0, len(uniq_b), size=len(uniq_b))
        idx = np.concatenate([idx_by_b[i] for i in draw])
        diffs.append(float(metric_fn(y_true[idx], pred_a[idx]) - metric_fn(y_true[idx], pred_b[idx])))
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    point = float(metric_fn(y_true, pred_a) - metric_fn(y_true, pred_b))
    return {"point": round(point, 4), "ci95": [round(float(lo), 4), round(float(hi), 4)], "n_boot": n_boot,
            "unit": "spatial block, seed-0 predictions"}


def l2_norm(X: np.ndarray) -> np.ndarray:
    return X / np.clip(np.linalg.norm(X, axis=1, keepdims=True), 1e-12, None)


def block_bootstrap_ci(
    y_true: np.ndarray,
    y_pred_or_score: np.ndarray,
    blocks: np.ndarray,
    metric_fn,
    n_boot: int = 1000,
    seed: int = SEED,
) -> list[float]:
    rng = np.random.default_rng(seed)
    uniq_b = np.unique(blocks)
    idx_by_b = [np.where(blocks == b)[0] for b in uniq_b]
    vals = []
    for _ in range(n_boot):
        draw = rng.integers(0, len(uniq_b), size=len(uniq_b))
        idx = np.concatenate([idx_by_b[i] for i in draw])
        if len(np.unique(y_true[idx])) < 2:
            continue
        vals.append(float(metric_fn(y_true[idx], y_pred_or_score[idx])))
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return [round(float(lo), 4), round(float(hi), 4)]


def select_linear_svm(X_tr_l2: np.ndarray, y_tr: np.ndarray, seed: int = SEED, fixed_C: float | None = None) -> LinearSVC:
    if fixed_C is not None:
        clf = LinearSVC(C=fixed_C, class_weight="balanced", tol=1e-3, max_iter=2000, random_state=seed)
        clf.fit(X_tr_l2, y_tr)
        return clf
    min_count = int(np.min(np.bincount(y_tr)))
    c_grid = [0.05, 0.2, 1.0, 5.0, 20.0]
    if min_count < 4:
        clf = LinearSVC(C=1.0, class_weight="balanced", tol=1e-3, max_iter=2000, random_state=seed)
        clf.fit(X_tr_l2, y_tr)
        return clf
    n_splits = min(3, min_count)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    best_c, best_score = 1.0, -1.0
    is_binary = len(np.unique(y_tr)) == 2
    for c in c_grid:
        scores = []
        for tr_i, va_i in skf.split(X_tr_l2, y_tr):
            m = LinearSVC(C=c, class_weight="balanced", tol=1e-3, max_iter=1500, random_state=seed)
            m.fit(X_tr_l2[tr_i], y_tr[tr_i])
            if is_binary:
                scores.append(average_precision_score(y_tr[va_i], m.decision_function(X_tr_l2[va_i])))
            else:
                scores.append(f1_score(y_tr[va_i], m.predict(X_tr_l2[va_i]), average="macro"))
        mean_sc = float(np.mean(scores))
        if mean_sc > best_score:
            best_score = mean_sc
            best_c = c
    clf = LinearSVC(C=best_c, class_weight="balanced", tol=1e-3, max_iter=2000, random_state=seed)
    clf.fit(X_tr_l2, y_tr)
    return clf


def select_logistic(X_tr_l2: np.ndarray, y_tr: np.ndarray, seed: int = SEED) -> LogisticRegression:
    min_count = int(np.min(np.bincount(y_tr)))
    c_grid = [0.1, 1.0, 10.0, 100.0]
    if min_count < 4:
        clf = LogisticRegression(C=10.0, class_weight="balanced", tol=1e-3, max_iter=800, random_state=seed)
        clf.fit(X_tr_l2, y_tr)
        return clf
    n_splits = min(3, min_count)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    best_c, best_score = 10.0, -1.0
    is_binary = len(np.unique(y_tr)) == 2
    for c in c_grid:
        scores = []
        for tr_i, va_i in skf.split(X_tr_l2, y_tr):
            m = LogisticRegression(C=c, class_weight="balanced", tol=1e-3, max_iter=600, random_state=seed)
            m.fit(X_tr_l2[tr_i], y_tr[tr_i])
            if is_binary:
                scores.append(average_precision_score(y_tr[va_i], m.predict_proba(X_tr_l2[va_i])[:, 1]))
            else:
                scores.append(f1_score(y_tr[va_i], m.predict(X_tr_l2[va_i]), average="macro"))
        mean_sc = float(np.mean(scores))
        if mean_sc > best_score:
            best_score = mean_sc
            best_c = c
    clf = LogisticRegression(C=best_c, class_weight="balanced", tol=1e-3, max_iter=800, random_state=seed)
    clf.fit(X_tr_l2, y_tr)
    return clf


P1_CONTRASTS: dict = {}  # budget -> paired block-bootstrap contrasts (kept out of the per-method summaries)


def run_p1_eurosat_multiclass(X_tr_all, y_tr_all, X_te, y_te, blocks_te):
    print("[P1] Running spatially blocked EuroSAT 10-way probe comparison...")
    X_tr_l2_all = l2_norm(X_tr_all)
    X_te_l2 = l2_norm(X_te)
    budgets = [5, 10, 50, 200]
    seeds = [0, 1, 2, 3, 4]
    methods = [
        "lsvm",
        "logistic",
        "centroid",
        "knn_cosine",
        "mlp_2layer",
        "random_forest",
        "hist_gbdt",
        "umap2d_knn",
        "umap2d_logistic",
    ]
    out = {}
    for b in budgets:
        res_b = {m: [] for m in methods}
        preds_seed0 = {}
        for s in seeds:
            rng = np.random.default_rng(1000 + s * 37 + b)
            idx_list = []
            for c in range(10):
                c_idx = np.where(y_tr_all == c)[0]
                idx_list.append(rng.choice(c_idx, size=b, replace=False))
            tr_idx = np.concatenate(idx_list)
            X_tr_raw, y_tr = X_tr_all[tr_idx], y_tr_all[tr_idx]
            X_tr_l2 = X_tr_l2_all[tr_idx]

            # 1. LSVM
            m_svm = select_linear_svm(X_tr_l2, y_tr, seed=s)
            p_svm = m_svm.predict(X_te_l2)
            res_b["lsvm"].append(float(f1_score(y_te, p_svm, average="macro")))

            # 2. Logistic Probe
            m_lr = select_logistic(X_tr_l2, y_tr, seed=s)
            p_lr = m_lr.predict(X_te_l2)
            res_b["logistic"].append(float(f1_score(y_te, p_lr, average="macro")))

            # 3. Nearest Class Centroid (L2-normalized prototypes)
            protos = np.stack([l2_norm(X_tr_l2[y_tr == c].mean(axis=0, keepdims=True))[0] for c in range(10)])
            p_cent = np.argmax(X_te_l2 @ protos.T, axis=1)
            res_b["centroid"].append(float(f1_score(y_te, p_cent, average="macro")))

            # 4. Cosine k-NN
            k_nn = min(5, b)
            m_knn = KNeighborsClassifier(n_neighbors=k_nn, metric="cosine", weights="distance")
            m_knn.fit(X_tr_l2, y_tr)
            p_knn = m_knn.predict(X_te_l2)
            res_b["knn_cosine"].append(float(f1_score(y_te, p_knn, average="macro")))

            # 5. 2-Layer MLP (fitted with train-only StandardScaler)
            m_mlp = Pipeline([
                ("sc", StandardScaler()),
                ("mlp", MLPClassifier(hidden_layer_sizes=(256,), alpha=1e-2, tol=1e-3, max_iter=180, early_stopping=False, random_state=s)),
            ])
            m_mlp.fit(X_tr_l2, y_tr)
            p_mlp = m_mlp.predict(X_te_l2)
            res_b["mlp_2layer"].append(float(f1_score(y_te, p_mlp, average="macro")))

            # 6. Random Forest (axis-aligned trees on L2-normalized embedding coords)
            m_rf = Pipeline([("l2", L2RowNormalizer()), ("rf", RandomForestClassifier(n_estimators=150, max_depth=16, random_state=s, n_jobs=1))])
            m_rf.fit(X_tr_raw, y_tr)
            p_rf = m_rf.predict(X_te)
            res_b["random_forest"].append(float(f1_score(y_te, p_rf, average="macro")))

            # 7. HistGBDT
            m_gb = Pipeline([("l2", L2RowNormalizer()), ("gb", HistGradientBoostingClassifier(max_iter=100, max_depth=6, random_state=s))])
            m_gb.fit(X_tr_raw, y_tr)
            p_gb = m_gb.predict(X_te)
            res_b["hist_gbdt"].append(float(f1_score(y_te, p_gb, average="macro")))

            # 8 & 9. Train-only 2D UMAP projection (evaluate on 3 seeds across all budgets)
            if s < 3:
                n_nb = min(15, max(5, len(X_tr_l2) // 5))
                reducer = umap.UMAP(
                    n_components=2, n_neighbors=n_nb, min_dist=0.1, metric="cosine", random_state=SEED + s, n_jobs=1
                )
                Z_tr_2d = reducer.fit_transform(X_tr_l2)
                Z_te_2d = reducer.transform(X_te_l2)
                uknn = KNeighborsClassifier(n_neighbors=min(5, b), weights="distance").fit(Z_tr_2d, y_tr)
                p_uknn = uknn.predict(Z_te_2d)
                res_b["umap2d_knn"].append(float(f1_score(y_te, p_uknn, average="macro")))

                ulr = Pipeline([("sc", StandardScaler()), ("lr", LogisticRegression(C=10.0, max_iter=1000))]).fit(Z_tr_2d, y_tr)
                p_ulr = ulr.predict(Z_te_2d)
                res_b["umap2d_logistic"].append(float(f1_score(y_te, p_ulr, average="macro")))
                if s == 0:
                    preds_seed0["umap2d_knn"] = p_uknn

            if s == 0:
                preds_seed0.update({
                    "lsvm": p_svm, "logistic": p_lr, "centroid": p_cent,
                    "knn_cosine": p_knn, "mlp_2layer": p_mlp, "random_forest": p_rf, "hist_gbdt": p_gb,
                })

        contrasts = {}
        for other in ["logistic", "mlp_2layer", "random_forest", "hist_gbdt", "knn_cosine", "centroid", "umap2d_knn"]:
            if other in preds_seed0:
                contrasts[f"lsvm_minus_{other}"] = paired_block_bootstrap_ci(
                    y_te, preds_seed0["lsvm"], preds_seed0[other], blocks_te,
                    lambda yt, yp: f1_score(yt, yp, average="macro"), n_boot=500)
        summary_b = {}
        P1_CONTRASTS[str(b)] = contrasts
        for m in methods:
            vals = res_b[m]
            entry = {
                "mean_macro_f1": round(float(np.mean(vals)), 4),
                "sd_seeds": round(float(np.std(vals)), 4),
                "n_seeds": len(vals),
            }
            if m in preds_seed0:
                entry["seed0_block_ci95"] = block_bootstrap_ci(
                    y_te, preds_seed0[m], blocks_te, lambda yt, yp: f1_score(yt, yp, average="macro"), n_boot=500
                )
            summary_b[m] = entry
        print(
            f"     n={b:3d}/class | LSVM={summary_b['lsvm']['mean_macro_f1']:.3f} | "
            f"LogReg={summary_b['logistic']['mean_macro_f1']:.3f} | "
            f"Centroid={summary_b['centroid']['mean_macro_f1']:.3f} | "
            f"kNN={summary_b['knn_cosine']['mean_macro_f1']:.3f} | "
            f"MLP={summary_b['mlp_2layer']['mean_macro_f1']:.3f} | "
            f"RF={summary_b['random_forest']['mean_macro_f1']:.3f} | "
            f"GBDT={summary_b['hist_gbdt']['mean_macro_f1']:.3f} | "
            f"UMAP2D={summary_b['umap2d_knn']['mean_macro_f1']:.3f}"
        )
        out[str(b)] = summary_b
    return out


def precision_at_k(y_true: np.ndarray, scores: np.ndarray, k: int = 50) -> float:
    top_idx = np.argsort(scores)[::-1][:k]
    return float(np.mean(y_true[top_idx]))


def run_p2_concept_search(X_tr_all, y_tr_all, X_te, y_te):
    print("[P2] Running imbalanced one-vs-rest concept search across 10 classes...")
    X_tr_l2_all = l2_norm(X_tr_all)
    X_te_l2 = l2_norm(X_te)
    pos_budgets = [2, 5, 25, 100]
    seeds = [0, 1, 2]
    methods = [
        "lsvm",
        "logistic",
        "centroid_diff",
        "pos_centroid",
        "max_sim_exemplar",
        "knn_prob",
        "mlp_2layer",
        "random_forest",
        "hist_gbdt",
    ]
    out = {}
    for kp in pos_budgets:
        kn = 10 * kp
        ap_by_m = {m: [] for m in methods}
        p50_by_m = {m: [] for m in methods}
        for s in seeds:
            rng = np.random.default_rng(5000 + s * 97 + kp)
            for c in range(10):
                pos_pool = np.where(y_tr_all == c)[0]
                neg_pool = np.where(y_tr_all != c)[0]
                p_idx = rng.choice(pos_pool, size=kp, replace=False)
                n_idx = rng.choice(neg_pool, size=kn, replace=False)
                tr_idx = np.concatenate([p_idx, n_idx])
                yb_tr = np.concatenate([np.ones(kp, dtype=np.int64), np.zeros(kn, dtype=np.int64)])
                yb_te = (y_te == c).astype(np.int64)

                X_tr_l2 = X_tr_l2_all[tr_idx]
                X_tr_raw = X_tr_all[tr_idx]

                # 1. LSVM (single weight vector w in R^1024)
                svm = select_linear_svm(X_tr_l2, yb_tr, seed=s)
                s_svm = svm.decision_function(X_te_l2)

                # 2. Logistic Probe (single weight vector w in R^1024)
                lr = select_logistic(X_tr_l2, yb_tr, seed=s)
                s_lr = X_te_l2 @ lr.coef_[0] + lr.intercept_[0]

                # 3. Centroid Difference (mu_+ - mu_-)
                mu_p = X_tr_l2[:kp].mean(axis=0)
                mu_n = X_tr_l2[kp:].mean(axis=0)
                w_cd = mu_p - mu_n
                w_cd = w_cd / max(float(np.linalg.norm(w_cd)), 1e-12)
                s_cd = X_te_l2 @ w_cd

                # 4. Positive Centroid only (mu_+)
                w_pos = mu_p / max(float(np.linalg.norm(mu_p)), 1e-12)
                s_pos = X_te_l2 @ w_pos

                # 5. Max-Sim over positive exemplars
                s_max = np.max(X_te_l2 @ X_tr_l2[:kp].T, axis=1)

                # 6. Cosine k-NN probability
                knn = KNeighborsClassifier(n_neighbors=min(10, len(yb_tr) - 1), metric="cosine", weights="distance")
                knn.fit(X_tr_l2, yb_tr)
                s_knn = knn.predict_proba(X_te_l2)[:, 1]

                # 7. 2-Layer MLP
                mlp = Pipeline([
                    ("sc", StandardScaler()),
                    ("mlp", MLPClassifier(hidden_layer_sizes=(128,), alpha=1e-2, tol=1e-3, max_iter=150, random_state=s)),
                ])
                mlp.fit(X_tr_l2, yb_tr)
                s_mlp = mlp.predict_proba(X_te_l2)[:, 1]

                # 8. Random Forest
                rf = Pipeline([("l2", L2RowNormalizer()), ("rf", RandomForestClassifier(n_estimators=100, max_depth=12, class_weight="balanced", random_state=s, n_jobs=1))])
                rf.fit(X_tr_raw, yb_tr)
                s_rf = rf.predict_proba(X_te)[:, 1]

                # 9. HistGBDT
                gb = Pipeline([("l2", L2RowNormalizer()), ("gb", HistGradientBoostingClassifier(max_iter=50, max_bins=64, max_depth=5, class_weight="balanced", random_state=s))])
                gb.fit(X_tr_raw, yb_tr)
                s_gb = gb.predict_proba(X_te)[:, 1]

                for m_name, sc in [
                    ("lsvm", s_svm), ("logistic", s_lr), ("centroid_diff", s_cd),
                    ("pos_centroid", s_pos), ("max_sim_exemplar", s_max), ("knn_prob", s_knn),
                    ("mlp_2layer", s_mlp), ("random_forest", s_rf), ("hist_gbdt", s_gb),
                ]:
                    ap_by_m[m_name].append(float(average_precision_score(yb_te, sc)))
                    p50_by_m[m_name].append(precision_at_k(yb_te, sc, k=50))

        summary_kp = {}
        for m in methods:
            summary_kp[m] = {
                "mean_ap": round(float(np.mean(ap_by_m[m])), 4),
                "sd_ap": round(float(np.std(ap_by_m[m])), 4),
                "mean_p_at_50": round(float(np.mean(p50_by_m[m])), 4),
                "n_runs": len(ap_by_m[m]),
                "single_vector_query": m in ("lsvm", "logistic", "centroid_diff", "pos_centroid"),
            }
        print(
            f"     k+={kp:3d} (k-={kn:4d}) | LSVM AP={summary_kp['lsvm']['mean_ap']:.3f} | "
            f"LogReg={summary_kp['logistic']['mean_ap']:.3f} | "
            f"CentDiff={summary_kp['centroid_diff']['mean_ap']:.3f} | "
            f"PosMean={summary_kp['pos_centroid']['mean_ap']:.3f} | "
            f"MaxSim={summary_kp['max_sim_exemplar']['mean_ap']:.3f} | "
            f"RF={summary_kp['random_forest']['mean_ap']:.3f} | "
            f"GBDT={summary_kp['hist_gbdt']['mean_ap']:.3f}"
        )
        out[str(kp)] = summary_kp
    return out


def run_p3_label_noise_stress(X_tr_all, y_tr_all, X_te, y_te):
    print("[P3] Running positive label-noise stress test (k+=25, k-=250)...")
    X_tr_l2_all = l2_norm(X_tr_all)
    X_te_l2 = l2_norm(X_te)
    kp, kn = 25, 250
    noise_rates = [0.0, 0.15, 0.30]
    seeds = [0, 1, 2]
    methods = [
        "lsvm_reg_c01",
        "lsvm_cv",
        "logistic_cv",
        "centroid_diff",
        "max_sim_1nn",
        "mlp_2layer",
        "random_forest",
        "hist_gbdt",
    ]
    out = {}
    for eta in noise_rates:
        n_flip = int(round(eta * kp))
        ap_by_m = {m: [] for m in methods}
        for s in seeds:
            rng = np.random.default_rng(8000 + s * 61 + int(eta * 100))
            for c in range(10):
                pos_pool = np.where(y_tr_all == c)[0]
                neg_pool = np.where(y_tr_all != c)[0]
                p_clean = rng.choice(pos_pool, size=kp - n_flip, replace=False)
                n_draw = rng.choice(neg_pool, size=kn + n_flip, replace=False)
                p_noisy = np.concatenate([p_clean, n_draw[:n_flip]])
                n_clean = n_draw[n_flip:]

                tr_idx = np.concatenate([p_noisy, n_clean])
                yb_tr = np.concatenate([np.ones(kp, dtype=np.int64), np.zeros(kn, dtype=np.int64)])
                yb_te = (y_te == c).astype(np.int64)

                X_tr_l2 = X_tr_l2_all[tr_idx]
                X_tr_raw = X_tr_all[tr_idx]

                # Regularized LSVM (C=0.1) vs CV LSVM
                s_svm01 = select_linear_svm(X_tr_l2, yb_tr, seed=s, fixed_C=0.1).decision_function(X_te_l2)
                s_svmcv = select_linear_svm(X_tr_l2, yb_tr, seed=s).decision_function(X_te_l2)
                lr = select_logistic(X_tr_l2, yb_tr, seed=s)
                s_lr = X_te_l2 @ lr.coef_[0] + lr.intercept_[0]

                mu_p, mu_n = X_tr_l2[:kp].mean(axis=0), X_tr_l2[kp:].mean(axis=0)
                w_cd = (mu_p - mu_n) / max(float(np.linalg.norm(mu_p - mu_n)), 1e-12)
                s_cd = X_te_l2 @ w_cd

                # Max-Sim (sensitive to corrupted positive exemplars!)
                s_max = np.max(X_te_l2 @ X_tr_l2[:kp].T, axis=1)

                mlp = Pipeline([
                    ("sc", StandardScaler()),
                    ("mlp", MLPClassifier(hidden_layer_sizes=(128,), alpha=1e-3, tol=1e-3, max_iter=150, random_state=s)),
                ]).fit(X_tr_l2, yb_tr)
                s_mlp = mlp.predict_proba(X_te_l2)[:, 1]

                rf = Pipeline([("l2", L2RowNormalizer()), ("rf", RandomForestClassifier(n_estimators=100, max_depth=14, class_weight="balanced", random_state=s, n_jobs=1))]).fit(X_tr_raw, yb_tr)
                s_rf = rf.predict_proba(X_te)[:, 1]

                gb = Pipeline([("l2", L2RowNormalizer()), ("gb", HistGradientBoostingClassifier(max_iter=50, max_bins=64, max_depth=5, class_weight="balanced", random_state=s))]).fit(X_tr_raw, yb_tr)
                s_gb = gb.predict_proba(X_te)[:, 1]

                for m_name, sc in [
                    ("lsvm_reg_c01", s_svm01), ("lsvm_cv", s_svmcv), ("logistic_cv", s_lr),
                    ("centroid_diff", s_cd), ("max_sim_1nn", s_max), ("mlp_2layer", s_mlp),
                    ("random_forest", s_rf), ("hist_gbdt", s_gb),
                ]:
                    ap_by_m[m_name].append(float(average_precision_score(yb_te, sc)))

        key = f"noise_{int(round(eta * 100))}pct"
        out[key] = {
            m: {"mean_ap": round(float(np.mean(ap_by_m[m])), 4), "sd_ap": round(float(np.std(ap_by_m[m])), 4)}
            for m in methods
        }
        print(
            f"     noise={int(eta*100):2d}% | LSVM(C=0.1)={out[key]['lsvm_reg_c01']['mean_ap']:.3f} | "
            f"LSVM(CV)={out[key]['lsvm_cv']['mean_ap']:.3f} | "
            f"LogReg={out[key]['logistic_cv']['mean_ap']:.3f} | "
            f"CentDiff={out[key]['centroid_diff']['mean_ap']:.3f} | "
            f"MaxSim={out[key]['max_sim_1nn']['mean_ap']:.3f} | "
            f"MLP={out[key]['mlp_2layer']['mean_ap']:.3f} | "
            f"GBDT={out[key]['hist_gbdt']['mean_ap']:.3f}"
        )
    return out


def extract_s2_nlcd_embeddings():
    print("[P4] Extracting canonical Clay v1.5 embeddings on 223 Texas Sentinel-2 scenes (>30 km spatial CV)...")
    ckpt_sha = sha256_file(CKPT_PATH)
    if ckpt_sha != REQUIRED_CKPT_SHA256:
        raise ValueError(f"Checkpoint SHA mismatch: {ckpt_sha}")

    sys.path.insert(0, str(MODEL_REPO))
    from claymodel.model import Encoder  # noqa: PLC0415

    sd = torch.load(str(CKPT_PATH), map_location="cpu", weights_only=True, mmap=True)["state_dict"]
    enc_state = {k.removeprefix("model.encoder."): v for k, v in sd.items() if k.startswith("model.encoder.")}
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    enc = Encoder(mask_ratio=0.0, patch_size=8, shuffle=False, dim=1024, depth=24, heads=16, dim_head=64, mlp_ratio=4.0)
    enc.load_state_dict(enc_state, strict=True)
    enc.eval().requires_grad_(False).to(device)

    metadata = yaml.safe_load((MODEL_REPO / "configs/metadata.yaml").read_text())
    s2_meta = metadata["sentinel-2-l2a"]
    s2_means = np.array([s2_meta["bands"]["mean"][b] for b in CLAY_S2_BAND_NAMES], dtype=np.float32)
    s2_stds = np.array([s2_meta["bands"]["std"][b] for b in CLAY_S2_BAND_NAMES], dtype=np.float32)
    s2_waves = [float(s2_meta["bands"]["wavelength"][b]) for b in CLAY_S2_BAND_NAMES]
    s2_gsd = float(s2_meta["gsd"])

    chips_df = pd.read_parquet(S2_ROOT / "data/chips.parquet")
    nlcd_df = pd.read_parquet(S2_ROOT / "results/labels_nlcd.parquet")
    px_map = {p.stem: p for p in sorted((S2_ROOT / "data/pixels").glob("*.npy"))}
    df = chips_df[chips_df["chip_id"].isin(px_map)].merge(nlcd_df, on="chip_id", how="inner")
    df = df[df["y_impervious"].notna()].sort_values("chip_id").reset_index(drop=True)

    raw_12 = np.stack([np.load(px_map[cid]) for cid in df["chip_id"]], axis=0)
    harm_10 = np.clip(raw_12[:, CLAY_S2_BANDS_12_IDX].astype(np.float32) - 1000.0, 0.0, None)
    norm_10 = (harm_10 - s2_means[None, :, None, None]) / s2_stds[None, :, None, None]

    # LATLON_ORDER "training": [sin(lon), cos(lon), sin(lat), cos(lat)] (default); "documented": [sin(lat), cos(lat), sin(lon), cos(lon)]
    times, latlons = [], []
    for t_str, la, lo in zip(df["datetime"], df["lat"], df["lon"]):
        d = dt.datetime.fromisoformat(str(t_str).replace("Z", "+00:00"))
        wk = d.isocalendar().week * 2.0 * math.pi / 52.0
        hr = d.hour * 2.0 * math.pi / 24.0
        times.append([math.sin(wk), math.cos(wk), math.sin(hr), math.cos(hr)])
        la_r, lo_r = math.radians(float(la)), math.radians(float(lo))
        latlons.append([math.sin(lo_r), math.cos(lo_r), math.sin(la_r), math.cos(la_r)] if LATLON_ORDER == "training"
                       else [math.sin(la_r), math.cos(la_r), math.sin(lo_r), math.cos(lo_r)])
    times = np.asarray(times, dtype=np.float32)
    latlons = np.asarray(latlons, dtype=np.float32)

    cls_list = []
    waves_t = torch.tensor(s2_waves, dtype=torch.float32, device=device)
    with torch.inference_mode():
        for i in range(0, len(norm_10), 32):
            datacube = {
                "pixels": torch.from_numpy(norm_10[i : i + 32]).to(device),
                "time": torch.from_numpy(times[i : i + 32]).to(device),
                "latlon": torch.from_numpy(latlons[i : i + 32]).to(device),
                "waves": waves_t,
                "gsd": torch.tensor(s2_gsd, dtype=torch.float32, device=device),
            }
            unmasked, _, _, _ = enc(datacube)
            cls_list.append(unmasked[:, 0, :].cpu().numpy())
    X_cls = np.concatenate(cls_list, axis=0)

    # >30 km single-linkage spatial GroupKFold
    lats = df["lat"].to_numpy()
    lons = df["lon"].to_numpy()
    la_m, lo_m = np.radians(lats)[:, None], np.radians(lons)[:, None]
    a = np.sin((la_m - la_m.T) / 2.0) ** 2 + np.cos(la_m) * np.cos(la_m.T) * np.sin((lo_m - lo_m.T) / 2.0) ** 2
    dmat = 2.0 * 6371.0088 * np.arcsin(np.clip(np.sqrt(a), 0.0, 1.0))
    clusters = fcluster(linkage(dmat[np.triu_indices(len(lats), k=1)], method="single"), t=30.0, criterion="distance")
    gkf = GroupKFold(n_splits=5)
    folds = np.empty(len(lats), dtype=np.int64)
    min_km = float("inf")
    for fi, (tr, te) in enumerate(gkf.split(X_cls, groups=clusters)):
        folds[te] = fi
        min_km = min(min_km, float(dmat[np.ix_(tr, te)].min()))
    assert min_km > 30.0

    return X_cls, df["y_impervious"].to_numpy(dtype=np.float64), df["y_canopy"].to_numpy(dtype=np.float64), folds, clusters, min_km


def run_p4_colocated_s2_nlcd():
    X_cls, y_imp, y_can, folds, clusters, min_km = extract_s2_nlcd_embeddings()
    X_l2 = l2_norm(X_cls)

    # Co-occurrence statistics on 1.28 km scenes:
    imp_pos = (y_imp >= 5.0).astype(np.int64)
    can_pos = (y_can >= 10.0).astype(np.int64)
    cooccur_count = int(np.sum((imp_pos == 1) & (can_pos == 1)))

    # Part A: Co-occurring multi-label binary retrieval (OOF AP across 5 >30.9 km spatial folds)
    binary_targets = {
        "impervious_ge_5pct": imp_pos,
        "canopy_ge_10pct": can_pos,
    }
    bin_results = {}
    for t_name, y_bin in binary_targets.items():
        oof_scores = {
            m: np.empty(len(y_bin), dtype=np.float64)
            for m in ["lsvm", "logistic", "centroid_diff", "pos_centroid", "knn_cosine", "mlp_2layer", "random_forest", "hist_gbdt", "umap2d_knn"]
        }
        for f in range(5):
            tr, te = folds != f, folds == f
            svm = select_linear_svm(X_l2[tr], y_bin[tr], seed=SEED + f)
            oof_scores["lsvm"][te] = svm.decision_function(X_l2[te])

            lr = select_logistic(X_l2[tr], y_bin[tr], seed=SEED + f)
            oof_scores["logistic"][te] = X_l2[te] @ lr.coef_[0] + lr.intercept_[0]

            mu_p = X_l2[tr][y_bin[tr] == 1].mean(axis=0)
            mu_n = X_l2[tr][y_bin[tr] == 0].mean(axis=0)
            w_cd = (mu_p - mu_n) / max(float(np.linalg.norm(mu_p - mu_n)), 1e-12)
            w_p = mu_p / max(float(np.linalg.norm(mu_p)), 1e-12)
            oof_scores["centroid_diff"][te] = X_l2[te] @ w_cd
            oof_scores["pos_centroid"][te] = X_l2[te] @ w_p

            knn = KNeighborsClassifier(n_neighbors=7, metric="cosine", weights="distance").fit(X_l2[tr], y_bin[tr])
            oof_scores["knn_cosine"][te] = knn.predict_proba(X_l2[te])[:, 1]

            mlp = Pipeline([
                ("sc", StandardScaler()),
                ("mlp", MLPClassifier(hidden_layer_sizes=(128,), alpha=1e-2, max_iter=300, random_state=SEED + f)),
            ]).fit(X_l2[tr], y_bin[tr])
            oof_scores["mlp_2layer"][te] = mlp.predict_proba(X_l2[te])[:, 1]

            rf = Pipeline([("l2", L2RowNormalizer()), ("rf", RandomForestClassifier(n_estimators=150, max_depth=12, class_weight="balanced", random_state=SEED + f, n_jobs=1))]).fit(X_cls[tr], y_bin[tr])
            oof_scores["random_forest"][te] = rf.predict_proba(X_cls[te])[:, 1]

            gb = Pipeline([("l2", L2RowNormalizer()), ("gb", HistGradientBoostingClassifier(max_iter=80, max_depth=5, class_weight="balanced", random_state=SEED + f))]).fit(X_cls[tr], y_bin[tr])
            oof_scores["hist_gbdt"][te] = gb.predict_proba(X_cls[te])[:, 1]

            red = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, metric="cosine", random_state=SEED + f, n_jobs=1)
            Z_tr = red.fit_transform(X_l2[tr])
            Z_te = red.transform(X_l2[te])
            uknn = KNeighborsClassifier(n_neighbors=7, weights="distance").fit(Z_tr, y_bin[tr])
            oof_scores["umap2d_knn"][te] = uknn.predict_proba(Z_te)[:, 1]

        bin_results[t_name] = {
            "n_positives": int(y_bin.sum()),
            "methods": {
                m: {
                    "oof_ap": round(float(average_precision_score(y_bin, sc)), 4),
                    "oof_auroc": round(float(roc_auc_score(y_bin, sc)), 4),
                }
                for m, sc in oof_scores.items()
            },
        }

    # Part B: Continuous NLCD fraction regression (OOF R^2 across 5 >30.9 km folds)
    reg_results = {}
    for t_name, y_reg in [("nlcd_impervious_pct", y_imp), ("nlcd_canopy_pct", y_can)]:
        oof_preds = {
            m: np.empty(len(y_reg), dtype=np.float64)
            for m in ["ridge_linear", "linear_svr", "knn_cosine", "mlp_2layer", "random_forest", "hist_gbdt", "umap2d_knn"]
        }
        for f in range(5):
            tr, te = folds != f, folds == f
            ridge = Pipeline([("sc", StandardScaler()), ("r", RidgeCV(alphas=np.logspace(-2, 6, 33)))]).fit(X_cls[tr], y_reg[tr])
            oof_preds["ridge_linear"][te] = ridge.predict(X_cls[te])

            svr = Pipeline([("sc", StandardScaler()), ("s", LinearSVR(C=1.0, epsilon=0.5, max_iter=5000, random_state=SEED + f))]).fit(X_cls[tr], y_reg[tr])
            oof_preds["linear_svr"][te] = svr.predict(X_cls[te])

            knn = KNeighborsRegressor(n_neighbors=7, metric="cosine", weights="distance").fit(X_l2[tr], y_reg[tr])
            oof_preds["knn_cosine"][te] = knn.predict(X_l2[te])

            mlp = Pipeline([("sc", StandardScaler()), ("m", MLPRegressor(hidden_layer_sizes=(128,), alpha=1.0, max_iter=400, random_state=SEED + f))]).fit(X_cls[tr], y_reg[tr])
            oof_preds["mlp_2layer"][te] = mlp.predict(X_cls[te])

            rf = Pipeline([("l2", L2RowNormalizer()), ("rf", RandomForestRegressor(n_estimators=150, max_depth=12, random_state=SEED + f, n_jobs=1))]).fit(X_cls[tr], y_reg[tr])
            oof_preds["random_forest"][te] = rf.predict(X_cls[te])

            gb = Pipeline([("l2", L2RowNormalizer()), ("gb", HistGradientBoostingRegressor(max_iter=80, max_depth=5, random_state=SEED + f))]).fit(X_cls[tr], y_reg[tr])
            oof_preds["hist_gbdt"][te] = gb.predict(X_cls[te])

            red = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, metric="cosine", random_state=SEED + f, n_jobs=1)
            Z_tr = red.fit_transform(X_l2[tr])
            Z_te = red.transform(X_l2[te])
            uknn = KNeighborsRegressor(n_neighbors=7, weights="distance").fit(Z_tr, y_reg[tr])
            oof_preds["umap2d_knn"][te] = uknn.predict(Z_te)

        reg_results[t_name] = {
            m: round(float(r2_score(y_reg, pr)), 4) for m, pr in oof_preds.items()
        }

    print(
        f"     S2 Co-located (N=223, co-occur={cooccur_count}) | "
        f"Mean Bin AP: LSVM={0.5*(bin_results['impervious_ge_5pct']['methods']['lsvm']['oof_ap']+bin_results['canopy_ge_10pct']['methods']['lsvm']['oof_ap']):.3f}, "
        f"CentDiff={0.5*(bin_results['impervious_ge_5pct']['methods']['centroid_diff']['oof_ap']+bin_results['canopy_ge_10pct']['methods']['centroid_diff']['oof_ap']):.3f}, "
        f"PosMean={0.5*(bin_results['impervious_ge_5pct']['methods']['pos_centroid']['oof_ap']+bin_results['canopy_ge_10pct']['methods']['pos_centroid']['oof_ap']):.3f}, "
        f"RF={0.5*(bin_results['impervious_ge_5pct']['methods']['random_forest']['oof_ap']+bin_results['canopy_ge_10pct']['methods']['random_forest']['oof_ap']):.3f}, "
        f"UMAP2D={0.5*(bin_results['impervious_ge_5pct']['methods']['umap2d_knn']['oof_ap']+bin_results['canopy_ge_10pct']['methods']['umap2d_knn']['oof_ap']):.3f}"
    )
    print(
        f"     S2 Regression Mean R2: Ridge={0.5*(reg_results['nlcd_impervious_pct']['ridge_linear']+reg_results['nlcd_canopy_pct']['ridge_linear']):.3f}, "
        f"LinSVR={0.5*(reg_results['nlcd_impervious_pct']['linear_svr']+reg_results['nlcd_canopy_pct']['linear_svr']):.3f}, "
        f"kNN={0.5*(reg_results['nlcd_impervious_pct']['knn_cosine']+reg_results['nlcd_canopy_pct']['knn_cosine']):.3f}, "
        f"RF={0.5*(reg_results['nlcd_impervious_pct']['random_forest']+reg_results['nlcd_canopy_pct']['random_forest']):.3f}, "
        f"UMAP2D={0.5*(reg_results['nlcd_impervious_pct']['umap2d_knn']+reg_results['nlcd_canopy_pct']['umap2d_knn']):.3f}"
    )
    return {
        "n_scenes": int(len(X_cls)),
        "n_spatial_clusters": int(len(np.unique(clusters))),
        "min_interfold_distance_km": round(float(min_km), 2),
        "cooccurring_imp5_and_can10_scenes": cooccur_count,
        "binary_retrieval": bin_results,
        "continuous_regression_r2": reg_results,
    }


def summarize_e1_aef_linear_vs_knn():
    d = json.loads(E1_AEF_JSON.read_text())
    out = {}
    for budget in ["1", "10", "max"]:
        lin_map = {
            s["task"]: float(s["point"])
            for s in d["summaries"]
            if s["arm"] == "clay_cls" and str(s["budget"]) == budget and s["head"] == "linear"
        }
        knn_map = {
            s["task"]: float(s["point"])
            for s in d["summaries"]
            if s["arm"] == "clay_cls" and str(s["budget"]) == budget and s["head"] == "knn3"
        }
        common = sorted(set(lin_map) & set(knn_map))
        deltas = [lin_map[t] - knn_map[t] for t in common]
        wins = sum(1 for x in deltas if x > 0)
        best_t = max(common, key=lambda t: lin_map[t] - knn_map[t])
        out[f"budget_{budget}"] = {
            "n_tasks": len(common),
            "linear_wins": wins,
            "mean_linear_minus_knn3": round(float(np.mean(deltas)), 4),
            "median_linear_minus_knn3": round(float(np.median(deltas)), 4),
            "max_gain_task": best_t,
            "max_gain_delta": round(float(lin_map[best_t] - knn_map[best_t]), 4),
        }
    return out


def main(out_path: Path = OUT_PATH):
    sys.stdout.reconfigure(line_buffering=True)
    t_all = time.perf_counter()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = pd.read_csv(EUROSAT_FEAT_DIR / "rows.csv")
    cls_all = np.load(EUROSAT_FEAT_DIR / "cls.npy")
    assert len(rows) == len(cls_all) == 25648

    tr_mask = (rows["split"] == "train").to_numpy()
    te_mask = (rows["split"] == "test").to_numpy()
    X_tr_all = cls_all[tr_mask].astype(np.float64)
    y_tr_all = rows.loc[tr_mask, "label"].to_numpy(dtype=np.int64)
    X_te = cls_all[te_mask].astype(np.float64)
    y_te = rows.loc[te_mask, "label"].to_numpy(dtype=np.int64)
    blocks_te = rows.loc[te_mask, "block_id"].to_numpy()

    p1 = run_p1_eurosat_multiclass(X_tr_all, y_tr_all, X_te, y_te, blocks_te)
    p2 = run_p2_concept_search(X_tr_all, y_tr_all, X_te, y_te)
    p3 = run_p3_label_noise_stress(X_tr_all, y_tr_all, X_te, y_te)
    p4 = run_p4_colocated_s2_nlcd()
    e1_lin_vs_knn = summarize_e1_aef_linear_vs_knn()

    sys.path.insert(0, str(MODEL_REPO))
    try:
        from provenance import provenance  # noqa: PLC0415  (experiments/suite/common/provenance.py)
        prov = provenance(__file__, {"latlon_order_p4": LATLON_ORDER, "eurosat_feat_dir": str(EUROSAT_FEAT_DIR.name),
                                     "out": out_path.name, "seed": SEED}, root=WORK_ROOT)
    except Exception as e:  # noqa: BLE001
        prov = {"error": f"provenance unavailable: {e}"}
    payload = {
        "provenance": prov,
        "latlon_order_p4": LATLON_ORDER,
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "runtime_seconds": round(time.perf_counter() - t_all, 2),
        "platform": platform.platform(),
        "python_version": sys.version.split()[0],
        "eurosat_split_summary": {
            "n_train": int(tr_mask.sum()),
            "n_train_blocks": int(rows.loc[tr_mask, "block_id"].nunique()),
            "n_test": int(te_mask.sum()),
            "n_test_blocks": int(rows.loc[te_mask, "block_id"].nunique()),
        },
        "p1_eurosat_multiclass_by_budget": p1,
        "p1_paired_contrasts_seed0_by_budget": P1_CONTRASTS,
        "p2_eurosat_binary_concept_search": p2,
        "p3_label_noise_stress_k25": p3,
        "p4_colocated_s2_nlcd_spatial_cv": p4,
        "e1_aef_15task_linear_vs_knn3": e1_lin_vs_knn,
    }
    out_path.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"[DONE] Wrote {out_path} in {payload['runtime_seconds']:.1f}s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Probe / LSVM benchmark on frozen Clay v1.5 embeddings.")
    ap.add_argument("--latlon-order", choices=["training", "documented"], default="training",
                    help="location-slot order for the S2/NLCD (P4) embeddings; EuroSAT features carry the order they were built with")
    ap.add_argument("--eurosat-feat-dir", default=None, help="directory with rows.csv and cls.npy (default data/eurosat/features)")
    ap.add_argument("--out-name", default="probe_and_lsvm_results.json")
    a = ap.parse_args()
    LATLON_ORDER = a.latlon_order
    if a.eurosat_feat_dir:
        EUROSAT_FEAT_DIR = Path(a.eurosat_feat_dir)
    main(OUT_PATH.parent / a.out_name)
