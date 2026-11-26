#!/usr/bin/env python3
"""Lesson 4 control (issue #30): give the non-linear baselines the same inner-CV tuning budget as the LSVM.

run_probe_and_lsvm_benchmark.py tunes C for the LSVM and logistic probes with a 3-fold inner CV (5 candidate values for the
LSVM) but fits RandomForest, HistGradientBoosting and k-NN with fixed settings. This script repeats the spatially blocked
EuroSAT comparison for the budgets 50 and 200 labels per class, with the SAME train/test split, the same seeds and sampling
as that script, and tunes
  - LinearSVC C                         {0.05, 0.2, 1, 5, 20}           (as in the main script)
  - RandomForest  max_features x min_samples_leaf   {sqrt, 0.1} x {1, 3}   (n_estimators 200)
  - HistGradientBoosting  learning_rate x max_depth {0.05, 0.1} x {3, 6}   (max_iter 100)
  - cosine k-NN   k {1, 3, 5, 10, 20}  x weights {uniform, distance}
by the inner 3-fold stratified CV macro-F1 on the sampled training labels (4-10 fits per candidate), then scores the test split
once. Output: macro-F1 per method, seed and budget, means, and LSVM minus tuned-baseline differences.

Run:  python experiments/probe_equal_tuning_control.py --feat-dir data/eurosat/features [--seeds 3] [--out experiments/results/probe_equal_tuning_control.json]
Features: rows.csv and cls.npy built by experiments/build_lesson_inputs.py eurosat. CPU only; n_jobs capped at 8.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC

HERE = Path(__file__).resolve().parent
SEED = 42


def l2(X):
    return X / np.clip(np.linalg.norm(X, axis=1, keepdims=True), 1e-12, None)


def inner_cv_pick(make, grid, X, y, seed):
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
    best, best_s = None, -1.0
    for params in grid:
        sc = []
        for a, b in skf.split(X, y):
            m = make(params).fit(X[a], y[a])
            sc.append(f1_score(y[b], m.predict(X[b]), average="macro"))
        if np.mean(sc) > best_s:
            best, best_s = params, float(np.mean(sc))
    return best, best_s


def main(feat_dir: Path, out: Path, n_seeds: int):
    rows = pd.read_csv(feat_dir / "rows.csv")
    cls = np.load(feat_dir / "cls.npy")
    tr = (rows["split"] == "train").to_numpy()
    te = (rows["split"] == "test").to_numpy()
    Xtr_all, ytr_all = l2(cls[tr].astype(np.float64)), rows.loc[tr, "label"].to_numpy(np.int64)
    Xte, yte = l2(cls[te].astype(np.float64)), rows.loc[te, "label"].to_numpy(np.int64)
    res = {}
    t0 = time.time()
    for b in (50, 200):
        per = {m: [] for m in ("lsvm_tuned", "rf_fixed", "rf_tuned", "hgb_fixed", "hgb_tuned", "knn_fixed_k5", "knn_tuned")}
        chosen = {m: [] for m in ("lsvm_tuned", "rf_tuned", "hgb_tuned", "knn_tuned")}
        for s in range(n_seeds):
            rng = np.random.default_rng(1000 + s * 37 + b)  # same sampling as run_probe_and_lsvm_benchmark.py
            idx = np.concatenate([rng.choice(np.where(ytr_all == c)[0], size=b, replace=False) for c in range(10)])
            X, y = Xtr_all[idx], ytr_all[idx]
            f1 = lambda m: float(f1_score(yte, m.predict(Xte), average="macro"))  # noqa: E731

            p, _ = inner_cv_pick(lambda C: LinearSVC(C=C, class_weight="balanced", tol=1e-3, max_iter=1500, random_state=s),
                                 [0.05, 0.2, 1.0, 5.0, 20.0], X, y, s)
            per["lsvm_tuned"].append(f1(LinearSVC(C=p, class_weight="balanced", tol=1e-3, max_iter=2000, random_state=s).fit(X, y)))
            chosen["lsvm_tuned"].append(p)

            per["rf_fixed"].append(f1(RandomForestClassifier(n_estimators=150, max_depth=16, random_state=s, n_jobs=8).fit(X, y)))
            g = [dict(max_features=mf, min_samples_leaf=msl) for mf, msl in product(["sqrt", 0.1], [1, 3])]
            p, _ = inner_cv_pick(lambda q: RandomForestClassifier(n_estimators=200, random_state=s, n_jobs=8, **q), g, X, y, s)
            per["rf_tuned"].append(f1(RandomForestClassifier(n_estimators=200, random_state=s, n_jobs=8, **p).fit(X, y)))
            chosen["rf_tuned"].append(p)

            per["hgb_fixed"].append(f1(HistGradientBoostingClassifier(max_iter=100, max_depth=6, random_state=s).fit(X, y)))
            g = [dict(learning_rate=lr, max_depth=d) for lr, d in product([0.05, 0.1], [3, 6])]
            p, _ = inner_cv_pick(lambda q: HistGradientBoostingClassifier(max_iter=100, random_state=s, **q), g, X, y, s)
            per["hgb_tuned"].append(f1(HistGradientBoostingClassifier(max_iter=100, random_state=s, **p).fit(X, y)))
            chosen["hgb_tuned"].append(p)

            per["knn_fixed_k5"].append(f1(KNeighborsClassifier(n_neighbors=5, metric="cosine", weights="distance").fit(X, y)))
            g = [dict(n_neighbors=k, weights=w) for k, w in product([1, 3, 5, 10, 20], ["uniform", "distance"])]
            p, _ = inner_cv_pick(lambda q: KNeighborsClassifier(metric="cosine", **q), g, X, y, s)
            per["knn_tuned"].append(f1(KNeighborsClassifier(metric="cosine", **p).fit(X, y)))
            chosen["knn_tuned"].append(p)
            print(f"budget {b} seed {s} done {time.time() - t0:.0f}s", {k: round(v[-1], 4) for k, v in per.items()}, flush=True)
        mean = {k: round(float(np.mean(v)), 4) for k, v in per.items()}
        res[str(b)] = {
            "per_seed_macro_f1": {k: [round(x, 4) for x in v] for k, v in per.items()},
            "mean_macro_f1": mean,
            "lsvm_minus_other": {k: round(mean["lsvm_tuned"] - mean[k], 4) for k in mean if k != "lsvm_tuned"},
            "chosen_hyperparameters": chosen,
        }
    payload = {"n_seeds": n_seeds, "budgets_labels_per_class": [50, 200], "inner_cv": "3-fold stratified, macro-F1",
               "test_split": "spatially blocked EuroSAT test blocks (same as run_probe_and_lsvm_benchmark.py)",
               "n_test": int(te.sum()), "results": res, "runtime_seconds": round(time.time() - t0, 1)}
    sys.path.insert(0, str(HERE / "suite" / "common"))
    try:
        from provenance import provenance
        payload["provenance"] = provenance(__file__, {"feat_dir": feat_dir.name, "seeds": n_seeds, "out": out.name}, root=HERE.parent)
    except Exception as e:  # noqa: BLE001
        payload["provenance"] = {"error": str(e)}
    out.write_text(json.dumps(payload, indent=1) + "\n")
    print("wrote", out.name)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--feat-dir", default=str(HERE.parent / "data/eurosat/features"))
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--out", default=str(HERE / "results" / "probe_equal_tuning_control.json"))
    a = ap.parse_args()
    main(Path(a.feat_dir), Path(a.out), a.seeds)
