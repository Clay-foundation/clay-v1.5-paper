#!/usr/bin/env python3
"""How much does the NLCD probe R^2 (Lessons 3 and 5 baseline) depend on how the 171 spatial clusters are split into 5 folds?

run_clay_v15_experiments.py assigns clusters to folds with sklearn's GroupKFold, whose assignment differs between sklearn versions
(1.5.2 vs 1.9.1 give different folds for the same clusters). The baseline mean R^2 therefore moves by ~0.01-0.02 between
environments (pre-audit file 0.8392, regenerated 0.8256 in the documented lat/lon order). This script re-embeds the 223 NLCD
chips (CPU) and re-scores the full-1,024-d, prefix-64 and random-orthogonal-64 Ridge probes under
  - the folds GroupKFold gives in the current environment, and
  - N random assignments of the same clusters to 5 folds (every assignment keeps the > 30 km cluster separation),
for both lat/lon orders, and reports mean, sd, min, max over the random assignments.

Run:  python experiments/fold_sensitivity_nlcd.py [--n-assign 100] [--out experiments/results/fold_sensitivity_nlcd.json]
Environment: CLAY_V15_CKPT, CLAY_S2_NLCD_ROOT (as run_clay_v15_experiments.py). About 15 minutes on 8 CPU threads.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_clay_v15_experiments as R  # noqa: E402


def main(n_assign: int, out: Path):
    enc_cpu, _enc_dev, _device, metadata, _art = R.load_verified_clay()
    chips = pd.read_parquet(R.S2_ROOT / "data/chips.parquet")
    lab = pd.read_parquet(R.S2_ROOT / "results/labels_nlcd.parquet")
    px = {p.stem: p for p in sorted((R.S2_ROOT / "data/pixels").glob("*.npy"))}
    df = chips[chips["chip_id"].isin(px)].merge(lab, on="chip_id", how="left").sort_values("chip_id").reset_index(drop=True)
    raw = np.stack([np.load(px[c]) for c in df["chip_id"]])[:, R.CLAY_S2_BANDS_12_IDX]
    harm = np.clip(raw.astype(np.float32) - 1000.0, 0.0, None)
    s2 = metadata["sentinel-2-l2a"]
    means = np.array([s2["bands"]["mean"][b] for b in R.CLAY_S2_BAND_NAMES], dtype=np.float32)
    stds = np.array([s2["bands"]["std"][b] for b in R.CLAY_S2_BAND_NAMES], dtype=np.float32)
    waves = [float(s2["bands"]["wavelength"][b]) for b in R.CLAY_S2_BAND_NAMES]
    norm = (harm - means[None, :, None, None]) / stds[None, :, None, None]
    times = np.stack([R.normalize_timestamp(t) for t in df["datetime"]])
    m = df["y_impervious"].notna().to_numpy()
    d = df[m].reset_index(drop=True)
    clusters, folds_sk, _ = R.assign_spatial_clusters(d["lat"].to_numpy(), d["lon"].to_numpy(), 30.0)
    y_imp, y_can = d["y_impervious"].to_numpy(float), d["y_canopy"].to_numpy(float)
    uniq = np.unique(clusters)
    res = {}
    t0 = time.time()
    for order in ("documented", "training"):
        R.LATLON_ORDER = order
        ll = np.stack([R.normalize_latlon(a, b) for a, b in zip(df["lat"], df["lon"])])
        cls, _ = R.forward_tokens(enc_cpu, norm, times, ll, waves, float(s2["gsd"]), "cpu", batch_size=16, return_patches=False)
        X = cls[m]
        print(order, "embedded", round(time.time() - t0), "s", flush=True)

        def score(folds, cols):
            Xs = X @ cols if isinstance(cols, np.ndarray) else X[:, cols]
            r = [R.ridge_oof_r2(Xs, y, folds, clusters, n_boot=0)["r2"] for y in (y_imp, y_can)]
            return float(np.mean(r))

        full, pre = slice(0, 1024), slice(0, 64)
        sk = {"full_1024_mean_r2": round(score(folds_sk, full), 4), "prefix_64_mean_r2": round(score(folds_sk, pre), 4)}
        # Issue #30 control: a random ORTHOGONAL 64-d projection of the class token (first 64 columns of a random rotation)
        rng_q = np.random.default_rng(R.SEED + 7)
        orth = []
        for _ in range(10):
            Q, _r = np.linalg.qr(rng_q.standard_normal((1024, 1024)))
            orth.append(score(folds_sk, Q[:, :64]))
        sk["random_orthogonal_64_mean_r2_over_10_rotations_groupkfold"] = {"mean": round(float(np.mean(orth)), 4), "sd": round(float(np.std(orth)), 4),
                                                                              "min": round(float(np.min(orth)), 4), "max": round(float(np.max(orth)), 4)}
        rng = np.random.default_rng(R.SEED)
        full_v, pre_v = [], []
        for k in range(n_assign):
            perm = rng.permutation(len(uniq))
            fold_of = {c: i % 5 for i, c in zip(range(len(uniq)), uniq[perm])}
            folds = np.array([fold_of[c] for c in clusters])
            full_v.append(score(folds, full))
            pre_v.append(score(folds, pre))
        st = lambda v: {"mean": round(float(np.mean(v)), 4), "sd": round(float(np.std(v)), 4), "min": round(float(np.min(v)), 4), "max": round(float(np.max(v)), 4)}  # noqa: E731
        res[order] = {"groupkfold_this_environment": sk, "random_fold_assignments": {"n": n_assign, "full_1024_mean_r2": st(full_v), "prefix_64_mean_r2": st(pre_v),
                      "prefix_minus_full": st(np.array(pre_v) - np.array(full_v))}}
        print(order, json.dumps(res[order]), flush=True)
    payload = {"n_chips_nlcd": int(m.sum()), "n_clusters": int(len(uniq)), "results": res, "note": "random assignments split the same clusters into 5 folds of ~equal cluster counts; "
               "compare the sklearn GroupKFold values with the spread over random assignments"}
    sys.path.insert(0, str(R.MODEL_REPO))
    try:
        from provenance import provenance
        payload["provenance"] = provenance(__file__, {"n_assign": n_assign, "out": out.name}, root=R.WORK_ROOT)
    except Exception as e:  # noqa: BLE001
        payload["provenance"] = {"error": str(e)}
    out.write_text(json.dumps(payload, indent=1) + "\n")
    print("wrote", out.name)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-assign", type=int, default=100)
    ap.add_argument("--out", default=str(HERE / "results" / "fold_sensitivity_nlcd.json"))
    a = ap.parse_args()
    main(a.n_assign, Path(a.out))
