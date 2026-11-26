#!/usr/bin/env python3
"""Micro-benchmark of the ELLE probe's per-chip cost, independent of the chips: a (N, 1024) float32 matrix and a
folded Ridge weight vector (random values; the cost does not depend on them). Reports best-of-7 seconds per chip for
(a) a batched dot product, (b) the same with float64, (c) a single-vector dot product, each pinned to 1 BLAS thread.
Also records the 1-minute load average, because shared-machine contention inflates the numbers.
Usage: OMP_NUM_THREADS=1 python latency_bench.py --out ../../results/elle_latency_microbench.json"""
import argparse, json, os, time
os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ.setdefault("OPENBLAS_NUM_THREADS", "1"); os.environ.setdefault("MKL_NUM_THREADS", "1")
import numpy as np

ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--n", type=int, default=100000)
a = ap.parse_args()
rng = np.random.default_rng(0)
X = rng.standard_normal((a.n, 1024)).astype(np.float32); w = rng.standard_normal(1024).astype(np.float32); b = np.float32(0.3)
X64 = X.astype(np.float64); w64 = w.astype(np.float64)
def best(fn, reps):
    out = []
    for _ in range(7):
        t0 = time.perf_counter()
        for _ in range(reps): fn()
        out.append((time.perf_counter() - t0) / reps)
    return min(out)
x1 = X[:1]
res = {"n_chips": a.n, "blas_threads": 1, "load_avg_1min_at_start": os.getloadavg()[0],
       "seconds_per_chip": {"batched_dot_float32": best(lambda: X @ w + b, 5) / a.n,
                            "batched_dot_float64": best(lambda: X64 @ w64 + 0.3, 5) / a.n,
                            "single_vector_dot_float32": best(lambda: x1 @ w + b, 20000)},
       "load_avg_1min_at_end": os.getloadavg()[0], "numpy": np.__version__,
       "note": "synthetic data of the right shape; the probe is a 1,024-term dot product after folding the scaler into the weights"}
json.dump(res, open(a.out, "w"), indent=1); print(json.dumps(res, indent=1))
