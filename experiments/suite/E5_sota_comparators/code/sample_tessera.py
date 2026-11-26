"""E5: sample TESSERA v1.1 (dClimate zarr via geotessera.store.GeoTesseraZarr) at E5 points.
mode pixel: 10 m embedding at the point (bulk per zone/year). mode patch: mean of 128x128 px patch (priority order, hard stop).
"""
import argparse, json, os, time
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import numpy as np, pandas as pd
from geotessera.store import GeoTesseraZarr
ap = argparse.ArgumentParser()
ap.add_argument("--points", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--mode", choices=["pixel", "patch"], required=True); ap.add_argument("--workers", type=int, default=16)
ap.add_argument("--hard_stop_s", type=float, default=10800); ap.add_argument("--limit", type=int, default=0)
a = ap.parse_args()
P = pd.read_csv(a.points).sort_values(["prio", "pid"])
if a.limit: P = P.head(a.limit)
gt = GeoTesseraZarr(); print("store", gt.url, gt.years, gt.model_version, flush=True)
t0 = time.time()
if a.mode == "pixel":
    X = np.full((len(P), 128), np.nan, np.float32)
    for yr in sorted(P.year.unique()):
        idx = np.where(P.year.values == yr)[0]
        for i in range(0, len(idx), 2000):
            ii = idx[i:i + 2000]
            X[ii] = gt.sample_points(list(zip(P.x.values[ii], P.y.values[ii])), year=int(yr))
            print(yr, i + len(ii), len(idx), round(time.time() - t0), int(np.isnan(X[ii]).all(1).sum()), flush=True)
    np.savez(a.out, pid=P.pid.values, tessera_pixel=X)
else:
    done = set(json.loads(l)["pid"] for l in open(a.out)) if os.path.exists(a.out) else set()
    P = P[~P.pid.isin(done)]; fo = open(a.out, "a"); n = [0, 0]
    def one(r):
        patch, tr, crs = gt.read_patch(float(r.x), float(r.y), int(r.year), 128)
        v = patch.reshape(-1, patch.shape[-1]); ok = np.isfinite(v).all(1)
        return {"pid": int(r.pid), "n_valid": int(ok.sum()),
                "tessera_mean128": v[ok].mean(0).round(5).tolist() if ok.any() else None, "tessera_pixel": patch[64, 64].round(5).tolist() if np.isfinite(patch[64, 64]).all() else None}
    ex = ThreadPoolExecutor(max(a.workers, 64)); it = (r for _, r in P.iterrows()); futs = set(); last = 0
    while True:
        while len(futs) < 3 * a.workers and time.time() - t0 < a.hard_stop_s:
            r = next(it, None)
            if r is None: break
            futs.add(ex.submit(one, r))
        if not futs: break
        dn, futs = wait(futs, timeout=30, return_when=FIRST_COMPLETED)
        for fu in dn:
            try: fo.write(json.dumps(fu.result()) + "\n"); n[0] += 1
            except Exception as e:
                n[1] += 1
                if n[1] < 20: print("ERR", type(e).__name__, str(e)[:200], flush=True)
        fo.flush()
        if time.time() - last > 120: last = time.time(); print(n, round(time.time() - t0), flush=True)
    fo.close(); ex.shutdown(wait=False, cancel_futures=True); print("done", n, flush=True)
print("wall", round(time.time() - t0, 1))
