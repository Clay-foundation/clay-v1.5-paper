"""E5: sample AlphaEarth Foundations annual embeddings (Source Cooperative COG mirror) at E5 points.
Mode ov32: overview factor 32 (320 m px; overview = mean of dequantised 10 m embeddings, renormalised).
Mode pixel: full-res 10 m pixel at the point.  Dequantisation: ((v/127.5)**2)*sign(v); -128 nodata.
"""
import argparse, os, time, json, threading
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
os.environ.setdefault("AWS_NO_SIGN_REQUEST", "YES"); os.environ.setdefault("AWS_REGION", "us-west-2")
os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR"); os.environ.setdefault("GDAL_HTTP_MAX_RETRY", "6")
os.environ.setdefault("GDAL_HTTP_RETRY_DELAY", "2"); os.environ.setdefault("VSI_CACHE", "FALSE")
import numpy as np, pandas as pd, rasterio
from rasterio.windows import Window

ap = argparse.ArgumentParser()
ap.add_argument("--blocks", required=True); ap.add_argument("--points", required=True)
ap.add_argument("--mode", choices=["ov32", "pixel"], required=True)
ap.add_argument("--out", required=True); ap.add_argument("--workers", type=int, default=16)
ap.add_argument("--hard_stop_s", type=float, default=3 * 3600); ap.add_argument("--limit", type=int, default=0)
a = ap.parse_args()
D = pd.read_csv(a.blocks).merge(pd.read_csv(a.points)[["pid", "prio"]], on="pid", how="inner")
deq = lambda v: (v.astype(np.float32) / 127.5) ** 2 * np.sign(v.astype(np.float32))
vsi = lambda p: p.replace("s3://", "/vsis3/")
done = set(json.loads(l)["pid"] for l in open(a.out)) if os.path.exists(a.out) else set()
D = D[~D.pid.isin(done)]
lock = threading.Lock(); fo = open(a.out, "a"); t0 = time.time(); n = [0, 0]

def do_file(path, g):
    with rasterio.open(vsi(path), overview_level=4) as s:  # overviews [2,4,8,16,32,..] -> index 4 = factor 32
        A = s.read(); tr = s.transform; full = s.width * 32
    recs = []
    for _, r in g.iterrows():
        cx, cy = (r.col + 0.5) / 32, (r.row + 0.5) / 32
        c0 = int(round(cx - 2)); r0 = int(round(cy - 2))
        win = A[:, max(r0, 0):max(r0, 0) + 4, max(c0, 0):max(c0, 0) + 4].reshape(64, -1)
        ok = (win != -128).all(0)
        one = A[:, min(int(cy), A.shape[1] - 1), min(int(cx), A.shape[2] - 1)]
        recs.append({"pid": int(r.pid), "n_valid": int(ok.sum()),
                     "aef_mean128": deq(win[:, ok]).mean(1).round(5).tolist() if ok.any() else None,
                     "aef_ov32_px": deq(one).round(5).tolist() if (one != -128).all() else None})
    return recs

def do_pixel(r):
    with rasterio.open(vsi(r.path)) as s:
        v = s.read(window=Window(int(r.col), int(r.row), 1, 1))[:, 0, 0]
    return [{"pid": int(r.pid), "aef_pixel": deq(v).round(5).tolist() if (v != -128).all() else None}]

if a.mode == "ov32":
    groups = sorted(D.groupby("path"), key=lambda kv: kv[1].prio.min())
    jobs = groups[:a.limit] if a.limit else groups; fn = lambda j: do_file(*j)
else:
    D = D.sort_values(["prio", "pid"]); D = D.head(a.limit) if a.limit else D
    jobs = [r for _, r in D.iterrows()]; fn = do_pixel
print(a.mode, "jobs", len(jobs), "already", len(done), flush=True)
ex = ThreadPoolExecutor(a.workers); it = iter(jobs); futs = set(); last = 0
while True:
    while len(futs) < 3 * a.workers and time.time() - t0 < a.hard_stop_s:
        j = next(it, None)
        if j is None: break
        futs.add(ex.submit(fn, j))
    if not futs: break
    dn, futs = wait(futs, timeout=30, return_when=FIRST_COMPLETED)
    for fu in dn:
        try:
            recs = fu.result()
            with lock:
                for rr in recs: fo.write(json.dumps(rr) + "\n")
                n[0] += len(recs)
        except Exception as e:
            n[1] += 1
            if n[1] < 20: print("ERR", type(e).__name__, str(e)[:200], flush=True)
    fo.flush()
    if time.time() - last > 120:
        last = time.time(); print(f"{n[0]} recs {n[1]} err {time.time()-t0:.0f}s", flush=True)
    if time.time() - t0 > a.hard_stop_s and not futs: print("hard stop", flush=True); break
fo.close(); ex.shutdown(wait=False, cancel_futures=True); print("done", n, round(time.time() - t0, 1), flush=True)
