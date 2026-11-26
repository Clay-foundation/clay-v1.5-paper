"""E1: fetch one least-cloudy Sentinel-2 L2A chip per request from Planetary Computer and embed it.

Requests CSV columns: chip_id, lon, lat, t0, t1 (ISO dates; search window).
Per request: STAC search (sentinel-2-l2a, point intersects, [t0,t1]); candidates sorted by eo:cloud_cover;
for up to MAX_CAND candidates read the chip's SCL (20 m) and take the first with cloud/shadow/nodata
fraction <= CLEAR_OK, else the lowest seen. Then read the 10 Clay bands as raw uint16 on a 10 m grid
(20 m bands nearest-upsampled x2, as rasterio does with out_shape). No harmonisation (raw PC values).
Chip: CHIP x CHIP px centred on the point, snapped to the 20 m grid of the item.
Embeddings per chip: Clay v1.5 CLS / mean / centre (2x2 central patch tokens), DINOv2-L/14-reg4 on display RGB
(224 px), DOFA-L (torchgeo, 10 bands with wavelengths, Clay S2 z-score, 224 px), per-band mean/std (raw).
Pixels are not kept. Output: out/shard_XXXX.npz (+ out/meta.csv), resumable.
"""
import argparse, csv, json, math, os, sys, time, threading, queue, traceback
from concurrent.futures import ThreadPoolExecutor, wait as fwait, FIRST_COMPLETED
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
os.environ.setdefault("CPL_VSIL_CURL_ALLOWED_EXTENSIONS", ".tif")
os.environ.setdefault("GDAL_HTTP_MAX_RETRY", "6")
os.environ.setdefault("GDAL_HTTP_RETRY_DELAY", "2")
os.environ.setdefault("GDAL_HTTP_MERGE_CONSECUTIVE_RANGES", "YES")
os.environ.setdefault("VSI_CACHE", "FALSE")
os.environ.setdefault("GDAL_HTTP_TIMEOUT", "30")
os.environ.setdefault("GDAL_HTTP_CONNECTTIMEOUT", "10")

import numpy as np
import rasterio
from rasterio.windows import Window
from rasterio.enums import Resampling
from pyproj import Transformer
import pystac_client, planetary_computer

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "common"))
BANDS = ["blue", "green", "red", "rededge1", "rededge2", "rededge3", "nir", "nir08", "swir16", "swir22"]
ASSET = {"blue": "B02", "green": "B03", "red": "B04", "rededge1": "B05", "rededge2": "B06", "rededge3": "B07",
         "nir": "B08", "nir08": "B8A", "swir16": "B11", "swir22": "B12"}
RES = {"B02": 10, "B03": 10, "B04": 10, "B08": 10, "B05": 20, "B06": 20, "B07": 20, "B8A": 20, "B11": 20, "B12": 20}
BAD_SCL = (0, 1, 3, 8, 9, 10)
MAX_CAND, CLEAR_OK = 4, 0.05
STAC = "https://planetarycomputer.microsoft.com/api/stac/v1"
_tl = threading.local()


class Limiter:
    """Shared adaptive rate limiter for STAC API calls (Planetary Computer rate-limits per client)."""
    def __init__(self, rps=3.0, lo=0.3, hi=6.0):
        self.rps, self.lo, self.hi = rps, lo, hi
        self.lock = threading.Lock(); self.next_t = time.time(); self.pause_until = 0.0; self.last_429 = time.time()

    def wait(self):
        with self.lock:
            now = time.time()
            if now - self.last_429 > 300 and self.rps < self.hi:
                self.rps = min(self.hi, self.rps + 0.5); self.last_429 = now
            t = max(self.next_t, now, self.pause_until)
            self.next_t = t + 1.0 / self.rps
        time.sleep(max(0.0, t - time.time()))

    def throttled(self):
        with self.lock:
            self.rps = max(self.lo, self.rps / 2); self.pause_until = time.time() + 60; self.last_429 = time.time()
            print(f"rate-limited: rps -> {self.rps:.2f}", flush=True)


LIM = Limiter()


def stac_search(**kw):
    for k in range(12):
        LIM.wait()
        try:
            return list(catalog().search(**kw).items())
        except Exception as e:
            msg = str(e).lower()
            if "rate limit" in msg or "429" in msg or "slow down" in msg:
                LIM.throttled(); continue
            if k >= 3:
                raise
            time.sleep(2 * (2 ** k))
    raise RuntimeError("rate-limited: retries exhausted")


def catalog():
    if not hasattr(_tl, "cat"):
        try:
            _tl.cat = pystac_client.Client.open(STAC, modifier=planetary_computer.sign_inplace, timeout=30)
        except TypeError:
            _tl.cat = pystac_client.Client.open(STAC, modifier=planetary_computer.sign_inplace)
    return _tl.cat


def retry(fn, n=5, base=2.0):
    for k in range(n):
        try:
            return fn()
        except Exception:
            if k == n - 1:
                raise
            time.sleep(base * (2 ** k))


def read_window(href, x0, y1, size_m, out_px, res):
    """Read the square [x0, x0+size_m] x [y1-size_m, y1] (item CRS) from a COG, boundless, fill 0."""
    with rasterio.open(href) as src:
        t = src.transform
        col = (x0 - t.c) / t.a; row = (y1 - t.f) / t.e
        n = int(round(size_m / res))
        w = Window(int(round(col)), int(round(row)), n, n)
        return src.read(1, window=w, out_shape=(out_px, out_px), boundless=True, fill_value=0,
                        resampling=Resampling.nearest)


def fetch_one(req, chip_px):
    lon, lat = float(req["lon"]), float(req["lat"])
    size_m = chip_px * 10
    geom = {"type": "Point", "coordinates": [lon, lat]}
    # server-side sort by scene cloud cover (pgstac sort extension), one page of 2*MAX_CAND items
    items = stac_search(collections=["sentinel-2-l2a"], intersects=geom, datetime=f"{req['t0']}/{req['t1']}",
                        max_items=2 * MAX_CAND, limit=2 * MAX_CAND,
                        sortby=[{"field": "eo:cloud_cover", "direction": "asc"}])
    if not items:
        return None, {"status": "no_items"}
    items.sort(key=lambda it: (it.properties.get("eo:cloud_cover", 100.0), it.id))
    best = None
    for it in items[:MAX_CAND]:
        epsg = it.properties.get("proj:epsg") or int(str(it.properties.get("proj:code", "EPSG:0")).split(":")[-1])
        tr = Transformer.from_crs(4326, epsg, always_xy=True)
        cx, cy = tr.transform(lon, lat)
        # snap chip origin to the 20 m grid (S2 grids are anchored at multiples of 60 m in UTM)
        x0 = math.floor((cx - size_m / 2) / 20.0) * 20.0
        y1 = math.floor((cy + size_m / 2) / 20.0) * 20.0
        try:
            scl = retry(lambda: read_window(it.assets["SCL"].href, x0, y1, size_m, chip_px // 2, 20), n=3)
        except Exception as e:
            continue
        bad = float(np.isin(scl, BAD_SCL).mean()); nod = float((scl == 0).mean())
        cand = (bad, it, epsg, x0, y1, nod)
        if best is None or bad < best[0]:
            best = cand
        if bad <= CLEAR_OK:
            break
    if best is None:
        return None, {"status": "scl_unreadable", "n_items": len(items)}
    bad, it, epsg, x0, y1, nod = best
    arr = np.zeros((len(BANDS), chip_px, chip_px), np.uint16)
    for b, name in enumerate(BANDS):
        a = ASSET[name]
        arr[b] = retry(lambda: read_window(it.assets[a].href, x0, y1, size_m, chip_px, RES[a]), n=3)
    zero = (arr == 0).any(0)
    zf = float(zero.mean())
    meta = {"status": "ok", "item": it.id, "datetime": it.properties["datetime"],
            "scene_cloud": it.properties.get("eo:cloud_cover"), "chip_bad_scl": round(bad, 4),
            "nodata_frac": round(zf, 4), "baseline": it.properties.get("s2:processing_baseline"),
            "epsg": epsg, "x0": x0, "y1": y1, "n_items": len(items)}
    if zf > 0.5:
        meta["status"] = "mostly_nodata"
        return None, meta
    if zf > 0:  # fill nodata with per-band median of valid pixels (recorded via nodata_frac)
        for b in range(len(BANDS)):
            v = arr[b][~zero]
            arr[b][zero] = np.median(v) if v.size else 1
    return arr, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--requests", required=True)
    ap.add_argument("--out", default="out")
    ap.add_argument("--chip", type=int, default=128)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--no-models", action="store_true")
    ap.add_argument("--stac-rps", type=float, default=3.0)
    args = ap.parse_args()
    LIM.rps = args.stac_rps
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    reqs = list(csv.DictReader(open(args.requests)))
    metap = out / "meta.csv"
    done = set()
    if metap.exists():
        done = {r["chip_id"] for r in csv.DictReader(open(metap)) if not r["status"].startswith("error")}
    reqs = [r for r in reqs if r["chip_id"] not in done]
    if args.limit:
        reqs = reqs[:args.limit]
    print(f"{len(done)} done, {len(reqs)} to fetch", flush=True)

    import torch
    from clayrt import ClayRuntime, DinoV2Runtime, to_display_rgb, platform_spec
    spec = platform_spec("sentinel-2-l2a")
    if not args.no_models:
        clay = ClayRuntime(device="cuda")
        dino = DinoV2Runtime(size=224)
        import timm  # timm builds this model at 518 px; rebuild at 224 px (pos-embed interpolated at load)
        dino.m = timm.create_model(dino.name, pretrained=True, num_classes=0, img_size=224).eval().to("cuda")
        from torchgeo.models import dofa_large_patch16_224, DOFALarge16_Weights
        dofa = dofa_large_patch16_224(weights=DOFALarge16_Weights.DOFA_MAE).eval().cuda()
        json.dump({"clay": clay.evidence, "dino": dino.name, "dofa": str(DOFALarge16_Weights.DOFA_MAE.url),
                   "torch": torch.__version__}, open(out / "runtime.json", "w"), indent=1)
    mean = torch.tensor(spec["means"]).view(1, -1, 1, 1); std = torch.tensor(spec["stds"]).view(1, -1, 1, 1)

    fields = ["chip_id", "status", "item", "datetime", "scene_cloud", "chip_bad_scl", "nodata_frac", "baseline",
              "epsg", "x0", "y1", "n_items", "shard", "fetch_s"]
    newfile = not metap.exists()
    mf = open(metap, "a", newline=""); mw = csv.DictWriter(mf, fieldnames=fields, extrasaction="ignore")
    if newfile:
        mw.writeheader()
    shard_i = len(list(out.glob("shard_*.npz")))
    buf = []  # (req, arr, meta)
    t_start = time.time(); n_ok = n_fail = 0

    def flush():
        nonlocal shard_i, buf
        if not buf:
            return
        X = np.stack([b[1] for b in buf]).astype(np.float32)
        ids = np.array([b[0]["chip_id"] for b in buf])
        feats = {"ids": ids}
        Xf = X.reshape(len(X), X.shape[1], -1)
        feats["stats"] = np.concatenate([Xf.mean(2), Xf.std(2)], 1).astype(np.float32)
        if not args.no_models:
            ll = [(float(b[0]["lat"]), float(b[0]["lon"])) for b in buf]
            dd = [datetime.fromisoformat(b[2]["datetime"].replace("Z", "+00:00")) for b in buf]
            for order, sfx in (("training", ""), ("documented", "_doc")):
                o = clay.embed(X, platform="sentinel-2-l2a", latlon=ll, date=dd, batch_size=64, latlon_order=order)
                n = o["patches"].shape[1]; c0 = n // 2 - 1
                feats["clay_cls" + sfx] = o["cls"].astype(np.float16); feats["clay_mean" + sfx] = o["mean"].astype(np.float16)
                feats["clay_center" + sfx] = o["patches"][:, c0:c0 + 2, c0:c0 + 2].mean((1, 2)).astype(np.float16)
                del o
            d = dino.embed(to_display_rgb(X, "sentinel-2-l2a"))
            feats["dino_cls"] = d["cls"].astype(np.float16); feats["dino_mean"] = d["mean"].astype(np.float16)
            with torch.inference_mode():
                fs = []
                for i in range(0, len(X), 64):
                    x = (torch.from_numpy(X[i:i + 64]) - mean) / std
                    x = torch.nn.functional.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
                    fs.append(dofa.forward_features(x.cuda(), spec["waves"]).float().cpu())
                feats["dofa"] = torch.cat(fs).numpy().astype(np.float16)
            for k, v in feats.items():
                if k != "ids" and not np.isfinite(v.astype(np.float32)).all():
                    raise ValueError(f"non-finite {k}")
        np.savez(out / f"shard_{shard_i:04d}.npz", **feats)
        (out / "pixels").mkdir(exist_ok=True)
        np.savez_compressed(out / "pixels" / f"pixels_{shard_i:04d}.npz", ids=ids, pixels=X.astype(np.uint16))
        for b in buf:
            m = dict(b[2]); m["chip_id"] = b[0]["chip_id"]; m["shard"] = shard_i
            mw.writerow(m)
        mf.flush(); shard_i += 1; buf = []

    STARTED = {}

    def task(r):
        t0 = time.time(); STARTED[r["chip_id"]] = t0
        try:
            arr, meta = fetch_one(r, args.chip)
        except Exception as e:
            arr, meta = None, {"status": "error:" + type(e).__name__ + ":" + str(e)[:120].replace("\n", " ")}
        meta["fetch_s"] = round(time.time() - t0, 2)
        return r, arr, meta

    ex = ThreadPoolExecutor(args.workers)
    pending = {}; it = iter(reqs); k = 0; TASK_TIMEOUT = 900
    def handle(r, arr, meta):
        nonlocal n_ok, n_fail, k
        k += 1
        if arr is None:
            m = dict(meta); m["chip_id"] = r["chip_id"]; m["shard"] = -1; mw.writerow(m); n_fail += 1
        else:
            buf.append((r, arr, meta)); n_ok += 1
        if len(buf) >= args.batch:
            flush()
        if k % 250 == 0:
            el = time.time() - t_start
            print(f"{k}/{len(reqs)} ok={n_ok} fail={n_fail} {(k/el):.2f}/s", flush=True)
    try:
        while True:
            while len(pending) < 4 * args.workers:
                r = next(it, None)
                if r is None:
                    break
                pending[ex.submit(task, r)] = (r, time.time())
            if not pending:
                break
            done_set, _ = fwait(list(pending), timeout=10, return_when=FIRST_COMPLETED)
            for fu in done_set:
                r, _t = pending.pop(fu); handle(*fu.result())
            now = time.time()
            for fu, (r, _sub) in list(pending.items()):
                t0 = STARTED.get(r["chip_id"])  # time from execution start, not submission (limiter waits excluded)
                if t0 is not None and now - t0 > TASK_TIMEOUT and not fu.done():
                    pending.pop(fu); handle(r, None, {"status": "error:task_timeout", "fetch_s": round(now - t0, 1)})
    finally:
        flush(); mf.close(); ex.shutdown(wait=False, cancel_futures=True)
    el = time.time() - t_start
    print(json.dumps({"n_req": len(reqs), "ok": n_ok, "fail": n_fail, "wall_s": round(el, 1),
                      "rate_per_s": round(len(reqs) / max(el, 1e-6), 3),
                      "gpu_max_mem_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2) if not args.no_models else 0}))


if __name__ == "__main__":
    main()
