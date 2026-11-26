"""E6: fetch 3 additional seasonal Sentinel-2 L2A chips per E1 chip and embed them (Clay v1.5 CLS/mean, DOFA-L, stats).

Requests CSV: chip_id, lon, lat, y0 (valid-year start, ISO), e1_q (quarter of E1's chip, 0-3), tier, order.
Valid year = [y0, y0 + 12 months). Quarter q = [y0 + 3q months, y0 + 3(q+1) months). For each chip, the 3 quarters != e1_q
are fetched. One STAC search per chip over the whole valid year (fields extension: ids, dates, cloud, baseline, asset hrefs),
then per quarter: candidates by scene eo:cloud_cover, read SCL for up to MAX_CAND, first with <=5% bad SCL (0,1,3,8,9,10)
else lowest seen -- identical to E1's rule. 10 Clay bands raw uint16, 128 px at 10 m, 20 m bands nearest-upsampled,
chip snapped to the 20 m grid, nodata>0.5 dropped, smaller nodata filled by per-band median (E1 rules).
Pixels are not kept (disk). Output: OUT/shard_XXXX.npz (keys ids = 'chip_id|q') + OUT/meta.csv. Resumable. Hard stop.
"""
import argparse, csv, json, math, os, sys, time, threading
from concurrent.futures import ThreadPoolExecutor, wait as fwait, FIRST_COMPLETED
from datetime import datetime
from pathlib import Path
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "common"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "E1_label_efficiency", "code"))
import fetch_embed as fe  # E1's module: Limiter, read_window, retry, band constants (sets GDAL env)
from pyproj import Transformer
import pystac_client, planetary_computer

ASSETS = [fe.ASSET[b] for b in fe.BANDS] + ["SCL"]
FIELDS = {"include": ["id", "properties.datetime", "properties.eo:cloud_cover", "properties.s2:processing_baseline",
                      "properties.proj:epsg", "properties.proj:code"] + [f"assets.{a}.href" for a in ASSETS], "exclude": []}
_tl = threading.local()


def catalog():
    if not hasattr(_tl, "cat"):
        _tl.cat = pystac_client.Client.open(fe.STAC, timeout=60)
    return _tl.cat


def add_months(d, m):
    import calendar
    y, mo = d.year + (d.month - 1 + m) // 12, (d.month - 1 + m) % 12 + 1
    return d.replace(year=y, month=mo, day=min(d.day, calendar.monthrange(y, mo)[1]))  # clamp month-end starts


def search_year(lon, lat, y0):
    t0 = y0; t1 = add_months(y0, 12)
    for k in range(12):
        fe.LIM.wait()
        try:
            s = catalog().search(collections=["sentinel-2-l2a"], intersects={"type": "Point", "coordinates": [lon, lat]},
                                 datetime=f"{t0:%Y-%m-%d}T00:00:00Z/{t1:%Y-%m-%d}T00:00:00Z", limit=1000, max_items=1000,
                                 fields=FIELDS, sortby=[{"field": "eo:cloud_cover", "direction": "asc"}])
            return list(s.items_as_dicts())
        except Exception as e:
            msg = str(e).lower()
            if "rate limit" in msg or "429" in msg or "slow down" in msg:
                fe.LIM.throttled(); continue
            if k >= 3:
                raise
            time.sleep(2 * (2 ** k))
    raise RuntimeError("rate-limited: retries exhausted")


def fetch_chip(req, chip_px=128):
    """Returns list of (q, arr|None, meta) for the 3 non-E1 quarters."""
    lon, lat = float(req["lon"]), float(req["lat"]); size_m = chip_px * 10
    y0 = datetime.fromisoformat(req["y0"]); e1q = int(req["e1_q"])
    items = search_year(lon, lat, y0)
    bounds = [add_months(y0, 3 * q) for q in range(5)]
    out = []
    for q in range(4):
        if q == e1q:
            continue
        lo, hi = bounds[q], bounds[q + 1]
        cand = []
        for it in items:
            dt = datetime.fromisoformat(it["properties"]["datetime"].replace("Z", "+00:00")).replace(tzinfo=None)
            if lo <= dt < hi:
                cand.append(it)
        cand.sort(key=lambda it: (it["properties"].get("eo:cloud_cover", 100.0), it["id"]))
        meta = {"q": q, "n_items": len(cand)}
        if not cand:
            meta["status"] = "no_items"; out.append((q, None, meta)); continue
        best = None
        for it in cand[:fe.MAX_CAND]:
            p = it["properties"]
            epsg = p.get("proj:epsg") or int(str(p.get("proj:code", "EPSG:0")).split(":")[-1])
            cx, cy = Transformer.from_crs(4326, epsg, always_xy=True).transform(lon, lat)
            x0 = math.floor((cx - size_m / 2) / 20.0) * 20.0; y1 = math.floor((cy + size_m / 2) / 20.0) * 20.0
            try:
                href = planetary_computer.sign(it["assets"]["SCL"]["href"])
                scl = fe.retry(lambda: fe.read_window(href, x0, y1, size_m, chip_px // 2, 20), n=3)
            except Exception:
                continue
            bad = float(np.isin(scl, fe.BAD_SCL).mean())
            if best is None or bad < best[0]:
                best = (bad, it, epsg, x0, y1)
            if bad <= fe.CLEAR_OK:
                break
        if best is None:
            meta["status"] = "scl_unreadable"; out.append((q, None, meta)); continue
        bad, it, epsg, x0, y1 = best
        arr = np.zeros((len(fe.BANDS), chip_px, chip_px), np.uint16)
        for b, name in enumerate(fe.BANDS):
            a = fe.ASSET[name]; href = planetary_computer.sign(it["assets"][a]["href"])
            arr[b] = fe.retry(lambda: fe.read_window(href, x0, y1, size_m, chip_px, fe.RES[a]), n=3)
        zero = (arr == 0).any(0); zf = float(zero.mean()); p = it["properties"]
        meta.update({"status": "ok", "item": it["id"], "datetime": p["datetime"], "scene_cloud": p.get("eo:cloud_cover"),
                     "chip_bad_scl": round(bad, 4), "nodata_frac": round(zf, 4), "baseline": p.get("s2:processing_baseline"),
                     "epsg": epsg, "x0": x0, "y1": y1})
        if zf > 0.5:
            meta["status"] = "mostly_nodata"; out.append((q, None, meta)); continue
        if zf > 0:
            for b in range(len(fe.BANDS)):
                v = arr[b][~zero]; arr[b][zero] = np.median(v) if v.size else 1
        out.append((q, arr, meta))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--requests", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=32); ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--limit", type=int, default=0); ap.add_argument("--stac-rps", type=float, default=3.0)
    ap.add_argument("--hard-stop-h", type=float, default=6.0)
    args = ap.parse_args(); fe.LIM.rps = args.stac_rps
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    reqs = list(csv.DictReader(open(args.requests)))
    reqs.sort(key=lambda r: int(r["order"]))
    metap = out / "meta.csv"; done = set()
    if metap.exists():
        done = {r["chip_id"] for r in csv.DictReader(open(metap)) if not r["status"].startswith("error")}
    reqs = [r for r in reqs if r["chip_id"] not in done]
    if args.limit:
        reqs = reqs[:args.limit]
    print(f"{len(done)} chips done, {len(reqs)} to fetch", flush=True)
    import torch
    from clayrt import ClayRuntime, platform_spec
    spec = platform_spec("sentinel-2-l2a")
    clay = ClayRuntime(device="cuda")
    from torchgeo.models import dofa_large_patch16_224, DOFALarge16_Weights
    dofa = dofa_large_patch16_224(weights=DOFALarge16_Weights.DOFA_MAE).eval().cuda()
    json.dump({"clay": clay.evidence, "dofa": str(DOFALarge16_Weights.DOFA_MAE.url), "torch": torch.__version__},
              open(out / "runtime.json", "w"), indent=1, default=str)
    mean = torch.tensor(spec["means"]).view(1, -1, 1, 1); std = torch.tensor(spec["stds"]).view(1, -1, 1, 1)
    fields = ["chip_id", "q", "status", "item", "datetime", "scene_cloud", "chip_bad_scl", "nodata_frac", "baseline",
              "epsg", "x0", "y1", "n_items", "shard", "fetch_s", "tier"]
    newfile = not metap.exists()
    mf = open(metap, "a", newline=""); mw = csv.DictWriter(mf, fieldnames=fields, extrasaction="ignore")
    if newfile:
        mw.writeheader()
    shard_i = len(list(out.glob("shard_*.npz"))); buf = []; t_start = time.time(); n_ok = n_fail = 0; k = 0

    def flush():
        nonlocal shard_i, buf
        if not buf:
            return
        X = np.stack([b[1] for b in buf]).astype(np.float32)
        ids = np.array([f"{b[0]['chip_id']}|{b[2]['q']}" for b in buf])
        Xf = X.reshape(len(X), X.shape[1], -1)
        feats = {"ids": ids, "stats": np.concatenate([Xf.mean(2), Xf.std(2)], 1).astype(np.float32)}
        ll = [(float(b[0]["lat"]), float(b[0]["lon"])) for b in buf]
        dd = [datetime.fromisoformat(b[2]["datetime"].replace("Z", "+00:00")) for b in buf]
        o = clay.embed(X, platform="sentinel-2-l2a", latlon=ll, date=dd, batch_size=64, latlon_order="training")
        feats["clay_cls"] = o["cls"].astype(np.float16); feats["clay_mean"] = o["mean"].astype(np.float16); del o
        with torch.inference_mode():
            fs = []
            for i in range(0, len(X), 64):
                x = (torch.from_numpy(X[i:i + 64]) - mean) / std
                x = torch.nn.functional.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
                fs.append(dofa.forward_features(x.cuda(), spec["waves"]).float().cpu())
            feats["dofa"] = torch.cat(fs).numpy().astype(np.float16)
        for kk, v in feats.items():
            if kk != "ids" and not np.isfinite(v.astype(np.float32)).all():
                raise ValueError(f"non-finite {kk}")
        np.savez(out / f"shard_{shard_i:04d}.npz", **feats)
        for b in buf:
            m = dict(b[2]); m["chip_id"] = b[0]["chip_id"]; m["shard"] = shard_i; m["tier"] = b[0]["tier"]; mw.writerow(m)
        mf.flush(); shard_i += 1; buf = []

    STARTED = {}

    def task(r):
        t0 = time.time(); STARTED[r["chip_id"]] = t0
        try:
            res = fetch_chip(r)
        except Exception as e:
            res = [(-1, None, {"q": -1, "status": "error:" + type(e).__name__ + ":" + str(e)[:120].replace("\n", " ")})]
        for _, _, m in res:
            m["fetch_s"] = round(time.time() - t0, 2)
        return r, res

    def handle(r, res):
        nonlocal n_ok, n_fail, k
        k += 1
        for q, arr, meta in res:
            if arr is None:
                m = dict(meta); m["chip_id"] = r["chip_id"]; m["shard"] = -1; m["tier"] = r["tier"]; mw.writerow(m); n_fail += 1
            else:
                buf.append((r, arr, meta)); n_ok += 1
        if len(buf) >= args.batch:
            flush()
        if k % 250 == 0:
            el = time.time() - t_start
            print(f"{k}/{len(reqs)} chips; seasons ok={n_ok} fail={n_fail}; {k/el:.2f} chips/s; stac rps {fe.LIM.rps:.2f}", flush=True)

    ex = ThreadPoolExecutor(args.workers); pending = {}; it = iter(reqs); TASK_TIMEOUT = 900; stop = False
    try:
        while True:
            if not stop and time.time() - t_start > args.hard_stop_h * 3600:
                stop = True; print("HARD STOP reached; draining", flush=True)
            while not stop and len(pending) < 3 * args.workers:
                r = next(it, None)
                if r is None:
                    break
                pending[ex.submit(task, r)] = r
            if not pending:
                break
            done_set, _ = fwait(list(pending), timeout=10, return_when=FIRST_COMPLETED)
            for fu in done_set:
                pending.pop(fu); handle(*fu.result())
            now = time.time()
            for fu, r in list(pending.items()):
                t0 = STARTED.get(r["chip_id"])
                if t0 is not None and now - t0 > TASK_TIMEOUT and not fu.done():
                    pending.pop(fu); handle(r, [(-1, None, {"q": -1, "status": "error:task_timeout"})])
    finally:
        flush(); mf.close(); ex.shutdown(wait=False, cancel_futures=True)
    el = time.time() - t_start
    print(json.dumps({"n_req": len(reqs), "chips_handled": k, "seasons_ok": n_ok, "seasons_fail": n_fail, "wall_s": round(el, 1),
                      "gpu_max_mem_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2)}), flush=True)
    os._exit(0)


if __name__ == "__main__":
    main()
