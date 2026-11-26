"""E2_tokens fetch: sample points, stream 256x256 chips (S2, S1, Landsat, NAIP) + WorldCover token labels from PC.
Usage: python fetch.py OUTDIR [--n-global 1500 --n-conus 2000 --workers 24]
Writes OUTDIR/chips/<sensor>/<pid>.npz and OUTDIR/points.jsonl, OUTDIR/log_<sensor>.jsonl."""
import argparse, hashlib, json, math, os, sys, threading, time, traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np
os.environ.setdefault("GDAL_HTTP_MAX_RETRY", "4"); os.environ.setdefault("GDAL_HTTP_RETRY_DELAY", "2")
os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR"); os.environ.setdefault("GDAL_HTTP_TIMEOUT", "60")
import pystac_client, planetary_computer, odc.stac
from odc.geo.geobox import GeoBox
from affine import Affine
from pyproj import Transformer, CRS

ap = argparse.ArgumentParser(); ap.add_argument("out"); ap.add_argument("--n-global", type=int, default=1500)
ap.add_argument("--n-conus", type=int, default=2000); ap.add_argument("--workers", type=int, default=24)
A = ap.parse_args(); OUT = Path(A.out); (OUT / "chips").mkdir(parents=True, exist_ok=True)
for s in ["s2", "s1", "ls", "naip"]: (OUT / "chips" / s).mkdir(exist_ok=True)
PC = "https://planetarycomputer.microsoft.com/api/stac/v1"
_tl = threading.local()
def cat():
    if not hasattr(_tl, "c"):
        _tl.c = pystac_client.Client.open(PC, modifier=planetary_computer.sign_inplace)
    return _tl.c
LOCK = threading.Lock()
def log(name, rec):
    with LOCK, open(OUT / f"log_{name}.jsonl", "a") as f: f.write(json.dumps(rec) + "\n")

S2B = ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"]
LSB = ["red", "green", "blue", "nir08", "swir16", "swir22"]

def item_epsg(it):
    pr = it.properties
    if pr.get("proj:epsg"): return int(pr["proj:epsg"])
    if pr.get("proj:code"): return int(str(pr["proj:code"]).split(":")[-1])
    for a in it.assets.values():
        e = a.extra_fields
        if e.get("proj:epsg"): return int(e["proj:epsg"])
        if e.get("proj:code"): return int(str(e["proj:code"]).split(":")[-1])
    raise KeyError("no epsg")

def utm_epsg(lat, lon):
    return (32600 if lat >= 0 else 32700) + int((lon + 180) // 6) + 1

def geobox_for(lat, lon, epsg, res, n=256, snap=None):
    tr = Transformer.from_crs(4326, epsg, always_xy=True); x, y = tr.transform(lon, lat)
    snap = snap or res
    x0 = math.floor((x - n * res / 2) / snap) * snap; y1 = math.floor((y + n * res / 2) / snap) * snap
    return GeoBox((n, n), Affine(res, 0, x0, 0, -res, y1), CRS.from_epsg(epsg))

def load(items, bands, gb, dtype, nodata, resampling="nearest"):
    return retry(lambda: _load(items, bands, gb, dtype, nodata, resampling))
def _load(items, bands, gb, dtype, nodata, resampling):
    items = [planetary_computer.sign(i) for i in items]
    ds = odc.stac.load(items, bands=bands, geobox=gb, dtype=dtype, nodata=nodata, resampling=resampling,
                       groupby="id", chunks=None, fail_on_error=True)
    return np.stack([ds[b].isel(time=0).values for b in bands])

RATE = {"next": 0.0, "dt": 0.35}  # >= 0.35 s between STAC requests across all threads
class Transient(Exception): pass
def throttle():
    with LOCK:
        now = time.time(); wait = max(0.0, RATE["next"] - now); RATE["next"] = max(now, RATE["next"]) + RATE["dt"]
    if wait: time.sleep(wait)
def is_transient(e):
    t = repr(e).lower()
    return any(k in t for k in ["rate limit", "429", "503", "502", "500", "timeout", "timed out", "connection", "temporarily", "reset by peer", "curl error", "http response code", "403", "chunk and warp failed"])
def retry(fn, tries=8):
    for k in range(tries):
        try: return fn()
        except Exception as e:
            if not is_transient(e) or k == tries - 1: raise
            time.sleep(min(120, 5 * 2 ** k) * (0.5 + np.random.rand()))

def search(coll, lat, lon, dt, query=None, n=50):
    d = 1e-4
    def f():
        throttle()
        return list(cat().search(collections=[coll], bbox=[lon - d, lat - d, lon + d, lat + d], datetime=dt,
                                 query=query, max_items=n).items())
    return retry(f)

def wc_labels(gb, res_wc):
    """WorldCover 2021 on chip extent at res_wc; majority per token cell -> (32,32) uint8, purity."""
    n_px = int(round(gb.shape[0] * gb.resolution.x / res_wc))
    g2 = GeoBox((n_px, n_px), Affine(res_wc, 0, gb.affine.c, 0, -res_wc, gb.affine.f), gb.crs)
    bb = gb.geographic_extent.boundingbox
    its = [t for t in WC_TILES if t.bbox[0] < bb.right and t.bbox[2] > bb.left and t.bbox[1] < bb.top and t.bbox[3] > bb.bottom]
    if not its: raise RuntimeError("no worldcover")
    ds = retry(lambda: odc.stac.load([planetary_computer.sign(t) for t in its], bands=["map"], geobox=g2, dtype="uint8", nodata=0, resampling="nearest",
                       groupby="solar_day", chunks=None))
    m = ds["map"].isel(time=0).values
    k = n_px // 32
    blk = m.reshape(32, k, 32, k).transpose(0, 2, 1, 3).reshape(32, 32, k * k)
    classes = np.array([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100])
    cnt = np.stack([(blk == v).sum(-1) for v in classes], -1)
    cnt_valid = cnt.copy(); cnt_valid[..., 0] = -1
    lab = classes[cnt_valid.argmax(-1)]; pur = cnt.max(-1) / (k * k)
    lab = np.where(cnt[..., 0] == k * k, 0, lab).astype(np.uint8)
    water_frac = float(((m == 80) | (m == 0)).mean())
    return lab, pur.astype(np.float16), water_frac

def fetch_s2(p):
    its = search("sentinel-2-l2a", p["lat"], p["lon"], "2021-01-01/2021-12-31", {"eo:cloud_cover": {"lt": 20}})
    its.sort(key=lambda i: i.properties["eo:cloud_cover"])
    tried = []
    for it in its[:4]:
        epsg = item_epsg(it); gb = geobox_for(p["lat"], p["lon"], epsg, 10, snap=60)
        arr = load([it], S2B + ["SCL"], gb, "uint16", 0)
        px, scl = arr[:10], arr[10]
        cloud = float(np.isin(scl, [3, 8, 9, 10]).mean())
        if (px == 0).any() or cloud > 0.05: tried.append((it.id, cloud, int((px == 0).sum()))); continue
        return dict(pixels=px, item=it.id, dt=it.properties["datetime"], epsg=epsg, gb=gb,
                    baseline=it.properties.get("s2:processing_baseline"), cloud=cloud), tried
    return None, tried

def fetch_s1(p, gb, s2dt):
    t = datetime.fromisoformat(s2dt.replace("Z", "+00:00"))
    its = search("sentinel-1-rtc", p["lat"], p["lon"], f"{(t-timedelta(days=60)).date()}/{(t+timedelta(days=60)).date()}")
    if not its: its = search("sentinel-1-rtc", p["lat"], p["lon"], "2021-01-01/2021-12-31")
    its = [i for i in its if {"vv", "vh"} <= set(i.assets)]
    its.sort(key=lambda i: abs((i.datetime - t).total_seconds()))
    tried = []
    for it in its[:3]:
        arr = load([it], ["vv", "vh"], gb, "float32", np.nan)
        ok = np.isfinite(arr) & (arr > 0)
        if not ok.all(): tried.append((it.id, int((~ok).sum()))); continue
        return dict(pixels=(10 * np.log10(arr)).astype(np.float32), item=it.id, dt=it.properties["datetime"]), tried
    return None, tried

def fetch_ls(p):
    its = search("landsat-c2-l2", p["lat"], p["lon"], "2021-01-01/2021-12-31",
                 {"eo:cloud_cover": {"lt": 20}, "platform": {"in": ["landsat-8", "landsat-9"]}})
    its.sort(key=lambda i: i.properties["eo:cloud_cover"])
    tried = []
    for it in its[:4]:
        epsg = item_epsg(it); gb = geobox_for(p["lat"], p["lon"], epsg, 30, snap=30)
        arr = load([it], LSB + ["qa_pixel"], gb, "uint16", 0)
        px, qa = arr[:6], arr[6].astype(np.int64)
        cloud = float((((qa >> 3) & 1) | ((qa >> 4) & 1)).mean())
        if (px == 0).any() or cloud > 0.05: tried.append((it.id, cloud, int((px == 0).sum()))); continue
        return dict(pixels=px, item=it.id, dt=it.properties["datetime"], epsg=epsg, gb=gb, cloud=cloud), tried
    return None, tried

def read_naip(it, gb):
    import rasterio
    from rasterio.vrt import WarpedVRT
    from rasterio.enums import Resampling
    href = planetary_computer.sign(it).assets["image"].href
    with rasterio.open(href) as src, WarpedVRT(src, crs=f"EPSG:{gb.crs.epsg}", transform=gb.affine, width=256, height=256,
                                               resampling=Resampling.bilinear, nodata=0) as v:
        return v.read()

def fetch_naip(p, its):
    its.sort(key=lambda i: i.datetime, reverse=True)
    tried = []
    for it in its[:2]:
        epsg = item_epsg(it); gb = geobox_for(p["lat"], p["lon"], epsg, 1.0, snap=1.0)
        im = retry(lambda: read_naip(it, gb))
        if im.shape[0] != 4 or (im == 0).all(0).any(): tried.append((it.id, list(im.shape))); continue
        return dict(pixels=im, item=it.id, dt=it.properties["datetime"], epsg=epsg, gb=gb), tried
    return None, tried

def save(sensor, pid, p, d, lab, pur):
    np.savez(OUT / "chips" / sensor / f"{pid}.npz", pixels=d["pixels"], labels=lab, purity=pur)
    log(sensor, {"pid": pid, "ok": True, "lat": p["lat"], "lon": p["lon"], "stratum": p["stratum"], "item": d["item"],
                 "datetime": d["dt"], "epsg": d.get("epsg"), "baseline": d.get("baseline"), "cloud": d.get("cloud"),
                 "bbox_native": list(d["gb"].extent.boundingbox) if "gb" in d else None})

def process(p):
    pid = p["pid"]
    # point acceptance: WorldCover at the S2 footprint
    gb0 = geobox_for(p["lat"], p["lon"], utm_epsg(p["lat"], p["lon"]), 10, snap=60)
    _, _, wf = wc_labels(gb0, 10)
    if wf > 0.5: return {"pid": pid, "accepted": False, "why": f"water {wf:.2f}"}
    naip_items = None
    if p["stratum"] == "conus":
        naip_items = search("naip", p["lat"], p["lon"], "2019-01-01/2022-12-31", n=20)
        if not naip_items: return {"pid": pid, "accepted": False, "why": "no naip"}
    res = {"pid": pid, "accepted": True}
    def guard(name, fn):
        try: fn()
        except Exception as e: log(name, {"pid": pid, "ok": False, "err": repr(e)[:300]})
    def do_s2():
        d, tried = fetch_s2(p)
        if d is None: log("s2", {"pid": pid, "ok": False, "tried": tried}); return
        lab, pur, _ = wc_labels(d["gb"], 10); save("s2", pid, p, d, lab, pur)
        def do_s1():
            d1, tried1 = fetch_s1(p, d["gb"], d["dt"])
            if d1 is None: log("s1", {"pid": pid, "ok": False, "tried": tried1}); return
            d1["gb"] = d["gb"]; d1["epsg"] = d["epsg"]; save("s1", pid, p, d1, lab, pur)
        guard("s1", do_s1)
    def do_ls():
        d, tried = fetch_ls(p)
        if d is None: log("ls", {"pid": pid, "ok": False, "tried": tried}); return
        lab, pur, _ = wc_labels(d["gb"], 10); save("ls", pid, p, d, lab, pur)
    def do_naip():
        d, tried = fetch_naip(p, naip_items)
        if d is None: log("naip", {"pid": pid, "ok": False, "tried": tried}); return
        lab, pur, _ = wc_labels(d["gb"], 1.0); save("naip", pid, p, d, lab, pur)
    guard("s2", do_s2); guard("ls", do_ls)
    if p["stratum"] == "conus": guard("naip", do_naip)
    return res

def candidates():
    rng = np.random.default_rng(20261105)
    tiles = list(WC_TILES)
    tiles.sort(key=lambda t: hashlib.sha256(t.id.encode()).hexdigest())
    print("worldcover tiles", len(tiles), flush=True)
    glob, con = [], []
    for k in range(6):  # up to 6 cycles over tiles
        for t in tiles:
            b = t.bbox; glob.append((rng.uniform(b[1], b[3]), rng.uniform(b[0], b[2])))
    for _ in range(8000):
        con.append((rng.uniform(24.5, 49.4), rng.uniform(-124.7, -66.9)))
    return glob, con

def run_stratum(name, pts, target):
    acc, i = 0, 0
    done_f = OUT / f"points_{name}.jsonl"
    prev = {}
    if done_f.exists():
        for l in open(done_f):
            r = json.loads(l)
            if not str(r.get("why", "")).startswith("error") or "no worldcover" in str(r.get("why")): prev[r["pid"]] = r
    acc = sum(r["accepted"] for r in prev.values())
    with ThreadPoolExecutor(A.workers) as ex:
        while acc < target and i < len(pts):
            batch = []
            while len(batch) < A.workers * 2 and i < len(pts):
                pid = f"{name[0]}{i:05d}"; lat, lon = pts[i]; i += 1
                if pid in prev: continue
                batch.append({"pid": pid, "lat": float(lat), "lon": float(lon), "stratum": name})
            futs = {ex.submit(process, p): p for p in batch}
            for f in as_completed(futs):
                p = futs[f]
                try: r = f.result()
                except Exception as e: r = {"pid": p["pid"], "accepted": False, "why": "error " + repr(e)[:200]}
                r.update(p)
                with LOCK, open(done_f, "a") as fh: fh.write(json.dumps(r) + "\n")
                acc += r["accepted"]
            print(name, "candidates", i, "accepted", acc, time.strftime("%H:%M:%S"), flush=True)
    # note: a final batch can overshoot the target; extra accepted points are kept and reported
    return acc

t0 = time.time()
# unsigned client: cached tiles must be re-signed at use (sign() skips hrefs that already carry a token, even expired)
def list_wc_tiles():
    """All WorldCover 2021 tiles, in 24 spatial queries of < 1000 items each (a single page): the Planetary Computer
    STAC API answers 'upstream request timeout' on page 2 and later of the unfiltered listing (seen 2026-11-18)."""
    seen = {}
    for lon0 in range(-180, 180, 90):
        for lat0 in range(-90, 90, 30):
            for t in pystac_client.Client.open(PC).search(collections=["esa-worldcover"], datetime="2021-01-01/2021-12-31",
                                                         bbox=[lon0, lat0, lon0 + 90, lat0 + 30], limit=1000).items():
                seen[t.id] = t
    return list(seen.values())
WC_TILES = retry(list_wc_tiles)
WC_TILES = [t for t in WC_TILES if "2021" in t.id or "v200" in t.id] or WC_TILES
glob, con = candidates()
run_stratum("conus", con, A.n_conus)
run_stratum("global", glob, A.n_global)
for s in ["s2", "s1", "ls", "naip"]:
    print(s, len(list((OUT / "chips" / s).glob("*.npz"))))
print("elapsed", round(time.time() - t0))
