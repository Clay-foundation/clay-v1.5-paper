"""E8 item 1 build: 2.56 km context-matched S2 (256 px @10 m) and NAIP (256 px @10 m, GDAL average)
centred on every E3 location. Implements E8 PROTOCOL.md (frozen 2026-11-05 22:25 UTC).
Usage: python e8_build.py E3_MANIFEST.csv E3_TILES_DIR OUTDIR [N_WORKERS] [MAX_GROUPS]
"""
import csv, hashlib, json, os, sys, time, threading, traceback
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
os.environ.setdefault("GDAL_HTTP_MAX_RETRY", "5")
os.environ.setdefault("GDAL_HTTP_RETRY_DELAY", "2")
os.environ.setdefault("VSI_CACHE", "TRUE")
import numpy as np
import planetary_computer as pc
import pystac_client
import odc.stac
import rasterio
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
from rasterio.windows import from_bounds
from odc.geo.geobox import GeoBox
from pyproj import Transformer

PC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
S2A = ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"]
HALF, PX = 1280, 256
_tl = threading.local()


def cat():
    if not hasattr(_tl, "c"):
        _tl.c = pystac_client.Client.open(PC_URL, modifier=pc.sign_inplace)
    return _tl.c


def search(**kw):
    for a in range(5):
        try:
            return list(cat().search(**kw).items())
        except Exception:
            if a == 4:
                raise
            time.sleep(3 * (a + 1))


def dt_of(it):
    return datetime.fromisoformat(it.properties["datetime"].replace("Z", "+00:00")).replace(tzinfo=None)


def chip_bounds(r):
    x0, y0, x1, y1 = [float(v) for v in r["cell_bounds"].split("|")]
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    return (cx - HALF, cy - HALF, cx + HALF, cy + HALF)


def read_naip(href, epsg, b, px):
    """Read R,G,B,NIR over bounds b (in EPSG epsg) to px x px with GDAL average (overviews allowed)."""
    with rasterio.open(href) as src:
        if src.crs.to_epsg() == epsg:
            # clip to the dataset extent and read decimated (uses overviews); paste into a zero canvas
            sb = src.bounds; res = (b[2] - b[0]) / px
            ib = (max(b[0], sb.left), max(b[1], sb.bottom), min(b[2], sb.right), min(b[3], sb.top))
            out = np.zeros((4, px, px), np.uint8)
            if ib[0] >= ib[2] or ib[1] >= ib[3]:
                return out
            c0 = int(round((ib[0] - b[0]) / res)); c1 = int(round((ib[2] - b[0]) / res))
            r0 = int(round((b[3] - ib[3]) / res)); r1 = int(round((b[3] - ib[1]) / res))
            if c1 <= c0 or r1 <= r0:
                return out
            ib = (b[0] + c0 * res, b[3] - r1 * res, b[0] + c1 * res, b[3] - r0 * res)
            w = from_bounds(*ib, src.transform)
            out[:, r0:r1, c0:c1] = src.read([1, 2, 3, 4], window=w, out_shape=(4, r1 - r0, c1 - c0),
                                            resampling=Resampling.average)
            return out
        res = (b[2] - b[0]) / px
        tr = rasterio.transform.from_origin(b[0], b[3], res, res)
        with WarpedVRT(src, crs=f"EPSG:{epsg}", transform=tr, width=px, height=px,
                       resampling=Resampling.average, nodata=0) as v:
            return v.read([1, 2, 3, 4])


def naip_mosaic(primary, epsg, b, px, year, lonlat_bbox, log):
    a = read_naip(primary.assets["image"].href, epsg, b, px)
    used = [primary.id]
    hole = (a == 0).all(0)
    if hole.mean() > 0:
        others = search(collections=["naip"], bbox=lonlat_bbox, datetime=f"{year}-01-01/{year}-12-31")
        t0 = dt_of(primary)
        others = sorted([o for o in others if o.id != primary.id], key=lambda o: (abs((dt_of(o) - t0).days), o.id))
        for o in others:
            if not hole.any():
                break
            try:
                f = read_naip(o.assets["image"].href, epsg, b, px)
            except Exception as e:
                log.setdefault("naip_fill_err", []).append(repr(e)[:200]); continue
            fill = hole & ~(f == 0).all(0)
            if fill.any():
                a[:, fill] = f[:, fill]; used.append(o.id); hole = (a == 0).all(0)
    return a, used


def process_group(key, rows, e3tiles, outdir):
    tid, s2key = key
    log = {"tid": tid, "s2_ids": s2key, "n": len(rows)}
    t_g = time.time()
    epsg = int(rows[0]["epsg"])
    to_ll = Transformer.from_crs(epsg, 4326, always_xy=True)
    bs = [chip_bounds(r) for r in rows]
    ub = (min(b[0] for b in bs), min(b[1] for b in bs), max(b[2] for b in bs), max(b[3] for b in bs))
    s2items = search(collections=["sentinel-2-l2a"], ids=s2key.split("|"))
    if len(s2items) != len(s2key.split("|")):
        log["warn"] = f"found {len(s2items)} of {len(s2key.split('|'))} S2 items"
    gb = GeoBox.from_bbox(ub, crs=f"EPSG:{epsg}", resolution=10)
    ds = odc.stac.load(s2items, bands=S2A + ["SCL"], geobox=gb, groupby="solar_day", dtype="uint16", nodata=0,
                       resampling="nearest", fail_on_error=True)
    A = np.stack([ds[b].isel(time=0).values for b in S2A + ["SCL"]])
    log["t_s2"] = round(time.time() - t_g, 1)
    naip_id = rows[0]["naip_id"]
    ni = search(collections=["naip"], ids=[naip_id])[0]
    year = dt_of(ni).year
    z = np.load(os.path.join(e3tiles, f"{tid:04d}.npz"))
    e3idx = {l: i for i, l in enumerate(z["loc_id"])}
    S2 = np.zeros((len(rows), 10, PX, PX), np.uint16); NA = np.zeros((len(rows), 4, PX, PX), np.uint8)
    meta = []
    for j, (r, b) in enumerate(zip(rows, bs)):
        c0 = int(round((b[0] - ub[0]) / 10)); r0 = int(round((ub[3] - b[3]) / 10))
        px = A[:, r0:r0 + PX, c0:c0 + PX]
        assert px.shape == (11, PX, PX), px.shape
        S2[j] = px[:10]
        clear = float(np.isin(px[10], [4, 5, 6, 7]).mean()); allpos = bool((px[:10] > 0).all())
        e3 = z["s2"][e3idx[r["loc_id"]]]
        same = bool(np.array_equal(px[:10, 112:144, 112:144], e3))
        llb = to_ll.transform_bounds(*b)
        try:
            a, used = naip_mosaic(ni, epsg, b, PX, year, list(llb), log)
            NA[j] = a; zfrac = float((a == 0).all(0).mean())
        except Exception as e:
            used, zfrac = [], 1.0; log.setdefault("naip_err", []).append(repr(e)[:300])
        meta.append({"loc_id": r["loc_id"], "s2_clear": round(clear, 5), "s2_allpos": allpos, "s2_central_equals_e3": same,
                     "s2_valid": allpos and clear >= 0.90, "naip_zero_frac": round(zfrac, 5), "naip_valid": zfrac < 0.01,
                     "naip_items": "|".join(used), "chip_bounds": "|".join(f"{v:.0f}" for v in b)})
    log["t_total"] = round(time.time() - t_g, 1)
    h = hashlib.sha1(s2key.encode()).hexdigest()[:8]
    np.savez(os.path.join(outdir, "tiles", f"{tid:04d}_{h}.npz"), s2=S2, naip10=NA,
             loc_id=np.array([r["loc_id"] for r in rows]))
    log.update(status="ok", n_s2_valid=sum(m["s2_valid"] for m in meta), n_naip_valid=sum(m["naip_valid"] for m in meta))
    return log, meta


def main():
    man, e3tiles, outdir = sys.argv[1:4]
    nw = int(sys.argv[4]) if len(sys.argv) > 4 else 12
    maxg = int(sys.argv[5]) if len(sys.argv) > 5 else 10**9
    os.makedirs(os.path.join(outdir, "tiles"), exist_ok=True)
    rows = list(csv.DictReader(open(man)))
    groups = defaultdict(list)
    for r in rows:
        groups[(int(r["tid"]), r["s2_ids"])].append(r)
    keys = sorted(groups)[:maxg]
    print("locations", len(rows), "groups", len(groups), "running", len(keys), flush=True)
    metas, logs = [], []
    t0 = time.time()
    with ThreadPoolExecutor(nw) as ex:
        futs = {ex.submit(process_group, k, groups[k], e3tiles, outdir): k for k in keys}
        for i, f in enumerate(as_completed(futs)):
            k = futs[f]
            try:
                lg, m = f.result()
            except Exception as e:
                lg, m = {"tid": k[0], "s2_ids": k[1], "status": "error", "err": repr(e)[:400],
                         "tb": traceback.format_exc()[-800:]}, []
            logs.append(lg); metas.extend(m)
            if i % 25 == 0 or i == len(keys) - 1:
                print(f"[{time.time()-t0:6.0f}s] {i+1}/{len(keys)} metas={len(metas)} "
                      f"s2ok={sum(x['s2_valid'] for x in metas)} naipok={sum(x['naip_valid'] for x in metas)} "
                      f"err={sum(l['status']=='error' for l in logs)}", flush=True)
    metas.sort(key=lambda m: m["loc_id"])
    with open(os.path.join(outdir, "e8_chips.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(metas[0].keys())); w.writeheader(); w.writerows(metas)
    json.dump(sorted(logs, key=lambda l: l["tid"]), open(os.path.join(outdir, "build_log.json"), "w"), indent=0, default=str)
    print("FINAL", len(metas), "s2_valid", sum(m["s2_valid"] for m in metas), "naip_valid", sum(m["naip_valid"] for m in metas),
          "both", sum(m["s2_valid"] and m["naip_valid"] for m in metas), "central_eq", sum(m["s2_central_equals_e3"] for m in metas),
          f"{time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
