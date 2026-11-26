"""E3 data build: co-located S2 / NAIP / S1 / Landsat chips + WorldCover labels over CONUS.
Implements PROTOCOL.md (frozen 2026-11-05). Usage: python build.py N_POINTS OUTDIR [N_WORKERS]
"""
import json, math, os, sys, time, threading, traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta

os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
os.environ.setdefault("GDAL_HTTP_MAX_RETRY", "5")
os.environ.setdefault("GDAL_HTTP_RETRY_DELAY", "2")
os.environ.setdefault("VSI_CACHE", "TRUE")

import numpy as np
import planetary_computer as pc
import pystac_client
import odc.stac
import rasterio
from odc.geo.geobox import GeoBox
from pyproj import Transformer
from rasterio.enums import Resampling
from rasterio.windows import from_bounds

PC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
CELL, PX = 320, 32
CLASSES = [10, 20, 30, 40, 50, 60, 80, 90]
PRIORITY = [60, 90, 80, 50, 20, 30, 40, 10]  # bare, wetland, water, built, shrub, grass, crop, tree
S2A = ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"]
LSA = ["red", "green", "blue", "nir08", "swir16", "swir22"]
PER_TILE, CAND = 8, 14
MAX_LOC = 10000
_tl = threading.local()


def cat():
    if not hasattr(_tl, "c"):
        _tl.c = pystac_client.Client.open(PC_URL, modifier=pc.sign_inplace)
    return _tl.c


def search(**kw):
    for a in range(4):
        try:
            return list(cat().search(**kw).items())
        except Exception:
            if a == 3:
                raise
            time.sleep(3 * (a + 1))


def dt_of(it):
    return datetime.fromisoformat(it.properties["datetime"].replace("Z", "+00:00")).replace(tzinfo=None)


def sample_points(n, seed=3):
    rng = np.random.default_rng(seed)
    s0, s1 = math.sin(math.radians(25.0)), math.sin(math.radians(49.4))
    lat = np.degrees(np.arcsin(rng.uniform(s0, s1, n)))
    lon = rng.uniform(-124.7, -67.0, n)
    return lon, lat


def days(items, t0, cloud_key=None, wcloud=2.0):
    """group items by solar day (UTC date), score = |dt| days + wcloud*min cloud."""
    g = {}
    for it in items:
        g.setdefault(dt_of(it).date(), []).append(it)
    out = []
    for d, its in g.items():
        dd = abs((datetime(d.year, d.month, d.day) - t0).days)
        cc = min(float(i.properties.get(cloud_key, 0)) for i in its) if cloud_key else 0.0
        out.append((dd + wcloud * cc, d, its))
    return sorted(out, key=lambda x: x[0])


def load(items, bands, gb, dtype, nodata):
    ds = odc.stac.load(items, bands=bands, geobox=gb, groupby="solar_day", dtype=dtype, nodata=nodata,
                       resampling="nearest", fail_on_error=True)
    return np.stack([ds[b].isel(time=0).values for b in bands])


def process_tile(tid, lon, lat, outdir):
    t_start = time.time()
    log = {"tid": tid, "lon": lon, "lat": lat}
    naip = search(collections=["naip"], intersects={"type": "Point", "coordinates": [lon, lat]},
                  datetime="2020-01-01/2022-12-31")
    if not naip:
        log["status"] = "no_naip"; return log, []
    naip.sort(key=lambda i: (abs(dt_of(i).year - 2021), -dt_of(i).year, i.id))
    ni = naip[0]; t0 = dt_of(ni)
    href = ni.assets["image"].href
    with rasterio.open(href) as src:
        crs = src.crs; b = src.bounds; tr = src.transform
        epsg = crs.to_epsg()
        x0 = math.ceil((b.left + 100) / CELL) * CELL; x1 = math.floor((b.right - 100) / CELL) * CELL
        y0 = math.ceil((b.bottom + 100) / CELL) * CELL; y1 = math.floor((b.top - 100) / CELL) * CELL
        if x1 - x0 > 6400:
            x0 += math.floor((x1 - x0 - 6400) / 2 / CELL) * CELL; x1 = x0 + 6400
        if y1 - y0 > 6400:
            y0 += math.floor((y1 - y0 - 6400) / 2 / CELL) * CELL; y1 = y0 + 6400
        nr, nc = (y1 - y0) // CELL, (x1 - x0) // CELL
        if nr < 2 or nc < 2:
            log["status"] = "tiny"; return log, []
        gb = GeoBox.from_bbox((x0, y0, x1, y1), crs=f"EPSG:{epsg}", resolution=10)
        assert gb.shape == (nr * PX, nc * PX), (gb.shape, nr, nc)
        gbL = GeoBox.from_bbox((x0 - 80, y0 - 80, x1 + 80, y1 + 80), crs=f"EPSG:{epsg}", resolution=10)
        to_ll = Transformer.from_crs(epsg, 4326, always_xy=True)
        llb = to_ll.transform_bounds(x0, y0, x1, y1)
        # WorldCover 2021
        wc = search(collections=["esa-worldcover"], bbox=list(llb), datetime="2021-01-01/2021-12-31")
        if not wc:
            log["status"] = "no_wc"; return log, []
        W = load(wc, ["map"], gb, "uint8", 0)[0]
        cellW = W.reshape(nr, PX, nc, PX).transpose(0, 2, 1, 3).reshape(nr, nc, PX * PX)
        lab = np.zeros((nr, nc), int); pur = np.zeros((nr, nc))
        for r in range(nr):
            for c in range(nc):
                v, n = np.unique(cellW[r, c], return_counts=True)
                k = n.argmax(); lab[r, c] = v[k]; pur[r, c] = n[k] / (PX * PX)
        okL = (pur >= 0.6) & np.isin(lab, CLASSES)
        if not okL.any():
            log["status"] = "no_label_cells"; return log, []
        win = f"{(t0 - timedelta(days=60)).date()}/{(t0 + timedelta(days=60)).date()}"
        # Sentinel-2
        s2items = search(collections=["sentinel-2-l2a"], bbox=list(llb), datetime=win,
                         query={"eo:cloud_cover": {"lt": 30}})
        S2 = np.zeros((nr, nc, 10, PX, PX), np.uint16); s2meta = {}; s2ok = np.zeros((nr, nc), bool)
        for _, d, its in days(s2items, t0, "eo:cloud_cover")[:3]:
            todo = okL & ~s2ok
            if not todo.any():
                break
            A = load(its, S2A + ["SCL"], gb, "uint16", 0)
            Ac = A.reshape(11, nr, PX, nc, PX).transpose(1, 3, 0, 2, 4)
            for r, c in zip(*np.where(todo)):
                px = Ac[r, c]
                if (px[:10] > 0).all() and np.isin(px[10], [4, 5, 6, 7]).mean() >= 0.95:
                    S2[r, c] = px[:10]; s2ok[r, c] = True
                    s2meta[(r, c)] = {"s2_ids": [i.id for i in its], "s2_date": str(dt_of(its[0])),
                                      "s2_baseline": its[0].properties.get("s2:processing_baseline"),
                                      "s2_cloud": float(its[0].properties.get("eo:cloud_cover", -1))}
        cand = okL & s2ok
        if not cand.any():
            log["status"] = "no_s2"; return log, []
        # selection (labels only)
        rng = np.random.default_rng(tid)
        pools = {k: [tuple(x) for x in rng.permutation(np.argwhere(cand & (lab == k)))] for k in PRIORITY}
        order = []
        while len(order) < CAND and any(pools.values()):
            for k in PRIORITY:
                if pools[k] and len(order) < CAND:
                    order.append(pools[k].pop(0))
        # NAIP reads
        sel, NA = [], []
        for (r, c) in order:
            cx0, cy1 = x0 + c * CELL, y1 - r * CELL
            w = from_bounds(cx0, cy1 - CELL, cx0 + CELL, cy1, tr)
            a = src.read([1, 2, 3, 4], window=w, out_shape=(4, 256, 256), resampling=Resampling.average)
            if ((a == 0).all(0)).mean() < 0.01:
                sel.append((r, c)); NA.append(a)
            if len(sel) >= PER_TILE:
                break
    if not sel:
        log["status"] = "no_naip_valid"; return log, []
    n = len(sel)
    # Sentinel-1 RTC
    S1 = np.full((n, 2, PX, PX), np.nan, np.float32); s1m = [None] * n
    try:
        s1items = search(collections=["sentinel-1-rtc"], bbox=list(llb), datetime=win)
        for _, d, its in days(s1items, t0)[:3]:
            if all(m is not None for m in s1m):
                break
            A = load(its, ["vv", "vh"], gb, "float32", float("nan"))
            for j, (r, c) in enumerate(sel):
                if s1m[j] is not None:
                    continue
                px = A[:, r * PX:(r + 1) * PX, c * PX:(c + 1) * PX]
                if np.isfinite(px).all() and (px > 0).all():
                    S1[j] = 10 * np.log10(np.where(px <= 0, 1e-10, px))
                    s1m[j] = {"s1_ids": [i.id for i in its], "s1_date": str(dt_of(its[0]))}
    except Exception as e:
        log["s1_err"] = repr(e)[:300]
    # Landsat C2 L2
    LS = np.full((n, 6, 16, 16), np.nan, np.float32); lsm = [None] * n
    try:
        lsitems = search(collections=["landsat-c2-l2"], bbox=list(llb), datetime=win,
                         query={"eo:cloud_cover": {"lt": 30}, "platform": {"in": ["landsat-8", "landsat-9"]}})
        for _, d, its in days(lsitems, t0, "eo:cloud_cover")[:3]:
            if all(m is not None for m in lsm):
                break
            A = load(its, LSA + ["qa_pixel"], gbL, "uint16", 0)
            for j, (r, c) in enumerate(sel):
                if lsm[j] is not None:
                    continue
                px = A[:, r * PX:r * PX + 48, c * PX:c * PX + 48]
                clear = ((px[6].astype(np.int64) >> 1) & 0b1111) == 0
                if (px[:6] > 0).all() and clear.mean() >= 0.95:
                    LS[j] = px[:6].astype(np.float32).reshape(6, 16, 3, 16, 3).mean((2, 4))
                    lsm[j] = {"ls_ids": [i.id for i in its], "ls_date": str(dt_of(its[0]))}
    except Exception as e:
        log["ls_err"] = repr(e)[:300]
    rows = []
    for j, (r, c) in enumerate(sel):
        cx0, cy1 = x0 + c * CELL, y1 - r * CELL
        clon, clat = to_ll.transform(cx0 + CELL / 2, cy1 - CELL / 2)
        row = {"loc_id": f"t{tid:04d}_r{r:02d}c{c:02d}", "tid": tid, "lon": clon, "lat": clat, "epsg": int(epsg),
               "cell_bounds": [cx0, cy1 - CELL, cx0 + CELL, cy1], "label": int(lab[r, c]), "purity": float(pur[r, c]),
               "naip_id": ni.id, "naip_date": str(t0), "has_s1": s1m[j] is not None, "has_ls": lsm[j] is not None}
        row.update(s2meta[(r, c)]); row.update(s1m[j] or {}); row.update(lsm[j] or {})
        rows.append(row)
    np.savez(os.path.join(outdir, "tiles", f"{tid:04d}.npz"), s2=S2[tuple(np.array(sel).T)], naip=np.stack(NA),
             s1=S1, ls=LS, loc_id=np.array([r["loc_id"] for r in rows]))
    log.update(status="ok", n=n, n_s1=sum(m is not None for m in s1m), n_ls=sum(m is not None for m in lsm),
               secs=round(time.time() - t_start, 1))
    return log, rows


def main():
    npts, outdir = int(sys.argv[1]), sys.argv[2]
    nw = int(sys.argv[3]) if len(sys.argv) > 3 else 12
    os.makedirs(os.path.join(outdir, "tiles"), exist_ok=True)
    lon, lat = sample_points(1500)
    lon, lat = lon[:npts], lat[:npts]
    allrows, logs, total = [], [], 0
    t0 = time.time()
    with ThreadPoolExecutor(nw) as ex:
        futs = {ex.submit(process_tile, i, float(lon[i]), float(lat[i]), outdir): i for i in range(len(lon))}
        for k, f in enumerate(as_completed(futs)):
            i = futs[f]
            try:
                lg, rows = f.result()
            except Exception as e:
                lg, rows = {"tid": i, "status": "error", "err": repr(e)[:500], "tb": traceback.format_exc()[-800:]}, []
            logs.append(lg); allrows.extend(rows)
            if k % 25 == 0:
                st = {}
                for l in logs:
                    st[l["status"]] = st.get(l["status"], 0) + 1
                print(f"[{time.time()-t0:7.0f}s] done {k+1}/{len(lon)} rows={len(allrows)} {st}", flush=True)
    allrows.sort(key=lambda r: r["loc_id"])
    # global cap (deterministic: lowest tile ids first)
    if len(allrows) > MAX_LOC:
        allrows = allrows[:MAX_LOC]
    json.dump(allrows, open(os.path.join(outdir, "rows.json"), "w"), default=lambda o: o.item() if hasattr(o, "item") else str(o))
    json.dump(sorted(logs, key=lambda l: l["tid"]), open(os.path.join(outdir, "tile_log.json"), "w"), indent=0, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    st = {}
    for l in logs:
        st[l["status"]] = st.get(l["status"], 0) + 1
    print("FINAL", len(allrows), st, f"{time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
