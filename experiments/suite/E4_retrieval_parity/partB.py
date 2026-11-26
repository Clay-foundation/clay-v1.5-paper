"""E4 Part B: parity of released Clay v1.5 embeddings vs local re-encoding (released checkpoint, documented preprocessing)."""
import hashlib, io, json, math, os, sys, time, traceback, re
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
import pyproj
os.environ.setdefault("PROJ_DATA", pyproj.datadir.get_data_dir())
import numpy as np, requests, pyarrow.parquet as pq, rasterio
from rasterio.windows import Window
from rasterio.enums import Resampling
from shapely import wkb
from shapely.ops import transform as shp_transform
sys.path.insert(0, "common")
from clayrt import ClayRuntime, platform_spec
BUCKET = "https://s3.us-west-2.amazonaws.com/us-west-2.opendata.source.coop"
PROXY = "https://data.source.coop/"
S = requests.Session()
LOG = {"sets": {}, "errors": []}


def h(s): return hashlib.sha256(s.encode()).hexdigest()


def ls(prefix, delimiter="/"):
    pre, keys, tok = [], [], None
    while True:
        p = {"list-type": "2", "prefix": prefix, "delimiter": delimiter}
        if tok: p["continuation-token"] = tok
        r = S.get(BUCKET + "/", params=p, timeout=60); r.raise_for_status()
        root = ET.fromstring(r.content); ns = {"s": root.tag.split("}")[0].strip("{")}
        pre += [e.text for e in root.findall("s:CommonPrefixes/s:Prefix", ns)]
        keys += [(e.find("s:Key", ns).text, int(e.find("s:Size", ns).text)) for e in root.findall("s:Contents", ns)]
        t = root.find("s:NextContinuationToken", ns)
        if t is None: break
        tok = t.text
    return pre, keys


def read_parquet(key):
    r = S.get(PROXY + key, timeout=600); r.raise_for_status()
    t = pq.read_table(io.BytesIO(r.content))
    return t, {k.decode(): v.decode()[:500] for k, v in (t.schema.metadata or {}).items() if k != b"geo"}, len(r.content)


def cos(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    return (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))


def summarize(rel, rec):
    c = cos(rel, rec)
    return {"n": int(len(c)), "median_cos": float(np.median(c)), "p05_cos": float(np.percentile(c, 5)), "min_cos": float(c.min()),
            "frac_gt_0.99": float((c > 0.99).mean()), "norm_ratio_median": float(np.median(np.linalg.norm(rel, axis=1) / np.linalg.norm(rec, axis=1))),
            "released_norm_median": float(np.median(np.linalg.norm(rel, axis=1))), "recomputed_norm_median": float(np.median(np.linalg.norm(rec, axis=1)))}, c


def run_arms(rt, px, platform, latlon_xy, date, gsd, arms, waves=None, means=None, stds=None):
    """latlon_xy: list of (lon, lat) bbox-centroid coords as in pipeline (centroid.coords[0])."""
    out = {}
    for arm in arms:
        if arm == "as_shipped":   # normalize_latlon(*(lon,lat)) -> lon in lat slots
            ll = [(x, y) for x, y in latlon_xy]; dd = date; norm = True
        elif arm == "latlon_fixed":
            ll = [(y, x) for x, y in latlon_xy]; dd = date; norm = True
        elif arm == "no_meta":
            ll = None; dd = None; norm = True
        elif arm == "no_norm":
            ll = [(x, y) for x, y in latlon_xy]; dd = date; norm = False
        out[arm] = rt.embed(px, platform=platform, latlon=ll, date=dd, gsd=gsd, normalize=norm, batch_size=16)["cls"]
    return out


def naip_set(rt, n_files=8, per_file=125):
    import pystac_client, planetary_computer
    cat = pystac_client.Client.open("https://planetarycomputer.microsoft.com/api/stac/v1", modifier=planetary_computer.sign_inplace)
    root = "clay/clay-v1-5-naip-2/"
    states, _ = ls(root)
    states = sorted(states, key=lambda p: h(f"e4-naip-state|{p}"))
    rel_all, rec_all, rows, attempts = [], {a: [] for a in ("as_shipped", "latlon_fixed", "no_meta", "no_norm")}, [], []
    old_cmp = []
    for sp in states:
        if len(rows) and len({r["file"] for r in rows}) >= n_files: break
        try:
            years, _ = ls(sp); years = sorted(years, key=lambda p: h(f"e4-naip-year|{p}"))
            done = False
            for yp in years[:3]:
                res, _ = ls(yp); rp = sorted(res, key=lambda p: h(f"e4-naip-res|{p}"))[0]
                q, _ = ls(rp + "rgbir_cog/"); qp = sorted(q, key=lambda p: h(f"e4-naip-quad|{p}"))[0]
                _, files = ls(qp); fk = sorted([k for k, s in files if k.endswith(".parquet")], key=lambda k: h(f"e4-naip-file|{k}"))[0]
                stem = fk.rsplit("/", 1)[1][:-8]; state = fk.split("/")[2]
                t, meta, nbytes = read_parquet(fk)
                emb = np.stack(t.column("embeddings").to_numpy(zero_copy_only=False)).astype("float32")
                geoms = [wkb.loads(g) for g in t.column("geometry").to_pylist()]
                for attempt in range(6):
                    try:
                        cands = list(cat.search(collections=["naip"], bbox=list(geoms[0].bounds), limit=100).items()); break
                    except Exception as e:
                        if attempt == 5: raise
                        time.sleep(15 * 2 ** attempt)
                match = [it for it in cands if it.id == f"{state}_{stem}" or it.id.startswith(f"{state}_{stem}_")]
                item = match[0] if len(match) == 1 else None
                attempts.append({"key": fk, "pc_item_prefix": f"{state}_{stem}", "pc_matches": [m.id for m in match]})
                if item is None: continue
                href = item.assets["image"].href
                with rasterio.open(href) as src:
                    T, crs, Hh, Ww = src.transform, src.crs, src.height, src.width
                    fwd = pyproj.Transformer.from_crs(4326, crs, always_xy=True).transform
                    pick = sorted(range(len(geoms)), key=lambda i: h(f"e4-naip-row|{fk}|{i}"))[:per_file]
                    px, ll, resid, xy = [], [], [], []
                    for i in pick:
                        gb = shp_transform(fwd, geoms[i]).bounds
                        fx = (gb[0] - T.c) / (256 * T.a); fy = (T.f - gb[3]) / (256 * -T.e)
                        x, y = int(round(fx)), int(round(fy)); resid.append(max(abs(fx - x), abs(fy - y)) * 256)
                        px.append(src.read(window=Window(x * 256, y * 256, 256, 256)).astype("float32"))
                        ll.append(geoms[i].centroid.coords[0]); xy.append((x, y))
                px = np.stack(px)
                datestr = meta.get("date"); date = datetime(int(datestr[:4]), int(datestr[4:6]), int(datestr[6:8]), 12)
                gsd = float(fk.split("/")[4].replace("cm", "")) / 100
                recs = run_arms(rt, px, "naip", ll, date, gsd, list(rec_all))
                for a in rec_all: rec_all[a].append(recs[a])
                rel_all.append(emb[pick])
                n_chips_expected = (Hh // 256) * (Ww // 256)
                for j, i in enumerate(pick):
                    rows.append({"file": fk, "row": int(i), "chip_x": xy[j][0], "chip_y": xy[j][1], "resid_px": float(resid[j]),
                                 "lon": ll[j][0], "lat": ll[j][1], "cos_as_shipped": float(cos(emb[[i]], recs["as_shipped"][[j]])[0])})
                LOG["sets"].setdefault("naip_files", []).append({"key": fk, "meta": meta, "rows_in_file": len(geoms), "dim": int(emb.shape[1]),
                    "pc_item": item.id, "pc_href": href.split("?")[0], "raster_shape": [Hh, Ww], "expected_chips_full_grid": n_chips_expected,
                    "max_resid_px": float(max(resid)), "gsd": gsd, "date": str(date), "bytes": nbytes})
                # superseded v1 release: same footprints?
                try:
                    t_old, meta_old, _ = read_parquet(fk.replace("clay-v1-5-naip-2/", "clay-v1-5-naip/"))
                    g_old = {bytes(g): k for k, g in enumerate(t_old.column("geometry").to_pylist())}
                    e_old = np.stack(t_old.column("embeddings").to_numpy(zero_copy_only=False)).astype("float32")
                    for j, i in enumerate(pick):
                        k = g_old.get(bytes(t.column("geometry")[i].as_py()))
                        if k is not None:
                            old_cmp.append((e_old[k], emb[i], recs["no_norm"][j], recs["as_shipped"][j]))
                except Exception as e:
                    LOG["errors"].append(f"naip old {fk}: {e}")
                print("naip", fk, len(pick), "median cos as_shipped", float(np.median(cos(emb[pick], recs["as_shipped"]))), flush=True)
                done = True; break
        except Exception as e:
            LOG["errors"].append(f"naip {sp}: {e!r}"); traceback.print_exc()
    rel = np.concatenate(rel_all); out = {"attempts": attempts}
    for a, v in rec_all.items():
        out[a], c = summarize(rel, np.concatenate(v))
    if old_cmp:
        eo, en, rnn, ras = map(np.stack, zip(*old_cmp))
        out["superseded_v1_vs_no_norm"] = summarize(eo, rnn)[0]; out["superseded_v1_vs_as_shipped"] = summarize(eo, ras)[0]
        out["superseded_v1_vs_rev2_released"] = summarize(eo, en)[0]
    out["footprint_resid_px_max"] = float(max(r["resid_px"] for r in rows))
    out["dims"] = sorted({int(rel.shape[1])})
    return out, rows


def s2_item_bands():
    return ["blue", "green", "red", "rededge1", "rededge2", "rededge3", "nir", "nir08", "swir16", "swir22"]


V0KEY = {"blue": "B02", "green": "B03", "red": "B04", "rededge1": "B05", "rededge2": "B06", "rededge3": "B07", "nir": "B08",
         "nir08": "B8A", "swir16": "B11", "swir22": "B12", "scl": "SCL"}


def read_s2_chip(assets, shape, x, y, cs=256, row_off=None, col_off=None):
    """stacchip Chipper semantics: window in each band's native res, out_shape cs x cs, nearest."""
    arrs = []
    for b in s2_item_bands():
        with rasterio.open(assets[b]) as src:
            f = shape[0] / src.height
            c0 = (x * cs if col_off is None else col_off); r0 = (y * cs if row_off is None else row_off)
            w = Window(math.floor(c0 / f), math.floor(r0 / f), math.ceil(cs / f), math.ceil(cs / f))
            arrs.append(src.read(1, window=w, out_shape=(cs, cs), resampling=Resampling.nearest, boundless=True, fill_value=0))
    return np.stack(arrs).astype("float32")


def s2_old_set(rt, n_files=6, max_rows=300):
    root = "clay/clay-v1-5-sentinel2/sentinel-s2-l2a-cogs/"
    zones, _ = ls(root)
    us = [z for z in zones if z.rstrip("/").split("/")[-1] in [str(i) for i in range(10, 20)]]
    files = []
    for z in sorted(us, key=lambda p: h(f"e4-s2-zone|{p}")):
        if len(files) >= n_files: break
        try:
            bands_, _ = ls(z)
            for lb in sorted(bands_, key=lambda p: h(f"e4-s2-lb|{p}"))[:1]:
                sq, _ = ls(lb); sqp = sorted(sq, key=lambda p: h(f"e4-s2-sq|{p}"))[0]
                yrs, _ = ls(sqp); yp = sorted(yrs, key=lambda p: h(f"e4-s2-y|{p}"))[0]
                mos, _ = ls(yp); mp = sorted(mos, key=lambda p: h(f"e4-s2-m|{p}"))[0]
                its, _ = ls(mp); ip = sorted(its, key=lambda p: h(f"e4-s2-i|{p}"))[0]
                _, ks = ls(ip); files += [k for k, s in ks if k.endswith(".parquet")][:1]
        except Exception as e:
            LOG["errors"].append(f"s2old list {z}: {e!r}")
    rel_all = []; rec = {}; rows = []; finfo = []
    for fk in files:
        try:
            t, meta, _ = read_parquet(fk)
            emb = np.stack(t.column("embeddings").to_numpy(zero_copy_only=False)).astype("float32")
            geoms = [wkb.loads(g) for g in t.column("geometry").to_pylist()]
            key = meta["uri"].replace("s3://sentinel-cogs/", "")
            if not key.endswith(".json"):
                key = key.rstrip("/") + ".json" if key.endswith("_L2A") else key
            jurl = "https://sentinel-cogs.s3.us-west-2.amazonaws.com/" + key
            item = S.get(jurl, timeout=60).json()
            A = item["assets"]; get = lambda b: (A.get(b) or A.get(V0KEY[b]))
            assets = {b: get(b)["href"].replace("s3://sentinel-cogs/", "https://sentinel-cogs.s3.us-west-2.amazonaws.com/") for b in s2_item_bands()}
            epsg = item["properties"].get("proj:epsg")
            tr10 = get("blue")["proj:transform"]; shape = get("blue")["proj:shape"]
            ts = [a.get("proj:transform") for a in A.values() if a.get("proj:transform")]
            tr_st = max(ts, key=lambda t_: t_[0]) if "proj:transform" not in item["properties"] else item["properties"]["proj:transform"]
            fwd = pyproj.Transformer.from_crs(4326, epsg, always_xy=True).transform
            dt = datetime.fromisoformat(item["properties"]["datetime"].replace("Z", "+00:00"))
            pick = sorted(range(len(geoms)), key=lambda i: h(f"e4-s2-row|{fk}|{i}"))[:max(1, max_rows // max(1, len(files)))]
            widths = []
            for hyp, T in (("10m", tr10), ("stacchip_transform", tr_st)):
                px, ll, res = [], [], []
                for i in pick:
                    gb = shp_transform(fwd, geoms[i]).bounds; widths.append(gb[2] - gb[0])
                    fx = (gb[0] - T[2]) / (256 * T[0]); fy = (T[5] - gb[3]) / (256 * -T[4])
                    x, y = int(round(fx)), int(round(fy)); res.append(max(abs(fx - x), abs(fy - y)))
                    px.append(read_s2_chip(assets, shape, x, y)); ll.append(geoms[i].centroid.coords[0])
                px = np.stack(px)
                ok = (px > 0).all(axis=(1, 2, 3))
                r = run_arms(rt, px, "sentinel-2-l2a", ll, dt, 10.0, ["as_shipped", "latlon_fixed", "no_meta"])
                for a, v in r.items(): rec.setdefault((hyp, a), []).append(v)
                finfo.append({"key": fk, "hyp": hyp, "transform_used": T, "resid_chip_units_max": float(max(res)), "n": len(pick),
                              "frac_chips_no_nodata": float(ok.mean()), "median_cos_as_shipped": float(np.median(cos(emb[pick], r["as_shipped"])))})
            rel_all.append(emb[pick])
            gb_ = t.column("geometry").to_pylist(); first = {}; dup_cos = []
            for i_, g_ in enumerate(gb_):
                if g_ in first: dup_cos.append(float(cos(emb[[first[g_]]], emb[[i_]])[0]))
                else: first[g_] = i_
            finfo.append({"key": fk, "n_unique_geoms": len(first), "n_duplicate_rows": len(dup_cos),
                          "dup_pair_cos_min": min(dup_cos) if dup_cos else None, "dup_pair_cos_median": float(np.median(dup_cos)) if dup_cos else None})
            finfo.append({"key": fk, "meta": meta, "rows": len(geoms), "dim": int(emb.shape[1]), "footprint_width_m_median": float(np.median(widths)),
                          "item_json": jurl, "asset_keys": sorted(A)[:30], "transform_10m": tr10, "transform_stacchip": tr_st})
            print("s2old", fk, finfo[-2:], flush=True)
        except Exception as e:
            LOG["errors"].append(f"s2old {fk}: {e!r}"); traceback.print_exc()
    out = {"files": finfo}
    if rel_all:
        rel = np.concatenate(rel_all)
        for (hyp, a), v in rec.items():
            out[f"{hyp}/{a}"] = summarize(rel, np.concatenate(v))[0]
    return out


if __name__ == "__main__":
    t0 = time.time()
    rt = ClayRuntime(device="cuda")
    LOG["clay_evidence"] = rt.evidence
    res = {}
    for name, fn in (("naip_rev2", naip_set), ("s2_deprecated", s2_old_set)):
        try:
            r = fn(rt)
            if isinstance(r, tuple):
                res[name], rows = r; json.dump(rows, open(f"manifestB_{name}.json", "w"), default=str)
            else:
                res[name] = r
        except Exception as e:
            LOG["errors"].append(f"{name}: {e!r}"); traceback.print_exc()
        json.dump({"results": res, "log": LOG, "elapsed_s": time.time() - t0}, open("resultsB.json", "w"), indent=1, default=str)
    print(json.dumps(res, indent=1, default=str)[:6000])
