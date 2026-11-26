"""E4 Part A: PV retrieval at a review budget on full S2 MGRS tiles (USPVDB labels).
Stages: embed (per region & chip size -> cache/*.npz) and score (retrieval + block bootstrap -> resultsA.json).

Usage (from this directory):
  E4_CACHE=cache E4_DATA=data python partA.py --stage embed      # documented-order + DINOv2 + band-stat arms (GPU, ~140 s read per tile)
  E4_CACHE=cache E4_DATA=data python partA.py --stage nometa     # metadata-off Clay arm
  E4_CACHE=cache E4_DATA=data E4_TRAIN_ARMS=1 python partA.py --stage train   # training-order Clay arms, then scores
  E4_CACHE=cache E4_TRAIN_ARMS=1 python partA.py --stage score --out resultsA.json
E4_DATA: directory with the USPVDB v4.0 GeoJSON (https://energy.usgs.gov/uspvdb/data/). regions.json lists the three
Sentinel-2 L2A scenes (Planetary Computer item ids). Cached embeddings are float16."""
import argparse, glob, hashlib, json, os, sys, time
HERE_ = os.path.dirname(os.path.abspath(__file__))
import pyproj
os.environ.setdefault("PROJ_DATA", pyproj.datadir.get_data_dir())
import numpy as np
sys.path.insert(0, os.path.join(HERE_, "..", "common"))
ASSET = {"blue": "B02", "green": "B03", "red": "B04", "rededge1": "B05", "rededge2": "B06", "rededge3": "B07",
         "nir": "B08", "nir08": "B8A", "swir16": "B11", "swir22": "B12"}
SIZES = (64, 256); KS = (1, 5, 20); M_NEG = 20; BUDGETS = (10, 100, 1000); SEEDS = (0, 1, 2, 3, 4)
ARMS = ("clay_cls", "clay_mean", "dinov2_rgb", "spectral_stats", "random"); BLOCK_M = 10_000; NBOOT = 1000
CACHE = os.environ.get("E4_CACHE", "cache")
DATA = os.environ.get("E4_DATA", "data")
TRAIN_ARMS = os.environ.get("E4_TRAIN_ARMS", "0") == "1"


def h(s):
    return hashlib.sha256(s.encode()).hexdigest()


def embed_region(reg, gpv, nometa=False, train=False):
    import rasterio, torch, planetary_computer, pystac_client
    from rasterio.enums import Resampling
    from rasterio import features
    from datetime import datetime
    from clayrt import ClayRuntime, DinoV2Runtime, to_display_rgb, platform_spec
    spec = platform_spec("sentinel-2-l2a")
    cat = pystac_client.Client.open("https://planetarycomputer.microsoft.com/api/stac/v1", modifier=planetary_computer.sign_inplace)
    it = cat.get_collection("sentinel-2-l2a").get_item(reg["item"])
    date = datetime.fromisoformat(it.properties["datetime"].replace("Z", "+00:00"))
    t0 = time.time(); bands = []
    with rasterio.open(it.assets["B02"].href) as src:
        H, W, tr, crs = src.height, src.width, src.transform, src.crs
    for b in spec["bands"]:
        with rasterio.open(it.assets[ASSET[b]].href) as src:  # 20 m bands -> 10 m nearest, as stacchip
            bands.append(src.read(1, out_shape=(H, W), resampling=Resampling.nearest))
    arr = np.stack(bands); del bands
    print(reg["state"], "read", arr.shape, round(time.time() - t0), "s", flush=True)
    g = gpv.to_crs(crs)
    old = g[g.p_year <= 2022]; amb = g[~(g.p_year <= 2022)]  # built 2023+ or unknown year: label ambiguous in summer-2023 image
    m_pv = features.rasterize(((geom, 1) for geom in old.geometry), out_shape=(H, W), transform=tr, dtype="uint8")
    m_amb = features.rasterize(((geom, 1) for geom in amb.geometry), out_shape=(H, W), transform=tr, dtype="uint8") if len(amb) else np.zeros((H, W), "uint8")
    to_ll = pyproj.Transformer.from_crs(crs, 4326, always_xy=True)
    rt = ClayRuntime(device="cuda"); dino = None if (nometa or train) else DinoV2Runtime(size=224)
    import timm  # shared DinoV2Runtime keeps timm's 518-px default; rebuild same weights at img_size=224 (pos-embed interpolated)
    if dino is not None: dino.m = timm.create_model(dino.name, pretrained=True, num_classes=0, img_size=224).eval().to("cuda")
    for cs in SIZES:
        n = H // cs; nn = n * cs
        def tiles(a):
            return a[..., :nn, :nn].reshape(*a.shape[:-2], n, cs, n, cs)
        pv = tiles(m_pv).mean(axis=(1, 3)).reshape(-1); am = tiles(m_amb).mean(axis=(1, 3)).reshape(-1)
        mn = tiles(arr).min(axis=(-3, -1)).min(axis=0).reshape(-1)  # min over band and pixels
        rr, cc = np.divmod(np.arange(n * n), n)
        valid = mn > 0
        x = tr.c + (cc + 0.5) * cs * tr.a; y = tr.f + (rr + 0.5) * cs * tr.e
        lon, lat = to_ll.transform(x, y)
        idx = np.where(valid)[0]
        out = {k: [] for k in ("cls", "mean", "dino", "spec")}
        chunk = 1024 if cs == 64 else 128
        for i in range(0, len(idx), chunk):
            sel = idx[i:i + chunk]
            px = np.stack([arr[:, r * cs:(r + 1) * cs, c * cs:(c + 1) * cs] for r, c in zip(rr[sel], cc[sel])]).astype("float32")
            if train:
                e = rt.embed(px, platform="sentinel-2-l2a", latlon=[(float(a), float(b)) for a, b in zip(lat[sel], lon[sel])], date=date,
                             batch_size=256 if cs == 64 else 16, latlon_order="training")
                out["cls"].append(e["cls"].astype("float16")); out["mean"].append(e["mean"].astype("float16")); del e
                continue
            if nometa:
                out["cls"].append(rt.embed(px, platform="sentinel-2-l2a", latlon=None, date=None, batch_size=256 if cs == 64 else 16)["cls"].astype("float16"))
                continue
            e = rt.embed(px, platform="sentinel-2-l2a", latlon=[(float(a), float(b)) for a, b in zip(lat[sel], lon[sel])], date=date,
                         batch_size=256 if cs == 64 else 16, latlon_order="documented")  # explicit: clayrt's default is now "training"
            out["cls"].append(e["cls"].astype("float16")); out["mean"].append(e["mean"].astype("float16")); del e
            out["dino"].append(dino.embed(to_display_rgb(px, "sentinel-2-l2a"), batch_size=128)["cls"].astype("float16"))
            out["spec"].append(np.concatenate([px.mean(axis=(2, 3)), px.std(axis=(2, 3))], 1).astype("float32"))
        ids_nm = np.array([f"{reg['state']}_{reg['tile']}_c{cs}_r{r:03d}_c{c:03d}" for r, c in zip(rr[idx], cc[idx])])
        if train:
            np.savez(f"{CACHE}/{reg['state']}_{cs}_train.npz", ids=ids_nm, cls=np.concatenate(out["cls"]), mean=np.concatenate(out["mean"]))
            print("train", reg["state"], cs, flush=True); continue
        if nometa:
            np.savez(f"{CACHE}/{reg['state']}_{cs}_nometa.npz", ids=ids_nm, cls=np.concatenate(out["cls"])); print("nometa", reg["state"], cs, flush=True); continue
        res = {k: np.concatenate(v) for k, v in out.items()}
        ids = np.array([f"{reg['state']}_{reg['tile']}_c{cs}_r{r:03d}_c{c:03d}" for r, c in zip(rr[idx], cc[idx])])
        blk = np.array([f"{reg['state']}|{int(a // BLOCK_M)}|{int(b // BLOCK_M)}" for a, b in zip(x[idx], y[idx])])
        np.savez(f"{CACHE}/{reg['state']}_{cs}.npz", ids=ids, row=rr[idx], col=cc[idx], x=x[idx], y=y[idx], lat=np.asarray(lat)[idx],
                 lon=np.asarray(lon)[idx], frac_pv=pv[idx], frac_amb=am[idx], block=blk, n_total=n * n, n_invalid=int((~valid).sum()), **res)
        print(reg["state"], cs, "chips", n * n, "valid", len(idx), "pos", int((pv[idx] >= 0.1).sum()), round(time.time() - t0), "s",
              "gpu_gb", round(torch.cuda.max_memory_allocated() / 1e9, 2), flush=True)
    if nometa or train: return None
    return {"item": reg["item"], "datetime": it.properties["datetime"], "baseline": it.properties.get("s2:processing_baseline"),
            "crs": str(crs), "transform": list(tr)[:6], "shape": [H, W], "n_sites_old": int(len(old)), "n_sites_amb": int(len(amb)),
            "clay_evidence": rt.evidence, "dino": dino.name}


def l2(a):
    a = a.astype("float32"); return a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-12)


def metrics_weighted(order, y, W, pos_total=None):
    """order: ranking (desc score). y: 0/1. W: (B,N) multiplicities (row of ones = point estimate). Returns dict of (B,) arrays."""
    ys = y[order].astype("float32"); Ws = W[:, order]
    n = np.cumsum(Ws, 1); tp = np.cumsum(Ws * ys, 1); P = tp[:, -1]
    out = {}
    for K in BUDGETS:
        j = np.minimum((n < K).sum(1), n.shape[1] - 1)
        r = np.arange(len(W)); out[f"P@{K}"] = tp[r, j] / n[r, j]; out[f"R@{K}"] = np.where(P > 0, tp[r, j] / np.maximum(P, 1e-9), np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        out["AP"] = np.where(P > 0, (Ws * ys * tp / n).sum(1) / P, np.nan)
    return out


def score(regions):
    rng_boot = np.random.default_rng(1729)
    results = {"per_seed": [], "summary": [], "contrasts": [], "pools": {}, "gallery": {}}
    for cs in SIZES:
        D = {r["state"]: dict(np.load(f"{CACHE}/{r['state']}_{cs}.npz")) for r in regions}
        allspec = np.concatenate([d["spec"][d["frac_amb"] < 0.1] for d in D.values()])
        mu, sd = allspec.mean(0), allspec.std(0) + 1e-6
        F = {}
        for s, d in D.items():
            keep = d["frac_amb"] < 0.1
            for k in list(d):
                if hasattr(d[k], "__len__") and getattr(d[k], "shape", ()) and d[k].shape[0] == len(keep):
                    d[k] = d[k][keep]
            d["y"] = (d["frac_pv"] >= 0.1).astype("int8")
            nm = f"{CACHE}/{s}_{cs}_nometa.npz"
            if os.path.exists(nm):
                z = np.load(nm); assert (z["ids"][keep] == d["ids"]).all(); d["cls_nometa"] = z["cls"][keep]
            F[s] = {"clay_cls": l2(d["cls"]), "clay_mean": l2(d["mean"]), "dinov2_rgb": l2(d["dino"]), "spectral_stats": l2((d["spec"] - mu) / sd)}
            if "cls_nometa" in d: F[s]["clay_cls_nometa"] = l2(d["cls_nometa"])
            tr_ = f"{CACHE}/{s}_{cs}_train.npz"
            if TRAIN_ARMS and os.path.exists(tr_):
                z = np.load(tr_); assert (z["ids"][keep] == d["ids"]).all()
                F[s]["clay_cls_train"] = l2(z["cls"][keep]); F[s]["clay_mean_train"] = l2(z["mean"][keep])
            results["gallery"][f"{s}_{cs}"] = {"n": int(len(d["y"])), "n_pos": int(d["y"].sum()), "prevalence": float(d["y"].mean()),
                                               "n_excluded_ambiguous": int((~keep).sum()), "n_invalid_nodata": int(d["n_invalid"]), "n_total": int(d["n_total"])}
        boots = {}  # (state) -> W
        for s, d in D.items():
            ub, inv = np.unique(d["block"], return_inverse=True)
            draws = rng_boot.integers(0, len(ub), size=(NBOOT, len(ub)))
            cnt = np.zeros((NBOOT, len(ub)), "float32")
            np.add.at(cnt, (np.repeat(np.arange(NBOOT), len(ub)), draws.ravel()), 1)
            boots[s] = np.vstack([np.ones((1, len(inv)), "float32"), cnt[:, inv]])  # row 0 = point estimate
        acc = {}  # (arm,k,mode,metric) -> {state: (B+1, seeds)}
        for s, d in D.items():
            others = [o for o in D if o != s]
            pos_pool = np.concatenate([D[o]["ids"][D[o]["y"] == 1] for o in others])
            neg_pool = np.concatenate([D[o]["ids"][(D[o]["frac_pv"] == 0) & (D[o]["frac_amb"] == 0)] for o in others])
            lookup = {o: {i: j for j, i in enumerate(D[o]["ids"])} for o in others}
            def vec(arm, idlist):
                return np.stack([F[i.split("_")[0]][arm][lookup[i.split("_")[0]][i]] for i in idlist])
            results["pools"][f"{s}_{cs}"] = {"n_query_pos_pool": int(len(pos_pool)), "n_query_neg_pool": int(len(neg_pool))}
            for seed in SEEDS:
                pr = pos_pool[np.argsort([h(f"claypaper-e4-qpos|{seed}|{i}") for i in pos_pool])]
                nr = neg_pool[np.argsort([h(f"claypaper-e4-qneg|{seed}|{i}") for i in neg_pool])][:M_NEG]
                rnd = np.random.default_rng(10_000 + seed).random(len(d["y"]))
                for k in KS:
                    q_ids = pr[:k]
                    for mode in ("pos", "pos_minus_neg"):
                        for arm in ARMS + tuple(a_ for a_ in ("clay_cls_nometa", "clay_cls_train", "clay_mean_train") if a_ in F[s]):
                            if arm == "random":
                                sc = rnd
                            else:
                                q = vec(arm, q_ids).mean(0)
                                if mode == "pos_minus_neg":
                                    q = q - vec(arm, nr).mean(0)
                                sc = F[s][arm] @ (q / np.linalg.norm(q))
                            order = np.argsort(-sc, kind="stable")
                            m = metrics_weighted(order, d["y"], boots[s])
                            for met, v in m.items():
                                acc.setdefault((arm, k, mode, met), {}).setdefault(s, []).append(v)
                                results["per_seed"].append({"chip": cs, "region": s, "arm": arm, "k": k, "mode": mode, "seed": seed,
                                                            "metric": met, "value": float(v[0]), "n_query_pos": int(len(q_ids))})
        def summarize(vals_by_state):
            per_reg = {s: np.nanmean(np.stack(v, 1), 1) for s, v in vals_by_state.items()}  # (B+1,) seed-mean
            pooled = np.nanmean(np.stack(list(per_reg.values()), 1), 1)
            return per_reg, pooled
        for (arm, k, mode, met), vb in acc.items():
            per_reg, pooled = summarize(vb)
            for s, v in list(per_reg.items()) + [("pooled", pooled)]:
                results["summary"].append({"chip": cs, "region": s, "arm": arm, "k": k, "mode": mode, "metric": met, "point": float(v[0]),
                                           "lo": float(np.nanpercentile(v[1:], 2.5)), "hi": float(np.nanpercentile(v[1:], 97.5))})
        for (k, mode, met) in {(a[1], a[2], a[3]) for a in acc}:
          for bname in ("clay_cls",) + (("clay_cls_train",) if ("clay_cls_train", k, mode, met) in acc else ()):
            base = summarize(acc[(bname, k, mode, met)])
            others = ("clay_mean", "dinov2_rgb", "spectral_stats", "random") + tuple(o for o in ("clay_cls_nometa",) if (o, k, mode, met) in acc)
            if bname == "clay_cls_train": others = others + ("clay_cls", "clay_mean_train")
            for other in others:
                oth = summarize(acc[(other, k, mode, met)])
                for s in list(base[0]) + ["pooled"]:
                    a = base[1] if s == "pooled" else base[0][s]; b = oth[1] if s == "pooled" else oth[0][s]
                    dlt = a - b
                    results["contrasts"].append({"chip": cs, "region": s, "contrast": f"{bname}-{other}", "k": k, "mode": mode, "metric": met,
                                                 "point": float(dlt[0]), "lo": float(np.nanpercentile(dlt[1:], 2.5)), "hi": float(np.nanpercentile(dlt[1:], 97.5))})
        print("scored", cs, flush=True)
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--stage", default="all"); ap.add_argument("--out", default=None, help="results JSON (default resultsA[_trainorder].json)"); ap.add_argument("--manifest-out", default="manifestA_chips.csv.gz"); a = ap.parse_args()
    import geopandas as gpd
    regions = json.load(open(os.path.join(HERE_, "regions.json")))
    os.makedirs(CACHE, exist_ok=True)
    if a.stage == "train":
        gpv = gpd.read_file(glob.glob(os.path.join(DATA, "*.geojson"))[0])
        for reg in regions:
            if not all(os.path.exists(f"{CACHE}/{reg['state']}_{cs}_train.npz") for cs in SIZES):
                embed_region(reg, gpv, train=True)
    if a.stage == "nometa":
        gpv = gpd.read_file(glob.glob(os.path.join(DATA, "*.geojson"))[0])
        for reg in regions:
            if not all(os.path.exists(f"{CACHE}/{reg['state']}_{cs}_nometa.npz") for cs in SIZES):
                embed_region(reg, gpv, nometa=True)
    if a.stage in ("embed", "all"):
        gpv = gpd.read_file(glob.glob(os.path.join(DATA, "*.geojson"))[0])
        meta = {}
        for reg in regions:
            if all(os.path.exists(f"{CACHE}/{reg['state']}_{cs}.npz") for cs in SIZES) and os.path.exists(f"{CACHE}/meta_{reg['state']}.json"):
                continue
            m = embed_region(reg, gpv); json.dump(m, open(f"{CACHE}/meta_{reg['state']}.json", "w"), indent=1, default=str)
    if a.stage in ("score", "all", "nometa", "train"):
        res = score(regions)
        res["scenes"] = {r["state"]: json.load(open(f"{CACHE}/meta_{r['state']}.json")) for r in regions}
        try:
            from provenance import provenance
            res["provenance"] = provenance(__file__, {"stage": a.stage, "train_arms": TRAIN_ARMS, "cache": os.path.basename(CACHE.rstrip("/"))})
        except Exception as e:  # noqa: BLE001
            res["provenance"] = {"error": str(e)}
        json.dump(res, open(a.out or ("resultsA_trainorder.json" if TRAIN_ARMS else "resultsA.json"), "w"), default=float)
        # manifest of chips
        import csv, gzip
        with gzip.open(a.manifest_out, "wt") as f:
            w = csv.writer(f); w.writerow(["chip_id", "chip_px", "region", "row", "col", "lat", "lon", "frac_pv", "frac_amb", "label", "block"])
            for cs in SIZES:
                for r in regions:
                    d = np.load(f"{CACHE}/{r['state']}_{cs}.npz")
                    for i in range(len(d["ids"])):
                        w.writerow([d["ids"][i], cs, r["state"], d["row"][i], d["col"][i], round(float(d["lat"][i]), 6), round(float(d["lon"][i]), 6),
                                    round(float(d["frac_pv"][i]), 4), round(float(d["frac_amb"][i]), 4),
                                    "excluded_ambiguous" if d["frac_amb"][i] >= 0.1 else int(d["frac_pv"][i] >= 0.1), d["block"][i]])
        print("done")
