"""E8 items 2-4 (CPU): pretraining overlap, E1 sensitivity (re-split, train-test distance), E2 location probe.
Implements E8 PROTOCOL.md (frozen 2026-11-05 22:25 UTC).
Usage: python e8_cpu.py --mgrs FGB --e1shards DIR --e1rows rows.csv --e2emb DIR --e2man manifest_E2.csv
                        --e3man manifest.csv --e4man manifestA_chips.csv.gz --out OUTDIR
"""
import argparse, glob, json, os, sys, time
import numpy as np, pandas as pd
sys.path.insert(0, "common")
import evalkit as ek
from sklearn.metrics import r2_score
from sklearn.neighbors import BallTree

t00 = time.time()
def say(*a): print(round(time.time() - t00), *a, flush=True)

REG = {"aster_ged", "openet_ensemble"}; CHG = {"lcmap_lcc", "lcmap_luc"}


def bal_acc(y, p):
    y = np.asarray(y, int); p = np.asarray(p, int); C = max(y.max(), p.max()) + 1
    n = np.bincount(y, minlength=C); tp = np.bincount(y[y == p], minlength=C); m = n > 0
    return float((tp[m] / n[m]).mean())


def resamples(blocks, n=1000, seed=1729):
    rng = np.random.default_rng(seed); ub = np.unique(blocks); rows = {b: np.where(blocks == b)[0] for b in ub}
    return [np.concatenate([rows[b] for b in rng.choice(ub, len(ub), replace=True)]) for _ in range(n)]


def ci(d):
    return [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]


def gcd_km(la1, lo1, la2, lo2):
    la1, lo1, la2, lo2 = map(np.radians, (la1, lo1, la2, lo2))
    a = np.sin((la2 - la1) / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2
    return 6371 * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


# ---------------- E1 helpers ----------------
def load_e1(shard_dir, rows_csv):
    F, ids = {"clay_cls": [], "stats": []}, []
    for f in sorted(glob.glob(f"{shard_dir}/shard_*.npz")):
        z = np.load(f); ids.extend(z["ids"].tolist())
        for k in F:
            F[k].append(z[k].astype(np.float32))
    F = {k: np.concatenate(v) for k, v in F.items()}
    cidx = {c: i for i, c in enumerate(ids)}
    R = pd.read_csv(rows_csv, keep_default_na=False, dtype={"chip": str, "chip_before": str, "chip_after": str})
    R["partition"] = pd.to_numeric(R["partition"], errors="coerce"); R["row"] = R["row"].astype(int)
    return cidx, F, R


def e1_task_rows(task, T, cidx):
    if "tier" in T.columns:
        T = T[(T.split == "train") | (T.tier.astype(int) == 0)]
    ok = (T.chip_before.isin(cidx) & T.chip_after.isin(cidx)) if task in CHG else T.chip.isin(cidx)
    return T[ok].reset_index(drop=True)


def e1_feats(task, T, cidx, F):
    out = {}
    for arm in ("clay_cls", "stats"):
        if task in CHG:
            out[arm] = F[arm][[cidx[c] for c in T.chip_after]] - F[arm][[cidx[c] for c in T.chip_before]]
        else:
            out[arm] = F[arm][[cidx[c] for c in T.chip]]
    return out


def e1_contrast(task, T, feats, tr, te, k=10):
    reg = task in REG
    y = T.label.values.astype(float if reg else int)
    strata = (T.partition.values if reg else T.label.values).astype(int)
    ids = (task + "|" + T.row.astype(str)).values
    metric = r2_score if reg else bal_acc
    P = {a: [] for a in feats}
    for sd in ek.SEEDS:
        sel = tr[ek.label_budget(ids[tr], strata[tr], k, sd)]
        for a, X in feats.items():
            m = ek.reg_head() if reg else ek.clf_head()
            if not reg:
                m.set_params(logisticregression__random_state=sd)
            P[a].append(m.fit(X[sel], y[sel]).predict(X[te]))
    yte = y[te]; blocks = ek.hblock(T.y.values[te], T.x.values[te]); RS = resamples(blocks)
    pt = {a: float(np.mean([metric(yte, p) for p in P[a]])) for a in P}
    bd = {a: np.array([np.mean([metric(yte[i], p[i]) for p in P[a]]) for i in RS]) for a in P}
    d = bd["clay_cls"] - bd["stats"]
    return {"clay_cls": [pt["clay_cls"], *ci(bd["clay_cls"])], "stats": [pt["stats"], *ci(bd["stats"])],
            "clay_cls-stats": [pt["clay_cls"] - pt["stats"], *ci(d)], "n_train": int(len(tr)), "n_test": int(len(te)),
            "n_test_blocks": int(len(np.unique(blocks))), "n_train_classes": int(len(np.unique(strata[tr]))),
            "n_test_classes": int(len(np.unique(strata[te])))}


def nn_dist_km(lat_q, lon_q, lat_r, lon_r):
    bt = BallTree(np.radians(np.c_[lat_r, lon_r]), metric="haversine")
    d, _ = bt.query(np.radians(np.c_[lat_q, lon_q]), k=1)
    return d[:, 0] * 6371.0


def dist_summary(d):
    return {"n": int(len(d)), "median_km": float(np.median(d)), "q25_km": float(np.percentile(d, 25)),
            "q75_km": float(np.percentile(d, 75)), "frac_lt_1km": float((d < 1).mean()),
            "frac_lt_10km": float((d < 10).mean()), "frac_lt_100km": float((d < 100).mean())}


def item3(a, out):
    cidx, F, R = load_e1(a.e1shards, a.e1rows)
    res = {"resplit": {}, "distance": {}}
    for task in sorted(R.task.unique()):
        T = e1_task_rows(task, R[R.task == task], cidx)
        feats = e1_feats(task, T, cidx, F)
        # AEF split (reproduces E1 n=10 contrast as a check)
        tr0 = np.where(T.split == "train")[0]; te0 = np.where(T.split == "test")[0]
        r = {"aef_split": e1_contrast(task, T, feats, tr0, te0)}
        # hblock re-split
        sp = ek.block_split(ek.hblock(T.y.values, T.x.values))
        tr1 = np.where(sp != "test")[0]; te1 = np.where(sp == "test")[0]
        strata = (T.partition.values if task in REG else T.label.values).astype(int)
        if len(np.unique(ek.hblock(T.y.values[te1], T.x.values[te1]))) < 2 or \
                len(np.unique(strata[tr1])) < len(np.unique(strata)):
            r["hblock_split"] = {"skipped": "fewer than 2 test blocks or a class missing in train",
                                 "n_train": int(len(tr1)), "n_test": int(len(te1))}
        else:
            r["hblock_split"] = e1_contrast(task, T, feats, tr1, te1)
        res["resplit"][task] = r
        lat, lon = T.y.values.astype(float), T.x.values.astype(float)
        res["distance"][task] = {"aef_split": dist_summary(nn_dist_km(lat[te0], lon[te0], lat[tr0], lon[tr0])),
                                 "hblock_split": dist_summary(nn_dist_km(lat[te1], lon[te1], lat[tr1], lon[tr1]))}
        say("E1", task, {k: v.get("clay_cls-stats") for k, v in r.items()}, res["distance"][task]["aef_split"]["median_km"])
        json.dump(res, open(os.path.join(out, "item3_e1_sensitivity.json"), "w"), indent=1)


# ---------------- item 2 ----------------
def item2(a, out):
    import geopandas as gpd
    from shapely.geometry import Point
    tiles = gpd.read_file(a.mgrs).to_crs(4326)
    res = {"mgrs_columns": [c for c in tiles.columns if c != "geometry"], "n_tiles": int(len(tiles))}
    namecol = next((c for c in tiles.columns if c.lower() in ("name", "mgrs", "tile", "mgrs_tile")), None)

    def inside(lat, lon):
        pts = gpd.GeoDataFrame(geometry=[Point(x, y) for x, y in zip(lon, lat)], crs=4326)
        j = gpd.sjoin(pts, tiles[["geometry"]], predicate="within", how="left")
        return j.groupby(level=0)["index_right"].apply(lambda s: s.notna().any()).reindex(range(len(pts))).values.astype(bool)

    def frac(m):
        return {"n": int(len(m)), "n_inside": int(m.sum()), "frac_inside": float(m.mean()) if len(m) else None}

    ids = []
    for f in sorted(glob.glob(f"{a.e1shards}/shard_*.npz")):
        ids.extend(np.load(f)["ids"].tolist())
    cidx = {c: i for i, c in enumerate(ids)}
    R = pd.read_csv(a.e1rows, keep_default_na=False, dtype={"chip": str, "chip_before": str, "chip_after": str})
    E1 = pd.concat([e1_task_rows(t, R[R.task == t], cidx).assign(task=t)[["task", "row", "split", "x", "y"]]
                    for t in sorted(R.task.unique())])
    E1.to_csv(os.path.join(out, "e1_rows_with_chip.csv"), index=False)
    E1 = E1.reset_index(drop=True)
    E1["inside"] = inside(E1.y.values, E1.x.values)
    res["E1_aef"] = {"test_pooled": frac(E1.inside[E1.split == "test"].values),
                     "train_pooled": frac(E1.inside[E1.split == "train"].values),
                     "test_unique_locations": frac(E1[E1.split == "test"].drop_duplicates(["x", "y"]).inside.values),
                     "per_task_test": {t: frac(g.inside[g.split == "test"].values) for t, g in E1.groupby("task")}}
    E2 = pd.read_csv(a.e2man); E2["inside"] = inside(E2.lat.values, E2.lon.values)
    res["E2"] = {"all_chips": frac(E2.inside.values), "test_chips": frac(E2.inside[E2.split == "test"].values),
                 "per_sensor_test": {s: frac(g.inside[g.split == "test"].values) for s, g in E2.groupby("sensor")},
                 "per_stratum_test": {s: frac(g.inside[g.split == "test"].values) for s, g in E2.groupby("stratum")}}
    E3 = pd.read_csv(a.e3man); E3["inside"] = inside(E3.lat.values, E3.lon.values)
    sp = ek.block_split(ek.hblock(E3.lat.values, E3.lon.values, 100.0))
    res["E3"] = {"all_locations": frac(E3.inside.values), "test_locations": frac(E3.inside.values[sp == "test"]),
                 "manifest_split_test": frac(E3.inside[E3.split == "test"].values)}
    E4 = pd.read_csv(a.e4man); E4["inside"] = inside(E4.lat.values, E4.lon.values)
    res["E4_partA"] = {"all_chips": frac(E4.inside.values),
                       "per_region_chipsize": {f"{r}_{p}": frac(g.inside.values) for (r, p), g in E4.groupby(["region", "chip_px"])}}
    if namecol:
        names = set(tiles[namecol].astype(str))
        res["E4_partA"]["tile_name_in_sample"] = {t: (t in names or ("T" + t) in names) for t in ("11SLU", "15STT", "18STE")}
        res["mgrs_name_column"] = namecol
    json.dump(res, open(os.path.join(out, "item2_overlap.json"), "w"), indent=1)
    say("item2", {k: v for k, v in res.items() if k in ("E3", "E4_partA")})


# ---------------- item 4 ----------------
def item4(a, out):
    res = {}
    for s in ("s2", "s1", "ls", "naip"):
        p = os.path.join(a.e2emb, f"emb_{s}.npz")
        if not os.path.exists(p):
            continue
        z = np.load(p, allow_pickle=True)
        lat, lon, split, blocks = z["lat"].astype(float), z["lon"].astype(float), z["split"], z["blocks"]
        arms = {"cls_on_training": z["cls"], "cls_on_documented": z["cls_doc"], "cls_off": z["cls_off"]}
        te = split == "test"; tr = ~te
        conus = (lon >= -124.7) & (lon <= -66.9) & (lat >= 24.5) & (lat <= 49.4)
        subsets = {"all": te, "conus": te & conus, "non_conus": te & ~conus}
        R = {"n_test": {k: int(v.sum()) for k, v in subsets.items()}}
        boots = {}
        for arm, X in arms.items():
            X = X.astype(np.float32)
            plat = ek.reg_head().fit(X[tr], lat[tr]).predict(X)
            plon = ek.reg_head().fit(X[tr], lon[tr]).predict(X)
            la, lo = np.radians(lat), np.radians(lon)
            V = np.c_[np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)]
            pv = ek.reg_head().fit(X[tr], V[tr]).predict(X); pv /= np.linalg.norm(pv, axis=1, keepdims=True)
            vlat, vlon = np.degrees(np.arcsin(np.clip(pv[:, 2], -1, 1))), np.degrees(np.arctan2(pv[:, 1], pv[:, 0]))
            d_ll = gcd_km(lat, lon, plat, plon); d_v = gcd_km(lat, lon, vlat, vlon)
            A = {}
            for sn, m in subsets.items():
                ii = np.where(m)[0]
                if len(ii) < 10:
                    A[sn] = {"n": int(len(ii))}; continue
                RS = resamples(blocks[ii])
                f = {"lon_r2": lambda j: r2_score(lon[ii][j], plon[ii][j]), "lat_r2": lambda j: r2_score(lat[ii][j], plat[ii][j]),
                     "median_km": lambda j: float(np.median(d_ll[ii][j])), "median_km_unitvec": lambda j: float(np.median(d_v[ii][j]))}
                A[sn] = {"n": int(len(ii)), "n_blocks": int(len(np.unique(blocks[ii])))}
                for k, fn in f.items():
                    bs = np.array([fn(j) for j in RS]); boots[(arm, sn, k)] = bs
                    A[sn][k] = [float(fn(np.arange(len(ii)))), *ci(bs)]
            R[arm] = A
        for sn in subsets:
            for k in ("lon_r2", "lat_r2", "median_km", "median_km_unitvec"):
                for other in ("cls_off", "cls_on_documented"):
                    if ("cls_on_training", sn, k) in boots and (other, sn, k) in boots:
                        R.setdefault("contrasts", {})[f"{sn}|{k}|training-{other}"] = [
                            R["cls_on_training"][sn][k][0] - R[other][sn][k][0], *ci(boots[("cls_on_training", sn, k)] - boots[(other, sn, k)])]
        res[s] = R
        say("item4", s, {arm: R[arm]["all"].get("lon_r2") for arm in arms}, {arm: R[arm]["all"].get("median_km") for arm in arms})
        json.dump(res, open(os.path.join(out, "item4_e2_location.json"), "w"), indent=1)


def main():
    ap = argparse.ArgumentParser()
    for k in ("mgrs", "e1shards", "e1rows", "e2emb", "e2man", "e3man", "e4man", "out"):
        ap.add_argument("--" + k, required=True)
    ap.add_argument("--items", default="3,2,4")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    for it in a.items.split(","):
        {"2": item2, "3": item3, "4": item4}[it](a, a.out)
    say("done")


if __name__ == "__main__":
    main()
