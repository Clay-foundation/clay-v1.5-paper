"""E5 evaluation: Clay v1.5 (E1 vectors) vs comparators on E1's AEF-suite rows, identical to E1 eval_aef.py
(label_budget seeds 0-4, budgets 1/10/max with E1's completeness rule, LR C=1 / RidgeCV + kNN-3, BA / R2,
1000 block-bootstrap resamples seed 1729, paired). Each comparator X is scored together with clay_cls and clay_center
on the rows where all three have features ("comparison set"), so contrasts are paired on identical rows."""
import argparse, glob, json, os, sys, time, warnings
import numpy as np, pandas as pd
from joblib import Parallel, delayed
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import r2_score, f1_score
try:
    from scipy.special import sph_harm_y as _shy
    def sph_harm(m, l, ph, th): return _shy(l, m, th, ph)
except ImportError:
    from scipy.special import sph_harm
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "common"))
import evalkit as ek

REG = {"aster_ged", "openet_ensemble"}; CHG = {"lcmap_lcc", "lcmap_luc"}
NMAX = {"africa_crop_mask": 200, "aster_ged": 200, "canada_crops_coarse": 75, "canada_crops_fine": 68, "descals": 200,
        "ethiopia_crops": 49, "glance": 300, "lcmap_lc": 300, "lcmap_lu": 300, "lcmap_lcc": 300, "lcmap_luc": 150,
        "lucas_lc": 300, "lucas_lu": 300, "openet_ensemble": 300, "us_trees": 300}

def bal_acc(y, p):
    y = np.asarray(y, int); p = np.asarray(p, int); C = max(y.max(), p.max()) + 1
    n = np.bincount(y, minlength=C); tp = np.bincount(y[y == p], minlength=C); m = n > 0
    return float((tp[m] / n[m]).mean())

def fit_predict(Xtr, ytr, Xte, reg, head, seed):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        if head == "linear":
            m = ek.reg_head() if reg else ek.clf_head()
            if not reg: m.set_params(logisticregression__random_state=seed)
        else:
            k = min(3, len(ytr)); m = KNeighborsRegressor(k) if reg else KNeighborsClassifier(k)
        m.fit(Xtr, ytr); p = m.predict(Xte)
        nc = sum("converge" in str(x.message).lower() for x in w)
    return p, nc

def sh_features(lat, lon, L=10):
    th = np.radians(90 - np.asarray(lat)); ph = np.radians(np.asarray(lon)); out = []
    for l in range(L + 1):
        for m in range(-l, l + 1):
            Y = sph_harm(abs(m), l, ph, th)
            out.append(np.sqrt(2) * (-1) ** m * (Y.imag if m < 0 else Y.real) if m != 0 else Y.real)
    return np.stack(out, 1).astype(np.float32)

def load_e1(shard_dir):
    keys = ["clay_cls", "clay_center"]; F = {k: [] for k in keys}; ids = []
    for f in sorted(glob.glob(f"{shard_dir}/shard_*.npz")):
        z = np.load(f); ids += z["ids"].tolist()
        for k in keys: F[k].append(z[k].astype(np.float32))
    return {c: i for i, c in enumerate(ids)}, {k: np.concatenate(v) for k, v in F.items()}

def load_e5_chip(feat_dir):
    if not feat_dir or not glob.glob(f"{feat_dir}/feat_*.npz"): return {}, {}
    F, ids = {}, []
    for f in sorted(glob.glob(f"{feat_dir}/feat_*.npz")):
        z = np.load(f); ids += z["ids"].astype(str).tolist()
        for k in z.files:
            if k in ("ids", "n_harmonised"): continue
            F.setdefault(k, []).append(z[k].astype(np.float32))
    return {c: i for i, c in enumerate(ids)}, {k: np.concatenate(v) for k, v in F.items()}

def load_points(a):
    """pid -> vector for point arms (NaN where missing)."""
    P = pd.read_csv(a.points); n = P.pid.max() + 1; out = {}
    def put(name, pid, vecs, d):
        A = np.full((n, d), np.nan, np.float32); A[np.asarray(pid)] = vecs; out[name] = A
    for path, names in [(a.aef, ["aef_mean128", "aef_ov32_px", "aef_pixel"]), (a.tessera_patch, ["tessera_mean128", "tessera_pixel"])]:
        for f in (path or "").split(","):
            if not f or not os.path.exists(f): continue
            recs = [json.loads(l) for l in open(f)]
            for nm in names:
                rr = [(r["pid"], r[nm]) for r in recs if r.get(nm) is not None]
                if rr:
                    d = len(rr[0][1]); A = out.get(nm, np.full((n, d), np.nan, np.float32))
                    A[[p for p, _ in rr]] = np.array([v for _, v in rr], np.float32); out[nm] = A
    if a.tessera_pixel and os.path.exists(a.tessera_pixel):
        z = np.load(a.tessera_pixel); put("tessera_pixel", z["pid"], z["tessera_pixel"], 128)
    return P, out

def run_set(task, T, feats, arms, refs, out_preds, tag, full_tr):
    reg = task in REG; full_ids = (task + "|" + full_tr.row.astype(str)).values
    full_strata = (full_tr.partition.values if reg else full_tr.label.values).astype(int)
    ok = np.all([np.isfinite(feats[a]).all(1) for a in arms], 0)
    Tm = T[ok].reset_index(drop=True); fm = {a: feats[a][ok] for a in arms}
    tr = np.where(Tm.split == "train")[0]; te = np.where(Tm.split == "test")[0]
    info = {"task": task, "set": tag, "arms": arms, "n_rows_e1": int(len(T)), "n_rows_set": int(ok.sum()),
            "n_train": int(len(tr)), "n_test": int(len(te)), "n_test_e1": int((T.split == "test").sum())}
    if len(tr) == 0 or len(te) == 0: info["skipped"] = "no rows"; return [], [], info
    y = Tm.label.values.astype(float if reg else int); strata = Tm.partition.values.astype(int) if reg else y.astype(int)
    ids = (task + "|" + Tm.row.astype(str)).values; yte = y[te]
    blocks = ek.hblock(Tm.y.values[te], Tm.x.values[te])
    rng = np.random.default_rng(1729); ub = np.unique(blocks); rows = {b: np.where(blocks == b)[0] for b in ub}
    RS = [np.concatenate([rows[b] for b in rng.choice(ub, len(ub), replace=True)]) for _ in range(1000)]
    metric = r2_score if reg else bal_acc
    budgets = {"1": 1, "10": 10, "max": NMAX[task]}; avail = set(ids[tr])
    complete = {b: all(set(full_ids[ek.label_budget(full_ids, full_strata, k, sd)]) <= avail for sd in ek.SEEDS) for b, k in budgets.items()}
    budgets = {b: k for b, k in budgets.items() if complete[b] or b != "max"}
    info.update(n_test_blocks=int(len(ub)), budget_complete=complete)
    jobs = [(b, sd, a, h, tr[ek.label_budget(ids[tr], strata[tr], k, sd)]) for b, k in budgets.items() for sd in ek.SEEDS for a in arms for h in ["linear", "knn3"]]
    res = Parallel(n_jobs=int(os.environ.get("E5_NJOBS", "8")), backend="threading")(delayed(fit_predict)(fm[a][s], y[s], fm[a][te], reg, h, sd) for b, sd, a, h, s in jobs)
    P = {}; nconv = 0
    for (b, sd, a, h, s), (p, nc) in zip(jobs, res): P.setdefault((b, a, h), {})[sd] = p; nconv += nc
    info["convergence_warnings"] = int(nconv)
    summ, contr, boot = [], [], {}
    for (b, a, h), d in P.items():
        preds = [d[sd] for sd in ek.SEEDS]
        dist = np.array([np.mean([metric(yte[i], p[i]) for p in preds]) for i in RS]); boot[(b, a, h)] = dist
        rec = {"task": task, "set": tag, "budget": b, "arm": a, "head": h, "metric": "r2" if reg else "balanced_accuracy",
               "point": float(np.mean([metric(yte, p) for p in preds])), "lo": float(np.percentile(dist, 2.5)), "hi": float(np.percentile(dist, 97.5)),
               "seed_values": [float(metric(yte, p)) for p in preds], "n_test": int(len(te))}
        if not reg: rec["macro_f1"] = float(np.mean([f1_score(yte, p, average="macro") for p in preds]))
        summ.append(rec); out_preds[f"{tag}|{task}|{b}|{a}|{h}"] = np.stack(preds)
    for h in ["linear", "knn3"]:
        for b in budgets:
            for lhs, rhs in refs:
                pa, pb = [P[(b, lhs, h)][sd] for sd in ek.SEEDS], [P[(b, rhs, h)][sd] for sd in ek.SEEDS]
                dist = boot[(b, lhs, h)] - boot[(b, rhs, h)]
                contr.append({"task": task, "set": tag, "budget": b, "head": h, "contrast": f"{lhs}-{rhs}",
                              "point": float(np.mean([metric(yte, p) for p in pa]) - np.mean([metric(yte, p) for p in pb])),
                              "lo": float(np.percentile(dist, 2.5)), "hi": float(np.percentile(dist, 97.5)), "n_test": int(len(te))})
    out_preds[f"{tag}|{task}|ids_test"] = ids[te]; out_preds[f"{tag}|{task}|blocks"] = blocks
    return summ, contr, info

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--e1_shards", required=True); ap.add_argument("--rows", required=True); ap.add_argument("--rows_e5", required=True)
    ap.add_argument("--points", required=True); ap.add_argument("--aef", default=""); ap.add_argument("--tessera_pixel", default="")
    ap.add_argument("--tessera_patch", default=""); ap.add_argument("--e5_chip", default=""); ap.add_argument("--out", required=True)
    ap.add_argument("--tasks", default=""); ap.add_argument("--only", default="")
    a = ap.parse_args()
    cidx, F1 = load_e1(a.e1_shards); c5, F5 = load_e5_chip(a.e5_chip); Pt, FP = load_points(a)
    key = {(round(x, 7), round(y, 7), int(yr)): int(p) for p, x, y, yr in zip(Pt.pid, Pt.x, Pt.y, Pt.year)}
    R = pd.read_csv(a.rows, keep_default_na=False, dtype={"chip": str, "chip_before": str, "chip_after": str})
    R["partition"] = pd.to_numeric(R["partition"], errors="coerce"); R["row"] = R.row.astype(int)
    R["tier"] = pd.to_numeric(R.tier, errors="coerce")
    E5 = pd.read_csv(a.rows_e5)[["task", "row", "year_before", "year_after"]]
    R = R.merge(E5, on=["task", "row"], how="left")
    tasks = a.tasks.split(",") if a.tasks else sorted(R.task.unique())
    S, C, I, preds = [], [], [], {}
    point_arms = [k for k in ["aef_mean128", "aef_ov32_px", "aef_pixel", "tessera_pixel", "tessera_mean128"] if k in FP]
    chip_arms = sorted(F5)
    print("point arms", point_arms, "chip arms", chip_arms, flush=True)
    for t in tasks:
        t0 = time.time(); chg = t in CHG
        T = R[(R.task == t) & ((R.split == "train") | (R.tier == 0))]
        okc = (T.chip_before.isin(cidx) & T.chip_after.isin(cidx)) if chg else T.chip.isin(cidx)
        full_tr = T[T.split == "train"]  # E1 rule: completeness vs all tier-filtered train rows, before chip filter
        T = T[okc].reset_index(drop=True)
        def chipfeat(Fd, ci, k):
            def g(cs): return np.stack([Fd[k][ci[c]] if c in ci else np.full(Fd[k].shape[1], np.nan, np.float32) for c in cs])
            return g(T.chip_after) - g(T.chip_before) if chg else g(T.chip)
        feats = {k: chipfeat(F1, cidx, k) for k in F1}
        for k in chip_arms: feats[k] = chipfeat(F5, c5, k)
        def pid(col): return np.array([key.get((round(x, 7), round(y, 7), int(yr)), -1) for x, y, yr in zip(T.x, T.y, T[col])])
        pa, pb = pid("year_after"), pid("year_before")
        for k in point_arms:
            A = FP[k]; get = lambda p: np.where((p >= 0)[:, None], A[np.clip(p, 0, None)], np.nan)
            feats[k] = get(pa) - get(pb) if chg else get(pb)
        if not chg:
            feats["sh10"] = sh_features(T.y.values, T.x.values, 10)
            if "satclip_l40" in feats: feats["clay_cls+satclip_l40"] = np.concatenate([feats["clay_cls"], feats["satclip_l40"]], 1)
        comps = [k for k in feats if k not in ("clay_cls", "clay_center")]
        if a.only: comps = [k for k in comps if k in a.only.split(",")]
        for X in comps:
            arms = ["clay_cls", "clay_center", X]
            refs = [("clay_cls", X), ("clay_center", X), ("clay_cls", "clay_center")]
            s, cc, i = run_set(t, T, feats, arms, refs, preds, X, full_tr)
            S += s; C += cc; I.append(i)
        print(t, len(T), comps, f"{time.time()-t0:.0f}s", flush=True)
        json.dump({"summaries": S, "contrasts": C, "sets": I}, open(a.out, "w"))
    np.savez_compressed(a.out.replace(".json", "_predictions.npz"), **preds)

if __name__ == "__main__":
    main()
