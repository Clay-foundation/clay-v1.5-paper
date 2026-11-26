"""Complete 15-task AEF/TESSERA comparison (Table 2 Panel C + OpenET) from cached embeddings, matched and paired.

Replaces the earlier unreleased producer of e1_e5_complete.json, which filtered every arm independently (different test rows per
arm), picked point-feature years by earliest/latest/nearest-year fallback, and reported seed summaries without intervals.

What this script does (all CPU, from cached features):
  * Rows = E1 rows (all train rows with an E1 chip + tier-0 test rows), exactly as eval_e5.py.
  * Point features (AEF, TESSERA) are joined at the row's label year(s) from --label_years (build_label_years.py): exact (x, y, year),
    no nearest-year fallback.  Rows without an exact match have NaN features and drop out of any set that contains that arm.
  * Each comparison is scored on a *set* of arms that share identical rows: rows where every arm in the set has a feature.
    Within a set all arms use the identical test rows and the identical training draws (evalkit.label_budget, seeds 0-4, drawn
    from the set's available train rows by the same hash), the same heads (LR C=1 / RidgeCV) and the same 1,000 shared 100 km
    block-bootstrap resamples (seed 1729), so every contrast is a paired contrast with a paired interval.
  * Sets: "chip" (all chip arms), "clay+<point arm>" (the three Clay readouts and one point arm) and "common" (all chip arms and
    aef_mean128, aef_ov32_px, tessera_pixel on identical rows; this set feeds one-row-one-set Table 2 columns).
  * Budgets 10 and max.  Each budget records the nominal draw (full tier-filtered train pool) and the achieved draw (set's pool);
    `budget_complete` is true only if every seed's nominal selection is inside the set's pool (E1's completeness rule).
    A budget is never silently relabelled: summarize_panelC.py counts incomplete max cells separately.
  * Per set and per arm: sha256 digests of the test ids, train-pool ids and, per budget and seed, the selected training ids,
    computed from the arrays actually passed to the fit.

Usage (from the repo root; all inputs under --data_root with the names below, each overridable):
  python experiments/suite/E5_sota_comparators/code/run_e1_e5_complete.py --data_root DATA --out OUT.json
  DATA/ = e1/shard_*.npz  e1/meta.csv  e5_gpu_feats.npz  e1_rows.csv  points.csv  aef_ov32.jsonl  aef_pixel.jsonl
          tessera_patch.jsonl  label_years.csv
Environment: E5_NJOBS (parallel fits, default 8), GIT_COMMIT (recorded in the JSON; default `git rev-parse HEAD`).
"""
import argparse, datetime, glob, hashlib, json, os, platform, subprocess, sys, time, warnings
import numpy as np, pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import r2_score
import sklearn, scipy, joblib

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "common"))
import evalkit as ek

REG = {"aster_ged", "openet_ensemble"}; CHG = {"lcmap_lcc", "lcmap_luc"}
NMAX = {"africa_crop_mask": 200, "aster_ged": 200, "canada_crops_coarse": 75, "canada_crops_fine": 68, "descals": 200,
        "ethiopia_crops": 49, "glance": 300, "lcmap_lc": 300, "lcmap_lu": 300, "lcmap_lcc": 300, "lcmap_luc": 150,
        "lucas_lc": 300, "lucas_lu": 300, "openet_ensemble": 300, "us_trees": 300}
CLAY = ["clay_cls", "clay_mean", "clay_center"]
CHIP_ARMS = CLAY + ["dinov2", "dofa", "olmo_mean", "olmo_center", "prithvi_mean", "prithvi_center"]
POINT_ARMS = ["aef_mean128", "aef_ov32_px", "aef_pixel", "tessera_mean128", "tessera_pixel"]
COMMON_POINT = ["aef_mean128", "aef_ov32_px", "tessera_pixel"]
CENTRE_PAIRS = [("clay_center", "clay_mean"), ("clay_center", "clay_cls"), ("clay_cls", "clay_mean"),
                ("olmo_center", "olmo_mean"), ("prithvi_center", "prithvi_mean")]
BUDGETS = {"10": lambda t: 10, "max": lambda t: NMAX[t]}


def sha(items):
    return hashlib.sha256("\n".join(sorted(map(str, items))).encode()).hexdigest()


def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""): h.update(b)
    return h.hexdigest()


def bal_acc(y, p):
    y = np.asarray(y, int); p = np.asarray(p, int); C = max(y.max(), p.max()) + 1
    n = np.bincount(y, minlength=C); tp = np.bincount(y[y == p], minlength=C); m = n > 0
    return float((tp[m] / n[m]).mean())


def fit_predict(Xtr, ytr, Xte, reg, seed):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = ek.reg_head() if reg else ek.clf_head()
        if not reg: m.set_params(logisticregression__random_state=seed)
        m.fit(Xtr, ytr)
        return m.predict(Xte)


def load_chip_arms(a):
    keys = ["clay_cls", "clay_mean", "clay_center", "dino_cls", "dofa"]
    F = {k: [] for k in keys}; ids = []
    for f in sorted(glob.glob(os.path.join(a.e1_shards, "shard_*.npz"))):
        z = np.load(f); ids.extend(z["ids"].tolist())
        for k in keys: F[k].append(z[k].astype(np.float32))
    F = {k: np.concatenate(v, 0) for k, v in F.items()}; F["dinov2"] = F.pop("dino_cls")
    cidx = {str(c): i for i, c in enumerate(ids)}
    z = np.load(a.gpu_feats); gids = z["ids"].astype(str).tolist(); gidx = {c: i for i, c in enumerate(gids)}
    G = {k: z[k].astype(np.float32) for k in z.files if k != "ids"}
    return cidx, F, gidx, G


def load_point_arms(a, n):
    out = {}
    for path, names in [(a.aef_ov32, ["aef_mean128", "aef_ov32_px"]), (a.aef_pixel, ["aef_pixel"]),
                        (a.tessera_patch, ["tessera_mean128", "tessera_pixel"])]:
        if not os.path.exists(path): print("missing", path, flush=True); continue
        recs = [json.loads(l) for l in open(path)]
        for nm in names:
            rr = [(r["pid"], r[nm]) for r in recs if r.get(nm) is not None]
            if rr:
                A = np.full((n, len(rr[0][1])), np.nan, np.float32)
                A[[p for p, _ in rr]] = np.array([v for _, v in rr], np.float32); out[nm] = A
    return out


def run_set(task, T, feats, arms, full_tr, reg, pairs):
    """Score `arms` on the rows where all have features; paired block bootstrap for `pairs`."""
    ok = np.all([np.isfinite(feats[x]).all(1) for x in arms], 0)
    Tm = T[ok].reset_index(drop=True); fm = {x: feats[x][ok] for x in arms}
    tr = np.where(Tm.split == "train")[0]; te = np.where(Tm.split == "test")[0]
    info = {"arms": arms, "n_rows_eligible": int(len(T)), "n_rows_set": int(ok.sum()), "n_train_pool": int(len(tr)),
            "n_test": int(len(te)), "n_test_eligible": int((T.split == "test").sum()), "n_train_pool_eligible": int((T.split == "train").sum())}
    if len(tr) == 0 or len(te) == 0:
        info["skipped"] = "no train or no test rows"; return info
    y = Tm.label.values.astype(float if reg else int); strata = Tm.partition.values.astype(int) if reg else y.astype(int)
    ids = (task + "|" + Tm.row.astype(str)).values; yte = y[te]
    full_ids = (task + "|" + full_tr.row.astype(str)).values
    full_strata = (full_tr.partition.values if reg else full_tr.label.values).astype(int)
    blocks = ek.hblock(Tm.y.values[te], Tm.x.values[te]); ub = np.unique(blocks)
    if len(ub) == 0:
        info["skipped"] = "no test blocks"; return info
    rng = np.random.default_rng(1729); rows = {b: np.where(blocks == b)[0] for b in ub}
    try:
        RS = [np.concatenate([rows[b] for b in rng.choice(ub, len(ub), replace=True)]) for _ in range(1000)]
    except Exception as e:
        print("bootstrap draw failed", task, arms, "n_te", len(te), "n_blocks", len(ub), ub.dtype, repr(ub[:3]), repr(e), flush=True); raise
    metric = r2_score if reg else bal_acc
    avail = set(ids[tr]); sel, sel_digest, bud_info = {}, {}, {}
    for b, kf in BUDGETS.items():
        k = kf(task)
        nominal = [full_ids[ek.label_budget(full_ids, full_strata, k, sd)] for sd in ek.SEEDS]
        sel[b] = [tr[ek.label_budget(ids[tr], strata[tr], k, sd)] for sd in ek.SEEDS]
        sel_digest[b] = [sha(ids[s]) for s in sel[b]]
        bud_info[b] = {"k_per_class": k, "budget_complete": bool(all(set(n) <= avail for n in nominal)),
                       "n_train_nominal": [int(len(n)) for n in nominal], "n_train_achieved": [int(len(s)) for s in sel[b]],
                       "train_selected_sha256": sel_digest[b]}
    jobs = [(b, si, x) for b in BUDGETS for si in range(len(ek.SEEDS)) for x in arms]
    res = Parallel(n_jobs=int(os.environ.get("E5_NJOBS", "8")), backend="threading")(
        delayed(fit_predict)(fm[x][sel[b][si]], y[sel[b][si]], fm[x][te], reg, ek.SEEDS[si]) for b, si, x in jobs)
    P = {}
    for (b, si, x), p in zip(jobs, res): P.setdefault((b, x), {})[si] = p
    boot, arm_out = {}, {x: {} for x in arms}
    te_dig = sha(ids[te]); pool_dig = sha(ids[tr])
    for (b, x), d in P.items():
        preds = [d[si] for si in range(len(ek.SEEDS))]
        dist = np.array([np.mean([metric(yte[i], p[i]) for p in preds]) for i in RS]); boot[(b, x)] = dist
        sv = [float(metric(yte, p)) for p in preds]
        arm_out[x][b] = {"point": float(np.mean(sv)), "lo": float(np.percentile(dist, 2.5)), "hi": float(np.percentile(dist, 97.5)),
                         "seed_values": sv, "n_test": int(len(te)), "n_train_mean": float(np.mean(bud_info[b]["n_train_achieved"])),
                         "test_ids_sha256": te_dig, "train_pool_sha256": pool_dig, "train_selected_sha256": sel_digest[b]}
    contrasts = {}
    for lhs, rhs in pairs:
        if lhs not in arms or rhs not in arms: continue
        contrasts[f"{lhs}-{rhs}"] = {}
        for b in BUDGETS:
            dist = boot[(b, lhs)] - boot[(b, rhs)]
            sv = [m1 - m2 for m1, m2 in zip(arm_out[lhs][b]["seed_values"], arm_out[rhs][b]["seed_values"])]
            contrasts[f"{lhs}-{rhs}"][b] = {"point": float(np.mean(sv)), "lo": float(np.percentile(dist, 2.5)),
                                            "hi": float(np.percentile(dist, 97.5)), "n_test": int(len(te)),
                                            "budget_complete": bud_info[b]["budget_complete"]}
    info.update({"n_test_blocks": int(len(ub)), "metric": "r2" if reg else "balanced_accuracy", "test_ids_sha256": te_dig,
                 "train_pool_sha256": pool_dig, "test_blocks_sha256": sha(ub), "budgets": bud_info, "arm_results": arm_out,
                 "contrasts": contrasts})
    return info


def merge(parts, out_path):
    """Combine runs over disjoint task groups (long runs are split over a few processes).  Tasks are copied verbatim."""
    out = {"tasks": {}, "meta": {"merged_from": []}}
    for p in parts:
        d = json.load(open(p)); dup = set(d["tasks"]) & set(out["tasks"])
        assert not dup, f"task in two parts: {dup}"
        out["tasks"].update(d["tasks"])
        out["meta"]["merged_from"].append({"file": os.path.basename(p), "tasks": sorted(d["tasks"]), "meta": d.get("meta", "unfinished .partial file of the same script")})
        if "script" not in out["meta"] and "meta" in d: out["meta"].update({k: v for k, v in d["meta"].items() if k != "args"})
    out["tasks"] = {t: out["tasks"][t] for t in sorted(out["tasks"])}
    json.dump(out, open(out_path, "w"), indent=1)
    print("merged", len(out["tasks"]), "tasks into", out_path)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data_root", default=os.environ.get("E5_DATA_ROOT", "data/e5"))
    ap.add_argument("--e1_shards"); ap.add_argument("--gpu_feats"); ap.add_argument("--rows"); ap.add_argument("--points")
    ap.add_argument("--aef_ov32"); ap.add_argument("--aef_pixel"); ap.add_argument("--tessera_patch"); ap.add_argument("--label_years")
    ap.add_argument("--out", required=True); ap.add_argument("--tasks", default="")
    ap.add_argument("--merge", nargs="+", default=None, help="merge per-task-group result files (complete or .partial) into --out and exit")
    a = ap.parse_args()
    if a.merge: return merge(a.merge, a.out)
    d = a.data_root; j = lambda *p: os.path.join(d, *p)
    a.e1_shards = a.e1_shards or j("e1"); a.gpu_feats = a.gpu_feats or j("e5_gpu_feats.npz"); a.rows = a.rows or j("e1_rows.csv")
    a.points = a.points or j("points.csv"); a.aef_ov32 = a.aef_ov32 or j("aef_ov32.jsonl"); a.aef_pixel = a.aef_pixel or j("aef_pixel.jsonl")
    a.tessera_patch = a.tessera_patch or j("tessera_patch.jsonl"); a.label_years = a.label_years or j("label_years.csv")
    t_start = time.time()
    cidx, F1, gidx, FG = load_chip_arms(a)
    P = pd.read_csv(a.points); npt = int(P.pid.max()) + 1
    key = {(round(float(x), 7), round(float(y), 7), int(yr)): int(p) for p, x, y, yr in zip(P.pid, P.x, P.y, P.year)}
    FP = load_point_arms(a, npt)
    R = pd.read_csv(a.rows, keep_default_na=False, dtype={"chip": str, "chip_before": str, "chip_after": str})
    R["partition"] = pd.to_numeric(R.partition, errors="coerce"); R["row"] = R.row.astype(int); R["tier"] = pd.to_numeric(R.tier, errors="coerce")
    Y = pd.read_csv(a.label_years)[["task", "row", "year_before", "year_after", "source"]]
    R = R.merge(Y, on=["task", "row"], how="left")
    tasks = a.tasks.split(",") if a.tasks else sorted(R.task.unique())
    out = {"tasks": {}}
    for t in tasks:
        t0 = time.time(); reg, chg = t in REG, t in CHG
        T = R[(R.task == t) & ((R.split == "train") | (R.tier == 0))]
        full_tr = T[T.split == "train"]  # completeness is judged against all tier-filtered train rows (E1 rule)
        okc = (T.chip_before.isin(cidx) & T.chip_after.isin(cidx)) if chg else T.chip.isin(cidx)
        T = T[okc].reset_index(drop=True)
        feats = {}
        def chipfeat(Fd, ci, k):
            def g(cs): return np.stack([Fd[k][ci[c]] if c in ci else np.full(Fd[k].shape[1], np.nan, np.float32) for c in cs])
            return g(T.chip_after) - g(T.chip_before) if chg else g(T.chip)
        for k in F1: feats[k] = chipfeat(F1, cidx, k)
        for k in FG: feats[k] = chipfeat(FG, gidx, k)
        def pid(col): return np.array([key.get((round(float(x), 7), round(float(y), 7), int(yr)), -1) if pd.notna(yr) else -1
                                       for x, y, yr in zip(T.x, T.y, T[col])])
        pa, pb = pid("year_after"), pid("year_before")
        for k, A in FP.items():
            get = lambda p: np.where((p >= 0)[:, None], A[np.clip(p, 0, None)], np.nan)
            feats[k] = get(pa) - get(pb) if chg else get(pb)
        cov = {"n_rows_eligible_with_chip": int(len(T)), "n_train_pool_full": int(len(full_tr)),
               "n_rows_label_year_resolved": int(T.year_before.notna().sum() if not chg else (T.year_before.notna() & T.year_after.notna()).sum()),
               "arms": {x: {"n_valid": int(np.isfinite(feats[x]).all(1).sum()),
                            "n_valid_test": int((np.isfinite(feats[x]).all(1) & (T.split == "test").values).sum()),
                            "n_valid_train": int((np.isfinite(feats[x]).all(1) & (T.split == "train").values).sum())}
                        for x in feats}}
        sets = {}
        chip = [x for x in CHIP_ARMS if x in feats]
        pairs = [(c, o) for c in CLAY for o in chip if o not in CLAY] + CENTRE_PAIRS
        sets["chip"] = run_set(t, T, feats, chip, full_tr, reg, pairs)
        for X in [x for x in POINT_ARMS if x in feats]:
            sets[f"clay+{X}"] = run_set(t, T, feats, CLAY + [X], full_tr, reg, [(c, X) for c in CLAY])
        cp = [x for x in COMMON_POINT if x in feats]
        sets["common"] = run_set(t, T, feats, chip + cp, full_tr, reg,
                                 [(c, o) for c in CLAY for o in chip + cp if o not in CLAY] + CENTRE_PAIRS +
                                 [("olmo_center", x) for x in cp] + [("olmo_mean", x) for x in cp])
        out["tasks"][t] = {"metric": "r2" if reg else "balanced_accuracy", "coverage": cov, "sets": sets}
        print(t, f"{time.time()-t0:.0f}s", {s: v.get("n_test") for s, v in sets.items()}, flush=True)
        json.dump(out, open(a.out + ".partial", "w"))
    commit = os.environ.get("GIT_COMMIT") or ""
    if not commit:
        try: commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=HERE, text=True).strip()
        except Exception: commit = "unknown"
    out["meta"] = {"script": "experiments/suite/E5_sota_comparators/code/run_e1_e5_complete.py", "script_sha256": sha_file(os.path.abspath(__file__)),
                   "git_commit": commit, "args": {k: v for k, v in vars(a).items()},
                   "utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
                   "versions": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__, "sklearn": sklearn.__version__,
                                "scipy": scipy.__version__, "joblib": joblib.__version__},
                   "evalkit_sha256": sha_file(os.path.join(HERE, "..", "..", "common", "evalkit.py")),
                   "inputs_sha256": {os.path.basename(p): sha_file(p) for p in [a.gpu_feats, a.rows, a.points, a.aef_ov32, a.aef_pixel, a.tessera_patch, a.label_years]
                                     if os.path.exists(p)},
                   "label_year_sources": Y.source.value_counts().to_dict(), "seeds": list(ek.SEEDS), "bootstrap": {"n": 1000, "seed": 1729, "block_km": 100},
                   "budgets": {b: "10 per class" if b == "10" else "NMAX per task (per class)" for b in BUDGETS}, "nmax": NMAX,
                   "wall_s": round(time.time() - t_start, 1)}
    json.dump(out, open(a.out, "w"), indent=1)
    if os.path.exists(a.out + ".partial"): os.remove(a.out + ".partial")


if __name__ == "__main__":
    main()
