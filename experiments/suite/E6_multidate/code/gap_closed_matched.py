"""E6 follow-up (issue #8): "gap closed by four dates" with ONE fixed readout, a NAMED comparator on MATCHED rows, one pooling
rule per column, explicit budgets, task set and eligibility rule.

Why a rerun: E6's published median (0.12) divides by transcribed AEF numbers, takes the better of two pooling rules per
task (mean vs concat) and mixes target types.  The matched note (RESULTS.md) divided by E5's AEF values computed on different rows.
Here every quantity in a (task, comparator, budget) cell comes from ONE row set and ONE set of training draws:

  rows      = E1 rows (all train rows with an E1 chip + tier-0 test rows), E6 fetch attempted for every chip, comparator feature present
              at the row's exact label year (label_years.csv, build_label_years.py); no nearest-year fallback.
  arms      = clay1 (E1 CLS, 1 date), clay4_mean (mean of 4 quarterly CLS), clay4_cat (4-slot concat of CLS)  [readout CLS, fixed];
              claymean1 / claymean4_mean (mean-patch readout, secondary);  comparator X in {aef_mean128, aef_ov32_px}.
  budgets   = 10 per class and max (NMAX per class); max only if E1's completeness rule holds on the matched set.
  heads     = linear (LR C=1 / RidgeCV), 5 label seeds, 1,000 shared 100 km block-bootstrap resamples (seed 1729): paired intervals.
  gap       = X - clay1 (point estimate on the matched rows).  A task is ELIGIBLE for a ratio only if gap >= --min_gap (default 0.05).
  closed    = (pooled 4-date arm - clay1) / (X - clay1).  Reported separately for each pooling rule; never best-of.  The interval is the
              paired-resample percentile of the ratio, given only when the gap is positive in all resamples.
  absolute  = pooled 4-date arm - clay1 with its paired interval, for every task (also the ineligible ones).

Usage (repo root):
  python experiments/suite/E6_multidate/code/gap_closed_matched.py --data_root DATA --out OUT.json
  DATA/ = e1/shard_*.npz  e6/shard_*.npz  e6/meta.csv  e1_rows.csv  points.csv  aef_ov32.jsonl  label_years.csv
Environment: E6_NJOBS (parallel fits, default 8), GIT_COMMIT.
"""
import argparse, datetime, glob, hashlib, json, os, platform, subprocess, sys, time
import numpy as np, pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import r2_score
import sklearn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "common"))
sys.path.insert(0, os.path.join(HERE, "..", "..", "E1_label_efficiency", "code"))
import evalkit as ek
from eval_aef import fit_predict, bal_acc, NMAX, REG, CHG
from eval_e6 import load, build_chip_feats

COMPARATORS = ["aef_mean128", "aef_ov32_px"]
ARMS = ["clay1", "clay4_mean", "clay4_cat", "claymean1", "claymean4_mean"]
POOL = {"mean": ("clay4_mean", "clay1"), "concat": ("clay4_cat", "clay1"), "mean_patchreadout": ("claymean4_mean", "claymean1")}
TIER_B_MAX = {"africa_crop_mask", "canada_crops_fine", "descals", "ethiopia_crops", "glance", "lcmap_lc", "lcmap_lu", "lcmap_lcc", "lcmap_luc", "openet_ensemble"}
BUDGETS = {"10": lambda t: 10, "max": lambda t: NMAX[t]}


def sha(items):
    return hashlib.sha256("\n".join(sorted(map(str, items))).encode()).hexdigest()


def run_cell(task, T, feats, X, full_tr, reg, njobs):
    arms = ARMS + [X]
    ok = np.all([np.isfinite(feats[a]).all(1) for a in arms], 0)
    Tm = T[ok].reset_index(drop=True); fm = {a: feats[a][ok] for a in arms}
    tr = np.where(Tm.split == "train")[0]; te = np.where(Tm.split == "test")[0]
    info = {"comparator": X, "n_rows_eligible": int(len(T)), "n_rows_set": int(ok.sum()), "n_train_pool": int(len(tr)), "n_test": int(len(te))}
    if len(tr) == 0 or len(te) == 0:
        info["skipped"] = "no train or no test rows"; return info
    y = Tm.label.values.astype(float if reg else int); strata = Tm.partition.values.astype(int) if reg else y.astype(int)
    ids = (task + "|" + Tm.row.astype(str)).values; yte = y[te]
    full_ids = (task + "|" + full_tr.row.astype(str)).values
    full_strata = (full_tr.partition.values if reg else full_tr.label.values).astype(int)
    blocks = ek.hblock(Tm.y.values[te], Tm.x.values[te]); ub = np.unique(blocks)
    rng = np.random.default_rng(1729); rows = {b: np.where(blocks == b)[0] for b in ub}
    RS = [np.concatenate([rows[b] for b in rng.choice(ub, len(ub), replace=True)]) for _ in range(1000)]
    metric = r2_score if reg else bal_acc
    avail = set(ids[tr]); res_b = {}
    for b, kf in BUDGETS.items():
        k = kf(task)
        nominal = [full_ids[ek.label_budget(full_ids, full_strata, k, sd)] for sd in ek.SEEDS]
        complete = bool(all(set(n) <= avail for n in nominal))
        if b == "max" and (not complete or task not in TIER_B_MAX):
            res_b[b] = {"run": False, "budget_complete": complete, "reason": "max budget incomplete on matched rows or not fetched by E6"}; continue
        sel = [tr[ek.label_budget(ids[tr], strata[tr], k, sd)] for sd in ek.SEEDS]
        jobs = [(si, a) for si in range(len(ek.SEEDS)) for a in arms]
        res = Parallel(n_jobs=njobs, backend="threading")(delayed(fit_predict)(fm[a][sel[si]], y[sel[si]], fm[a][te], reg, "linear", ek.SEEDS[si]) for si, a in jobs)
        P = {}
        for (si, a), (p, _) in zip(jobs, res): P.setdefault(a, {})[si] = p
        dist, sv = {}, {}
        for a in arms:
            preds = [P[a][si] for si in range(len(ek.SEEDS))]
            sv[a] = [float(metric(yte, p)) for p in preds]
            dist[a] = np.array([np.mean([metric(yte[i], p[i]) for p in preds]) for i in RS])
        cell = {"run": True, "budget_complete": complete, "n_train_achieved": [int(len(s)) for s in sel], "test_ids_sha256": sha(ids[te]),
                "train_selected_sha256": [sha(ids[s]) for s in sel],
                "arms": {a: {"point": float(np.mean(sv[a])), "lo": float(np.percentile(dist[a], 2.5)), "hi": float(np.percentile(dist[a], 97.5)),
                             "seed_values": sv[a]} for a in arms}}
        gap_pt = cell["arms"][X]["point"] - cell["arms"]["clay1"]["point"]
        gap_d = dist[X] - dist["clay1"]
        cell["gap"] = {"point": float(gap_pt), "lo": float(np.percentile(gap_d, 2.5)), "hi": float(np.percentile(gap_d, 97.5))}
        cell["pooling"] = {}
        for name, (a4, a1) in POOL.items():
            dd = dist[a4] - dist[a1]; pt = cell["arms"][a4]["point"] - cell["arms"][a1]["point"]
            rec = {"abs_change": float(pt), "lo": float(np.percentile(dd, 2.5)), "hi": float(np.percentile(dd, 97.5))}
            if name != "mean_patchreadout":
                rec["gap_closed"] = float(pt / gap_pt) if gap_pt != 0 else None
                if (gap_d > 0).all():
                    q = dd / gap_d; rec["gap_closed_lo"] = float(np.percentile(q, 2.5)); rec["gap_closed_hi"] = float(np.percentile(q, 97.5))
            cell["pooling"][name] = rec
        res_b[b] = cell
    info["budgets"] = res_b; info["n_test_blocks"] = int(len(ub))
    return info


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data_root", default=os.environ.get("E5_DATA_ROOT", "data/e5"))
    ap.add_argument("--e6_dir"); ap.add_argument("--min_gap", type=float, default=0.05)
    ap.add_argument("--out", required=True); ap.add_argument("--tasks", default="")
    a = ap.parse_args(); d = a.data_root; a.e6_dir = a.e6_dir or os.path.join(d, "e6")
    t_start = time.time()
    e1idx, F1 = load(os.path.join(d, "e1"), ["clay_cls", "clay_mean", "stats", "dofa"])
    e6idx, F6 = load(a.e6_dir, ["clay_cls", "clay_mean", "stats", "dofa"])
    meta = pd.read_csv(os.path.join(a.e6_dir, "meta.csv"), dtype={"chip_id": str})
    # E1's own quarter for a chip = the one quarter id that is absent from E6's fetch list (E6 fetched the other three)
    tried = meta[meta.q >= 0].groupby("chip_id").q.apply(set)
    e1q = {c: next(iter({0, 1, 2, 3} - s)) for c, s in tried.items() if len(s) == 3}
    attempted = set(meta.chip_id[~meta.status.astype(str).str.startswith("error")]) | set(meta.chip_id)  # --errors-as-missing semantics
    P = pd.read_csv(os.path.join(d, "points.csv")); n = int(P.pid.max()) + 1
    key = {(round(float(x), 7), round(float(y), 7), int(yr)): int(p) for p, x, y, yr in zip(P.pid, P.x, P.y, P.year)}
    FP = {}
    recs = [json.loads(l) for l in open(os.path.join(d, "aef_ov32.jsonl"))]
    for nm in COMPARATORS:
        rr = [(r["pid"], r[nm]) for r in recs if r.get(nm) is not None]
        A = np.full((n, len(rr[0][1])), np.nan, np.float32); A[[p for p, _ in rr]] = np.array([v for _, v in rr], np.float32); FP[nm] = A
    R = pd.read_csv(os.path.join(d, "e1_rows.csv"), keep_default_na=False, dtype={"chip": str, "chip_before": str, "chip_after": str})
    R["partition"] = pd.to_numeric(R.partition, errors="coerce"); R["row"] = R.row.astype(int); R["tier"] = pd.to_numeric(R.tier, errors="coerce")
    Y = pd.read_csv(os.path.join(d, "label_years.csv"))[["task", "row", "year_before", "year_after"]]
    R = R.merge(Y, on=["task", "row"], how="left")
    tasks = a.tasks.split(",") if a.tasks else sorted(R.task.unique())
    out = {"tasks": {}}
    for t in tasks:
        t0 = time.time(); reg, chg = t in REG, t in CHG
        T = R[(R.task == t) & ((R.split == "train") | (R.tier == 0))]
        full_tr = T[T.split == "train"]
        okc = (T.chip_before.isin(e1idx) & T.chip_after.isin(e1idx)) if chg else T.chip.isin(e1idx)
        T = T[okc]
        att = (T.chip_before.isin(attempted) & T.chip_after.isin(attempted)) if chg else T.chip.isin(attempted)
        T = T[att.values].reset_index(drop=True)
        args = (e1idx, F1, e6idx, F6, e1q)
        if chg:
            fa, _ = build_chip_feats(T.chip_after.tolist(), *args); fb, _ = build_chip_feats(T.chip_before.tolist(), *args)
            feats = {k: fa[k] - fb[k] for k in fa}
        else:
            feats, _ = build_chip_feats(T.chip.tolist(), *args)
        feats = {k: feats[k] for k in ARMS}
        def pid(col): return np.array([key.get((round(float(x), 7), round(float(y), 7), int(yr)), -1) if pd.notna(yr) else -1 for x, y, yr in zip(T.x, T.y, T[col])])
        pa, pb = pid("year_after"), pid("year_before")
        for k, A in FP.items():
            get = lambda p: np.where((p >= 0)[:, None], A[np.clip(p, 0, None)], np.nan)
            feats[k] = get(pa) - get(pb) if chg else get(pb)
        out["tasks"][t] = {"metric": "r2" if reg else "balanced_accuracy",
                           "cells": {X: run_cell(t, T, feats, X, full_tr, reg, int(os.environ.get("E6_NJOBS", "8"))) for X in COMPARATORS}}
        print(t, f"{time.time()-t0:.0f}s", {X: c.get("n_test") for X, c in out["tasks"][t]["cells"].items()}, flush=True)
    commit = os.environ.get("GIT_COMMIT") or "unknown"
    out["meta"] = {"script": "experiments/suite/E6_multidate/code/gap_closed_matched.py", "git_commit": commit, "args": vars(a),
                   "utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
                   "versions": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__, "sklearn": sklearn.__version__},
                   "min_gap": a.min_gap, "seeds": list(ek.SEEDS), "bootstrap": {"n": 1000, "seed": 1729, "block_km": 100},
                   "n_chips_with_e1_quarter": len(e1q), "wall_s": round(time.time() - t_start, 1)}
    json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
