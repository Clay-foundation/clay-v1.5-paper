"""E1(a) evaluation: frozen features -> fixed heads at AEF budgets -> BA / R2 with 100 km block bootstrap.

Implements PROTOCOL.md exactly: evalkit.label_budget (seeds 0-4), evalkit.clf_head / reg_head (primary),
kNN k=min(3,n) on raw features (secondary), 1000 block resamples (evalkit.hblock 100 km, numpy
default_rng(1729), the same draw sequence as evalkit.block_bootstrap) shared by all arms of a task (paired).
Usage: python eval_aef.py --shards DIR --rows rows.csv --meta meta.csv --out results_aef.json
"""
import os
import argparse, glob, json, sys, warnings, time
import numpy as np, pandas as pd
from joblib import Parallel, delayed
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import balanced_accuracy_score, r2_score, f1_score
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "common"))
import evalkit as ek

ARMS = ["clay_cls", "clay_mean", "clay_center", "clay_cls_doc", "dinov2", "dofa", "stats", "coords"]
SRC = {"clay_cls": "clay_cls", "clay_mean": "clay_mean", "clay_center": "clay_center", "clay_cls_doc": "clay_cls_doc", "dinov2": "dino_cls",
       "dofa": "dofa", "stats": "stats"}
REG = {"aster_ged", "openet_ensemble"}
CHG = {"lcmap_lcc", "lcmap_luc"}
NMAX = {"africa_crop_mask": 200, "aster_ged": 200, "canada_crops_coarse": 75, "canada_crops_fine": 68, "descals": 200,
        "ethiopia_crops": 49, "glance": 300, "lcmap_lc": 300, "lcmap_lu": 300, "lcmap_lcc": 300, "lcmap_luc": 150,
        "lucas_lc": 300, "lucas_lu": 300, "openet_ensemble": 300, "us_trees": 300}


def load_features(shard_dir):
    F, ids = {k: [] for k in SRC.values()}, []
    for f in sorted(glob.glob(f"{shard_dir}/shard_*.npz")):
        z = np.load(f)
        ids.extend(z["ids"].tolist())
        for k in F:
            F[k].append(z[k].astype(np.float32))
    F = {k: np.concatenate(v) for k, v in F.items()}
    return {c: i for i, c in enumerate(ids)}, F


def coords(lat, lon):
    la, lo = np.radians(lat), np.radians(lon)
    return np.stack([np.sin(la), np.cos(la), np.sin(lo), np.cos(lo)], 1).astype(np.float32)


def bal_acc(y, p):
    """Equal to sklearn.balanced_accuracy_score (mean recall over classes present in y), via bincount."""
    y = np.asarray(y, int); p = np.asarray(p, int); C = max(y.max(), p.max()) + 1
    n = np.bincount(y, minlength=C); tp = np.bincount(y[y == p], minlength=C); m = n > 0
    return float((tp[m] / n[m]).mean())


def fit_predict(Xtr, ytr, Xte, reg, head, seed):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        if head == "linear":
            m = ek.reg_head() if reg else ek.clf_head()
            if not reg:
                m.set_params(logisticregression__random_state=seed)
        else:
            k = min(3, len(ytr))
            m = KNeighborsRegressor(k) if reg else KNeighborsClassifier(k)
        m.fit(Xtr, ytr)
        p = m.predict(Xte)
        nconv = sum("converge" in str(x.message).lower() for x in w)
    return p, nconv


def run_task(task, T, cidx, F, out_preds):
    reg, chg = task in REG, task in CHG
    if "tier" in T.columns:  # Amendment 2: test set = tier-0 test rows
        T = T[(T.split == "train") | (T.tier.astype(int) == 0)]
    Tall_tr = T[T.split == "train"]
    full_ids = (task + "|" + Tall_tr.row.astype(str)).values
    full_strata = (Tall_tr.partition.values if reg else Tall_tr.label.values).astype(int)
    if chg:
        ok = T.chip_before.isin(cidx) & T.chip_after.isin(cidx)
    else:
        ok = T.chip.isin(cidx)
    T = T[ok].reset_index(drop=True)
    tr = np.where(T.split == "train")[0]; te = np.where(T.split == "test")[0]
    if len(tr) == 0 or len(te) == 0:
        return [], [], {"task": task, "skipped": "no train or no test rows with chips"}, {"ids_test": np.array([]), "y_test": np.array([]), "blocks": np.array([])}
    feats = {}
    for arm in ARMS:
        if arm == "coords":
            if chg:
                continue
            feats[arm] = coords(T.y.values, T.x.values)
        elif chg:
            a = F[SRC[arm]][[cidx[c] for c in T.chip_after]]; b = F[SRC[arm]][[cidx[c] for c in T.chip_before]]
            feats[arm] = a - b
        else:
            feats[arm] = F[SRC[arm]][[cidx[c] for c in T.chip]]
    y = T.label.values.astype(float if reg else int)
    strata = T.partition.values.astype(int) if reg else y.astype(int)
    ids = (task + "|" + T.row.astype(str)).values
    yte = y[te]
    blocks = ek.hblock(T.y.values[te], T.x.values[te])
    # shared resamples (same RNG semantics as evalkit.block_bootstrap)
    rng = np.random.default_rng(1729); ub = np.unique(blocks); rows = {b: np.where(blocks == b)[0] for b in ub}
    RS = [np.concatenate([rows[b] for b in rng.choice(ub, len(ub), replace=True)]) for _ in range(1000)]
    metric = r2_score if reg else bal_acc
    budgets = {"1": 1, "10": 10, "max": NMAX[task]}
    # Amendment 2: a budget is run only if, for every seed, the selection on available rows equals the full-data one
    complete = {}
    avail = set(ids[tr])
    for bname, k in budgets.items():
        complete[bname] = all(set(full_ids[ek.label_budget(full_ids, full_strata, k, sd)]) <= avail for sd in ek.SEEDS)
    budgets = {b: k for b, k in budgets.items() if complete[b] or b != "max"}
    jobs = []
    for bname, k in budgets.items():
        for seed in ek.SEEDS:
            sel = tr[ek.label_budget(ids[tr], strata[tr], k, seed)]
            for arm in feats:
                for head in ["linear", "knn3"]:
                    jobs.append((bname, seed, arm, head, sel))
    res = Parallel(n_jobs=int(__import__("os").environ.get("E1_NJOBS", "1")), backend="threading")(delayed(fit_predict)(feats[a][s], y[s], feats[a][te], reg, h, sd)
                                              for b, sd, a, h, s in jobs)
    P = {}
    nconv = 0
    for (b, sd, a, h, s), (p, nc) in zip(jobs, res):
        P.setdefault((b, a, h), {})[sd] = (p, len(s)); nconv += nc
    summ, contr = [], []
    boot = {}
    for (b, a, h), d in P.items():
        preds = [d[sd][0] for sd in ek.SEEDS]
        point = float(np.mean([metric(yte, p) for p in preds]))
        dist = np.array([np.mean([metric(yte[i], p[i]) for p in preds]) for i in RS])
        boot[(b, a, h)] = dist
        rec = {"task": task, "budget": b, "arm": a, "head": h, "metric": "r2" if reg else "balanced_accuracy",
               "point": point, "lo": float(np.percentile(dist, 2.5)), "hi": float(np.percentile(dist, 97.5)),
               "seed_values": [float(metric(yte, p)) for p in preds], "n_train_mean": float(np.mean([d[sd][1] for sd in ek.SEEDS]))}
        if not reg:
            rec["macro_f1"] = float(np.mean([f1_score(yte, p, average="macro") for p in preds]))
        summ.append(rec)
        out_preds[f"{task}|{b}|{a}|{h}"] = np.stack(preds)
    for h in ["linear", "knn3"]:
        for b in budgets:
            for ref in ["stats", "dinov2", "dofa", "coords", "clay_mean", "clay_center", "clay_cls_doc"]:
                if (b, ref, h) not in boot:
                    continue
                pa = [P[(b, "clay_cls", h)][sd][0] for sd in ek.SEEDS]; pb = [P[(b, ref, h)][sd][0] for sd in ek.SEEDS]
                point = float(np.mean([metric(yte, p) for p in pa]) - np.mean([metric(yte, p) for p in pb]))
                dist = boot[(b, "clay_cls", h)] - boot[(b, ref, h)]
                contr.append({"task": task, "budget": b, "head": h, "contrast": f"clay_cls-{ref}", "point": point,
                              "lo": float(np.percentile(dist, 2.5)), "hi": float(np.percentile(dist, 97.5))})
    info = {"task": task, "n_rows_requested": int(len(ok)), "n_rows_with_chip": int(ok.sum()), "n_train": int(len(tr)),
            "n_test": int(len(te)), "n_test_blocks": int(len(ub)), "n_classes": int(len(np.unique(strata))),
            "convergence_warnings": int(nconv), "budget_complete": complete,
            "n_train_rows_full": int(len(Tall_tr)), "n_train_rows_missing_chip": int(len(Tall_tr) - len(tr))}
    return summ, contr, info, {"ids_test": ids[te], "y_test": yte, "blocks": blocks}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", required=True); ap.add_argument("--rows", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--tasks", default="")
    a = ap.parse_args()
    cidx, F = load_features(a.shards)
    R = pd.read_csv(a.rows, keep_default_na=False, dtype={"chip": str, "chip_before": str, "chip_after": str})
    R["partition"] = pd.to_numeric(R["partition"], errors="coerce")
    R["row"] = R["row"].astype(int)
    tasks = a.tasks.split(",") if a.tasks else sorted(R.task.unique())
    S, C, I, preds = [], [], [], {}
    for t in tasks:
        t0 = time.time()
        s, c, i, tinfo = run_task(t, R[R.task == t], cidx, F, preds)
        S += s; C += c; I.append(i)
        preds[f"{t}|ids_test"] = tinfo["ids_test"]; preds[f"{t}|y_test"] = tinfo["y_test"]; preds[f"{t}|blocks"] = tinfo["blocks"]
        print(t, i, f"{time.time()-t0:.0f}s", flush=True)
    json.dump({"summaries": S, "contrasts": C, "tasks": I}, open(a.out, "w"), indent=1)
    np.savez_compressed(a.out.replace(".json", "_predictions.npz"), **preds)


if __name__ == "__main__":
    main()
