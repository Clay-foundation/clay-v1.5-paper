"""E6 evaluation: E1 single-date arms vs 4-season arms, same rows/splits/budgets/heads/bootstrap as E1 (eval_aef.py).
Usage: python eval_e6.py --e1shards DIR --e6shards DIR --e6meta meta.csv --rows rows.csv --req requests_e6.csv --out results_e6.json
"""
import argparse, glob, json, os, sys, time, warnings
import numpy as np, pandas as pd
from joblib import Parallel, delayed
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "common"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "E1_label_efficiency", "code"))
import evalkit as ek
from eval_aef import fit_predict, bal_acc, NMAX, REG, CHG
from sklearn.metrics import r2_score, f1_score

ARMS = ["clay1", "clay4_mean", "clay4_cat", "stats1", "stats4_cat", "dofa1", "dofa4_mean", "claymean1", "claymean4_mean"]
CONTRASTS = [("clay4_mean", "clay1"), ("clay4_cat", "clay1"), ("clay4_cat", "stats4_cat"), ("clay4_mean", "dofa4_mean"),
             ("clay4_cat", "clay4_mean"), ("stats4_cat", "stats1"), ("dofa4_mean", "dofa1"), ("claymean4_mean", "claymean1"),
             ("clay1", "stats1"), ("clay4_mean", "stats4_cat")]


def load(shard_dir, keys):
    F, ids = {k: [] for k in keys}, []
    for f in sorted(glob.glob(f"{shard_dir}/shard_*.npz")):
        z = np.load(f); ids.extend(z["ids"].tolist())
        for k in keys:
            F[k].append(z[k].astype(np.float32))
    return {c: i for i, c in enumerate(ids)}, {k: np.concatenate(v) for k, v in F.items()}


def build_chip_feats(chips, e1idx, F1, e6idx, F6, e1q):
    """Per chip: 4-season stacks. Returns dict arm -> array, and n_seasons_available."""
    n = len(chips); out = {}; navail = np.zeros(n, int)
    pairs = {"clay": ("clay_cls", "clay_cls"), "stats": ("stats", "stats"), "dofa": ("dofa", "dofa"), "claymean": ("clay_mean", "clay_mean")}
    i1 = np.array([e1idx[c] for c in chips])
    for name, (k1, k6) in pairs.items():
        base = F1[k1][i1]; d = base.shape[1]
        S = np.repeat(base[:, None, :], 4, 1).copy(); avail = np.zeros((n, 4), bool)
        for j, c in enumerate(chips):
            q1 = e1q.get(c)  # None: chip outside E6's request set (never selected by a budget that is run)
            if q1 is None:
                avail[j, 0] = True; continue
            avail[j, q1] = True
            for q in range(4):
                if q == q1:
                    continue
                r = e6idx.get(f"{c}|{q}")
                if r is not None:
                    S[j, q] = F6[k6][r]; avail[j, q] = True
        navail = avail.sum(1)
        out[name + "1"] = base
        out[name + "4_mean"] = (S * avail[:, :, None]).sum(1) / avail.sum(1, keepdims=True)
        out[name + "4_cat"] = S.reshape(n, 4 * d)  # missing slots hold the E1 vector
    return out, navail


def run_task(task, T, ctx, out_preds, attempted):
    reg, chg = task in REG, task in CHG
    T = T[(T.split == "train") | (T.tier.astype(int) == 0)]
    e1idx = ctx["e1idx"]
    ok = (T.chip_before.isin(e1idx) & T.chip_after.isin(e1idx)) if chg else T.chip.isin(e1idx)
    T = T[ok].reset_index(drop=True)
    att = (T.chip_before.isin(attempted) & T.chip_after.isin(attempted)) if chg else T.chip.isin(attempted)
    tr = np.where(T.split == "train")[0]; te = np.where(T.split == "test")[0]
    y = T.label.values.astype(float if reg else int)
    strata = T.partition.values.astype(int) if reg else y.astype(int)
    ids = (task + "|" + T.row.astype(str)).values
    if chg:
        fa, na = build_chip_feats(T.chip_after.tolist(), *ctx["args"]); fb, nb = build_chip_feats(T.chip_before.tolist(), *ctx["args"])
        feats = {k: fa[k] - fb[k] for k in fa}; nav = np.minimum(na, nb)
    else:
        feats, nav = build_chip_feats(T.chip.tolist(), *ctx["args"])
    feats = {a: feats[{"clay1": "clay1", "claymean1": "claymean1"}.get(a, a)] for a in ARMS}
    yte = y[te]; blocks = ek.hblock(T.y.values[te], T.x.values[te])
    rng = np.random.default_rng(1729); ub = np.unique(blocks); rows = {b: np.where(blocks == b)[0] for b in ub}
    RS = [np.concatenate([rows[b] for b in rng.choice(ub, len(ub), replace=True)]) for _ in range(1000)]
    metric = r2_score if reg else bal_acc
    budgets = {"1": 1, "10": 10, "max": NMAX[task]}
    ran, coverage = {}, {}
    for bname, k in budgets.items():
        sel_all = set()
        for sd in ek.SEEDS:
            sel_all |= set(tr[ek.label_budget(ids[tr], strata[tr], k, sd)])
        used = np.array(sorted(sel_all | set(te)))
        coverage[bname] = float(att[used].mean())
        ran[bname] = bool(att[used].all()) and (bname != "max" or ctx["e1_max_ok"].get(task, False))
    jobs = []
    for bname, k in budgets.items():
        if not ran[bname]:
            continue
        for seed in ek.SEEDS:
            sel = tr[ek.label_budget(ids[tr], strata[tr], k, seed)]
            for arm in ARMS:
                for head in ["linear", "knn3"]:
                    jobs.append((bname, seed, arm, head, sel))
    res = Parallel(n_jobs=int(os.environ.get("E6_NJOBS", "8")), backend=os.environ.get("E6_BACKEND", "loky"))(
        delayed(fit_predict)(feats[a][s], y[s], feats[a][te], reg, h, sd) for b, sd, a, h, s in jobs)
    P = {}; nconv = 0
    for (b, sd, a, h, s), (p, nc) in zip(jobs, res):
        P.setdefault((b, a, h), {})[sd] = (p, len(s)); nconv += nc
    summ, contr, boot = [], [], {}
    for (b, a, h), d in P.items():
        preds = [d[sd][0] for sd in ek.SEEDS]
        point = float(np.mean([metric(yte, p) for p in preds]))
        dist = np.array([np.mean([metric(yte[i], p[i]) for p in preds]) for i in RS]); boot[(b, a, h)] = dist
        rec = {"task": task, "budget": b, "arm": a, "head": h, "metric": "r2" if reg else "balanced_accuracy", "point": point,
               "lo": float(np.percentile(dist, 2.5)), "hi": float(np.percentile(dist, 97.5)),
               "seed_values": [float(metric(yte, p)) for p in preds], "n_train_mean": float(np.mean([d[sd][1] for sd in ek.SEEDS]))}
        if not reg:
            rec["macro_f1"] = float(np.mean([f1_score(yte, p, average="macro") for p in preds]))
        summ.append(rec); out_preds[f"{task}|{b}|{a}|{h}"] = np.stack(preds)
    for h in ["linear", "knn3"]:
        for b in budgets:
            for A, B in CONTRASTS:
                if (b, A, h) not in boot:
                    continue
                pa = [P[(b, A, h)][sd][0] for sd in ek.SEEDS]; pb = [P[(b, B, h)][sd][0] for sd in ek.SEEDS]
                point = float(np.mean([metric(yte, p) for p in pa]) - np.mean([metric(yte, p) for p in pb]))
                dist = boot[(b, A, h)] - boot[(b, B, h)]
                contr.append({"task": task, "budget": b, "head": h, "contrast": f"{A}-{B}", "point": point,
                              "lo": float(np.percentile(dist, 2.5)), "hi": float(np.percentile(dist, 97.5))})
    info = {"task": task, "n_train": int(len(tr)), "n_test": int(len(te)), "n_test_blocks": int(len(ub)),
            "budget_run": ran, "e6_attempt_coverage": coverage, "convergence_warnings": int(nconv),
            "frac_rows_all4_test": float((nav[te] == 4).mean()), "frac_rows_all4_train": float((nav[tr] == 4).mean()) if len(tr) else None,
            "mean_seasons_test": float(nav[te].mean())}
    return summ, contr, info


def main():
    ap = argparse.ArgumentParser()
    for k in ["e1shards", "e6shards", "e6meta", "rows", "req", "out", "e1results"]:
        ap.add_argument("--" + k, required=True)
    ap.add_argument("--tasks", default="")
    ap.add_argument("--errors-as-missing", action="store_true", help="after the retry pass: chips whose fetch errored count as attempted, seasons missing")
    a = ap.parse_args()
    keys = ["clay_cls", "clay_mean", "stats", "dofa"]
    e1idx, F1 = load(a.e1shards, keys); e6idx, F6 = load(a.e6shards, keys)
    req = pd.read_csv(a.req, dtype={"chip_id": str}); e1q = dict(zip(req.chip_id, req.e1_q.astype(int)))
    meta = pd.read_csv(a.e6meta, dtype={"chip_id": str})
    attempted = set(meta.chip_id[~meta.status.astype(str).str.startswith("error")])
    if a.errors_as_missing:
        attempted |= set(meta.chip_id)
    E1 = json.load(open(a.e1results)); e1_max_ok = {t["task"]: t["budget_complete"]["max"] for t in E1["tasks"]}
    R = pd.read_csv(a.rows, keep_default_na=False, dtype={"chip": str, "chip_before": str, "chip_after": str})
    R["partition"] = pd.to_numeric(R["partition"], errors="coerce"); R["row"] = R["row"].astype(int)
    ctx = {"e1idx": e1idx, "args": (e1idx, F1, e6idx, F6, e1q), "e1_max_ok": e1_max_ok}
    tasks = a.tasks.split(",") if a.tasks else sorted(R.task.unique())
    S, C, I, preds = [], [], [], {}
    for t in tasks:
        t0 = time.time(); s, cc, i = run_task(t, R[R.task == t], ctx, preds, attempted)
        S += s; C += cc; I.append(i); print(t, i, f"{time.time()-t0:.0f}s", flush=True)
    json.dump({"summaries": S, "contrasts": C, "tasks": I}, open(a.out, "w"), indent=1)
    np.savez_compressed(a.out.replace(".json", "_predictions.npz"), **preds)


if __name__ == "__main__":
    main()
