"""E2_tokens probes (CPU). Usage: python probe.py EMBDIR OUT.json [--quick]"""
import hashlib, json, sys, time
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, "common")
from evalkit import clf_head, reg_head, SEEDS
from claymodel.utils import posemb_sincos_2d_with_gsd, posemb_sincos_2d
from sklearn.metrics import r2_score, f1_score, accuracy_score
from joblib import Parallel, delayed
from threadpoolctl import threadpool_limits
def _fitpred(Xtr, ytr, Xte):
    with threadpool_limits(6):
        return clf_head().fit(Xtr, ytr).predict(Xte)
def fitpred_seeds(jobs):
    """jobs: list of (Xtr, ytr, Xte) -> list of predictions, run in parallel processes"""
    return Parallel(n_jobs=min(len(jobs), 5))(delayed(_fitpred)(*j) for j in jobs)

EMB = Path(sys.argv[1]); OUTF = Path(sys.argv[2]); QUICK = "--quick" in sys.argv
NB = 50 if QUICK else 1000
SEEDS = SEEDS[:1] if QUICK else SEEDS
GSD = {"s2": 10.0, "s1": 10.0, "ls": 30.0, "naip": 1.0}
WC = [10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100]
R = {"meta": {"n_boot": NB, "seeds": list(SEEDS)}}
t00 = time.time()
def say(*a): print(round(time.time() - t00), *a, flush=True)

def boot_idx(blocks, nb=NB, seed=1729):
    rng = np.random.default_rng(seed); ub = np.unique(blocks)
    return ub, [rng.choice(len(ub), len(ub), replace=True) for _ in range(nb)]

# ---------- metric helpers using per-block sufficient statistics (exact block bootstrap) ----------
def conf_by_block(y, p, tblk, ub, labels):
    L = len(labels); li = {c: i for i, c in enumerate(labels)}
    yi = np.array([li.get(v, -1) for v in y]); pi = np.array([li.get(v, -1) for v in p])
    bi = np.searchsorted(ub, tblk)
    ok = yi >= 0
    pi = np.where(pi < 0, L, pi)  # predicted class outside label set -> extra column
    M = np.zeros((len(ub), L, L + 1)); np.add.at(M, (bi[ok], yi[ok], pi[ok]), 1)
    return M
def f1_miou(C):
    tp = np.diagonal(C[:, :C.shape[0]]) if C.ndim == 2 else None
    L = C.shape[0]; tp = np.array([C[i, i] for i in range(L)]); fn = C.sum(1) - tp; fp = C[:, :L].sum(0) - tp
    present = (tp + fn) > 0
    f1 = np.where(present, 2 * tp / np.maximum(2 * tp + fp + fn, 1), np.nan); iou = np.where(present, tp / np.maximum(tp + fp + fn, 1), np.nan)
    return float(np.nanmean(f1)), float(np.nanmean(iou)), float(tp.sum() / max(C.sum(), 1))
def seg_eval(Ms, ub, bidx):
    """Ms: list over seeds of per-block confusion [nb,L,L+1]. Returns point+CI for f1/miou/acc (seed-mean)."""
    pt = np.mean([f1_miou(M.sum(0)) for M in Ms], 0)
    bs = np.array([np.mean([f1_miou(M[b].sum(0)) for M in Ms], 0) for b in bidx])
    out = {}
    for k, n in enumerate(["macro_f1", "miou", "acc"]):
        out[n] = [float(pt[k]), float(np.percentile(bs[:, k], 2.5)), float(np.percentile(bs[:, k], 97.5))]
    return out, bs
def paired(bsA, bsB, ptA, ptB, k=0):
    d = bsA[:, k] - bsB[:, k]; return [float(ptA - ptB), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]

def r2_by_block(y, p, tblk, ub):
    """sufficient stats per block for R2: n, sum y, sum y2, sum (y-p)^2"""
    bi = np.searchsorted(ub, tblk); S = np.zeros((len(ub), 4))
    for j, v in enumerate([np.ones_like(y), y, y * y, (y - p) ** 2]): np.add.at(S[:, j], bi, v)
    return S
def r2_from(S):
    n, sy, syy, sse = S.sum(0); return float(1 - sse / max(syy - sy * sy / n, 1e-12))
def r2_eval(Ss, ub, bidx):
    pt = np.mean([r2_from(S) for S in Ss]); bs = np.array([np.mean([r2_from(S[b]) for S in Ss]) for b in bidx])
    return [float(pt), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))], bs

def seed_rows(pids_tok, chip_idx, pids, s):
    """for train tokens: the sha256(seed)-ranked 16 of the 64 tokens per chip"""
    key = np.array([int(hashlib.sha256(f"{s}|{pids[c]}|{q}".encode()).hexdigest()[:12], 16) for c, q in zip(chip_idx, pids_tok)])
    order = np.lexsort((key, chip_idx)); rank = np.empty(len(order), int)
    # rank within chip
    ci = chip_idx[order]; starts = np.r_[0, np.flatnonzero(np.diff(ci)) + 1]
    r = np.arange(len(order)) - np.repeat(starts, np.diff(np.r_[starts, len(order)]))
    rank[order] = r
    return np.flatnonzero(rank < 16)

def gcd_km(la1, lo1, la2, lo2):
    la1, lo1, la2, lo2 = map(np.radians, (la1, lo1, la2, lo2))
    a = np.sin((la2 - la1) / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2
    return 6371 * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))

def code(gsd, std=False):
    if std: return posemb_sincos_2d(32, 32, 1016).numpy()
    return posemb_sincos_2d_with_gsd(32, 32, 1016, gsd=torch.tensor(gsd)).numpy()

E = {s: dict(np.load(EMB / f"emb_{s}.npz", allow_pickle=True)) for s in ["s2", "s1", "ls", "naip"] if (EMB / f"emb_{s}.npz").exists()}
R["n_chips"] = {s: {"total": int(len(e["pids"])), "test": int((e["split"] == "test").sum()),
                    "test_blocks": int(len(np.unique(e["blocks"][e["split"] == "test"])))} for s, e in E.items()}
say("loaded", R["n_chips"])

# ---------------- (i) position probe + (iii) dense probe per sensor ----------------
R["position"], R["dense"], R["dense_contrasts"] = {}, {}, {}
for s, e in E.items():
    test_chip = e["split"] == "test"
    ch = e["tok_chip"]; tr = ~test_chip[ch]; te = test_chip[ch]
    pos = e["tok_pos"].astype(int); row, col = pos // 32, pos % 32
    tblk = e["blocks"][ch[te]]; ub, bidx = boot_idx(e["blocks"][test_chip])
    feats = {"patches": e["tok_patch"], "bandstats": e["tok_bstat"]}
    if "tok_dino" in e: feats["dinov2_patches"] = e["tok_dino"]
    # position
    P = {}
    cods = {"code_implemented": code(GSD[s]), "code_standard": code(GSD[s], std=True)}
    for name, X in list(feats.items()) + [(k, v[pos]) for k, v in cods.items()]:
        Ss = []
        res = {}
        for tgt, y in [("row", row), ("col", col)]:
            m = reg_head().fit(X[tr].astype(np.float32), y[tr]); p = m.predict(X[te].astype(np.float32))
            S = r2_by_block(y[te].astype(float), p, tblk, ub); r, bs = r2_eval([S], ub, bidx); res[tgt] = r; Ss.append(bs)
        bsm = (Ss[0] + Ss[1]) / 2; ptm = (res["row"][0] + res["col"][0]) / 2
        res["mean"] = [ptm, float(np.percentile(bsm, 2.5)), float(np.percentile(bsm, 97.5))]; res["_bs"] = bsm
        P[name] = res
    for name in cods:  # checkerboard: positions with (r+c) odd -> fit, even -> test
        X = cods[name]; rr, cc = np.arange(1024) // 32, np.arange(1024) % 32; odd = (rr + cc) % 2 == 1
        P[name]["checkerboard_r2"] = {t: float(r2_score(y[~odd], reg_head().fit(X[odd], y[odd]).predict(X[~odd]))) for t, y in [("row", rr), ("col", cc)]}
    P["contrast_patches_minus_dinov2"] = paired(P["patches"]["_bs"][:, None], P["dinov2_patches"]["_bs"][:, None], P["patches"]["mean"][0], P["dinov2_patches"]["mean"][0]) if "dinov2_patches" in P else None
    for v in P.values():
        if isinstance(v, dict): v.pop("_bs", None)
    R["position"][s] = P; say("position", s, {k: v["mean"][0] for k, v in P.items() if isinstance(v, dict)})
    # dense
    y_all = e["tok_label"].astype(int); valid = y_all > 0
    labels = [c for c in WC if (y_all[te & valid] == c).any()]
    feats_d = dict(feats); feats_d["cls_broadcast"] = e["cls"][ch]; feats_d["mean_broadcast"] = e["mean"][ch]
    D, BS = {}, {}
    trv = np.flatnonzero(tr & valid); tev = te & valid
    SR = {sd: trv[seed_rows(pos[trv], ch[trv], e["pids"], sd)] for sd in SEEDS}
    for name, X in feats_d.items():
        Xte = X[tev].astype(np.float32)
        preds = fitpred_seeds([(X[SR[sd]].astype(np.float32), y_all[SR[sd]], Xte) for sd in SEEDS]); rows = SR[SEEDS[-1]]
        Ms = [conf_by_block(y_all[tev], p, e["blocks"][ch[tev]], ub, labels) for p in preds]
        D[name], BS[name] = seg_eval(Ms, ub, bidx)
        D[name]["n_train_tokens_per_seed"] = int(len(rows)); D[name]["n_test_tokens"] = int(tev.sum())
        say("dense", s, name, D[name]["macro_f1"])
    D["labels"] = labels
    R["dense"][s] = D
    C = {}
    for a in ["cls_broadcast", "mean_broadcast", "bandstats", "dinov2_patches"]:
        if a in BS:
            C[f"patches_minus_{a}"] = {"macro_f1": paired(BS["patches"], BS[a], D["patches"]["macro_f1"][0], D[a]["macro_f1"][0], 0),
                                       "miou": paired(BS["patches"], BS[a], D["patches"]["miou"][0], D[a]["miou"][0], 1)}
    R["dense_contrasts"][s] = C
    json.dump(R, open(OUTF, "w"), indent=1)

# ---------------- (ii-b) S2 GSD lie ----------------
if "s2" in E:
    e = E["s2"]; test_chip = e["split"] == "test"; ch = e["tok_chip"]; tr = ~test_chip[ch]; te = test_chip[ch]
    pos = e["tok_pos"].astype(int); row, col = pos // 32, pos % 32
    ub, bidx = boot_idx(e["blocks"][test_chip]); y_all = e["tok_label"].astype(int); valid = y_all > 0
    labels = R["dense"]["s2"]["labels"]; trv = np.flatnonzero(tr & valid); tev = te & valid; tblk = e["blocks"][ch[tev]]
    G = {"cosine": {}, "dense_transfer": {}, "dense_retrained": {}, "position_retrained": {}}
    # cosine: mean over test chips with block bootstrap of chip means
    cb = e["blocks"][test_chip]; bi = np.searchsorted(ub, cb)
    for g in [1, 5, 30, 100]:
        G["cosine"][str(g)] = {}
        for k in ["cls", "mean", "patch"]:
            v = e[f"cos_{k}_g{g}"][test_chip]; S = np.zeros((len(ub), 2)); np.add.at(S[:, 0], bi, v); np.add.at(S[:, 1], bi, 1)
            bs = [S[b, 0].sum() / S[b, 1].sum() for b in bidx]
            G["cosine"][str(g)][k] = [float(v.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
    def _fit(Xtr, ytr):
        with threadpool_limits(6): return clf_head().fit(Xtr, ytr)
    SR = [trv[seed_rows(pos[trv], ch[trv], e["pids"], sd)] for sd in SEEDS]
    models = list(zip(SR, Parallel(n_jobs=5)(delayed(_fit)(e["tok_patch"][r].astype(np.float32), y_all[r]) for r in SR)))
    base_Ms = [conf_by_block(y_all[tev], m.predict(e["tok_patch"][tev].astype(np.float32)), tblk, ub, labels) for _, m in models]
    base, base_bs = seg_eval(base_Ms, ub, bidx); G["dense_transfer"]["10"] = base; G["dense_retrained"]["10"] = base
    G["contrast_transfer_vs_10"] = {}; G["contrast_retrained_vs_10"] = {}
    for g in [1, 5, 30, 100]:
        X = e[f"tok_patch_g{g}"]
        Ms = [conf_by_block(y_all[tev], m.predict(X[tev].astype(np.float32)), tblk, ub, labels) for _, m in models]
        G["dense_transfer"][str(g)], bs = seg_eval(Ms, ub, bidx)
        G["contrast_transfer_vs_10"][str(g)] = paired(bs, base_bs, G["dense_transfer"][str(g)]["macro_f1"][0], base["macro_f1"][0])
        Xte = X[tev].astype(np.float32)
        Ms = [conf_by_block(y_all[tev], p, tblk, ub, labels) for p in fitpred_seeds([(X[r].astype(np.float32), y_all[r], Xte) for r, _ in models])]
        G["dense_retrained"][str(g)], bs = seg_eval(Ms, ub, bidx)
        G["contrast_retrained_vs_10"][str(g)] = paired(bs, base_bs, G["dense_retrained"][str(g)]["macro_f1"][0], base["macro_f1"][0])
        tb2 = e["blocks"][ch[te]]; rr = {}
        for tgt, y in [("row", row), ("col", col)]:
            p = reg_head().fit(X[tr].astype(np.float32), y[tr]).predict(X[te].astype(np.float32))
            rr[tgt] = r2_eval([r2_by_block(y[te].astype(float), p, tb2, ub)], ub, bidx)[0]
        rr["mean"] = (rr["row"][0] + rr["col"][0]) / 2
        G["position_retrained"][str(g)] = rr
        say("gsd lie", g, G["dense_transfer"][str(g)]["macro_f1"], G["dense_retrained"][str(g)]["macro_f1"], rr["mean"])
    G["position_retrained"]["10"] = {k: R["position"]["s2"]["patches"][k] for k in ["row", "col"]} | {"mean": R["position"]["s2"]["patches"]["mean"][0]}
    # predict supplied gsd from cls/mean across 5 arms
    gs = [1, 5, 10, 30, 100]
    for k in ["cls", "mean"]:
        X = np.concatenate([e[k] if g == 10 else e[f"{k}_g{g}"] for g in gs]); y = np.concatenate([np.full(len(e["pids"]), np.log10(g)) for g in gs])
        tc = np.tile(test_chip, len(gs)); blk = np.tile(e["blocks"], len(gs))
        p = reg_head().fit(X[~tc], y[~tc]).predict(X[tc])
        G[f"supplied_gsd_probe_{k}"] = {"r2": r2_eval([r2_by_block(y[tc], p, blk[tc], ub)], ub, bidx)[0], "rmse_log10": float(np.sqrt(np.mean((y[tc] - p) ** 2)))}
    R["gsd_lie_s2"] = G; json.dump(R, open(OUTF, "w"), indent=1); say("gsd probe", G["supplied_gsd_probe_cls"])

# ---------------- (ii-a) sensor / GSD probe across sensors ----------------
Xs = {k: [] for k in ["cls", "mean"]}; ys, gs_, tc, blk = [], [], [], []
for s, e in E.items():
    for k in Xs: Xs[k].append(e[k])
    ys += [s] * len(e["pids"]); gs_ += [np.log10(GSD[s])] * len(e["pids"]); tc.append(e["split"] == "test"); blk.append(e["blocks"])
ys = np.array(ys); gs_ = np.array(gs_); tc = np.concatenate(tc); blk = np.concatenate(blk)
ub, bidx = boot_idx(blk[tc]); S2G = {}
for k in Xs:
    X = np.concatenate(Xs[k]); m = clf_head().fit(X[~tc], ys[~tc]); p = m.predict(X[tc])
    labs = sorted(set(ys)); M = conf_by_block(ys[tc], p, blk[tc], ub, labs)
    sens, _ = seg_eval([M], ub, bidx)
    pg = reg_head().fit(X[~tc], gs_[~tc]).predict(X[tc])
    S2G[k] = {"sensor": sens, "log10_gsd_r2": r2_eval([r2_by_block(gs_[tc], pg, blk[tc], ub)], ub, bidx)[0],
              "log10_gsd_rmse": float(np.sqrt(np.mean((gs_[tc] - pg) ** 2))), "n_train": int((~tc).sum()), "n_test": int(tc.sum())}
R["sensor_gsd_probe"] = S2G; json.dump(R, open(OUTF, "w"), indent=1); say("sensor", {k: v["sensor"]["acc"][0] for k, v in S2G.items()})

# ---------------- (iv) location leakage ----------------
def llcode(la, lo):
    a, b = np.radians(la), np.radians(lo); return np.stack([np.sin(a), np.cos(a), np.sin(b), np.cos(b)], 1)
R["location"] = {}
for s, e in E.items():
    tcs = e["split"] == "test"; ub, bidx = boot_idx(e["blocks"][tcs]); L = {}
    arms = {"cls_on": e["cls"], "cls_off": e["cls_off"], "mean_on": e["mean"], "mean_off": e["mean_off"],
            "cls_on_documented_order": e["cls_doc"], "mean_on_documented_order": e["mean_doc"],
            "latlon_code": llcode(e["lat"], e["lon"]), "chip_bandstats": e["chipstats"]}
    bss = {}
    for a, X in arms.items():
        res = {}; P = {}
        for tgt, y in [("lat", e["lat"]), ("lon", e["lon"])]:
            p = reg_head().fit(X[~tcs], y[~tcs]).predict(X[tcs]); P[tgt] = p
            res[tgt], bs = r2_eval([r2_by_block(y[tcs], p, e["blocks"][tcs], ub)], ub, bidx); bss[(a, tgt)] = bs
        d = gcd_km(e["lat"][tcs], e["lon"][tcs], P["lat"], P["lon"])
        res["median_km"] = float(np.median(d)); L[a] = res
    for tgt in ["lat", "lon"]:
        L[f"contrast_cls_on_minus_off_{tgt}"] = paired(bss[("cls_on", tgt)][:, None], bss[("cls_off", tgt)][:, None], L["cls_on"][tgt][0], L["cls_off"][tgt][0])
    cu = lambda a: a / np.linalg.norm(a, axis=1, keepdims=True)
    L["contrast_cls_on_training_minus_documented_lat"] = paired(bss[("cls_on", "lat")][:, None], bss[("cls_on_documented_order", "lat")][:, None], L["cls_on"]["lat"][0], L["cls_on_documented_order"]["lat"][0])
    L["cos_cls_on_documented_mean"] = float((cu(e["cls"]) * cu(e["cls_doc"])).sum(1).mean())
    L["cos_cls_on_off_mean"] = float((cu(e["cls"]) * cu(e["cls_off"])).sum(1).mean())
    R["location"][s] = L; say("loc", s, {a: (L[a]["lat"][0], L[a]["median_km"]) for a in arms})
R["meta"]["elapsed_s"] = round(time.time() - t00)
json.dump(R, open(OUTF, "w"), indent=1); say("done")
