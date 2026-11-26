"""E8 item 1 evaluation (CPU): context-matched cross-sensor transfer vs E3 32 px on the same locations.
Arms, heads, seeds and bootstrap follow E3 evaluate.py (frozen) and E8 PROTOCOL.md (frozen 2026-11-05 22:25 UTC).
Usage: python e8_eval.py E3_MANIFEST E8_EMB E3_SUB E8_CHIPS [E8_NATIVE|-] OUT.json
"""
import csv, json, os, sys, time
import numpy as np
from joblib import Parallel, delayed
from scipy.linalg import orthogonal_procrustes
from sklearn.metrics import f1_score, accuracy_score
from sklearn.preprocessing import StandardScaler
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
import evalkit as ek

man, embp, e3p, chipsp, natp, outp = sys.argv[1:7]
rows = list(csv.DictReader(open(man))); N = len(rows)
ids = np.array([r["loc_id"] for r in rows]); y_all = np.array([int(r["label"]) for r in rows])
lat = np.array([float(r["lat"]) for r in rows]); lon = np.array([float(r["lon"]) for r in rows])
tid = np.array([int(r["tid"]) for r in rows])
blocks = ek.hblock(lat, lon, 100.0); split = ek.block_split(blocks)
assert all(split == np.array([r["split"] for r in rows]))
is_tr, is_te = split != "test", split == "test"
E8 = dict(np.load(embp)); E3 = dict(np.load(e3p))
assert list(E8["loc_id"]) == list(ids) and list(E3["loc_id"]) == list(ids)
CLASSES = [10, 20, 30, 40, 50, 60, 80]
L = E8["valid_s2"] & E8["valid_naip"] & np.isin(y_all, CLASSES)
chips = {r["loc_id"]: r for r in csv.DictReader(open(chipsp))}
NAT = None
if natp != "-" and os.path.exists(natp):
    z = np.load(natp); pos = {l: i for i, l in enumerate(ids)}
    NAT = {"mask": np.zeros(N, bool)}
    for k in ("clay_naipnat_on_cls", "clay_naipnat_on_mean"):
        a = np.full((N, 1024), np.nan, np.float32); ii = [pos[l] for l in z["loc_id"]]; a[ii] = z[k]; NAT[k] = a
    NAT["mask"][[pos[l] for l in z["loc_id"]]] = True

CFG = {"s2_32": ("s232", E3), "naip_320": ("naip32", E3), "s2_256": ("s2256", E8), "naip_256": ("naip256", E8)}


def feat(name, cfg):
    if cfg == "naip_nat":
        return NAT[{"clay_on_cls": "clay_naipnat_on_cls", "clay_on_mean": "clay_naipnat_on_mean"}[name]]
    key, src = CFG[cfg]
    if name.startswith("clay"):
        _, meta, ro = name.split("_"); return src[f"clay_{key}_{meta}_{ro}"].astype(np.float32)
    return src[f"dino_{key}_cls"].astype(np.float32)


F1 = lambda y, p: f1_score(y, p, average="macro")
SEEDS, KS, NS = ek.SEEDS, (10, 50), (100, 1000)


def boot_idx(b, n=1000, seed=1729):
    rng = np.random.default_rng(seed); ub = np.unique(b); rw = {u: np.where(b == u)[0] for u in ub}
    return [np.concatenate([rw[u] for u in rng.choice(ub, len(ub), replace=True)]) for _ in range(n)]


def run_pair(A, B, fname, m):
    XA, XB = feat(fname, A), feat(fname, B)
    tr, te = np.where(m & is_tr)[0], np.where(m & is_te)[0]
    ytr, yte, bte = y_all[tr], y_all[te], blocks[te]
    P = {}
    hA = ek.clf_head().fit(XA[tr], ytr); hB = ek.clf_head().fit(XB[tr], ytr)
    P["majority"] = [np.full(len(te), np.bincount(ytr).argmax())]
    P["native_full"] = [hB.predict(XB[te])]; P["native_source"] = [hA.predict(XA[te])]
    P["zeroshot"] = [hA.predict(XB[te])]
    scB = StandardScaler().fit(XB[tr]); P["zeroshot_bstd"] = [hA[-1].predict(scB.transform(XB[te]))]
    for k in KS:
        P[f"native_k{k}"] = [ek.clf_head().fit(XB[bi], y_all[bi]).predict(XB[te])
                             for bi in [tr[ek.label_budget(ids[tr], ytr, k, sd)] for sd in SEEDS]]
    for n in NS:
        pr = []
        for sd in SEEDS:
            pi = tr[ek.n_budget(ids[tr], min(n, len(tr)), sd)]
            muA, muB = XA[pi].mean(0), XB[pi].mean(0)
            R, _ = orthogonal_procrustes(XB[pi] - muB, XA[pi] - muA)
            pr.append(hA.predict((XB[te] - muB) @ R + muA))
        P[f"procrustes_n{n}"] = pr
    return {"A": A, "B": B, "feature": fname, "te": te, "P": P, "n_train": int(len(tr)), "n_test": int(len(te)),
            "n_test_blocks": int(len(np.unique(bte)))}


def summarise(r, RS):
    yte = y_all[r["te"]]; out = {"pair": f"{r['A']}->{r['B']}", "feature": r["feature"], "n_train": r["n_train"],
                                 "n_test": r["n_test"], "n_test_blocks": r["n_test_blocks"], "arms": {}, "contrasts": {}}
    S = {a: np.array([np.mean([F1(yte[i], p[i]) for p in P]) for i in RS]) for a, P in r["P"].items()}
    pt = {a: float(np.mean([F1(yte, p) for p in P])) for a, P in r["P"].items()}
    for a in r["P"]:
        out["arms"][a] = {"macro_f1": pt[a], "lo": float(np.percentile(S[a], 2.5)), "hi": float(np.percentile(S[a], 97.5)),
                          "acc": float(np.mean([accuracy_score(yte, p) for p in r["P"][a]]))}
    for a, b in [("zeroshot", "native_full"), ("zeroshot", "majority"), ("procrustes_n1000", "native_full"),
                 ("zeroshot_bstd", "zeroshot"), ("procrustes_n1000", "zeroshot")]:
        d = S[a] - S[b]; out["contrasts"][f"{a}-{b}"] = {"diff": pt[a] - pt[b], "lo": float(np.percentile(d, 2.5)), "hi": float(np.percentile(d, 97.5))}
    eff = (S["zeroshot"] - S["majority"]) / np.maximum(S["native_full"] - S["majority"], 1e-9)
    out["contrasts"]["transfer_efficiency"] = {"value": (pt["zeroshot"] - pt["majority"]) / max(pt["native_full"] - pt["majority"], 1e-9),
                                               "lo": float(np.percentile(eff, 2.5)), "hi": float(np.percentile(eff, 97.5))}
    return out, S, pt


def cross(r1, S1, p1, a1, b1, r2, S2, p2, a2, b2):
    """paired contrast (a1-b1 in run1) - (a2-b2 in run2); both runs on the same test rows -> shared resamples."""
    assert np.array_equal(r1["te"], r2["te"])
    d = (S1[a1] - (S1[b1] if b1 else 0)) - (S2[a2] - (S2[b2] if b2 else 0))
    pt = (p1[a1] - (p1[b1] if b1 else 0)) - (p2[a2] - (p2[b2] if b2 else 0))
    return {"diff": float(pt), "lo": float(np.percentile(d, 2.5)), "hi": float(np.percentile(d, 97.5))}


def ranks(Q, G, center=False, tiles=None):
    if center:
        Q = Q - Q.mean(0); G = G - G.mean(0)
    Q = Q / np.linalg.norm(Q, axis=1, keepdims=True); G = G / np.linalg.norm(G, axis=1, keepdims=True)
    rk = np.empty(len(Q), int)
    for a in range(0, len(Q), 1000):
        S = Q[a:a + 1000] @ G.T; ix = np.arange(a, a + S.shape[0])
        tru = S[np.arange(S.shape[0]), ix]
        if tiles is not None:
            same = tiles[ix][:, None] == tiles[None, :]; same[np.arange(S.shape[0]), ix] = False; S = np.where(same, -np.inf, S)
        rk[a:a + 1000] = (S > tru[:, None]).sum(1) + 1
    return rk


def raw_feats(arr):
    X = arr.astype(np.float32)
    mu = np.nanmean(X, axis=(0, 2, 3), keepdims=True); sd = np.nanstd(X, axis=(0, 2, 3), keepdims=True)
    pix = ((X - mu) / sd).reshape(len(X), -1)
    st = np.concatenate([X.mean((2, 3)), X.std((2, 3))], 1); st = (st - np.nanmean(st, 0)) / np.nanstd(st, 0)
    return pix, st


def main():
    t0 = time.time(); R = {"protocol": "E8 PROTOCOL.md frozen 2026-11-05 22:25 UTC", "classes": CLASSES}
    R["n"] = {"e3_locations": int(N), "s2_256_valid": int(E8["valid_s2"].sum()), "naip_256_valid": int(E8["valid_naip"].sum()),
              "L": int(L.sum()), "L_train": int((L & is_tr).sum()), "L_test": int((L & is_te).sum()),
              "L_test_blocks": int(len(np.unique(blocks[L & is_te]))),
              "s2_central_equals_e3": int(sum(chips[l]["s2_central_equals_e3"] == "True" for l in ids if l in chips)),
              "class_counts_L": {str(c): {"train": int(((y_all == c) & L & is_tr).sum()), "test": int(((y_all == c) & L & is_te).sum())} for c in CLASSES}}
    feats = ["clay_on_cls", "clay_on_mean", "clay_off_cls", "dino_cls"]
    tasks = []
    for f in feats:
        tasks += [("s2_256", "naip_256", f, "L"), ("s2_32", "naip_320", f, "L"), ("naip_256", "s2_256", f, "L"),
                  ("naip_320", "s2_32", f, "L"), ("s2_32", "s2_256", f, "L"), ("s2_256", "s2_32", f, "L"),
                  ("naip_320", "naip_256", f, "L"), ("naip_256", "naip_320", f, "L")]
    tasks += [("s2_256", "naip_256", "clay_on_center", "L"), ("naip_256", "s2_256", "clay_on_center", "L")]
    if NAT is not None:
        M = L & NAT["mask"]; R["n"]["native_subset"] = int(M.sum()); R["n"]["native_subset_test"] = int((M & is_te).sum())
        for f in ("clay_on_cls", "clay_on_mean"):
            tasks += [("s2_256", "naip_nat", f, "NAT"), ("s2_256", "naip_256", f, "NAT"), ("s2_32", "naip_320", f, "NAT")]
    masks = {"L": L, "NAT": (L & NAT["mask"]) if NAT is not None else None}
    runs = Parallel(n_jobs=int(os.environ.get("NJ", "8")), backend="threading")(delayed(run_pair)(A, B, f, masks[mk]) for A, B, f, mk in tasks)
    print("pairs done", f"{time.time()-t0:.0f}s", flush=True)
    RSc = {}; SUM, KEEP = [], {}
    for (A, B, f, mk), r in zip(tasks, runs):
        key = mk
        if key not in RSc:
            RSc[key] = boot_idx(blocks[r["te"]])
        s, S, pt = summarise(r, RSc[key]); s["set"] = mk; SUM.append(s); KEEP[(A, B, f, mk)] = (r, S, pt)
    R["classification"] = SUM
    # primary + secondary cross-config contrasts
    X = {}
    for f in feats + ["clay_on_center"]:
        k256, k32 = ("s2_256", "naip_256", f, "L"), ("s2_32", "naip_320", f, "L")
        if k256 in KEEP and k32 in KEEP:
            a, b = KEEP[k256], KEEP[k32]
            X[f"{f}|delta256-delta32 (zeroshot-native_full)"] = cross(*a, "zeroshot", "native_full", *b, "zeroshot", "native_full")
            X[f"{f}|zeroshot256-zeroshot32"] = cross(*a, "zeroshot", None, *b, "zeroshot", None)
            X[f"{f}|procrustes1000_256-32"] = cross(*a, "procrustes_n1000", None, *b, "procrustes_n1000", None)
            X[f"{f}|native_naip256-native_naip320"] = cross(*a, "native_full", None, *b, "native_full", None)
        ks, kn = ("s2_32", "s2_256", f, "L"), ("s2_32", "naip_320", f, "L")
        if ks in KEEP and kn in KEEP:
            X[f"{f}|ctx_offset(s2_32->s2_256 zs-nat) - sensor_offset(s2_32->naip_320 zs-nat)"] = cross(
                *KEEP[ks], "zeroshot", "native_full", *KEEP[kn], "zeroshot", "native_full")
    kc, k32 = ("s2_256", "naip_256", "clay_on_center", "L"), ("s2_32", "naip_320", "clay_on_cls", "L")
    X["center256 vs cls32|delta(zeroshot-native_full)"] = cross(*KEEP[kc], "zeroshot", "native_full", *KEEP[k32], "zeroshot", "native_full")
    X["center256 vs cls32|zeroshot"] = cross(*KEEP[kc], "zeroshot", None, *KEEP[k32], "zeroshot", None)
    X["center256 vs cls32|zeroshot_bstd"] = cross(*KEEP[kc], "zeroshot_bstd", None, *KEEP[k32], "zeroshot_bstd", None)
    X["center256 vs cls32|native_full"] = cross(*KEEP[kc], "native_full", None, *KEEP[k32], "native_full", None)
    if NAT is not None:
        for f in ("clay_on_cls", "clay_on_mean"):
            a, b = KEEP[("s2_256", "naip_nat", f, "NAT")], KEEP[("s2_256", "naip_256", f, "NAT")]
            X[f"{f}|NAT: zeroshot(s2_256->naip_nat) - zeroshot(s2_256->naip_256)"] = cross(*a, "zeroshot", None, *b, "zeroshot", None)
            X[f"{f}|NAT: delta_nat - delta_256"] = cross(*a, "zeroshot", "native_full", *b, "zeroshot", "native_full")
    R["cross_contrasts"] = X
    print("contrasts done", f"{time.time()-t0:.0f}s", flush=True)
    # retrieval + cosine on all L locations (train + test)
    ii = np.where(L)[0]; bq = blocks[ii]; RSq = boot_idx(bq)
    raw = {"s2_32": raw_feats(E3["raw_s232_rgbn"]), "naip_320": raw_feats(E3["raw_naip32_rgbn"]),
           "s2_256": raw_feats(E8["raw_s2256_rgbn"]), "naip_256": raw_feats(E8["raw_naip256_rgbn"])}
    def getF(name, cfg):
        if name == "raw_pixels": return raw[cfg][0]
        if name == "raw_stats": return raw[cfg][1]
        return feat(name, cfg)
    retr, HITS = [], {}
    combos = [("s2_256", "naip_256"), ("s2_32", "naip_320"), ("s2_32", "s2_256"), ("naip_320", "naip_256")]
    for (Qc, Gc) in combos:
        for name in ["clay_on_cls", "clay_off_cls", "clay_on_mean", "clay_on_center", "dino_cls", "raw_pixels", "raw_stats"]:
            if name == "clay_on_center" and "32" in Qc + Gc and not (Qc.endswith("256") and Gc.endswith("256")):
                continue
            Q, G = getF(name, Qc)[ii], getF(name, Gc)[ii]
            for cen in (False, True):
                for excl in (False, True):
                    rk = ranks(Q, G, cen, tid[ii] if excl else None)
                    d = {"query": Qc, "gallery": Gc, "method": name, "centered": cen, "exclude_same_tile": excl,
                         "n_gallery": int(len(ii)), "median_rank": float(np.median(rk)), "chance_r10": 10 / len(ii)}
                    for k in (1, 10):
                        h = (rk <= k).astype(float); st = np.array([h[j].mean() for j in RSq])
                        d[f"r@{k}"] = float(h.mean()); d[f"r@{k}_lo"] = float(np.percentile(st, 2.5)); d[f"r@{k}_hi"] = float(np.percentile(st, 97.5))
                        HITS[(Qc, Gc, name, cen, excl, k)] = (h, st)
                    retr.append(d)
    R["retrieval"] = retr
    rc = {}
    for name in ["clay_on_cls", "clay_off_cls", "clay_on_mean", "dino_cls", "raw_pixels"]:
        for cen in (False, True):
            for excl in (False, True):
                a = HITS.get(("s2_256", "naip_256", name, cen, excl, 10)); b = HITS.get(("s2_32", "naip_320", name, cen, excl, 10))
                if a and b:
                    d = a[1] - b[1]
                    rc[f"{name}|centered={cen}|excl_tile={excl}|R@10 256-32"] = {"diff": float(a[0].mean() - b[0].mean()),
                                                                                  "lo": float(np.percentile(d, 2.5)), "hi": float(np.percentile(d, 97.5))}
    R["retrieval_contrasts"] = rc
    if NAT is not None:
        jj = np.where(L & NAT["mask"])[0]; RSn = boot_idx(blocks[jj]); nat = []
        for Gc in ("naip_nat", "naip_256", "naip_320"):
            Qc = "s2_32" if Gc == "naip_320" else "s2_256"
            Q, G = feat("clay_on_cls", Qc)[jj], feat("clay_on_cls", Gc)[jj]
            for cen in (False, True):
                rk = ranks(Q, G, cen); h = (rk <= 10).astype(float); st = np.array([h[j].mean() for j in RSn])
                nat.append({"query": Qc, "gallery": Gc, "centered": cen, "n_gallery": int(len(jj)), "r@10": float(h.mean()),
                            "r@10_lo": float(np.percentile(st, 2.5)), "r@10_hi": float(np.percentile(st, 97.5)), "median_rank": float(np.median(rk))})
            Qn = Q / np.linalg.norm(Q, axis=1, keepdims=True); Gn = G / np.linalg.norm(G, axis=1, keepdims=True)
            pm = np.random.default_rng(0).permutation(len(jj))
            nat.append({"query": Qc, "gallery": Gc, "paired_cos": float((Qn * Gn).sum(1).mean()), "unpaired_cos": float((Qn * Gn[pm]).sum(1).mean())})
        R["native_subset_retrieval"] = nat
    cosd = []
    perm = np.random.default_rng(0).permutation(len(ii))
    for (Ac, Bc) in combos:
        for name in ["clay_on_cls", "clay_off_cls", "clay_on_mean", "clay_on_center", "dino_cls"]:
            if name == "clay_on_center" and not (Ac.endswith("256") and Bc.endswith("256")):
                continue
            Q, G = getF(name, Ac)[ii], getF(name, Bc)[ii]
            Q = Q / np.linalg.norm(Q, axis=1, keepdims=True); G = G / np.linalg.norm(G, axis=1, keepdims=True)
            pc_, uc = (Q * G).sum(1), (Q * G[perm]).sum(1); wa, wb = (Q * Q[perm]).sum(1), (G * G[perm]).sum(1)
            gap = np.array([pc_[j].mean() - uc[j].mean() for j in RSq])
            cosd.append({"A": Ac, "B": Bc, "feature": name, "paired_cos": float(pc_.mean()), "unpaired_cos": float(uc.mean()),
                         "within_A_cos": float(wa.mean()), "within_B_cos": float(wb.mean()),
                         "paired_minus_unpaired": [float(pc_.mean() - uc.mean()), float(np.percentile(gap, 2.5)), float(np.percentile(gap, 97.5))]})
    R["cosine"] = cosd
    json.dump(R, open(outp, "w"), indent=1)
    print("done", f"{time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
