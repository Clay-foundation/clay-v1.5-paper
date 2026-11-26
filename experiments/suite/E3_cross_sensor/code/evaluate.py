"""E3 evaluation: cross-sensor head transfer + paired-chip retrieval. Implements PROTOCOL.md.
Usage: python evaluate.py rows.json emb.npz OUTDIR [--overlay-trainorder emb_trainingorder.npz]

rows.json is built from ../manifest.csv by rows_from_manifest.py. With --overlay-trainorder, every metadata-on Clay
arm (clay_<sensor>_on_{cls,mean}) is replaced by the embedding computed with the TRAINING lat/lon order
([sin lon, cos lon, sin lat, cos lat]); metadata-off and DINOv2 arms are unchanged. Without it the metadata-on arms use
the documented order ([sin lat, cos lat, sin lon, cos lon])."""
import json, os, sys, time
import numpy as np
from joblib import Parallel, delayed
from scipy.linalg import orthogonal_procrustes
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, accuracy_score

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "common")); sys.path.insert(0, "common")
import evalkit as ek

_argv = [a for a in sys.argv[1:] if not a.startswith("--overlay-trainorder")]
_overlay = sys.argv[sys.argv.index("--overlay-trainorder") + 1] if "--overlay-trainorder" in sys.argv else None
if _overlay:
    _argv = [a for a in _argv if a != _overlay]
rows_p, emb_p, outdir = _argv[:3]
rows = json.load(open(rows_p)); E = dict(np.load(emb_p, allow_pickle=False))
LATLON_ORDER = "documented"
if _overlay:
    _O = np.load(_overlay, allow_pickle=False)
    assert list(_O["loc_id"]) == list(E["loc_id"]), "overlay rows differ from base embeddings"
    for _s in ("s2", "naip", "s1", "ls"):
        for _ro in ("cls", "mean"):
            E[f"clay_{_s}_on_{_ro}"] = _O[f"clay_{_s}_ontr_{_ro}"]
    LATLON_ORDER = "training"
N = len(rows)
assert list(E["loc_id"]) == [r["loc_id"] for r in rows]
ids = np.array([r["loc_id"] for r in rows]); y_all = np.array([r["label"] for r in rows])
lat = np.array([r["lat"] for r in rows]); lon = np.array([r["lon"] for r in rows])
blocks = ek.hblock(lat, lon, 100.0); split = ek.block_split(blocks)
is_tr = split != "test"; is_te = split == "test"
has = {s: E[f"has_{s}"].astype(bool) for s in ("s2", "naip", "s1", "ls")}
# class list on S2 ∩ NAIP
both = has["s2"] & has["naip"]
MIN_TR, MIN_TE = int(os.environ.get("E3_MIN_TRAIN", "100")), int(os.environ.get("E3_MIN_TEST", "20"))  # smoke test only: lower both (experiments/repro_smoke.sh)
classes = [int(c) for c in np.unique(y_all) if ((y_all == c) & both & is_tr).sum() >= MIN_TR and ((y_all == c) & both & is_te).sum() >= MIN_TE]
keepc = np.isin(y_all, classes)
print("N", N, "classes", classes, "train", int((both & is_tr & keepc).sum()), "test", int((both & is_te & keepc).sum()),
      "test blocks", len(np.unique(blocks[both & is_te & keepc])), flush=True)


def feat(name, s):
    if name.startswith("clay"):  # clay_on_cls
        _, meta, ro = name.split("_"); return E[f"clay_{s}_{meta}_{ro}"].astype(np.float32)
    _, ro = name.split("_"); return E[f"dino_{s}_{ro}"].astype(np.float32)


F1 = lambda y, p: f1_score(y, p, average="macro")
PAIRS = [("s2", "naip"), ("naip", "s2"), ("s2", "s1"), ("s1", "s2"), ("s2", "ls"), ("ls", "s2")]
FEATS = ["clay_on_cls", "clay_on_mean", "clay_off_cls", "dino_cls"]
KS, NS = (10, 50), (100, 1000)


def boot(pred_lists, y, b):
    out = {}
    for arm, P in pred_lists.items():
        pt, lo, hi = ek.block_bootstrap(F1, P, y, b)
        out[arm] = {"macro_f1": pt, "lo": lo, "hi": hi, "acc": float(np.mean([accuracy_score(y, p) for p in P])),
                    "n_seeds": len(P), "per_seed_f1": [float(F1(y, p)) for p in P]}
    return out


def run_pair(A, B, fname):
    if fname.startswith("dino") and ("s1" in (A, B)):
        return None
    m = has[A] & has[B] & keepc
    tr, te = np.where(m & is_tr)[0], np.where(m & is_te)[0]
    XA, XB = feat(fname, A), feat(fname, B)
    ytr, yte, bte = y_all[tr], y_all[te], blocks[te]
    P = {}
    hA = ek.clf_head().fit(XA[tr], ytr)
    hB = ek.clf_head().fit(XB[tr], ytr)
    P["majority"] = [np.full(len(te), np.bincount(ytr).argmax())]
    P["native_full"] = [hB.predict(XB[te])]
    P["zeroshot"] = [hA.predict(XB[te])]
    scB = StandardScaler().fit(XB[tr]); lr = hA[-1]
    P["zeroshot_bstd"] = [lr.predict(scB.transform(XB[te]))]
    for k in KS:
        nat, tk = [], []
        for sd in ek.SEEDS:
            bi = tr[ek.label_budget(ids[tr], ytr, k, sd)]
            nat.append(ek.clf_head().fit(XB[bi], y_all[bi]).predict(XB[te]))
            tk.append(ek.clf_head().fit(np.vstack([XA[tr], XB[bi]]), np.concatenate([ytr, y_all[bi]])).predict(XB[te]))
        P[f"native_k{k}"] = nat; P[f"transfer_k{k}"] = tk
    for n in NS:
        pr = []
        for sd in ek.SEEDS:
            pi = tr[ek.n_budget(ids[tr], min(n, len(tr)), sd)]
            muA, muB = XA[pi].mean(0), XB[pi].mean(0)
            R, _ = orthogonal_procrustes(XB[pi] - muB, XA[pi] - muA)
            pr.append(hA.predict((XB[te] - muB) @ R + muA))
        P[f"procrustes_n{n}"] = pr
    res = boot(P, yte, bte)
    c = {}
    for a, b2 in [("zeroshot", "native_full"), ("zeroshot", "majority"), ("procrustes_n1000", "native_full"),
                  ("transfer_k50", "native_k50"), ("transfer_k10", "native_k10"), ("zeroshot_bstd", "zeroshot")]:
        pt, lo, hi = ek.paired_block_bootstrap(F1, P[a], P[b2], yte, bte); c[f"{a}-{b2}"] = {"diff": pt, "lo": lo, "hi": hi}
    # transfer efficiency with shared resamples
    rng = np.random.default_rng(1729); ub = np.unique(bte); rws = {u: np.where(bte == u)[0] for u in ub}
    def eff(ix):
        z, nf, mj = F1(yte[ix], P["zeroshot"][0][ix]), F1(yte[ix], P["native_full"][0][ix]), F1(yte[ix], P["majority"][0][ix])
        return (z - mj) / max(nf - mj, 1e-9)
    st = [eff(np.concatenate([rws[u] for u in rng.choice(ub, len(ub))])) for _ in range(ek.NBOOT)]
    c["transfer_efficiency"] = {"value": float(eff(np.arange(len(te)))), "lo": float(np.percentile(st, 2.5)), "hi": float(np.percentile(st, 97.5))}
    return {"pair": f"{A}->{B}", "feature": fname, "n_train": int(len(tr)), "n_test": int(len(te)),
            "n_test_blocks": int(len(ub)), "arms": res, "contrasts": c}


def rank_retrieval(Q, G, center=False):
    if center:
        Q = Q - Q.mean(0); G = G - G.mean(0)
    Q = Q / np.linalg.norm(Q, axis=1, keepdims=True); G = G / np.linalg.norm(G, axis=1, keepdims=True)
    ranks = np.empty(len(Q), int)
    for a in range(0, len(Q), 1000):
        S = Q[a:a + 1000] @ G.T
        tru = S[np.arange(S.shape[0]), np.arange(a, a + S.shape[0])]
        ranks[a:a + 1000] = (S > tru[:, None]).sum(1) + 1
    return ranks


def retr_stats(ranks, b):
    rng = np.random.default_rng(1729); ub = np.unique(b); rws = {u: np.where(b == u)[0] for u in ub}
    out = {"n_gallery": int(len(ranks)), "median_rank": float(np.median(ranks)), "chance_r1": 1 / len(ranks)}
    bs = [np.concatenate([rws[u] for u in rng.choice(ub, len(ub))]) for _ in range(ek.NBOOT)]
    for k in (1, 10):
        h = (ranks <= k).astype(float)
        st = [h[ix].mean() for ix in bs]
        out[f"r@{k}"] = float(h.mean()); out[f"r@{k}_lo"] = float(np.percentile(st, 2.5)); out[f"r@{k}_hi"] = float(np.percentile(st, 97.5))
    return out


def raw_feats(s):
    X = E[f"raw_{s}_rgbn"].astype(np.float32)
    mu = np.nanmean(X, axis=(0, 2, 3), keepdims=True); sd = np.nanstd(X, axis=(0, 2, 3), keepdims=True)
    pix = ((X - mu) / sd).reshape(len(X), -1)
    stats = np.concatenate([X.mean((2, 3)), X.std((2, 3))], 1)
    stats = (stats - np.nanmean(stats, 0)) / np.nanstd(stats, 0)
    return pix, stats


def main():
    t0 = time.time(); os.makedirs(outdir, exist_ok=True)
    tasks = [(A, B, f) for (A, B) in PAIRS for f in FEATS]
    res = Parallel(n_jobs=int(os.environ.get("NJ", "12")), verbose=5)(delayed(run_pair)(A, B, f) for A, B, f in tasks)
    clf = [r for r in res if r is not None]
    print("clf done", f"{time.time()-t0:.0f}s", flush=True)
    # retrieval
    retr = []
    rp = {s: raw_feats(s) for s in ("s2", "naip")}
    for (Qs, Gs) in [("s2", "naip"), ("naip", "s2"), ("s2", "s1")]:
        m = has[Qs] & has[Gs]; ii = np.where(m)[0]
        meths = {f"clay_{meta}_{ro}": (feat(f"clay_{meta}_{ro}", Qs), feat(f"clay_{meta}_{ro}", Gs))
                 for meta in ("on", "off") for ro in ("cls", "mean")}
        if "s1" not in (Qs, Gs):
            meths["dino_cls"] = (feat("dino_cls", Qs), feat("dino_cls", Gs))
            meths["raw_pixels"] = (rp[Qs][0], rp[Gs][0]); meths["raw_stats"] = (rp[Qs][1], rp[Gs][1])
        for name, (Q, G) in meths.items():
            for cen in (False, True):
                rk = rank_retrieval(Q[ii], G[ii], cen)
                d = {"query": Qs, "gallery": Gs, "method": name, "centered": cen}; d.update(retr_stats(rk, blocks[ii]))
                retr.append(d)
    # descriptive paired cosine
    cosd = []
    for (A, B) in [("s2", "naip"), ("s2", "s1"), ("s2", "ls")]:
        ii = np.where(has[A] & has[B])[0]
        for name in ("clay_on_cls", "clay_off_cls", "dino_cls"):
            if name.startswith("dino") and B == "s1":
                continue
            Q, G = feat(name, A)[ii], feat(name, B)[ii]
            Q = Q / np.linalg.norm(Q, axis=1, keepdims=True); G = G / np.linalg.norm(G, axis=1, keepdims=True)
            paired = float((Q * G).sum(1).mean())
            rng = np.random.default_rng(0); perm = rng.permutation(len(ii))
            cosd.append({"pair": f"{A}-{B}", "feature": name, "paired_cos": paired, "unpaired_cos": float((Q * G[perm]).sum(1).mean()),
                         "within_A_cos": float((Q * Q[perm]).sum(1).mean()), "within_B_cos": float((G * G[perm]).sum(1).mean())})
    out = {"protocol": "PROTOCOL.md frozen 2026-11-05", "n_locations": N, "classes": classes,
           "class_counts": {str(c): {"train": int(((y_all == c) & both & is_tr).sum()), "test": int(((y_all == c) & both & is_te).sum())} for c in classes},
           "n_test_blocks_s2naip": int(len(np.unique(blocks[both & is_te & keepc]))),
           "classification": clf, "retrieval": retr, "cosine": cosd,
           "clay_evidence": json.loads(str(E["clay_evidence"])), "latlon_order_metadata_on": LATLON_ORDER,
           "n_pairs": {f"{r['pair']}|{r['feature']}": {"n_train": r["n_train"], "n_test": r["n_test"]} for r in clf if r["feature"] == "clay_on_cls"}}
    try:
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "common"))
        from provenance import provenance
        out["provenance"] = provenance(__file__, {"rows": os.path.basename(rows_p), "emb": os.path.basename(emb_p),
                                                  "overlay_trainorder": os.path.basename(_overlay) if _overlay else None})
    except Exception as e:  # noqa: BLE001
        out["provenance"] = {"error": str(e)}
    json.dump(out, open(os.path.join(outdir, "results.json"), "w"), indent=1)
    print("done", f"{time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
