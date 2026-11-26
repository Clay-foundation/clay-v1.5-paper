"""E7 analysis: pooled water IoU / mIoU, chip and seed-and-chip bootstraps, paired contrasts, retained fraction.
Usage: python analyze_e7.py RES_DIR OUT.json CHIPS.json
RES_DIR holds <arm>_s<seed>.npz (confusion counts per chip, written by train_e7.py) and the matching .json logs for all five arms.
Writes arms, paired contrasts (with chip-only, 100 km-block and seed-and-chip intervals), retained fraction and the per-run training
cost (wall seconds, peak GPU memory, trainable parameters)."""
import glob, json, sys, os
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "common"))
from evalkit import hblock
RES = sys.argv[1]; OUT = sys.argv[2]; CHIPS = sys.argv[3]
if not os.path.exists(CHIPS):
    sys.exit(f"{CHIPS} not found: run fetch_data.py in this directory first; it writes chips.json")
ARMS = ["clay_frozen_linear", "clay_frozen_upernet", "clay_finetune_upernet", "unet_scratch", "dofa_finetune_upernet"]
SPL = ["test", "bolivia", "valid"]
meta = {r["chip"]: r for r in json.load(open(CHIPS))}
C, logs = {}, {}
for a in ARMS:
    fs = sorted(glob.glob(f"{RES}/{a}_s[0-9].npz"))
    if not fs: continue
    C[a] = {s: np.stack([np.load(f)[f"conf_{s}"] for f in fs]) for s in SPL}  # [seed, chip, 4]
    logs[a] = [json.load(open(f.replace(".npz", ".json"))) for f in fs]
    ids = {s: np.load(fs[0])[f"ids_{s}"] for s in SPL}
    for f in fs: assert all((np.load(f)[f"ids_{s}"] == ids[s]).all() for s in SPL)

def met(conf):  # conf [..., chip, 4] -> water IoU, mIoU over pooled chips
    tp, fp, fn, tn = [conf[..., k].sum(-1) for k in range(4)]
    w = tp / (tp + fp + fn); l = tn / (tn + fn + fp)
    return w, (w + l) / 2

rng_seed, NB = 1729, 1000
def draws(s, mode, nseed=3):
    rng = np.random.default_rng(rng_seed); n = len(ids[s])
    if mode == "chip":
        return [rng.integers(0, n, n) for _ in range(NB)]
    lat = np.array([meta[c]["lat"] for c in ids[s]]); lon = np.array([meta[c]["lon"] for c in ids[s]])
    b = hblock(lat, lon); ub = np.unique(b); rows = {u: np.where(b == u)[0] for u in ub}
    return [np.concatenate([rows[u] for u in rng.choice(ub, len(ub))]) for _ in range(NB)]

def seedmean(a, s, idx, sidx=None, k=0):
    cf = C[a][s][:, idx] if sidx is None else C[a][s][sidx][:, idx]
    return met(cf)[k].mean()

out = {"arms": {}, "contrasts": {}, "retained_fraction": {}, "n_chips": {s: int(len(ids[s])) for s in SPL}, "n_boot": NB, "boot_seed": rng_seed}
KEYS = {"water_iou": 0, "miou": 1}
for s in SPL:
    D = {"chip": draws(s, "chip"), "block100km": draws(s, "block")}
    srng = np.random.default_rng(rng_seed + 1)
    S = {a: [srng.integers(0, C[a][s].shape[0], C[a][s].shape[0]) for _ in range(NB)] for a in C}  # seed draws per arm
    out["n_blocks_" + s] = int(len(np.unique(hblock([meta[c]["lat"] for c in ids[s]], [meta[c]["lon"] for c in ids[s]]))))
    full = np.arange(len(ids[s]))
    for a in C:
        for mk, k in KEYS.items():
            per_seed = met(C[a][s])[k]; pt = float(per_seed.mean())
            r = {"point": pt, "seeds": per_seed.tolist(), "seed_min": float(per_seed.min()), "seed_max": float(per_seed.max())}
            for dn, dd in D.items():
                v = [seedmean(a, s, i, k=k) for i in dd]; r[f"ci_{dn}"] = [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
            v = [seedmean(a, s, i, S[a][j], k=k) for j, i in enumerate(D["chip"])]
            r["ci_seed_and_chip"] = [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
            out["arms"].setdefault(a, {}).setdefault(s, {})[mk] = r
    pairs = [("clay_finetune_upernet", "clay_frozen_linear"), ("clay_finetune_upernet", "clay_frozen_upernet"),
             ("clay_frozen_upernet", "clay_frozen_linear"), ("clay_finetune_upernet", "unet_scratch"),
             ("clay_finetune_upernet", "dofa_finetune_upernet"), ("clay_frozen_upernet", "unet_scratch"),
             ("clay_frozen_linear", "unet_scratch")]
    for x, y in pairs:
        if x not in C or y not in C: continue
        for mk, k in KEYS.items():
            pt = seedmean(x, s, full, k=k) - seedmean(y, s, full, k=k)
            r = {"point": float(pt)}
            for dn, dd in D.items():
                v = [seedmean(x, s, i, k=k) - seedmean(y, s, i, k=k) for i in dd]; r[f"ci_{dn}"] = [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
            v = [seedmean(x, s, i, S[x][j], k=k) - seedmean(y, s, i, S[y][j], k=k) for j, i in enumerate(D["chip"])]
            r["ci_seed_and_chip"] = [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
            out["contrasts"].setdefault(f"{x} - {y}", {}).setdefault(s, {})[mk] = r
    if all(a in C for a in ("unet_scratch", "clay_finetune_upernet")):
        for X in ("clay_frozen_linear", "clay_frozen_upernet"):
            if X not in C: continue
            for mk, k in KEYS.items():
                f = lambda i, sx=None, sf=None, su=None: (seedmean(X, s, i, sx, k) - seedmean("unet_scratch", s, i, su, k)) / (seedmean("clay_finetune_upernet", s, i, sf, k) - seedmean("unet_scratch", s, i, su, k))
                den = [seedmean("clay_finetune_upernet", s, i, S["clay_finetune_upernet"][j], k) - seedmean("unet_scratch", s, i, S["unet_scratch"][j], k) for j, i in enumerate(D["chip"])]
                v = [f(i, S[X][j], S["clay_finetune_upernet"][j], S["unet_scratch"][j]) for j, i in enumerate(D["chip"])]
                dlo, dhi = np.percentile(den, [2.5, 97.5])
                out["retained_fraction"].setdefault(X, {}).setdefault(s, {})[mk] = {
                    "point": float(f(full)), "ci_seed_and_chip": [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))],
                    "denominator_ci": [float(dlo), float(dhi)], "ill_defined": bool(dlo <= 0 <= dhi)}
out["external_context_not_comparable"] = {
    "note": "Different recipes; reported only as context (see RESULTS.md). No frozen TerraMind score exists in the cited comparison.",
    "prithvi_eo_2_0_arxiv_2412_02732_table_iv": {"test_water_iou": "82.2-83.1", "test_miou": "89.7-90.3",
                                                "recipe": "6 HLS bands, 448-px resize, UPerNet head, 10-trial hyper-parameter tuning, 10 seeds"},
    "terramind_arxiv_2504_11171_table_8": {"model": "TerraMindv1-B S-2 full fine-tune", "test_water_iou": 82.26, "test_miou": 89.70},
    "this_study": "10 S2 bands, 256-px crops, 100 epochs, fixed per-arm recipe, 3 seeds per arm; follow-up arms elsewhere in the paper use 35 epochs and 1 seed"}
try:
    from provenance import provenance
    out["provenance"] = provenance(__file__, {"res": os.path.basename(RES.rstrip("/")), "out": os.path.basename(OUT)})
except Exception as e:  # noqa: BLE001
    out["provenance"] = {"error": str(e)}
out["training"] = {a: [{k: l.get(k) for k in ("seed", "best_epoch", "best_valid_wiou", "train_s", "peak_mem_gb", "trainable_params", "lr_head", "lr_enc")} for l in logs[a]] for a in logs}
os.makedirs(os.path.dirname(OUT) or ".", exist_ok=True)
json.dump(out, open(OUT, "w"), indent=1)
for a in out["arms"]:
    print(a, *[f"{s}: wIoU {out['arms'][a][s]['water_iou']['point']:.4f} {np.round(out['arms'][a][s]['water_iou']['ci_seed_and_chip'],4)}" for s in ("test", "bolivia")])
for c_, v in out["contrasts"].items():
    print(c_, *[f"{s}: {v[s]['water_iou']['point']:+.4f} {np.round(v[s]['water_iou']['ci_seed_and_chip'],4)}" for s in ("test", "bolivia")])
for X, v in out["retained_fraction"].items():
    print("R", X, {s: (round(v[s]['water_iou']['point'], 3), np.round(v[s]['water_iou']['ci_seed_and_chip'], 3).tolist(), v[s]['water_iou']['ill_defined']) for s in ("test", "bolivia")})
print({k: v for k, v in out.items() if k.startswith("n_")})
