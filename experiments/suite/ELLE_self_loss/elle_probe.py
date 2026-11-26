#!/usr/bin/env python3
"""ELLE check: can a Ridge probe on the frozen Clay v1.5 class token predict the chip's own reconstruction loss?

Target (per chip): mean absolute error between the normalised input patches and the decoder's reconstruction when the
encoder sees ALL patches (mask_ratio 0, no channel dropout) -- the "unmasked" ELLE loss of the PI28 pipeline -- and, as a
second target, the MAE-style loss on 75 % masked patches with a fixed mask seed. Chips: the Sentinel-2 256 px chips of the
E2 token study (suite/E2_tokens/code/fetch.py builds them; manifest.csv lists id, lat, lon, date, 100 km block).
Probe: StandardScaler + RidgeCV on the 1,024-d CLS, 5-fold GroupKFold over 100 km blocks. Reports Pearson r, R^2, Spearman
(out-of-fold) with block-bootstrap CIs, two baselines (band mean/std; mean patch token), and latency of the probe
(batched dot product, sklearn predict batched, sklearn predict on one vector) next to the encoder and decoder cost.

Usage: python elle_probe.py --chips DIR_WITH_s2_npz --manifest ../E2_tokens/manifest.csv --out ../../results/elle_probe.json
       [--n 3000] [--latlon-order training|documented] [--batch 16]
Environment: CLAY_V15_CKPT (checkpoint)."""
from __future__ import annotations

import argparse, json, math, os, sys, time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from einops import rearrange
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
COMMON = HERE.parent / "common"
sys.path.insert(0, str(COMMON))
from claymodel.model import Decoder, Encoder  # noqa: E402
from clayrt import CKPT_SHA256, encode_latlon, encode_time, platform_spec, sha256  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--chips", required=True, help="directory containing <pid>.npz S2 chips with a 'pixels' array (10, 256, 256) uint16")
ap.add_argument("--manifest", default=str(HERE.parent / "E2_tokens" / "manifest.csv"))
ap.add_argument("--ckpt", default=os.environ.get("CLAY_V15_CKPT", str(HERE.parents[2] / "checkpoints/v1.5/clay-v1.5.ckpt")))
ap.add_argument("--out", default=str(HERE.parents[1] / "results" / "elle_probe.json"))
ap.add_argument("--n", type=int, default=3000)
ap.add_argument("--batch", type=int, default=16)
ap.add_argument("--latlon-order", choices=["training", "documented"], default="training")
ap.add_argument("--seed", type=int, default=42)
a = ap.parse_args()
dev = "cuda" if torch.cuda.is_available() else "cpu"
if dev == "cuda":
    torch.cuda.set_per_process_memory_fraction(3.8 / 24.0)  # stay under the 4 GB shared-GPU allowance
rng = np.random.default_rng(a.seed)

man = pd.read_csv(a.manifest)
man = man[(man.sensor == "s2")].reset_index(drop=True)
have = {p.stem for p in Path(a.chips).glob("*.npz")}
man = man[man.pid.isin(have)].sort_values("pid").reset_index(drop=True)
man = man.iloc[np.sort(rng.choice(len(man), size=min(a.n, len(man)), replace=False))].reset_index(drop=True)
print(len(man), "chips", flush=True)

assert sha256(a.ckpt) == CKPT_SHA256
sd = torch.load(a.ckpt, map_location="cpu", weights_only=True, mmap=True)["state_dict"]
enc = Encoder(mask_ratio=0.0, patch_size=8, shuffle=False, dim=1024, depth=24, heads=16, dim_head=64, mlp_ratio=4.0)
enc.load_state_dict({k.removeprefix("model.encoder."): v for k, v in sd.items() if k.startswith("model.encoder.")}, strict=True)
dec = Decoder(mask_ratio=0.0, patch_size=8, encoder_dim=1024, dim=512, depth=4, heads=4, dim_head=64, mlp_ratio=4.0)
dec.load_state_dict({k.removeprefix("model.decoder."): v for k, v in sd.items() if k.startswith("model.decoder.")}, strict=True)
enc.eval().to(dev); dec.eval().to(dev)
dec_masked = Decoder(mask_ratio=0.75, patch_size=8, encoder_dim=1024, dim=512, depth=4, heads=4, dim_head=64, mlp_ratio=4.0)
dec_masked.load_state_dict(dec.state_dict()); dec_masked.eval().to(dev)
enc_masked = Encoder(mask_ratio=0.75, patch_size=8, shuffle=False, dim=1024, depth=24, heads=16, dim_head=64, mlp_ratio=4.0)
enc_masked.load_state_dict(enc.state_dict()); enc_masked.eval().to(dev)

spec = platform_spec("sentinel-2-l2a")
means = torch.tensor(spec["means"])[None, :, None, None]; stds = torch.tensor(spec["stds"])[None, :, None, None]
waves = torch.tensor(spec["waves"], device=dev); gsd = torch.tensor(float(spec["gsd"]), device=dev)


def patches_of(x):
    return rearrange(x, "B C (h p1) (w p2) -> B (h w) (C p1 p2)", p1=8, p2=8)


cls, meanp, stat, l_un, l_mk = [], [], [], [], []
t_enc = t_dec = 0.0
torch.manual_seed(a.seed)
with torch.inference_mode():
    for i in range(0, len(man), a.batch):
        sub = man.iloc[i:i + a.batch]
        px = np.stack([np.load(Path(a.chips) / f"{p}.npz")["pixels"] for p in sub.pid]).astype(np.float32)
        x = (torch.from_numpy(px) - means) / stds
        tt = torch.tensor([encode_time(datetime.fromisoformat(d.replace("Z", "+00:00"))) for d in sub.datetime], dtype=torch.float32)
        ll = torch.tensor([encode_latlon(la, lo, order=a.latlon_order) for la, lo in zip(sub.lat, sub.lon)], dtype=torch.float32)
        cube = {"pixels": x.to(dev), "time": tt.to(dev), "latlon": ll.to(dev), "gsd": gsd, "waves": waves}
        torch.cuda.synchronize() if dev == "cuda" else None; t0 = time.perf_counter()
        tok, ui, mi, mm = enc(cube)
        torch.cuda.synchronize() if dev == "cuda" else None; t1 = time.perf_counter()
        pred, _ = dec(tok, ui, mi, mm, cube["time"], cube["latlon"], gsd, waves)
        torch.cuda.synchronize() if dev == "cuda" else None; t2 = time.perf_counter()
        t_enc += t1 - t0; t_dec += t2 - t1
        tgt = patches_of(cube["pixels"])
        l_un.append((pred - tgt).abs().mean((1, 2)).cpu())
        cls.append(tok[:, 0].float().cpu()); meanp.append(tok[:, 1:].float().mean(1).cpu())
        stat.append(torch.cat([px.mean((2, 3)), px.std((2, 3))], 1) if False else torch.from_numpy(np.concatenate([px.mean((2, 3)), px.std((2, 3))], 1)))
        # masked (75 %) loss, fixed RNG stream
        tok_m, ui_m, mi_m, mm_m = enc_masked(cube)
        pred_m, _ = dec_masked(tok_m, ui_m, mi_m, mm_m, cube["time"], cube["latlon"], gsd, waves)
        per_patch = (pred_m - tgt).abs().mean(-1)
        l_mk.append(((per_patch * mm_m).sum(1) / mm_m.sum(1)).cpu())
        if (i // a.batch) % 20 == 0:
            print(f"  {i + len(sub)}/{len(man)}", flush=True)
X = torch.cat(cls).numpy().astype(np.float64); Xm = torch.cat(meanp).numpy().astype(np.float64); Xs = torch.cat(stat).numpy().astype(np.float64)
y_un = torch.cat(l_un).numpy().astype(np.float64); y_mk = torch.cat(l_mk).numpy().astype(np.float64)
groups = man.block100km.values


def oof(Xf, y):
    pred = np.empty_like(y)
    for tr, te in GroupKFold(5).split(Xf, y, groups):
        m = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 6, 33))).fit(Xf[tr], y[tr])
        pred[te] = m.predict(Xf[te])
    return pred


def stats(y, p):
    ub = np.unique(groups); idx = {u: np.where(groups == u)[0] for u in ub}
    r0 = pearsonr(y, p)[0]
    br = []
    bg = np.random.default_rng(1729)
    for _ in range(1000):
        ii = np.concatenate([idx[u] for u in bg.choice(ub, len(ub))])
        if np.std(y[ii]) > 0 and np.std(p[ii]) > 0:
            br.append(pearsonr(y[ii], p[ii])[0])
    r2 = 1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    return {"pearson_r": float(r0), "pearson_r_ci95": [float(np.percentile(br, 2.5)), float(np.percentile(br, 97.5))],
            "r2": float(r2), "spearman": float(spearmanr(y, p)[0])}


res = {"n_chips": int(len(man)), "n_blocks": int(len(np.unique(groups))), "latlon_order": a.latlon_order,
       "loss_unmasked": {"mean": float(y_un.mean()), "sd": float(y_un.std())}, "loss_masked75": {"mean": float(y_mk.mean()), "sd": float(y_mk.std())},
       "cv": "5-fold GroupKFold over 100 km blocks; Ridge alpha chosen by inner RidgeCV on the training folds only"}
for tname, y in (("unmasked_l1", y_un), ("masked75_l1_fixed_seed", y_mk)):
    res[tname] = {"cls_ridge": stats(y, oof(X, y)), "mean_patch_ridge": stats(y, oof(Xm, y)), "band_stats_ridge": stats(y, oof(Xs, y))}
res["correlation_between_targets"] = float(pearsonr(y_un, y_mk)[0])

# latency of the probe, fit on all chips (timing only)
mdl = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 6, 33))).fit(X, y_un)
sc, rg = mdl[0], mdl[1]
w_eff = rg.coef_ / sc.scale_; b_eff = rg.intercept_ - float((rg.coef_ * sc.mean_ / sc.scale_).sum())
Xf32 = X.astype(np.float32); w32 = w_eff.astype(np.float32)
def timeit(fn, reps):
    best = []
    for _ in range(5):
        t0 = time.perf_counter()
        for _ in range(reps): fn()
        best.append((time.perf_counter() - t0) / reps)
    return min(best)
t_dot = timeit(lambda: Xf32 @ w32 + b_eff, 200) / len(Xf32)
t_skb = timeit(lambda: mdl.predict(X), 20) / len(X)
t_sk1 = timeit(lambda: mdl.predict(X[:1]), 200)
res["latency_per_chip_seconds"] = {
    "folded_dot_product_batched_float32": t_dot, "sklearn_predict_batched": t_skb, "sklearn_predict_single_vector": t_sk1,
    "clay_encoder_forward_gpu_per_chip": t_enc / len(man), "clay_decoder_forward_gpu_per_chip": t_dec / len(man),
    "note": "the probe cost excludes obtaining the CLS token (already in the catalogue); the encoder+decoder figures are what the probe avoids when the decoder pass is skipped",
    "device": torch.cuda.get_device_name(0) if dev == "cuda" else "cpu"}
res["probe_check_dot_equals_predict_max_abs_diff"] = float(np.abs((X @ w_eff + b_eff) - mdl.predict(X)).max())
try:
    sys.path.insert(0, str(COMMON)); from provenance import provenance
    res["provenance"] = provenance(__file__, vars(a), root=HERE)
except Exception as e:  # noqa: BLE001
    res["provenance"] = {"error": str(e)}
Path(a.out).write_text(json.dumps(res, indent=1))
print(json.dumps({k: res[k] for k in ("unmasked_l1", "latency_per_chip_seconds")}, indent=1))
