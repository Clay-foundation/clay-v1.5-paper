#!/usr/bin/env python3
"""Complete all remaining single-date cells for Table 2:
1. GEO-Bench DOFA-L:
   - m-cashew-plant (LP + FT mIoU)
   - m-bigearthnet (LP + FT micro-F1)
   - m-brick-kiln, m-eurosat, m-so2sat, m-forestnet (FT OA)
   m-brick-kiln uses the 9 physical reflectance bands selected by channel index from the documented true
   GEO-Bench order (no B06, TCI composites excluded; audit fix #15).
2. Sen1Floods11 (S2) UPerNet water IoU (E7 protocol: 100 epochs, validation every 5, seeds 0,1,2 by default;
   audit fix #27 -- the earlier run used 35 epochs and one seed). Settings are stored per cell under `_settings`.
   - dofa_frozen_upernet (LP)
   - olmo_frozen_upernet (LP) & olmo_finetune_upernet (FT)
   - prithvi_frozen_upernet (LP)
   - dinov2_frozen_upernet (LP) & dinov2_finetune_upernet (FT)

Usage: python experiments/run_geobench_and_e7_complete.py [--output-dir DIR] [--resume] [--parts gb,e7]
         [--e7-epochs 100] [--e7-seeds 0,1,2] [--e7-arms a,b] [--gb-cells m_cashew,m_bigearthnet,m-brick-kiln,...]
Default output is a FRESH directory experiments/runs/remaining_<UTC>/remaining_all.json (audit fix #35).
"""
import argparse
import ast, gc, glob, io, json, math, os, pickle, shutil, subprocess, sys, time, types, zipfile
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import rasterio
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler

import torch.distributed.fsdp
if not hasattr(torch.distributed.fsdp, "fully_shard"):
    torch.distributed.fsdp.fully_shard = lambda *a, **kw: None
import olmoearth_pretrain.config, olmoearth_pretrain.datatypes, olmoearth_pretrain.evals
olmoearth_pretrain.config.require_olmo_core = lambda *a, **kw: None
sys.modules["olmoearth_pretrain.train.masking"] = olmoearth_pretrain.datatypes
m_pkg = types.ModuleType("olmoearth_pretrain.evals.models")
m_pkg.__path__ = [os.path.join(olmoearth_pretrain.evals.__path__[0], "models")]
sys.modules["olmoearth_pretrain.evals.models"] = m_pkg

from olmoearth_pretrain.config import Config
from olmoearth_pretrain.data.constants import Modality
from olmoearth_pretrain.data.normalize import Normalizer, Strategy
from olmoearth_pretrain.datatypes import MaskedOlmoEarthSample
from olmoearth_pretrain.nn.pooling import PoolingType, pool_unmasked_tokens
from olmoearth_pretrain.evals.models.prithviv2.prithviv2 import PrithviV2, PrithviV2Models

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "experiments"))
from run_standard_benchmarks import (
    WORK_DIR, H5_PYTHON, S2_10B_WAVES, L8_6B_WAVES, KILN_BANDS, EUROSAT_BANDS, gb_channels, gb_waves,
    clean_nonfinite, compute_train_band_stats, eval_dense_segmentation, _fit_gpu_linear_lbfgs,
    load_geobench_splits, olmo_files, provenance, utc_now, cfg_fingerprint, resolve_output, sha256_file,
)
from torchgeo.models import dofa_large_patch16_224, DOFALarge16_Weights

torch.cuda.set_per_process_memory_fraction(10 / 24.0)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

SCRIPT = str(Path(__file__).resolve())
OUT_FILE = None
RES: dict = {}
RESUME = False


def save_res():
    RES["_meta"] = provenance(SCRIPT)
    json.dump(RES, open(OUT_FILE, "w"), indent=2)


def have(key: str, contract: dict) -> bool:
    """Reuse a stored cell only under --resume and only if produced by this exact script and contract."""
    prod = RES.get("_producers", {}).get(key, {})
    return RESUME and key in RES and prod.get("script_sha256") == sha256_file(SCRIPT) and prod.get("cfg") == cfg_fingerprint(contract)


def put(key: str, value, contract: dict, base: dict | None = None):
    """Store a cell. `contract` is what is recorded (may carry result-side extras such as input checks or per-seed
    values); `base` is the pre-run contract that `have()` fingerprints, so --resume matches (default: contract)."""
    RES[key] = value
    RES.setdefault("_settings", {})[key] = contract
    RES.setdefault("_producers", {})[key] = {"script_sha256": sha256_file(SCRIPT), "cfg": cfg_fingerprint(base if base is not None else contract), "utc_timestamp": utc_now()}
    save_res()


def load_gb_direct(repo_file, task_name, band_specs, kind):
    """Pinned-revision GEO-Bench loader shared with run_standard_benchmarks (seg labels as int64)."""
    splits = load_geobench_splits(repo_file, task_name, band_specs, kind=kind)
    if kind == "seg":
        splits = {k: (X, y.astype(np.int64)) for k, (X, y) in splits.items()}
    load_gb_direct.last_meta = dict(load_geobench_splits.last_meta)
    return splits


load_gb_direct.last_meta = {}

@torch.no_grad()
def ext_dofa_frozen(dofa, X, means, stds, waves, need_pat=False):
    mu_t = torch.tensor(means, dtype=torch.float32, device="cuda").view(1, -1, 1, 1)
    sd_t = torch.tensor(stds, dtype=torch.float32, device="cuda").view(1, -1, 1, 1)
    w_t = torch.tensor(waves, dtype=torch.float32, device="cuda")
    cls_l, mean_l, pat_l = [], [], []
    for i in range(0, len(X), 64):
        xb = torch.as_tensor(X[i:i+64], dtype=torch.float32, device="cuda")
        if xb.shape[-2:] != (224, 224):
            xb = F.interpolate(xb, size=(224, 224), mode="bilinear", align_corners=False)
        xb = (xb - mu_t) / sd_t
        B = len(xb)
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.float16):
            tok, _ = dofa.patch_embed(xb, w_t)
            tok = tok + dofa.pos_embed[:, 1:, :]
            cls_tok = (dofa.cls_token + dofa.pos_embed[:, :1, :]).expand(B, -1, -1)
            tok = torch.cat((cls_tok, tok), dim=1)
            for blk in dofa.blocks:
                tok = blk(tok)
        tok = tok.float()
        cls_l.append(dofa.fc_norm(tok[:, 0]).cpu().numpy())
        mean_l.append(dofa.fc_norm(tok[:, 1:].mean(1)).cpu().numpy())
        if need_pat:
            pat_l.append(dofa.fc_norm(tok[:, 1:]).reshape(B, 14, 14, 1024).half().cpu().numpy())
    out = {"cls": np.concatenate(cls_l, 0), "mean": np.concatenate(mean_l, 0)}
    if need_pat:
        out["patches"] = np.concatenate(pat_l, 0)
    return out

def finetune_dofa_cls(X_tr, y_tr, X_va, y_va, X_te, y_te, means, stds, waves, multilabel=False, epochs=12):
    m = dofa_large_patch16_224(weights=DOFALarge16_Weights.DOFA_MAE).cuda()
    n_out = y_tr.shape[1] if multilabel else (int(max(y_tr.max(), y_va.max(), y_te.max())) + 1)
    head = nn.Linear(1024, n_out).cuda()
    opt = torch.optim.AdamW([
        {"params": head.parameters(), "lr": 5e-4},
        {"params": m.parameters(), "lr": 5e-5},
    ], weight_decay=0.05)
    mu_t = torch.tensor(means, dtype=torch.float32, device="cuda").view(1, -1, 1, 1)
    sd_t = torch.tensor(stds, dtype=torch.float32, device="cuda").view(1, -1, 1, 1)
    w_t = torch.tensor(waves, dtype=torch.float32, device="cuda")

    def prep_batch(arr):
        xb = torch.as_tensor(arr, dtype=torch.float32, device="cuda")
        if xb.shape[-2:] != (224, 224):
            xb = F.interpolate(xb, size=(224, 224), mode="bilinear", align_corners=False)
        return (xb - mu_t) / sd_t

    bs = 32
    n_ep = min(len(X_tr), 3200)
    steps = math.ceil(n_ep / bs) * epochs
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    gen = torch.Generator().manual_seed(42)

    def forward_pool(xb, train_mode):
        B = len(xb)
        tok, _ = m.patch_embed(xb, w_t)
        tok = tok + m.pos_embed[:, 1:, :]
        cls_tok = (m.cls_token + m.pos_embed[:, :1, :]).expand(B, -1, -1)
        h = torch.cat((cls_tok, tok), dim=1)
        for blk in m.blocks:
            h = checkpoint(blk, h, use_reentrant=False) if train_mode else blk(h)
        return head(m.fc_norm(h[:, 1:].mean(1)))

    @torch.no_grad()
    def eval_split(X_arr, yp):
        m.eval(); head.eval()
        preds = []
        for i in range(0, len(X_arr), 64):
            xb = prep_batch(X_arr[i:i+64])
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                logits = forward_pool(xb, False).float()
            if multilabel:
                preds.append((torch.sigmoid(logits) > 0.5).cpu().numpy().astype(int))
            else:
                preds.append(logits.argmax(1).cpu().numpy())
        p = np.concatenate(preds, 0)
        m.train(); head.train()
        if multilabel:
            return float(f1_score(yp.astype(int), p, average="micro")) * 100.0
        return float(accuracy_score(yp, p)) * 100.0

    best_va, best_te = -1.0, -1.0
    m.train(); head.train()
    for ep in range(epochs):
        perm = torch.randperm(len(X_tr), generator=gen)[:n_ep].numpy()
        for s in range(0, len(perm), bs):
            idx = perm[s:s+bs]
            xb = prep_batch(X_tr[idx])
            if torch.rand(1, generator=gen) < 0.5: xb = xb.flip(-1)
            if torch.rand(1, generator=gen) < 0.5: xb = xb.flip(-2)
            yb = torch.as_tensor(y_tr[idx], device="cuda")
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                logits = forward_pool(xb, True)
                loss = F.binary_cross_entropy_with_logits(logits, yb.float()) if multilabel else F.cross_entropy(logits, yb.long())
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
        if (ep + 1) % 4 == 0 or ep + 1 == epochs:
            va = eval_split(X_va, y_va)
            if va > best_va:
                best_va = va
                best_te = eval_split(X_te, y_te)
    del m, head
    torch.cuda.empty_cache()
    return round(best_te, 2)

def finetune_dofa_seg(X_tr, y_tr, X_va, y_va, X_te, y_te, means, stds, waves, n_cls, epochs=25):
    m = dofa_large_patch16_224(weights=DOFALarge16_Weights.DOFA_MAE).cuda()
    head = nn.Sequential(
        nn.ConvTranspose2d(1024, 256, 4, 4), nn.BatchNorm2d(256), nn.GELU(),
        nn.ConvTranspose2d(256, 128, 4, 4), nn.BatchNorm2d(128), nn.GELU(),
        nn.Conv2d(128, n_cls, 1)
    ).cuda()
    opt = torch.optim.AdamW([
        {"params": head.parameters(), "lr": 5e-4},
        {"params": m.parameters(), "lr": 5e-5},
    ], weight_decay=0.05)
    mu_t = torch.tensor(means, dtype=torch.float32, device="cuda").view(1, -1, 1, 1)
    sd_t = torch.tensor(stds, dtype=torch.float32, device="cuda").view(1, -1, 1, 1)
    w_t = torch.tensor(waves, dtype=torch.float32, device="cuda")
    H_mask, W_mask = y_tr.shape[1], y_tr.shape[2]

    def prep(X):
        xt = torch.from_numpy(X).float()
        if xt.shape[-2:] != (224, 224):
            xt = F.interpolate(xt, size=(224, 224), mode="bilinear", align_corners=False)
        return ((xt.cuda() - mu_t) / sd_t).cpu().half()

    Xtr_p, Xva_p, Xte_p = prep(X_tr), prep(X_va), prep(X_te)
    bs = 16
    steps = math.ceil(len(Xtr_p) / bs) * epochs
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    gen = torch.Generator().manual_seed(42)

    def forward_seg(xb, train_mode):
        B = len(xb)
        tok, _ = m.patch_embed(xb, w_t)
        tok = tok + m.pos_embed[:, 1:, :]
        cls_tok = (m.cls_token + m.pos_embed[:, :1, :]).expand(B, -1, -1)
        h = torch.cat((cls_tok, tok), dim=1)
        for blk in m.blocks:
            h = checkpoint(blk, h, use_reentrant=False) if train_mode else blk(h)
        f = m.fc_norm(h[:, 1:]).transpose(1, 2).reshape(B, 1024, 14, 14)
        logits = head(f)
        return F.interpolate(logits, size=(H_mask, W_mask), mode="bilinear", align_corners=False)

    @torch.no_grad()
    def eval_split(Xp, yp):
        m.eval(); head.eval()
        conf = np.zeros((n_cls, n_cls), dtype=np.int64)
        for i in range(0, len(Xp), 32):
            xb = Xp[i:i+32].cuda().float()
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                up = forward_seg(xb, False).float()
            pred = up.argmax(1).cpu().numpy().reshape(-1)
            gt = yp[i:i+32].reshape(-1).astype(np.int64)
            valid = (gt >= 0) & (gt < n_cls)
            conf += np.bincount(gt[valid] * n_cls + pred[valid], minlength=n_cls*n_cls).reshape(n_cls, n_cls)
        m.train(); head.train()
        inter = np.diag(conf).astype(np.float64)
        union = conf.sum(1) + conf.sum(0) - inter
        return float(np.mean([inter[k]/union[k] for k in range(n_cls) if union[k] > 0])) * 100.0

    best_va, best_te = -1.0, -1.0
    m.train(); head.train()
    for ep in range(epochs):
        perm = torch.randperm(len(Xtr_p), generator=gen)
        for s in range(0, len(Xtr_p), bs):
            idx = perm[s:s+bs]
            xb = Xtr_p[idx].cuda().float()
            yb = torch.as_tensor(y_tr[idx.numpy()], dtype=torch.long, device="cuda")
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                logits = forward_seg(xb, True)
                loss = F.cross_entropy(logits, yb, ignore_index=-1)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
        if (ep + 1) % 5 == 0 or ep + 1 == epochs:
            va = eval_split(Xva_p, y_va)
            if va > best_va:
                best_va = va
                best_te = eval_split(Xte_p, y_te)
    del m, head, Xtr_p, Xva_p, Xte_p
    torch.cuda.empty_cache()
    return round(best_te, 2)

bands_s2_std = [
    "02 - Blue", "03 - Green", "04 - Red",
    "05 - Vegetation Red Edge", "06 - Vegetation Red Edge", "07 - Vegetation Red Edge",
    "08 - NIR", "08A - Vegetation Red Edge", "11 - SWIR", "12 - SWIR",
]
bands_s2_wild = [b + "_*" for b in bands_s2_std]
bands_l8 = [
    "04 - Red_*", "03 - Green_*", "02 - Blue_*",
    "05 - NIR_*", "06 - SWIR1_*", "07 - SWIR2_*",
]
C_CASHEW = dict(model="dofa_l", waves_um=S2_10B_WAVES, lp="dense 1x1 ridge/logreg (run_standard_benchmarks.eval_dense_segmentation)", ft_epochs=25, ft_lr_head=5e-4, ft_lr_enc=5e-5, ft_seed=42, normaliser="train zscore")
C_BEN = dict(model="dofa_l", waves_um=S2_10B_WAVES, ft_epochs=20, ft_lr_head=5e-4, ft_lr_enc=5e-5, ft_seed=42, normaliser="train zscore")
# FT cells: (task id, repo file, band spec, waves, band names)
FT_CELLS = [
    ("m-brick-kiln", "classification_v1.0/m-brick-kiln.zip", gb_channels("m-brick-kiln", KILN_BANDS), gb_waves(KILN_BANDS), KILN_BANDS),
    ("m-eurosat", "classification_v1.0/m-eurosat.zip", gb_channels("m-eurosat", EUROSAT_BANDS), gb_waves(EUROSAT_BANDS), EUROSAT_BANDS),
    ("m-so2sat", "classification_v1.0/m-so2sat.zip", bands_s2_std, S2_10B_WAVES, ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"]),
    ("m-forestnet", "classification_v1.0/m-forestnet.zip", bands_l8, L8_6B_WAVES, ["R", "G", "B", "NIR", "SWIR1", "SWIR2"]),
]


def run_gb(sel):
    # 1. m-cashew-plant (LP + FT)
    if sel("m_cashew") and not have("m_cashew_dofa", C_CASHEW):
        t0 = time.time()
        print("=== m-cashew-plant DOFA-L LP + FT ===", flush=True)
        splits = load_gb_direct("segmentation_v1.0/m-cashew-plant.zip", "m-cashew-plant", bands_s2_wild, "seg")
        X_tr, y_tr = splits["train"]; X_va, y_va = splits["val"]; X_te, y_te = splits["test"]
        means, stds, _, _ = compute_train_band_stats(X_tr)
        X_tr = clean_nonfinite(X_tr, means); X_va = clean_nonfinite(X_va, means); X_te = clean_nonfinite(X_te, means)
        dofa = dofa_large_patch16_224(weights=DOFALarge16_Weights.DOFA_MAE).cuda().eval()
        d_tr = ext_dofa_frozen(dofa, X_tr, means, stds, S2_10B_WAVES, True)
        d_va = ext_dofa_frozen(dofa, X_va, means, stds, S2_10B_WAVES, True)
        d_te = ext_dofa_frozen(dofa, X_te, means, stds, S2_10B_WAVES, True)
        del dofa; torch.cuda.empty_cache()
        n_cls = int(max(y_tr.max(), y_va.max(), y_te.max())) + 1
        lp_res = eval_dense_segmentation(d_tr["patches"], y_tr, d_va["patches"], y_va, d_te["patches"], y_te, n_cls)
        ft_miou = finetune_dofa_seg(X_tr, y_tr, X_va, y_va, X_te, y_te, means, stds, S2_10B_WAVES, n_cls, epochs=25)
        put("m_cashew_dofa", {"lp_miou": lp_res["miou"], "ft_miou": ft_miou}, dict(C_CASHEW, input_check=load_gb_direct.last_meta), base=C_CASHEW)
        print(f"m-cashew-plant DOFA-L done in {time.time()-t0:.1f}s:", RES["m_cashew_dofa"], flush=True)
        del X_tr, X_va, X_te, d_tr, d_va, d_te; gc.collect()

    # 2. m-bigearthnet (LP + FT)
    if sel("m_bigearthnet") and not have("m_bigearthnet_dofa", C_BEN):
        t0 = time.time()
        print("=== m-bigearthnet DOFA-L LP + FT ===", flush=True)
        splits = load_gb_direct("classification_v1.0/m-bigearthnet.zip", "m-bigearthnet", bands_s2_std, "multilabel")
        X_tr, y_tr = splits["train"]; X_va, y_va = splits["val"]; X_te, y_te = splits["test"]
        means, stds, _, _ = compute_train_band_stats(X_tr)
        X_tr = clean_nonfinite(X_tr, means); X_va = clean_nonfinite(X_va, means); X_te = clean_nonfinite(X_te, means)
        dofa = dofa_large_patch16_224(weights=DOFALarge16_Weights.DOFA_MAE).cuda().eval()
        d_tr = ext_dofa_frozen(dofa, X_tr, means, stds, S2_10B_WAVES, False)
        d_va = ext_dofa_frozen(dofa, X_va, means, stds, S2_10B_WAVES, False)
        d_te = ext_dofa_frozen(dofa, X_te, means, stds, S2_10B_WAVES, False)
        del dofa; torch.cuda.empty_cache()
        out_ben = {}
        for ro in ("cls", "mean"):
            scaler = StandardScaler().fit(d_tr[ro])
            Xtr = torch.tensor(scaler.transform(d_tr[ro]), dtype=torch.float32, device="cuda")
            Xva = torch.tensor(scaler.transform(d_va[ro]), dtype=torch.float32, device="cuda")
            Xte = torch.tensor(scaler.transform(d_te[ro]), dtype=torch.float32, device="cuda")
            Ytr_t = torch.tensor(y_tr, dtype=torch.float32, device="cuda")
            best_f1, best_te_f1 = -1.0, -1.0
            for c in (0.01, 0.1, 1.0, 10.0):
                W, b = _fit_gpu_linear_lbfgs(Xtr, Ytr_t, y_tr.shape[1], 0.5 / (c * len(Xtr)), mode="bce")
                with torch.no_grad():
                    p_va = (torch.sigmoid(Xva @ W.t() + b) > 0.5).cpu().numpy().astype(int)
                    va_f1 = float(f1_score(y_va.astype(int), p_va, average="micro"))
                    if va_f1 > best_f1:
                        best_f1 = va_f1
                        p_te = (torch.sigmoid(Xte @ W.t() + b) > 0.5).cpu().numpy().astype(int)
                        best_te_f1 = float(f1_score(y_te.astype(int), p_te, average="micro"))
            out_ben[ro] = round(best_te_f1 * 100.0, 2)
        ft_f1 = finetune_dofa_cls(X_tr, y_tr, X_va, y_va, X_te, y_te, means, stds, S2_10B_WAVES, multilabel=True, epochs=20)
        put("m_bigearthnet_dofa", {"lp_cls": out_ben["cls"], "lp_mean": out_ben["mean"], "ft_f1": ft_f1}, dict(C_BEN, input_check=load_gb_direct.last_meta), base=C_BEN)
        print(f"m-bigearthnet DOFA-L done in {time.time()-t0:.1f}s:", RES["m_bigearthnet_dofa"], flush=True)
        del X_tr, X_va, X_te, d_tr, d_va, d_te; gc.collect()


    # 3. DOFA-L FT on m-brick-kiln, m-eurosat, m-so2sat, m-forestnet
    for tid, rfile, bspec, wv, bnames in FT_CELLS:
        k = f"{tid}_dofa_ft"
        c = dict(model="dofa_l", bands=bnames, waves_um=wv, ft_epochs=20, ft_lr_head=5e-4, ft_lr_enc=5e-5, ft_seed=42, normaliser="train zscore")
        if sel(tid) and not have(k, c):
            t0 = time.time()
            print(f"=== {tid} DOFA-L FT ===", flush=True)
            splits = load_gb_direct(rfile, tid, bspec, "cls")
            c_base = dict(c)
            c["input_check"] = load_gb_direct.last_meta
            X_tr, y_tr = splits["train"]; X_va, y_va = splits["val"]; X_te, y_te = splits["test"]
            means, stds, _, _ = compute_train_band_stats(X_tr)
            X_tr = clean_nonfinite(X_tr, means); X_va = clean_nonfinite(X_va, means); X_te = clean_nonfinite(X_te, means)
            ft_oa = finetune_dofa_cls(X_tr, y_tr, X_va, y_va, X_te, y_te, means, stds, wv, multilabel=False, epochs=20)
            put(k, ft_oa, c, base=c_base)
            print(f"{tid} DOFA-L FT done in {time.time()-t0:.1f}s: {ft_oa}", flush=True)
            del X_tr, X_va, X_te; gc.collect()


# 4. Sen1Floods11 (S2) UPerNet arms (E7 protocol)
DATA_E7 = os.environ.get("CLAY_E7_DATA", str(HERE / "data/E7_finetune"))
BIDX = [1, 2, 3, 4, 5, 6, 7, 8, 11, 12]
S2_MEANS = np.array([1105.0, 1355.0, 1552.0, 1887.0, 2422.0, 2630.0, 2743.0, 2785.0, 2388.0, 1835.0], np.float32)[:, None, None]
S2_STDS  = np.array([1809.0, 1757.0, 1888.0, 1870.0, 1732.0, 1697.0, 1684.0, 1647.0, 1470.0, 1379.0], np.float32)[:, None, None]

def cbr(i, o, k=3):
    return nn.Sequential(nn.Conv2d(i, o, k, padding=k // 2, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True))

class Neck(nn.Module):
    def __init__(self, din, stride=16, c=256):
        super().__init__()
        self.proj = nn.ModuleList([nn.Conv2d(din, c, 1) for _ in range(4)])
        up = lambda: nn.Sequential(nn.ConvTranspose2d(c, c, 2, 2), nn.BatchNorm2d(c), nn.GELU())
        if stride == 8:
            self.rs = nn.ModuleList([up(), nn.Identity(), nn.MaxPool2d(2), nn.MaxPool2d(4)])
        else:
            self.rs = nn.ModuleList([nn.Sequential(up(), nn.ConvTranspose2d(c, c, 2, 2)), up(), nn.Identity(), nn.MaxPool2d(2)])
    def forward(self, feats):
        return [r(p(f)) for f, p, r in zip(feats, self.proj, self.rs)]

class UPerHead(nn.Module):
    def __init__(self, c=256, ncls=2, scales=(1, 2, 3, 6)):
        super().__init__()
        self.ppm = nn.ModuleList([nn.Sequential(nn.AdaptiveAvgPool2d(s), cbr(c, c, 1)) for s in scales])
        self.bott = cbr(c * (1 + len(scales)), c)
        self.lat = nn.ModuleList([cbr(c, c, 1) for _ in range(3)])
        self.fpn = nn.ModuleList([cbr(c, c) for _ in range(3)])
        self.fuse = cbr(4 * c, c); self.cls = nn.Conv2d(c, ncls, 1)
    def forward(self, f):
        top = f[3]; hw = top.shape[-2:]
        top = self.bott(torch.cat([top] + [F.interpolate(p(top), hw, mode="bilinear", align_corners=False) for p in self.ppm], 1))
        lats = [l(x) for l, x in zip(self.lat, f[:3])] + [top]
        for i in range(2, -1, -1):
            lats[i] = lats[i] + F.interpolate(lats[i + 1], lats[i].shape[-2:], mode="bilinear", align_corners=False)
        outs = [self.fpn[i](lats[i]) for i in range(3)] + [lats[3]]
        hw0 = outs[0].shape[-2:]
        outs = [outs[0]] + [F.interpolate(o, hw0, mode="bilinear", align_corners=False) for o in outs[1:]]
        return self.cls(self.fuse(torch.cat(outs, 1)))

def load_e7_data():
    rows = json.load(open(f"{DATA_E7}/chips.json"))
    norm_olmo = Normalizer(Strategy.COMPUTED)
    idx10 = [0, 1, 2, 6, 3, 4, 5, 7, 8, 9]
    D = {}
    for r in rows:
        c = r["chip"]
        raw = rasterio.open(f"{DATA_E7}/S2Hand/{c}_S2Hand.tif").read()[BIDX].astype(np.float32)
        y = rasterio.open(f"{DATA_E7}/LabelHand/{c}_LabelHand.tif").read(1).astype(np.int64)
        nod = (raw == 0).any(0)
        x_z = (raw - S2_MEANS) / S2_STDS; x_z[:, nod] = 0.0
        # RGB [0,1] for DINOv2 (B04, B03, B02 = indices 2, 1, 0)
        rgb = np.clip(raw[[2, 1, 0]] / 4000.0, 0.0, 1.0)
        rgb[:, nod] = 0.0
        # OlmoEarth 12-band normalized
        s2_12 = np.zeros((512, 512, 1, 12), dtype=np.float32)
        s2_12[..., :10] = np.transpose(raw[idx10], (1, 2, 0))[..., None, :]
        s2_olmo = norm_olmo.normalize(Modality.SENTINEL2_L2A, s2_12[None])[0, ..., 0, :].astype(np.float32) # (512, 512, 12)
        s2_olmo[..., 10:] = 0.0
        s2_olmo = np.transpose(s2_olmo, (2, 0, 1)) # (12, 512, 512)
        s2_olmo[:, nod] = 0.0
        D[c] = dict(
            raw=torch.from_numpy(raw).half(),
            xz=torch.from_numpy(x_z).half(),
            rgb=torch.from_numpy(rgb).half(),
            olmo=torch.from_numpy(s2_olmo).half(),
            y=torch.from_numpy(y),
            split=r["split"][0],
        )
    return D

class E7SegModel(nn.Module):
    def __init__(self, arm):
        super().__init__()
        self.arm = arm
        self.trainable = ("finetune" in arm)
        if arm.startswith("dofa"):
            m = dofa_large_patch16_224(weights=DOFALarge16_Weights.DOFA_MAE)
            pe = m.pos_embed.data; g0 = int(math.isqrt(pe.shape[1] - 1)); g = 256 // 16
            grid = F.interpolate(pe[:, 1:].reshape(1, g0, g0, -1).permute(0, 3, 1, 2), (g, g), mode="bicubic", align_corners=False).permute(0, 2, 3, 1).reshape(1, g*g, -1)
            m.pos_embed = nn.Parameter(torch.cat([pe[:, :1], grid], 1), requires_grad=self.trainable)
            m.requires_grad_(self.trainable)
            self.bb = m
            self.norm = getattr(m, "norm", None) or m.fc_norm
            self.waves = torch.tensor(S2_10B_WAVES, dtype=torch.float32, device="cuda")
            self.neck = Neck(1024, stride=16); self.head = UPerHead()
        elif arm.startswith("dinov2"):
            import timm
            m = timm.create_model("vit_large_patch14_reg4_dinov2.lvd142m", pretrained=True, num_classes=0, dynamic_img_size=True)
            m.requires_grad_(self.trainable)
            self.bb = m
            cfg = m.pretrained_cfg
            self.register_buffer("im_mu", torch.tensor(cfg["mean"], dtype=torch.float32).view(1, 3, 1, 1))
            self.register_buffer("im_sd", torch.tensor(cfg["std"], dtype=torch.float32).view(1, 3, 1, 1))
            self.neck = Neck(1024, stride=16); self.head = UPerHead()
        elif arm.startswith("prithvi"):
            m = PrithviV2(load_directory=os.environ.get("CLAY_PRITHVI_DIR", os.path.expanduser("~/.cache/clay_paper/prithvi")), size=PrithviV2Models.VIT_300)
            m.requires_grad_(self.trainable)
            self.bb = m
            self.neck = Neck(1024, stride=16); self.head = UPerHead()
        elif arm.startswith("olmo"):
            p, w = olmo_files()
            cfg = json.load(open(p))
            ecfg = cfg["model"]["encoder_config"]
            ecfg.pop("attention_pooler_config", None)
            enc = Config.from_dict(ecfg).build()
            sd = torch.load(w, map_location="cpu", weights_only=False)
            enc_sd = {k[len("encoder."):]: v for k, v in sd.items() if k.startswith("encoder.") and not k.startswith("encoder.attention_pooler")}
            enc.load_state_dict(enc_sd, strict=True)
            enc.requires_grad_(self.trainable)
            self.bb = enc
            self.neck = Neck(768, stride=8); self.head = UPerHead()

    def forward_bb(self, x):
        B = x.shape[0]
        if self.arm.startswith("dofa"):
            h, _ = self.bb.patch_embed(x, self.waves)
            h = h + self.bb.pos_embed[:, 1:]
            h = torch.cat([(self.bb.cls_token + self.bb.pos_embed[:, :1]).expand(B, -1, -1), h], 1)
            out = []
            for i, blk in enumerate(self.bb.blocks):
                h = checkpoint(blk, h, use_reentrant=False) if (self.trainable and self.training) else blk(h)
                if i + 1 in (6, 12, 18, 24):
                    out.append(self.norm(h)[:, 1:].transpose(1, 2).reshape(B, 1024, 16, 16))
            return out
        elif self.arm.startswith("dinov2"):
            x224 = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
            x224 = (x224 - self.im_mu) / self.im_sd
            feats = self.bb.get_intermediate_layers(x224, n=[5, 11, 17, 23], reshape=True, norm=True)
            return list(feats)
        elif self.arm.startswith("prithvi"):
            idx10 = [0, 1, 2, 6, 3, 4, 5, 7, 8, 9]
            s2_12 = torch.zeros((B, 256, 256, 1, 12), dtype=x.dtype, device=x.device)
            s2_12[..., :10] = x[:, idx10].permute(0, 2, 3, 1).unsqueeze(3)
            samp = MaskedOlmoEarthSample(
                timestamps=torch.tensor([[[15, 6, 2020]]] * B, dtype=torch.int64, device=x.device),
                sentinel2_l2a=s2_12,
                sentinel2_l2a_mask=torch.zeros((B, 256, 256, 1, 1), dtype=torch.int64, device=x.device),
            )
            inp = self.bb.prepare_input(samp)
            f_list = self.bb.model.forward_features(inp)
            out = []
            for idx in (5, 11, 17, 23):
                t = f_list[idx][:, 1:, :].transpose(1, 2).reshape(B, 1024, 14, 14)
                out.append(t)
            return out
        elif self.arm.startswith("olmo"):
            s2_12 = x.permute(0, 2, 3, 1).unsqueeze(3) # (B, 256, 256, 1, 12)
            samp = MaskedOlmoEarthSample(
                timestamps=torch.tensor([[[15, 6, 2020]]] * B, dtype=torch.int64, device=x.device),
                sentinel2_l2a=s2_12,
                sentinel2_l2a_mask=torch.zeros((B, 256, 256, 1, 1), dtype=torch.int64, device=x.device),
            )
            out_o = self.bb(samp, patch_size=8, fast_pass=True)["tokens_and_masks"]
            grid = pool_unmasked_tokens(out_o, PoolingType.MEAN, spatial_pooling=True) # (B, 32, 32, 768)
            f = grid.permute(0, 3, 1, 2) # (B, 768, 32, 32)
            return [f, f, f, f]

    def forward(self, x):
        H = x.shape[-1]
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            if not self.trainable:
                with torch.no_grad():
                    f = self.forward_bb(x)
                f = [t.float() for t in f]
            else:
                f = self.forward_bb(x)
            o = self.head(self.neck(f))
        return F.interpolate(o.float(), (H, H), mode="bilinear", align_corners=False)

def run_e7_arm(D, arm, epochs=100, seed=0, val_every=5):
    torch.manual_seed(seed); np.random.seed(seed)
    ids = {s: sorted(c for c in D if D[c]["split"] == s) for s in ("train", "valid", "test")}
    model = E7SegModel(arm).cuda()
    model.train()
    if not model.trainable:
        model.bb.eval()
    lr_head = 1e-4
    enc = [p for n, p in model.named_parameters() if n.startswith("bb.") and p.requires_grad]
    rest = [p for n, p in model.named_parameters() if not n.startswith("bb.") and p.requires_grad]
    groups = [{"params": rest, "lr": lr_head}] + ([{"params": enc, "lr": 0.1 * lr_head}] if enc else [])
    opt = torch.optim.AdamW(groups, betas=(0.9, 0.999), weight_decay=0.05)
    bs = 8; spe = math.ceil(len(ids["train"]) / bs); total = spe * epochs; warm = spe * 5
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(total - warm, 1))))
    gen = torch.Generator().manual_seed(seed)
    key = "xz" if arm.startswith("dofa") else ("rgb" if arm.startswith("dinov2") else ("raw" if arm.startswith("prithvi") else "olmo"))

    @torch.no_grad()
    def eval_ids(id_list):
        model.eval()
        conf = np.zeros(4, np.int64)
        tiles = [(ci, a, b) for ci in range(len(id_list)) for a in (0, 256) for b in (0, 256)]
        for s in range(0, len(tiles), 16):
            T = tiles[s:s+16]
            xb = torch.stack([D[id_list[ci]][key][:, a:a+256, b:b+256].float() for ci, a, b in T]).cuda()
            p = model(xb).argmax(1).cpu()
            for k, (ci, a, b) in enumerate(T):
                y = D[id_list[ci]]["y"][a:a+256, b:b+256]; v = y >= 0; pk, yk = p[k][v], y[v]
                conf += [int(((pk == 1) & (yk == 1)).sum()), int(((pk == 1) & (yk == 0)).sum()),
                         int(((pk == 0) & (yk == 1)).sum()), int(((pk == 0) & (yk == 0)).sum())]
        model.train()
        if not model.trainable: model.bb.eval()
        tp, fp, fn, tn = conf
        return float(tp / max(tp + fp + fn, 1)) * 100.0

    best_va, best_te = -1.0, -1.0
    t0 = time.time()
    for ep in range(epochs):
        perm = torch.randperm(len(ids["train"]), generator=gen).tolist()
        for s in range(spe):
            b_ids = [ids["train"][i] for i in perm[s*bs:(s+1)*bs]]
            xs, ys = [], []
            for c in b_ids:
                d = D[c]
                i, j = torch.randint(0, 257, (2,), generator=gen).tolist()
                x = d[key][:, i:i+256, j:j+256].float(); y = d["y"][i:i+256, j:j+256]
                if torch.rand(1, generator=gen) < 0.5: x, y = x.flip(-1), y.flip(-1)
                if torch.rand(1, generator=gen) < 0.5: x, y = x.flip(-2), y.flip(-2)
                k = int(torch.randint(0, 4, (1,), generator=gen)); x, y = torch.rot90(x, k, (-2, -1)), torch.rot90(y, k, (-2, -1))
                xs.append(x); ys.append(y)
            xb = torch.stack(xs).cuda(); yb = torch.stack(ys).cuda()
            loss = F.cross_entropy(model(xb), yb, ignore_index=-1)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
        if (ep + 1) % val_every == 0 or ep + 1 == epochs:
            va = eval_ids(ids["valid"])
            if va > best_va:
                best_va = va
                best_te = eval_ids(ids["test"])
            print(f"  [{arm}] ep {ep+1}/{epochs} valid={va:.2f} best_test={best_te:.2f} ({time.time()-t0:.0f}s)", flush=True)
    del model
    torch.cuda.empty_cache()
    return round(best_te, 1)

E7_ARMS = ["dofa_frozen_upernet", "olmo_frozen_upernet", "olmo_finetune_upernet", "prithvi_frozen_upernet", "dinov2_frozen_upernet", "dinov2_finetune_upernet"]


def run_e7(arms, epochs, seeds, val_every):
    D_e7 = None
    for arm in arms:
        k = f"sen1floods11_s2_{arm}"
        c = dict(arm=arm, epochs=epochs, seeds=seeds, val_every=val_every, batch=8, lr_decoder=1e-4, lr_encoder_ratio=0.1,
                 warmup_epochs=5, tile="256 random crop, flips + rot90", protocol="E7 PROTOCOL.md (100 epochs, valid every 5, seeds 0-2)",
                 matched_to_clay_e7=(epochs == 100 and sorted(seeds) == [0, 1, 2] and val_every == 5))
        if have(k, c):
            continue
        if D_e7 is None:
            print("=== Loading Sen1Floods11 (S2) data ===", flush=True)
            D_e7 = load_e7_data()
        print(f"=== Running {arm} on Sen1Floods11 (S2): epochs={epochs} seeds={seeds} ===", flush=True)
        per_seed = {}
        for sd in seeds:
            t0 = time.time()
            per_seed[str(sd)] = run_e7_arm(D_e7, arm, epochs=epochs, seed=sd, val_every=val_every)
            print(f"  seed {sd}: {per_seed[str(sd)]} ({time.time()-t0:.0f}s)", flush=True)
        vals = list(per_seed.values())
        put(k, round(float(np.mean(vals)), 1), dict(c, per_seed_test_water_iou=per_seed, seed_min=min(vals), seed_max=max(vals)), base=c)
        print(f"Result {k}: {RES[k]}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--parts", default="gb,e7")
    ap.add_argument("--gb-cells", default=None, help="subset: m_cashew,m_bigearthnet,m-brick-kiln,m-eurosat,m-so2sat,m-forestnet")
    ap.add_argument("--e7-epochs", type=int, default=100)
    ap.add_argument("--e7-seeds", default="0,1,2")
    ap.add_argument("--e7-val-every", type=int, default=5)
    ap.add_argument("--e7-arms", default=",".join(E7_ARMS))
    a = ap.parse_args()
    RESUME = a.resume
    OUT_FILE = resolve_output(a.output_dir, a.resume, "remaining", "remaining_all.json")
    print(f"[output] {OUT_FILE}", flush=True)
    if a.resume and OUT_FILE.exists():
        RES.update(json.load(open(OUT_FILE)))
    parts = a.parts.split(",")
    if "gb" in parts:
        cells = set(a.gb_cells.split(",")) if a.gb_cells else None
        run_gb(lambda name: cells is None or name in cells)
    if "e7" in parts:
        run_e7(a.e7_arms.split(","), a.e7_epochs, [int(x) for x in a.e7_seeds.split(",")], a.e7_val_every)
    save_res()
    print("ALL REQUESTED EVALUATIONS COMPLETE", flush=True)
