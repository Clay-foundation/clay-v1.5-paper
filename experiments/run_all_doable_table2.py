#!/usr/bin/env python3
"""Peer-model cells of Table 2 (OlmoEarth v1.3 Base, Panopticon, Copernicus-FM, DOFA-L, DINOv2-L, Clay).

Parts (select with --parts, default all):
  tasks    OlmoEarth / Panopticon / Copernicus-FM on 13 tasks (matched 224x224 linear probes)
  s1       EuroSAT-S1: Copernicus-FM, OlmoEarth (S1), Panopticon, DINOv2-L (3-ch SAR composite)
  togo     CropHarvest-Togo 12-month S1 series: Clay, DOFA-L, Panopticon, Copernicus-FM, DINOv2-L

Input contracts (audit fixes #9, #15, #17, #27); each cell stores its contract under `_contract`:
  * GEO-Bench m-brick-kiln / m-eurosat bands are selected by channel index from the documented true order
    (geobench 1.1.0 known issues; ServiceNow/geo-bench#29); kiln has 9 physical bands (no B06), TCI excluded.
  * Copernicus-FM gets the TRUE footprint (native chip pixels x GSD, before the 224 resize) and the
    per-band bandwidths of the ICCV 2025 supplement (Table 4: S1/S2/S3); proxies for non-S1/S2/S3 sensors are marked.
  * OlmoEarth gets its own `Normalizer(Strategy.COMPUTED)` on S2-native tasks (the train-split z-score arm is kept
    as a labelled diagnostic `olmo_v13_trainz`); Landsat-8 uses the library's band order (B8, B1..B7, B9..B11).
  * CropHarvest-Togo: head / refit policy fixed a priori (C chosen on a train/val split, refit on all training
    labels, test scored once); DOFA gets DOFA's SAR convention (3.75 um, 3.75 um).

Output (audit fix #35): fresh `experiments/runs/table2_<UTC>/all_doable_table2.json` unless `--output-dir`;
`--resume` skips only cells whose producer fingerprint (script sha256 + contract hash) is identical.
"""
import ast, gc, glob, io, json, math, os, pickle, shutil, subprocess, sys, tarfile, time, types, urllib.request, zipfile
from pathlib import Path
from types import SimpleNamespace

import argparse
HERE = Path(__file__).resolve().parents[1]
if "PROJ_DATA" in os.environ and "PROJ_LIB" not in os.environ:
    os.environ["PROJ_LIB"] = os.environ["PROJ_DATA"]

import numpy as np
import pandas as pd
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
from olmoearth_pretrain.datatypes import MaskedOlmoEarthSample, MaskValue
from olmoearth_pretrain.nn.pooling import PoolingType, pool_unmasked_tokens

sys.path.insert(0, str(HERE / "experiments"))
from run_standard_benchmarks import (
    SCRATCH, WORK_DIR, H5_PYTHON, S3_OLCI_WAVES, S2_10B_WAVES, RGB_WAVES, NAIP_WAVES, L8_6B_WAVES, S1_2B_WAVES,
    DOFA_S1_2B_WAVES, KILN_BANDS, EUROSAT_BANDS, gb_channels, gb_waves, S2_WAVE_UM,
    clean_nonfinite, compute_train_band_stats, ModelSuite,
    eval_single_label, eval_multilabel, eval_dense_segmentation, eval_dense_regression, eval_chip_regression,
    load_torchgeo_cls, load_copernicus_s3, load_geobench_splits, olmo_files, _fit_gpu_linear_lbfgs,
    provenance, utc_now, cfg_fingerprint, resolve_output, sha256_file,
)
from torchgeo.datasets import CopernicusBenchEuroSATS1, RESISC45, UCMerced
from torchgeo.models import (
    dofa_large_patch16_224, DOFALarge16_Weights,
    copernicusfm_base, CopernicusFM_Base_Weights,
)
import timm

torch.cuda.set_per_process_memory_fraction(10 / 24.0)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

SCRIPT = str(Path(__file__).resolve())
OUT_FILE: Path | None = None
RES: dict = {}
RESUME = False


def save_res():
    RES["_meta"] = provenance(SCRIPT, {"olmoearth_repo_revision": os.environ.get("CLAY_OLMO_REVISION", "pinned in run_standard_benchmarks.OLMO_REVISION")})
    json.dump(RES, open(OUT_FILE, "w"), indent=2)


def cell_done(tid: str, model: str, contract: dict) -> bool:
    """True only under --resume, when the stored cell came from this exact script and contract."""
    fp = cfg_fingerprint(contract)
    prod = RES.get("_producers", {}).get(f"{tid}/{model}", {})
    return RESUME and model in RES.get(tid, {}) and prod.get("script_sha256") == sha256_file(SCRIPT) and prod.get("cfg") == fp


def put_cell(tid: str, model: str, value, contract: dict):
    RES.setdefault(tid, {})[model] = value
    RES[tid].setdefault("_contract", {})[model] = contract
    RES.setdefault("_producers", {})[f"{tid}/{model}"] = {"script_sha256": sha256_file(SCRIPT), "cfg": cfg_fingerprint(contract), "utc_timestamp": utc_now()}
    save_res()


def load_gb_direct(repo_file, task_name, band_specs, kind):
    """Pinned-revision GEO-Bench loader (shared with run_standard_benchmarks); seg labels as int64."""
    splits = load_geobench_splits(repo_file, task_name, band_specs, kind=kind)
    if kind == "seg":
        splits = {k: (X, y.astype(np.int64)) for k, (X, y) in splits.items()}
    load_gb_direct.last_meta = dict(load_geobench_splits.last_meta)
    return splits


load_gb_direct.last_meta = {}

# Copernicus-FM spectral widths (nm), ICCV 2025 supplement Table 4 (S2 TOA 13 bands, S3 OLCI 21 bands, S1 GRD).
COP_S2_BW = {"B01": 20, "B02": 65, "B03": 35, "B04": 30, "B05": 15, "B06": 15, "B07": 20, "B08": 115,
             "B8A": 20, "B09": 20, "B10": 30, "B11": 90, "B12": 180}
COP_S3_BW = [15, 10, 10, 10, 10, 10, 10, 10, 7.5, 7.5, 10, 7.5, 7.5, 3.75, 2.5, 15, 20, 10, 10, 20, 40]
COP_S1_WAVE, COP_S1_BW = [5.0e7, 5.0e7], [1.0e9, 1.0e9]
S2_SLOT_ORDER = ["B02", "B03", "B04", "B08", "B05", "B06", "B07", "B8A", "B11", "B12", "B01", "B09"]


# Model Loaders
def get_olmo():
    p, w = olmo_files()
    cfg = json.load(open(p))
    ecfg = cfg["model"]["encoder_config"]
    ecfg.pop("attention_pooler_config", None)
    enc = Config.from_dict(ecfg).build().cuda().eval()
    sd = torch.load(w, map_location="cpu", weights_only=False)
    enc_sd = {k[len("encoder."):]: v for k, v in sd.items() if k.startswith("encoder.") and not k.startswith("encoder.attention_pooler")}
    enc.load_state_dict(enc_sd, strict=True)
    return enc

def get_pan():
    torch.hub._validate_not_a_forked_repo = lambda a, b, c: True
    return torch.hub.load("panopticon-FM/panopticon", "panopticon_vitb14").cuda().eval()

def get_cop():
    return copernicusfm_base(weights=CopernicusFM_Base_Weights.CopernicusFM_ViT).cuda().eval()

def get_dinov2():
    m = timm.create_model("vit_large_patch14_reg4_dinov2.lvd142m", pretrained=True, num_classes=0, dynamic_img_size=True).cuda().eval()
    return m

# Feature Extractors
@torch.no_grad()
def ext_pan(pan, X, means, stds, chn_ids_nm, need_pat=False):
    mu_t = torch.tensor(means, dtype=torch.float32, device="cuda").view(1, -1, 1, 1)
    sd_t = torch.tensor(stds, dtype=torch.float32, device="cuda").view(1, -1, 1, 1)
    cls_l, mean_l, pat_l = [], [], []
    bs = 8 if len(chn_ids_nm) > 12 else 24
    for i in range(0, len(X), bs):
        xb = torch.as_tensor(X[i:i+bs], dtype=torch.float32, device="cuda")
        if xb.shape[-2:] != (224, 224):
            xb = F.interpolate(xb, size=(224, 224), mode="bilinear", align_corners=False)
        xb = (xb - mu_t) / sd_t
        B = len(xb)
        chn = torch.tensor([chn_ids_nm] * B, dtype=torch.float32, device="cuda")
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            d = pan.forward_features({"imgs": xb, "chn_ids": chn})
        c_tok = d["x_norm_clstoken"].float().cpu().numpy()
        p_tok = d["x_norm_patchtokens"].float()
        m_tok = p_tok.mean(1).cpu().numpy()
        cls_l.append(c_tok); mean_l.append(m_tok)
        if need_pat:
            pat_l.append(p_tok.reshape(B, 16, 16, 768).half().cpu().numpy())
    out = {"cls": np.concatenate(cls_l, 0), "mean": np.concatenate(mean_l, 0)}
    if need_pat: out["patches"] = np.concatenate(pat_l, 0)
    return out

@torch.no_grad()
def ext_cop(cop, X, means, stds, waves_nm, bws_nm, area_km2, need_pat=False):
    """Copernicus-FM features. `area_km2` is the TRUE chip footprint (native pixels x GSD, before the 224 resize)."""
    from torchgeo.models.copernicusfm import resize_abs_pos_embed
    mu_t = torch.tensor(means, dtype=torch.float32, device="cuda").view(1, -1, 1, 1)
    sd_t = torch.tensor(stds, dtype=torch.float32, device="cuda").view(1, -1, 1, 1)
    wvs = torch.tensor(waves_nm, dtype=torch.float32, device="cuda")
    bws = torch.tensor(bws_nm, dtype=torch.float32, device="cuda")
    area_km2 = max(1e-6, float(area_km2))
    cls_l, mean_l, pat_l = [], [], []
    bs = 16 if len(waves_nm) > 12 else 32
    for i in range(0, len(X), bs):
        xb = torch.as_tensor(X[i:i+bs], dtype=torch.float32, device="cuda")
        if xb.shape[-2:] != (224, 224):
            xb = F.interpolate(xb, size=(224, 224), mode="bilinear", align_corners=False)
        xb = (xb - mu_t) / sd_t
        B = len(xb)
        meta = torch.tensor([[float("nan"), float("nan"), float("nan"), area_km2]] * B, dtype=torch.float32, device="cuda")
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            h = cop.patch_embed_spectral(xb, wavelengths=wvs, bandwidths=bws, kernel_size=None)
            n_sqrt = int(math.sqrt(h.size(1)))
            n_orig = int(math.sqrt(cop.num_patches))
            pe = resize_abs_pos_embed(cop.pos_embed, n_sqrt, (n_orig, n_orig), num_prefix_tokens=1)
            coord_embed = cop.coord_fc(cop.coord_token)
            area_embed = cop.scale_fc(cop.get_area_pos_embed(meta[:, 3], pe.shape[-1]))
            time_embed = cop.time_fc(cop.time_token)
            pe = pe + coord_embed + area_embed + time_embed
            h = h + pe[:, 1:, :]
            cls_tokens = (cop.cls_token + pe[:, :1, :]).expand(B, -1, -1)
            h = torch.cat((cls_tokens, h), dim=1)
            for blk in cop.blocks:
                h = blk(h)
            norm_fn = getattr(cop, "norm", None) or cop.fc_norm
            h = norm_fn(h).float()
        cls_l.append(h[:, 0].cpu().numpy())
        mean_l.append(h[:, 1:].mean(1).cpu().numpy())
        if need_pat:
            pat_l.append(h[:, 1:].reshape(B, 14, 14, 768).half().cpu().numpy())
    out = {"cls": np.concatenate(cls_l, 0), "mean": np.concatenate(mean_l, 0)}
    if need_pat: out["patches"] = np.concatenate(pat_l, 0)
    return out

_OLMO_NORM = Normalizer(Strategy.COMPUTED)


def olmo_affine(modality):
    """Per-band (slope, offset) of Olmo's own COMPUTED normaliser, x_norm = x * slope + offset (library is the source)."""
    n = len(modality.band_order)
    lo = np.asarray(_OLMO_NORM.normalize(modality, np.zeros(n)), dtype=np.float32)
    hi = np.asarray(_OLMO_NORM.normalize(modality, np.ones(n)), dtype=np.float32)
    return hi - lo, lo


@torch.no_grad()
def ext_olmo(olmo, X, means, stds, chan_map, modality_name="sentinel2_l2a", norm_scale=None, need_pat=False):
    """OlmoEarth v1.3 Base features.

    chan_map: list of (source channel index, OlmoEarth band name) in the modality's band_order naming
              (S2: B02,B03,B04,B08,B05,B06,B07,B8A,B11,B12,B01,B09; Landsat: B8,B1..B7,B9,B10,B11).
    norm_scale: None -> train-split per-band z-score (legacy; used where the input is not S2/L8 DN);
                float -> multiply raw values by it (to DN), then Olmo's own Normalizer(COMPUTED).
    Band sets with no mapped band are masked MISSING; bands of a present set without data are 0 after normalisation.
    """
    modality = Modality.SENTINEL2_L2A if modality_name == "sentinel2_l2a" else Modality.LANDSAT
    order = modality.band_order
    nb = len(order)
    mu_t = torch.tensor(means, dtype=torch.float32, device="cuda").view(1, -1, 1, 1)
    sd_t = torch.tensor(stds, dtype=torch.float32, device="cuda").view(1, -1, 1, 1)
    slope, off = olmo_affine(modality)
    slope_t, off_t = torch.tensor(slope, device="cuda"), torch.tensor(off, device="cuda")
    src_idx = [c for c, _ in chan_map]
    slots = [order.index(b) for _, b in chan_map]
    present = [any(b in {bb for _, bb in chan_map} for b in bs.bands) for bs in modality.band_sets]
    nbs = len(modality.band_sets)
    if modality_name == "landsat":
        # OlmoEarth v1.3 Base tokenises Landsat as ONE band group holding all 11 bands in library order
        # (config.json tokenization_config.overrides.landsat); the mask column that gates tokens is that group,
        # not the library's default [B8 | B1..B11] split, so the group is present whenever any band is supplied.
        present = [True] * nbs
    mean_l, pat_l = [], []
    hw, p_sz, bs = 224, 8, 24
    for i in range(0, len(X), bs):
        xb = torch.as_tensor(X[i:i+bs], dtype=torch.float32, device="cuda")
        if xb.shape[-2:] != (hw, hw):
            xb = F.interpolate(xb, size=(hw, hw), mode="bilinear", align_corners=False)
        B = len(xb)
        if norm_scale is None:
            xn = (xb - mu_t) / sd_t
            xsel = xn[:, src_idx]
        else:
            xs = xb[:, src_idx] * float(norm_scale)
            xsel = xs * slope_t[slots].view(1, -1, 1, 1) + off_t[slots].view(1, -1, 1, 1)
        arr = torch.zeros((B, hw, hw, 1, nb), dtype=torch.float32, device="cuda")
        arr[..., slots] = xsel.permute(0, 2, 3, 1).unsqueeze(3)
        mask = torch.zeros((B, hw, hw, 1, nbs), dtype=torch.int64, device="cuda")
        for k, pr in enumerate(present):
            if not pr:
                mask[..., k] = MaskValue.MISSING.value
        ts = torch.tensor([[[15, 5, 2020]]] * B, dtype=torch.int64, device="cuda")
        if modality_name == "sentinel2_l2a":
            samp = MaskedOlmoEarthSample(timestamps=ts, sentinel2_l2a=arr, sentinel2_l2a_mask=mask)
        else:
            samp = MaskedOlmoEarthSample(timestamps=ts, landsat=arr, landsat_mask=mask)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            out_o = olmo(samp, patch_size=p_sz, fast_pass=True)["tokens_and_masks"]
            vec = pool_unmasked_tokens(out_o, PoolingType.MEAN, spatial_pooling=False).float()
            mean_l.append(vec.cpu().numpy())
            if need_pat:
                grid = pool_unmasked_tokens(out_o, PoolingType.MEAN, spatial_pooling=True).float()
                pat_l.append(grid.half().cpu().numpy())
    out = {"mean": np.concatenate(mean_l, 0)}
    if need_pat:
        out["patches"] = np.concatenate(pat_l, 0)
    return out


@torch.no_grad()
def ext_olmo_s1(olmo, X, batch_size=64):
    """EuroSAT-S1 (VV, VH in dB) through Olmo's own S1 normaliser; 64 px native, patch size 4."""
    N, C, H, W = X.shape
    slope, off = olmo_affine(Modality.SENTINEL1)
    s1 = np.transpose(X, (0, 2, 3, 1))[..., None, :].astype(np.float32) * slope + off
    mean_l = []
    for b0 in range(0, N, batch_size):
        b1 = min(N, b0 + batch_size)
        B = b1 - b0
        ts_t = torch.tensor([[15, 6, 2020]] * B, dtype=torch.int64, device="cuda")[:, None, :]
        mask_t = torch.zeros((B, H, W, 1, 1), dtype=torch.int64, device="cuda")
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            samp = MaskedOlmoEarthSample(timestamps=ts_t, sentinel1=torch.from_numpy(s1[b0:b1]).cuda(), sentinel1_mask=mask_t)
            out = olmo(samp, patch_size=4, fast_pass=True)["tokens_and_masks"]
            mean_l.append(pool_unmasked_tokens(out, PoolingType.MEAN, spatial_pooling=False).float().cpu().numpy())
    return {"mean": np.concatenate(mean_l, 0)}


def score_task_dict(kind, tr_d, y_tr, va_d, y_va, te_d, y_te, n_classes=None):
    if kind == "cls":
        best_oa, best_res = -1.0, None
        for ro in ("cls", "mean"):
            if ro in tr_d:
                r = eval_single_label(tr_d[ro], y_tr, va_d[ro], y_va, te_d[ro], y_te)
                if r["val_oa"] > best_oa:
                    best_oa = r["val_oa"]; best_res = r
        return best_res
    elif kind == "multilabel":
        best_map, best_res = -1.0, None
        for ro in ("cls", "mean"):
            if ro in tr_d:
                r = eval_multilabel(tr_d[ro], y_tr, va_d[ro], y_va, te_d[ro], y_te)
                if r["val_macro_map"] > best_map:
                    best_map = r["val_macro_map"]; best_res = r
        return best_res
    elif kind == "multilabel_f1":
        out = {}
        for ro in ("cls", "mean"):
            if ro in tr_d:
                scaler = StandardScaler().fit(tr_d[ro])
                Xtr = torch.tensor(scaler.transform(tr_d[ro]), dtype=torch.float32, device="cuda")
                Xva = torch.tensor(scaler.transform(va_d[ro]), dtype=torch.float32, device="cuda")
                Xte = torch.tensor(scaler.transform(te_d[ro]), dtype=torch.float32, device="cuda")
                Ytr_t = torch.tensor(y_tr, dtype=torch.float32, device="cuda")
                best_f1, best_te = -1.0, -1.0
                for c in (0.01, 0.1, 1.0, 10.0):
                    W, b = _fit_gpu_linear_lbfgs(Xtr, Ytr_t, y_tr.shape[1], 0.5 / (c * len(Xtr)), mode="bce")
                    with torch.no_grad():
                        p_va = (torch.sigmoid(Xva @ W.t() + b) > 0.5).cpu().numpy().astype(int)
                        va_f1 = float(f1_score(y_va.astype(int), p_va, average="micro"))
                        if va_f1 > best_f1:
                            best_f1 = va_f1
                            p_te = (torch.sigmoid(Xte @ W.t() + b) > 0.5).cpu().numpy().astype(int)
                            best_te = float(f1_score(y_te.astype(int), p_te, average="micro"))
                out[ro] = round(best_te * 100.0, 2)
        return out
    elif kind == "seg":
        return eval_dense_segmentation(tr_d["patches"], y_tr, va_d["patches"], y_va, te_d["patches"], y_te, n_classes)
    elif kind == "biomass":
        dense_m = eval_dense_regression(tr_d["patches"], y_tr, va_d["patches"], y_va, te_d["patches"], y_te)
        y_tr_m = y_tr.reshape(len(y_tr), -1).mean(1)
        y_va_m = y_va.reshape(len(y_va), -1).mean(1)
        y_te_m = y_te.reshape(len(y_te), -1).mean(1)
        ro = "mean" if "mean" in tr_d else "cls"
        chip_m = eval_chip_regression(tr_d[ro], y_tr_m, va_d[ro], y_va_m, te_d[ro], y_te_m)
        return {"dense": dense_m, "chip": chip_m}


# ----------------------------------------------------------------------------------------------
# Task table. chan_map for OlmoEarth: (source channel, Olmo band). cop_bw: Copernicus-FM bandwidths (nm).
# ----------------------------------------------------------------------------------------------
def _s2(names):  # (source idx, S2 band) for a list of physical band names in source order
    return [(i, b) for i, b in enumerate(names)]


S2_STD10 = ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"]
bands_s2_std = [
    "02 - Blue", "03 - Green", "04 - Red",
    "05 - Vegetation Red Edge", "06 - Vegetation Red Edge", "07 - Vegetation Red Edge",
    "08 - NIR", "08A - Vegetation Red Edge", "11 - SWIR", "12 - SWIR",
]
bands_s2_wild = [b + "_*" for b in bands_s2_std]
bands_l8 = ["04 - Red_*", "03 - Green_*", "02 - Blue_*", "05 - NIR_*", "06 - SWIR1_*", "07 - SWIR2_*"]
OLCI_TO_S2 = [(3, "B02"), (5, "B03"), (7, "B04"), (16, "B08"), (10, "B05"), (11, "B06"), (15, "B07"), (17, "B8A"), (1, "B01"), (19, "B09")]
# Proxy widths (S2 band at the same wavelength) where the supplement lists no widths for the sensor (aerial RGB/NIR, Landsat-8).
RGB_BW = [COP_S2_BW["B04"], COP_S2_BW["B03"], COP_S2_BW["B02"]]
NAIP_BW = RGB_BW + [COP_S2_BW["B08"]]
L8_BW = [COP_S2_BW[b] for b in ("B04", "B03", "B02", "B8A", "B11", "B12")]

TASKS = [
    dict(tid="m-pv4ger-seg", src="geobench", arg=("segmentation_v1.0/m-pv4ger-seg.zip", ["Red", "Green", "Blue"]), kind="seg", n_cls=2,
         waves=RGB_WAVES, gsd=0.1, olmo=dict(mod="sentinel2_l2a", map=[(0, "B04"), (1, "B03"), (2, "B02")], scale=None), cop_bw=RGB_BW),
    dict(tid="m-chesapeake", src="geobench", arg=("segmentation_v1.0/m-chesapeake.zip", ["Red", "Green", "Blue", "NearInfrared"]), kind="seg", n_cls=7,
         waves=NAIP_WAVES, gsd=1.0, olmo=dict(mod="sentinel2_l2a", map=[(0, "B04"), (1, "B03"), (2, "B02"), (3, "B08")], scale=None), cop_bw=NAIP_BW),
    dict(tid="biomass-s3", src="s3", arg=True, kind="biomass", n_cls=None, waves=S3_OLCI_WAVES, gsd=300.0, native_hw=94,
         olmo=dict(mod="sentinel2_l2a", map=OLCI_TO_S2, scale=None), cop_bw=COP_S3_BW),
    dict(tid="lc100cls-s3", src="s3", arg=False, kind="multilabel", n_cls=None, waves=S3_OLCI_WAVES, gsd=300.0, native_hw=94,
         olmo=dict(mod="sentinel2_l2a", map=OLCI_TO_S2, scale=None), cop_bw=COP_S3_BW),
    dict(tid="m-pv4ger", src="geobench", arg=("classification_v1.0/m-pv4ger.zip", ["Red", "Green", "Blue"]), kind="cls", n_cls=None,
         waves=RGB_WAVES, gsd=0.1, olmo=dict(mod="sentinel2_l2a", map=[(0, "B04"), (1, "B03"), (2, "B02")], scale=None), cop_bw=RGB_BW),
    dict(tid="ucmerced", src="torchgeo", arg=(UCMerced, 256), kind="cls", n_cls=None, waves=RGB_WAVES, gsd=0.3,
         olmo=dict(mod="sentinel2_l2a", map=[(0, "B04"), (1, "B03"), (2, "B02")], scale=None), cop_bw=RGB_BW),
    dict(tid="resisc45", src="torchgeo", arg=(RESISC45, 256), kind="cls", n_cls=None, waves=RGB_WAVES, gsd=1.0,
         olmo=dict(mod="sentinel2_l2a", map=[(0, "B04"), (1, "B03"), (2, "B02")], scale=None), cop_bw=RGB_BW),
    dict(tid="m-brick-kiln", src="geobench", arg=("classification_v1.0/m-brick-kiln.zip", gb_channels("m-brick-kiln", KILN_BANDS)), kind="cls", n_cls=None,
         band_names=KILN_BANDS, waves=gb_waves(KILN_BANDS), gsd=10.0,
         olmo=dict(mod="sentinel2_l2a", map=_s2(KILN_BANDS), scale=1.0), cop_bw=[COP_S2_BW[b] for b in KILN_BANDS]),
    dict(tid="m-eurosat", src="geobench", arg=("classification_v1.0/m-eurosat.zip", gb_channels("m-eurosat", EUROSAT_BANDS)), kind="cls", n_cls=None,
         band_names=EUROSAT_BANDS, waves=gb_waves(EUROSAT_BANDS), gsd=10.0,
         olmo=dict(mod="sentinel2_l2a", map=_s2(EUROSAT_BANDS), scale=1.0), cop_bw=[COP_S2_BW[b] for b in EUROSAT_BANDS]),
    dict(tid="m-so2sat", src="geobench", arg=("classification_v1.0/m-so2sat.zip", bands_s2_std), kind="cls", n_cls=None,
         band_names=S2_STD10, waves=S2_10B_WAVES, gsd=10.0,
         olmo=dict(mod="sentinel2_l2a", map=_s2(S2_STD10), scale=1.0e4), cop_bw=[COP_S2_BW[b] for b in S2_STD10]),  # so2sat is reflectance in [0,1]
    dict(tid="m-forestnet", src="geobench", arg=("classification_v1.0/m-forestnet.zip", bands_l8), kind="cls", n_cls=None,
         waves=L8_6B_WAVES, gsd=15.0,
         olmo=dict(mod="landsat", map=[(0, "B4"), (1, "B3"), (2, "B2"), (3, "B5"), (4, "B6"), (5, "B7")], scale=None), cop_bw=L8_BW),
    dict(tid="m-bigearthnet", src="geobench", arg=("classification_v1.0/m-bigearthnet.zip", bands_s2_std), kind="multilabel_f1", n_cls=None,
         band_names=S2_STD10, waves=S2_10B_WAVES, gsd=10.0,
         olmo=dict(mod="sentinel2_l2a", map=_s2(S2_STD10), scale=1.0), cop_bw=[COP_S2_BW[b] for b in S2_STD10]),
    dict(tid="m-cashew-plant", src="geobench", arg=("segmentation_v1.0/m-cashew-plant.zip", bands_s2_wild), kind="seg", n_cls=7,
         band_names=S2_STD10, waves=S2_10B_WAVES, gsd=10.0,
         olmo=dict(mod="sentinel2_l2a", map=_s2(S2_STD10), scale=1.0), cop_bw=[COP_S2_BW[b] for b in S2_STD10]),
]


def _load_task(t):
    kind_loader = "seg" if t["kind"] == "seg" else ("multilabel" if "multilabel" in t["kind"] else "cls")
    if t["src"] == "geobench":
        rfile, bspec = t["arg"]
        sp = load_gb_direct(rfile, t["tid"], bspec, kind_loader)
        return sp, dict(load_gb_direct.last_meta)
    if t["src"] == "torchgeo":
        cls_t, rhw = t["arg"]
        return load_torchgeo_cls(cls_t, resize_hw=rhw), {}
    return load_copernicus_s3(is_biomass=t["arg"]), {}


def run_tasks(only):
    for t in TASKS:
        tid = t["tid"]
        if only and tid not in only:
            continue
        # contracts per model (hashing them drives --resume)
        c_olmo = dict(model="olmo_v13", bands_map=t["olmo"]["map"], modality=t["olmo"]["mod"], normaliser=("olmo_COMPUTED_x%s" % t["olmo"]["scale"]) if t["olmo"]["scale"] else "train_zscore", olmoearth_revision=os.environ.get("CLAY_OLMO_REVISION", "pinned"), waves_um=t["waves"])
        c_olmo_z = dict(c_olmo, model="olmo_v13_trainz", normaliser="train_zscore")
        c_pan = dict(model="panopticon", waves_nm=[w * 1000.0 for w in t["waves"]], normaliser="train_zscore")
        c_cop = dict(model="copernicus_fm", waves_nm=[w * 1000.0 for w in t["waves"]], bandwidths_nm=t["cop_bw"], gsd_m=t["gsd"], normaliser="train_zscore", footprint="native_hw*gsd (before 224 resize)")
        todo = [("olmo_v13", c_olmo), ("panopticon", c_pan), ("copernicus_fm", c_cop)]
        if t["olmo"]["scale"] is not None:
            todo.append(("olmo_v13_trainz", c_olmo_z))
        if all(cell_done(tid, m, c) for m, c in todo):
            print(f"[resume-skip] {tid}", flush=True)
            continue
        t0 = time.time()
        print(f"=== Evaluating {tid} (OlmoEarth v1.3, Panopticon, Copernicus-FM) ===", flush=True)
        splits, gb_meta = _load_task(t)
        X_tr, y_tr = splits["train"]; X_va, y_va = splits["val"]; X_te, y_te = splits["test"]
        del splits
        means, stds, _, _ = compute_train_band_stats(X_tr)
        X_tr = clean_nonfinite(X_tr, means); X_va = clean_nonfinite(X_va, means); X_te = clean_nonfinite(X_te, means)
        kind = t["kind"]; need_pat = kind in ("seg", "biomass"); n_cls = t["n_cls"]
        native_hw = t.get("native_hw", X_tr.shape[-1])
        area_km2 = (native_hw * float(t["gsd"]) / 1000.0) ** 2
        for c in (c_olmo, c_olmo_z, c_pan):
            c["native_hw"] = native_hw
        c_cop.update(native_hw=native_hw, area_km2=area_km2)
        if gb_meta:
            for c in (c_olmo, c_olmo_z, c_pan, c_cop):
                c["input_check"] = gb_meta
        RES.setdefault(tid, {})

        def olmo_cell(model, contract, scale):
            if cell_done(tid, model, contract):
                return
            olmo = get_olmo()
            kw = dict(chan_map=t["olmo"]["map"], modality_name=t["olmo"]["mod"], norm_scale=scale, need_pat=need_pat)
            o = [ext_olmo(olmo, X, means, stds, **kw) for X in (X_tr, X_va, X_te)]
            del olmo; torch.cuda.empty_cache()
            put_cell(tid, model, score_task_dict(kind, o[0], y_tr, o[1], y_va, o[2], y_te, n_cls), contract)

        olmo_cell("olmo_v13", c_olmo, t["olmo"]["scale"])
        if t["olmo"]["scale"] is not None:
            olmo_cell("olmo_v13_trainz", c_olmo_z, None)

        waves_nm = [w * 1000.0 for w in t["waves"]]
        if not cell_done(tid, "panopticon", c_pan):
            pan = get_pan()
            p = [ext_pan(pan, X, means, stds, waves_nm, need_pat) for X in (X_tr, X_va, X_te)]
            del pan; torch.cuda.empty_cache()
            put_cell(tid, "panopticon", score_task_dict(kind, p[0], y_tr, p[1], y_va, p[2], y_te, n_cls), c_pan)
        if not cell_done(tid, "copernicus_fm", c_cop):
            cop = get_cop()
            c = [ext_cop(cop, X, means, stds, waves_nm, t["cop_bw"], area_km2, need_pat) for X in (X_tr, X_va, X_te)]
            del cop; torch.cuda.empty_cache()
            put_cell(tid, "copernicus_fm", score_task_dict(kind, c[0], y_tr, c[1], y_va, c[2], y_te, n_cls), c_cop)
        print(f"Done {tid} in {time.time()-t0:.1f}s:", {k: v for k, v in RES[tid].items() if not k.startswith("_")}, flush=True)
        del X_tr, X_va, X_te; gc.collect()


def run_eurosat_s1():
    from torchgeo.datasets import CopernicusBenchEuroSATS1
    tid = "eurosat-s1"
    native = 64
    area = (native * 10.0 / 1000.0) ** 2
    c_cop = dict(model="copernicus_fm", waves_nm=COP_S1_WAVE, bandwidths_nm=COP_S1_BW, native_hw=native, area_km2=area, normaliser="train_zscore")
    c_olmo = dict(model="olmo_v13", modality="sentinel1", normaliser="olmo_COMPUTED(S1 dB)", patch_size=4, native_hw=native)
    c_pan = dict(model="panopticon", chn_ids=[-1.0, -2.0], normaliser="train_zscore")
    c_dv2 = dict(model="dinov2", input="[VV,VH,(VV+VH)/2] train-zscore")
    cells = [("copernicus_fm", c_cop), ("olmo_v13", c_olmo), ("panopticon", c_pan), ("dinov2", c_dv2)]
    if all(cell_done(tid, m, c) for m, c in cells):
        return
    print("=== EuroSAT-S1 (Copernicus-FM, OlmoEarth, Panopticon, DINOv2-L) ===", flush=True)
    splits = load_torchgeo_cls(CopernicusBenchEuroSATS1, resize_hw=64)
    X_tr, y_tr = splits["train"]; X_va, y_va = splits["val"]; X_te, y_te = splits["test"]
    means, stds, _, _ = compute_train_band_stats(X_tr)
    X_tr = clean_nonfinite(X_tr, means); X_va = clean_nonfinite(X_va, means); X_te = clean_nonfinite(X_te, means)
    RES.setdefault(tid, {})
    if not cell_done(tid, "copernicus_fm", c_cop):
        cop = get_cop()
        c = [ext_cop(cop, X, means, stds, COP_S1_WAVE, COP_S1_BW, area, False) for X in (X_tr, X_va, X_te)]
        del cop; torch.cuda.empty_cache()
        put_cell(tid, "copernicus_fm", score_task_dict("cls", c[0], y_tr, c[1], y_va, c[2], y_te), c_cop)
    if not cell_done(tid, "olmo_v13", c_olmo):
        olmo = get_olmo()
        o = [ext_olmo_s1(olmo, X) for X in (X_tr, X_va, X_te)]
        del olmo; torch.cuda.empty_cache()
        put_cell(tid, "olmo_v13", score_task_dict("cls", o[0], y_tr, o[1], y_va, o[2], y_te), c_olmo)
    if not cell_done(tid, "panopticon", c_pan):
        pan = get_pan()
        p = [ext_pan(pan, X, means, stds, [-1.0, -2.0], False) for X in (X_tr, X_va, X_te)]
        del pan; torch.cuda.empty_cache()
        put_cell(tid, "panopticon", score_task_dict("cls", p[0], y_tr, p[1], y_va, p[2], y_te), c_pan)
    if not cell_done(tid, "dinov2", c_dv2):
        def make_sar3(X):
            return np.concatenate([X, 0.5 * (X[:, 0:1] + X[:, 1:2])], axis=1)
        Xtr3, Xva3, Xte3 = make_sar3(X_tr), make_sar3(X_va), make_sar3(X_te)
        mu3, sd3, _, _ = compute_train_band_stats(Xtr3)
        dv2 = get_dinov2()
        @torch.no_grad()
        def ext_dv2_sar(X):
            mu_t = torch.tensor(mu3, dtype=torch.float32, device="cuda").view(1, 3, 1, 1)
            sd_t = torch.tensor(sd3, dtype=torch.float32, device="cuda").view(1, 3, 1, 1)
            cls_l, mean_l = [], []
            for i in range(0, len(X), 64):
                xb = torch.as_tensor(X[i:i+64], dtype=torch.float32, device="cuda")
                xb = F.interpolate(xb, size=(224, 224), mode="bilinear", align_corners=False)
                xb = (xb - mu_t) / sd_t
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    f = dv2.forward_features(xb)
                cls_l.append(f[:, 0].float().cpu().numpy())
                mean_l.append(f[:, 5:].mean(1).float().cpu().numpy())
            return {"cls": np.concatenate(cls_l, 0), "mean": np.concatenate(mean_l, 0)}
        put_cell(tid, "dinov2", score_task_dict("cls", ext_dv2_sar(Xtr3), y_tr, ext_dv2_sar(Xva3), y_va, ext_dv2_sar(Xte3), y_te), c_dv2)
        del dv2; torch.cuda.empty_cache()
    print("EuroSAT-S1 done:", {k: v for k, v in RES[tid].items() if not k.startswith("_")}, flush=True)


# ----------------------------------------------------------------------------------------------
# CropHarvest-Togo (12-month S1 series). Head / refit policy is FIXED a priori and uses train/val only.
# ----------------------------------------------------------------------------------------------
TOGO_ZENODO = "https://zenodo.org/records/7257688/files"
TOGO_COP_AREA_KM2 = max(1e-3, (1 * 10.0 / 1000.0) ** 2)   # Copernicus-FM area embedding is defined on [1e-3, 5.1e8] km2


def load_togo():
    ch_dir = SCRATCH / "cropharvest_cache"     # persistent download cache (not the per-process work dir)
    ch_dir.mkdir(parents=True, exist_ok=True)
    feat_tar, lbl_json = ch_dir / "features.tar.gz", ch_dir / "labels.geojson"
    if not lbl_json.exists():
        urllib.request.urlretrieve(f"{TOGO_ZENODO}/labels.geojson?download=1", str(lbl_json))
    if not feat_tar.exists():
        urllib.request.urlretrieve(f"{TOGO_ZENODO}/features.tar.gz?download=1", str(feat_tar))
    df = pd.json_normalize(json.loads(lbl_json.read_text())["features"])
    togo_tr = df[df["properties.dataset"] == "togo"].reset_index(drop=True)
    togo_te = df[df["properties.dataset"] == "togo-eval"].reset_index(drop=True)
    need_tr = {f"{int(r['properties.index'])}_togo.h5": int(r["properties.is_crop"]) for _, r in togo_tr.iterrows()}
    need_te = {f"{int(r['properties.index'])}_togo-eval.h5": int(r["properties.is_crop"]) for _, r in togo_te.iterrows()}
    work = WORK_DIR / "cropharvest"
    work.mkdir(parents=True, exist_ok=True)
    helper = work / "_extract_togo.py"
    helper.write_text("""import io, json, sys, tarfile, h5py, numpy as np
from pathlib import Path
work = Path(sys.argv[1]); tar = sys.argv[2]
need_tr = json.loads((work / "need_tr.json").read_text())
need_te = json.loads((work / "need_te.json").read_text())
X_tr, y_tr, X_te, y_te = [], [], [], []
with tarfile.open(tar, "r:gz") as tf:
    for m in tf.getmembers():
        bname = Path(m.name).name
        if bname in need_tr or bname in need_te:
            with h5py.File(io.BytesIO(tf.extractfile(m).read()), "r") as hf:
                arr = hf["array"][:].astype(np.float32)  # (12, 18)
            if bname in need_tr:
                X_tr.append(arr); y_tr.append(need_tr[bname])
            else:
                X_te.append(arr); y_te.append(need_te[bname])
np.save(work / "X_tr.npy", np.stack(X_tr, 0)); np.save(work / "y_tr.npy", np.array(y_tr, dtype=np.int64))
np.save(work / "X_te.npy", np.stack(X_te, 0)); np.save(work / "y_te.npy", np.array(y_te, dtype=np.int64))
""")
    (work / "need_tr.json").write_text(json.dumps(need_tr))
    (work / "need_te.json").write_text(json.dumps(need_te))
    subprocess.run([H5_PYTHON, str(helper), str(work), str(feat_tar)], check=True)
    out = (np.load(work / "X_tr.npy"), np.load(work / "y_tr.npy"), np.load(work / "X_te.npy"), np.load(work / "y_te.npy"))
    shutil.rmtree(work)
    ident = {"labels_sha256": sha256_file(str(lbl_json)), "features_sha256": sha256_file(str(feat_tar)), "zenodo_record": 7257688}
    return out, ident


def togo_eval(Z_tr_all, y_tr_all, Z_te, y_te, tr_idx, va_idx):
    """Frozen recipe, decided before any test score:
       1) scaler on the train split; C in {0.01,0.1,1,10} chosen by validation accuracy (LR, L2);
       2) refit on ALL training labels with that C (the stated full-training refit);
       3) the test set is scored ONCE with that refit model -> `oa` (primary).
    Diagnostics (not selectable, reported for the audit record): the split-fit model's test OA, and an LSVM with
    its own val-chosen C. Each is a single frozen prediction; nothing is selected on test labels."""
    scaler = StandardScaler().fit(Z_tr_all[tr_idx])
    Xs = torch.tensor(scaler.transform(Z_tr_all[tr_idx]), dtype=torch.float32, device="cuda")
    Xv = torch.tensor(scaler.transform(Z_tr_all[va_idx]), dtype=torch.float32, device="cuda")
    ys = torch.tensor(y_tr_all[tr_idx], dtype=torch.long, device="cuda")
    K = int(max(y_tr_all.max(), y_te.max())) + 1
    best_c, best_va, best_wb = 1.0, -1.0, None
    for c in (0.01, 0.1, 1.0, 10.0):
        W, b = _fit_gpu_linear_lbfgs(Xs, ys, K, 0.5 / (c * len(Xs)), mode="ce")
        va = float(accuracy_score(y_tr_all[va_idx], (Xv @ W.t() + b).argmax(1).cpu().numpy()))
        if va > best_va:
            best_c, best_va, best_wb = c, va, (W, b)
    # split-fit diagnostic prediction (scored once)
    Xt_s = torch.tensor(scaler.transform(Z_te), dtype=torch.float32, device="cuda")
    oa_split = float(accuracy_score(y_te, (Xt_s @ best_wb[0].t() + best_wb[1]).argmax(1).cpu().numpy()))
    # primary: refit on all training labels
    sc_all = StandardScaler().fit(Z_tr_all)
    Xa = torch.tensor(sc_all.transform(Z_tr_all), dtype=torch.float32, device="cuda")
    ya = torch.tensor(y_tr_all, dtype=torch.long, device="cuda")
    W, b = _fit_gpu_linear_lbfgs(Xa, ya, K, 0.5 / (best_c * len(Xa)), mode="ce")
    Xt = torch.tensor(sc_all.transform(Z_te), dtype=torch.float32, device="cuda")
    oa = float(accuracy_score(y_te, (Xt @ W.t() + b).argmax(1).cpu().numpy()))
    # LSVM diagnostic with its own val-selected C, refit on all training labels
    Ypm = torch.full((len(Xs), K), -1.0, device="cuda"); Ypm.scatter_(1, ys.unsqueeze(1), 1.0)
    bc, bv = 0.1, -1.0
    for c in (0.001, 0.01, 0.1, 1.0):
        W2, b2 = _fit_gpu_linear_lbfgs(Xs, Ypm, K, 0.5 / (c * len(Xs)), mode="sq_hinge")
        va = float(accuracy_score(y_tr_all[va_idx], (Xv @ W2.t() + b2).argmax(1).cpu().numpy()))
        if va > bv:
            bc, bv = c, va
    Ya = torch.full((len(Xa), K), -1.0, device="cuda"); Ya.scatter_(1, ya.unsqueeze(1), 1.0)
    W3, b3 = _fit_gpu_linear_lbfgs(Xa, Ya, K, 0.5 / (bc * len(Xa)), mode="sq_hinge")
    lsvm = float(accuracy_score(y_te, (Xt @ W3.t() + b3).argmax(1).cpu().numpy()))
    return {"oa": round(oa * 100.0, 2), "val_oa": round(best_va * 100.0, 2), "best_C": best_c,
            "diag_split_fit_oa": round(oa_split * 100.0, 2), "diag_lsvm_refit_oa": round(lsvm * 100.0, 2), "lsvm_best_C": bc}


def run_togo():
    tid = "cropharvest_togo"
    base = dict(split_seed=42, val_fraction=0.15, c_grid=[0.01, 0.1, 1.0, 10.0], head="LR(L2) val-chosen C, refit on all train, test scored once",
                normaliser="S1 z-score (train mean/std over months and chips)", pooling="mean/std/max over 12 months of the per-month embedding")
    cells = {
        "clay": dict(base, model="clay_v15", waves_um=S1_2B_WAVES, gsd=10.0, readout="CLS", image="1x1 pixel replicated to 224x224"),
        "dofa_s1": dict(base, model="dofa_l", waves_um=DOFA_S1_2B_WAVES, readout="fc_norm(CLS)", image="1x1 pixel replicated to 224x224"),
        "dofa_s1_clay_waves_diag": dict(base, model="dofa_l", waves_um=S1_2B_WAVES, readout="fc_norm(CLS)", note="DIAGNOSTIC: original (Clay) SAR wavelength convention"),
        "panopticon_s1": dict(base, model="panopticon", chn_ids=[-1.0, -2.0]),
        "copernicus_fm_s1": dict(base, model="copernicus_fm", waves_nm=COP_S1_WAVE, bandwidths_nm=COP_S1_BW, area_km2=TOGO_COP_AREA_KM2, note="true pixel footprint is 10 m x 10 m = 1e-4 km2, below the model's supported area range [1e-3, 5.1e8] km2 (torchgeo FourierExpansion assert); clamped to the lower bound 1e-3 km2"),
        "dinov2_s1": dict(base, model="dinov2", input="[VV,VH,(VV+VH)/2]"),
    }
    if all(cell_done(tid, m, c) for m, c in cells.items()):
        return
    print("=== CropHarvest-Togo ===", flush=True)
    (X_tr_raw, y_tr_all, X_te_raw, y_te), ident = load_togo()
    for c in cells.values():
        c["data_identity"] = ident
        c["n_train"], c["n_test"] = int(len(y_tr_all)), int(len(y_te))
    s1_tr = X_tr_raw[:, :, :2]; s1_te = X_te_raw[:, :, :2]
    mu = s1_tr.mean(axis=(0, 1), keepdims=True); sd = s1_tr.std(axis=(0, 1), keepdims=True) + 1e-6
    s1_tr_z = (s1_tr - mu) / sd; s1_te_z = (s1_te - mu) / sd
    rng = np.random.RandomState(42)
    perm = rng.permutation(len(y_tr_all)); n_va = int(0.15 * len(y_tr_all))
    va_idx, tr_idx = perm[:n_va], perm[n_va:]
    RES.setdefault(tid, {})

    def agg(E):
        return np.concatenate([E.mean(1), E.std(1), E.max(1)], axis=1)

    def run_cell(name, embed_fn):
        if cell_done(tid, name, cells[name]):
            return
        Z_tr, Z_te = embed_fn(s1_tr_z), embed_fn(s1_te_z)
        put_cell(tid, name, togo_eval(Z_tr, y_tr_all, Z_te, y_te, tr_idx, va_idx), cells[name])

    def expand(S_z, i, n, C):
        return torch.from_numpy(S_z.reshape(-1, C, 1, 1)[i:i + n]).cuda().expand(-1, C, 224, 224)

    # Clay
    suite = ModelSuite.clay_only(); suite.clay.cuda()
    @torch.no_grad()
    def emb_clay(S_z):
        N, T, C = S_z.shape
        outs = []
        w_t = torch.tensor(S1_2B_WAVES, dtype=torch.float32, device="cuda")
        g_t = torch.tensor(10.0, dtype=torch.float32, device="cuda")
        for i in range(0, N * T, 128):
            xb = expand(S_z, i, 128, C)
            B = len(xb)
            cube = {"pixels": xb.contiguous(), "time": torch.zeros((B, 4), device="cuda"), "latlon": torch.zeros((B, 4), device="cuda"), "gsd": g_t, "waves": w_t}
            with torch.autocast("cuda", dtype=torch.float16):
                tok, _, _, _ = suite.clay(cube)
            outs.append(tok.float()[:, 0].cpu().numpy())
        return agg(np.concatenate(outs, 0).reshape(N, T, -1))
    run_cell("clay", emb_clay)
    del suite; torch.cuda.empty_cache()

    # DOFA (own SAR convention) + diagnostic with Clay's convention
    dofa = dofa_large_patch16_224(weights=DOFALarge16_Weights.DOFA_MAE).cuda().eval()
    def make_dofa(waves):
        w_s1 = torch.tensor(waves, dtype=torch.float32, device="cuda")
        @torch.no_grad()
        def emb(S_z):
            N, T, C = S_z.shape
            outs = []
            for i in range(0, N * T, 256):
                xb = expand(S_z, i, 256, C); B = len(xb)
                with torch.amp.autocast("cuda", dtype=torch.float16):
                    tok, _ = dofa.patch_embed(xb.contiguous(), w_s1)
                    tok = tok + dofa.pos_embed[:, 1:, :]
                    cls_tok = (dofa.cls_token + dofa.pos_embed[:, :1, :]).expand(B, -1, -1)
                    tok = torch.cat((cls_tok, tok), dim=1)
                    for blk in dofa.blocks:
                        tok = blk(tok)
                outs.append(dofa.fc_norm(tok.float()[:, 0]).cpu().numpy())
            return agg(np.concatenate(outs, 0).reshape(N, T, -1))
        return emb
    run_cell("dofa_s1", make_dofa(DOFA_S1_2B_WAVES))
    run_cell("dofa_s1_clay_waves_diag", make_dofa(S1_2B_WAVES))
    del dofa; torch.cuda.empty_cache()

    pan = get_pan()
    @torch.no_grad()
    def emb_pan_ts(S_z):
        N, T, C = S_z.shape
        outs = []
        for i in range(0, N * T, 256):
            xb = expand(S_z, i, 256, C).contiguous()
            chn = torch.tensor([[-1.0, -2.0]] * len(xb), dtype=torch.float32, device="cuda")
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                f = pan({"imgs": xb, "chn_ids": chn}).float()
            outs.append(f.cpu().numpy())
        return agg(np.concatenate(outs, 0).reshape(N, T, -1))
    run_cell("panopticon_s1", emb_pan_ts)
    del pan; torch.cuda.empty_cache()

    cop = get_cop()
    c_area = TOGO_COP_AREA_KM2
    @torch.no_grad()
    def emb_cop_ts(S_z):
        N, T, C = S_z.shape
        flat = S_z.reshape(N * T, C, 1, 1).astype(np.float32)
        mean0, std0 = np.zeros(C, np.float32), np.ones(C, np.float32)   # already z-scored
        outs = []
        for i in range(0, N * T, 128):
            xb = np.broadcast_to(flat[i:i + 128], (len(flat[i:i + 128]), C, 224, 224)).copy()
            outs.append(ext_cop(cop, xb, mean0, std0, COP_S1_WAVE, COP_S1_BW, c_area, False)["cls"])
        return agg(np.concatenate(outs, 0).reshape(N, T, -1))
    run_cell("copernicus_fm_s1", emb_cop_ts)
    del cop; torch.cuda.empty_cache()

    dv2 = get_dinov2()
    @torch.no_grad()
    def emb_dv2_ts(S_z):
        N, T, _ = S_z.shape
        S3 = np.concatenate([S_z, 0.5 * (S_z[:, :, 0:1] + S_z[:, :, 1:2])], axis=2)
        outs = []
        for i in range(0, N * T, 256):
            xb = expand(S3, i, 256, 3).contiguous()
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                f = dv2.forward_features(xb)[:, 0].float()
            outs.append(f.cpu().numpy())
        return agg(np.concatenate(outs, 0).reshape(N, T, -1))
    run_cell("dinov2_s1", emb_dv2_ts)
    del dv2; torch.cuda.empty_cache()
    print("CropHarvest-Togo done:", {k: v for k, v in RES[tid].items() if not k.startswith("_")}, flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output-dir", default=None, help="default: fresh experiments/runs/table2_<UTC>/")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--parts", default="tasks,s1,togo")
    ap.add_argument("--tasks", default=None, help="comma-separated subset of task ids for the `tasks` part")
    a = ap.parse_args()
    RESUME = a.resume
    OUT_FILE = resolve_output(a.output_dir, a.resume, "table2", "all_doable_table2.json")
    print(f"[output] {OUT_FILE}", flush=True)
    if a.resume and OUT_FILE.exists():
        RES.update(json.load(open(OUT_FILE)))
    parts = a.parts.split(",")
    if "tasks" in parts:
        run_tasks(a.tasks.split(",") if a.tasks else None)
    if "s1" in parts:
        run_eurosat_s1()
    if "togo" in parts:
        run_togo()
    save_res()
    print("ALL REQUESTED TABLE 2 CELLS COMPLETE", flush=True)
