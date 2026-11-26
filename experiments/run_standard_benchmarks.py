#!/usr/bin/env python3
"""Standard vanilla multi-sensor & multi-scale benchmark suite (E9).

Evaluates frozen Clay v1.5 alongside frozen DOFA-L, Scale-MAE-L, DINOv2-L, and
non-learned Band Stats across 12 non-crop standard benchmarks spanning 0.1 m to
300 m GSD and 6 sensor families (including unseen 21-band Sentinel-3 OLCI).

Strictly follows PROTOCOL-COMMON.md and PROTOCOL_standard_benchmarks.md:
  - Peak GPU VRAM <= 3.5 GB (models moved to GPU one at a time in fp16 autocast).
  - Peak scratch disk <= 8 GB under the run work directory
    (temporary archives deleted immediately after each dataset finishes).
  - Incremental checkpointing to <output-dir>/standard_benchmarks.json after every dataset.

Output policy (audit fix #35): by default every invocation writes into a FRESH directory
`experiments/runs/standard_benchmarks_<UTC timestamp>/`; nothing committed is read as a cache.
Pass `--output-dir DIR --resume` to continue an earlier run explicitly; a task is skipped only if
its stored producer fingerprint (script sha256 + task configuration hash) equals the current one.

Usage:
  python experiments/run_standard_benchmarks.py [--output-dir DIR] [--resume] [--tasks m-brick-kiln,ucmerced]
  python experiments/run_standard_benchmarks.py --verify-bands-only      # m-brick-kiln / m-eurosat band check
"""
from __future__ import annotations

import argparse
import datetime as _dt

import gc
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import rasterio
import torch
import torch.nn.functional as F
import yaml
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, r2_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

HERE = Path(__file__).resolve().parents[1]
SCRATCH = Path(os.environ.get("CLAY_SCRATCH", str(HERE / "data/standard_benchmarks")))
# Per-process work directory: concurrent runs on a shared machine never delete each other's data.
WORK_DIR = SCRATCH / f"_work_{os.getpid()}"
RUNS_ROOT = Path(os.environ.get("CLAY_RUNS_ROOT", str(HERE / "experiments/runs")))
H5_PYTHON = os.environ.get("CLAY_H5_PYTHON", sys.executable)
CLAY_REPO = os.environ.get("CLAY_MODEL_REPO", str(HERE / "experiments/suite/common"))
CLAY_CKPT = os.environ.get("CLAY_V15_CKPT", str(HERE / "checkpoints/v1.5/clay-v1.5.ckpt"))
CLAY_SHA256 = "21432069250b9b3f9a65ffd0071c5ad56b793247285ab0604edf7f531d4798d0"
# Smoke tests only (experiments/repro_smoke.sh): N > 0 keeps N random samples (seeded) of each split, so the
# whole pipeline runs in minutes. Never set for a result that is reported. `--limit N` sets it too.
SMOKE_LIMIT = int(os.environ.get("CLAY_SMOKE_LIMIT", "0"))

sys.path.insert(0, CLAY_REPO)

# Official ESA Sentinel-3 OLCI 21-band center wavelengths (um)
S3_OLCI_WAVES = [
    0.400, 0.4125, 0.4425, 0.490, 0.510, 0.560, 0.620, 0.665,
    0.67375, 0.68125, 0.70875, 0.75375, 0.76125, 0.764375, 0.7675, 0.77875,
    0.865, 0.885, 0.900, 0.940, 1.020,
]

S2_10B_WAVES = [0.493, 0.560, 0.665, 0.704, 0.740, 0.783, 0.842, 0.865, 1.610, 2.190]
RGB_WAVES = [0.650, 0.560, 0.480]
NAIP_WAVES = [0.650, 0.560, 0.480, 0.842]
L8_6B_WAVES = [0.650, 0.560, 0.480, 0.860, 1.600, 2.200]
S1_2B_WAVES = [3.500, 4.000]
DOFA_S1_2B_WAVES = [3.750, 3.750]

# --- GEO-Bench archive identity and the documented band order (audit fix #15) -------------------
GEOBENCH_HF_REPO = "recursix/geo-bench-1.0"
GEOBENCH_HF_REVISION = os.environ.get("CLAY_GEOBENCH_REVISION", "1e5754337032f097396db62cc15a5ff88f4618e8")
# Names stored in the HDF5 files of m-eurosat / m-brick-kiln (13 channels; sorted key order == channel order).
# They are WRONG for most channels (geobench 1.1.0 "Known issues"; ServiceNow/geo-bench issue 29).
GB_STORED_NAMES = [
    "01 - Coastal aerosol", "02 - Blue", "03 - Green", "04 - Red", "05 - Vegetation Red Edge",
    "06 - Vegetation Red Edge", "07 - Vegetation Red Edge", "08 - NIR", "08A - Vegetation Red Edge",
    "09 - Water vapour", "10 - SWIR - Cirrus", "11 - SWIR", "12 - SWIR",
]
# Documented TRUE channel order (index = position in sorted HDF5 keys).
GB_TRUE_ORDER = {
    "m-eurosat": ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B09", "B10", "B11", "B12", "B8A"],
    "m-brick-kiln": ["B01", "B02", "B03", "B04", "B05", "B07", "B8A", "B08", "B11", "B12", "TCI_R", "TCI_G", "TCI_B"],
}
S2_WAVE_UM = {"B01": 0.443, "B02": 0.493, "B03": 0.560, "B04": 0.665, "B05": 0.704, "B06": 0.740,
              "B07": 0.783, "B08": 0.842, "B8A": 0.865, "B11": 1.610, "B12": 2.190}
EUROSAT_BANDS = ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"]   # 10 physical bands
KILN_BANDS = ["B02", "B03", "B04", "B05", "B07", "B08", "B8A", "B11", "B12"]               # 9: B06 is absent, TCI_* excluded
CLAY_META_S2_NAME = {"B02": "blue", "B03": "green", "B04": "red", "B05": "rededge1", "B06": "rededge2",
                     "B07": "rededge3", "B08": "nir", "B8A": "nir08", "B11": "swir16", "B12": "swir22"}


OLMO_REPO = "allenai/OlmoEarth-v1_3-Base"
OLMO_REVISION = os.environ.get("CLAY_OLMO_REVISION", "9e9e918a369701c105dde7ef56789b0ff1d47f50")


def olmo_files() -> tuple[str, str]:
    """(config.json, weights.pth) of OlmoEarth v1.3 Base at a pinned HF revision (downloaded if absent)."""
    from huggingface_hub import hf_hub_download
    return (hf_hub_download(OLMO_REPO, "config.json", revision=OLMO_REVISION),
            hf_hub_download(OLMO_REPO, "weights.pth", revision=OLMO_REVISION))


def gb_channels(task: str, bands: list[str]) -> list[int]:
    """Channel indices (position in sorted HDF5 keys) of the named PHYSICAL bands, from the documented order."""
    order = GB_TRUE_ORDER[task]
    return [order.index(b) for b in bands]


def gb_waves(bands: list[str]) -> list[float]:
    return [S2_WAVE_UM[b] for b in bands]

CLAY_METADATA = yaml.safe_load((Path(CLAY_REPO) / "configs/metadata.yaml").read_text())


# --- provenance (audit fixes #35, #37) -----------------------------------------------------------
def utc_now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def git_head() -> str:
    try:
        out = subprocess.run(["git", "-C", str(HERE), "rev-parse", "HEAD"], capture_output=True, text=True, timeout=20)
        head = out.stdout.strip() or "unknown"
        dirty = subprocess.run(["git", "-C", str(HERE), "status", "--porcelain", "--", "experiments"],
                               capture_output=True, text=True, timeout=20).stdout.strip()
        return head + ("+dirty" if dirty else "")
    except Exception:
        return "unknown"


def provenance(script: str, extra: dict | None = None) -> dict:
    import importlib.metadata as md
    pk = {}
    for name in ("torch", "torchgeo", "numpy", "scikit-learn", "timm", "h5py", "rasterio", "olmoearth_pretrain"):
        try:
            pk[name] = md.version(name)
        except Exception:
            pk[name] = "n/a"
    return {
        "script": Path(script).name,
        "script_sha256": sha256_file(script),
        "git_commit": git_head(),
        "argv": [a.replace(str(Path.home()), "~") for a in sys.argv[1:]],   # no absolute home paths in committed JSON
        "utc_timestamp": utc_now(),
        "packages": pk,
        "python": sys.version.split()[0],
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        **(extra or {}),
    }


def fresh_output_dir(prefix: str) -> Path:
    d = RUNS_ROOT / f"{prefix}_{_dt.datetime.now(_dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    d.mkdir(parents=True, exist_ok=False)
    return d


def cfg_fingerprint(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:16]


def resolve_output(output_dir: str | None, resume: bool, prefix: str, filename: str) -> Path:
    """Fresh dir by default. An existing dir with an existing result file requires --resume."""
    if output_dir is None:
        return fresh_output_dir(prefix) / filename
    d = Path(output_dir)
    d.mkdir(parents=True, exist_ok=True)
    out = d / filename
    if out.exists() and not resume:
        raise SystemExit(f"{out} exists; pass --resume to continue it explicitly, or choose a new --output-dir")
    return out


def can_skip(entry: dict | None, script: str, fingerprint: str) -> bool:
    """A stored cell may be reused only if it was produced by this exact script and configuration."""
    if not entry:
        return False
    prod = entry.get("producer", {}) if isinstance(entry, dict) else {}
    return prod.get("script_sha256") == sha256_file(script) and prod.get("cfg") == fingerprint


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 24):
            h.update(chunk)
    return h.hexdigest()


def clean_nonfinite(x: np.ndarray, fallback_means: np.ndarray | None = None) -> np.ndarray:
    """Impute non-finite pixels with per-chip per-band finite mean (Lesson 5)."""
    if np.isfinite(x).all():
        return x
    x = x.copy()
    B, C, H, W = x.shape
    flat = x.reshape(B, C, H * W)
    finite = np.isfinite(flat)
    with np.errstate(invalid="ignore"):
        sums = np.where(finite, flat, 0.0).sum(axis=-1)
        cnts = finite.sum(axis=-1)
        chip_means = np.where(cnts > 0, sums / np.maximum(cnts, 1), np.nan)
    if fallback_means is not None:
        chip_means = np.where(np.isfinite(chip_means), chip_means, fallback_means[None, :])
    else:
        chip_means = np.nan_to_num(chip_means, nan=0.0)
    bad = ~finite
    if bad.any():
        fill = np.broadcast_to(chip_means[:, :, None], flat.shape)
        flat[bad] = fill[bad]
    return flat.reshape(B, C, H, W)


def compute_train_band_stats(x_train: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute per-band mean, std, p01, p99 strictly on training split."""
    C = x_train.shape[1]
    means = np.zeros(C, dtype=np.float32)
    stds = np.ones(C, dtype=np.float32)
    p01 = np.zeros(C, dtype=np.float32)
    p99 = np.ones(C, dtype=np.float32)
    rng = np.random.default_rng(42)
    n_sub = min(len(x_train), 2000)
    idx = rng.choice(len(x_train), size=n_sub, replace=False) if len(x_train) > n_sub else np.arange(len(x_train))
    for c in range(C):
        col = x_train[:, c]                      # float32 view; accumulate in float64 without a float64 copy (memory)
        fin = col[np.isfinite(col)]
        means[c] = float(fin.mean(dtype=np.float64)) if fin.size else 0.0
        stds[c] = max(float(fin.std(dtype=np.float64)), 1e-6) if fin.size else 1.0
        sub_fin = x_train[idx, c]
        sub_fin = sub_fin[np.isfinite(sub_fin)]
        if sub_fin.size:
            p01[c], p99[c] = np.percentile(sub_fin, [1.0, 99.0])
    return means, stds, p01, p99


def to_rgb01(x: np.ndarray, rgb_idx: list[int], is_uint8: bool, is_unit_float: bool,
             p01: np.ndarray, p99: np.ndarray) -> np.ndarray:
    # chunked over chips (identical values; avoids 3 full-size float32 temporaries on large splits)
    out = np.empty((len(x), len(rgb_idx), *x.shape[2:]), dtype=np.float32)
    lo = p01[rgb_idx][None, :, None, None]
    hi = p99[rgb_idx][None, :, None, None]
    for i in range(0, len(x), 512):
        rgb = x[i:i + 512][:, rgb_idx].astype(np.float32)
        if is_uint8:
            out[i:i + 512] = np.clip(rgb / 255.0, 0.0, 1.0)
        elif is_unit_float:
            out[i:i + 512] = np.clip(rgb, 0.0, 1.0)
        else:
            out[i:i + 512] = np.clip((rgb - lo) / np.maximum(hi - lo, 1e-6), 0.0, 1.0)
    return out


# ==============================================================================
# Model Wrapper Suite (Loaded Once on CPU, Moved to GPU One at a Time)
# ==============================================================================
class ModelSuite:
    def __init__(self):
        from claymodel.model import Encoder
        import timm
        from torchgeo.models import (
            DOFALarge16_Weights,
            ScaleMAELarge16_Weights,
            dofa_large_patch16_224,
            scalemae_large_patch16,
        )

        print("[init] Verifying Clay v1.5 checkpoint SHA-256...", flush=True)
        got_sha = sha256_file(CLAY_CKPT)
        if got_sha != CLAY_SHA256:
            raise RuntimeError(f"Clay SHA mismatch: {got_sha}")
        sd = torch.load(CLAY_CKPT, map_location="cpu", weights_only=True, mmap=True)["state_dict"]
        state = {k.removeprefix("model.encoder."): v for k, v in sd.items() if k.startswith("model.encoder.")}
        self.clay = Encoder(
            mask_ratio=0.0, patch_size=8, shuffle=False,
            dim=1024, depth=24, heads=16, dim_head=64, mlp_ratio=4.0,
        )
        self.clay.load_state_dict(state, strict=True)
        self.clay.eval().requires_grad_(False)

        print("[init] Loading DOFA-L (DOFA_MAE)...", flush=True)
        self.dofa = dofa_large_patch16_224(weights=DOFALarge16_Weights.DOFA_MAE)
        self.dofa.eval().requires_grad_(False)

        print("[init] Loading Scale-MAE-L (FMOW_RGB)...", flush=True)
        self.smae = scalemae_large_patch16(weights=ScaleMAELarge16_Weights.FMOW_RGB, num_classes=0)
        self.smae.eval().requires_grad_(False)

        print("[init] Loading DINOv2-L/14-reg4...", flush=True)
        self.dv2 = timm.create_model(
            "vit_large_patch14_reg4_dinov2.lvd142m", pretrained=True, num_classes=0, img_size=224,
        )
        self.dv2.eval().requires_grad_(False)
        cfg = self.dv2.pretrained_cfg
        self.imnet_mean = torch.tensor(cfg["mean"], dtype=torch.float32).view(1, 3, 1, 1)
        self.imnet_std = torch.tensor(cfg["std"], dtype=torch.float32).view(1, 3, 1, 1)

    @classmethod
    def clay_only(cls):
        """Clay encoder only (no peer models); same checkpoint check and construction as __init__."""
        from claymodel.model import Encoder
        self = cls.__new__(cls)
        if sha256_file(CLAY_CKPT) != CLAY_SHA256:
            raise RuntimeError("Clay checkpoint SHA-256 mismatch")
        sd = torch.load(CLAY_CKPT, map_location="cpu", weights_only=True, mmap=True)["state_dict"]
        state = {k.removeprefix("model.encoder."): v for k, v in sd.items() if k.startswith("model.encoder.")}
        self.clay = Encoder(mask_ratio=0.0, patch_size=8, shuffle=False, dim=1024, depth=24, heads=16, dim_head=64, mlp_ratio=4.0)
        self.clay.load_state_dict(state, strict=True)
        self.clay.eval().requires_grad_(False)
        return self

    @torch.inference_mode()
    def extract_clay(self, x: np.ndarray, means: np.ndarray, stds: np.ndarray,
                     waves: list[float], gsd: float, target_hw: int | None = None,
                     need_patches: bool = False, batch_size: int = 32) -> dict[str, np.ndarray]:
        self.clay.to("cuda")
        try:
            N, C, H, W = x.shape
            H_use = target_hw or H
            gsd_eff = float(gsd) * (H / float(H_use))
            mu_t = torch.tensor(means, dtype=torch.float32, device="cuda").view(1, C, 1, 1)
            sd_t = torch.tensor(stds, dtype=torch.float32, device="cuda").view(1, C, 1, 1)
            w_t = torch.tensor(waves, dtype=torch.float32, device="cuda")
            g_t = torch.tensor(gsd_eff, dtype=torch.float32, device="cuda")

            cls_l, mean_l, pat_l = [], [], []
            gh = H_use // 8
            for i in range(0, N, batch_size):
                xb = torch.as_tensor(x[i:i + batch_size], dtype=torch.float32, device="cuda")
                if xb.shape[-1] != H_use or xb.shape[-2] != H_use:
                    xb = F.interpolate(xb, size=(H_use, H_use), mode="bilinear", align_corners=False)
                xb = (xb - mu_t) / sd_t
                bsz = xb.shape[0]
                cube = {
                    "pixels": xb,
                    "time": torch.zeros((bsz, 4), dtype=torch.float32, device="cuda"),
                    "latlon": torch.zeros((bsz, 4), dtype=torch.float32, device="cuda"),
                    "gsd": g_t,
                    "waves": w_t,
                }
                with torch.autocast("cuda", dtype=torch.float16):
                    tok, _, _, _ = self.clay(cube)
                tok = tok.float()
                cls_l.append(tok[:, 0].cpu().numpy())
                mean_l.append(tok[:, 1:].mean(dim=1).cpu().numpy())
                if need_patches:
                    pat_l.append(tok[:, 1:].reshape(bsz, gh, gh, 1024).half().cpu().numpy())
            out = {"cls": np.concatenate(cls_l, axis=0), "mean": np.concatenate(mean_l, axis=0)}
            if need_patches:
                out["patches"] = np.concatenate(pat_l, axis=0)
            return out
        finally:
            self.clay.to("cpu")
            torch.cuda.empty_cache()

    @torch.inference_mode()
    def extract_dofa(self, x: np.ndarray, means: np.ndarray, stds: np.ndarray,
                     waves: list[float], need_patches: bool = False,
                     batch_size: int = 64) -> dict[str, np.ndarray]:
        self.dofa.to("cuda")
        try:
            N, C, _, _ = x.shape
            mu_t = torch.tensor(means, dtype=torch.float32, device="cuda").view(1, C, 1, 1)
            sd_t = torch.tensor(stds, dtype=torch.float32, device="cuda").view(1, C, 1, 1)
            w_t = torch.tensor(waves, dtype=torch.float32, device="cuda")

            cls_l, mean_l, pat_l = [], [], []
            for i in range(0, N, batch_size):
                xb = torch.as_tensor(x[i:i + batch_size], dtype=torch.float32, device="cuda")
                if xb.shape[-2:] != (224, 224):
                    xb = F.interpolate(xb, size=(224, 224), mode="bilinear", align_corners=False)
                xb = (xb - mu_t) / sd_t
                bsz = xb.shape[0]
                with torch.autocast("cuda", dtype=torch.float16):
                    tok, _ = self.dofa.patch_embed(xb, w_t)
                    tok = tok + self.dofa.pos_embed[:, 1:, :]
                    cls_tok = (self.dofa.cls_token + self.dofa.pos_embed[:, :1, :]).expand(bsz, -1, -1)
                    tok = torch.cat((cls_tok, tok), dim=1)
                    for blk in self.dofa.blocks:
                        tok = blk(tok)
                tok = tok.float()
                c_out = self.dofa.fc_norm(tok[:, 0])
                m_out = self.dofa.fc_norm(tok[:, 1:].mean(dim=1))
                cls_l.append(c_out.cpu().numpy())
                mean_l.append(m_out.cpu().numpy())
                if need_patches:
                    p_out = self.dofa.fc_norm(tok[:, 1:]).reshape(bsz, 14, 14, 1024)
                    pat_l.append(p_out.half().cpu().numpy())
            out = {"cls": np.concatenate(cls_l, axis=0), "mean": np.concatenate(mean_l, axis=0)}
            if need_patches:
                out["patches"] = np.concatenate(pat_l, axis=0)
            return out
        finally:
            self.dofa.to("cpu")
            torch.cuda.empty_cache()

    @torch.inference_mode()
    def extract_scalemae(self, rgb01: np.ndarray, gsd_orig: float, h_orig: int,
                         need_patches: bool = False, batch_size: int = 64) -> dict[str, np.ndarray]:
        self.smae.to("cuda")
        try:
            N = len(rgb01)
            self.smae.res = float(gsd_orig) * (float(h_orig) / 224.0)
            mu_t = self.imnet_mean.to("cuda")
            sd_t = self.imnet_std.to("cuda")
            cls_l, mean_l, pat_l = [], [], []
            for i in range(0, N, batch_size):
                xb = torch.as_tensor(rgb01[i:i + batch_size], dtype=torch.float32, device="cuda")
                if xb.shape[-2:] != (224, 224):
                    xb = F.interpolate(xb, size=(224, 224), mode="bilinear", align_corners=False)
                xb = (xb - mu_t) / sd_t
                bsz = xb.shape[0]
                with torch.autocast("cuda", dtype=torch.float16):
                    f = self.smae.forward_features(xb)
                f = f.float()
                cls_l.append(f[:, 0].cpu().numpy())
                mean_l.append(f[:, 1:].mean(dim=1).cpu().numpy())
                if need_patches:
                    pat_l.append(f[:, 1:].reshape(bsz, 14, 14, 1024).half().cpu().numpy())
            out = {"cls": np.concatenate(cls_l, axis=0), "mean": np.concatenate(mean_l, axis=0)}
            if need_patches:
                out["patches"] = np.concatenate(pat_l, axis=0)
            return out
        finally:
            self.smae.to("cpu")
            torch.cuda.empty_cache()

    @torch.inference_mode()
    def extract_dinov2(self, rgb01: np.ndarray, need_patches: bool = False,
                       batch_size: int = 64) -> dict[str, np.ndarray]:
        self.dv2.to("cuda")
        try:
            N = len(rgb01)
            mu_t = self.imnet_mean.to("cuda")
            sd_t = self.imnet_std.to("cuda")
            npre = getattr(self.dv2, "num_prefix_tokens", 5)
            cls_l, mean_l, pat_l = [], [], []
            for i in range(0, N, batch_size):
                xb = torch.as_tensor(rgb01[i:i + batch_size], dtype=torch.float32, device="cuda")
                if xb.shape[-2:] != (224, 224):
                    xb = F.interpolate(xb, size=(224, 224), mode="bilinear", align_corners=False)
                xb = (xb - mu_t) / sd_t
                bsz = xb.shape[0]
                with torch.autocast("cuda", dtype=torch.float16):
                    f = self.dv2.forward_features(xb)
                f = f.float()
                cls_l.append(f[:, 0].cpu().numpy())
                mean_l.append(f[:, npre:].mean(dim=1).cpu().numpy())
                if need_patches:
                    pat_l.append(f[:, npre:].reshape(bsz, 16, 16, 1024).half().cpu().numpy())
            out = {"cls": np.concatenate(cls_l, axis=0), "mean": np.concatenate(mean_l, axis=0)}
            if need_patches:
                out["patches"] = np.concatenate(pat_l, axis=0)
            return out
        finally:
            self.dv2.to("cpu")
            torch.cuda.empty_cache()


def extract_band_stats(x: np.ndarray) -> np.ndarray:
    # per-chip statistics, computed in chunks of chips (identical values, no full-size temporaries)
    out = []
    for i in range(0, len(x), 256):
        flat = x[i:i + 256].astype(np.float32, copy=False).reshape(min(256, len(x) - i), x.shape[1], -1)
        out.append(np.concatenate([flat.mean(axis=-1), flat.std(axis=-1)], axis=-1))
    return np.concatenate(out, axis=0)


# ==============================================================================
# Probe Evaluation Functions (Val-Tuned C/Alpha, Refit on Train Only -> Test)
# ==============================================================================
def _fit_gpu_linear_lbfgs(Xtr_t: torch.Tensor, target_t: torch.Tensor, n_out: int,
                          reg_weight: float, mode: str, max_iter: int = 150) -> tuple[torch.Tensor, torch.Tensor]:
    """Exact full-batch L-BFGS solver on GPU matching sklearn LogisticRegression / LinearSVC(squared_hinge)."""
    D = Xtr_t.shape[1]
    W = torch.zeros((n_out, D), dtype=torch.float32, device="cuda", requires_grad=True)
    b = torch.zeros((n_out,), dtype=torch.float32, device="cuda", requires_grad=True)
    opt = torch.optim.LBFGS([W, b], lr=1.0, max_iter=max_iter, tolerance_grad=1e-6,
                            tolerance_change=1e-9, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        logits = Xtr_t @ W.t() + b
        if mode == "ce":
            loss_data = F.cross_entropy(logits, target_t, reduction="mean")
        elif mode == "bce":
            loss_data = F.binary_cross_entropy_with_logits(logits, target_t, reduction="sum") / Xtr_t.shape[0]
        elif mode == "sq_hinge":
            margins = 1.0 - target_t * logits
            loss_data = torch.clamp(margins, min=0.0).pow(2).sum() / Xtr_t.shape[0]
        else:
            raise ValueError(mode)
        loss = loss_data + reg_weight * (W * W).sum()
        loss.backward()
        return loss

    opt.step(closure)
    return W.detach(), b.detach()


def eval_single_label(z_tr: np.ndarray, y_tr: np.ndarray,
                      z_va: np.ndarray, y_va: np.ndarray,
                      z_te: np.ndarray, y_te: np.ndarray) -> dict:
    scaler = StandardScaler().fit(z_tr)
    Xtr = torch.tensor(scaler.transform(z_tr), dtype=torch.float32, device="cuda")
    Xva = torch.tensor(scaler.transform(z_va), dtype=torch.float32, device="cuda")
    Xte = torch.tensor(scaler.transform(z_te), dtype=torch.float32, device="cuda")
    ytr_t = torch.tensor(y_tr, dtype=torch.long, device="cuda")
    N = Xtr.shape[0]
    K = int(max(y_tr.max(), y_va.max(), y_te.max())) + 1

    # 1. Multinomial L2 Logistic Regression (exact match to sklearn C grid)
    best_c_lr, best_va_lr, best_wb_lr = 1.0, -1.0, None
    for c in (0.01, 0.1, 1.0, 10.0):
        reg_w = 0.5 / (c * N)
        W, b = _fit_gpu_linear_lbfgs(Xtr, ytr_t, K, reg_w, mode="ce")
        with torch.no_grad():
            pred_va = (Xva @ W.t() + b).argmax(dim=1).cpu().numpy()
        sc = float(accuracy_score(y_va, pred_va))
        if sc > best_va_lr:
            best_va_lr, best_c_lr, best_wb_lr = sc, c, (W, b)
    with torch.no_grad():
        pred_lr = (Xte @ best_wb_lr[0].t() + best_wb_lr[1]).argmax(dim=1).cpu().numpy()

    # 2. One-vs-Rest Squared-Hinge Linear SVM (exact match to sklearn LinearSVC)
    Y_pm1 = torch.full((N, K), -1.0, dtype=torch.float32, device="cuda")
    Y_pm1.scatter_(1, ytr_t.unsqueeze(1), 1.0)
    best_c_svm, best_va_svm, best_wb_svm = 0.1, -1.0, None
    for c in (0.001, 0.01, 0.1, 1.0):
        reg_w = 0.5 / (c * N)
        W, b = _fit_gpu_linear_lbfgs(Xtr, Y_pm1, K, reg_w, mode="sq_hinge")
        with torch.no_grad():
            pred_va = (Xva @ W.t() + b).argmax(dim=1).cpu().numpy()
        sc = float(accuracy_score(y_va, pred_va))
        if sc > best_va_svm:
            best_va_svm, best_c_svm, best_wb_svm = sc, c, (W, b)
    with torch.no_grad():
        pred_svm = (Xte @ best_wb_svm[0].t() + best_wb_svm[1]).argmax(dim=1).cpu().numpy()

    return {
        "oa": round(float(accuracy_score(y_te, pred_lr)) * 100.0, 2),
        "macro_f1": round(float(f1_score(y_te, pred_lr, average="macro")) * 100.0, 2),
        "val_oa": round(best_va_lr * 100.0, 2),
        "best_C": best_c_lr,
        "lsvm_oa": round(float(accuracy_score(y_te, pred_svm)) * 100.0, 2),
        "lsvm_macro_f1": round(float(f1_score(y_te, pred_svm, average="macro")) * 100.0, 2),
    }


def eval_multilabel(z_tr: np.ndarray, y_tr: np.ndarray,
                    z_va: np.ndarray, y_va: np.ndarray,
                    z_te: np.ndarray, y_te: np.ndarray) -> dict:
    scaler = StandardScaler().fit(z_tr)
    Xtr = torch.tensor(scaler.transform(z_tr), dtype=torch.float32, device="cuda")
    Xva = torch.tensor(scaler.transform(z_va), dtype=torch.float32, device="cuda")
    Xte = torch.tensor(scaler.transform(z_te), dtype=torch.float32, device="cuda")
    N = Xtr.shape[0]

    valid_va = (y_tr.sum(0) > 0) & (y_va.sum(0) > 0)
    valid_te = (y_tr.sum(0) > 0) & (y_te.sum(0) > 0)
    idx_tr = np.where(y_tr.sum(0) > 0)[0]
    Ytr_t = torch.tensor(y_tr[:, idx_tr], dtype=torch.float32, device="cuda")

    best_c, best_va_map, best_scores_te = 1.0, -1.0, None
    for c in (0.01, 0.1, 1.0, 10.0):
        reg_w = 0.5 / (c * N)
        W, b = _fit_gpu_linear_lbfgs(Xtr, Ytr_t, len(idx_tr), reg_w, mode="bce")
        with torch.no_grad():
            p_va_sub = torch.sigmoid(Xva @ W.t() + b).cpu().numpy()
        prob_va = np.zeros_like(y_va, dtype=np.float32)
        prob_va[:, idx_tr] = p_va_sub
        va_map = float(average_precision_score(y_va[:, valid_va], prob_va[:, valid_va], average="macro"))
        if va_map > best_va_map:
            best_va_map = va_map
            best_c = c
            with torch.no_grad():
                p_te_sub = torch.sigmoid(Xte @ W.t() + b).cpu().numpy()
            prob_te = np.zeros_like(y_te, dtype=np.float32)
            prob_te[:, idx_tr] = p_te_sub
            best_scores_te = prob_te

    macro_map = float(average_precision_score(y_te[:, valid_te], best_scores_te[:, valid_te], average="macro"))
    micro_map = float(average_precision_score(y_te[:, valid_te], best_scores_te[:, valid_te], average="micro"))
    return {
        "macro_map": round(macro_map * 100.0, 2),
        "micro_map": round(micro_map * 100.0, 2),
        "val_macro_map": round(best_va_map * 100.0, 2),
        "n_eval_classes": int(valid_te.sum()),
        "best_C": best_c,
    }


def eval_dense_segmentation(pat_tr: np.ndarray, m_tr: np.ndarray,
                            pat_va: np.ndarray, m_va: np.ndarray,
                            pat_te: np.ndarray, m_te: np.ndarray,
                            n_classes: int) -> dict:
    """Per-token linear classifier upsampled bilinearly to full pixel mask resolution."""
    Ntr, Gh, Gw, D = pat_tr.shape
    H_mask, W_mask = m_tr.shape[1], m_tr.shape[2]

    m_tr_tok = F.interpolate(
        torch.as_tensor(m_tr[:, None], dtype=torch.float32),
        size=(Gh, Gw), mode="nearest",
    )[:, 0].numpy().astype(np.int64)

    flat_X = pat_tr.reshape(-1, D)
    flat_y = m_tr_tok.reshape(-1)
    rng = np.random.default_rng(42)
    n_sub = min(len(flat_y), 180_000)
    sub_idx = rng.choice(len(flat_y), size=n_sub, replace=False) if len(flat_y) > n_sub else np.arange(len(flat_y))
    X_sub = flat_X[sub_idx].astype(np.float32)
    y_sub = flat_y[sub_idx]

    scaler = StandardScaler().fit(X_sub)
    X_sub_t = torch.tensor(scaler.transform(X_sub), dtype=torch.float32, device="cuda")
    y_sub_t = torch.tensor(y_sub, dtype=torch.long, device="cuda")
    mu = torch.tensor(scaler.mean_, dtype=torch.float32, device="cuda")
    sc = torch.tensor(scaler.scale_, dtype=torch.float32, device="cuda")

    def score_split(clf_w: torch.Tensor, clf_b: torch.Tensor, pats: np.ndarray, masks: np.ndarray) -> tuple[float, list[float]]:
        conf = np.zeros((n_classes, n_classes), dtype=np.int64)
        for i in range(0, len(pats), 64):
            pb = torch.as_tensor(pats[i:i + 64], dtype=torch.float32, device="cuda")
            pb = (pb - mu) / sc
            logits = torch.einsum("bhwd,kd->bkhw", pb, clf_w) + clf_b.view(1, -1, 1, 1)
            up = F.interpolate(logits, size=(H_mask, W_mask), mode="bilinear", align_corners=False)
            pred = up.argmax(dim=1).cpu().numpy().reshape(-1)
            gt = masks[i:i + 64].reshape(-1).astype(np.int64)
            valid = (gt >= 0) & (gt < n_classes)
            idx_cm = gt[valid] * n_classes + pred[valid]
            conf += np.bincount(idx_cm, minlength=n_classes * n_classes).reshape(n_classes, n_classes)
        inter = np.diag(conf).astype(np.float64)
        union = conf.sum(1) + conf.sum(0) - inter
        ious = [float(inter[k] / union[k]) for k in range(n_classes) if union[k] > 0]
        return float(np.mean(ious)), ious

    best_c, best_va_miou, best_wb = 1.0, -1.0, None
    N_s = X_sub_t.shape[0]
    for c in (0.01, 0.1, 1.0, 10.0):
        w_t, b_t = _fit_gpu_linear_lbfgs(X_sub_t, y_sub_t, n_classes, 0.5 / (c * N_s), mode="ce")
        va_miou, _ = score_split(w_t, b_t, pat_va, m_va)
        if va_miou > best_va_miou:
            best_va_miou, best_c, best_wb = va_miou, c, (w_t, b_t)

    te_miou, te_ious = score_split(best_wb[0], best_wb[1], pat_te, m_te)
    out = {
        "miou": round(te_miou * 100.0, 2),
        "val_miou": round(best_va_miou * 100.0, 2),
        "best_C": best_c,
    }
    if n_classes == 2 and len(te_ious) == 2:
        out["fg_iou"] = round(te_ious[1] * 100.0, 2)
    return out


def eval_dense_regression(pat_tr: np.ndarray, y_tr: np.ndarray,
                          pat_va: np.ndarray, y_va: np.ndarray,
                          pat_te: np.ndarray, y_te: np.ndarray) -> dict:
    """Per-token linear Ridge upsampled bilinearly to full pixel target grid."""
    Ntr, Gh, Gw, D = pat_tr.shape
    H_t, W_t = y_tr.shape[1], y_tr.shape[2]
    y_tr_tok = F.adaptive_avg_pool2d(
        torch.as_tensor(y_tr[:, None], dtype=torch.float32), output_size=(Gh, Gw),
    )[:, 0].numpy()

    X_flat = pat_tr.reshape(-1, D).astype(np.float32)
    y_flat = y_tr_tok.reshape(-1).astype(np.float32)
    rng = np.random.default_rng(42)
    if len(y_flat) > 250_000:
        idx = rng.choice(len(y_flat), size=250_000, replace=False)
        X_flat, y_flat = X_flat[idx], y_flat[idx]

    scaler = StandardScaler().fit(X_flat)
    X_s = scaler.transform(X_flat)
    mu = torch.tensor(scaler.mean_, dtype=torch.float32, device="cuda")
    sc = torch.tensor(scaler.scale_, dtype=torch.float32, device="cuda")

    def score_split(w_t: torch.Tensor, b_val: float, pats: np.ndarray, targets: np.ndarray) -> tuple[float, float]:
        sse = 0.0
        sum_y = 0.0
        sum_y2 = 0.0
        n_pix = 0
        for i in range(0, len(pats), 64):
            pb = torch.as_tensor(pats[i:i + 64], dtype=torch.float32, device="cuda")
            pb = (pb - mu) / sc
            pred_tok = (pb @ w_t).unsqueeze(1) + b_val
            pred = F.interpolate(pred_tok, size=(H_t, W_t), mode="bilinear", align_corners=False)[:, 0]
            gt = torch.as_tensor(targets[i:i + 64], dtype=torch.float32, device="cuda")
            diff = pred - gt
            sse += float((diff * diff).sum().item())
            sum_y += float(gt.sum().item())
            sum_y2 += float((gt * gt).sum().item())
            n_pix += gt.numel()
        rmse = math.sqrt(sse / n_pix)
        mean_y = sum_y / n_pix
        sst = sum_y2 - n_pix * (mean_y ** 2)
        r2 = 1.0 - (sse / max(sst, 1e-9))
        return rmse, r2

    best_a, best_va_rmse, best_wb = 100.0, 1e18, None
    for a in (1.0, 10.0, 100.0, 1000.0, 10000.0):
        reg = Ridge(alpha=a, random_state=42).fit(X_s, y_flat)
        w_t = torch.tensor(reg.coef_.astype(np.float32), device="cuda")
        b_v = float(reg.intercept_)
        va_rmse, _ = score_split(w_t, b_v, pat_va, y_va)
        if va_rmse < best_va_rmse:
            best_va_rmse, best_a, best_wb = va_rmse, a, (w_t, b_v)

    te_rmse, te_r2 = score_split(best_wb[0], best_wb[1], pat_te, y_te)
    return {
        "pixel_rmse": round(te_rmse, 2),
        "pixel_r2": round(te_r2, 4),
        "val_pixel_rmse": round(best_va_rmse, 2),
        "best_alpha": best_a,
    }


def eval_chip_regression(z_tr: np.ndarray, y_tr_mean: np.ndarray,
                         z_va: np.ndarray, y_va_mean: np.ndarray,
                         z_te: np.ndarray, y_te_mean: np.ndarray) -> dict:
    scaler = StandardScaler().fit(z_tr)
    Xtr, Xva, Xte = scaler.transform(z_tr), scaler.transform(z_va), scaler.transform(z_te)
    best_a, best_rmse, best_reg = 10.0, 1e18, None
    for a in (0.1, 1.0, 10.0, 100.0, 1000.0):
        reg = Ridge(alpha=a).fit(Xtr, y_tr_mean)
        rmse = float(np.sqrt(np.mean((reg.predict(Xva) - y_va_mean) ** 2)))
        if rmse < best_rmse:
            best_rmse, best_a, best_reg = rmse, a, reg
    pred = best_reg.predict(Xte)
    return {
        "chip_r2": round(float(r2_score(y_te_mean, pred)), 4),
        "chip_rmse": round(float(np.sqrt(np.mean((pred - y_te_mean) ** 2))), 2),
    }


# ==============================================================================
# Dataset Loaders (One-at-a-Time Unpacking + Immediate Scratch Cleanup)
# ==============================================================================
GEOBENCH_HELPER = r"""
import ast, hashlib, io, json, os, pickle, sys, zipfile
from pathlib import Path
from types import SimpleNamespace
import h5py, numpy as np
from huggingface_hub import hf_hub_download

(repo_file, task_name, band_spec_json, kind, out_dir_str, hf_repo, hf_rev, stored_names_json,
 true_order_json) = sys.argv[1:10]
band_specs = json.loads(band_spec_json)   # list of str (HDF5 key / prefix*) or int (channel index in sorted keys)
stored_names = json.loads(stored_names_json)
true_order = json.loads(true_order_json).get(task_name)
out_dir = Path(out_dir_str)
out_dir.mkdir(parents=True, exist_ok=True)

zip_path = hf_hub_download(hf_repo, repo_file, repo_type="dataset", revision=hf_rev)

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()

class StubUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module.startswith("geobench"):
            class Dummy(SimpleNamespace):
                def __setstate__(self, st):
                    if isinstance(st, dict): self.__dict__.update(st)
            return Dummy
        return super().find_class(module, name)

def channel_keys(hf):
    return sorted(k for k in hf.keys() if k != "label")

band_check = {"archive": os.path.basename(os.path.realpath(zip_path)), "archive_sha256": sha256(zip_path),
              "hf_repo": hf_repo, "hf_revision": hf_rev}

with zipfile.ZipFile(zip_path, "r") as zf:
    part = json.loads(zf.read(f"{task_name}/default_partition.json"))
    # ---- sample-chip verification of the documented physical band order (m-eurosat, m-brick-kiln)
    if true_order is not None:
        rows = []
        for sid in part["train"][:300]:
            with h5py.File(io.BytesIO(zf.read(f"{task_name}/{sid}.hdf5")), "r") as hf:
                keys = channel_keys(hf)
                assert keys == stored_names, f"unexpected stored channel names: {keys}"
                rows.append(np.stack([hf[k][:] for k in keys]).astype(np.float64))
        X = np.stack(rows)
        fl = X.transpose(1, 0, 2, 3).reshape(X.shape[1], -1)
        C = np.corrcoef(fl)
        def corr(a, b):
            return round(float(C[true_order.index(a), true_order.index(b)]), 4)
        chk = {"stored_names_ok": True, "n_chips": int(X.shape[0]), "true_order": true_order,
               "channel_min_mean_max": [[round(float(X[:, c].min()), 2), round(float(X[:, c].mean()), 2), round(float(X[:, c].max()), 2)] for c in range(X.shape[1])]}
        if task_name == "m-brick-kiln":
            # the last three channels are 8-bit true-colour composites of B04/B03/B02
            chk["corr"] = {"TCI_R~B04": corr("TCI_R", "B04"), "TCI_G~B03": corr("TCI_G", "B03"), "TCI_B~B02": corr("TCI_B", "B02"),
                           "B8A~B08": corr("B8A", "B08"), "B07~B08": corr("B07", "B08"), "B11~B12": corr("B11", "B12")}
            ok = (min(chk["corr"][k] for k in ("TCI_R~B04", "TCI_G~B03", "TCI_B~B02")) > 0.99
                  and X[:, 10:].max() <= 255.0 and chk["corr"]["B8A~B08"] > 0.95)
        else:
            chk["corr"] = {"B8A~B08": corr("B8A", "B08"), "B06~B07": corr("B06", "B07"), "B11~B12": corr("B11", "B12")}
            ok = chk["corr"]["B8A~B08"] > 0.9
        chk["physical_check_passed"] = bool(ok)
        band_check["band_check"] = chk
        assert ok, f"physical band-order check failed for {task_name}: {chk['corr']}"
    for sp_in, sp_out in [("train", "train"), ("valid", "val"), ("test", "test")]:
        ids = part[sp_in]
        xs, ys = [], []
        for sample_id in ids:
            h5_bytes = zf.read(f"{task_name}/{sample_id}.hdf5")
            with h5py.File(io.BytesIO(h5_bytes), "r") as hf:
                hkeys = list(hf.keys())
                ckeys = channel_keys(hf)
                chans = []
                for bspec in band_specs:
                    if isinstance(bspec, int):
                        k = ckeys[bspec]
                    elif bspec.endswith("*"):
                        pref = bspec[:-1]
                        k = sorted([hk for hk in hkeys if hk.startswith(pref)])[0]
                    else:
                        k = bspec
                    chans.append(hf[k][:])
                arr = np.stack(chans, axis=0)
                if arr.dtype == np.float64:
                    arr = arr.astype(np.float32)
                xs.append(arr)
                if kind == "seg":
                    ys.append(hf["label"][:].astype(np.uint8))
                else:
                    raw = hf.attrs["pickle"]
                    b = ast.literal_eval(raw) if isinstance(raw, str) and raw.startswith("b") else raw.encode("latin1")
                    obj = StubUnpickler(io.BytesIO(b)).load()
                    lbl = obj["label"] if isinstance(obj, dict) else getattr(obj, "label")
                    ys.append(np.asarray(lbl, dtype=np.float32) if kind == "multilabel" else int(lbl))
        np.save(out_dir / f"X_{sp_out}.npy", np.stack(xs, axis=0))
        if kind == "seg":
            np.save(out_dir / f"y_{sp_out}.npy", np.stack(ys, axis=0))
        elif kind == "multilabel":
            np.save(out_dir / f"y_{sp_out}.npy", np.stack(ys, axis=0))
        else:
            np.save(out_dir / f"y_{sp_out}.npy", np.array(ys, dtype=np.int64))
(out_dir / "band_check.json").write_text(json.dumps(band_check))
"""


def _limit_splits(splits: dict) -> dict:
    """Keep SMOKE_LIMIT seeded-random samples per split (no-op unless a smoke run set it)."""
    if SMOKE_LIMIT <= 0:
        return splits
    rng = np.random.default_rng(0)
    out = {}
    for sp, (X, y) in splits.items():
        i = np.sort(rng.permutation(len(X))[:SMOKE_LIMIT])
        out[sp] = (X[i], y[i])
    print(f"[smoke] CLAY_SMOKE_LIMIT={SMOKE_LIMIT}: splits cut to {[len(v[0]) for v in out.values()]} (train/val/test); numbers are not results", flush=True)
    return out


def load_geobench_splits(repo_file: str, task_name: str, band_specs: list, kind: str = "cls",
                         is_seg: bool | None = None):
    """Load a GEO-Bench task from the pinned HF revision. `band_specs` entries are HDF5 keys, `prefix*`
    wildcards, or ints (channel index in sorted-key order, used with GB_TRUE_ORDER for the two archives
    whose stored names are wrong). Returns splits; archive identity / band check go to
    `load_geobench_splits.last_meta`. Work dir is per-process."""
    if is_seg is not None:
        kind = "seg" if is_seg else "cls"
    if WORK_DIR.exists():
        shutil.rmtree(WORK_DIR)
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    helper_py = WORK_DIR / "_gb_extract.py"
    helper_py.write_text(GEOBENCH_HELPER)
    env = os.environ.copy()
    env["PIP_CONFIG_FILE"] = "/dev/null"
    subprocess.run(
        [H5_PYTHON, str(helper_py), repo_file, task_name, json.dumps(band_specs), kind, str(WORK_DIR),
         GEOBENCH_HF_REPO, GEOBENCH_HF_REVISION, json.dumps(GB_STORED_NAMES), json.dumps(GB_TRUE_ORDER)],
        check=True, env=env,
    )
    splits = {}
    for sp in ("train", "val", "test"):
        splits[sp] = (np.load(WORK_DIR / f"X_{sp}.npy"), np.load(WORK_DIR / f"y_{sp}.npy"))
    load_geobench_splits.last_meta = json.loads((WORK_DIR / "band_check.json").read_text())
    shutil.rmtree(WORK_DIR)
    return _limit_splits(splits)


load_geobench_splits.last_meta = {}


def load_torchgeo_cls(cls_type, resize_hw: int = 256):
    if WORK_DIR.exists():
        shutil.rmtree(WORK_DIR)
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    splits = {}
    for sp in ("train", "val", "test"):
        ds = cls_type(root=str(WORK_DIR), split=sp, download=(sp == "train"), checksum=False)
        # Memory-lean: fill a preallocated array (no list + np.stack copy). Integral 0..255 RGB imagery
        # (RESISC45; UCMerced chips are not integral-valued after loading (the lossless check failed on them), so it stays float32) is stored
        # as uint8: lossless (checked per chip) and 4x smaller; every consumer casts to float32 per batch.
        arr, ys = None, []
        for i in range(len(ds)):
            item = ds[i]
            img = item["image"]  # (C, H, W)
            if img.shape[-2:] != (resize_hw, resize_hw):
                img = F.interpolate(img[None].float(), size=(resize_hw, resize_hw), mode="bilinear", align_corners=False)[0].to(img.dtype)
            a = img.numpy()
            if arr is None:
                as_u8 = cls_type.__name__ == "RESISC45"
                arr = np.empty((len(ds), *a.shape), dtype=np.uint8 if as_u8 else a.dtype)
            if arr.dtype == np.uint8 and a.dtype != np.uint8:
                u = a.astype(np.uint8)
                assert a.min() >= 0 and a.max() <= 255 and np.array_equal(u, a), "uint8 storage would be lossy"
                a = u
            arr[i] = a
            ys.append(int(item["label"]))
        splits[sp] = (arr, np.array(ys, dtype=np.int64))
    shutil.rmtree(WORK_DIR)
    return _limit_splits(splits)


def load_copernicus_s3(is_biomass: bool):
    from torchgeo.datasets import CopernicusBenchBiomassS3, CopernicusBenchLC100ClsS3
    if WORK_DIR.exists():
        shutil.rmtree(WORK_DIR)
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    cls_type = CopernicusBenchBiomassS3 if is_biomass else CopernicusBenchLC100ClsS3
    splits = {}
    for sp in ("train", "val", "test"):
        ds = cls_type(root=str(WORK_DIR), split=sp, mode="static", download=(sp == "train"), checksum=False)
        if not is_biomass:
            # Align static_fnames-{split}.csv to multilabel-{split}.csv by PID
            # (fixes upstream TorchGeo row-order mismatch on static_fnames-test.csv).
            sf_map = dict(zip(ds.static_files.iloc[:, 0].astype(str), ds.static_files.iloc[:, 1].astype(str)))
            pids = ds.files.iloc[:, 0].astype(str).tolist()
            ds.static_files = ds.static_files.iloc[:len(pids)].copy()
            ds.static_files.iloc[:, 0] = pids
            ds.static_files.iloc[:, 1] = [sf_map[p] for p in pids]
        xs, ys = [], []
        for i in range(len(ds)):
            item = ds[i]
            img = item["image"].float()
            # Clean non-finite (-inf nodata in S3 OLCI) BEFORE bilinear resize!
            arr = img.numpy()[None]
            arr = clean_nonfinite(arr)[0]
            img_t = torch.from_numpy(arr)[None]
            if img_t.shape[-2:] != (96, 96):
                img_t = F.interpolate(img_t, size=(96, 96), mode="bilinear", align_corners=False)
            xs.append(img_t[0].numpy())
            if is_biomass:
                msk = item["mask"].float()
                if msk.ndim == 2:
                    msk = msk[None]
                if msk.shape[-2:] != (282, 282):
                    msk = F.interpolate(msk[None], size=(282, 282), mode="bilinear", align_corners=False)[0]
                ys.append(msk[0].numpy())
            else:
                ys.append(item["label"].numpy().astype(np.int64))
        splits[sp] = (np.stack(xs, axis=0), np.stack(ys, axis=0))
    shutil.rmtree(WORK_DIR)
    return _limit_splits(splits)


# ==============================================================================
# Unified Benchmark Runner Across All 12 Non-Crop Tasks
# ==============================================================================
def build_tasks_cfg():

    from torchgeo.datasets import CopernicusBenchEuroSATS1, RESISC45, UCMerced

    return [
        {
            "id": "m-eurosat",
            "source": "geobench",
            "repo_file": "classification_v1.0/m-eurosat.zip",
            # Physical B02..B08, B8A, B11, B12 selected by channel index from the documented true order
            "bands": gb_channels("m-eurosat", EUROSAT_BANDS),
            "band_names": EUROSAT_BANDS,
            "waves": gb_waves(EUROSAT_BANDS),
            "rgb_idx": [2, 1, 0],
            "gsd": 10.0,
            "clay_hw": 64,
            "also_clay_224": True,
            "kind": "cls",
            "is_uint8": False,
            "is_unit_float": False,
            "clay_meta": ("sentinel-2-l2a", 1.0),
            "sensor_label": "Sentinel-2 (10 m, 10b)",
        },
        {
            "id": "ucmerced",
            "source": "torchgeo_cls",
            "cls_type": UCMerced,
            "resize_hw": 256,
            "waves": RGB_WAVES,
            "rgb_idx": [0, 1, 2],
            "gsd": 0.3,
            "clay_hw": 256,
            "kind": "cls",
            "is_uint8": True,
            "is_unit_float": False,
            "clay_meta": ("linz", 1.0),
            "sensor_label": "USGS Aerial RGB (0.3 m, 3b)",
        },
        {
            "id": "m-pv4ger",
            "source": "geobench",
            "repo_file": "classification_v1.0/m-pv4ger.zip",
            "bands": ["Red", "Green", "Blue"],
            "waves": RGB_WAVES,
            "rgb_idx": [0, 1, 2],
            "gsd": 0.1,
            "clay_hw": 256,
            "kind": "cls",
            "is_uint8": True,
            "is_unit_float": False,
            "clay_meta": ("linz", 1.0),
            "sensor_label": "Aerial RGB (0.1 m, 3b)",
        },
        {
            "id": "m-pv4ger-seg",
            "source": "geobench",
            "repo_file": "segmentation_v1.0/m-pv4ger-seg.zip",
            "bands": ["Red", "Green", "Blue"],
            "waves": RGB_WAVES,
            "rgb_idx": [0, 1, 2],
            "gsd": 0.1,
            "clay_hw": 256,
            "kind": "seg",
            "n_classes": 2,
            "is_uint8": True,
            "is_unit_float": False,
            "clay_meta": ("linz", 1.0),
            "sensor_label": "Aerial RGB (0.1 m, 3b)",
        },
        {
            "id": "resisc45",
            "source": "torchgeo_cls",
            "cls_type": RESISC45,
            "resize_hw": 256,
            "waves": RGB_WAVES,
            "rgb_idx": [0, 1, 2],
            "gsd": 1.0,
            "clay_hw": 256,
            "kind": "cls",
            "is_uint8": True,
            "is_unit_float": False,
            "clay_meta": ("linz", 1.0),
            "sensor_label": "Multi-Scale RGB (0.2-30 m, 3b)",
        },
        {
            "id": "m-chesapeake",
            "source": "geobench",
            "repo_file": "segmentation_v1.0/m-chesapeake.zip",
            "bands": ["Red", "Green", "Blue", "NearInfrared"],
            "waves": NAIP_WAVES,
            "rgb_idx": [0, 1, 2],
            "gsd": 1.0,
            "clay_hw": 256,
            "kind": "seg",
            "n_classes": 7,
            "is_uint8": False,
            "is_unit_float": True,
            "clay_meta": ("naip", 255.0),
            "sensor_label": "USDA NAIP (1.0 m, 4b)",
        },
        {
            "id": "m-brick-kiln",
            "source": "geobench",
            "repo_file": "classification_v1.0/m-brick-kiln.zip",
            # 9 physical reflectance bands (B06 is absent from this archive); the last three stored
            # channels are 8-bit TCI composites and are excluded. Selected by channel index (audit fix #15).
            "bands": gb_channels("m-brick-kiln", KILN_BANDS),
            "band_names": KILN_BANDS,
            "waves": gb_waves(KILN_BANDS),
            "rgb_idx": [2, 1, 0],
            "gsd": 10.0,
            "clay_hw": 64,
            "also_clay_224": True,
            "kind": "cls",
            "is_uint8": False,
            "is_unit_float": False,
            "clay_meta": ("sentinel-2-l2a", 1.0),
            "clay_meta_bands": [CLAY_META_S2_NAME[b] for b in KILN_BANDS],
            "sensor_label": "Sentinel-2 (10 m, 9b: no B06)",
        },
        {
            "id": "m-so2sat",
            "source": "geobench",
            "repo_file": "classification_v1.0/m-so2sat.zip",
            "bands": [
                "02 - Blue", "03 - Green", "04 - Red",
                "05 - Vegetation Red Edge", "06 - Vegetation Red Edge", "07 - Vegetation Red Edge",
                "08 - NIR", "08A - Vegetation Red Edge", "11 - SWIR", "12 - SWIR",
                "06 - VV.LEE Filtered", "05 - VH.LEE Filtered",
            ],
            "waves": S2_10B_WAVES + S1_2B_WAVES,
            "dofa_waves": S2_10B_WAVES + DOFA_S1_2B_WAVES,
            "rgb_idx": [2, 1, 0],
            "gsd": 10.0,
            "clay_hw": 32,
            "also_clay_224": True,
            "kind": "cls",
            "is_uint8": False,
            "is_unit_float": False,
            "so2sat_s1s2": True,
            "clay_meta": None,
            "sensor_label": "Sentinel-2 + S1 (10 m, 10b / 12b)",
        },
        {
            "id": "eurosat-s1",
            "source": "torchgeo_cls",
            "cls_type": CopernicusBenchEuroSATS1,
            "resize_hw": 64,
            "waves": S1_2B_WAVES,
            "dofa_waves": DOFA_S1_2B_WAVES,
            "rgb_idx": None,
            "gsd": 10.0,
            "clay_hw": 64,
            "also_clay_224": True,
            "kind": "cls",
            "is_uint8": False,
            "is_unit_float": False,
            "clay_meta": ("sentinel-1-rtc", 1.0),
            "sensor_label": "Sentinel-1 SAR (10 m, 2b VV/VH)",
        },
        {
            "id": "m-forestnet",
            "source": "geobench",
            "repo_file": "classification_v1.0/m-forestnet.zip",
            "bands": [
                "04 - Red_*", "03 - Green_*", "02 - Blue_*",
                "05 - NIR_*", "06 - SWIR1_*", "07 - SWIR2_*",
            ],
            "waves": L8_6B_WAVES,
            "rgb_idx": [0, 1, 2],
            "gsd": 15.0,
            "clay_hw": 256,
            "kind": "cls",
            "is_uint8": True,
            "is_unit_float": False,
            "clay_meta": None,
            "sensor_label": "Landsat-8 (15 m, 6b)",
        },
        {
            "id": "lc100cls-s3",
            "source": "copernicus_s3",
            "is_biomass": False,
            "waves": S3_OLCI_WAVES,
            "rgb_idx": [7, 5, 3],  # Oa08 (665nm), Oa06 (560nm), Oa04 (490nm)
            "gsd": 300.0,
            "clay_hw": 96,
            "also_clay_224": True,
            "kind": "multilabel",
            "is_uint8": False,
            "is_unit_float": False,
            "clay_meta": None,
            "sensor_label": "Sentinel-3 OLCI (300 m, 21b, unseen)",
        },
        {
            "id": "biomass-s3",
            "source": "copernicus_s3",
            "is_biomass": True,
            "waves": S3_OLCI_WAVES,
            "rgb_idx": [7, 5, 3],
            "gsd": 300.0,
            "clay_hw": 96,
            "kind": "biomass",
            "is_uint8": False,
            "is_unit_float": False,
            "clay_meta": None,
            "sensor_label": "Sentinel-3 OLCI (300 m, 21b, unseen)",
        },
    ]


def run_all(out_json: Path, resume: bool, only: list[str] | None):
    SCRATCH.mkdir(parents=True, exist_ok=True)
    script = str(Path(__file__).resolve())
    results = {"tasks": {}, "meta": {}}
    if resume and out_json.exists():
        results = json.loads(out_json.read_text())
        print(f"[resume] {out_json} holds {sorted(results['tasks'])}", flush=True)
    suite = ModelSuite()
    tasks_cfg = build_tasks_cfg()
    unknown = set(only or []) - {c["id"] for c in tasks_cfg}
    if unknown:
        raise SystemExit(f"unknown task ids: {sorted(unknown)}")

    for cfg in tasks_cfg:
        tid = cfg["id"]
        if only and tid not in only:
            continue
        fp = cfg_fingerprint({k: v for k, v in cfg.items() if k != "cls_type"} | {"cls_type": str(cfg.get("cls_type"))})
        if resume and can_skip(results["tasks"].get(tid), script, fp):
            print(f"[resume-skip] {tid}: stored cell has identical script sha256 and configuration.", flush=True)
            continue
        if tid in results["tasks"]:
            print(f"[stale] {tid}: stored cell was produced by a different script/configuration; recomputing.", flush=True)

        t0 = time.time()
        print(f"\n==================== {tid} ({cfg['sensor_label']}) ====================", flush=True)
        if cfg["source"] == "geobench":
            splits = load_geobench_splits(cfg["repo_file"], tid, cfg["bands"], kind=cfg["kind"])
            gb_meta = dict(load_geobench_splits.last_meta)
        elif cfg["source"] == "torchgeo_cls":
            splits = load_torchgeo_cls(cfg["cls_type"], resize_hw=cfg["resize_hw"])
        elif cfg["source"] == "copernicus_s3":
            splits = load_copernicus_s3(is_biomass=cfg["is_biomass"])
        else:
            raise ValueError(cfg["source"])

        X_tr, y_tr = splits["train"]
        X_va, y_va = splits["val"]
        X_te, y_te = splits["test"]
        del splits

        # Convert Lee-filtered S1 linear power in m-so2sat channels 10..11 to dB
        if cfg.get("so2sat_s1s2"):
            for arr in (X_tr, X_va, X_te):
                arr[:, 10:12] = 10.0 * np.log10(np.clip(arr[:, 10:12], 1e-5, None))

        means_full, stds_full, p01_full, p99_full = compute_train_band_stats(X_tr)
        X_tr = clean_nonfinite(X_tr, means_full)
        X_va = clean_nonfinite(X_va, means_full)
        X_te = clean_nonfinite(X_te, means_full)

        # For m-so2sat, primary multi-spectral arm is 10-band S2; also evaluate 12-band S1+S2 fusion!
        if cfg.get("so2sat_s1s2"):
            X_tr_main, X_va_main, X_te_main = X_tr[:, :10], X_va[:, :10], X_te[:, :10]
            means_main, stds_main = means_full[:10], stds_full[:10]
            waves_main = S2_10B_WAVES
        else:
            X_tr_main, X_va_main, X_te_main = X_tr, X_va, X_te
            means_main, stds_main = means_full, stds_full
            waves_main = cfg["waves"]

        need_pat = cfg["kind"] in ("seg", "biomass")
        h_orig = X_tr_main.shape[-2]
        task_res = {
            "sensor": cfg["sensor_label"],
            "gsd_m": cfg["gsd"],
            "n_bands": int(X_tr_main.shape[1]),
            "orig_hw": int(h_orig),
            "splits": [int(len(X_tr)), int(len(X_va)), int(len(X_te))],
            "kind": cfg["kind"],
            "models": {},
        }
        if cfg["source"] == "geobench":
            task_res["input_contract"] = {
                "bands": cfg.get("band_names", cfg["bands"]),
                "channel_indices_or_keys": cfg["bands"],
                "wavelengths_um_clay": waves_main,
                "wavelengths_um_dofa": waves_main if cfg.get("so2sat_s1s2") else cfg.get("dofa_waves", waves_main),
                **gb_meta,
            }

        def score_features(prefix: str, tr_d: dict, va_d: dict, te_d: dict):
            if cfg["kind"] == "cls":
                for ro in ("cls", "mean"):
                    if ro in tr_d:
                        task_res["models"][f"{prefix}_{ro}"] = eval_single_label(
                            tr_d[ro], y_tr, va_d[ro], y_va, te_d[ro], y_te,
                        )
            elif cfg["kind"] == "multilabel":
                for ro in ("cls", "mean"):
                    if ro in tr_d:
                        task_res["models"][f"{prefix}_{ro}"] = eval_multilabel(
                            tr_d[ro], y_tr, va_d[ro], y_va, te_d[ro], y_te,
                        )
            elif cfg["kind"] == "seg":
                task_res["models"][f"{prefix}_patch"] = eval_dense_segmentation(
                    tr_d["patches"], y_tr, va_d["patches"], y_va, te_d["patches"], y_te, cfg["n_classes"],
                )
            elif cfg["kind"] == "biomass":
                dense_m = eval_dense_regression(
                    tr_d["patches"], y_tr, va_d["patches"], y_va, te_d["patches"], y_te,
                )
                y_tr_m = y_tr.reshape(len(y_tr), -1).mean(1)
                y_va_m = y_va.reshape(len(y_va), -1).mean(1)
                y_te_m = y_te.reshape(len(y_te), -1).mean(1)
                chip_cls = eval_chip_regression(tr_d["cls"], y_tr_m, va_d["cls"], y_va_m, te_d["cls"], y_te_m)
                chip_mean = eval_chip_regression(tr_d["mean"], y_tr_m, va_d["mean"], y_va_m, te_d["mean"], y_te_m)
                task_res["models"][f"{prefix}_patch"] = dense_m
                task_res["models"][f"{prefix}_cls"] = chip_cls
                task_res["models"][f"{prefix}_mean"] = chip_mean

        # 1. Clay v1.5 (train-split per-band z-score parity)
        print(f"  -> Clay v1.5 (hw={cfg['clay_hw']})...", flush=True)
        c_tr = suite.extract_clay(X_tr_main, means_main, stds_main, waves_main, cfg["gsd"], cfg["clay_hw"], need_pat)
        c_va = suite.extract_clay(X_va_main, means_main, stds_main, waves_main, cfg["gsd"], cfg["clay_hw"], need_pat)
        c_te = suite.extract_clay(X_te_main, means_main, stds_main, waves_main, cfg["gsd"], cfg["clay_hw"], need_pat)
        score_features("clay_v15", c_tr, c_va, c_te)
        del c_tr, c_va, c_te
        gc.collect()

        # Diagnostic: Clay upsampled to 224x224 on small (<224) chips to match DOFA/DINOv2 token count
        if cfg.get("also_clay_224"):
            print("  -> Clay v1.5 @224px (token-grid parity diagnostic)...", flush=True)
            cu_tr = suite.extract_clay(X_tr_main, means_main, stds_main, waves_main, cfg["gsd"], 224, False)
            cu_va = suite.extract_clay(X_va_main, means_main, stds_main, waves_main, cfg["gsd"], 224, False)
            cu_te = suite.extract_clay(X_te_main, means_main, stds_main, waves_main, cfg["gsd"], 224, False)
            score_features("clay_v15_224", cu_tr, cu_va, cu_te)
            del cu_tr, cu_va, cu_te
            gc.collect()

        # Diagnostic: Clay fixed metadata.yaml constants where applicable
        if cfg["clay_meta"] is not None:
            plat, mult = cfg["clay_meta"]
            m_info = CLAY_METADATA[plat]
            b_order = list(cfg.get("clay_meta_bands") or m_info["band_order"])
            c_means = np.array([float(m_info["bands"]["mean"][b]) / mult for b in b_order], dtype=np.float32)
            c_stds = np.array([float(m_info["bands"]["std"][b]) / mult for b in b_order], dtype=np.float32)
            c_waves = [float(m_info["bands"]["wavelength"][b]) for b in b_order]
            print(f"  -> Clay v1.5 fixed metadata ({plat})...", flush=True)
            cm_tr = suite.extract_clay(X_tr_main, c_means, c_stds, c_waves, cfg["gsd"], cfg["clay_hw"], need_pat)
            cm_va = suite.extract_clay(X_va_main, c_means, c_stds, c_waves, cfg["gsd"], cfg["clay_hw"], need_pat)
            cm_te = suite.extract_clay(X_te_main, c_means, c_stds, c_waves, cfg["gsd"], cfg["clay_hw"], need_pat)
            score_features("clay_meta", cm_tr, cm_va, cm_te)
            del cm_tr, cm_va, cm_te
            gc.collect()

        # 2. DOFA-L
        dofa_waves_main = cfg.get("dofa_waves", waves_main) if not cfg.get("so2sat_s1s2") else waves_main
        print("  -> DOFA-L (@224px)...", flush=True)
        d_tr = suite.extract_dofa(X_tr_main, means_main, stds_main, dofa_waves_main, need_pat)
        d_va = suite.extract_dofa(X_va_main, means_main, stds_main, dofa_waves_main, need_pat)
        d_te = suite.extract_dofa(X_te_main, means_main, stds_main, dofa_waves_main, need_pat)
        score_features("dofa_l", d_tr, d_va, d_te)
        del d_tr, d_va, d_te
        gc.collect()

        # Optional 12-band S1+S2 single-pass fusion on m-so2sat
        if cfg.get("so2sat_s1s2"):
            print("  -> S1+S2 12-band single-pass fusion (Clay & DOFA)...", flush=True)
            cs_tr = suite.extract_clay(X_tr, means_full, stds_full, cfg["waves"], cfg["gsd"], 64, False)
            cs_va = suite.extract_clay(X_va, means_full, stds_full, cfg["waves"], cfg["gsd"], 64, False)
            cs_te = suite.extract_clay(X_te, means_full, stds_full, cfg["waves"], cfg["gsd"], 64, False)
            score_features("clay_s1s2_64", cs_tr, cs_va, cs_te)
            del cs_tr, cs_va, cs_te
            dofa_w_12 = cfg.get("dofa_waves", cfg["waves"])
            ds_tr = suite.extract_dofa(X_tr, means_full, stds_full, dofa_w_12, False)
            ds_va = suite.extract_dofa(X_va, means_full, stds_full, dofa_w_12, False)
            ds_te = suite.extract_dofa(X_te, means_full, stds_full, dofa_w_12, False)
            score_features("dofa_s1s2", ds_tr, ds_va, ds_te)
            del ds_tr, ds_va, ds_te
            gc.collect()

        # 3. Scale-MAE-L & DINOv2-L (optical tasks with RGB channels)
        if cfg["rgb_idx"] is not None:
            rgb_tr = to_rgb01(X_tr_main, cfg["rgb_idx"], cfg["is_uint8"], cfg["is_unit_float"], p01_full, p99_full)
            rgb_va = to_rgb01(X_va_main, cfg["rgb_idx"], cfg["is_uint8"], cfg["is_unit_float"], p01_full, p99_full)
            rgb_te = to_rgb01(X_te_main, cfg["rgb_idx"], cfg["is_uint8"], cfg["is_unit_float"], p01_full, p99_full)

            print("  -> Scale-MAE-L (@224px)...", flush=True)
            s_tr = suite.extract_scalemae(rgb_tr, cfg["gsd"], h_orig, need_pat)
            s_va = suite.extract_scalemae(rgb_va, cfg["gsd"], h_orig, need_pat)
            s_te = suite.extract_scalemae(rgb_te, cfg["gsd"], h_orig, need_pat)
            score_features("scalemae_l", s_tr, s_va, s_te)
            del s_tr, s_va, s_te
            gc.collect()

            print("  -> DINOv2-L (@224px)...", flush=True)
            v_tr = suite.extract_dinov2(rgb_tr, need_pat)
            v_va = suite.extract_dinov2(rgb_va, need_pat)
            v_te = suite.extract_dinov2(rgb_te, need_pat)
            score_features("dinov2_l", v_tr, v_va, v_te)
            del v_tr, v_va, v_te, rgb_tr, rgb_va, rgb_te
            gc.collect()

        # 4. Non-learned Band Stats baseline
        print("  -> Band Stats control...", flush=True)
        if cfg["kind"] == "cls":
            task_res["models"]["band_stats"] = eval_single_label(
                extract_band_stats(X_tr_main), y_tr,
                extract_band_stats(X_va_main), y_va,
                extract_band_stats(X_te_main), y_te,
            )
        elif cfg["kind"] == "multilabel":
            task_res["models"]["band_stats"] = eval_multilabel(
                extract_band_stats(X_tr_main), y_tr,
                extract_band_stats(X_va_main), y_va,
                extract_band_stats(X_te_main), y_te,
            )
        elif cfg["kind"] == "seg":
            # Per-pixel raw spectral channels (transposed to B, H, W, C)
            task_res["models"]["band_stats"] = eval_dense_segmentation(
                np.transpose(X_tr_main, (0, 2, 3, 1)), y_tr,
                np.transpose(X_va_main, (0, 2, 3, 1)), y_va,
                np.transpose(X_te_main, (0, 2, 3, 1)), y_te,
                cfg["n_classes"],
            )
        elif cfg["kind"] == "biomass":
            b_dense = eval_dense_regression(
                np.transpose(X_tr_main, (0, 2, 3, 1)), y_tr,
                np.transpose(X_va_main, (0, 2, 3, 1)), y_va,
                np.transpose(X_te_main, (0, 2, 3, 1)), y_te,
            )
            y_tr_m = y_tr.reshape(len(y_tr), -1).mean(1)
            y_va_m = y_va.reshape(len(y_va), -1).mean(1)
            y_te_m = y_te.reshape(len(y_te), -1).mean(1)
            b_chip = eval_chip_regression(
                extract_band_stats(X_tr_main), y_tr_m,
                extract_band_stats(X_va_main), y_va_m,
                extract_band_stats(X_te_main), y_te_m,
            )
            task_res["models"]["band_stats"] = {**b_dense, **b_chip}

        task_res["elapsed_s"] = round(time.time() - t0, 1)
        task_res["producer"] = {"script_sha256": sha256_file(script), "cfg": fp, "utc_timestamp": utc_now()}
        results["tasks"][tid] = task_res
        results["meta"] = provenance(script, {
            "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1e9, 3),
            "clay_ckpt_sha256": CLAY_SHA256,
            "geobench_hf_repo": GEOBENCH_HF_REPO,
            "geobench_hf_revision": GEOBENCH_HF_REVISION,
            **({"smoke_limit": SMOKE_LIMIT} if SMOKE_LIMIT else {}),
        })
        out_json.write_text(json.dumps(results, indent=2) + "\n")

        # Print concise summary line
        summary_pairs = []
        for mk, mv in task_res["models"].items():
            for metric_key in ("oa", "miou", "macro_map", "pixel_r2"):
                if metric_key in mv:
                    summary_pairs.append(f"{mk}={mv[metric_key]}")
                    break
        print(f"[done {tid} in {task_res['elapsed_s']}s] " + " | ".join(summary_pairs), flush=True)

        del X_tr, y_tr, X_va, y_va, X_te, y_te, X_tr_main, X_va_main, X_te_main
        gc.collect()

    print("\nALL 12 STANDARD BENCHMARKS COMPLETE!", flush=True)


def verify_bands_only() -> None:
    """Run only the sample-chip physical band-order check for m-brick-kiln and m-eurosat (CPU)."""
    for tid, bands in (("m-brick-kiln", KILN_BANDS), ("m-eurosat", EUROSAT_BANDS)):
        load_geobench_splits(f"classification_v1.0/{tid}.zip", tid, gb_channels(tid, bands), kind="cls")
        chk = load_geobench_splits.last_meta["band_check"]
        print(tid, json.dumps({k: chk[k] for k in ("corr", "physical_check_passed", "n_chips")}), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output-dir", default=None, help="default: fresh experiments/runs/standard_benchmarks_<UTC>/")
    ap.add_argument("--resume", action="store_true", help="continue an existing --output-dir (skips only identical-producer cells)")
    ap.add_argument("--tasks", default=None, help="comma-separated task ids (default: all 12)")
    ap.add_argument("--verify-bands-only", action="store_true")
    ap.add_argument("--limit", type=int, default=None, help="smoke test only: keep N samples per split (see SMOKE_LIMIT)")
    a = ap.parse_args()
    if a.limit is not None:
        SMOKE_LIMIT = a.limit
    if a.verify_bands_only:
        verify_bands_only()
    else:
        out = resolve_output(a.output_dir, a.resume, "standard_benchmarks", "standard_benchmarks.json")
        print(f"[output] {out}", flush=True)
        run_all(out, a.resume, a.tasks.split(",") if a.tasks else None)
