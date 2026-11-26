#!/usr/bin/env python3
"""Clean, deterministic, leak-free primary evaluation suite for Clay v1.5.

All experiments execute directly against the SHA-256-verified public Clay v1.5
checkpoint (`clay-v1.5.ckpt`), the model source of `Clay-foundation/model`
(commit `f14e698f3c237cabf8d28dec669a362d66625381`, vendored in
experiments/suite/common/claymodel and verified by SHA-256 of the file bodies), and
public Earth observation datasets (Copernicus Sentinel-2 L2A, USGS NLCD
fractional impervious / tree-canopy products, EuroSAT MS, NASA EMIT wavelength
schedule).

Run (inputs are built by `build_lesson_inputs.py`; see its docstring):
    python experiments/run_clay_v15_experiments.py [--latlon-order documented|training] [--out-name NAME]
Environment overrides: CLAY_V15_CKPT, CLAY_MODEL_REPO, CLAY_S2_NLCD_ROOT, CLAY_EUROSAT_DIR,
CLAY_EUROSAT_ZIP, CLAY_EUROSAT_RESULTS.

Strict ML & reproducibility guarantees:
  1. Weights loaded via `torch.load(..., weights_only=True, mmap=True)` after
     full SHA-256 verification.
  2. Strict spatial cross-validation: single-linkage Haversine clustering at a
     30 km threshold guarantees that the minimum distance between ANY training
     chip and ANY test chip across all 5 folds exceeds 30 km.
  3. Zero preprocessing leakage: all `StandardScaler`, `PCA`, `RidgeCV`, and
     `LogisticRegressionCV` estimators sit inside `sklearn.pipeline.Pipeline`
     and are fitted strictly within each training fold.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import platform
import sys
import zipfile
import time
import warnings
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import pandas as pd
import rasterio
from rasterio.io import MemoryFile
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegressionCV, RidgeCV
from sklearn.metrics import accuracy_score, roc_auc_score, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn.functional as F
import yaml

# ---------------------------------------------------------------------------
# Pinned paths & cryptographic contracts
# ---------------------------------------------------------------------------
WORK_ROOT = Path(__file__).resolve().parent.parent
MODEL_REPO = Path(os.environ.get("CLAY_MODEL_REPO", str(WORK_ROOT / "experiments/suite/common")))
CKPT_PATH = Path(os.environ.get("CLAY_V15_CKPT", str(WORK_ROOT / "checkpoints/v1.5/clay-v1.5.ckpt")))
# One chip root shared with run_probe_and_lsvm_benchmark.py; built by build_lesson_inputs.py s2-nlcd.
S2_ROOT = Path(os.environ.get("CLAY_S2_NLCD_ROOT", str(WORK_ROOT / "data/s2_nlcd_320")))
EUROSAT_DIR = Path(os.environ.get("CLAY_EUROSAT_DIR", str(WORK_ROOT / "data/eurosat")))  # images.json (build_lesson_inputs.py eurosat)
EUROSAT_ZIP = Path(os.environ.get("CLAY_EUROSAT_ZIP", str(WORK_ROOT / "data/EuroSAT_MS.zip")))
EUROSAT_AWS_RESULTS = Path(os.environ.get("CLAY_EUROSAT_RESULTS", str(WORK_ROOT / "experiments/results/eurosat-v1.1")))
OUT_DIR = WORK_ROOT / "experiments/results"

# The model repo directory is vendored (not an independent git checkout), so `git rev-parse` there returns the
# paper repo's HEAD. Provenance is checked instead by SHA-256 of the file bodies (vendored header stripped).
UPSTREAM_MODEL_COMMIT = "f14e698f3c237cabf8d28dec669a362d66625381"
REQUIRED_SOURCE_BODY_SHA256 = {
    "claymodel/model.py": "ebf06103d64a8746509f8a4f900bbebf35eb4b54482bcaeecb953de0df81dcf1",
    "claymodel/factory.py": "11dd00918a88fa076bea4739ce889233c0444ea459ebb2d38bb8bec90e1294c7",
    "claymodel/backbone.py": "f9ef3e410be8736f60d55cb3a8ed6c2cdf3b4e7972fa4716d410618514d3a4bf",
    "claymodel/utils.py": "007e4b8eede28f96132e3d5a34735fcf1fdb03a1940f3fa550689404549e63ca",
    "configs/metadata.yaml": "ad94c587861bf61b447db40d38f9781d36c5a20121cf0211a85cc6fb5b93d356",
}
VENDOR_MARKER = b"--- end of vendored-file header"
LATLON_ORDER = "documented"  # set from --latlon-order in main(); see normalize_latlon

REQUIRED_CKPT_SHA256 = "21432069250b9b3f9a65ffd0071c5ad56b793247285ab0604edf7f531d4798d0"
EXPECTED_ENCODER_TENSORS = 265
EXPECTED_ENCODER_PARAMS = 311_442_944

S2_BANDS_12 = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12"]
CLAY_S2_BANDS_12_IDX = [1, 2, 3, 4, 5, 6, 7, 8, 10, 11]  # B02..B08, B8A, B11, B12
CLAY_S2_BAND_NAMES = [
    "blue", "green", "red", "rededge1", "rededge2",
    "rededge3", "nir", "nir08", "swir16", "swir22",
]
# EuroSAT 13-band GeoTIFF 0-based indices for Clay's 10 S2 bands (B01..B12, B8A at idx 12)
EUROSAT_CLAY_IDX = [1, 2, 3, 4, 5, 6, 7, 12, 10, 11]

SEED = 42


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def normalize_timestamp(iso_str: str) -> np.ndarray:
    d = dt.datetime.fromisoformat(str(iso_str).replace("Z", "+00:00"))
    week = d.isocalendar().week * 2.0 * math.pi / 52.0
    hour = d.hour * 2.0 * math.pi / 24.0
    return np.array(
        [math.sin(week), math.cos(week), math.sin(hour), math.cos(hour)],
        dtype=np.float32,
    )


def normalize_latlon(lat: float, lon: float) -> np.ndarray:
    """Location slots fed to Clay. "documented" = [sin lat, cos lat, sin lon, cos lon] (as the docs describe);
    "training" = [sin lon, cos lon, sin lat, cos lat] (the order Clay v1.0/v1.5 were trained with and the order used by
    experiments/suite/common/clayrt.py and the released embedding products). The paper's perturbation experiments used
    "documented"; pass --latlon-order training for the sensitivity run."""
    la = math.radians(float(lat))
    lo = math.radians(float(lon))
    if LATLON_ORDER == "training":
        return np.array([math.sin(lo), math.cos(lo), math.sin(la), math.cos(la)], dtype=np.float32)
    return np.array([math.sin(la), math.cos(la), math.sin(lo), math.cos(lo)], dtype=np.float32)


def haversine_matrix_km(lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
    la = np.radians(lats)[:, None]
    lo = np.radians(lons)[:, None]
    dla = la - la.T
    dlo = lo - lo.T
    a = np.sin(dla / 2.0) ** 2 + np.cos(la) * np.cos(la.T) * np.sin(dlo / 2.0) ** 2
    return 2.0 * 6371.0088 * np.arcsin(np.clip(np.sqrt(a), 0.0, 1.0))


def assign_spatial_clusters(lats: np.ndarray, lons: np.ndarray, min_gap_km: float = 30.0):
    """Single-linkage clustering at `min_gap_km` guarantees >min_gap_km between clusters."""
    dmat = haversine_matrix_km(lats, lons)
    condensed = dmat[np.triu_indices(len(lats), k=1)]
    Z = linkage(condensed, method="single")
    clusters = fcluster(Z, t=min_gap_km, criterion="distance")
    gkf = GroupKFold(n_splits=5)
    folds = np.empty(len(lats), dtype=np.int64)
    min_cross_km = float("inf")
    for fold_idx, (tr, te) in enumerate(gkf.split(np.zeros(len(lats)), groups=clusters)):
        folds[te] = fold_idx
        cross_min = float(dmat[np.ix_(tr, te)].min())
        min_cross_km = min(min_cross_km, cross_min)
    if min_cross_km <= min_gap_km:
        raise RuntimeError(f"Spatial buffer violated: {min_cross_km:.2f} km <= {min_gap_km} km")
    return clusters, folds, min_cross_km


def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    Xc = X - X.mean(axis=0, keepdims=True)
    Yc = Y - Y.mean(axis=0, keepdims=True)
    num = np.linalg.norm(Xc.T @ Yc, ord="fro") ** 2
    den = np.linalg.norm(Xc.T @ Xc, ord="fro") * np.linalg.norm(Yc.T @ Yc, ord="fro")
    return float(num / max(den, 1e-12))


def ridge_oof_r2(
    X: np.ndarray,
    y: np.ndarray,
    folds: np.ndarray,
    clusters: np.ndarray,
    n_boot: int = 1000,
    seed: int = SEED,
) -> dict:
    alphas = np.logspace(-2, 6, 33)
    oof = np.empty_like(y, dtype=np.float64)
    models = []
    for f in range(int(folds.max()) + 1):
        tr = folds != f
        te = folds == f
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("ridge", RidgeCV(alphas=alphas)),
        ])
        pipe.fit(X[tr], y[tr])
        oof[te] = pipe.predict(X[te])
        models.append((f, pipe))
    r2 = float(r2_score(y, oof))
    if n_boot <= 0:
        return {"r2": r2, "oof": oof, "models": models}
    rng = np.random.default_rng(seed)
    uniq_c = np.unique(clusters)
    idx_by_c = [np.where(clusters == c)[0] for c in uniq_c]
    boots = []
    for _ in range(n_boot):
        sampled = rng.integers(0, len(uniq_c), size=len(uniq_c))
        b_idx = np.concatenate([idx_by_c[i] for i in sampled])
        if np.std(y[b_idx]) > 1e-8:
            boots.append(r2_score(y[b_idx], oof[b_idx]))
    ci_lo, ci_hi = np.percentile(boots, [2.5, 97.5])
    return {
        "r2": r2,
        "ci95": [float(ci_lo), float(ci_hi)],
        "oof": oof,
        "models": models,
    }


def eval_unchanged_models(models, X_pert: np.ndarray, y: np.ndarray, folds: np.ndarray) -> float:
    oof = np.empty_like(y, dtype=np.float64)
    for f, pipe in models:
        te = folds == f
        oof[te] = pipe.predict(X_pert[te])
    return float(r2_score(y, oof))


def cosine_rows(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    An = A / np.clip(np.linalg.norm(A, axis=1, keepdims=True), 1e-12, None)
    Bn = B / np.clip(np.linalg.norm(B, axis=1, keepdims=True), 1e-12, None)
    return np.sum(An * Bn, axis=1)


def top1_retrieval(query: np.ndarray, gallery: np.ndarray) -> float:
    qn = query / np.clip(np.linalg.norm(query, axis=1, keepdims=True), 1e-12, None)
    gn = gallery / np.clip(np.linalg.norm(gallery, axis=1, keepdims=True), 1e-12, None)
    sims = qn @ gn.T
    nn_idx = np.argmax(sims, axis=1)
    return float(np.mean(nn_idx == np.arange(len(query))))


# ---------------------------------------------------------------------------
# Step 0: Verify and load canonical Clay v1.5 encoder
# ---------------------------------------------------------------------------
def load_verified_clay():
    t0 = time.perf_counter()
    source_hashes = {}
    for rel, want in REQUIRED_SOURCE_BODY_SHA256.items():
        raw = (MODEL_REPO / rel).read_bytes()
        body = raw.split(VENDOR_MARKER, 1)[1].split(b"\n", 1)[1] if VENDOR_MARKER in raw else raw
        got = hashlib.sha256(body).hexdigest()
        if got != want:
            raise ValueError(f"{rel}: body sha256 {got} != pinned upstream {want} (commit {UPSTREAM_MODEL_COMMIT})")
        source_hashes[rel] = got

    ckpt_sha = sha256_file(CKPT_PATH)
    if ckpt_sha != REQUIRED_CKPT_SHA256:
        raise ValueError(f"Checkpoint SHA256 mismatch: {ckpt_sha}")

    sys.path.insert(0, str(MODEL_REPO))
    from claymodel.model import Encoder  # noqa: PLC0415

    torch.set_float32_matmul_precision("highest")

    ckpt = torch.load(str(CKPT_PATH), map_location="cpu", weights_only=True, mmap=True)
    sd = ckpt["state_dict"]
    prefix_counts = {}
    for k in sd:
        p = ".".join(k.split(".")[:2])
        prefix_counts[p] = prefix_counts.get(p, 0) + 1

    enc_state = {
        k.removeprefix("model.encoder."): v
        for k, v in sd.items()
        if k.startswith("model.encoder.")
    }
    if len(enc_state) != EXPECTED_ENCODER_TENSORS:
        raise ValueError(f"Expected {EXPECTED_ENCODER_TENSORS} tensors, got {len(enc_state)}")

    enc_cpu = Encoder(
        mask_ratio=0.0, patch_size=8, shuffle=False,
        dim=1024, depth=24, heads=16, dim_head=64, mlp_ratio=4.0,
    )
    enc_cpu.load_state_dict(enc_state, strict=True)
    enc_cpu.eval().requires_grad_(False)
    n_params = sum(p.numel() for p in enc_cpu.parameters())
    if n_params != EXPECTED_ENCODER_PARAMS:
        raise ValueError(f"Expected {EXPECTED_ENCODER_PARAMS} params, got {n_params}")

    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    enc_dev = Encoder(
        mask_ratio=0.0, patch_size=8, shuffle=False,
        dim=1024, depth=24, heads=16, dim_head=64, mlp_ratio=4.0,
    )
    enc_dev.load_state_dict(enc_state, strict=True)
    enc_dev.eval().requires_grad_(False).to(device)

    metadata = yaml.safe_load((MODEL_REPO / "configs/metadata.yaml").read_text())
    saved_meta = {k: ckpt.get(k) for k in ("epoch", "global_step", "pytorch-lightning_version", "hyper_parameters")}

    info = {
        "upstream_model_commit": UPSTREAM_MODEL_COMMIT,
        "source_files_body_sha256": source_hashes,
        "checkpoint_path": CKPT_PATH.name,  # file name only: no machine-specific path in results
        "checkpoint_bytes": CKPT_PATH.stat().st_size,
        "checkpoint_sha256": ckpt_sha,
        "total_state_dict_tensors": len(sd),
        "state_dict_prefix_counts": prefix_counts,
        "encoder_tensors": len(enc_state),
        "encoder_parameters": n_params,
        "saved_checkpoint_header": {
            "epoch": saved_meta.get("epoch"),
            "global_step": saved_meta.get("global_step"),
            "pytorch_lightning_version": saved_meta.get("pytorch-lightning_version"),
            "teacher": saved_meta.get("hyper_parameters", {}).get("teacher"),
            "dolls": saved_meta.get("hyper_parameters", {}).get("dolls"),
            "doll_weights": saved_meta.get("hyper_parameters", {}).get("doll_weights"),
            "lr": saved_meta.get("hyper_parameters", {}).get("lr"),
            "mask_ratio_train": saved_meta.get("hyper_parameters", {}).get("mask_ratio"),
            "norm_pix_loss": saved_meta.get("hyper_parameters", {}).get("norm_pix_loss"),
        },
        "runtime_device": device,
        "python_version": sys.version.split()[0],
        "torch_version": torch.__version__,
        "platform": platform.platform(),
        "load_seconds": round(time.perf_counter() - t0, 3),
    }
    return enc_cpu, enc_dev, device, metadata, info


@torch.inference_mode()
def forward_tokens(
    enc,
    pixels_norm: np.ndarray,
    times: np.ndarray,
    latlons: np.ndarray,
    waves_um: list[float] | np.ndarray,
    gsd_m: float,
    device: str,
    batch_size: int = 32,
    return_patches: bool = False,
):
    N = len(pixels_norm)
    cls_out = []
    mean_out = []
    patch_out = [] if return_patches else None
    w_t = torch.tensor(waves_um, dtype=torch.float32, device=device)
    g_t = torch.tensor([float(gsd_m)], dtype=torch.float32, device=device)
    for i in range(0, N, batch_size):
        cube = {
            "pixels": torch.from_numpy(pixels_norm[i : i + batch_size]).to(device, dtype=torch.float32),
            "time": torch.from_numpy(times[i : i + batch_size]).to(device, dtype=torch.float32),
            "latlon": torch.from_numpy(latlons[i : i + batch_size]).to(device, dtype=torch.float32),
            "waves": w_t,
            "gsd": g_t,
        }
        tokens, _, masked_idx, mask_mat = enc(cube)
        if masked_idx.numel() != 0 or mask_mat.count_nonzero().item() != 0:
            raise RuntimeError("Masking unexpectedly active at inference")
        tok_np = tokens.cpu().numpy().astype(np.float32)
        cls_out.append(tok_np[:, 0, :])
        mean_out.append(tok_np[:, 1:, :].mean(axis=1))
        if return_patches:
            patch_out.append(tok_np[:, 1:, :])
    cls_arr = np.concatenate(cls_out, axis=0)
    mean_arr = np.concatenate(mean_out, axis=0)
    if return_patches:
        return cls_arr, mean_arr, np.concatenate(patch_out, axis=0)
    return cls_arr, mean_arr


# ---------------------------------------------------------------------------
# Experiment A: DynamicEmbedding Wavelength Hypernetwork Anatomy
# ---------------------------------------------------------------------------
@torch.inference_mode()
def run_exp_a_wavelength_hypernetwork(enc_cpu, metadata: dict) -> dict:
    from claymodel.utils import posemb_sincos_1d  # noqa: PLC0415

    pe_mod = enc_cpu.patch_embedding

    def get_filter_matrix(waves_um: np.ndarray) -> np.ndarray:
        w = torch.tensor(waves_um, dtype=torch.float32)
        w_enc = posemb_sincos_1d(w, pe_mod.wave_dim)
        w_enc = pe_mod.fclayer(w_enc)
        weight, _ = pe_mod.weight_generator(w_enc)  # [C_in, 1024*8*8]
        return (weight * 0.02).cpu().numpy().astype(np.float64)

    # Canonical NASA EMIT 285-channel VSWIR schedule (381.0 to 2493.96 nm, 7.44 nm spacing)
    emit_um = np.linspace(0.3810, 2.49396, 285)
    emit_nm = emit_um * 1000.0
    M_emit = get_filter_matrix(emit_um)  # (285, 65536)

    # Uncentered SVD
    _, s_raw, _ = np.linalg.svd(M_emit, full_matrices=False)
    e_raw = s_raw**2
    p_raw = e_raw / e_raw.sum()
    cum_raw = np.cumsum(p_raw)
    eff_rank_shannon_raw = float(np.exp(-np.sum(p_raw * np.log(np.clip(p_raw, 1e-30, None)))))
    part_ratio_raw = float((e_raw.sum() ** 2) / np.sum(e_raw**2))

    # Wavelength-centered SVD (removing broadband DC albedo mode)
    M_cent = M_emit - M_emit.mean(axis=0, keepdims=True)
    U_c, s_cent, _ = np.linalg.svd(M_cent, full_matrices=False)
    e_cent = s_cent**2
    p_cent = e_cent / e_cent.sum()
    cum_cent = np.cumsum(p_cent)
    eff_rank_shannon_cent = float(np.exp(-np.sum(p_cent * np.log(np.clip(p_cent, 1e-30, None)))))
    part_ratio_cent = float((e_cent.sum() ** 2) / np.sum(e_cent**2))

    # Spectral impulse response kernel K(lambda_i, lambda_j) on centered filters
    M_norm = M_cent / np.clip(np.linalg.norm(M_cent, axis=1, keepdims=True), 1e-12, None)
    K_cent = M_norm @ M_norm.T  # (285, 285)

    # Also compute FWHM on centered kernel at 550, 850, 1650, 2300 nm
    probe_nms = [550.0, 850.0, 1650.0, 2300.0]
    fwhm_by_nm = {}
    impulse_slices = {}
    for target_nm in probe_nms:
        idx = int(np.argmin(np.abs(emit_nm - target_nm)))
        prof = K_cent[idx]
        # Half-maximum width around idx where correlation stays >= 0.5
        left = idx
        while left > 0 and prof[left] >= 0.5:
            left -= 1
        right = idx
        while right < len(prof) - 1 and prof[right] >= 0.5:
            right += 1
        fwhm_nm = float(emit_nm[right] - emit_nm[left])
        fwhm_by_nm[str(int(target_nm))] = round(fwhm_nm, 1)
        impulse_slices[str(int(target_nm))] = [round(float(x), 4) for x in prof]

    # Mode zero-crossing / autocorrelation length of top 5 centered left singular vectors U_c[:, k]
    mode_autocorr_fwhm_nm = {}
    for k in range(5):
        u = U_c[:, k]
        u = u - u.mean()
        ac = np.correlate(u, u, mode="full")[len(u) - 1 :]
        ac = ac / max(ac[0], 1e-12)
        lag = int(np.argmax(ac < 0.5)) if np.any(ac < 0.5) else len(ac) - 1
        # Full width = 2 * half-width lag * 7.44 nm
        mode_autocorr_fwhm_nm[f"mode_{k+1}"] = round(float(2 * lag * 7.44), 1)

    # Spectral passband frequency-response curve: zero-mean local Gabor/wavelet packets
    # centered at 2,300 nm (SWIR2 CH4 window) and 1,100 nm with varying feature width FWHM (nm)
    widths_nm = [10.0, 15.0, 25.0, 40.0, 60.0, 100.0, 160.0, 250.0, 400.0]
    gain_vs_width_2300 = {}
    gain_vs_width_850 = {}
    ref_gain_dc = float(np.linalg.norm(M_emit.T @ (np.ones(285) / np.sqrt(285))))
    for w_nm in widths_nm:
        for center_nm, store in [(2300.0, gain_vs_width_2300), (850.0, gain_vs_width_850)]:
            sigma = w_nm / 2.35482
            # Zero-mean second-derivative Mexican-hat (Ricker) absorption feature of width w_nm
            z = (emit_nm - center_nm) / sigma
            sig = (1.0 - z**2) * np.exp(-0.5 * z**2)
            sig = sig - sig.mean()
            sig = sig / np.linalg.norm(sig)
            resp = float(np.linalg.norm(M_emit.T @ sig))
            store[str(int(w_nm))] = {
                "raw_gain": round(resp, 4),
                "relative_to_dc": round(resp / ref_gain_dc, 4),
            }

    # Explicit CH4 2,300 nm narrowband comb (15 nm alternating lines) vs smooth 250 nm SWIR slope
    env = np.exp(-0.5 * ((emit_nm - 2300.0) / 35.0) ** 2)
    ch4_comb = env * np.cos(2.0 * math.pi * (emit_nm - 2300.0) / 22.0)
    ch4_comb -= ch4_comb.mean()
    ch4_comb /= np.linalg.norm(ch4_comb)
    swir_broad = np.exp(-0.5 * ((emit_nm - 2200.0) / 180.0) ** 2)
    swir_broad -= swir_broad.mean()
    swir_broad /= np.linalg.norm(swir_broad)
    ch4_gain = float(np.linalg.norm(M_emit.T @ ch4_comb))
    broad_gain = float(np.linalg.norm(M_emit.T @ swir_broad))

    # Sensor rank summary across Clay's pretraining platforms + EMIT
    sensor_schedules = {
        "LINZ (3 bands)": list(metadata["linz"]["bands"]["wavelength"].values()),
        "NAIP (4 bands)": list(metadata["naip"]["bands"]["wavelength"].values()),
        "Landsat (6 bands)": list(metadata["landsat-c2l2-sr"]["bands"]["wavelength"].values()),
        "MODIS (7 bands)": list(metadata["modis"]["bands"]["wavelength"].values()),
        "Sentinel-2 (10 bands)": list(metadata["sentinel-2-l2a"]["bands"]["wavelength"].values()),
        "Sentinel-1 SAR (2 pseudo-bands)": list(metadata["sentinel-1-rtc"]["bands"]["wavelength"].values()),
        "EMIT (285 bands)": emit_um.tolist(),
    }
    sensor_ranks = {}
    for s_name, w_list in sensor_schedules.items():
        Ms = get_filter_matrix(np.array(w_list, dtype=np.float64))
        _, sv, _ = np.linalg.svd(Ms, full_matrices=False)
        ev = sv**2
        pv = ev / ev.sum()
        er = float(np.exp(-np.sum(pv * np.log(np.clip(pv, 1e-30, None)))))
        k99 = int(np.searchsorted(np.cumsum(pv), 0.99) + 1)
        sensor_ranks[s_name] = {
            "n_bands": len(w_list),
            "shannon_eff_rank": round(er, 3),
            "k99": min(k99, len(w_list)),
        }

    # Check S1 pseudo-wavelengths (3.5, 4.0) similarity to optical/SWIR.
    # Issue #22: the manuscript quotes "mean |cos| = 0.044" for this comparison; no stored run produced it. The block below
    # evaluates every plausible definition on the real checkpoint and stores all of them, so the text can name the one it uses.
    s2_waves_list = list(metadata["sentinel-2-l2a"]["bands"]["wavelength"].values())
    s1_waves_list = [3.5, 4.0]
    M_s1 = get_filter_matrix(np.array(s1_waves_list))
    M_s2 = get_filter_matrix(np.array(s2_waves_list))

    def _rn(M):
        return M / np.clip(np.linalg.norm(M, axis=1, keepdims=True), 1e-12, None)

    def _cos_stats(A, B):
        C = _rn(A) @ _rn(B).T
        return {"mean_abs_cos": round(float(np.mean(np.abs(C))), 4), "mean_signed_cos": round(float(np.mean(C)), 4),
                "max_abs_cos": round(float(np.max(np.abs(C))), 4)}

    s1_n = _rn(M_s1)
    s2_n = _rn(M_s2)
    s1_s2_cos = (s1_n @ s2_n.T).tolist()
    M_joint = get_filter_matrix(np.array(s1_waves_list + s2_waves_list))  # one 12-band forward: attention sees both sets
    M_joint_c = M_joint - M_joint.mean(axis=0, keepdims=True)
    mu_s2 = M_s2.mean(axis=0, keepdims=True)
    mu_emit = M_emit.mean(axis=0, keepdims=True)
    Qs2, _ = np.linalg.qr(M_s2.T)  # orthonormal basis of span(S2 filters)
    in_span = [float(np.sum((Qs2 @ (Qs2.T @ v)) ** 2) / np.sum(v**2)) for v in M_s1]

    def _enc(w):
        return posemb_sincos_1d(torch.tensor(np.asarray(w), dtype=torch.float32), pe_mod.wave_dim).double().numpy()

    s1_vs_s2_definitions = {
        "stored_definition_uncentered_each_set_in_own_context": _cos_stats(M_s1, M_s2),
        "uncentered_joint_12band_context": _cos_stats(M_joint[:2], M_joint[2:]),
        "centred_each_set_on_own_band_mean": _cos_stats(M_s1 - M_s1.mean(0, keepdims=True), M_s2 - M_s2.mean(0, keepdims=True)),
        "centred_joint_context_on_joint_band_mean": _cos_stats(M_joint_c[:2], M_joint_c[2:]),
        "both_centred_on_S2_band_mean": _cos_stats(M_s1 - mu_s2, M_s2 - mu_s2),
        "both_centred_on_EMIT285_band_mean": _cos_stats(M_s1 - mu_emit, M_s2 - mu_emit),
        "row_centred_pearson_over_65536_weights": _cos_stats(M_s1 - M_s1.mean(1, keepdims=True), M_s2 - M_s2.mean(1, keepdims=True)),
        "raw_sincos_wavelength_encodings": _cos_stats(_enc(s1_waves_list), _enc(s2_waves_list)),
        "s2_vs_s2_within_optical_uncentred_offdiag_mean_abs": round(float(np.mean(np.abs(
            (s2_n @ s2_n.T)[np.triu_indices(len(s2_waves_list), 1)]))), 4),
        "random_gaussian_65536d_baseline_mean_abs": _cos_stats(
            np.random.default_rng(SEED).standard_normal((2, M_s1.shape[1])),
            np.random.default_rng(SEED + 1).standard_normal((10, M_s1.shape[1])))["mean_abs_cos"],
        "s1_energy_fraction_inside_span_of_s2_filters": [round(x, 4) for x in in_span],
        "definition_note": "No definition tested yields 0.044. The stored code implements the first entry (0.7358). "
                           "Random 65,536-d vectors give ~0.004, so 0.04 would be near-orthogonal; the real S1 filters "
                           "lie almost entirely inside the span of the S2 filters.",
    }

    return {
        "posemb_sincos_1d_optical_span_rad": round(float(emit_um[-1] - emit_um[0]), 4),
        "posemb_sincos_1d_optical_cycles": round(float((emit_um[-1] - emit_um[0]) / (2.0 * math.pi)), 4),
        "emit_n_bands": int(len(emit_um)),
        "emit_nm_min": round(float(emit_nm[0]), 1),
        "emit_nm_max": round(float(emit_nm[-1]), 1),
        "emit_step_nm": round(float(emit_nm[1] - emit_nm[0]), 2),
        "singular_values_raw_top16": [round(float(x), 3) for x in s_raw[:16]],
        "cum_variance_raw_top16": [round(float(x), 6) for x in cum_raw[:16]],
        "shannon_eff_rank_raw": round(eff_rank_shannon_raw, 3),
        "participation_ratio_raw": round(part_ratio_raw, 3),
        "singular_values_centered_top16": [round(float(x), 3) for x in s_cent[:16]],
        "cum_variance_centered_top16": [round(float(x), 6) for x in cum_cent[:16]],
        "shannon_eff_rank_centered": round(eff_rank_shannon_cent, 3),
        "participation_ratio_centered": round(part_ratio_cent, 3),
        "k99_raw": int(np.searchsorted(cum_raw, 0.99) + 1),
        "k99_centered": int(np.searchsorted(cum_cent, 0.99) + 1),
        "k999_raw": int(np.searchsorted(cum_raw, 0.999) + 1),
        "k999_centered": int(np.searchsorted(cum_cent, 0.999) + 1),
        "impulse_fwhm_nm_centered": fwhm_by_nm,
        "mode_autocorr_fwhm_nm": mode_autocorr_fwhm_nm,
        "emit_nm_grid": [round(float(x), 2) for x in emit_nm],
        "impulse_slices_centered": impulse_slices,
        "top4_centered_modes": [
            [round(float(v), 5) for v in U_c[:, k]] for k in range(4)
        ],
        "gain_vs_feature_width_nm_2300": gain_vs_width_2300,
        "gain_vs_feature_width_nm_850": gain_vs_width_850,
        "ch4_2300nm_comb_vs_swir_broad_ratio": round(ch4_gain / broad_gain, 4),
        "ch4_2300nm_comb_vs_dc_ratio": round(ch4_gain / ref_gain_dc, 4),
        "sensor_ranks": sensor_ranks,
        "s1_vv_vh_interband_cosine": round(float(s1_n[0] @ s1_n[1]), 4),
        "s1_vs_s2_mean_abs_cosine": round(float(np.mean(np.abs(s1_s2_cos))), 4),
        "s1_vs_s2_cosine_definitions": s1_vs_s2_definitions,
    }


# ---------------------------------------------------------------------------
# Main execution across Experiments A, B, C, D + EuroSAT summary
# ---------------------------------------------------------------------------
def main(out_name: str = "v15_clean_metrics.json"):
    t_start = time.perf_counter()
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("[1/6] Loading and verifying canonical Clay v1.5 checkpoint...")
    enc_cpu, enc_dev, device, metadata, art_info = load_verified_clay()
    print(f"      Loaded {art_info['encoder_parameters']:,} params on {device} in {art_info['load_seconds']}s")

    # Load 320 public Sentinel-2 L2A 128x128 chips + metadata + NLCD
    chips_df = pd.read_parquet(S2_ROOT / "data/chips.parquet")
    nlcd_df = pd.read_parquet(S2_ROOT / "results/labels_nlcd.parquet")
    px_map = {p.stem: p for p in sorted((S2_ROOT / "data/pixels").glob("*.npy"))}
    df = chips_df[chips_df["chip_id"].isin(px_map)].copy()
    df = df.merge(nlcd_df, on="chip_id", how="left").sort_values("chip_id").reset_index(drop=True)
    N_ALL = len(df)
    assert N_ALL == 320, f"Expected 320 chips, got {N_ALL}"

    raw_12_u16 = np.stack([np.load(px_map[cid]) for cid in df["chip_id"]], axis=0)  # (320, 12, 128, 128)
    raw_10_u16 = raw_12_u16[:, CLAY_S2_BANDS_12_IDX, :, :]  # (320, 10, 128, 128)
    # Harmonize Planetary Computer PB 04.00+ BOA_ADD_OFFSET (-1000 DN) to physical S2 reflectance*10000
    harm_10_f32 = np.clip(raw_10_u16.astype(np.float32) - 1000.0, 0.0, None)

    s2_meta = metadata["sentinel-2-l2a"]
    s2_means = np.array([s2_meta["bands"]["mean"][b] for b in CLAY_S2_BAND_NAMES], dtype=np.float32)
    s2_stds = np.array([s2_meta["bands"]["std"][b] for b in CLAY_S2_BAND_NAMES], dtype=np.float32)
    s2_waves = [float(s2_meta["bands"]["wavelength"][b]) for b in CLAY_S2_BAND_NAMES]
    s2_gsd = float(s2_meta["gsd"])  # 10.0

    norm_10 = (harm_10_f32 - s2_means[None, :, None, None]) / s2_stds[None, :, None, None]
    times_all = np.stack([normalize_timestamp(t) for t in df["datetime"]], axis=0)
    latlons_all = np.stack([normalize_latlon(la, lo) for la, lo in zip(df["lat"], df["lon"])], axis=0)

    # Spatial clusters (>30 km strict inter-fold separation) on all 320 chips & on 223 NLCD chips
    clusters_all, folds_all, min_km_all = assign_spatial_clusters(
        df["lat"].to_numpy(), df["lon"].to_numpy(), min_gap_km=30.0
    )
    nlcd_mask = df["y_impervious"].notna().to_numpy()
    df_nlcd = df[nlcd_mask].reset_index(drop=True)
    clusters_nlcd, folds_nlcd, min_km_nlcd = assign_spatial_clusters(
        df_nlcd["lat"].to_numpy(), df_nlcd["lon"].to_numpy(), min_gap_km=30.0
    )
    print(
        f"      Spatial CV verified: N_all={N_ALL} ({len(np.unique(clusters_all))} clusters, "
        f"min cross-fold={min_km_all:.1f} km); N_nlcd={len(df_nlcd)} "
        f"({len(np.unique(clusters_nlcd))} clusters, min cross-fold={min_km_nlcd:.1f} km)"
    )

    # CPU vs MPS float32 parity check on first 8 chips
    cls_cpu8, _, pat_cpu8 = forward_tokens(
        enc_cpu, norm_10[:8], times_all[:8], latlons_all[:8], s2_waves, s2_gsd, "cpu", batch_size=8, return_patches=True
    )
    cls_dev8, _, pat_dev8 = forward_tokens(
        enc_dev, norm_10[:8], times_all[:8], latlons_all[:8], s2_waves, s2_gsd, device, batch_size=8, return_patches=True
    )
    art_info["cpu_vs_device_parity"] = {
        "n_chips_tested": 8,
        "cls_max_abs_diff": float(np.max(np.abs(cls_cpu8 - cls_dev8))),
        "cls_min_cosine": float(np.min(cosine_rows(cls_cpu8, cls_dev8))),
        "patch_max_abs_diff": float(np.max(np.abs(pat_cpu8 - pat_dev8))),
        "patch_min_cosine": float(
            np.min(cosine_rows(pat_cpu8.reshape(-1, 1024), pat_dev8.reshape(-1, 1024)))
        ),
    }
    print(f"      CPU vs {device} parity: min CLS cos={art_info['cpu_vs_device_parity']['cls_min_cosine']:.8f}")

    # Extract reference CLS, mean(Z_patch), and full Z_patch (320, 256, 1024)
    print("[2/6] Running Exp A (Hypernetwork) & Reference Sentinel-2 inference...")
    exp_a = run_exp_a_wavelength_hypernetwork(enc_cpu, metadata)

    t_inf = time.perf_counter()
    cls_ref, mean_ref, patch_ref = forward_tokens(
        enc_dev, norm_10, times_all, latlons_all, s2_waves, s2_gsd, device, batch_size=32, return_patches=True
    )
    inf_sec = time.perf_counter() - t_inf
    art_info["s2_320_inference_seconds"] = round(inf_sec, 3)
    art_info["s2_320_chips_per_sec"] = round(N_ALL / inf_sec, 2)
    print(f"      Embedded 320 chips (81,920 spatial tokens) in {inf_sec:.2f}s ({N_ALL/inf_sec:.1f} chips/s)")

    # -----------------------------------------------------------------------
    # Experiment B: Two-Stage SAM+MRL -> DINOv2-reg4 Fossil Audit
    # -----------------------------------------------------------------------
    print("[3/6] Running Exp B (Stage-1 MRL Fossil & Coordinate Exchangeability)...")
    cls_var = np.var(cls_ref, axis=0)
    mean_var = np.var(mean_ref, axis=0)
    dim_blocks = [(0, 16), (16, 32), (32, 64), (64, 128), (128, 256), (256, 512), (512, 1024)]
    var_by_block = {}
    for lo, hi in dim_blocks:
        var_by_block[f"{lo}:{hi}"] = {
            "cls_mean_var": round(float(cls_var[lo:hi].mean()), 6),
            "patch_mean_var": round(float(mean_var[lo:hi].mean()), 6),
            "cls_share_pct": round(float(100.0 * cls_var[lo:hi].sum() / cls_var.sum()), 2),
        }

    X_cls_nlcd = cls_ref[nlcd_mask]
    X_mean_nlcd = mean_ref[nlcd_mask]
    y_imp = df_nlcd["y_impervious"].to_numpy(dtype=np.float64)
    y_can = df_nlcd["y_canopy"].to_numpy(dtype=np.float64)

    # Spectral20 baseline (10-band physical reflectance mean + std per chip)
    spec_mean = harm_10_f32[nlcd_mask].mean(axis=(2, 3))
    spec_std = harm_10_f32[nlcd_mask].std(axis=(2, 3))
    X_spec20 = np.hstack([spec_mean, spec_std])

    spec20_imp = ridge_oof_r2(X_spec20, y_imp, folds_nlcd, clusters_nlcd)
    spec20_can = ridge_oof_r2(X_spec20, y_can, folds_nlcd, clusters_nlcd)
    cls_full_imp = ridge_oof_r2(X_cls_nlcd, y_imp, folds_nlcd, clusters_nlcd)
    cls_full_can = ridge_oof_r2(X_cls_nlcd, y_can, folds_nlcd, clusters_nlcd)
    mean_full_imp = ridge_oof_r2(X_mean_nlcd, y_imp, folds_nlcd, clusters_nlcd)
    mean_full_can = ridge_oof_r2(X_mean_nlcd, y_can, folds_nlcd, clusters_nlcd)

    dims_grid = [16, 32, 64, 128, 256, 512, 1024]
    exchangeability = {}
    rng_b = np.random.default_rng(SEED)
    for d in dims_grid:
        # Ordered prefix [0:d]
        p_imp = ridge_oof_r2(X_cls_nlcd[:, :d], y_imp, folds_nlcd, clusters_nlcd, n_boot=500)
        p_can = ridge_oof_r2(X_cls_nlcd[:, :d], y_can, folds_nlcd, clusters_nlcd, n_boot=500)
        pm_imp = ridge_oof_r2(X_mean_nlcd[:, :d], y_imp, folds_nlcd, clusters_nlcd, n_boot=0)
        pm_can = ridge_oof_r2(X_mean_nlcd[:, :d], y_can, folds_nlcd, clusters_nlcd, n_boot=0)

        # Ordered tail [1024-d:1024]
        t_imp = ridge_oof_r2(X_cls_nlcd[:, 1024 - d :], y_imp, folds_nlcd, clusters_nlcd, n_boot=500)
        t_can = ridge_oof_r2(X_cls_nlcd[:, 1024 - d :], y_can, folds_nlcd, clusters_nlcd, n_boot=500)

        # In-fold PCA-d on CLS (upper bound for linear subspace compression)
        n_comp_max = min(d, int(np.min([np.sum(folds_nlcd != f) for f in range(5)])) - 1)
        oof_pca_imp = np.empty_like(y_imp)
        oof_pca_can = np.empty_like(y_can)
        for f in range(5):
            tr, te = folds_nlcd != f, folds_nlcd == f
            for y_vec, oof_buf in [(y_imp, oof_pca_imp), (y_can, oof_pca_can)]:
                pipe = Pipeline([
                    ("sc1", StandardScaler()),
                    ("pca", PCA(n_components=min(d, n_comp_max), random_state=SEED)),
                    ("sc2", StandardScaler()),
                    ("ridge", RidgeCV(alphas=np.logspace(-2, 6, 33))),
                ])
                pipe.fit(X_cls_nlcd[tr], y_vec[tr])
                oof_buf[te] = pipe.predict(X_cls_nlcd[te])
        pca_imp_r2 = float(r2_score(y_imp, oof_pca_imp))
        pca_can_r2 = float(r2_score(y_can, oof_pca_can))

        # 20 random d-subsets (coordinate exchangeability baseline)
        if d < 1024:
            rand_imp, rand_can, rand_mean_avg, rand_cka = [], [], [], []
            for _ in range(20):
                cols = np.sort(rng_b.choice(1024, size=d, replace=False))
                ri = ridge_oof_r2(X_cls_nlcd[:, cols], y_imp, folds_nlcd, clusters_nlcd, n_boot=0)["r2"]
                rc = ridge_oof_r2(X_cls_nlcd[:, cols], y_can, folds_nlcd, clusters_nlcd, n_boot=0)["r2"]
                rmi = ridge_oof_r2(X_mean_nlcd[:, cols], y_imp, folds_nlcd, clusters_nlcd, n_boot=0)["r2"]
                rmc = ridge_oof_r2(X_mean_nlcd[:, cols], y_can, folds_nlcd, clusters_nlcd, n_boot=0)["r2"]
                rand_imp.append(ri)
                rand_can.append(rc)
                rand_mean_avg.append(0.5 * (rmi + rmc))
                rand_cka.append(linear_cka(cls_ref[:, cols], cls_ref))
        else:
            rand_imp = [p_imp["r2"]]
            rand_can = [p_can["r2"]]
            rand_mean_avg = [0.5 * (pm_imp["r2"] + pm_can["r2"])]
            rand_cka = [1.0]

        rand_avg = 0.5 * (np.array(rand_imp) + np.array(rand_can))
        pref_avg = 0.5 * (p_imp["r2"] + p_can["r2"])
        tail_avg = 0.5 * (t_imp["r2"] + t_can["r2"])
        pref_mean_avg = 0.5 * (pm_imp["r2"] + pm_can["r2"])

        exchangeability[str(d)] = {
            "cls_prefix_imp_r2": round(p_imp["r2"], 4),
            "cls_prefix_can_r2": round(p_can["r2"], 4),
            "cls_prefix_mean_r2": round(pref_avg, 4),
            "cls_prefix_imp_ci95": [round(x, 4) for x in p_imp["ci95"]],
            "cls_prefix_can_ci95": [round(x, 4) for x in p_can["ci95"]],
            "cls_tail_imp_r2": round(t_imp["r2"], 4),
            "cls_tail_can_r2": round(t_can["r2"], 4),
            "cls_tail_mean_r2": round(tail_avg, 4),
            "cls_random_imp_mean": round(float(np.mean(rand_imp)), 4),
            "cls_random_imp_sd": round(float(np.std(rand_imp)), 4),
            "cls_random_can_mean": round(float(np.mean(rand_can)), 4),
            "cls_random_can_sd": round(float(np.std(rand_can)), 4),
            "cls_random_mean_r2": round(float(np.mean(rand_avg)), 4),
            "cls_random_mean_sd": round(float(np.std(rand_avg)), 4),
            "cls_prefix_minus_random_r2": round(float(pref_avg - np.mean(rand_avg)), 4),
            "cls_prefix_minus_tail_r2": round(float(pref_avg - tail_avg), 4),
            "cls_infold_pca_mean_r2": round(0.5 * (pca_imp_r2 + pca_can_r2), 4),
            "zpatch_prefix_mean_r2": round(pref_mean_avg, 4),
            "zpatch_random_mean_r2": round(float(np.mean(rand_mean_avg)), 4),
            "zpatch_prefix_minus_random_r2": round(float(pref_mean_avg - np.mean(rand_mean_avg)), 4),
            "cls_prefix_cka_to_full": round(linear_cka(cls_ref[:, :d], cls_ref), 4),
            "cls_random_cka_to_full": round(float(np.mean(rand_cka)), 4),
        }

    # Check Stage-2 proj.weight norm across input dimensions [0..1023] in checkpoint!
    ckpt_sd = torch.load(str(CKPT_PATH), map_location="cpu", weights_only=True, mmap=True)["state_dict"]
    proj_w = ckpt_sd["model.proj.weight"].numpy()  # (1024, 1024): out_features x in_features
    proj_col_norm = np.linalg.norm(proj_w, axis=0)  # sensitivity of Stage-2 head to each CLS dim j
    proj_norm_blocks = {
        f"{lo}:{hi}": round(float(proj_col_norm[lo:hi].mean()), 5) for lo, hi in dim_blocks
    }

    exp_b = {
        "n_chips_nlcd": int(len(df_nlcd)),
        "n_spatial_clusters": int(len(np.unique(clusters_nlcd))),
        "min_interfold_distance_km": round(min_km_nlcd, 2),
        "cls_coord_variance_32_bins": [
            round(float(cls_var[i * 32 : (i + 1) * 32].mean()), 6) for i in range(32)
        ],
        "patch_coord_variance_32_bins": [
            round(float(mean_var[i * 32 : (i + 1) * 32].mean()), 6) for i in range(32)
        ],
        "variance_by_mrl_block": var_by_block,
        "stage2_proj_input_col_norm_by_block": proj_norm_blocks,
        "baselines": {
            "spectral20": {
                "imp_r2": round(spec20_imp["r2"], 4),
                "imp_ci95": [round(x, 4) for x in spec20_imp["ci95"]],
                "can_r2": round(spec20_can["r2"], 4),
                "can_ci95": [round(x, 4) for x in spec20_can["ci95"]],
                "mean_r2": round(0.5 * (spec20_imp["r2"] + spec20_can["r2"]), 4),
            },
            "cls_1024": {
                "imp_r2": round(cls_full_imp["r2"], 4),
                "imp_ci95": [round(x, 4) for x in cls_full_imp["ci95"]],
                "can_r2": round(cls_full_can["r2"], 4),
                "can_ci95": [round(x, 4) for x in cls_full_can["ci95"]],
                "mean_r2": round(0.5 * (cls_full_imp["r2"] + cls_full_can["r2"]), 4),
            },
            "mean_zpatch_1024": {
                "imp_r2": round(mean_full_imp["r2"], 4),
                "imp_ci95": [round(x, 4) for x in mean_full_imp["ci95"]],
                "can_r2": round(mean_full_can["r2"], 4),
                "can_ci95": [round(x, 4) for x in mean_full_can["ci95"]],
                "mean_r2": round(0.5 * (mean_full_imp["r2"] + mean_full_can["r2"]), 4),
            },
        },
        "exchangeability_by_d": exchangeability,
    }

    # -----------------------------------------------------------------------
    # Experiment C: CLS vs Z_patch Spatial Readout, Quadrant Binding, & Scale
    # -----------------------------------------------------------------------
    print("[4/6] Running Exp C (Dense 80 m Z_patch Readout, Quadrant Binding, Bandsets, GSD)...")
    # C1: Dense 80 m patch-grid readout across all 320 chips (81,920 8x8 patches)
    # Reshape harm_10_f32 (320, 10, 128, 128) -> (320, 10, 16, 8, 16, 8)
    patches_px = harm_10_f32.reshape(N_ALL, 10, 16, 8, 16, 8).transpose(0, 2, 4, 1, 3, 5)  # (320,16,16,10,8,8)
    blue_p = patches_px[:, :, :, 0].mean(axis=(-2, -1))
    green_p = patches_px[:, :, :, 1].mean(axis=(-2, -1))
    red_p = patches_px[:, :, :, 2].mean(axis=(-2, -1))
    nir_p = patches_px[:, :, :, 6].mean(axis=(-2, -1))
    swir1_p = patches_px[:, :, :, 8].mean(axis=(-2, -1))
    nir_sd_p = patches_px[:, :, :, 6].std(axis=(-2, -1))

    eps = 1.0
    ndvi_patch = ((nir_p - red_p) / (nir_p + red_p + eps)).reshape(-1)
    ndwi_patch = ((green_p - nir_p) / (green_p + nir_p + eps)).reshape(-1)
    swir_ratio_patch = ((swir1_p - nir_p) / (swir1_p + nir_p + eps)).reshape(-1)
    texture_patch = np.log1p(nir_sd_p).reshape(-1)

    Z_flat = patch_ref.reshape(-1, 1024)  # (81920, 1024)
    CLS_broad = np.repeat(cls_ref, 256, axis=0)  # (81920, 1024)
    rng_c = np.random.default_rng(SEED)
    Z_shuf = np.empty_like(patch_ref)
    for i in range(N_ALL):
        Z_shuf[i] = patch_ref[i, rng_c.permutation(256)]
    Z_shuf_flat = Z_shuf.reshape(-1, 1024)
    folds_patch = np.repeat(folds_all, 256)

    dense_targets = {
        "patch_ndvi": ndvi_patch,
        "patch_ndwi": ndwi_patch,
        "patch_swir_nir_index": swir_ratio_patch,
        "patch_log_nir_texture": texture_patch,
    }
    dense_results = {}
    alphas_fast = np.logspace(0, 5, 16)
    for t_name, y_t in dense_targets.items():
        res_t = {}
        # Compute theoretical between-chip variance ceiling (chip mean predictor!)
        y_grid = y_t.reshape(N_ALL, 256)
        chip_mean_broad = np.repeat(y_grid.mean(axis=1), 256)
        within_chip_var_frac = 1.0 - float(r2_score(y_t, chip_mean_broad))
        res_t["oracle_chip_mean_r2_ceiling"] = round(float(r2_score(y_t, chip_mean_broad)), 4)
        res_t["within_chip_variance_share"] = round(within_chip_var_frac, 4)

        for feat_name, X_f in [
            ("z_patch_aligned", Z_flat),
            ("cls_broadcast", CLS_broad),
            ("z_patch_shuffled_within_chip", Z_shuf_flat),
        ]:
            oof_p = np.empty_like(y_t, dtype=np.float64)
            for f in range(5):
                tr, te = folds_patch != f, folds_patch == f
                pipe = Pipeline([
                    ("scaler", StandardScaler()),
                    ("ridge", RidgeCV(alphas=alphas_fast)),
                ])
                pipe.fit(X_f[tr], y_t[tr])
                oof_p[te] = pipe.predict(X_f[te])
            # Total R^2 AND within-chip anomaly R^2 (y - chip_mean)!
            r2_tot = float(r2_score(y_t, oof_p))
            y_anom = y_t - chip_mean_broad
            oof_anom = oof_p - np.repeat(oof_p.reshape(N_ALL, 256).mean(axis=1), 256)
            r2_within = float(r2_score(y_anom, oof_anom))
            res_t[feat_name] = {
                "total_r2": round(r2_tot, 4),
                "within_chip_anomaly_r2": round(r2_within, 4),
            }
        dense_results[t_name] = res_t

    # C2a: Natural unedited S2 quadrant localization (which quadrant has peak NDVI / NDWI / texture?)
    # Pool 16x16 tokens into 4 quadrants: NW (0:8,0:8), NE (0:8,8:16), SW (8:16,0:8), SE (8:16,8:16)
    Z_grid = patch_ref.reshape(N_ALL, 16, 16, 1024)
    Z_quads = np.stack([
        Z_grid[:, 0:8, 0:8].mean(axis=(1, 2)),
        Z_grid[:, 0:8, 8:16].mean(axis=(1, 2)),
        Z_grid[:, 8:16, 0:8].mean(axis=(1, 2)),
        Z_grid[:, 8:16, 8:16].mean(axis=(1, 2)),
    ], axis=1)  # (320, 4, 1024)

    natural_quad_results = {}
    for q_name, p_arr in [
        ("peak_vegetation_quadrant", ndvi_patch.reshape(N_ALL, 16, 16)),
        ("peak_water_index_quadrant", ndwi_patch.reshape(N_ALL, 16, 16)),
        ("peak_texture_quadrant", texture_patch.reshape(N_ALL, 16, 16)),
    ]:
        q_vals = np.stack([
            p_arr[:, 0:8, 0:8].mean(axis=(1, 2)),
            p_arr[:, 0:8, 8:16].mean(axis=(1, 2)),
            p_arr[:, 8:16, 0:8].mean(axis=(1, 2)),
            p_arr[:, 8:16, 8:16].mean(axis=(1, 2)),
        ], axis=1)  # (320, 4)
        # Filter to scenes with meaningful quadrant contrast (top - second > 0.25 * global SD)
        sorted_q = np.sort(q_vals, axis=1)
        margin = sorted_q[:, -1] - sorted_q[:, -2]
        keep = margin > 0.15 * np.std(q_vals)
        yk = np.argmax(q_vals[keep], axis=1)

        # Evaluate 4-way accuracy & AUROC on CLS vs shared 1,024-param quadrant scorer w^T Z_q
        oof_cls_prob = np.zeros((keep.sum(), 4), dtype=np.float64)
        oof_z_scores = np.zeros((keep.sum(), 4), dtype=np.float64)
        folds_k = folds_all[keep]
        cls_k = cls_ref[keep]
        zq_k = Z_quads[keep]
        qk_vals = q_vals[keep]
        for f in range(5):
            tr, te = folds_k != f, folds_k == f
            clf_cls = Pipeline([
                ("sc", StandardScaler()),
                ("lr", LogisticRegressionCV(Cs=10, cv=3, max_iter=1000, random_state=SEED)),
            ])
            clf_cls.fit(cls_k[tr], yk[tr])
            oof_cls_prob[te] = clf_cls.predict_proba(cls_k[te])

            # Shared 1,024-weight linear head trained on quadrant value regression
            reg_q = Pipeline([
                ("sc", StandardScaler()),
                ("ridge", RidgeCV(alphas=alphas_fast)),
            ])
            reg_q.fit(zq_k[tr].reshape(-1, 1024), qk_vals[tr].reshape(-1))
            oof_z_scores[te] = reg_q.predict(zq_k[te].reshape(-1, 1024)).reshape(-1, 4)

        # Softmax Z scores for AUROC
        z_exp = np.exp(oof_z_scores - oof_z_scores.max(axis=1, keepdims=True))
        oof_z_prob = z_exp / z_exp.sum(axis=1, keepdims=True)
        natural_quad_results[q_name] = {
            "n_scenes": int(keep.sum()),
            "cls_acc": round(float(accuracy_score(yk, np.argmax(oof_cls_prob, axis=1))), 4),
            "cls_auroc_ovr": round(float(roc_auc_score(yk, oof_cls_prob, multi_class="ovr")), 4),
            "zpatch_shared1024_acc": round(float(accuracy_score(yk, np.argmax(oof_z_scores, axis=1))), 4),
            "zpatch_shared1024_auroc_ovr": round(float(roc_auc_score(yk, oof_z_prob, multi_class="ovr")), 4),
        }

    # C2b: Controlled 2x2 EuroSAT Compositional Feature-Location Binding Benchmark
    # Every 128x128 composite contains 1 Built tile (Industrial/Residential) in quadrant q_B,
    # 1 Water tile (SeaLake/River) in q_W != q_B, and 2 Vegetation tiles (Forest/AnnualCrop).
    euro_imgs = json.loads((EUROSAT_DIR / "images.json").read_text())
    pools = {"train": {"built": [], "water": [], "veg": []}, "test": {"built": [], "water": [], "veg": []}}
    euro_zip = zipfile.ZipFile(EUROSAT_ZIP)
    for item in euro_imgs:
        sp = item["split"]
        cl = item["class"]
        with MemoryFile(euro_zip.read(item["name"])) as mf, mf.open() as src:
            arr13 = src.read().astype(np.float32)  # (13, 64, 64), Harmonized EuroSAT L1C/S2 scale
        arr10 = arr13[EUROSAT_CLAY_IDX]  # (10, 64, 64)
        grp = "built" if cl in ("Industrial", "Residential") else ("water" if cl in ("SeaLake", "River") else "veg")
        pools[sp][grp].append(arr10)

    def build_compositional_mosaic_split(split_name: str, n_per_pair: int, seed_offset: int):
        rng_m = np.random.default_rng(SEED + seed_offset)
        p = pools[split_name]
        mosaics = []
        q_built_list = []
        q_water_list = []
        # 12 ordered (q_built, q_water) pairs with q_built != q_water -> perfectly balanced!
        for qb in range(4):
            for qw in range(4):
                if qb == qw:
                    continue
                other_qs = [q for q in range(4) if q not in (qb, qw)]
                for _ in range(n_per_pair):
                    canvas = np.empty((10, 128, 128), dtype=np.float32)
                    tiles = {
                        qb: p["built"][int(rng_m.integers(len(p["built"])))],
                        qw: p["water"][int(rng_m.integers(len(p["water"])))],
                        other_qs[0]: p["veg"][int(rng_m.integers(len(p["veg"])))],
                        other_qs[1]: p["veg"][int(rng_m.integers(len(p["veg"])))],
                    }
                    for q_idx, t_arr in tiles.items():
                        r0 = 0 if q_idx in (0, 1) else 64
                        c0 = 0 if q_idx in (0, 2) else 64
                        canvas[:, r0 : r0 + 64, c0 : c0 + 64] = t_arr
                    mosaics.append(canvas)
                    q_built_list.append(qb)
                    q_water_list.append(qw)
        px = np.stack(mosaics, axis=0)
        px_norm = (px - s2_means[None, :, None, None]) / s2_stds[None, :, None, None]
        zeros4 = np.zeros((len(px), 4), dtype=np.float32)
        cls_m, mean_m, pat_m = forward_tokens(
            enc_dev, px_norm, zeros4, zeros4, s2_waves, s2_gsd, device, batch_size=32, return_patches=True
        )
        pg = pat_m.reshape(len(px), 16, 16, 1024)
        zq = np.stack([
            pg[:, 0:8, 0:8].mean(axis=(1, 2)),
            pg[:, 0:8, 8:16].mean(axis=(1, 2)),
            pg[:, 8:16, 0:8].mean(axis=(1, 2)),
            pg[:, 8:16, 8:16].mean(axis=(1, 2)),
        ], axis=1)
        return cls_m, mean_m, zq, np.array(q_built_list), np.array(q_water_list)

    # 12 ordered (q_B, q_W) pairs * 20 = 240 train mosaics; 12 * 20 = 240 strictly disjoint test mosaics!
    cls_tr, mean_tr, zq_tr, qb_tr, qw_tr = build_compositional_mosaic_split("train", 20, 100)
    cls_te, mean_te, zq_te, qb_te, qw_te = build_compositional_mosaic_split("test", 20, 200)

    compositional_binding = {}
    for target_label, y_tr, y_te in [("built_quadrant", qb_tr, qb_te), ("water_quadrant", qw_tr, qw_te)]:
        res_bind = {}
        for rep_name, X_tr_b, X_te_b in [
            ("cls_1024", cls_tr, cls_te),
            ("mean_zpatch_1024", mean_tr, mean_te),
            ("zpatch_concat_4096", zq_tr.reshape(len(zq_tr), -1), zq_te.reshape(len(zq_te), -1)),
        ]:
            clf = Pipeline([
                ("sc", StandardScaler()),
                ("lr", LogisticRegressionCV(Cs=10, cv=4, max_iter=1000, random_state=SEED)),
            ])
            clf.fit(X_tr_b, y_tr)
            prob = clf.predict_proba(X_te_b)
            pred = np.argmax(prob, axis=1)
            res_bind[rep_name] = {
                "acc_4way": round(float(accuracy_score(y_te, pred)), 4),
                "auroc_ovr": round(float(roc_auc_score(y_te, prob, multi_class="ovr")), 4),
            }
        # Shared 1,024-weight quadrant presence scorer w^T Z_q (only 1,024 params vs 4,096 for CLS!)
        y_bin_tr = np.eye(4, dtype=np.int64)[y_tr].reshape(-1)
        clf_shared = Pipeline([
            ("sc", StandardScaler()),
            ("lr", LogisticRegressionCV(Cs=10, cv=4, max_iter=1000, random_state=SEED)),
        ])
        clf_shared.fit(zq_tr.reshape(-1, 1024), y_bin_tr)
        s_te = clf_shared.decision_function(zq_te.reshape(-1, 1024)).reshape(-1, 4)
        s_exp = np.exp(s_te - s_te.max(axis=1, keepdims=True))
        p_shared = s_exp / s_exp.sum(axis=1, keepdims=True)
        res_bind["zpatch_shared_1024w"] = {
            "acc_4way": round(float(accuracy_score(y_te, np.argmax(s_te, axis=1))), 4),
            "auroc_ovr": round(float(roc_auc_score(y_te, p_shared, multi_class="ovr")), 4),
        }
        compositional_binding[target_label] = res_bind

    # C3a: Cross-Bandset Alignment & Zero-Shot Probe Transfer on 320 real S2 scenes
    band_subsets = {
        "S2_10band_ref": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        "Landsat_6band_equiv": [2, 1, 0, 7, 8, 9],  # R, G, B, NIR08, SWIR16, SWIR22
        "NAIP_4band_RGBN": [2, 1, 0, 6],            # R, G, B, NIR
        "RGB_3band": [2, 1, 0],                     # R, G, B
    }
    bandset_results = {}
    for b_name, idx_list in band_subsets.items():
        if b_name == "S2_10band_ref":
            c_b, m_b = cls_ref, mean_ref
        else:
            w_sub = [s2_waves[j] for j in idx_list]
            c_b, m_b = forward_tokens(
                enc_dev, norm_10[:, idx_list], times_all, latlons_all, w_sub, s2_gsd, device, batch_size=32
            )
        cos_cls = cosine_rows(c_b, cls_ref)
        cos_mean = cosine_rows(m_b, mean_ref)
        ret_cls = top1_retrieval(c_b, cls_ref)
        ret_mean = top1_retrieval(m_b, mean_ref)
        # Zero-shot transfer of unchanged 10-band linear probe!
        zs_imp_cls = eval_unchanged_models(cls_full_imp["models"], c_b[nlcd_mask], y_imp, folds_nlcd)
        zs_can_cls = eval_unchanged_models(cls_full_can["models"], c_b[nlcd_mask], y_can, folds_nlcd)
        zs_imp_mean = eval_unchanged_models(mean_full_imp["models"], m_b[nlcd_mask], y_imp, folds_nlcd)
        zs_can_mean = eval_unchanged_models(mean_full_can["models"], m_b[nlcd_mask], y_can, folds_nlcd)
        # Matched-sensor refitted probe
        refit_imp = ridge_oof_r2(c_b[nlcd_mask], y_imp, folds_nlcd, clusters_nlcd, n_boot=0)["r2"]
        refit_can = ridge_oof_r2(c_b[nlcd_mask], y_can, folds_nlcd, clusters_nlcd, n_boot=0)["r2"]
        bandset_results[b_name] = {
            "n_bands": len(idx_list),
            "cls_cos_to_10b_mean": round(float(cos_cls.mean()), 4),
            "mean_zpatch_cos_to_10b_mean": round(float(cos_mean.mean()), 4),
            "cls_cross_sensor_top1_acc": round(ret_cls, 4),
            "mean_zpatch_cross_sensor_top1_acc": round(ret_mean, 4),
            "cls_zeroshot_10b_probe_imp_r2": round(zs_imp_cls, 4),
            "cls_zeroshot_10b_probe_can_r2": round(zs_can_cls, 4),
            "cls_zeroshot_10b_probe_mean_r2": round(0.5 * (zs_imp_cls + zs_can_cls), 4),
            "mean_zpatch_zeroshot_10b_probe_mean_r2": round(0.5 * (zs_imp_mean + zs_can_mean), 4),
            "cls_matched_probe_imp_r2": round(refit_imp, 4),
            "cls_matched_probe_can_r2": round(refit_can, 4),
            "cls_matched_probe_mean_r2": round(0.5 * (refit_imp + refit_can), 4),
        }

    # C3b: Physical GSD Conditioning Sweep at fixed pixels
    gsd_sweep = {}
    for gsd_val in [0.5, 1.0, 5.0, 10.0, 30.0, 100.0, 500.0]:
        if gsd_val == 10.0:
            c_g, m_g = cls_ref, mean_ref
        else:
            c_g, m_g = forward_tokens(
                enc_dev, norm_10, times_all, latlons_all, s2_waves, gsd_val, device, batch_size=32
            )
        zs_i = eval_unchanged_models(cls_full_imp["models"], c_g[nlcd_mask], y_imp, folds_nlcd)
        zs_c = eval_unchanged_models(cls_full_can["models"], c_g[nlcd_mask], y_can, folds_nlcd)
        gsd_sweep[str(gsd_val)] = {
            "cls_cos_to_10m": round(float(cosine_rows(c_g, cls_ref).mean()), 4),
            "mean_zpatch_cos_to_10m": round(float(cosine_rows(m_g, mean_ref).mean()), 4),
            "cls_top1_to_10m": round(top1_retrieval(c_g, cls_ref), 4),
            "unchanged_10m_probe_mean_r2": round(0.5 * (zs_i + zs_c), 4),
        }

    exp_c = {
        "dense_80m_patch_readout": dense_results,
        "natural_s2_quadrant_localization": natural_quad_results,
        "eurosat_2x2_compositional_binding": compositional_binding,
        "cross_bandset_alignment": bandset_results,
        "gsd_conditioning_sweep": gsd_sweep,
    }

    # -----------------------------------------------------------------------
    # Experiment D: Imperfect Operational Observations & Contract Sensitivity
    # -----------------------------------------------------------------------
    print("[5/6] Running Exp D (Missing-Pixel Cliff, JPEG2000 Wavelet Compression, Contracts)...")

    # Helper to generate realistic deterministic scan-gap / cloud-patch masks at exact fraction
    def make_missing_masks(n_chips: int, target_frac: float, slc_off_mode: bool = False) -> np.ndarray:
        rng_d = np.random.default_rng(SEED + int(round(target_frac * 1000)) + (77 if slc_off_mode else 0))
        yy, xx = np.meshgrid(np.arange(128), np.arange(128), indexing="ij")
        masks = np.zeros((n_chips, 128, 128), dtype=bool)
        if slc_off_mode:
            # SYNTHETIC Landsat-7-style SLC-off gap pattern (sawtooth wedges, ~22% missing) imposed on Sentinel-2 chips; a simulation, not real SLC-off data
            dist_from_nadir = np.abs(xx - 63.5) / 64.0  # 0 at center, ~1 at edge
            period = 14.0
            for i in range(n_chips):
                phase = rng_d.uniform(0.0, period)
                saw = np.abs(((yy + 0.12 * xx + phase) % period) / period - 0.5)
                m = saw < (0.22 * dist_from_nadir)
                masks[i] = m
            return masks
        # Coherent contiguous stripe/cloud-bar mask of exact pixel count
        n_miss = int(round(target_frac * 128 * 128))
        for i in range(n_chips):
            m = np.zeros((128, 128), dtype=bool)
            # Place tilted scan/cloud strips until reaching n_miss
            placed = 0
            while placed < n_miss:
                r0 = int(rng_d.integers(0, 128))
                h_bar = max(1, int(rng_d.integers(1, max(2, int(24 * target_frac) + 3))))
                c0 = int(rng_d.integers(0, 128))
                w_bar = int(rng_d.integers(16, 129))
                r1 = min(128, r0 + h_bar)
                c1 = min(128, c0 + w_bar)
                m[r0:r1, c0:c1] = True
                placed = int(m.sum())
            # Trim excess boundary pixels so fraction is exact
            if placed > n_miss:
                on_idx = np.flatnonzero(m)
                drop = on_idx[: placed - n_miss]
                m.reshape(-1)[drop] = False
            masks[i] = m
        return masks

    gap_specs = [
        ("1pct", 0.01, False),
        ("2pct", 0.02, False),
        ("5pct", 0.05, False),
        ("10pct", 0.10, False),
        ("22pct_slc_off", 0.22, True),
        ("50pct", 0.50, False),
    ]
    missing_pixel_results = {}
    z_dn0_vals = (-s2_means / s2_stds).tolist()
    for g_label, frac, is_slc in gap_specs:
        m_bool = make_missing_masks(N_ALL, frac, slc_off_mode=is_slc)
        actual_frac = float(m_bool.mean())
        m_exp = np.broadcast_to(m_bool[:, None, :, :], norm_10.shape)

        # (1) Standard GeoTIFF nodata DN=0 fill BEFORE normalization -> z_b = -mean_b / std_b
        px_dn0 = harm_10_f32.copy()
        px_dn0[m_exp] = 0.0
        norm_dn0 = (px_dn0 - s2_means[None, :, None, None]) / s2_stds[None, :, None, None]
        cls_dn0, mean_dn0 = forward_tokens(
            enc_dev, norm_dn0, times_all, latlons_all, s2_waves, s2_gsd, device, batch_size=32
        )

        # (2) Normalized zero / band-mean fill (z_b = 0.0, matching ClayMAE channel dropout!)
        norm_meanfill = norm_10.copy()
        norm_meanfill[m_exp] = 0.0
        cls_mf, mean_mf = forward_tokens(
            enc_dev, norm_meanfill, times_all, latlons_all, s2_waves, s2_gsd, device, batch_size=32
        )

        # (3) Scene-specific unmasked band-mean fill (zero global stats required!)
        px_sc = harm_10_f32.copy()
        for i in range(N_ALL):
            valid = ~m_bool[i]
            b_means = px_sc[i, :, valid].mean(axis=0) if np.any(valid) else s2_means
            for b in range(10):
                px_sc[i, b, m_bool[i]] = b_means[b]
        norm_sc = (px_sc - s2_means[None, :, None, None]) / s2_stds[None, :, None, None]
        cls_sc, _ = forward_tokens(
            enc_dev, norm_sc, times_all, latlons_all, s2_waves, s2_gsd, device, batch_size=32
        )

        dn0_imp = eval_unchanged_models(cls_full_imp["models"], cls_dn0[nlcd_mask], y_imp, folds_nlcd)
        dn0_can = eval_unchanged_models(cls_full_can["models"], cls_dn0[nlcd_mask], y_can, folds_nlcd)
        mf_imp = eval_unchanged_models(cls_full_imp["models"], cls_mf[nlcd_mask], y_imp, folds_nlcd)
        mf_can = eval_unchanged_models(cls_full_can["models"], cls_mf[nlcd_mask], y_can, folds_nlcd)
        sc_imp = eval_unchanged_models(cls_full_imp["models"], cls_sc[nlcd_mask], y_imp, folds_nlcd)
        sc_can = eval_unchanged_models(cls_full_can["models"], cls_sc[nlcd_mask], y_can, folds_nlcd)

        missing_pixel_results[g_label] = {
            "missing_fraction": round(actual_frac, 4),
            "dn0_cls_cos": round(float(cosine_rows(cls_dn0, cls_ref).mean()), 4),
            "dn0_zpatch_cos": round(float(cosine_rows(mean_dn0, mean_ref).mean()), 4),
            "dn0_top1_self_retrieval": round(top1_retrieval(cls_dn0, cls_ref), 4),
            "dn0_unchanged_probe_mean_r2": round(0.5 * (dn0_imp + dn0_can), 4),
            "global_meanfill_cls_cos": round(float(cosine_rows(cls_mf, cls_ref).mean()), 4),
            "global_meanfill_zpatch_cos": round(float(cosine_rows(mean_mf, mean_ref).mean()), 4),
            "global_meanfill_top1_self_retrieval": round(top1_retrieval(cls_mf, cls_ref), 4),
            "global_meanfill_unchanged_probe_mean_r2": round(0.5 * (mf_imp + mf_can), 4),
            "scene_meanfill_cls_cos": round(float(cosine_rows(cls_sc, cls_ref).mean()), 4),
            "scene_meanfill_top1_self_retrieval": round(top1_retrieval(cls_sc, cls_ref), 4),
            "scene_meanfill_unchanged_probe_mean_r2": round(0.5 * (sc_imp + sc_can), 4),
        }

    # D2: Lossy JPEG2000 Wavelet Compression via OpenJPEG on all 320 chips
    harm_u16 = np.round(harm_10_f32).astype(np.uint16)
    jp2_results = {}
    for target_label, qual in [("10to1", 10.0), ("20to1", 5.0), ("40to1", 2.5), ("80to1", 1.25)]:
        rec_all = np.empty_like(harm_u16, dtype=np.float32)
        ratios = []
        mse_list = []
        for i in range(N_ALL):
            with MemoryFile() as mem:
                with mem.open(
                    driver="JP2OpenJPEG", width=128, height=128, count=10,
                    dtype="uint16", QUALITY=qual, REVERSIBLE="NO",
                ) as ds:
                    ds.write(harm_u16[i])
                comp_bytes = len(mem.getbuffer())
                ratios.append(harm_u16[i].nbytes / comp_bytes)
                with mem.open() as ds2:
                    rec_i = ds2.read().astype(np.float32)
            rec_all[i] = rec_i
            mse_list.append(float(np.mean((rec_i - harm_10_f32[i]) ** 2)))
        norm_jp2 = (rec_all - s2_means[None, :, None, None]) / s2_stds[None, :, None, None]
        cls_j, mean_j = forward_tokens(
            enc_dev, norm_jp2, times_all, latlons_all, s2_waves, s2_gsd, device, batch_size=32
        )
        j_imp = eval_unchanged_models(cls_full_imp["models"], cls_j[nlcd_mask], y_imp, folds_nlcd)
        j_can = eval_unchanged_models(cls_full_can["models"], cls_j[nlcd_mask], y_can, folds_nlcd)
        mean_mse = float(np.mean(mse_list))
        psnr_db = 10.0 * math.log10((10000.0**2) / max(mean_mse, 1e-6))
        jp2_results[target_label] = {
            "openjpeg_quality": qual,
            "achieved_ratio_mean": round(float(np.mean(ratios)), 2),
            "psnr_db_re_10000": round(psnr_db, 2),
            "cls_cos": round(float(cosine_rows(cls_j, cls_ref).mean()), 4),
            "mean_zpatch_cos": round(float(cosine_rows(mean_j, mean_ref).mean()), 4),
            "cls_top1_self_retrieval": round(top1_retrieval(cls_j, cls_ref), 4),
            "unchanged_probe_imp_r2": round(j_imp, 4),
            "unchanged_probe_can_r2": round(j_can, 4),
            "unchanged_probe_mean_r2": round(0.5 * (j_imp + j_can), 4),
        }

    # D3: Preprocessing Contract Sensitivity Audit
    # (1) Unharmonized Planetary Computer PB 04.00+ (+1000 BOA offset)
    norm_pb04 = (raw_10_u16.astype(np.float32) - s2_means[None, :, None, None]) / s2_stds[None, :, None, None]
    cls_pb04, _ = forward_tokens(
        enc_dev, norm_pb04, times_all, latlons_all, s2_waves, s2_gsd, device, batch_size=32
    )
    # (2) Wavelength unit bug (`waves * 1000` in nm, matching `quickstart.md` error)
    s2_waves_nm = [w * 1000.0 for w in s2_waves]
    cls_nm, _ = forward_tokens(
        enc_dev, norm_10, times_all, latlons_all, s2_waves_nm, s2_gsd, device, batch_size=32
    )
    # (3) Per-split sample normalization (reproducing AEF S19.3.3 protocol: normalize image pixels
    #     by evaluation-split statistics rather than Clay's metadata.yaml physical table!)
    #     Test both full-fold normalization (~178 train chips) and low-shot n=10 split normalization!
    harm_nlcd = harm_10_f32[nlcd_mask]
    times_nlcd = times_all[nlcd_mask]
    latlons_nlcd = latlons_all[nlcd_mask]

    oof_foldnorm_imp = np.empty_like(y_imp)
    oof_foldnorm_can = np.empty_like(y_can)
    foldnorm_cos = []
    for f in range(5):
        tr, te = folds_nlcd != f, folds_nlcd == f
        mu_f = harm_nlcd[tr].mean(axis=(0, 2, 3), keepdims=True)
        sd_f = np.clip(harm_nlcd[tr].std(axis=(0, 2, 3), keepdims=True), 1e-3, None)
        c_tr, _ = forward_tokens(
            enc_dev, (harm_nlcd[tr] - mu_f) / sd_f, times_nlcd[tr], latlons_nlcd[tr], s2_waves, s2_gsd, device
        )
        c_te, _ = forward_tokens(
            enc_dev, (harm_nlcd[te] - mu_f) / sd_f, times_nlcd[te], latlons_nlcd[te], s2_waves, s2_gsd, device
        )
        foldnorm_cos.extend(cosine_rows(c_te, X_cls_nlcd[te]).tolist())
        for y_v, buf in [(y_imp, oof_foldnorm_imp), (y_can, oof_foldnorm_can)]:
            pipe = Pipeline([("sc", StandardScaler()), ("ridge", RidgeCV(alphas=np.logspace(-2, 6, 33)))])
            pipe.fit(c_tr, y_v[tr])
            buf[te] = pipe.predict(c_te)

    # Low-shot n=10 comparison: Reference metadata.yaml vs Per-10-chip-split normalization
    # across 15 deterministic low-shot draws per fold!
    rng_ls = np.random.default_rng(SEED + 999)
    ls_ref_r2s, ls_splitnorm_r2s, ls_splitnorm_cos = [], [], []
    for draw in range(8):
        oof_r_i, oof_r_c = np.empty_like(y_imp), np.empty_like(y_can)
        oof_s_i, oof_s_c = np.empty_like(y_imp), np.empty_like(y_can)
        for f in range(5):
            tr_idx = np.flatnonzero(folds_nlcd != f)
            te_idx = np.flatnonzero(folds_nlcd == f)
            sub10 = rng_ls.choice(tr_idx, size=12, replace=False)
            # Reference metadata.yaml embeddings on exact same 12-chip train subset!
            for y_v, buf in [(y_imp, oof_r_i), (y_can, oof_r_c)]:
                pipe = Pipeline([("sc", StandardScaler()), ("ridge", RidgeCV(alphas=np.logspace(-1, 6, 25)))])
                pipe.fit(X_cls_nlcd[sub10], y_v[sub10])
                buf[te_idx] = pipe.predict(X_cls_nlcd[te_idx])
            # Per-split pixel normalization on the 12-chip subset (AEF S19.3.3 low-shot protocol)
            mu_10 = harm_nlcd[sub10].mean(axis=(0, 2, 3), keepdims=True)
            sd_10 = np.clip(harm_nlcd[sub10].std(axis=(0, 2, 3), keepdims=True), 1e-3, None)
            c_sub10, _ = forward_tokens(
                enc_dev, (harm_nlcd[sub10] - mu_10) / sd_10, times_nlcd[sub10], latlons_nlcd[sub10], s2_waves, s2_gsd, device
            )
            c_te10, _ = forward_tokens(
                enc_dev, (harm_nlcd[te_idx] - mu_10) / sd_10, times_nlcd[te_idx], latlons_nlcd[te_idx], s2_waves, s2_gsd, device
            )
            ls_splitnorm_cos.extend(cosine_rows(c_te10, X_cls_nlcd[te_idx]).tolist())
            for y_v, buf in [(y_imp, oof_s_i), (y_can, oof_s_c)]:
                pipe = Pipeline([("sc", StandardScaler()), ("ridge", RidgeCV(alphas=np.logspace(-1, 6, 25)))])
                pipe.fit(c_sub10, y_v[sub10])
                buf[te_idx] = pipe.predict(c_te10)
        ls_ref_r2s.append(0.5 * (r2_score(y_imp, oof_r_i) + r2_score(y_can, oof_r_c)))
        ls_splitnorm_r2s.append(0.5 * (r2_score(y_imp, oof_s_i) + r2_score(y_can, oof_s_c)))

    contracts = {
        "reference_metadata_boa_harmonized_um": {
            "cls_cos_to_ref": round(float(cosine_rows(cls_ref, cls_ref).mean()), 4),
            "matched_probe_mean_r2": round(0.5 * (cls_full_imp["r2"] + cls_full_can["r2"]), 4),
            "unchanged_ref_probe_mean_r2": round(0.5 * (cls_full_imp["r2"] + cls_full_can["r2"]), 4),
        },
        "unharmonized_pb04_plus1000_offset": {
            "cls_cos_to_ref": round(float(cosine_rows(cls_pb04, cls_ref).mean()), 4),
            "matched_probe_mean_r2": round(
                0.5 * (
                    ridge_oof_r2(cls_pb04[nlcd_mask], y_imp, folds_nlcd, clusters_nlcd, n_boot=0)["r2"]
                    + ridge_oof_r2(cls_pb04[nlcd_mask], y_can, folds_nlcd, clusters_nlcd, n_boot=0)["r2"]
                ),
                4,
            ),
            "unchanged_ref_probe_mean_r2": round(
                0.5 * (
                    eval_unchanged_models(cls_full_imp["models"], cls_pb04[nlcd_mask], y_imp, folds_nlcd)
                    + eval_unchanged_models(cls_full_can["models"], cls_pb04[nlcd_mask], y_can, folds_nlcd)
                ),
                4,
            ),
        },
        "per_fold_pixel_norm_n178": {
            "cls_cos_to_ref": round(float(np.mean(foldnorm_cos)), 4),
            "matched_probe_mean_r2": round(
                0.5 * (float(r2_score(y_imp, oof_foldnorm_imp)) + float(r2_score(y_can, oof_foldnorm_can))),
                4,
            ),
        },
        "lowshot_12chip_comparison": {
            "reference_metadata_mean_r2": round(float(np.mean(ls_ref_r2s)), 4),
            "per_split_pixel_norm_mean_r2": round(float(np.mean(ls_splitnorm_r2s)), 4),
            "delta_r2": round(float(np.mean(ls_splitnorm_r2s) - np.mean(ls_ref_r2s)), 4),
            "per_split_cls_cos_to_ref": round(float(np.mean(ls_splitnorm_cos)), 4),
        },
        "wavelength_nm_x1000_mismatch": {
            "cls_cos_to_ref": round(float(cosine_rows(cls_nm, cls_ref).mean()), 4),
            "matched_probe_mean_r2": round(
                0.5 * (
                    ridge_oof_r2(cls_nm[nlcd_mask], y_imp, folds_nlcd, clusters_nlcd, n_boot=0)["r2"]
                    + ridge_oof_r2(cls_nm[nlcd_mask], y_can, folds_nlcd, clusters_nlcd, n_boot=0)["r2"]
                ),
                4,
            ),
            "unchanged_ref_probe_mean_r2": round(
                0.5 * (
                    eval_unchanged_models(cls_full_imp["models"], cls_nm[nlcd_mask], y_imp, folds_nlcd)
                    + eval_unchanged_models(cls_full_can["models"], cls_nm[nlcd_mask], y_can, folds_nlcd)
                ),
                4,
            ),
        },
    }

    exp_d = {
        "dn0_normalized_z_per_band": [round(float(x), 3) for x in z_dn0_vals],
        "missing_pixel_cliff_and_rescue": missing_pixel_results,
        "jpeg2000_wavelet_compression": jp2_results,
        "preprocessing_contracts": contracts,
    }

    # -----------------------------------------------------------------------
    # Step 6: Load verified EuroSAT AWS reproducibility summary (Appendix A)
    # -----------------------------------------------------------------------
    print("[6/6] Packaging verified EuroSAT reproducibility summary & writing JSON...")
    scores_csv = EUROSAT_AWS_RESULTS / "scores.csv"
    audit_json = EUROSAT_AWS_RESULTS / "review-audit.json"
    if scores_csv.exists():
        euro_scores = pd.read_csv(scores_csv)
        audit = json.loads(audit_json.read_text()) if audit_json.exists() else None
        eurosat_summary = {
            "source": "carried over, not regenerated by this script (see results/eurosat-v1.1/README.md)",
            # None (not True) when no audit file is shipped: the earlier pass/fail flag is not reproducible from this repo.
            "audit_passed": (bool(audit["passed"]) if audit is not None and "passed" in audit else None),
            "scores_rows": euro_scores.to_dict(orient="records"),
        }
    else:
        eurosat_summary = {"source": "missing", "audit_passed": None, "scores_rows": []}

    sys.path.insert(0, str(MODEL_REPO))
    try:
        from provenance import provenance  # noqa: PLC0415  (experiments/suite/common/provenance.py)
        prov = provenance(__file__, {"latlon_order": LATLON_ORDER, "out_name": out_name, "seed": SEED,
                                     "chip_root": str(S2_ROOT.name)}, root=WORK_ROOT)
    except Exception as e:  # noqa: BLE001
        prov = {"error": f"provenance unavailable: {e}"}
    payload = {
        "provenance": prov,
        "latlon_order": LATLON_ORDER,
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "total_runtime_seconds": round(time.perf_counter() - t_start, 2),
        "artifact_verification": art_info,
        "expA_wavelength_hypernetwork": exp_a,
        "expB_mrl_fossil_audit": exp_b,
        "expC_spatial_readout_and_conditioning": exp_c,
        "expD_operational_and_contracts": exp_d,
        "appendixA_eurosat_repro": eurosat_summary,
    }

    out_json = OUT_DIR / out_name
    out_json.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    print(f"DONE in {payload['total_runtime_seconds']}s -> {out_json}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Clay v1.5 primary evaluation suite (Experiments A-D).")
    ap.add_argument("--latlon-order", choices=["documented", "training"], default="documented",
                    help="location-slot order fed to Clay (default: documented, the setting of the paper's perturbation runs)")
    ap.add_argument("--out-name", default=None, help="results JSON file name under experiments/results/")
    a = ap.parse_args()
    LATLON_ORDER = a.latlon_order
    main(a.out_name or ("v15_clean_metrics.json" if a.latlon_order == "documented" else "v15_clean_metrics_latlon_training.json"))
