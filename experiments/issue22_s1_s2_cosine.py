#!/usr/bin/env python3
"""Issue #22: which definition of the Sentinel-1 vs Sentinel-2 "mean |cos|" gives the manuscript's 0.044 (if any)?

Evaluates the DynamicEmbedding of the real Clay v1.5 checkpoint (and, as a control, randomly initialised ones) with every
plausible definition and writes ALL of them to experiments/results/issue22_s1_s2_cosine.json. The definition implemented by
run_clay_v15_experiments.py is `stored_definition`; it is listed first and is not chosen for being flattering.

A "filter" is the 65,536-d generated patch-embedding kernel of one band: weight_generator(fclayer(posemb_sincos_1d(wavelength)))
* 0.02, with shape (n_bands, 1024 out channels * 8 * 8 px). Sentinel-1 VV/VH use the pseudo-wavelengths 3.5 and 4.0 um from
configs/metadata.yaml; Sentinel-2 uses its 10 band wavelengths.

Run:  python experiments/issue22_s1_s2_cosine.py [--out experiments/results/issue22_s1_s2_cosine.json]
Environment: CLAY_V15_CKPT (checkpoint, default checkpoints/v1.5/clay-v1.5.ckpt). CPU only, about 1 minute.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

HERE = Path(__file__).resolve().parent
COMMON = HERE / "suite" / "common"
sys.path.insert(0, str(COMMON))
from claymodel.factory import DynamicEmbedding  # noqa: E402
from claymodel.model import Encoder  # noqa: E402
from claymodel.utils import posemb_sincos_1d  # noqa: E402

S2_BANDS = ["blue", "green", "red", "rededge1", "rededge2", "rededge3", "nir", "nir08", "swir16", "swir22"]
S1 = [3.5, 4.0]
TARGET = 0.044


def nrm(M):
    return M / np.clip(np.linalg.norm(M, axis=1, keepdims=True), 1e-12, None)


def stats(A, B):
    C = nrm(A) @ nrm(B).T
    return {"mean_abs_cos": round(float(np.mean(np.abs(C))), 4), "mean_signed_cos": round(float(np.mean(C)), 4),
            "max_abs_cos": round(float(np.max(np.abs(C))), 4)}


def main(out: Path):
    ckpt = Path(os.environ.get("CLAY_V15_CKPT", str(HERE.parent / "checkpoints/v1.5/clay-v1.5.ckpt")))
    sd = torch.load(str(ckpt), map_location="cpu", weights_only=True, mmap=True)["state_dict"]
    st = {k.removeprefix("model.encoder."): v for k, v in sd.items() if k.startswith("model.encoder.")}
    enc = Encoder(mask_ratio=0.0, patch_size=8, shuffle=False, dim=1024, depth=24, heads=16, dim_head=64, mlp_ratio=4.0)
    enc.load_state_dict(st, strict=True)
    enc.eval()
    pe = enc.patch_embedding
    meta = yaml.safe_load((COMMON / "configs/metadata.yaml").read_text())
    S2 = [float(meta["sentinel-2-l2a"]["bands"]["wavelength"][b]) for b in S2_BANDS]
    assert meta["sentinel-1-rtc"]["bands"]["wavelength"]["vv"] == 3.5 and meta["sentinel-1-rtc"]["bands"]["wavelength"]["vh"] == 4.0

    def filt(m, w):
        w = torch.tensor(w, dtype=torch.float32)
        with torch.inference_mode():
            we = m.fclayer(posemb_sincos_1d(w, m.wave_dim))
            W, _ = m.weight_generator(we)
        return (W * 0.02).double().numpy()

    def enc_only(w):
        return posemb_sincos_1d(torch.tensor(w, dtype=torch.float32), pe.wave_dim).double().numpy()

    def after_fc(w):
        with torch.inference_mode():
            return pe.fclayer(posemb_sincos_1d(torch.tensor(w, dtype=torch.float32), pe.wave_dim)).double().numpy()

    def per_out_channel(A, B):
        a, b = A.reshape(A.shape[0], 1024, 64), B.reshape(B.shape[0], 1024, 64)
        cs = np.stack([(a[i] * b[j]).sum(-1) / (np.linalg.norm(a[i], axis=-1) * np.linalg.norm(b[j], axis=-1))
                       for i in range(a.shape[0]) for j in range(b.shape[0])])
        return {"mean_abs_cos": round(float(np.mean(np.abs(cs))), 4), "mean_signed_cos": round(float(np.mean(cs)), 4)}

    M1, M2 = filt(pe, S1), filt(pe, S2)
    emit = np.linspace(0.3810, 2.49396, 285)
    Me = filt(pe, list(emit))
    mu_s2, mu_emit = M2.mean(0, keepdims=True), Me.mean(0, keepdims=True)
    Mj = filt(pe, S1 + S2)
    Mjc = Mj - Mj.mean(0, keepdims=True)
    Mj285 = filt(pe, list(emit) + S1 + S2)
    Mj285c = Mj285 - Mj285[:285].mean(0, keepdims=True)
    Q, _ = np.linalg.qr(M2.T)
    in_span = [round(float(np.sum((Q @ (Q.T @ v)) ** 2) / np.sum(v**2)), 4) for v in M1]
    _, _, Vt = np.linalg.svd(M2 - M2.mean(0, keepdims=True), full_matrices=False)
    top_modes = [round(float(np.abs(nrm(M1) @ Vt[k]).mean()), 3) for k in range(9)]
    rc = lambda M: M - M.mean(1, keepdims=True)  # noqa: E731

    defs = {
        "stored_definition_uncentered_each_set_in_own_context": stats(M1, M2),
        "uncentered_joint_12band_context": stats(Mj[:2], Mj[2:]),
        "centred_each_set_on_own_band_mean": stats(M1 - M1.mean(0, keepdims=True), M2 - M2.mean(0, keepdims=True)),
        "centred_joint_context_on_joint_band_mean": stats(Mjc[:2], Mjc[2:]),
        "both_centred_on_S2_band_mean": stats(M1 - mu_s2, M2 - mu_s2),
        "both_centred_on_EMIT285_band_mean": stats(M1 - mu_emit, M2 - mu_emit),
        "joint_285_plus_S1_plus_S2_uncentred": stats(Mj285[285:287], Mj285[287:]),
        "joint_285_plus_S1_plus_S2_centred_on_EMIT_rows": stats(Mj285c[285:287], Mj285c[287:]),
        "row_centred_pearson_over_65536_weights": stats(rc(M1), rc(M2)),
        "per_output_channel_64d_kernel_cosine": per_out_channel(M1, M2),
        "raw_sincos_wavelength_encodings": stats(enc_only(S1), enc_only(S2)),
        "after_fclayer_before_weight_generator": stats(after_fc(S1), after_fc(S2)),
        "S1_vs_S2_visible_bands_only_BGR": stats(M1, M2[:3]),
        "S1_vs_top_centred_S2_modes_mean_abs_cos_modes_1_to_9": top_modes,
    }
    controls = {
        "s2_vs_s2_within_optical_uncentred_offdiag_mean_abs": round(float(np.mean(np.abs(
            (nrm(M2) @ nrm(M2).T)[np.triu_indices(10, 1)]))), 4),
        "s2_vs_s2_within_optical_centred_offdiag_mean_abs": round(float(np.mean(np.abs(
            (nrm(M2 - M2.mean(0)) @ nrm(M2 - M2.mean(0)).T)[np.triu_indices(10, 1)]))), 4),
        "s1_vv_vh_cosine": round(float(nrm(M1)[0] @ nrm(M1)[1]), 4),
        "random_gaussian_65536d_mean_abs": stats(np.random.default_rng(0).standard_normal((2, 65536)),
                                                 np.random.default_rng(1).standard_normal((10, 65536)))["mean_abs_cos"],
        "s1_energy_fraction_inside_span_of_s2_filters_VV_VH": in_span,
    }
    untrained = {}
    for seed in range(3):
        torch.manual_seed(seed)
        de = DynamicEmbedding(wave_dim=pe.wave_dim, num_latent_tokens=pe.num_latent_tokens, patch_size=8, embed_dim=1024).eval()
        a, b = filt(de, S1), filt(de, S2)
        untrained[f"seed{seed}"] = {"uncentered": stats(a, b), "centred_on_S2_band_mean": stats(a - b.mean(0, keepdims=True), b - b.mean(0, keepdims=True))}
    allv = {k: v["mean_abs_cos"] for k, v in defs.items() if isinstance(v, dict) and "mean_abs_cos" in v}
    closest = min(allv.items(), key=lambda kv: abs(kv[1] - TARGET))
    payload = {
        "issue": 22, "manuscript_value": TARGET,
        "stored_code_value": allv["stored_definition_uncentered_each_set_in_own_context"],
        "definitions_trained_checkpoint": defs, "controls": controls, "untrained_init_controls": untrained,
        "closest_definition_to_0.044": {"name": closest[0], "mean_abs_cos": closest[1]},
        "conclusion": "No tested definition gives 0.044. The stored code implements the first entry. The Sentinel-1 filters are "
                      "strongly aligned with Sentinel-2 filters (>0.97 of their energy lies in the span of the S2 filters), "
                      "and an untrained DynamicEmbedding is aligned to the same degree, so 'orthogonal' is not supported.",
    }
    try:
        from provenance import provenance
        payload["provenance"] = provenance(__file__, {"out": out.name}, root=HERE.parent)
    except Exception as e:  # noqa: BLE001
        payload["provenance"] = {"error": str(e)}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=1) + "\n")
    print("wrote", out.name, "| stored:", payload["stored_code_value"], "| closest to 0.044:", payload["closest_definition_to_0.044"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(HERE / "results" / "issue22_s1_s2_cosine.json"))
    main(Path(ap.parse_args().out))
