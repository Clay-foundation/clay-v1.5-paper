#!/usr/bin/env python3
"""DOFA-L on the MADOS cells of Table 2 (frozen linear probe and fine-tune), same tiles/labels/probe as Clay.

Reuses experiments/run_clay_missing_cells.py by import, so only the encoder changes:
  * data      : run_clay_missing_cells.load_mados() -- the public Zenodo MADOS release cut into 80 px tiles
                (2657 / 1186 / 1311 tiles train/val/test, 15 labelled classes, -1 = unlabelled, ignored),
                10 Sentinel-2 bands, train-split per-band z-score.
  * input     : DOFA's own contract as in run_standard_benchmarks.py: tile bilinearly resized to 224 px,
                Sentinel-2 wavelengths in micrometres (rsb.S2_10B_WAVES), 14 x 14 patch tokens.
  * LP (cell lp_mados) : rsb.eval_dense_segmentation (per-token 1x1 linear head on the final-normed patch tokens,
                val-selected C, bilinear upsampling to the 80 px mask, mIoU over classes with union>0) -- identical to Clay's.
  * FT (cell ft_mados) : run_clay_missing_cells.ft_seg (E7 recipe: AdamW wd 0.05, head lr 1e-4, encoder lr 1e-5,
                5-epoch warm-up + cosine, batch 8, flips + rot90, bf16, gradient checkpointing, E7 Neck + UPerHead on taps
                6/12/18/24, best-validation epoch evaluated every 5 epochs, test scored once) with E7's DofaBackbone.

Usage:
  CLAY_SCRATCH=<scratch> CLAY_V15_CKPT=<ckpt> CLAY_MADOS_ZIP=<MADOS.zip> \\
    python experiments/run_dofa_missing_cells.py --cells lp_mados,ft_mados --seeds 0,1,2 --epochs 20 --out <json>
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import run_clay_missing_cells as rcm
from run_clay_missing_cells import rsb

SCRIPT = Path(__file__).resolve()
DEFAULT_OUT = rcm.HERE / "experiments/results/dofa_missing_cells.json"


def prov(extra=None):
    return rsb.provenance(str(SCRIPT), {"rsb_script_sha256": rsb.sha256_file(rsb.__file__),
                                        "rcm_script_sha256": rsb.sha256_file(rcm.__file__), **(extra or {})})


class DofaOnly(rsb.ModelSuite):
    """rsb.ModelSuite.extract_dofa with only DOFA-L (torchgeo DOFA_MAE) loaded."""

    def __init__(self):
        from torchgeo.models import DOFALarge16_Weights, dofa_large_patch16_224
        self.dofa = dofa_large_patch16_224(weights=DOFALarge16_Weights.DOFA_MAE)
        self.dofa.eval().requires_grad_(False)


def dofa_feats(suite, X, mu, sd):
    for attempt in range(30):  # shared GPU: wait and retry on CUDA OOM
        try:
            return suite.extract_dofa(X, mu, sd, rsb.S2_10B_WAVES, True)["patches"]
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache(); print(f"[oom] shared GPU full, retry {attempt+1}", flush=True); time.sleep(60)
    raise RuntimeError("GPU stayed full")


def lp_mados(out):
    (Xtr, ytr), (Xva, yva), (Xte, yte), mu, sd = rcm.load_mados()
    suite = DofaOnly()
    f = {k: dofa_feats(suite, X, mu, sd) for k, X in (("tr", Xtr), ("va", Xva), ("te", Xte))}
    with rcm.IgnoreCE():
        r = rsb.eval_dense_segmentation(f["tr"], ytr, f["va"], yva, f["te"], yte, 15)
    rec = {"table2_row": "MADOS", "model": "dofa_l", "cell": "LP", "status": "rerun_here",
           "metric": "test mIoU (%) over the 15 labelled classes with union>0 (unlabelled=ignored)", "value": r["miou"],
           "readout": "final-normed patch tokens (14x14 on the 224 px resized tile), 1x1 linear head, bilinear upsampling to 80 px",
           "head": "multinomial linear, val-selected C (same code as the Clay cell)", "seeds": "deterministic (fixed subsample seed 42)",
           "split_sizes_tiles": [len(ytr), len(yva), len(yte)], "details": r,
           "input": {"bands": "10 S2 bands as Clay", "wavelengths_um": rsb.S2_10B_WAVES, "resize_px": 224, "normalisation": "train-split per-band z-score"},
           "caveat": "same tiles/probe as the repo's Clay MADOS LP (own tile preparation of the Zenodo release); not a replication of third-party numbers",
           "data": rcm.MADOS_URL, "provenance": prov()}
    rcm.merge_write(out, "mados/LP", rec)


class DofaFT(nn.Module):
    """Drop-in for rcm.FTModel(seg): E7 DofaBackbone + Neck(stride 16) + UPerHead; tile resized to 224 px, logits back to tile size."""

    def __init__(self, E7, kind, n_out, waves, gsd):
        super().__init__()
        self.bb = E7.DofaBackbone(size=224)          # parameter names start with "bb." -> encoder lr group in ft_seg
        self.bb.waves = list(waves)                  # micrometres (DOFA convention)
        self.neck, self.head = E7.Neck(1024, 16), E7.UPerHead(ncls=n_out)

    def forward(self, x):
        H = x.shape[-1]
        x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            o = self.head(self.neck(self.bb(x)))
        return F.interpolate(o.float(), (H, H), mode="bilinear", align_corners=False)


def ft_mados(seeds, epochs, out):
    E7 = rcm.load_e7()
    rcm.FTModel = DofaFT                              # ft_seg builds its model through this module-level name
    (Xtr, ytr), (Xva, yva), (Xte, yte), mu, sd = rcm.load_mados()
    nz = lambda X: torch.from_numpy(((X - mu[None, :, None, None]) / sd[None, :, None, None]).astype(np.float16))
    T = lambda y: torch.from_numpy(y.astype(np.int16))
    data = dict(X_tr=nz(Xtr), y_tr=T(ytr), X_va=nz(Xva), y_va=T(yva), waves=rsb.S2_10B_WAVES, gsd=10.0, n_cls=15, crop=80,
                tests={"test": (nz(Xte), T(yte))})
    runs = []
    for s in seeds:
        runs.append(rcm.ft_seg(E7, "mados-dofa", s, epochs, 8, data))
        allr = rcm.union_seeds(out, "mados/FT", runs)
        vals = [r["test_miou"] for r in allr]
        rec = {"table2_row": "MADOS", "model": "dofa_l", "cell": "FT", "status": "rerun_here",
               "metric": "test mIoU (%) over the 15 labelled classes with union>0 (unlabelled=ignored)",
               "value": round(float(np.mean(vals)), 2), "value_std_over_seeds": round(float(np.std(vals)), 2) if len(vals) > 1 else None,
               "head": "E7 UPerNet neck+head on DOFA taps (6,12,18,24), full encoder trainable (DOFA wavelength generator included)",
               "recipe": {"optimizer": "AdamW wd 0.05", "lr_head": 1e-4, "lr_encoder": 1e-5, "schedule": "5-epoch warm-up + cosine",
                          "epochs": epochs, "batch_size": 8, "input": "80 px tile bilinearly resized to 224 px, S2 wavelengths in um",
                          "precision": "bf16, gradient checkpointing", "selection": "best-validation mIoU epoch (every 5 epochs); test scored once",
                          "source": "experiments/suite/E7_finetune/code/train_e7.py (DofaBackbone, Neck, UPerHead) via run_clay_missing_cells.ft_seg"},
               "seeds": [r["seed"] for r in allr], "per_seed": allr,
               "best_epoch_before_last": [r["best_epoch"] < epochs for r in allr],
               "split_sizes_tiles": [len(ytr), len(yva), len(yte)], "n_classes": 15, "data": rcm.MADOS_URL, "provenance": prov()}
        rcm.merge_write(out, "mados/FT", rec)


class DofaDetBackbone(nn.Module):
    """DOFA-L as the detection encoder: 128 px patch bilinearly resized to 256 px -> 16x16 token map (same grid as Clay's 8 px patches);
    wavelengths in micrometres, SAR = DOFA_S1_2B_WAVES (3.75/3.75) as run_standard_benchmarks.  Last-layer (24) normed tokens."""

    def __init__(self, E7, waves):
        super().__init__()
        self.inner = E7.DofaBackbone(size=256)
        self.inner.waves = list(waves)

    def forward(self, x, tt=None, ll=None, taps=(24,)):
        x = F.interpolate(x, size=(256, 256), mode="bilinear", align_corners=False)
        return [self.inner(x)[-1]]


class DinoDetBackbone(nn.Module):
    """DINOv2-L/14 reg4 (timm, as run_all_doable_table2.get_dinov2) with 3 bands (R,G,B = first three Landsat bands B4,B3,B2, train-split
    z-score per band); 128 px resized to 224 px -> 16x16 token map; last-layer normed patch tokens (register tokens dropped)."""

    def __init__(self):
        super().__init__()
        import timm
        self.m = timm.create_model("vit_large_patch14_reg4_dinov2.lvd142m", pretrained=True, num_classes=0, img_size=224)
        self.m.set_grad_checkpointing(True)

    def forward(self, x, tt=None, ll=None, taps=(24,)):
        x = F.interpolate(x[:, :3], size=(224, 224), mode="bilinear", align_corners=False)
        t = self.m.forward_features(x)[:, self.m.num_prefix_tokens:]
        B, N, D = t.shape; n = int(N ** 0.5)
        return [t.transpose(1, 2).reshape(B, D, n, n)]


def ft_vessel_other(model, kind, seeds, steps, out):
    """Same cell as run_clay_missing_cells.ft_det_cell (data, head, budget, metric) with a different backbone."""
    E7 = rcm.load_e7()
    spec = rcm.DET_SPEC[kind]
    if model == "dofa_l":
        waves = rsb.DOFA_S1_2B_WAVES if kind == "vessel-s1" else spec["waves"]
        rcm.DET_BACKBONE_FACTORY = lambda E7_, k: DofaDetBackbone(E7_, waves)
        rcm.TAG = ""
    else:
        rcm.DET_BACKBONE_FACTORY = lambda E7_, k: DinoDetBackbone()
        rcm.TAG = "_dinov2"
    rcm.DET_MODEL = model
    rcm.ft_det_cell(kind, seeds, steps, out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cells", required=True, help="lp_mados,ft_mados,ft_dofa_vessel-s1,ft_dofa_vessel-l8,ft_dinov2_vessel-l8")
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--steps", type=int, default=6000, help="detection cells: optimiser steps (same as the Clay vessel cells)")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    rcm.SMOKE = a.smoke
    out = Path(a.out)
    rsb.SCRATCH.mkdir(parents=True, exist_ok=True)
    torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
    for cell in a.cells.split(","):
        t0 = time.time(); print(f"\n===== {cell} =====", flush=True)
        if cell == "lp_mados":
            lp_mados(out)
        elif cell == "ft_mados":
            ft_mados([int(s) for s in a.seeds.split(",")], a.epochs, out)
        elif cell in ("ft_dofa_vessel-s1", "ft_dofa_vessel-l8", "ft_dinov2_vessel-l8"):
            _, mdl, kind = cell.split("_", 2)
            ft_vessel_other({"dofa": "dofa_l", "dinov2": "dinov2_l"}[mdl], kind, [int(x) for x in a.seeds.split(",")], a.steps, out)
        else:
            raise SystemExit(f"unknown cell {cell}")
        print(f"===== {cell} done in {time.time()-t0:.0f}s =====", flush=True)


if __name__ == "__main__":
    main()
