#!/usr/bin/env python3
"""Instantiate the Clay v0 model exactly as released at tag v0.0.1 and count encoder/decoder parameters.

Why: README/DRAFT/LaTeX say the v0 encoder has 90.6M parameters; audit issue #33 says 95.3M.
Source code: Clay-foundation/model tag v0.0.1 (commit 0145e55bcf6bb... see provenance), src/model_clay.py,
CLAYModule defaults (model_size='small', image_size=512, patch_size=32) and CLAY defaults (13 bands,
6 band groups). The released v0 checkpoints (made-with-clay/Clay-legacy v0/*.ckpt) store the same
hyper-parameters; a separate shape-only read of those files gave encoder 95,271,168 / decoder 32,431,104
(recorded in the output JSON under 'checkpoint_shape_read'; the script checks that both agree).

Usage:
  git clone https://github.com/Clay-foundation/model && git -C model archive v0.0.1 | tar -x -C v0src
  python experiments/table1/count_v0_params.py --v0-src v0src [--out experiments/table1/v0_param_count.json]
Needs: torch, einops, vit-pytorch (pip install vit-pytorch). Lightning/geopandas/etc. are stubbed
because only nn.Module construction is needed. No GPU. Seed fixed (irrelevant to counts).
"""
from __future__ import annotations

import argparse
import json
import sys
import types
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments" / "suite" / "common"))
from provenance import provenance  # noqa: E402

# Shape-only reads of the released v0 checkpoints (HTTP range read of the zip, no tensors downloaded);
# produced by the audit script legacy_ckpt_params.py on the HF Clay-legacy repo. Recorded, not recomputed here.
CKPT_SHAPE_READ = {
    "source_repo": "made-with-clay/Clay-legacy",
    "files": ["v0/Clay_v0.1_epoch-24_val-loss-0.46.ckpt", "v0/Clay-v0.0_epoch-02_val-loss-0.52.ckpt"],
    "encoder": 95271168, "decoder": 32431104, "tensors": 267,
    "hparams": {"model_size": "small", "mask_ratio": 0.7, "image_size": 512, "patch_size": 32, "lr": 0.0001},
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--v0-src", required=True, help="directory containing the v0.0.1 source tree (src/model_clay.py)")
    ap.add_argument("--out", default=str(ROOT / "experiments/table1/v0_param_count.json"))
    a = ap.parse_args()

    for m in ("lightning", "geopandas", "pandas", "pyarrow", "shapely"):
        if m not in sys.modules:
            try:
                __import__(m)
            except Exception:
                sys.modules[m] = mock.MagicMock()
    # lightning.LightningModule must be a real base class
    lt = types.ModuleType("lightning")
    import torch
    import inspect
    from types import SimpleNamespace

    class _LM(torch.nn.Module):  # minimal stand-in for lightning.LightningModule
        def save_hyperparameters(self, **_):
            loc = inspect.currentframe().f_back.f_locals
            self.hparams = SimpleNamespace(**{k: v for k, v in loc.items() if k not in ("self", "__class__")})

    lt.LightningModule = _LM
    sys.modules["lightning"] = lt
    sys.modules["pyarrow"] = mock.MagicMock()
    sys.path.insert(0, a.v0_src)
    import torch  # noqa: F811

    torch.manual_seed(0)
    from src import model_clay  # type: ignore

    mod = model_clay.CLAYModule()  # released defaults: small, 512 px, patch 32, mask 0.75 (ckpt: 0.7)
    enc = sum(p.numel() for p in mod.model.encoder.parameters())
    dec = sum(p.numel() for p in mod.model.decoder.parameters())
    res = {
        "model_size": mod.hparams.model_size, "image_size": mod.hparams.image_size, "patch_size": mod.hparams.patch_size,
        "encoder_dim": 768, "encoder_depth": 12, "encoder_heads": 12,
        "encoder_params": enc, "decoder_params": dec, "total_params": enc + dec,
        "encoder_params_millions": enc / 1e6,
        "paper_claim_encoder_millions": 90.6,
        "matches_checkpoint_shape_read": enc == CKPT_SHAPE_READ["encoder"] and dec == CKPT_SHAPE_READ["decoder"],
        "checkpoint_shape_read": CKPT_SHAPE_READ,
        "provenance": provenance(__file__, {"v0_src": "<dir>"}, ROOT),
    }
    Path(a.out).write_text(json.dumps(res, indent=2) + "\n")
    print(json.dumps({k: v for k, v in res.items() if k not in ("provenance", "checkpoint_shape_read")}, indent=1))


if __name__ == "__main__":
    main()
