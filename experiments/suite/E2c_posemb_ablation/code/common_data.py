"""Chip loading and model construction shared by train/eval (E2c)."""
import json, sys, math
from pathlib import Path
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from einops import rearrange, reduce
sys.path.insert(0, "common")
import claymodel.model as CM
from clayrt import platform_spec
from evalkit import hblock, block_split
from posembs import ARMS

PLAT = {"s2": "sentinel-2-l2a", "s1": "sentinel-1-rtc", "ls": "landsat-c2l2-sr", "naip": "naip"}
SENSORS = ["s2", "s1", "ls", "naip"]

def chip_index(DATA, s):
    """Same pid order, blocks and split as E2 embed.py."""
    logs = [json.loads(l) for l in open(DATA / f"log_{s}.jsonl")]
    meta = {r["pid"]: r for r in logs if r.get("ok")}
    pids = sorted(p for p in meta if (DATA / "chips" / s / f"{p}.npz").exists())
    lat = np.array([meta[p]["lat"] for p in pids]); lon = np.array([meta[p]["lon"] for p in pids])
    blocks = hblock(lat, lon); split = block_split(blocks)
    return pids, lat, lon, blocks, split

def load_norm(DATA, s, pids):
    spec = platform_spec(PLAT[s]); mu = np.array(spec["means"], np.float32)[:, None, None]; sd = np.array(spec["stds"], np.float32)[:, None, None]
    X = np.empty((len(pids), len(spec["means"]), 256, 256), np.float16)
    for i, p in enumerate(pids):
        x = np.load(DATA / "chips" / s / f"{p}.npz")["pixels"].astype(np.float32)
        assert np.isfinite(x).all()
        X[i] = (x - mu) / sd
    return X, spec

class SmallClay(nn.Module):
    """clay_mae_small encoder+decoder (pinned Clay source), MAE reconstruction only (no teacher)."""
    def __init__(self, arm, mask_ratio=0.75):
        super().__init__()
        a = dict(dim=384, depth=6, heads=6, dim_head=64, mlp_ratio=2, decoder_dim=192, decoder_depth=4,
                 decoder_heads=4, decoder_dim_head=64, decoder_mlp_ratio=2)
        self.arm = arm; self.patch_size = 8
        self.encoder = CM.Encoder(mask_ratio=mask_ratio, patch_size=8, shuffle=True, dim=a["dim"], depth=a["depth"],
                                  heads=a["heads"], dim_head=a["dim_head"], mlp_ratio=a["mlp_ratio"])
        self.decoder = CM.Decoder(mask_ratio=mask_ratio, patch_size=8, encoder_dim=a["dim"], dim=a["decoder_dim"],
                                  depth=a["decoder_depth"], heads=a["decoder_heads"], dim_head=a["decoder_dim_head"],
                                  mlp_ratio=a["decoder_mlp_ratio"])
    def _pe(self):
        CM.posemb_sincos_2d_with_gsd = ARMS[self.arm]  # both Encoder and Decoder look this name up at call time
    def forward(self, px, waves, gsd):
        """Clay ClayMAE.forward minus teacher/proj; time+latlon zeros (so Clay's channel dropout never fires)."""
        self._pe(); B = px.shape[0]
        z = torch.zeros(B, 4, device=px.device)
        enc, ui, mi, mm = self.encoder({"pixels": px, "time": z, "latlon": z, "gsd": gsd, "waves": waves})
        pix, _ = self.decoder(enc, ui, mi, mm, z, z, gsd, waves)
        tgt = rearrange(px, "B C (h p1) (w p2) -> B (h w) (C p1 p2)", p1=8, p2=8)
        l = reduce(F.l1_loss(tgt.float(), pix.float(), reduction="none"), "B L D -> B L", "mean")
        per_chip = (l * mm).sum(1) / mm.sum(1)
        return per_chip.mean(), per_chip
    @torch.no_grad()
    def tokens(self, px, waves, gsd):
        self._pe(); B = px.shape[0]; z = torch.zeros(B, 4, device=px.device)
        mr, sh = self.encoder.mask_ratio, self.encoder.shuffle
        self.encoder.mask_ratio, self.encoder.shuffle = 0.0, False
        tok, _, mi, _ = self.encoder({"pixels": px, "time": z, "latlon": z, "gsd": gsd, "waves": waves})
        self.encoder.mask_ratio, self.encoder.shuffle = mr, sh
        assert mi.numel() == 0
        return tok[:, 0].float(), tok[:, 1:].float()
