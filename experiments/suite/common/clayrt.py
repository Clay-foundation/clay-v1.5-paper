"""Shared runtime for the Clay paper experiments.

Clay v1.5 encoder from pinned source (Clay-foundation/model @ f14e698), loaded strictly from
the released checkpoint (SHA-256 pinned). Eval mode, no masking, no shuffling.

    rt = ClayRuntime(ckpt_path, device="cuda")
    out = rt.embed(pixels_raw, platform="sentinel-2-l2a", latlon=(lat, lon) or None, date=dt or None)
    out["cls"]  -> [B,1024]; out["patches"] -> [B, H/8, W/8, 1024]; out["mean"] -> [B,1024]

`pixels_raw` are raw sensor values (B,C,H,W) in the platform's band order from configs/metadata.yaml.
The runtime applies the platform mean/std, wavelengths and GSD. To use a band subset or a custom
sensor, pass bands=[...] (names from metadata) or waves/means/stds/gsd explicitly.
Metadata convention: missing time/location = zeros (as in training).
"""
from __future__ import annotations

import hashlib, json, math, os, sys, time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
CKPT_SHA256 = "21432069250b9b3f9a65ffd0071c5ad56b793247285ab0604edf7f531d4798d0"
DEFAULT_CKPT = os.environ.get("CLAY_V15_CKPT", "checkpoints/v1.5/clay-v1.5.ckpt")
EXPECTED_TENSORS, EXPECTED_ELEMENTS = 265, 311_442_944
METADATA = yaml.safe_load((HERE / "configs/metadata.yaml").read_text())


def sha256(path, chunk=1 << 24):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()


def encode_time(date: datetime | None):
    if date is None:
        return [0.0] * 4
    week = date.isocalendar().week * 2 * math.pi / 52
    hour = date.hour * 2 * math.pi / 24
    return [math.sin(week), math.cos(week), math.sin(hour), math.cos(hour)]


LATLON_ORDER = "training"  # "training" = [sin lon, cos lon, sin lat, cos lat] as Clay v1.0/v1.5 were trained;
                           # "documented" = [sin lat, cos lat, sin lon, cos lon] as the docs/inference code describe.
# stacchip prechip.py (since 2024-04-19): `lon_norm, lat_norm = normalize_latlon(bounds)` while normalize_latlon
# returns (lat, lon); datamodule stacks (lat_norm, lon_norm) -> the model saw longitude in the "lat" slots.


def encode_latlon(lat: float | None, lon: float | None, order: str | None = None):
    if lat is None:
        return [0.0] * 4
    la, lo = math.radians(lat), math.radians(lon)
    if (order or LATLON_ORDER) == "training":
        return [math.sin(lo), math.cos(lo), math.sin(la), math.cos(la)]
    return [math.sin(la), math.cos(la), math.sin(lo), math.cos(lo)]


def platform_spec(platform: str, bands: list[str] | None = None):
    m = METADATA[platform]
    bands = bands or list(m["band_order"])
    return {"bands": bands,
            "waves": [float(m["bands"]["wavelength"][b]) for b in bands],
            "means": [float(m["bands"]["mean"][b]) for b in bands],
            "stds": [float(m["bands"]["std"][b]) for b in bands],
            "gsd": float(m["gsd"])}


class ClayRuntime:
    def __init__(self, ckpt: str = DEFAULT_CKPT, device: str = "cuda", verify_sha: bool = True,
                 dtype=torch.float32):
        from claymodel.model import Encoder
        t0 = time.time()
        if verify_sha:
            got = sha256(ckpt)
            if got != CKPT_SHA256:
                raise ValueError(f"checkpoint sha256 {got} != pinned {CKPT_SHA256}")
        sd = torch.load(ckpt, map_location="cpu", weights_only=True, mmap=True)["state_dict"]
        state = {k.removeprefix("model.encoder."): v for k, v in sd.items() if k.startswith("model.encoder.")}
        enc = Encoder(mask_ratio=0.0, patch_size=8, shuffle=False, dim=1024, depth=24, heads=16,
                      dim_head=64, mlp_ratio=4.0)
        res = enc.load_state_dict(state, strict=True)
        n_el = sum(p.numel() for p in enc.parameters())
        if len(state) != EXPECTED_TENSORS or n_el != EXPECTED_ELEMENTS or res.missing_keys or res.unexpected_keys:
            raise ValueError(f"strict load failed: {len(state)} tensors, {n_el} params, {res}")
        self.proj_w = sd["model.proj.weight"].float().numpy()  # CLS -> DINOv2 space (for analysis)
        self.enc = enc.eval().requires_grad_(False).to(device=device, dtype=dtype)
        self.device, self.dtype = device, dtype
        self.evidence = {"ckpt": str(ckpt), "sha256": CKPT_SHA256 if verify_sha else "unverified",
                         "tensors": len(state), "params": n_el, "load_s": round(time.time() - t0, 1),
                         "source": "Clay-foundation/model@f14e698f3c237cabf8d28dec669a362d66625381",
                         "torch": torch.__version__}

    @torch.inference_mode()
    def embed(self, pixels, platform: str | None = None, bands=None, waves=None, means=None, stds=None,
              gsd=None, latlon=None, date=None, normalize=True, batch_size=32, latlon_order=None):
        """pixels: np/tensor (B,C,H,W) raw values. latlon: None | (lat,lon) | list of (lat,lon) | None per item.
        date: None | datetime | list. Returns dict of numpy arrays: cls, mean, patches."""
        if platform is not None:
            spec = platform_spec(platform, bands)
            waves = waves or spec["waves"]; means = means or spec["means"]; stds = stds or spec["stds"]
            gsd = gsd or spec["gsd"]
        x = torch.as_tensor(np.asarray(pixels), dtype=torch.float32)
        if not torch.isfinite(x).all():
            raise ValueError(f"non-finite input pixels ({int((~torch.isfinite(x)).sum())} values); mask or fill before embedding")
        B, C, H, W = x.shape
        assert H == W and H % 8 == 0, "square, patch-divisible chips required"
        assert len(waves) == C, f"{len(waves)} wavelengths for {C} channels"
        if normalize:
            x = (x - torch.tensor(means)[None, :, None, None]) / torch.tensor(stds)[None, :, None, None]
        ll = latlon if isinstance(latlon, list) else [latlon] * B
        dd = date if isinstance(date, list) else [date] * B
        ll = torch.tensor([encode_latlon(*(p if p is not None else (None, None)), order=latlon_order) for p in ll], dtype=torch.float32)
        tt = torch.tensor([encode_time(d) for d in dd], dtype=torch.float32)
        w = torch.tensor(waves, dtype=torch.float32, device=self.device)
        g = torch.tensor(float(gsd), dtype=torch.float32, device=self.device)
        cls, mean, pat = [], [], []
        for i in range(0, B, batch_size):
            cube = {"pixels": x[i:i + batch_size].to(self.device, self.dtype), "time": tt[i:i + batch_size].to(self.device),
                    "latlon": ll[i:i + batch_size].to(self.device), "gsd": g, "waves": w}
            tok, _, masked_idx, _ = self.enc(cube)
            assert masked_idx.numel() == 0
            tok = tok.float()
            n = H // 8
            cls.append(tok[:, 0].cpu()); mean.append(tok[:, 1:].mean(1).cpu())
            pat.append(tok[:, 1:].reshape(-1, n, n, tok.shape[-1]).cpu())
        out = {"cls": torch.cat(cls).numpy(), "mean": torch.cat(mean).numpy(), "patches": torch.cat(pat).numpy()}
        if not all(np.isfinite(v).all() for v in out.values()):
            raise ValueError("non-finite embeddings")
        return out


class DinoV2Runtime:
    """Frozen DINOv2 ViT-L/14-reg4 (Clay's teacher) on RGB. Input: (B,3,H,W) float in [0,1] (display RGB).
    Returns pooled ('cls' = timm pooled output) and patch tokens."""
    def __init__(self, name="vit_large_patch14_reg4_dinov2.lvd142m", device="cuda", size=224):
        import timm
        self.m = timm.create_model(name, pretrained=True, num_classes=0, img_size=size).eval().to(device)
        cfg = self.m.pretrained_cfg
        self.mean = torch.tensor(cfg["mean"]).view(1, 3, 1, 1); self.std = torch.tensor(cfg["std"]).view(1, 3, 1, 1)
        self.device, self.size, self.name = device, size, name

    @torch.inference_mode()
    def embed(self, rgb01, batch_size=64):
        x = torch.as_tensor(np.asarray(rgb01), dtype=torch.float32)
        x = torch.nn.functional.interpolate(x, size=(self.size, self.size), mode="bilinear", align_corners=False)
        x = (x - self.mean) / self.std
        cls, mean = [], []
        for i in range(0, len(x), batch_size):
            f = self.m.forward_features(x[i:i + batch_size].to(self.device))
            npre = getattr(self.m, "num_prefix_tokens", 1)
            cls.append(self.m.forward_head(f, pre_logits=True).float().cpu()); mean.append(f[:, npre:].mean(1).float().cpu())
        return {"cls": torch.cat(cls).numpy(), "mean": torch.cat(mean).numpy()}


def to_display_rgb(pixels, platform, bands=("red", "green", "blue"), lo=2, hi=98):
    """Raw sensor array (B,C,H,W) -> per-chip percentile-stretched RGB in [0,1] for natural-image baselines."""
    order = list(METADATA[platform]["band_order"])
    idx = [order.index(b) for b in bands]
    x = np.asarray(pixels, dtype=np.float32)[:, idx]
    p_lo = np.percentile(x, lo, axis=(2, 3), keepdims=True); p_hi = np.percentile(x, hi, axis=(2, 3), keepdims=True)
    return np.clip((x - p_lo) / np.maximum(p_hi - p_lo, 1e-6), 0, 1)


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))
