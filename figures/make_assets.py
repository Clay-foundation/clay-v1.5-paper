#!/usr/bin/env python3
"""Regenerate Figure 1's chip input / reconstruction pair from the REAL released checkpoint.

Scene: one public-domain USDA NAIP item (US Government work) fetched from the Microsoft Planetary Computer
STAC API (collection 'naip'); the item id, bbox, acquisition date and licence are written to the JSON next to
the PNGs. Model: Clay v1.5 encoder + decoder loaded strictly from the pinned checkpoint (SHA-256 checked),
mask ratio 0.75 (as in pre-training), fixed seed. The input and the decoder prediction are displayed with the
SAME per-band stretch (percentiles computed on the input), so any colour difference is the model's, not the plot's.

Usage (GPU optional):
  python figures/make_assets.py --ckpt checkpoints/v1.5/clay-v1.5.ckpt --out-dir figures/assets_regen \
      [--lon -97.5 --lat 40.5 --year 2022 --seed 0]
Needs torch, einops, rasterio, pystac-client, planetary-computer, pillow, pyyaml.
Writes: asset_chip_in.png, asset_chip_rec.png, assets_provenance.json (+ metrics: masked-patch L1 in normalised units).
Does NOT overwrite the committed figures/asset_*.png; copy them over deliberately after inspection.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from einops import rearrange
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments" / "suite" / "common"))
from provenance import provenance  # noqa: E402

import clayrt  # noqa: E402
from claymodel.model import Decoder, Encoder  # noqa: E402


def fetch_naip(lon: float, lat: float, year: int, size: int = 256):
    import planetary_computer as pc
    import pystac_client
    import rasterio
    from rasterio.windows import Window
    from rasterio.warp import transform

    cat = pystac_client.Client.open("https://planetarycomputer.microsoft.com/api/stac/v1", modifier=pc.sign_inplace)
    items = list(cat.search(collections=["naip"], intersects={"type": "Point", "coordinates": [lon, lat]},
                            datetime=f"{year}-01-01/{year}-12-31", max_items=1).items())
    if not items:
        raise SystemExit("no NAIP item found; change --lon/--lat/--year")
    it = items[0]
    with rasterio.open(it.assets["image"].href) as src:
        xs, ys = transform("EPSG:4326", src.crs, [lon], [lat])
        r, c = src.index(xs[0], ys[0])
        arr = src.read(window=Window(c - size // 2, r - size // 2, size, size)).astype(np.float32)  # R,G,B,NIR
    meta = {"stac_item_id": it.id, "collection": "naip", "datetime": it.datetime.isoformat(), "bbox": it.bbox,
            "gsd_m": 1.0, "licence": "Public domain (USDA NAIP, US Government work)", "centre_lonlat": [lon, lat]}
    return arr, it, meta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=clayrt.DEFAULT_CKPT)
    ap.add_argument("--out-dir", default=str(ROOT / "figures" / "assets_regen"))
    ap.add_argument("--lon", type=float, default=-97.5)
    ap.add_argument("--lat", type=float, default=40.5)
    ap.add_argument("--year", type=int, default=2022)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)

    got = clayrt.sha256(a.ckpt)
    if got != clayrt.CKPT_SHA256:
        raise SystemExit(f"checkpoint sha256 mismatch: {got}")
    sd = torch.load(a.ckpt, map_location="cpu", weights_only=True, mmap=True)["state_dict"]
    enc = Encoder(mask_ratio=0.75, patch_size=8, shuffle=True, dim=1024, depth=24, heads=16, dim_head=64, mlp_ratio=4.0)
    dec = Decoder(mask_ratio=0.75, patch_size=8, encoder_dim=1024, dim=512, depth=4, heads=4, dim_head=64, mlp_ratio=4.0)
    r1 = enc.load_state_dict({k[len("model.encoder."):]: v for k, v in sd.items() if k.startswith("model.encoder.")}, strict=True)
    r2 = dec.load_state_dict({k[len("model.decoder."):]: v for k, v in sd.items() if k.startswith("model.decoder.")}, strict=True)
    enc, dec = enc.eval().to(a.device), dec.eval().to(a.device)

    raw, item, meta = fetch_naip(a.lon, a.lat, a.year)  # (4,256,256) R,G,B,NIR raw 0-255
    spec = clayrt.platform_spec("naip")
    mean = torch.tensor(spec["means"])[:, None, None]; std = torch.tensor(spec["stds"])[:, None, None]
    x = ((torch.from_numpy(raw) - mean) / std)[None]
    waves = torch.tensor(spec["waves"], device=a.device); gsd = torch.tensor(spec["gsd"], device=a.device)
    date = item.datetime.replace(tzinfo=None) if item.datetime else datetime(a.year, 6, 1)
    tt = torch.tensor([clayrt.encode_time(date)], dtype=torch.float32, device=a.device)
    ll = torch.tensor([clayrt.encode_latlon(a.lat, a.lon)], dtype=torch.float32, device=a.device)

    torch.manual_seed(a.seed); np.random.seed(a.seed)
    with torch.inference_mode():
        cube = {"pixels": x.to(a.device), "time": tt, "latlon": ll, "gsd": gsd, "waves": waves}
        enc_tok, un_idx, m_idx, m_mat = enc(cube)
        pix, _ = dec(enc_tok, un_idx, m_idx, m_mat, tt, ll, gsd, waves)
    patches = rearrange(x.to(a.device), "B C (h p1) (w p2) -> B (h w) (C p1 p2)", p1=8, p2=8)
    masked_l1 = float(((pix - patches).abs().mean(-1) * m_mat).sum() / m_mat.sum())
    base_zero = float(((patches).abs().mean(-1) * m_mat).sum() / m_mat.sum())  # predict the dataset mean (0 after z-scoring)
    unmasked_l1 = float(((pix - patches).abs().mean(-1) * (1 - m_mat)).sum() / (1 - m_mat).sum())
    rec = rearrange(pix.float().cpu(), "B (h w) (C p1 p2) -> B C (h p1) (w p2)", h=32, w=32, p1=8, p2=8)[0]
    # Standard MAE visualisation: visible (unmasked) patches come from the input, masked patches from the decoder.
    # (The decoder is only trained on masked patches; its output on visible patches is not a reconstruction.)
    keep = m_mat.float().cpu().reshape(1, 32, 32)[0].repeat_interleave(8, 0).repeat_interleave(8, 1)[None]
    rec = rec * keep + x[0] * (1 - keep)
    rec_raw = rec * std + mean

    # False colour NIR,R,G (as the committed asset); one stretch for both images, from the input.
    idx = [3, 0, 1]
    lo = np.percentile(raw[idx], 2, axis=(1, 2), keepdims=True); hi = np.percentile(raw[idx], 98, axis=(1, 2), keepdims=True)
    def show(a_):
        v = np.clip((a_[idx] - lo) / np.maximum(hi - lo, 1e-6), 0, 1)
        im = Image.fromarray((v.transpose(1, 2, 0) * 255).astype(np.uint8))
        return im.resize((128, 128), Image.BILINEAR)
    show(raw).save(out / "asset_chip_in.png"); show(rec_raw.numpy()).save(out / "asset_chip_rec.png")
    meta.update({
        "model": "Clay v1.5", "ckpt_sha256": got, "mask_ratio": 0.75, "seed": a.seed,
        "encoder_load": str(r1), "decoder_load": str(r2),
        "masked_patch_l1_normalised": masked_l1,
        "baseline_masked_l1_predict_dataset_mean": base_zero,
        "unmasked_patch_l1_normalised": unmasked_l1,
        "masked_patches": int(m_mat.sum().item()), "total_patches": int(m_mat.numel()),
        "display": "false colour NIR,R,G; 2-98 percentile stretch computed on the input, applied to both; 128x128 bilinear",
        "note": ("Prediction image = decoder output on the 768 masked patches, input pixels on the 256 visible ones. "
                 "Default scene was picked by eye as a clear example; a second scene (urban, --lon -117.24 --lat 32.70) "
                 "reconstructs much worse (masked L1 0.657 vs 0.781 for predicting the dataset mean) and is reported in the audit log. "
                 "The committed strip assets (asset_strip_*.png) are NOT regenerated here."),
        "provenance": provenance(__file__, {**vars(a), "ckpt": Path(a.ckpt).name, "out_dir": Path(a.out_dir).name}, ROOT)})
    (out / "assets_provenance.json").write_text(json.dumps(meta, indent=2, default=str) + "\n")
    print(json.dumps({k: meta[k] for k in ("stac_item_id", "datetime", "masked_patch_l1_normalised")}, default=str))


if __name__ == "__main__":
    main()
