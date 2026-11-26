"""Embed the E1 chips with single-date OlmoEarth (v1.3 Base) and Prithvi-EO-2.0-300M-TL and cache the pooled features.

Recovered from the (previously unreleased) producer of e1_e5_complete.json and made portable: every path is a CLI flag.
Needs a GPU (about 10 GB) and the `olmoearth_pretrain` package (separate environment).  The CPU stage
run_e1_e5_complete.py only reads the cached npz, so this script is needed only to regenerate that cache.

  python embed_gpu_arms.py --pixels_dir E1_OUT/pixels --meta E1_OUT/meta.csv --out e5_gpu_feats.npz \
         [--olmo_dir HF_SNAPSHOT_DIR_WITH_config.json_and_weights.pth] [--prithvi_dir DIR]

Defaults: --olmo_dir = newest ~/.cache/huggingface/hub/models--allenai--OlmoEarth-v1_3-Base/snapshots/*,
          --prithvi_dir = ~/.cache/huggingface/hub/prithvi.
Output npz keys: ids, olmo_mean, olmo_center, prithvi_mean, prithvi_center (float16).  `center` = mean of the central
4x4 (OlmoEarth, patch 4 on 128 px -> 32x32 grid) / 2x2 (Prithvi, 14x14 grid) tokens; `mean` = mean of all tokens.
"""
import argparse, glob, json, os, sys, time, types
from datetime import datetime
import numpy as np
import pandas as pd
import torch
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


def embed_all(pixels_dir, meta_csv, out, olmo_dir, prithvi_dir, max_shards=0):
    meta = pd.read_csv(meta_csv)
    c2meta = {}
    for _, r in meta.iterrows():
        dt_str = str(r.datetime)
        try:
            dt = datetime.fromisoformat(dt_str.replace("Z", "+00:00"))
            ts = [dt.day, dt.month - 1, dt.year]
        except Exception:
            ts = [15, 6, 2020]
        c2meta[str(r.chip_id)] = (ts, float(r.baseline) if pd.notna(r.baseline) else 2.12)

    p = os.path.join(olmo_dir, "config.json"); w = os.path.join(olmo_dir, "weights.pth")
    cfg = json.load(open(p))
    ecfg = cfg["model"]["encoder_config"]
    ecfg.pop("attention_pooler_config", None)
    olmo = Config.from_dict(ecfg).build().cuda().eval()
    sd = torch.load(w, map_location="cpu", weights_only=False)
    enc_sd = {k[len("encoder."):]: v for k, v in sd.items() if k.startswith("encoder.") and not k.startswith("encoder.attention_pooler")}
    olmo.load_state_dict(enc_sd, strict=True)
    norm_olmo = Normalizer(Strategy.COMPUTED)

    prithvi = PrithviV2(load_directory=prithvi_dir, size=PrithviV2Models.VIT_300).cuda().eval()

    idx10 = [0, 1, 2, 6, 3, 4, 5, 7, 8, 9]
    files = sorted(glob.glob(f"{pixels_dir}/pixels_*.npz"))
    if max_shards: files = files[:max_shards]  # verification subset: compare against a cached full run with compare_gpu_cache.py
    ids_all = []
    feats = {"olmo_mean": [], "olmo_center": [], "prithvi_mean": [], "prithvi_center": []}
    t0 = time.time()
    for si, f in enumerate(files):
        z = np.load(f)
        cids = [str(c) for c in z["ids"].tolist()]
        px = z["pixels"].astype(np.float32)  # (N, 10, 128, 128)
        N = len(cids)
        ts_list = []
        for i, cid in enumerate(cids):
            ts, bl = c2meta.get(cid, ([15, 6, 2020], 2.12))
            ts_list.append(ts)
            if bl >= 4.0:
                px[i] = np.clip(px[i] - 1000.0, 0.0, None)
        s2_12 = np.zeros((N, 128, 128, 1, 12), dtype=np.float32)
        s2_12[..., :10] = np.transpose(px[:, idx10], (0, 2, 3, 1))[..., None, :]
        s2_olmo = norm_olmo.normalize(Modality.SENTINEL2_L2A, s2_12).astype(np.float32)
        s2_olmo[..., 10:] = 0.0

        bs = 64
        om_m, om_c, pr_m, pr_c = [], [], [], []
        for b0 in range(0, N, bs):
            b1 = min(N, b0 + bs)
            B = b1 - b0
            ts_t = torch.tensor(ts_list[b0:b1], dtype=torch.int64, device="cuda")[:, None, :]
            mask_t = torch.zeros((B, 128, 128, 1, 1), dtype=torch.int64, device="cuda")
            with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
                samp_o = MaskedOlmoEarthSample(
                    timestamps=ts_t,
                    sentinel2_l2a=torch.from_numpy(s2_olmo[b0:b1]).cuda(),
                    sentinel2_l2a_mask=mask_t,
                )
                out_o = olmo(samp_o, patch_size=4, fast_pass=True)["tokens_and_masks"]
                grid_o = pool_unmasked_tokens(out_o, PoolingType.MEAN, spatial_pooling=True).float() # (B, 32, 32, 768)
                om_m.append(grid_o.mean(dim=(1, 2)).cpu().numpy())
                om_c.append(grid_o[:, 14:18, 14:18, :].mean(dim=(1, 2)).cpu().numpy())

                samp_p = MaskedOlmoEarthSample(
                    timestamps=ts_t,
                    sentinel2_l2a=torch.from_numpy(s2_12[b0:b1]).cuda(),
                    sentinel2_l2a_mask=mask_t,
                )
                grid_p = prithvi(samp_p, spatial_pool=True).float() # (B, 14, 14, 1024)
                pr_m.append(grid_p.mean(dim=(1, 2)).cpu().numpy())
                pr_c.append(grid_p[:, 6:8, 6:8, :].mean(dim=(1, 2)).cpu().numpy())

        ids_all.extend(cids)
        feats["olmo_mean"].append(np.concatenate(om_m, 0).astype(np.float16))
        feats["olmo_center"].append(np.concatenate(om_c, 0).astype(np.float16))
        feats["prithvi_mean"].append(np.concatenate(pr_m, 0).astype(np.float16))
        feats["prithvi_center"].append(np.concatenate(pr_c, 0).astype(np.float16))
        if (si + 1) % 20 == 0 or si + 1 == len(files):
            print(f"Embedded shard {si+1}/{len(files)} ({len(ids_all)} chips) in {time.time()-t0:.1f}s", flush=True)

    feats = {k: np.concatenate(v, 0) for k, v in feats.items()}
    np.savez_compressed(out, ids=np.array(ids_all), **feats)
    print("Saved GPU feats to", out, flush=True)
    del olmo, prithvi
    torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pixels_dir", required=True); ap.add_argument("--meta", required=True); ap.add_argument("--out", required=True)
    hub = os.path.expanduser("~/.cache/huggingface/hub")
    snaps = sorted(glob.glob(f"{hub}/models--allenai--OlmoEarth-v1_3-Base/snapshots/*"))
    ap.add_argument("--olmo_dir", default=snaps[-1] if snaps else "")
    ap.add_argument("--prithvi_dir", default=f"{hub}/prithvi")
    ap.add_argument("--max_gpu_gb", type=float, default=10.0)
    ap.add_argument("--max_shards", type=int, default=0, help="embed only the first N pixel shards (verification subset)")
    a = ap.parse_args()
    torch.cuda.set_per_process_memory_fraction(a.max_gpu_gb / (torch.cuda.get_device_properties(0).total_memory / 2**30))
    torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
    embed_all(a.pixels_dir, a.meta, a.out, a.olmo_dir, a.prithvi_dir, a.max_shards)


if __name__ == "__main__":
    main()
