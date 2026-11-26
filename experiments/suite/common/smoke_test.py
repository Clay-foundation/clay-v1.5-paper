"""Smoke test: verified checkpoint load, then real S2 chips -> Clay + DINOv2 teacher-alignment check.

Runs on CPU (minutes) or GPU. From the repo root:

    uv run python experiments/suite/common/smoke_test.py --ckpt checkpoints/v1.5/clay-v1.5.ckpt

Published values it reproduces (paper Sec. 2.3 / PROTOCOL-COMMON.md):
  * checkpoint SHA-256 21432069...98d0, strict load of 265 tensors, 311,442,944 encoder parameters;
  * projected class token aligns with DINOv2 on the same chip (mean cosine 0.87 on the diagonal vs
    0.72 off-diagonal; the pinned scene reproduces these two values, see PINNED_ITEM).
Inputs are public: Sentinel-2 L2A from the Planetary Computer STAC API (anonymous, signed on the fly) and the
DINOv2 weights from the Hugging Face hub. `--skip-stac` runs the checkpoint stage only (no network except HF).
Writes a results JSON with provenance (default experiments/results/smoke_test.json).
"""
import argparse, json, os, sys, time
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np
import torch
from clayrt import DEFAULT_CKPT, EXPECTED_ELEMENTS, EXPECTED_TENSORS, ClayRuntime, DinoV2Runtime, platform_spec, to_display_rgb
from provenance import provenance

# Scene of the published check (first hit of a <5% cloud search, SF Bay, Jun-Aug 2023). Other scenes give different
# absolute cosines, e.g. the water-heavy tile S2A_MSIL2A_20230618T184921_R113_T10SEG_20230619T034036 gives about 0.65 / 0.57.
PINNED_ITEM = "S2A_MSIL2A_20230827T184921_R113_T10SEH_20241024T140931"
ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", default=DEFAULT_CKPT, help="Clay v1.5 checkpoint (env CLAY_V15_CKPT; fetch_checkpoint.py downloads it)")
ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
ap.add_argument("--out", default=str(HERE.parents[1] / "results" / "smoke_test.json"))
ap.add_argument("--n-chips", type=int, default=16)
ap.add_argument("--dino-size", type=int, default=518)
ap.add_argument("--skip-stac", action="store_true")
ap.add_argument("--stac-item", default=PINNED_ITEM, help="Sentinel-2 L2A item id (default: the pinned San Francisco Bay scene used for the published check)")
args = ap.parse_args()
device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
torch.manual_seed(0)

t0 = time.time()
rt = ClayRuntime(args.ckpt, device=device)
res = {"device": device, "checkpoint": rt.evidence, "clay_load_s": round(time.time() - t0, 1)}
assert rt.evidence["tensors"] == EXPECTED_TENSORS and rt.evidence["params"] == EXPECTED_ELEMENTS
print(f"checkpoint OK: {rt.evidence['tensors']} tensors, {rt.evidence['params']:,} params ({device})")

if not args.skip_stac:
    import odc.stac, planetary_computer, pystac_client
    from shapely.geometry import shape
    cat = pystac_client.Client.open("https://planetarycomputer.microsoft.com/api/stac/v1", modifier=planetary_computer.sign_inplace)
    spec = platform_spec("sentinel-2-l2a")
    asset = {"blue": "B02", "green": "B03", "red": "B04", "rededge1": "B05", "rededge2": "B06", "rededge3": "B07",
             "nir": "B08", "nir08": "B8A", "swir16": "B11", "swir22": "B12"}
    it = next(cat.search(collections=["sentinel-2-l2a"], ids=[args.stac_item]).items())
    cen = shape(it.geometry).representative_point(); lon0, lat0 = cen.x, cen.y
    d = 0.06
    ds = odc.stac.load([it], bands=[asset[b] for b in spec["bands"]], resolution=10, bbox=[lon0 - d, lat0 - d, lon0 + d, lat0 + d],
                       dtype="uint16", nodata=0)
    arr = np.stack([ds[asset[b]].isel(time=0).values for b in spec["bands"]]).astype("float32")
    C, H, W = arr.shape
    chips = [arr[:, i:i + 256, j:j + 256] for i in range(0, H - 255, 256) for j in range(0, W - 255, 256)]
    chips = np.stack(chips); chips = chips[(chips > 0).all(axis=(1, 2, 3))][:args.n_chips]
    assert len(chips) >= 8, f"only {len(chips)} valid chips"
    date = datetime.fromisoformat(it.properties["datetime"].replace("Z", "+00:00"))
    t0 = time.time()
    out = rt.embed(chips, platform="sentinel-2-l2a", latlon=(lat0, lon0), date=date)
    t_embed = time.time() - t0
    dino = DinoV2Runtime(size=args.dino_size, device=device).embed(to_display_rgb(chips, "sentinel-2-l2a"))
    proj = out["cls"] @ rt.proj_w.T
    cos = lambda a, b: (a / np.linalg.norm(a, axis=1, keepdims=True)) @ (b / np.linalg.norm(b, axis=1, keepdims=True)).T
    S = cos(proj, dino["cls"])
    res.update({"n_chips": len(chips), "stac_item": it.id,
                "clay_shapes": {k: list(v.shape) for k, v in out.items()}, "dino_shape": list(dino["cls"].shape),
                "diag_mean_cos": float(np.diag(S).mean()),
                "offdiag_mean_cos": float(S[~np.eye(len(S), dtype=bool)].mean()),
                "top1_match": float((S.argmax(1) == np.arange(len(S))).mean()),
                "clay_embed_s": round(t_embed, 1)})
    print(f"alignment: diag {res['diag_mean_cos']:.3f} vs off-diag {res['offdiag_mean_cos']:.3f}, top-1 {res['top1_match']:.2f}")
if device == "cuda":
    res["gpu_mem_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
res["provenance"] = provenance(__file__, vars(args), root=HERE)
ok = True
if "diag_mean_cos" in res and args.stac_item == PINNED_ITEM:
    ok = abs(res["diag_mean_cos"] - 0.87) < 0.01 and abs(res["offdiag_mean_cos"] - 0.72) < 0.01
    res["matches_published_0.87_0.72"] = ok
Path(args.out).parent.mkdir(parents=True, exist_ok=True)
Path(args.out).write_text(json.dumps(res, indent=1, default=str) + "\n")
print(json.dumps(res, indent=1, default=str))
if not ok:
    sys.exit("smoke test: alignment does not match the published 0.87 / 0.72")
