"""E8 item 1 embeddings (GPU). Clay v1.5 (training lat/lon order, meta ON; meta OFF) and DINOv2-L on
S2-256 (10 m) and NAIP-256 (10 m, gsd=10), plus a compact copy of the cached E3 32 px embeddings.
Usage: python e8_embed.py E3_MANIFEST BUILD_DIR E3_EMB E3_EMB_TR OUTDIR
"""
import csv, glob, json, os, sys, time
from datetime import datetime
import numpy as np, torch
sys.path.insert(0, "common")
import clayrt
from clayrt import ClayRuntime, DinoV2Runtime, to_display_rgb
assert clayrt.LATLON_ORDER == "training"

man, bdir, e3emb, e3tr, out = sys.argv[1:6]
os.makedirs(out, exist_ok=True)
rows = list(csv.DictReader(open(man))); N = len(rows)
ids = [r["loc_id"] for r in rows]; pos = {l: i for i, l in enumerate(ids)}
meta = {m["loc_id"]: m for m in csv.DictReader(open(os.path.join(bdir, "e8_chips.csv")))}
S2 = np.zeros((N, 10, 256, 256), np.uint16); NA = np.zeros((N, 4, 256, 256), np.uint8); got = np.zeros(N, bool)
for f in sorted(glob.glob(os.path.join(bdir, "tiles", "*.npz"))):
    z = np.load(f)
    for j, l in enumerate(z["loc_id"]):
        S2[pos[l]] = z["s2"][j]; NA[pos[l]] = z["naip10"][j]; got[pos[l]] = True
v_s2 = np.array([got[i] and meta.get(l, {}).get("s2_valid") == "True" for i, l in enumerate(ids)])
v_na = np.array([got[i] and meta.get(l, {}).get("naip_valid") == "True" for i, l in enumerate(ids)])
print("N", N, "loaded", got.sum(), "s2 valid", v_s2.sum(), "naip valid", v_na.sum(), "both", (v_s2 & v_na).sum(), flush=True)
P = datetime.fromisoformat
ll = [(float(r["lat"]), float(r["lon"])) for r in rows]
dates = {"s2": [P(r["s2_date"]) for r in rows], "naip": [P(r["naip_date"]) for r in rows]}
O = {"loc_id": np.array(ids), "valid_s2": v_s2, "valid_naip": v_na}
t0 = time.time()
rt = ClayRuntime(device="cuda")
O["clay_evidence"] = np.array(json.dumps(rt.evidence))
spec = {"s2": (S2, "sentinel-2-l2a", 10.0, v_s2), "naip": (NA, "naip", 10.0, v_na)}
for s, (X, plat, g, v) in spec.items():
    ii = np.where(v)[0]
    for m in ("on", "off"):
        cl = np.full((N, 1024), np.nan, np.float16); mn = cl.copy(); ce = cl.copy()
        for a in range(0, len(ii), 64):
            sl = ii[a:a + 64]
            kw = dict(latlon=[ll[i] for i in sl], date=[dates[s][i] for i in sl], latlon_order="training") if m == "on" else {}
            e = rt.embed(X[sl].astype(np.float32), platform=plat, gsd=g, batch_size=8, **kw)
            cl[sl] = e["cls"]; mn[sl] = e["mean"]; ce[sl] = e["patches"][:, 14:18, 14:18].mean((1, 2))
        O[f"clay_{s}256_{m}_cls"] = cl; O[f"clay_{s}256_{m}_mean"] = mn; O[f"clay_{s}256_{m}_center"] = ce
        print(s, m, len(ii), f"{time.time()-t0:.0f}s", "gpu GB", round(torch.cuda.max_memory_allocated() / 1e9, 2), flush=True)
del rt; torch.cuda.empty_cache()
dino = DinoV2Runtime(size=224)
for s, (X, plat, g, v) in spec.items():
    ii = np.where(v)[0]; cl = np.full((N, 1024), np.nan, np.float16)
    for a in range(0, len(ii), 256):
        sl = ii[a:a + 256]
        cl[sl] = dino.embed(to_display_rgb(X[sl].astype(np.float32), plat), batch_size=32)["cls"]
    O[f"dino_{s}256_cls"] = cl
    print("dino", s, f"{time.time()-t0:.0f}s", flush=True)
del dino; torch.cuda.empty_cache()
# raw RGBN controls on the 256 chips (S2 idx 2,1,0,6 -> R,G,B,NIR), 8x8 mean-pooled to 32x32
O["raw_s2256_rgbn"] = S2[:, [2, 1, 0, 6]].astype(np.float32).reshape(N, 4, 32, 8, 32, 8).mean((3, 5)).astype(np.float16)
O["raw_naip256_rgbn"] = NA.astype(np.float32).reshape(N, 4, 32, 8, 32, 8).mean((3, 5)).astype(np.float16)
np.savez_compressed(os.path.join(out, "e8_emb.npz"), **O)
# compact copy of the cached E3 embeddings needed for the matched comparison
E = np.load(e3emb); T = np.load(e3tr)
assert list(E["loc_id"]) == ids and list(T["loc_id"]) == ids
K = {"loc_id": E["loc_id"]}
for s in ("s2", "naip"):
    K[f"clay_{s}32_on_cls"] = T[f"clay_{s}_ontr_cls"]; K[f"clay_{s}32_on_mean"] = T[f"clay_{s}_ontr_mean"]
    K[f"clay_{s}32_off_cls"] = E[f"clay_{s}_off_cls"]; K[f"dino_{s}32_cls"] = E[f"dino_{s}_cls"]
    K[f"raw_{s}32_rgbn"] = E[f"raw_{s}_rgbn"]
np.savez_compressed(os.path.join(out, "e3_sub.npz"), **K)
print("saved", f"{time.time()-t0:.0f}s", flush=True)
