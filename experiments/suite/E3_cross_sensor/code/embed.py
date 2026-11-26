"""E3 embeddings: Clay v1.5 (meta ON/OFF) for S2/NAIP/S1/Landsat, DINOv2-L RGB, raw RGBN controls.
Usage: python embed.py BUILD_DIR OUT.npz"""
import glob, json, os, sys, time
from datetime import datetime
import numpy as np, torch
sys.path.insert(0, "common")
from clayrt import ClayRuntime, DinoV2Runtime, to_display_rgb

bdir, outp = sys.argv[1], sys.argv[2]
rows = json.load(open(os.path.join(bdir, "rows.json")))
keep = {r["loc_id"] for r in rows}
D = {k: [] for k in ("s2", "naip", "s1", "ls", "loc_id")}
for f in sorted(glob.glob(os.path.join(bdir, "tiles", "*.npz"))):
    z = np.load(f)
    m = np.array([l in keep for l in z["loc_id"]])
    for k in D:
        D[k].append(z[k][m])
D = {k: np.concatenate(v) for k, v in D.items()}
order = {l: i for i, l in enumerate(D["loc_id"])}
idx = np.array([order[r["loc_id"]] for r in rows])
D = {k: v[idx] for k, v in D.items()}
assert list(D["loc_id"]) == [r["loc_id"] for r in rows]
N = len(rows); print("N", N, flush=True)
P = lambda s: datetime.fromisoformat(s)
ll = [(r["lat"], r["lon"]) for r in rows]
dates = {"s2": [P(r["s2_date"]) for r in rows], "naip": [P(r["naip_date"]) for r in rows],
         "s1": [P(r["s1_date"]) if r["has_s1"] else None for r in rows],
         "ls": [P(r["ls_date"]) if r["has_ls"] else None for r in rows]}
has = {"s2": np.ones(N, bool), "naip": np.ones(N, bool), "s1": np.array([r["has_s1"] for r in rows]),
       "ls": np.array([r["has_ls"] for r in rows])}
plat = {"s2": ("sentinel-2-l2a", None), "naip": ("naip", 1.25), "s1": ("sentinel-1-rtc", None),
        "ls": ("landsat-c2l2-sr", None)}
out = {"loc_id": D["loc_id"]}
for s in has:
    out[f"has_{s}"] = has[s]
t0 = time.time()
rt = ClayRuntime(device="cuda")
out["clay_evidence"] = np.array(json.dumps(rt.evidence))
for s, (p, g) in plat.items():
    ii = np.where(has[s])[0]
    X = D[s][ii].astype(np.float32)
    for meta in ("on", "off"):
        cl = np.full((N, 1024), np.nan, np.float16); mn = cl.copy()
        bs = 8 if s == "naip" else 64
        for a in range(0, len(ii), 512):
            sl = slice(a, a + 512)
            kw = dict(latlon=[ll[i] for i in ii[sl]], date=[dates[s][i] for i in ii[sl]]) if meta == "on" else {}
            e = rt.embed(X[sl], platform=p, gsd=g, batch_size=bs, **kw)
            cl[ii[sl]] = e["cls"]; mn[ii[sl]] = e["mean"]
        out[f"clay_{s}_{meta}_cls"] = cl; out[f"clay_{s}_{meta}_mean"] = mn
        print(s, meta, len(ii), f"{time.time()-t0:.0f}s", "gpu GB", round(torch.cuda.max_memory_allocated() / 1e9, 2), flush=True)
del rt; torch.cuda.empty_cache()
dino = DinoV2Runtime(size=224)
for s in ("s2", "naip", "ls"):
    ii = np.where(has[s])[0]
    cl = np.full((N, 1024), np.nan, np.float16); mn = cl.copy()
    for a in range(0, len(ii), 512):
        sl = ii[a:a + 512]
        e = dino.embed(to_display_rgb(D[s][sl].astype(np.float32), plat[s][0]), batch_size=64)
        cl[sl] = e["cls"]; mn[sl] = e["mean"]
    out[f"dino_{s}_cls"] = cl; out[f"dino_{s}_mean"] = mn
    print("dino", s, f"{time.time()-t0:.0f}s", flush=True)
# raw RGBN controls (S2 bands blue,green,red,...,nir at idx 2,1,0,6 -> R,G,B,NIR)
out["raw_s2_rgbn"] = D["s2"][:, [2, 1, 0, 6]].astype(np.float32).astype(np.float16)
out["raw_naip_rgbn"] = D["naip"].astype(np.float32).reshape(N, 4, 32, 8, 32, 8).mean((3, 5)).astype(np.float16)
out["raw_ls_rgbn"] = D["ls"][:, [0, 1, 2, 3]].astype(np.float16)
out["raw_s1"] = D["s1"].astype(np.float16)
np.savez_compressed(outp, **out)
print("saved", outp, f"{time.time()-t0:.0f}s", "gpu GB", round(torch.cuda.max_memory_allocated() / 1e9, 2), flush=True)
