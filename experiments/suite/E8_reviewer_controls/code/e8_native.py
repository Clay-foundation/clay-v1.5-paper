"""E8 item 1 secondary arm: NAIP-native (1.25 m) over the 2.56 km footprint, 8x8 tiles of 256 px, Clay CLS/mean
per tile (gsd 1.25, chip-centre lat/lon, NAIP date, training order), mean-pooled. Locations in sha256('claypaper-e8|loc_id')
order among those with valid S2-256 and NAIP-256; target 2,000; hard time cap.
Usage: python e8_native.py E3_MANIFEST BUILD_DIR OUTDIR TARGET CAP_SECONDS
"""
import csv, hashlib, json, os, sys, time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import numpy as np, torch
sys.path.insert(0, "common"); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import clayrt
from clayrt import ClayRuntime
from e8_build import read_naip, search, chip_bounds
assert clayrt.LATLON_ORDER == "training"

man, bdir, out, target, cap = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), float(sys.argv[5])
os.makedirs(out, exist_ok=True)
rows = {r["loc_id"]: r for r in csv.DictReader(open(man))}
meta = {m["loc_id"]: m for m in csv.DictReader(open(os.path.join(bdir, "e8_chips.csv")))}
L = [l for l, m in meta.items() if m["s2_valid"] == "True" and m["naip_valid"] == "True" and int(rows[l]["label"]) in (10, 20, 30, 40, 50, 60, 80)]
L.sort(key=lambda l: hashlib.sha256(f"claypaper-e8|{l}".encode()).hexdigest())
L = L[:target]
print("queue", len(L), flush=True)


def fetch(l):
    r, m = rows[l], meta[l]
    b = chip_bounds(r); epsg = int(r["epsg"])
    items = {i.id: i for i in search(collections=["naip"], ids=m["naip_items"].split("|"))}
    a = None
    for iid in m["naip_items"].split("|"):
        f = read_naip(items[iid].assets["image"].href, epsg, b, 2048)
        if a is None:
            a = f
        else:
            hole = (a == 0).all(0); fill = hole & ~(f == 0).all(0); a[:, fill] = f[:, fill]
    return l, a, float((a == 0).all(0).mean())


t0 = time.time()
rt = ClayRuntime(device="cuda")
ids, cls, mean, zf = [], [], [], []
with ThreadPoolExecutor(6) as ex:
    futs = [ex.submit(fetch, l) for l in L[:12]]; nxt = 12
    while futs:
        f = futs.pop(0)
        if nxt < len(L) and time.time() - t0 < cap:
            futs.append(ex.submit(fetch, L[nxt])); nxt += 1
        try:
            l, a, z = f.result()
        except Exception as e:
            print("fetch fail", repr(e)[:200], flush=True); continue
        r = rows[l]
        X = a.reshape(4, 8, 256, 8, 256).transpose(1, 3, 0, 2, 4).reshape(64, 4, 256, 256).astype(np.float32)
        for attempt in range(30):  # shared GPU: on OOM wait for memory (another service may hold it), then retry
            try:
                e = rt.embed(X, platform="naip", gsd=1.25, latlon=(float(r["lat"]), float(r["lon"])),
                             date=datetime.fromisoformat(r["naip_date"]), batch_size=8, latlon_order="training")
                break
            except torch.OutOfMemoryError:
                torch.cuda.empty_cache(); print("OOM, waiting", attempt, flush=True); time.sleep(60)
        else:
            raise RuntimeError("persistent OOM")
        ids.append(l); cls.append(e["cls"].mean(0)); mean.append(e["mean"].mean(0)); zf.append(z)
        if len(ids) % 100 == 0:
            print(len(ids), f"{time.time()-t0:.0f}s", "gpu GB", round(torch.cuda.max_memory_allocated() / 1e9, 2), flush=True)
            np.savez(os.path.join(out, "e8_native.npz"), loc_id=np.array(ids), clay_naipnat_on_cls=np.array(cls, np.float16),
                     clay_naipnat_on_mean=np.array(mean, np.float16), zero_frac=np.array(zf))
        if time.time() - t0 > cap:
            print("time cap reached", flush=True)
            for g in futs:
                g.cancel()
            break
np.savez(os.path.join(out, "e8_native.npz"), loc_id=np.array(ids), clay_naipnat_on_cls=np.array(cls, np.float16),
         clay_naipnat_on_mean=np.array(mean, np.float16), zero_frac=np.array(zf))
json.dump({"queued": len(L), "done": len(ids), "elapsed_s": round(time.time() - t0), "cap_s": cap},
          open(os.path.join(out, "e8_native.json"), "w"))
print("done", len(ids), f"{time.time()-t0:.0f}s", flush=True)
