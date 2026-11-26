"""E1(b) amendment 3: re-embed the frozen EuroSAT pilot IDs with Clay v1.5 in the TRAINING lat/lon order.

Pilot input contract unchanged otherwise: native 64x64, bands [1,2,3,4,5,6,7,12,10,11], raw DN with Clay S2 mean/std,
nominal GSD 10 m, time = zeros, lat/lon = manifest centroid. Also embeds in the documented order as a
reproduction check against the pilot's stored embeddings (same order the pilot used).
Outputs: embeddings_clay_training.npz, embeddings_clay_documented.npz (ids, cls, mean), clay_extract.json.
"""
import os
import csv, json, sys, time, zipfile
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "common"))
SEL = [1, 2, 3, 4, 5, 6, 7, 12, 10, 11]


def main():
    from rasterio.io import MemoryFile
    from clayrt import ClayRuntime
    rows = [r for r in csv.DictReader(open("manifest.csv", newline="")) if r["split"] != "excluded"]
    ids = [r["id"] for r in rows]; ll = [(float(r["lat"]), float(r["lon"])) for r in rows]
    z = zipfile.ZipFile("EuroSAT_MS.zip")
    names = {n: n for n in z.namelist()}
    rt = ClayRuntime(device="cuda")
    out = {o: {"cls": [], "mean": []} for o in ("training", "documented")}
    t0 = time.time(); B = 256
    for i in range(0, len(ids), B):
        X = []
        for k in ids[i:i + B]:
            with MemoryFile(z.read(names[k])) as mf, mf.open() as src:
                X.append(src.read().astype(np.float32)[SEL])
        X = np.stack(X)
        for o in out:
            e = rt.embed(X, platform="sentinel-2-l2a", latlon=ll[i:i + B], date=None, batch_size=64, latlon_order=o)
            out[o]["cls"].append(e["cls"]); out[o]["mean"].append(e["mean"])
    for o, d in out.items():
        np.savez(f"embeddings_clay_{o}.npz", ids=np.asarray(ids), cls=np.concatenate(d["cls"]), mean=np.concatenate(d["mean"]))
    json.dump({"n": len(ids), "evidence": rt.evidence, "wall_s": round(time.time() - t0, 1)}, open("clay_extract.json", "w"), indent=1)


if __name__ == "__main__":
    main()
