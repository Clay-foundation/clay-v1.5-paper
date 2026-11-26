"""E1(b): EuroSAT pilot — Clay training-tile overlap + DINOv2-L / DOFA-L embeddings for the frozen pilot IDs.

Inputs (workdir): manifest.csv (frozen pilot manifest), EuroSAT_MS.zip (Zenodo 7711810, MD5-checked),
mgrs_sample_v02.fgb (Clay training MGRS tiles), common/ (clayrt).
Outputs: overlap.csv (per eligible id: in_tile_centroid, in_tile_footprint), overlap.json,
embeddings_dino.npz (ids, cls, mean), embeddings_dofa.npz (ids, cls, mean=cls), extract.json.
Band selection follows the frozen pilot: zero-based [1,2,3,4,5,6,7,12,10,11] -> B02..B08,B8A,B11,B12.
"""
import os
import csv, json, sys, time, zipfile
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "common"))

SEL = [1, 2, 3, 4, 5, 6, 7, 12, 10, 11]


def main():
    rows = [r for r in csv.DictReader(open("manifest.csv", newline="")) if r["split"] != "excluded"]
    ids = [r["id"] for r in rows]
    print(len(rows), "eligible", flush=True)
    # ---- overlap with Clay training MGRS tiles
    import geopandas as gpd
    from shapely.geometry import Point, box
    from pyproj import Transformer
    tiles = gpd.read_file("mgrs_sample_v02.fgb").to_crs(4326)
    tr = Transformer.from_crs(3035, 4326, always_xy=True)
    pts = gpd.GeoDataFrame({"id": ids, "split": [r["split"] for r in rows]},
                           geometry=[Point(float(r["lon"]), float(r["lat"])) for r in rows], crs=4326)
    fp = []
    for r in rows:
        xs, ys = tr.transform([float(r["bounds3035_xmin"]), float(r["bounds3035_xmax"])],
                              [float(r["bounds3035_ymin"]), float(r["bounds3035_ymax"])])
        fp.append(box(min(xs), min(ys), max(xs), max(ys)))
    fps = gpd.GeoDataFrame({"id": ids}, geometry=fp, crs=4326)
    j1 = gpd.sjoin(pts, tiles[["geometry"]], predicate="within", how="left")
    in_c = set(j1.loc[j1["index_right"].notna(), "id"])
    j2 = gpd.sjoin(fps, tiles[["geometry"]], predicate="intersects", how="left")
    in_f = set(j2.loc[j2["index_right"].notna(), "id"])
    with open("overlap.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["id", "split", "in_tile_centroid", "in_tile_footprint"])
        for r in rows:
            w.writerow([r["id"], r["split"], int(r["id"] in in_c), int(r["id"] in in_f)])
    summ = {"n_tiles": len(tiles), "tile_columns": [c for c in tiles.columns if c != "geometry"]}
    for sp in ["train", "val", "test"]:
        s = [r["id"] for r in rows if r["split"] == sp]
        summ[sp] = {"n": len(s), "centroid_in_tile": sum(i in in_c for i in s), "footprint_intersects_tile": sum(i in in_f for i in s)}
    json.dump(summ, open("overlap.json", "w"), indent=1); print(json.dumps(summ), flush=True)

    # ---- comparator embeddings
    import torch, rasterio
    from rasterio.io import MemoryFile
    from clayrt import DinoV2Runtime, to_display_rgb, platform_spec
    from torchgeo.models import dofa_large_patch16_224, DOFALarge16_Weights
    spec = platform_spec("sentinel-2-l2a")
    dino = DinoV2Runtime(size=224)
    import timm  # timm builds this model at 518 px; rebuild at 224 px (pos-embed interpolated at load)
    dino.m = timm.create_model(dino.name, pretrained=True, num_classes=0, img_size=224).eval().to("cuda")
    dofa = dofa_large_patch16_224(weights=DOFALarge16_Weights.DOFA_MAE).eval().cuda()
    mean = torch.tensor(spec["means"]).view(1, -1, 1, 1); std = torch.tensor(spec["stds"]).view(1, -1, 1, 1)
    z = zipfile.ZipFile("EuroSAT_MS.zip")
    names = {n.split("/", 1)[-1] if not n.startswith("EuroSAT_MS/") else n: n for n in z.namelist()}
    out = {"dino_cls": [], "dino_mean": [], "dofa": []}
    t0 = time.time(); B = 256
    for i in range(0, len(ids), B):
        X = []
        for k in ids[i:i + B]:
            with MemoryFile(z.read(names[k])) as mf, mf.open() as src:
                a = src.read().astype(np.float32)
            assert a.shape == (13, 64, 64), (k, a.shape)
            X.append(a[SEL])
        X = np.stack(X)
        d = dino.embed(to_display_rgb(X, "sentinel-2-l2a"))
        out["dino_cls"].append(d["cls"]); out["dino_mean"].append(d["mean"])
        with torch.inference_mode():
            x = (torch.from_numpy(X) - mean) / std
            x = torch.nn.functional.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
            for j in range(0, len(x), 64):
                out["dofa"].append(dofa.forward_features(x[j:j + 64].cuda(), spec["waves"]).float().cpu().numpy())
        if (i // B) % 20 == 0:
            print(i, round(time.time() - t0, 1), flush=True)
    o = {k: np.concatenate(v).astype(np.float32) for k, v in out.items()}
    assert all(np.isfinite(v).all() for v in o.values())
    ida = np.asarray(ids)
    np.savez("embeddings_dino.npz", ids=ida, cls=o["dino_cls"], mean=o["dino_mean"])
    np.savez("embeddings_dofa.npz", ids=ida, cls=o["dofa"], mean=o["dofa"])
    json.dump({"n": len(ids), "dino": dino.name, "dino_dim": o["dino_cls"].shape[1], "dofa_dim": o["dofa"].shape[1],
               "dofa_weights": DOFALarge16_Weights.DOFA_MAE.url, "wall_s": round(time.time() - t0, 1),
               "gpu_max_mem_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2)}, open("extract.json", "w"), indent=1)


if __name__ == "__main__":
    main()
