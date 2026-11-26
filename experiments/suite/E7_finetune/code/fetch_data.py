"""Download Sen1Floods11 v1.1 hand-labelled S2Hand + LabelHand chips, split CSVs and per-chip STAC dates."""
import csv, io, json, os, sys, urllib.request, concurrent.futures as cf
from pathlib import Path
import numpy as np, rasterio
B = "https://storage.googleapis.com/sen1floods11/v1.1"
HERE = Path(__file__).resolve().parents[4]
OUT = Path(os.environ.get("CLAY_E7_DATA", str(HERE / "data/E7_finetune"))); OUT.mkdir(parents=True, exist_ok=True)
(OUT / "S2Hand").mkdir(exist_ok=True); (OUT / "LabelHand").mkdir(exist_ok=True)
get = lambda u: urllib.request.urlopen(u, timeout=120).read()
splits = {}
for s, f in [("train", "flood_train_data.csv"), ("valid", "flood_valid_data.csv"), ("test", "flood_test_data.csv"), ("bolivia", "flood_bolivia_data.csv")]:
    txt = get(f"{B}/splits/flood_handlabeled/{f}").decode()
    (OUT / f).write_text(txt)
    splits[s] = [r[0].replace("_S1Hand.tif", "") for r in csv.reader(io.StringIO(txt)) if r]
    print(s, len(splits[s]))
chips = sorted({c for v in splits.values() for c in v})
def dl(c):
    for d, suf in [("S2Hand", "S2Hand"), ("LabelHand", "LabelHand")]:
        p = OUT / d / f"{c}_{suf}.tif"
        if not p.exists():
            p.write_bytes(get(f"{B}/data/flood_events/HandLabeled/{d}/{c}_{suf}.tif"))
    meta = {}
    for u in [f"{B}/catalog/sen1floods11_hand_labeled_source/{c}/{c}.json",
              f"{B}/catalog/sen1floods11_hand_labeled_label/{c}/{c}.json"]:
        try:
            j = json.loads(get(u)); meta = {"url": u, "datetime": j.get("properties", {}).get("datetime"), "props": j.get("properties")}; break
        except Exception as e:
            meta = {"err": str(e)}
    return c, meta
with cf.ThreadPoolExecutor(16) as ex:
    metas = dict(ex.map(dl, chips))
rows = []
for c in chips:
    with rasterio.open(OUT / "S2Hand" / f"{c}_S2Hand.tif") as r:
        b = r.bounds; crs = r.crs; dt = r.dtypes[0]; cnt = r.count; shp = r.shape
        from rasterio.warp import transform_bounds
        w, s_, e, n = transform_bounds(crs, "EPSG:4326", *b)
    rows.append({"chip": c, "split": [k for k, v in splits.items() if c in v], "lat": (s_ + n) / 2, "lon": (w + e) / 2,
                 "dtype": dt, "bands": cnt, "shape": shp, "crs": str(crs), "stac": metas[c]})
json.dump(rows, open(OUT / "chips.json", "w"), indent=1, default=str)
print(len(rows), rows[0])
x = rasterio.open(OUT / "S2Hand" / f"{chips[0]}_S2Hand.tif").read()
y = rasterio.open(OUT / "LabelHand" / f"{chips[0]}_LabelHand.tif").read()
print("S2", x.dtype, x.shape, np.nanpercentile(x, [1, 50, 99], axis=(1, 2)).round(1).tolist(), "nan", int(np.isnan(x).sum()) if x.dtype.kind == "f" else 0)
print("label", np.unique(y, return_counts=True))
os.system(f"du -sh {OUT}")
