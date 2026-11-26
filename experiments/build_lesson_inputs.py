#!/usr/bin/env python3
"""Build the inputs that `run_clay_v15_experiments.py` and `run_probe_and_lsvm_benchmark.py` read.

Two subcommands (both write under --out, default `data/`):

  s2-nlcd   The 320 Sentinel-2 L2A chips (128 x 128 px at 10 m, 12 bands, raw DN) and their NLCD labels.
            Reads results/s2_nlcd_320_manifest.csv (chip id, centre, STAC item id, UTM origin, labels,
            sha256 of the expected pixel array), downloads the pixels from Microsoft Planetary Computer
            (public, anonymous signing), and checks every array against the manifest hash.
            Writes <out>/s2_nlcd_320/data/{chips.parquet,pixels/*.npy} and <out>/s2_nlcd_320/results/labels_nlcd.parquet.
            Pass --pixels-from DIR to copy existing arrays instead of downloading (hashes are still checked).

  eurosat   Clay v1.5 class-token features for the 25,648 eligible EuroSAT-MS chips of the spatially blocked pilot split
            (results/eurosat_pilot_manifest.csv, split in {train,val,test}), plus the image list used by the
            compositional-mosaic test. Needs EuroSAT_MS.zip (Zenodo 7711810) and the Clay checkpoint.
            Writes <out>/eurosat/features/{rows.csv,cls.npy} and <out>/eurosat/images.json.
            --latlon-order documented|training chooses the 4 location slots ([sin lat, cos lat, sin lon, cos lon] or
            [sin lon, cos lon, sin lat, cos lat]); time is zeros, GSD 10 m, raw DN with Clay's S2 mean/std.

Environment: CLAY_V15_CKPT (checkpoint), EUROSAT_ZIP (zip path), PYTHONPATH not needed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
WORK_ROOT = HERE.parent
RESULTS = HERE / "results"
S2_BANDS_12 = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12"]
CHIP_PX = 128
CHIP_M = 1280.0
STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
EUROSAT_SEL = [1, 2, 3, 4, 5, 6, 7, 12, 10, 11]  # zero-based indices into the 13-band EuroSAT tif -> Clay's 10 S2 bands
EUROSAT_ZIP_MD5 = "091174add3c8e680a49244acf185b9f0"


def sha256_array(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


# ----------------------------------------------------------------------------------------------
# s2-nlcd
# ----------------------------------------------------------------------------------------------
def _read_chip(item, row) -> np.ndarray:
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.warp import transform_bounds

    bounds_utm = (row.utm_x, row.utm_y - CHIP_M, row.utm_x + CHIP_M, row.utm_y)
    out = np.zeros((len(S2_BANDS_12), CHIP_PX, CHIP_PX), dtype=np.uint16)
    for bi, band in enumerate(S2_BANDS_12):
        with rasterio.open(item.assets[band].href) as src:
            b = bounds_utm if src.crs.to_epsg() == int(row.utm_epsg) else transform_bounds(
                f"EPSG:{int(row.utm_epsg)}", src.crs, *bounds_utm)
            win = rasterio.windows.from_bounds(*b, transform=src.transform)
            arr = src.read(1, window=win, out_shape=(CHIP_PX, CHIP_PX), resampling=Resampling.bilinear,
                           boundless=True, fill_value=0)
        out[bi] = arr.astype(np.uint16)
    return out


def build_s2_nlcd(args) -> None:
    os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
    os.environ.setdefault("GDAL_HTTP_MAX_RETRY", "5")
    os.environ.setdefault("GDAL_HTTP_RETRY_DELAY", "2")
    man = pd.read_csv(RESULTS / "s2_nlcd_320_manifest.csv", float_precision="round_trip")
    root = Path(args.out) / "s2_nlcd_320"
    (root / "data" / "pixels").mkdir(parents=True, exist_ok=True)
    (root / "results").mkdir(parents=True, exist_ok=True)
    man[["chip_id", "lat", "lon", "utm_epsg", "utm_x", "utm_y", "datetime", "s2_item_id", "cloud_pct", "stratum"]
        ].to_parquet(root / "data" / "chips.parquet", index=False)
    man[["chip_id", "y_impervious", "y_canopy", "src_impervious", "src_canopy"]
        ].to_parquet(root / "results" / "labels_nlcd.parquet", index=False)
    todo = man[~man.chip_id.map(lambda c: (root / "data" / "pixels" / f"{c}.npy").exists())]
    report = {"n_chips": int(len(man)), "n_to_fetch": int(len(todo)), "hash_mismatch": [], "failed": []}
    if args.pixels_from:
        for r in todo.itertuples():
            shutil.copy(Path(args.pixels_from) / f"{r.chip_id}.npy", root / "data" / "pixels" / f"{r.chip_id}.npy")
    elif len(todo):
        import planetary_computer
        import pystac_client

        cat = pystac_client.Client.open(STAC_URL, modifier=planetary_computer.sign_inplace)
        ids = todo.s2_item_id.unique().tolist()
        items = {}
        for i in range(0, len(ids), 50):
            for it in cat.search(collections=["sentinel-2-l2a"], ids=ids[i:i + 50]).items():
                items[it.id] = it
        print(f"signed {len(items)}/{len(ids)} STAC items", flush=True)

        def work(r):
            if r.s2_item_id not in items:
                raise RuntimeError(f"STAC item {r.s2_item_id} not found")
            return r.chip_id, _read_chip(items[r.s2_item_id], r)

        with ThreadPoolExecutor(args.workers) as ex:
            futs = [ex.submit(work, r) for r in todo.itertuples()]
            for k, f in enumerate(as_completed(futs)):
                try:
                    cid, arr = f.result()
                    np.save(root / "data" / "pixels" / f"{cid}.npy", arr)
                except Exception as e:  # noqa: BLE001
                    report["failed"].append(str(e)[:200])
                if k % 40 == 0:
                    print(f"  {k + 1}/{len(futs)}", flush=True)
    for r in man.itertuples():
        p = root / "data" / "pixels" / f"{r.chip_id}.npy"
        if not p.exists():
            continue
        if sha256_array(np.load(p)) != r.pixel_sha256:
            report["hash_mismatch"].append(r.chip_id)
    report["n_hash_ok"] = int(len(man) - len(report["hash_mismatch"]) - len(report["failed"]))
    (root / "build_report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps({k: (v if not isinstance(v, list) else len(v)) for k, v in report.items()}))
    if report["hash_mismatch"]:
        print("WARNING: pixel arrays differ from the manifest hashes (Planetary Computer reprocessing or resampling "
              "differences); results will not be bit-identical to the committed JSON.", file=sys.stderr)


# ----------------------------------------------------------------------------------------------
# eurosat
# ----------------------------------------------------------------------------------------------
def build_eurosat(args) -> None:
    import hashlib as _h
    from rasterio.io import MemoryFile

    sys.path.insert(0, str(HERE / "suite" / "common"))
    from clayrt import ClayRuntime  # noqa: PLC0415

    zpath = Path(args.zip)
    if args.check_md5:
        h = _h.md5()
        with zpath.open("rb") as f:
            for c in iter(lambda: f.read(1 << 24), b""):
                h.update(c)
        if h.hexdigest() != EUROSAT_ZIP_MD5:
            raise SystemExit(f"EuroSAT_MS.zip md5 {h.hexdigest()} != expected {EUROSAT_ZIP_MD5}")
    man = pd.read_csv(RESULTS / "eurosat_pilot_manifest.csv")
    man = man[man.split != "excluded"].reset_index(drop=True)
    if args.limit:  # smoke test only (experiments/repro_smoke.sh): a seeded random subset of the manifest rows
        man = man.sample(n=min(args.limit, len(man)), random_state=0).sort_index().reset_index(drop=True)
    out = Path(args.out) / "eurosat"
    (out / "features").mkdir(parents=True, exist_ok=True)
    z = zipfile.ZipFile(zpath)
    rt = ClayRuntime(ckpt=args.ckpt, device="cuda" if args.device == "cuda" else "cpu")
    cls, B = [], 256
    for i in range(0, len(man), B):
        sub = man.iloc[i:i + B]
        X = []
        for k in sub["id"]:
            with MemoryFile(z.read(k)) as mf, mf.open() as src:
                X.append(src.read().astype(np.float32)[EUROSAT_SEL])
        ll = list(zip(sub["lat"].astype(float), sub["lon"].astype(float)))
        e = rt.embed(np.stack(X), platform="sentinel-2-l2a", latlon=ll, date=None, batch_size=64,
                     latlon_order=args.latlon_order)
        cls.append(e["cls"])
        if (i // B) % 10 == 0:
            print(f"  {i + len(sub)}/{len(man)}", flush=True)
    np.save(out / "features" / "cls.npy", np.concatenate(cls).astype(np.float32))
    man.rename(columns={"class_name": "class"}).assign(block_id=man.block_id).to_csv(out / "features" / "rows.csv", index=False)
    imgs = [{"name": r["id"], "class": r["class_name"], "split": r["split"]} for _, r in man.iterrows()
            if r["split"] in ("train", "test")]
    (out / "images.json").write_text(json.dumps(imgs))
    (out / "features" / "build.json").write_text(json.dumps(
        {"latlon_order": args.latlon_order, "n": int(len(man)), "clay_evidence": rt.evidence,
         "zip": str(zpath.name), "zip_md5_expected": EUROSAT_ZIP_MD5}, indent=1))
    print("wrote", out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("s2-nlcd")
    a.add_argument("--out", default=str(WORK_ROOT / "data"))
    a.add_argument("--workers", type=int, default=8)
    a.add_argument("--pixels-from", default=None, help="directory with existing <chip_id>.npy arrays to copy instead of downloading")
    b = sub.add_parser("eurosat")
    b.add_argument("--out", default=str(WORK_ROOT / "data"))
    b.add_argument("--zip", default=os.environ.get("EUROSAT_ZIP", str(WORK_ROOT / "data" / "EuroSAT_MS.zip")))
    b.add_argument("--ckpt", default=os.environ.get("CLAY_V15_CKPT", str(WORK_ROOT / "checkpoints/v1.5/clay-v1.5.ckpt")))
    b.add_argument("--latlon-order", choices=["documented", "training"], default="documented")
    b.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    b.add_argument("--check-md5", action="store_true")
    b.add_argument("--limit", type=int, default=0, help="smoke test only: embed a seeded random subset of N chips")
    args = ap.parse_args()
    {"s2-nlcd": build_s2_nlcd, "eurosat": build_eurosat}[args.cmd](args)


if __name__ == "__main__":
    main()
