#!/usr/bin/env python3
"""Regenerate the 'corpus' and 'sensor_wavelengths_nm' blocks of
experiments/results/corpus_and_carbon_provenance.json (Table 1 inventory).

Source of the per-collection chip counts: Clay-foundation/model,
docs/release-notes/data_sampling.md at commit f14e698f3c237cabf8d28dec669a362d66625381,
section "Dataset size" (the table captioned "used for Clay v1 training").
Bands and GSD come from experiments/suite/common/configs/metadata.yaml; the wavelengths
actually fed to the model in the benchmark scripts come from the constants in
experiments/run_standard_benchmarks.py (parsed with ast, not imported).
Step-share multipliers are derived by formula below, never typed in.

Usage (from the repo root, no GPU, no network if --data-sampling is given):
    python experiments/table1/build_table1.py [--data-sampling PATH] [--json PATH]
Default for --data-sampling is to download the pinned raw file from GitHub and check its SHA-256.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import sys
import urllib.request
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments" / "suite" / "common"))
from provenance import provenance  # noqa: E402

UPSTREAM_COMMIT = "f14e698f3c237cabf8d28dec669a362d66625381"
UPSTREAM_PATH = "docs/release-notes/data_sampling.md"
UPSTREAM_URL = f"https://raw.githubusercontent.com/Clay-foundation/model/{UPSTREAM_COMMIT}/{UPSTREAM_PATH}"
UPSTREAM_SHA256 = "506149ee5db2e0c9d81daf2120645e142255feb87fbe5390806898bf7c897e54"

# Display metadata (labels only; numbers come from the sources above).
COLLECTIONS = [  # (id, upstream row name, label, domain)
    ("linz", "LINZ", "LINZ Aerial (3b)", "RGB aerial orthoimagery"),
    ("naip", "NAIP", "USDA NAIP (4b)", "RGB + NIR aerial orthoimagery"),
    ("sentinel-2-l2a", "Sentinel-2-l2a", "Sentinel-2 L2A (10b)", "VSWIR (0.49-2.19 um)"),
    ("sentinel-1-rtc", "Sentinel-1-rtc", "Sentinel-1 SAR (2b)", "C-band SAR (gamma0 VV, VH in dB)"),
    ("landsat-c2l1", "Landsat-c2l1", "Landsat C2 L1 (6b)", "Top-of-atmosphere VSWIR"),
    ("landsat-c2l2-sr", "Landsat-c2l2-sr", "Landsat C2 L2 (6b)", "Surface-reflectance VSWIR"),
    ("modis", "MODIS", "MODIS MOD09A1 (7b)", "Global 8-day surface reflectance"),
]

# Historical values that were in the JSON before this script existed. NO source was found for any of
# them (searched: Clay-foundation org code search, model repo history, openclay, lgnd monorepo, the
# original experiment scratch). They are kept only as a record, flagged unsourced, and are NOT used.
SUPERSEDED_UNSOURCED = {
    "per_collection_chips": {
        "naip": 21017896, "linz": 3274626, "sentinel-2-l2a": 18697944, "sentinel-1-rtc": 16092150,
        "landsat-c2l1": 5831942, "landsat-c2l2-sr": 5806866, "modis": 1347940,
    },
    "parquet_index_rows_pre_filter": 72240719,
    "parquet_row_groups": 95945,
    "total_chips_post_filter": 72069364,
    "uncompressed_tb": 33.8,
    "note": ("Per-row values differ from the pinned upstream table in every row (deltas sum to exactly 0, so "
             "the total equals the upstream total). No file, parquet index, script or log containing these "
             "integers was found. Treat as unsourced; the old provenance_note claiming extraction from "
             "data_sampling.md was incorrect."),
}


def load_upstream(path: str | None) -> tuple[str, str]:
    if path:
        raw = Path(path).read_bytes()
        src = f"local file {Path(path).name}"
    else:
        raw = urllib.request.urlopen(UPSTREAM_URL, timeout=60).read()
        src = UPSTREAM_URL
    sha = hashlib.sha256(raw).hexdigest()
    if sha != UPSTREAM_SHA256:
        raise SystemExit(f"data_sampling.md SHA-256 mismatch: got {sha}, expected {UPSTREAM_SHA256}")
    return raw.decode(), src


def parse_table(text: str) -> tuple[dict[str, int], int]:
    lines = text.splitlines()
    start = next(i for i, l in enumerate(lines) if l.startswith("## Dataset size"))
    rows, first = {}, None
    for j in range(start, len(lines)):
        m = re.match(r"\|\s*([A-Za-z0-9-]+)\s*\|\s*([\d,]+)\s*\|", lines[j])
        if m:
            rows[m.group(1)] = int(m.group(2).replace(",", ""))
            first = first or j + 1
        elif rows and not lines[j].startswith("|"):
            break
    return rows, first


def literal_lists(py_path: Path) -> dict[str, list[float]]:
    out = {}
    for node in ast.parse(py_path.read_text()).body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            n = node.targets[0].id
            if n in ("S3_OLCI_WAVES", "S2_10B_WAVES", "L8_6B_WAVES", "NAIP_WAVES", "S1_2B_WAVES", "RGB_WAVES"):
                out[n] = ast.literal_eval(node.value)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-sampling", default=None, help="local copy of data_sampling.md@pinned commit")
    ap.add_argument("--json", default=str(ROOT / "experiments/results/corpus_and_carbon_provenance.json"))
    ap.add_argument("--metadata", default=str(ROOT / "experiments/suite/common/configs/metadata.yaml"))
    ap.add_argument("--bench-script", default=str(ROOT / "experiments/run_standard_benchmarks.py"))
    a = ap.parse_args()

    text, src = load_upstream(a.data_sampling)
    counts, first_line = parse_table(text)
    meta = yaml.safe_load(Path(a.metadata).read_text())
    total = sum(counts[u] for _, u, _, _ in COLLECTIONS)
    assert set(counts) == {u for _, u, _, _ in COLLECTIONS}, counts
    n_coll = len(COLLECTIONS)
    naip = counts["NAIP"]

    cols, total_band_values = [], 0
    for cid, up, label, domain in COLLECTIONS:
        n = counts[up]
        gsd = float(meta[cid]["gsd"])
        bands = len(meta[cid]["band_order"])
        total_band_values += n * bands * 256 * 256
        cols.append({
            "id": cid, "label": label, "domain": domain, "bands": bands, "gsd_m": gsd,
            "token_m": 8.0 * gsd,
            "chips": n, "chips_millions": n / 1e6,
            "chip_share": n / total,
            "step_share": 1.0 / n_coll,
            # exposure per chip relative to a uniform draw over all chips: (1/7) / (n/N)
            "oversampling_vs_uniform_chip_share": (1.0 / n_coll) / (n / total),
            # the multiplier printed in DRAFT.md Table 1: NAIP chips / this collection's chips
            "oversampling_vs_naip": naip / n,
        })

    corpus = {
        "source": {
            "repo": "Clay-foundation/model", "commit": UPSTREAM_COMMIT, "path": UPSTREAM_PATH,
            "table_first_line": first_line, "fetched_from": src, "sha256": UPSTREAM_SHA256,
            "what_the_table_is": ("Upstream table is captioned 'distribution of imagery chips used for Clay v1 "
                                  "training'. The paper applies it to Clay v1.5; the same counts are present in "
                                  "every upstream revision of the doc since 8001245. No separate v1.5 inventory "
                                  "was found (see unsourced_previous_values)."),
        },
        "chip_px": [256, 256],
        "patch_stride_px": 8,
        "total_chips": total,
        "total_chips_millions": total / 1e6,
        "mgrs_tiles": 2728,
        "mgrs_tiles_source": "upstream data_sampling.md ('a sample of 2728 MGRS tiles total'); 3225 draws before dedup",
        "train_val_split": [0.8, 0.2],
        "equalized_step_share_per_collection": 1.0 / n_coll,
        "step_share_definition": ("Share of optimiser steps (one batch per platform, cycling), NOT share of chips. "
                                  "Multipliers: oversampling_vs_uniform_chip_share = (1/7)/(chips_i/N); "
                                  "oversampling_vs_naip = chips_NAIP/chips_i (the values printed in DRAFT.md, "
                                  "normalised so NAIP = 1.0)."),
        "reported_dataset_size_tb": 33.8,
        "reported_dataset_size_basis": ("Upstream data_sampling.md line 190: 'a dataset with a size of 33.8 TB of "
                                        "imagery, with about 70 million chips created'. Whether this is source scenes "
                                        "or packed chips, and compressed or not, is not stated upstream. The word "
                                        "'uncompressed' in the paper is NOT supported by the source."),
        "size_plausibility_check": {
            "total_band_values": total_band_values,
            "tb_if_1_byte_per_value": total_band_values / 1e12,
            "tb_if_2_bytes_per_value": total_band_values * 2 / 1e12,
            "tb_if_4_bytes_per_value": total_band_values * 4 / 1e12,
            "implied_bytes_per_value_at_33.8TB": 33.8e12 / total_band_values,
            "note": ("Arithmetic only: chips x bands x 256 x 256. 33.8 TB corresponds to about 1.3 bytes per "
                     "band value, i.e. neither raw float32 nor raw uint16 chips; it cannot be the uncompressed "
                     "size of the 72.07M packed chips under any single dtype."),
        },
        "collections": cols,
        "unsourced_previous_values": SUPERSEDED_UNSOURCED,
    }

    lists = literal_lists(Path(a.bench_script))
    s2 = meta["sentinel-2-l2a"]["bands"]["wavelength"]
    l8 = meta["landsat-c2l1"]["bands"]["wavelength"]
    to_nm = lambda xs: [round(float(x) * 1000.0, 3) for x in xs]  # noqa: E731
    wl = {
        "source": "experiments/run_standard_benchmarks.py constants (what the benchmarks fed the model) cross-checked against metadata.yaml",
        "s2_10b": to_nm(lists["S2_10B_WAVES"]),
        "l8_6b_order_red_green_blue_nir_swir16_swir22": to_nm(lists["L8_6B_WAVES"]),
        "naip_4b": to_nm(lists["NAIP_WAVES"]),
        "s1_2b_pseudo_wavelength_nm": to_nm(lists["S1_2B_WAVES"]),
        "s3_olci_21b": to_nm(lists["S3_OLCI_WAVES"]),
        "metadata_yaml_s2_10b": to_nm([s2[k] for k in meta["sentinel-2-l2a"]["band_order"]]),
        "metadata_yaml_l8_6b": to_nm([l8[k] for k in meta["landsat-c2l1"]["band_order"]]),
        "bench_constants_match_metadata_yaml": {
            "s2_10b": to_nm(lists["S2_10B_WAVES"]) == to_nm([s2[k] for k in meta["sentinel-2-l2a"]["band_order"]]),
            "l8_6b": to_nm(lists["L8_6B_WAVES"]) == to_nm([l8[k] for k in meta["landsat-c2l1"]["band_order"]]),
        },
        "previous_unsourced_values_nm": {
            "s2_10b": [493.0, 560.0, 665.0, 704.0, 740.0, 783.0, 833.0, 865.0, 1614.0, 2202.0],
            "l8_6b": [482.0, 561.0, 655.0, 865.0, 1609.0, 2201.0],
            "note": "Nominal sensor band centres (e.g. S2 833 nm = B8, Landsat-8 482/561/655), not the values given to Clay.",
        },
    }

    p = Path(a.json)
    d = json.loads(p.read_text()) if p.exists() else {}
    d["provenance_note"] = (
        "corpus and sensor_wavelengths_nm are regenerated by experiments/table1/build_table1.py from "
        f"Clay-foundation/model {UPSTREAM_PATH}@{UPSTREAM_COMMIT[:7]}, metadata.yaml and "
        "run_standard_benchmarks.py. carbon_and_compute_2024 holds hand-transcribed AWS Customer Carbon "
        "Footprint Tool values (not independently verifiable here) and is checked by "
        "experiments/table1/check_carbon.py.")
    d["corpus"] = corpus
    d["sensor_wavelengths_nm"] = wl
    d.setdefault("provenance", {})["build_table1"] = provenance(
        __file__, {"data_sampling": UPSTREAM_URL if not a.data_sampling else Path(a.data_sampling).name}, ROOT)
    p.write_text(json.dumps(d, indent=2) + "\n")
    print(f"total={total}  wrote {p}")
    for c in cols:
        print(f"{c['id']:16s} {c['chips']:>10,d} {c['chips_millions']:6.2f}M  chip_share={c['chip_share']*100:5.2f}%  "
              f"x_naip={c['oversampling_vs_naip']:.2f}  x_uniform={c['oversampling_vs_uniform_chip_share']:.2f}")


if __name__ == "__main__":
    main()
