# E3 — Cross-sensor transfer in Clay v1.5 token space (frozen 2026-11-05, before any score)

Follows `../PROTOCOL-COMMON.md`. Question: does a head trained on Clay embeddings of sensor A work on sensor B at the same place and season? Community reports say an S2-trained head "returned nothing" on NAIP.

## Data (CONUS, co-located, same season)
- **Tiles.** 1,500 points drawn uniformly by area over the CONUS bounding box (lon −124.7…−67.0, lat 25.0…49.4; numpy seed 3). A point is kept if a Planetary Computer `naip` item from 2020–2022 covers it; we take the item whose year is closest to 2021 (ties: later). Points without NAIP (ocean, Canada, Mexico) are dropped. Working extent = NAIP item footprint in its native UTM CRS, shrunk by 100 m, snapped to a 320 m grid, capped at 6.4 × 6.4 km around its centre.
- **Cells.** 320 m × 320 m cells on that grid. Label = ESA WorldCover 2021 (`esa-worldcover`, v200) majority class in the cell at 10 m. Keep a cell if purity ≥ 0.60 and the class is in {10 tree, 20 shrub, 30 grass, 40 crop, 50 built, 60 bare, 80 water, 90 wetland}.
- **Sentinel-2 L2A** (`sentinel-2-l2a`): 10 bands in Clay order, raw PC values with no harmonisation, 32 × 32 px at 10 m on the cell. Acquisition: eo:cloud_cover < 30 and within ±60 days of the NAIP date; candidate days ranked by |Δt| + 2·cloud%. Up to 3 days are tried; a cell takes the first day where all pixels are non-zero and ≥ 95 % of SCL ∈ {4,5,6,7}. We record `s2:processing_baseline` and the date per chip.
- **NAIP** (`naip`): R, G, B, NIR (uint8) over the same 320 m footprint, read natively with GDAL average resampling to 256 × 256 (1.25 m). Embedded with gsd = 1.25. A cell is valid if < 1 % of its pixels are all-zero.
- **Sentinel-1 RTC** (`sentinel-1-rtc`): VV, VH on the same 32 × 32 grid at 10 m. Linear γ0 is converted to dB as in Clay's datamodule: values ≤ 0 → 1e-10, then 10·log10. Day closest to the NAIP date within ±60 days (up to 3 tried). A cell is valid if every pixel is finite and > 0 before the conversion.
- **Landsat C2 L2** (`landsat-c2-l2`, Landsat 8/9, SR bands red, green, blue, nir08, swir16, swir22 as raw DN): 480 m footprint centred on the cell. Loaded at 10 m nearest-neighbour, then 3 × 3 block-averaged to 16 × 16 px at 30 m (a declared approximation of the native 30 m grid, shift ≤ 10 m). Same date and cloud rules as S2 (cloud < 30, ±60 days). A cell is valid if it has no zero DN and ≥ 95 % of pixels have QA_PIXEL bits 1–4 clear.
- **Per-tile selection** (labels only; no embeddings or scores are used). Candidate cells have a valid S2 chip. Up to 8 cells are chosen per tile by round-robin over present classes in the fixed priority order bare, wetland, water, built, shrub, grass, crop, tree, picking uniformly within class (seed = tile index). Then NAIP is read. S1 and Landsat are attached where valid. Total is capped at 10,000 locations.
- Sensors missing for a location drop it only from the pairs that need that sensor. If the 1,500 points yield fewer than 5,000 S2+NAIP locations, we report the shortfall and do not add points after scoring.

## Embeddings
- Clay v1.5 (`clayrt.ClayRuntime`), CLS (primary) and mean patch token (secondary).
- **Metadata ON** (default): chip-centre lat/lon and the sensor's own acquisition datetime.
- **Metadata OFF**: zeros.
- GSD: S2 10, NAIP 1.25, S1 10, Landsat 30.
- DINOv2 ViT-L/14-reg4 at 224 px on the display RGB (`to_display_rgb`, 2–98 % per-chip stretch) for S2, NAIP and Landsat. Not run on S1.
- Raw controls:
  - *raw-pixels*: shared R, G, B, NIR (NAIP average-pooled to 32 × 32), each band z-scored per sensor over all chips, flattened;
  - *raw-stats*: per-chip mean and std of R, G, B, NIR, z-scored per sensor.

## Split, heads, budgets
- `evalkit.hblock` at 100 km, then `evalkit.block_split` (70/15/15). Train = train ∪ val (no tuning is done); test = test blocks.
- Classes: those with ≥ 100 train and ≥ 20 test locations in the S2 ∩ NAIP set. The same class list is used for every pair. Locations of other classes are dropped.
- Head: `evalkit.clf_head` (StandardScaler + LogisticRegression C = 1). Budgets k ∈ {10, 50} labels/class via `evalkit.label_budget`, 5 seeds.

## Arms, per ordered pair A→B
Pairs: S2→NAIP, NAIP→S2, S2→S1, S1→S2 (required); S2→Landsat, Landsat→S2 (secondary). Each pair uses the locations that have both A and B.
1. **native-full**: head on all B train labels, scored on B test. **native-k**: head on k labels/class of B.
2. **zero-shot transfer**: head on all A train labels, applied unchanged to B test (A's scaler included). *Secondary (2b)*: the same LR weights after re-standardising B with its own unlabeled train mean/std.
3. **transfer+k**: one head refit on the pooled A train labels ∪ k labels/class of B train, with equal sample weights.
4. **Procrustes-n**: n ∈ {100, 1000} unlabeled paired train locations (`evalkit.n_budget`, 5 seeds). Centre each sensor on the means of those n chips; R = `scipy.linalg.orthogonal_procrustes`(B_c, A_c); map B → (B − μ_B)R + μ_A; apply the full A head.
- Also a **majority-class** reference (train-majority prediction).
- Features: Clay CLS meta-ON (primary); Clay mean meta-ON, Clay CLS meta-OFF, and DINOv2 (S2↔NAIP, S2↔Landsat only) as secondary.
- Metrics: macro-F1 (primary), accuracy. 1,000 block-bootstrap resamples over test blocks (seed 1729); paired contrasts share resamples.

## Retrieval of paired chips
- Query each S2 chip against the gallery of all NAIP chips, using all S2 ∩ NAIP locations (no training).
- Record the rank of the true pair by cosine similarity: recall@1, recall@10, median rank. Chance R@1 = 1/N.
- Primary similarity: cosine on raw features. Secondary: cosine after per-sensor mean-centering (an unsupervised alignment).
- Methods: Clay CLS meta-ON, Clay CLS meta-OFF, Clay mean (ON/OFF), DINOv2 RGB, raw-pixels, raw-stats.
- Secondary directions: NAIP→S2; S2→S1 for Clay ON/OFF and raw-stats (VV/VH vs S2 has no shared bands; raw not applicable to S1 → reported as n/a).
- CIs: 1,000 bootstrap resamples of query blocks with the gallery fixed (seed 1729).
- **Metadata test.** Under metadata ON, both chips carry the same lat/lon, and dates within 60 days. ON ≫ OFF retrieval would mean location/time encoding drives the apparent alignment.

## Pre-registered primary endpoint
- Δ = macro-F1(zero-shot S2→NAIP) − macro-F1(native-full NAIP), Clay CLS meta-ON, with a paired block-bootstrap CI.
- We also report transfer efficiency (zero-shot − majority) / (native − majority).
- Reading: a shared token space predicts Δ near 0 (we pre-declare |Δ| < 0.10 as "transfer works"). Zero-shot macro-F1 at or near the majority baseline supports the community report.
- All other numbers are secondary/descriptive.

## Outputs
`results.json`, `RESULTS.md`, `fig_e3_cross_sensor.{png,pdf}`, `manifest.csv` (location id, tile, lon/lat, CRS cell bounds, label, purity, per-sensor item ids and dates, S2 processing baseline, split, block) and `code/`.

## Amendment — lat/lon order (2026-11-05, written before the amended scores)
- **What changed.** Clay v1.0/v1.5 were trained with longitude in the first two lat/lon slots: [sin lon, cos lon, sin lat, cos lat]. This comes from a stacchip unpacking swap; see the PROTOCOL-COMMON amendment. Our original metadata-ON embeddings used the documented order [sin lat, cos lat, sin lon, cos lon], which is out of distribution for the model.
- **Relabelling.** Those original numbers stay unchanged in `results.json`, relabelled **"documented order (out-of-distribution metadata)"**.
- **Re-run.** All Clay metadata-ON embeddings (CLS and mean; S2, NAIP, S1, Landsat) are re-run in the **training order** with the updated `common/clayrt.py`. The cached chips are reused: no new sampling, same split, heads, budgets, seeds and bootstrap.
- **Re-scored.** Every classification arm for the six pairs, the retrieval (S2→NAIP, NAIP→S2, S2→S1; raw and centred) and the paired-cosine descriptives are re-scored with the frozen `evaluate.py` functions.
- **Primary endpoint** is unchanged (S2→NAIP zero-shot minus native, CLS). It is reported in both orders, and the training order becomes the canonical metadata-ON value.
- **Metadata-OFF arms** are unaffected and not re-run.
- **Sanity check.** Re-embedding 64 chips per sensor in the documented order must reproduce the stored embeddings (minimum cosine ≈ 1).
