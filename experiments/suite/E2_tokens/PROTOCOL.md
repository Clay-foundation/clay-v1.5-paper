# E2_tokens — what Clay's class and patch tokens carry (frozen 2026-11-05, before any score)

Follows `../PROTOCOL-COMMON.md`. Clay v1.5 via `common/clayrt.py` (strict load, SHA-pinned), eval mode, no masking, real time + location metadata unless an arm says otherwise.

## Data and sampling (declared before fetching)
- Points. Two strata, RNG `numpy.default_rng(20261105)`:
  - GLOBAL: ESA WorldCover 2021 (PC `esa-worldcover`, v200) 3°×3° tiles; tiles ranked by sha256 hash, one uniform random point per tile, cycling until 1,500 points are accepted.
  - CONUS: uniform random points in lon [-124.7,-66.9], lat [24.5,49.4], until 2,000 points are accepted.
  - A point is accepted if the WorldCover window at the Sentinel-2 footprint (2.56 km) has ≤ 50 % water (class 80) + nodata, and (CONUS only) a NAIP item exists.
- Chips (256×256), centred on the point, on the scene's native projected CRS, nearest resampling unless stated:
  - Sentinel-2 L2A (10 m, 10 Clay bands, raw DN, no harmonisation), all points; year 2021, scene cloud < 20 %, lowest-cloud first; reject a chip with any nodata or > 5 % SCL cloud/shadow (SCL 3, 8, 9, 10) and try the next scene (≤ 4 scenes). `s2:processing_baseline` recorded.
  - Sentinel-1 RTC (10 m, VV/VH → dB = 10·log10), on the S2 chip's geobox, scene nearest in time to the S2 scene within ±60 days (else any 2021 scene); reject non-finite/≤ 0 pixels.
  - Landsat C2 L2 (30 m, Landsat-8/9, raw DN, `landsat-c2l2-sr` metadata), all points; 2021, cloud < 20 %; reject nodata or > 5 % QA_PIXEL cloud/shadow (bits 3, 4), ≤ 4 scenes.
  - NAIP (CONUS points only), latest item 2019–2022, resampled to 1.0 m (bilinear) to match Clay's NAIP GSD = 1.0; reject nodata.
- Dense labels: WorldCover 2021 `map` on the same CRS/extent, at 10 m (1 m nearest for NAIP), majority class per token (8×8-pixel) cell → 32×32 label grid; token purity recorded, no purity filter; nodata tokens dropped.
- Expected yield ~2,000–3,500 chips per sensor (whatever the fetch delivers is reported; failures are counted, not replaced).

## Splits, token subsampling, heads
- Split: `evalkit.hblock` 100 km on the chip centre → `block_split`; train = train ∪ val (fixed heads, no tuning), test = test. Co-located chips of different sensors share a block and split.
- Bootstrap blocks: the 100 km blocks of test chips; 1,000 resamples, seed 1729; paired contrasts share resamples.
- Training tokens: 64 fixed tokens per train chip (sha256-ranked positions). Logistic heads use, for seed s∈{0..4}, the sha256(s)-ranked 16 of those 64 per chip (seed = token draw). Ridge heads use all 64 (deterministic).
- Test tokens: every second row and column (16×16 = 256 tokens per test chip), all test chips.
- Heads: `evalkit.clf_head` (StandardScaler + LogReg C=1) and `evalkit.reg_head` (RidgeCV, inner CV on train).

## Feature arms
- Clay `patches` (1024), `cls` broadcast to all tokens, `mean` broadcast.
- Raw per-token band statistics: per-band mean and std of the normalised input over the 8×8 cell (2C features).
- DINOv2 ViT-L/14-reg4 patch tokens on display RGB (`to_display_rgb`), input 448 px → 32×32 token grid aligned to Clay's; S2, Landsat, NAIP (no S1).
- Position-code controls: the implemented `posemb_sincos_2d_with_gsd` (dim 1016) at each sensor's GSD; the standard `posemb_sincos_2d` (dim 1016) as reference.

## Analyses
(i) Position probe. Ridge from token features → row and column (0–31), separately; report R² on test tokens (mean of row/col R² and each). Per sensor, arms: Clay patches, DINOv2 patches, band stats, implemented code at the sensor's GSD, standard code. The codes are identical across chips, so their R² is a linear-decodability ceiling; an extra checkerboard variant (train on (r+c) odd positions, test on even) is also reported for the codes.
(ii-a) Sensor / GSD probes. Chip-level `cls` and `mean` from all four sensors (real metadata). Logistic → sensor (macro-F1, acc); ridge → log10 GSD (R², RMSE). Within-sensor GSD is constant, so this is also confounded with band count/wavelength code; reported as such.
(ii-b) GSD lie. The same S2 chips embedded with gsd ∈ {1, 5, 10(true), 30, 100}. Report: cosine to the true-GSD embedding for `cls`, `mean` and per-token `patches` (mean over tokens), mean over test chips with block-bootstrap CI; WorldCover dense probe trained on true-GSD patches and applied to each lie (transfer; paired contrast vs gsd 10), and retrained per GSD; position probe retrained per GSD; ridge → log10(supplied gsd) from `cls`/`mean` across the 5 arms (split by chip block).
(iii) Dense WorldCover probe per sensor: arms patches, cls-broadcast, mean-broadcast, band stats, DINOv2 patches. Metrics: macro-F1 and mIoU over the classes present in the sensor's test labels, token-level, pooled over test chips, seed-mean with block-bootstrap CI.
(iv) Location leakage: chip `cls` (and `mean`) with metadata ON (real lat/lon + time) vs OFF (lat/lon and time zeros), same chips, per sensor. Ridge → (lat, lon); R² for each and median great-circle error (km). Controls: the 4-d lat/lon code itself (ceiling), chip-level band stats (content only). Also reported: cosine between ON and OFF `cls`.

## Endpoints
- Primary (pre-registered): Sentinel-2 dense WorldCover macro-F1, Clay patches minus cls-broadcast, paired block bootstrap.
- Key secondary: S2 position R² (mean row/col), Clay patches vs DINOv2 patches; S2 location R² (lat) cls ON minus OFF; S2 dense macro-F1 transfer drop at gsd 1 and 100 vs 10.
- All other numbers are descriptive.

## Amendment 1 (2026-11-05, before any full-data embedding or score)
- Lat/lon slot order. Clay v1.0/v1.5 were trained with `latlon = [sin lon, cos lon, sin lat, cos lat]` (stacchip order bug; found by E4, fixed in `common/clayrt.py`, `LATLON_ORDER='training'`). All metadata-ON Clay arms here use the **training order**. The embedding job was patched before its embed stage started, so no score was ever computed with the old order.
- Added sensitivity arm for (iv): `cls`/`mean` with metadata ON in the **documented order** `[sin lat, cos lat, sin lon, cos lon]` (out-of-distribution metadata), on the same chips; reported next to ON (training order) and OFF.
- DINOv2 input size: 448 px (as frozen above, giving the 32×32 patch grid aligned to Clay's tokens) via a local timm wrapper (`img_size=448`), not the 224 px `DinoV2Runtime` default.
- Fetch engineering (no change to sampling rules): a global STAC request throttle with backoff retry, plus a local WorldCover tile cache. These were added after a smoke test in which Planetary Computer rate-limited requests (`APIError: rate limited`). Points that fail only for transient errors are not counted as rejections.
