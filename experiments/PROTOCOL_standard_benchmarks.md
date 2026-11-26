# Protocol: E9 Standard Multi-Sensor & Multi-Scale Vanilla Benchmarks (Frozen 2026-11-12)

Frozen prior to running any downstream evaluation on one workstation (`NVIDIA GeForce RTX 4090`). Follows `PROTOCOL-COMMON.md` (`≤ 10 GB` VRAM per job, `≤ 25 GB` scratch usage under `~/.claude-science-scratch/claypaper/E9_standard_benchmarks/`, official splits only, zero test-set tuning).

## 1. Benchmark Suite (12 Non-Crop Tasks Across 6 Sensor Families & 3,000x GSD)

All tasks use their official published `train`, `val`/`valid`, and `test` partitions with zero agriculture/crop-type benchmarks:

1. **`m-pv4ger`** (`GEO-Bench v1.0`): `0.1 m` RGB aerial (`320x320`, `uint8`), binary solar PV classification (`11,814 / 999 / 999`), metric: Test Accuracy (`OA`) & Macro-F1.
2. **`m-pv4ger-seg`** (`GEO-Bench v1.0`): `0.1 m` RGB aerial (`320x320`, `uint8`), binary solar PV segmentation (`3,000 / 403 / 403`), metric: Test mIoU (2-class mean IoU and positive-class PV IoU) via frozen `Z_patch` per-token linear head bilinearly upsampled to `320x320`.
3. **`UCMerced`** (`TorchGeo`): `0.3 m` USGS RGB aerial (`256x256`, `uint8`), 21-class land-use classification (`1,260 / 420 / 420`), metric: Test `OA`.
4. **`RESISC-45`** (`TorchGeo`): `0.2-30 m` multi-scale RGB (`256x256`, `uint8`, nominal `gsd=1.0 m`), 45-class scene classification (`18,900 / 6,300 / 6,300`), metric: Test `OA`.
5. **`m-chesapeake`** (`GEO-Bench v1.0`): `1.0 m` USDA NAIP 4-band RGB+NIR (`256x256`, `[0,1]`), 7-class land-cover segmentation (`3,000 / 1,000 / 1,000`), metric: Test 7-class `mIoU` via frozen `Z_patch` per-token linear head bilinearly upsampled to `256x256`.
6. **`m-eurosat`** (`GEO-Bench v1.0`): `10 m` Sentinel-2 10-band VSWIR (`64x64`, `int16`; channels selected by index from the documented true order `B01..B08, B09, B10, B11, B12, B8A`, i.e. `B8A, B11, B12` reside under keys `12 - SWIR`, `10 - SWIR - Cirrus`, `11 - SWIR`; the same sample-chip band check is run), 10-class LULC (`2,000 / 1,000 / 1,000`), metric: Test `OA`.
7. **`m-brick-kiln`** (`GEO-Bench v1.0`): `10 m` Sentinel-2, `64x64`, binary brick-kiln detection (`15,063 / 999 / 999`), metric: Test `OA` & positive F1. **The distributed archive stores 13 channels under wrong names** (geobench 1.1.0 "Known issues"; ServiceNow/geo-bench issue 29). The true channel order is `B01, B02, B03, B04, B05, B07, B8A, B08, B11, B12, TCI_R, TCI_G, TCI_B`; `B06`, `B09`, `B10` are absent and the last three channels are 8-bit true-colour composites. The runners therefore select the **9 physical reflectance bands `B02, B03, B04, B05, B07, B08, B8A, B11, B12` by channel index** (not by stored name), pass their own wavelengths (`0.493 ... 2.190 um`), and exclude the TCI channels. Every run verifies on 300 sample chips that the stored names are the documented 13, that `TCI_R/G/B` correlate > 0.99 with `B04/B03/B02` and are <= 255, and that `B8A`~`B08` > 0.95 (stored in `input_contract.band_check`); the archive SHA-256 and HF revision are recorded. The earlier "10 standard reflectance bands" description does not apply to this archive.
8. **`m-so2sat`** (`GEO-Bench v1.0`): `10 m` Sentinel-2 10-band VSWIR (`32x32`, reflectance `[0,1]`; plus 12-band S2 + Lee-filtered S1 VV/VH dB fusion arm), 17-class Local Climate Zone urban morphology (`19,992 / 986 / 986`, geographic city split), metric: Test `OA`.
9. **`EuroSAT-S1`** (`Copernicus-Bench`): `10 m` Sentinel-1 C-band SAR `VV, VH` (`64x64`, `float32` dB), 10-class LULC (`16,200 / 5,400 / 5,400`), metric: Test `OA`. (Note: in `standard_benchmarks.json`, both Clay and DOFA-L received Clay's polarization-distinct pseudo-wavelengths `[3.5, 4.0] um` for `EuroSAT-S1` and `m-so2sat` 12-band; TorchGeo DOFA's nominal S1 convention is `[3.75, 3.75] um`.)
10. **`m-forestnet`** (`GEO-Bench v1.0`): `15 m` composited Landsat-8 6-band VSWIR (`332x332`, `uint8`), 12-class deforestation driver classification (`6,464 / 989 / 993`), metric: Test `OA`.
11. **`LC100Cls-S3`** (`Copernicus-Bench`, **unseen sensor**): `300 m` Sentinel-3 OLCI 21-band radiance (`0.400-1.020 um`, `94x94`), 23-class multi-label global land cover (`5,181 / 1,727 / 1,727`, `static` mode; joining `static_fnames-test.csv` to `multilabel-test.csv` explicitly by `PID` to fix TorchGeo's row-order mismatch on `test`), metric: Test macro mAP & micro mAP.
12. **`Biomass-S3`** (`Copernicus-Bench`, **unseen sensor**): `300 m` Sentinel-3 OLCI 21-band radiance (`94x94`) predicting `100 m` ESA CCI forest aboveground biomass (`282x282`, `t/ha`, `3,000 / 1,000 / 1,000`, `static` mode), metric: Test pixel-level RMSE & $R^2$ from frozen `Z_patch` linear Ridge head (plus chip-mean $R^2$/RMSE from pooled tokens).

## 2. Frozen Models & Preprocessing Parity

- **Nodata handling:** Non-finite pixels (`NaN`, `±inf`) are imputed with each chip's unmasked per-band mean prior to normalization (falling back to training-split band mean if an entire band is missing).
- **Primary radiometric normalization (strict parity across Clay v1.5 and DOFA-L):** Per-band `(mean, std)` computed strictly on the `train` split of each dataset (`std` floored at `1e-6`). Secondary diagnostic arm (`clay_meta`) applies fixed `configs/metadata.yaml` constants on sensors in native units.
- **Frozen backbones (`fp16` autocast on RTX 4090; verified `fp32` vs `fp16` `CLS` cosine `= 1.000000`):**
  1. **Clay v1.5** (`clay-v1.5.ckpt`, SHA-256 `21432069...98d0`, `311.4M` params, zero metadata `[0]*4` time/latlon for parity): native chip size when `H in [32..256]` and `H % 8 == 0`; `256x256` (with scale-adjusted `gsd = gsd_0 * H / 256`) when `H > 256` (`m-pv4ger`, `m-pv4ger-seg`, `m-forestnet`) or non-divisible (`96x96` for `94x94` Sentinel-3 OLCI). Readouts: `CLS`, `mean`, and `Z_patch` (for dense tasks; plus `clay_up224` diagnostic on small `32x32`/`64x64` chips).
  2. **DOFA-L** (`DOFALarge16_Weights.DOFA_MAE`, `304M` params): `224x224` bilinear resize, identical wavelengths in `um` and identical `train`-split per-band `(mean, std)`. Readouts: `mean` (official `DOFA` default with `fc_norm`) and `CLS`, plus `14x14` `Z_patch` for dense tasks (in `standard_benchmarks.json`, `extract_dofa` applied `self.dofa.fc_norm` to `mean` and parameterless `F.layer_norm` followed by `StandardScaler` to `cls` and `patches`).
  3. **Scale-MAE-L** (`ScaleMAELarge16_Weights.FMOW_RGB`, `304M` params): `224x224` RGB (`uint8 / 255` or `train`-split `1-99%` stretched RGB `[0,1]`), ImageNet normalization, `res = gsd_0 * H / 224.0`. Omitted on pure SAR (`EuroSAT-S1`). Readouts: `CLS`, `mean`, `14x14` `Z_patch`.
  4. **DINOv2-L** (`vit_large_patch14_reg4_dinov2.lvd142m`, `304M` params): `224x224` RGB (`[0,1]`), ImageNet normalization. Omitted on pure SAR (`EuroSAT-S1`). Readouts: `CLS`, `mean`, `16x16` `Z_patch`.
  5. **Band Stats**: Non-learned per-chip `[mean, std]` across channels (`2C` features; per-pixel `C` channels for dense segmentation/biomass).

## 3. Probes & Hyperparameter Selection (Zero Test-Set Tuning)

- Feature standardization: `StandardScaler` fitted strictly on `train`.
- **Single-label classification** (`9` tasks): L2-regularized `LogisticRegression` (`lbfgs`, `max_iter=1000`) and `LinearSVC` (`dual="auto"`, `max_iter=3000`), selecting regularization `C in {0.01, 0.1, 1.0, 10.0}` solely by `val` accuracy, then refitting on `train` (never `train+val`) and scoring once on `test`.
- **Multi-label classification** (`LC100Cls-S3`): `OneVsRest` L2-regularized `LogisticRegression` (`C in {0.01, 0.1, 1.0, 10.0}` selected on `val` macro mAP), scored on `test`.
- **Dense segmentation** (`m-pv4ger-seg`, `m-chesapeake`): Ridge / multinomial linear $1\times 1$ projection fitted via exact normal equations $(\bm{Z}^\top \bm{Z} + \lambda \bm{I})^{-1}\bm{Z}^\top \bm{Y}$ or subsampled pixel `LogisticRegression` (`200k` stratified/uniform train tokens, `C`/`alpha` selected on `val` mIoU), logits bilinearly upsampled from the token grid to full pixel mask resolution before computing `test` pixel confusion matrix and `mIoU`.
- **Biomass regression** (`Biomass-S3`): Ridge linear projection (`alpha in {1, 10, 100, 1000, 10000}` selected on `val` RMSE) on `Z_patch` bilinearly upsampled to `282x282` (pixel-level `test` RMSE & $R^2$), plus chip-mean `Ridge` on pooled tokens.

## 4. Peer-model input contracts and runner output policy (audit fixes #9, #15, #17, #27, #35, #37)

These apply to `run_all_doable_table2.py` (OlmoEarth v1.3 Base, Panopticon, Copernicus-FM, DOFA-L, DINOv2-L, Clay) and
`run_geobench_and_e7_complete.py`; every stored cell carries its contract (`_contract` / `_settings`) and a producer fingerprint.

- **Copernicus-FM:** receives the *true* chip footprint (native pixels x GSD, taken before the 224 px resize; km^2) and the per-band spectral
  bandwidths of the ICCV 2025 supplement (S2: B01 20, B02 65, B03 35, B04 30, B05 15, B06 15, B07 20, B08 115, B8A 20, B09 20, B10 30, B11 90,
  B12 180 nm; S3 OLCI 21 widths; S1: centre 5.0e7 nm, width 1e9 nm). Aerial RGB/NIR and Landsat-8 inputs use the width of the S2 band at the
  same wavelength (marked as proxies). CropHarvest-Togo pixels have a 1e-4 km^2 footprint, below the model's supported area range
  `[1e-3, 5.1e8]` km^2; the area is clamped to the lower bound and the clamp is recorded in the contract.
- **OlmoEarth v1.3 Base:** S2-native tasks use OlmoEarth's own `Normalizer(Strategy.COMPUTED)` (raw DN scale). The train-split z-score arm is kept as
  the diagnostic cell `olmo_v13_trainz`. The checkpoint's tokenisation has a single band group per modality (Landsat: B8,B1..B7,B9..B11 in library order;
  S2: all 12 bands in library order); bands are placed by name and absent bands are left at the normalised zero.
- **CropHarvest-Togo:** head policy fixed before any test score: `C in {0.01,0.1,1,10}` chosen on a seeded 85/15 train/val split, refit on all
  training labels with that `C`, test scored once (`oa`). The split-fit and LSVM test accuracies are stored as labelled diagnostics only. DOFA-L gets
  its own SAR convention (both polarisations 3.75 um); the Clay convention (3.5, 4.0 um) is kept as `dofa_s1_clay_waves_diag`.
- **Output policy:** every runner writes to `--output-dir` (default a fresh `experiments/runs/<name>_<UTC>/`), never to a committed file. `--resume`
  is explicit and skips a cell only if its stored `script_sha256` and contract hash equal the current ones. Each JSON records script, sha256, git
  commit, argv, UTC time, package versions and GPU.
- **Sen1Floods11 follow-up arms** follow the Clay E7 protocol (100 epochs, validation every 5, seeds 0, 1, 2); per-seed values, seed range and the
  number of seeds actually run are stored per cell (`experiments/aggregate_e7_seeds.py` combines per-seed runs).
