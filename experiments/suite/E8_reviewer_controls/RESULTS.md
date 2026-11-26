# E8 — Reviewer-requested controls (results, 2026-11-06)

Protocol: `PROTOCOL.md`, frozen 2026-11-05 22:25 UTC before any E8 score. CIs are 95 % from 1,000 block-bootstrap resamples (100 km blocks, seed 1729); paired contrasts share resamples. All numbers are in `results.json`.

**Deviations.**
- The NAIP-native pooled arm (1.25 m, 64 tiles per location) was **not run to completion and has no scores**. Attempt 1 hit a CUDA OOM because an Ollama model (16–18 GB) was reloaded on the shared GPU. Attempt 2 completed fewer than 100 of the 2,000 queued locations in 1 h 43 min (NAIP blob connection failures, 2048 px reads) and was cancelled.
- In item 1, 331 S2-256 chips were invalid (119 had nodata, 212 were < 90 % clear) and 32 NAIP chips were invalid. Both chips were valid at 6,230 of the 6,592 E3 locations. Classification uses the **L = 6,190** of these in the 7 E3 classes (5,212 train, 978 test, 85 test blocks).
- The central 32 × 32 px of 5,967 / 6,592 (90.5 %) new S2 chips is byte-identical to the cached E3 chip. The other 9.5 % differ despite using the same item IDs; the mismatch is concentrated in multi-item mosaics, and the cause was not investigated.
- Item-1 heads were fit sequentially. A threaded parallel run produced NaNs in scipy and was discarded before any score was read.
- Item 3(a) reproduces the E1 AEF-split n = 10 contrasts to 3 decimals on 14 tasks. LUCAS-LU gives +0.039 here, against +0.041 reported.

## 1. E3 context-matched control (S2 → NAIP, Clay CLS, metadata on, training order)

Both sensors were chipped over the same 2.56 km footprint at 10 m: S2 at 256 px, and NAIP average-resampled to 256 px with gsd = 10. The E3 numbers below are recomputed on the same L.

| | E3 320 m chips | 2.56 km chips, CLS | 2.56 km, centre 4×4 patch tokens |
|---|---|---|---|
| native NAIP head | 0.833 [0.795, 0.858] | 0.481 [0.421, 0.527] | 0.798 [0.753, 0.833] |
| native S2 head | 0.727 [0.684, 0.757] | 0.543 [0.470, 0.594] | 0.828 [0.792, 0.858] |
| zero-shot S2 head → NAIP | 0.242 [0.201, 0.274] | 0.259 [0.201, 0.312] | 0.350 [0.297, 0.393] |
| zero-shot, NAIP re-standardised | 0.234 | 0.339 [0.279, 0.383] | 0.549 [0.492, 0.588] |
| Procrustes, 1,000 pairs | 0.368 [0.332, 0.391] | 0.160 [0.147, 0.169] | 0.229 [0.209, 0.246] |
| Δ = zero-shot − native | −0.591 [−0.629, −0.548] | −0.222 [−0.278, −0.161] | −0.448 [−0.504, −0.391] |
| R@10, S2 → NAIP (chance 0.16 %) | 1.7 % [1.3, 2.1] | 27.3 % [24.8, 29.8] | 6.9 % [5.7, 8.1] |
| R@10, per-sensor centred | 2.5 % | 41.0 % [38.1, 43.7] | 8.0 % |
| paired − unpaired cosine | 0.035 [0.032, 0.039] | 0.167 [0.160, 0.172] | 0.238 [0.222, 0.254] |

(The majority baseline is 0.076.)

**Primary endpoint 1a.**
- Δ256 − Δ32 = **+0.369 [0.300, 0.434]**, but |Δ256| = 0.22 ≥ 0.10. By the pre-declared reading, context "explains part of the gap".
- The table shows that this narrowing comes almost entirely from the **native** head falling: −0.352 [−0.409, −0.294]. Zero-shot does not improve: 0.259 vs 0.242, +0.017 [−0.040, 0.073].
- The cause is that labels describe the central 320 m cell, while a 2.56 km CLS token summarises 64× the area.
- With the centre-patch readout, which matches the labelled cell, the native heads recover (0.80 / 0.83). Transfer still fails there:
  - Δ = −0.448;
  - zero-shot improves over E3 by +0.108 [0.061, 0.155];
  - re-standardised zero-shot improves by +0.315 [0.252, 0.364].
- Procrustes is worse at 2.56 km than at 320 m in every readout.
- **Reading:** matching context and GSD does not make a Clay head portable from S2 to NAIP. The E3 cross-sensor conclusion stands.

**Alignment does improve with matched context.**
- Paired retrieval rises from 1.7 % to 27.3 % (+25.6 pp [23.2, 28.0]; centred +38.4 pp).
- The paired-vs-random cosine gap grows about fivefold.
- Same pattern with metadata off: R@10 26.1 %, centred 40.1 %.
- The mean token barely moves (1.0 → 3.0 %).
- The reference arms align much better:
  - DINOv2-L on display RGB goes from 4.2 % to 90.0 % R@10;
  - raw RGBN pixels go from 11.9 % to 17.1 %.
- Clay CLS therefore beats raw pixels but stays far below an RGB image model on this paired-chip test.
- Gallery chips from the same E3 tile overlap the query footprint. Excluding them changes Clay CLS R@10 only slightly (28.3 %).

**1b — same sensor, different chip size (CLS).**
- S2 320 m head → S2 2.56 km: 0.295 [0.231, 0.340], against native 0.543 (**Δctx = −0.248**).
- S2 2.56 km head → S2 320 m: 0.185 vs native 0.727 (Δ −0.542).
- NAIP 320 m @1.25 m → NAIP 2.56 km @10 m: 0.205 vs 0.481. The reverse direction gives 0.244 vs 0.833.
- Zero-shot macro-F1 after a chip-size change alone (0.19–0.30) is in the same range as after a sensor change (0.24–0.26).
- Retrieving the same location across chip sizes is also poor: S2 320 m → S2 2.56 km R@10 = 5.3 %.
- **Reading:** a frozen-CLS head does not transfer across chip size even within one sensor. Chip context is a first-order offset, comparable to the sensor offset. A Clay CLS head should be used only at the chip size and GSD it was fit on.

DINOv2 behaves the same at the head level: zero-shot 0.367 vs native 0.459 at 2.56 km, and 0.388 vs 0.734 at 320 m.

Figure: `fig_e8_context.png` / `.pdf`. Panel a: dotted line = majority class.

## 2. Pretraining overlap (centroid inside a Clay training MGRS tile, `mgrs_sample_v02.fgb`, 2,728 tiles)

- **E1 AEF test rows** (tier-0, with chip): **36.8 %** (3,342 / 9,088); train rows 29.8 %.
  - Per task: ASTER 14.6 %, Africa 21.0 %, Descals 21.3 %, Canada fine 22.7 %, GLanCE 24.9 %, LCMAP LCC 25.0 %, Canada coarse 25.6 %, LCMAP LUC 28.0 %, OpenET 28.0 %, LCMAP LC 30.7 %, LCMAP LU 31.7 %, US trees 39.8 %, LUCAS LU 52.3 %, LUCAS LC 53.3 %, Ethiopia 54.0 %.
- **E2 test chips:** **23.9 %** (446 / 1,868). By sensor: S2 22.8 %, S1 23.4 %, Landsat 23.2 %, NAIP 27.5 %. By stratum: CONUS 26.7 %, global 18.5 %.
- **E3 test locations:** **21.4 %** (220 / 1,027); all locations 26.1 %.
- **E4 Part A chips:** **7.6 %** (6,983 / 92,309): CA 9.4 %, NC 13.5 %, TX 0 % (64 px chips; 256 px chips similar).
  - None of the three test tiles (11SLU, 15STT, 18STE) is in the sample by name. The inside fraction comes from neighbouring sample tiles' overlapping footprints.
- This is spatial overlap only; it says nothing about acquisition date or chip identity.

## 3. E1 sensitivity (cached E1 shards; CPU)

**(a) Re-split with 100 km hblock blocks** (clay_cls − stats, n = 10/class). The first value is the AEF split as in E1, the second the hblock re-split.

| Task | AEF split | hblock re-split |
|---|---|---|
| Africa crop | +0.258 | +0.257 [0.152, 0.338] |
| ASTER R² | −0.001 | +0.040 [−0.000, 0.089] |
| Canada coarse | +0.047 | +0.022 [−0.018, 0.059] |
| Canada fine | +0.054 | +0.039 [0.017, 0.053] |
| Descals | +0.145 | +0.154 [0.099, 0.211] |
| Ethiopia | +0.022 | −0.025 [−0.180, 0.200] |
| GLanCE | +0.096 | +0.121 [0.094, 0.148] |
| LCMAP LC | +0.081 | +0.099 [0.063, 0.137] |
| LCMAP LCC | +0.002 | −0.022 [−0.070, 0.024] |
| LCMAP LU | +0.098 | +0.126 [0.090, 0.160] |
| LCMAP LUC | +0.026 | +0.011 [−0.053, 0.084] |
| LUCAS LC | +0.023 | +0.021 [0.002, 0.038] |
| LUCAS LU | +0.041 | +0.033 [0.003, 0.062] |
| OpenET R² | +0.203 | +0.076 [0.039, 0.114] |
| US trees | +0.016 | +0.018 [0.002, 0.029] |

- The sign flips on three tasks whose CIs include 0 under both splits: ASTER −0.001 → +0.040, Ethiopia +0.022 → −0.025 and LCMAP LCC +0.002 → −0.022.
- 10 of 15 tasks keep CIs above 0 (11 under the AEF split), and 13 of 15 point estimates are positive.
- Canada coarse loses significance, and ASTER moves to the edge of significance.
- OpenET shrinks most, +0.203 → +0.076.
- Ethiopia has only 8 test rows in 5 blocks under the re-split, so it is uninformative.

**(b) Distance from each AEF test row to its nearest train row (AEF split).**
- No test row lies within 1 km of a train row on any task.
- Median distances: Canada fine 3.1 km and Canada coarse 4.6 km (92 % / 86 % within 10 km); US trees 18.7 km; Ethiopia 19.7 km; OpenET 20.1 km; LUCAS LC 20.8 km; Descals 26.8 km; LCMAP 28–64 km; Africa 34.3 km; GLanCE 62.5 km; ASTER 94.4 km.
- The Canada tasks are therefore effectively spatially interleaved. OpenET (21 % of test rows within 10 km) is the task most sensitive to the split, as (a) shows.

**(c) Readouts from E1 `results.json`** (linear head; full table with CIs in `results.json`):
- At n = 10, the centre-patch readout has a higher point estimate than CLS on 11 of 15 tasks, by up to +0.13 (Canada fine 0.341 vs 0.216; Canada coarse 0.352 vs 0.260).
- Coordinates alone beat every Clay readout on US trees (0.117 vs 0.073–0.079) at n = 10, and on Ethiopia at max (0.401 vs 0.277–0.322).
- Realised max n per class: 200 for Africa and Descals, 68 for Canada fine, 49 for Ethiopia, 300 for GLanCE / LCMAP LC, LU, LCC / OpenET, and 150 for LCMAP LUC.
- Max was not run for ASTER, Canada coarse, LUCAS LC/LU or US trees. AEF train rows available: ASTER 99.95 %, Canada coarse 70.6 %, LUCAS LC 15.5 %, LUCAS LU 16.7 %, US trees 15.6 %. Canada fine had 76.1 % and Ethiopia 70.9 %, but their max budgets were complete; all other tasks had 100 %.
- **LCMAP land-use change, band statistics at max:** BA 0.580 [0.491, 0.670]. The CI includes chance (0.50), so it is **not** above chance. Clay CLS is 0.520 [0.423, 0.616].

## 4. E2 location probe (ridge on CLS, fit on all train chips; test-chip subsets)

| S2 test chips | lon R², meta on (training) | meta off | median error, on / off (km) |
|---|---|---|---|
| all (535) | 0.759 [0.679, 0.823] | 0.484 [0.390, 0.556] | 1,409 / 2,687 |
| CONUS (314) | −0.77 [−1.30, −0.40] | −4.80 | 1,167 / 2,034 |
| non-CONUS (221) | 0.533 [0.370, 0.665] | 0.070 [−0.108, 0.214] | 2,292 / 4,302 |

- The documented order gives the same values as the training order on S2: lon R² 0.763, median 1,369 km.
- A unit-vector probe roughly halves the error: training order 853 km overall, 635 km CONUS, 1,248 km non-CONUS; metadata off 1,549 km.
- S1 meta on: lon R² 0.862, 1,041 km; off 0.235, 3,726 km. Landsat: 0.812 / 0.499, 1,200 / 2,430 km.
- NAIP (CONUS only): 0.834 / 0.725, 373 / 491 km.
- Within CONUS, the globally fit probe is worse than predicting the CONUS mean longitude (negative R²). Its decodable location is a coarse CONUS-vs-rest signal.
- For context, the same linear probe on the exact sin/cos lat/lon input code gives S2 lon R² 0.854 and a 1,328 km median (E2 `results.json`). The probe itself, not only the embedding, limits precision.
- **Reading:** metadata-on CLS carries location at continental scale. It does not localise a chip within CONUS under a linear probe.

## Caveats
- Item 1 keeps E3's 320 m-cell labels on 2.56 km chips, by design. The CLS native-head drop is a label-footprint effect, and the centre-patch readout is the fairer 2.56 km comparison.
- NAIP at 10 m was produced with GDAL average resampling from COG overviews. 54 % of chips were mosaicked from more than one same-year NAIP item.
- The NAIP-native pooled arm is missing (see Deviations).
- WorldCover labels are noisy (purity ≥ 0.60).
- Overlap fractions are spatial only.

## Audit confirmation (2026-11-17)
`code/e8_confirm.py` copies the manuscript-facing numbers out of `../../results/e8_reviewer_controls.json` into `../../results/e8_confirmed_numbers.json`, each with the JSON key it came from; nothing is recomputed, and every value below matches this file as committed.
- **Overlap (key `overlap`).** 3,342 / 9,088 AEF-suite test rows (36.8 %) have their centroid inside a sampled Clay pretraining MGRS tile footprint (train rows 7,405 / 24,878 = 29.8 %). This is spatial proximity only; there is no acquisition-date or chip-identity match.
- **Held-out sensitivity (keys `e1_sensitivity_n10_clay_cls_minus_stats`, `e1_sensitivity_summary`).** OpenET 10-shot Clay CLS minus band statistics: +0.203 [0.123, 0.290] on the supplied AEF split (500 test rows, 335 blocks) and +0.076 [0.039, 0.114] on the 100 km held-out split (485 rows, 92 blocks). Sign flips on ASTER (−0.001 → +0.040), Ethiopia (+0.022 → −0.025) and LCMAP LCC (+0.002 → −0.022); in all three the CI includes 0 under both splits. CI above 0: 11/15 tasks on the AEF split, 10/15 on the hblock split. The bootstrap unit differs between the two evaluations (fixed split vs 100 km blocks), so they answer different questions.
- **Chip-size confound next to E3 (key `chip_size_confound`).** S2→NAIP, Clay CLS, same 5,212 / 978 train/test locations: at 320 m / 32 px native 0.833 [0.795, 0.858], zero-shot 0.242 [0.201, 0.274], Δ = −0.591 [−0.629, −0.548]; at 2.56 km native 0.481 [0.421, 0.527], zero-shot 0.259 [0.201, 0.312], Δ = −0.222 [−0.278, −0.161]; with the centre 4 × 4 patch tokens at 2.56 km native 0.798, zero-shot 0.350, Δ = −0.448 [−0.504, −0.391]. A chip-size change alone, same sensor, gives zero-shot macro-F1 0.185–0.295 (S2: 0.295 and 0.185; NAIP: 0.205 and 0.244), the same range as the sensor change (0.24–0.26). On the full E3 pool of 6,540 S2/NAIP locations the drop is −0.575 (`../E3_cross_sensor/RESULTS.md`); the −0.591 here is E3's design re-scored on the L = 6,190 locations that have valid 2.56 km chips.
- **Footprints.** Landsat chips in E3 are 480 m, not 320 m.
