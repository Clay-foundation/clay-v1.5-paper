# E1 — Label efficiency of frozen Clay v1.5: results (2026-11-05)

**What was measured.**
- **(a)** All 15 tasks of the AlphaEarth Foundations (AEF) evaluation release (Zenodo 16585402). Each row gets one least-cloudy Sentinel-2 L2A chip from Planetary Computer: 128 px / 1.28 km, raw values, inside the row's valid period. For change tasks, the embedding difference after − before is used. AEF's own train/test split is used; test is capped at 50 per class (500 for regression).
- Arms: Clay v1.5 CLS (primary), Clay mean patch, DOFA-L (10 bands), DINOv2-L/14-reg4 (RGB), per-band mean/std, coordinates.
- Heads: fixed linear (logistic C = 1 / RidgeCV). Budgets: 1, 10 and AEF's max labels per class. 5 seeds, 95% 100 km block-bootstrap CIs, paired contrasts.
- Clay received lat/lon in the order it was **trained** with ([sin lon, cos lon, sin lat, cos lat]). The documented order is a sensitivity arm.
- **(b)** The frozen EuroSAT spatial-split pilot, checked against Clay's training MGRS tiles and extended with DINOv2-L and DOFA-L.

**Primary endpoint (pre-registered): Clay CLS − band statistics at 10 labels/class.**
- **Positive on 11 of 15 tasks, negative on none.** Largest gains: Africa crop mask +0.258 BA [0.196, 0.326], OpenET +0.203 R² [0.123, 0.290], oil palm +0.145 [0.088, 0.196], LCMAP land use +0.098 [0.068, 0.132], GLanCE +0.096 [0.072, 0.119].
- **Null on 4 tasks:** ASTER emissivity −0.001 R² [−0.024, 0.023], Ethiopia crops +0.022 [−0.054, 0.100], LCMAP cover change +0.002 [−0.053, 0.055], LCMAP use change +0.026 [−0.041, 0.091].

**Against learned comparators at n = 10.** Clay CLS exceeds DOFA-L on 11/15 and DINOv2-L on 9/15 tasks, and is below neither on any task. The gaps are mostly small: DOFA by 0.012–0.064 BA where significant, except Africa (+0.110) and OpenET (+0.058 R²).

**More labels narrow the gaps.** n = max could be run on 10 tasks; for the other five the max-budget rows were not all fetched in time.
- Clay CLS beats statistics on 6/10, DINOv2 on 4/10 and DOFA-L on 3/10, and is significantly below none.
- DINOv2 nominally leads on the Africa crop mask (0.880 vs 0.830 BA; Δ −0.050 [−0.114, 0.020]).

**Change detection fails in this setup.** For both LCMAP change tasks, every arm stays near chance at every budget (Clay CLS max BA 0.500 and 0.520). AEF reports 0.78 / 0.79. Differencing two single-date chips does not carry annual change information.

**Against AEF's published numbers.** These are AEF's numbers transcribed from the paper (Fig. 1A labels; §6 text): max trial, best of three transfer methods, multi-source multi-date annual summaries. They are not a matched protocol.
- At max, Clay CLS is below AEF on every task: LCMAP LC 0.660 vs 0.84, LCMAP LU 0.670 vs 0.87, GLanCE 0.538 vs 0.68, Canada fine 0.276 vs 0.49, OpenET R² 0.346 vs 0.58.
- Clay CLS is close on oil palm (0.787 vs 0.79) and Africa crop mask (0.830 vs 0.89).
- ASTER R² reaches 0.713 [0.660, 0.755] at only 10 per bin, against AEF's max-trial 0.72.
- AEF's own Clay baseline (768-D, multi-date) is printed only for LUCAS (LU 0.32, LC 0.26). We could not run LUCAS at max; at 10/class we get 0.193 / 0.128.

**Lat/lon order.** Documented vs training order changes no task's score at n = 10, except US trees (+0.004 [0.001, 0.008] for training order).

**(b) EuroSAT.**
- **Overlap:** 69.1% of the pilot's test patches (2,194 / 3,176) have centroids inside Clay training MGRS tiles. This is spatial overlap only; acquisition dates were not matched.
- **Primary endpoint after excluding them** (CLS − spectral20 + xy2, 50/class, macro-F1, unchanged frozen script): +0.162 [0.051, 0.250] with training-order Clay (982 test patches, 28 blocks), against +0.175 [0.107, 0.238] on the full test set. With the pilot's documented-order embeddings the values are +0.159 [0.047, 0.246] and +0.169 [0.101, 0.230]. The advantage survives the exclusion.
- **Comparators at 50/class (full test set):**
  - DOFA-L matches Clay: Δ +0.005 [−0.006, 0.017].
  - DINOv2-L RGB is lower: Clay − DINOv2 +0.041 [0.014, 0.063].

**What this means.**
- On single-date chips, frozen Clay CLS is a reliable low-label default: it never loses to the controls or the two learned comparators.
- It is not a decisive winner. A multispectral comparator (DOFA-L) is equal on EuroSAT and usually within a few points on AEF tasks, and its advantage over DINOv2 and DOFA shrinks as labels grow.
- The large gap to AEF on annual or phenology targets (crops, land use, change, ET) is consistent with one date carrying little seasonal information. Our design cannot separate that from model quality.

**Caveats.**
- The observation window and context differ from AEF's (single date vs annual multi-source; 1.28 km chip vs per-pixel embedding), and so do heads, repeats (5 vs 500–1,000) and CI type (95% bootstrap vs 1σ).
- Test sets are capped.
- Max-budget results are missing for ASTER, Canada coarse, LUCAS LC/LU and US trees: the fetch hit Planetary Computer rate limits, and the scope was reduced by dated amendments before any score.
- 100 km blocks put most CONUS test rows in their own block, so those CIs are close to row-level.
- The 1-shot OpenET R² of band statistics is −3.5, so contrasts against it at n = 1 are not meaningful.
- 81% of chips are processing baseline 02.12; 13% are ≥ 04.00 (BOA offset, unharmonised, as in Clay training).

## Note — readout and coordinates contrasts quoted in the paper (added 2026-11-06, recomputed 2026-11-17 from `experiments/results/e1_label_efficiency_aef.json` `contrasts` (contrast `clay_cls-clay_center` and `clay_cls-coords`, linear head, training lat/lon order); the counts below were re-derived from that file and match: CLS minus centre has interval above 0 on 0 tasks, below 0 on 7/15 at n = 10 and 5/10 at max, with 3/15 and 1/10 point estimates above 0)

Clay CLS − Clay centre (2×2 central patch tokens); negative = centre better.

| task | n=10 | max |
|---|---|---|
| africa_crop_mask | +0.030 [-0.022, +0.086] | -0.040 [-0.108, +0.022] |
| aster_ged | -0.016 [-0.039, +0.003] | -- |
| canada_crops_coarse | -0.092 [-0.117, -0.061] | -- |
| canada_crops_fine | -0.124 [-0.144, -0.105] | -0.166 [-0.192, -0.143] |
| descals | -0.000 [-0.053, +0.048] | +0.060 [-0.012, +0.131] |
| ethiopia_crops | +0.017 [-0.042, +0.069] | -0.045 [-0.140, +0.093] |
| glance | -0.025 [-0.047, -0.003] | -0.040 [-0.085, +0.009] |
| lcmap_lc | -0.041 [-0.068, -0.013] | -0.077 [-0.141, -0.023] |
| lcmap_lcc | -0.036 [-0.096, +0.015] | -0.130 [-0.243, -0.008] |
| lcmap_lu | -0.069 [-0.099, -0.041] | -0.077 [-0.127, -0.032] |
| lcmap_luc | -0.008 [-0.058, +0.040] | +0.000 [-0.129, +0.128] |
| lucas_lc | -0.056 [-0.065, -0.048] | -- |
| lucas_lu | -0.040 [-0.058, -0.024] | -- |
| openet_ensemble | -0.018 [-0.089, +0.053] | -0.107 [-0.186, -0.033] |
| us_trees | +0.007 [-0.001, +0.013] | -- |

Centre significantly better: 7/15 at n=10, 5/10 at max; CLS significantly better: 0 at either budget.

Clay centre-token change detection (difference of two centre readouts), balanced accuracy:
- lcmap_lcc 10: 0.558 [0.511, 0.609]
- lcmap_lcc max: 0.630 [0.526, 0.726]
- lcmap_luc 10: 0.524 [0.482, 0.564]
- lcmap_luc max: 0.520 [0.426, 0.615]

Clay CLS − coordinates (single sinusoid per axis, E1 `coords` arm):

| task | n=10 | max |
|---|---|---|
| africa_crop_mask | +0.320 [+0.257, +0.387] | +0.320 [+0.193, +0.458] |
| aster_ged | +0.654 [+0.570, +0.741] | -- |
| canada_crops_coarse | +0.073 [+0.042, +0.096] | -- |
| canada_crops_fine | +0.085 [+0.067, +0.103] | +0.105 [+0.077, +0.131] |
| descals | +0.153 [+0.068, +0.241] | +0.280 [+0.180, +0.378] |
| ethiopia_crops | +0.001 [-0.138, +0.121] | -0.124 [-0.231, +0.018] |
| glance | +0.290 [+0.262, +0.317] | +0.375 [+0.331, +0.423] |
| lcmap_lc | +0.259 [+0.219, +0.300] | +0.270 [+0.203, +0.332] |
| lcmap_lu | +0.299 [+0.254, +0.343] | +0.333 [+0.273, +0.395] |
| lucas_lc | +0.039 [+0.028, +0.049] | -- |
| lucas_lu | +0.081 [+0.062, +0.100] | -- |
| openet_ensemble | +0.730 [+0.608, +0.864] | +0.914 [+0.771, +1.069] |
| us_trees | -0.037 [-0.048, -0.027] | -- |

n=10: Clay significantly above coordinates on 11/13, below on 1 (us_trees). max: above on 7/8, below on 0.

## Correction (2026-11-06)
The sentence above 'it never loses to the controls' is superseded: coordinates beat Clay CLS on US trees at n=10 (−0.037 [−0.048, −0.027]); see the table above and E5/E8.

## Reproduction note (added 2026-11-17)
- The evaluator now imports `evalkit` relative to its own location (`experiments/suite/common/evalkit.py`), so `python experiments/suite/E1_label_efficiency/code/eval_aef.py -h` runs from the repository root. Before this change the import used a working-directory-relative `common/` path and failed unless the script was started from the original scratch directory (issue #35). Dependencies: numpy, pandas, scikit-learn, joblib.
- Counts in this file that were re-derived from `e1_label_efficiency_aef.json` on 2026-11-17 and confirmed: CLS minus band statistics above 0 on 11/15 (n = 10) and 6/10 (max), below 0 on none; CLS minus DOFA-L above 0 on 11/15 and 3/10; CLS minus DINOv2 above 0 on 9/15 and 4/10; CLS minus coordinates above 0 on 11/13 (below on 1) and 7/8 (below on 0).
- The row-to-chip map and the Planetary Computer fetch (`fetch_embed.py`) need network access to Planetary Computer, a GPU and the Clay checkpoint (`../common/clayrt.py`); the cached chip embeddings (`shard_*.npz`, not in the repository) are the starting point for any CPU rerun (see `../E5_sota_comparators/PROTOCOL.md`, "Reproduction").
