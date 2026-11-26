# E6 — Does giving Clay four seasonal dates close its gap to AlphaEarth? (2026-11-06)

**What was measured.** E1's AEF-suite rows, splits, label draws, test caps (50 per class, 500 for regression), heads (logistic C = 1 / RidgeCV) and 100 km block bootstrap are reused unchanged. For 30,424 E1 chips, the three missing quarters of each row's valid year were fetched: one least-cloudy S2 L2A chip per quarter with E1's selection rule, giving four seasonal chips per row. 91,259 seasonal chips were fetched and embedded. Of the new chips, 94.7% have ≤5% bad SCL pixels, and ≥99.8% of the rows used have all four seasons. Arms: Clay CLS from 1 date (E1's), the mean of the 4 dates, and the concat of the 4 dates (4096-D). DOFA-L and band statistics were run with the same dates as matched controls. For the change tasks, each arm is the 4-season end-year feature minus the 4-season start-year feature. Re-embedding E1 pixels gives cosine ≥ 0.99999 to E1's stored vectors.

**At 10 labels/class, extra dates help modestly, and not only Clay.**
- Clay 4-date concat − 1 date: CI above 0 on 6 of 15 tasks, below 0 on 1 (OpenET), null on 8. The 4-date mean gives the same 6 / 1 / 8.
- Gains are small: GLanCE +0.049 BA [0.031, 0.069], LCMAP LU +0.037 [0.015, 0.058], LCMAP LC +0.027 [0.008, 0.047], ASTER +0.026 R² [0.013, 0.040], US trees +0.011 [0.006, 0.018], LUCAS LC +0.012 [0.005, 0.019].
- DOFA-L gains about as much from the same dates: 9 positive, 1 negative.
- Clay keeps its lead over the matched multi-date control: concat − stats4 is positive on 12/15 tasks and negative on none.

**At the maximum label budget, most of the gap to AEF remains.**
- Max was run on the 10 tasks where E1 ran it. Clay concat − 1 date is significant on 3 of them: GLanCE +0.060 [0.022, 0.103], LCMAP LU +0.050 [0.006, 0.100] and Canada crops fine +0.019 [0.004, 0.035]. It is null on 6 and negative on OpenET.
- The 4-date mean is significant on none.
- Pre-committed rule: gap closed = best 4-date arm minus 1 date, divided by AEF minus 1 date.
  - Per task: Africa 0.00, Canada fine 0.09, Ethiopia 0.18, GLanCE 0.42, LCMAP LC 0.15, LCMAP LU 0.25, LCMAP cover change 0.18, LCMAP use change 0.07. Oil palm gives −1.3 (its denominator is only 0.01) and OpenET −2.3.
  - Median 0.12; 0.17 without the two degenerate tasks. That is below 0.5, so per the pre-committed wording: **the gap to AEF is not explained by date count alone.**
- The controls gain at least as much as Clay. Band statistics close 0.45 of their own GLanCE gap and 0.34 of their LCMAP LU gap, against Clay's 0.42 and 0.25. The extra dates are therefore not a Clay-specific benefit.
- Change detection stays near chance with 4 + 4 dates. LCMAP cover change is 0.52 (AEF 0.78) and LCMAP use change 0.54 (AEF 0.79), with all CIs including 0.

**Negative result: OpenET.**
- Every multi-date arm collapses on OpenET: Clay concat −0.67 R² at max [−0.85, −0.51]; statistics and DOFA fall similarly.
- OpenET labels are monthly. Seasonal pooling or quarter-slotted concat erases which date matches the label month. This is a design mismatch for monthly targets, not evidence about Clay.

**What this means.** Four seasonal snapshots add a few points on land-cover/use targets for every encoder. They leave Clay well short of AEF on crops, land use and change. The remaining gap therefore points to what AEF does beyond date count: dense multi-source temporal input, per-pixel embeddings and annual summarisation inside the model. It does not point to the single-date window alone.

**Caveats.**
- These are 4 quarterly snapshots, not a dense time series. Quarters are not hemisphere-adjusted.
- AEF numbers in the section above are transcribed, not matched (best-of-3 heads, multi-source, 1,000 repeats), and the "best 4-date arm" takes the better of mean and concat per task. Both are superseded for manuscript use by the 2026-11-17 correction at the end of this file.
- n = max was not run for ASTER, Canada coarse, LUCAS LC/LU and US trees, as in E1. LUCAS at max (reviewer item v) would need about 55k new chip reads and was not run.
- The re-fitted 1-date logistic heads differ from E1's published values by ≤ 0.007 BA (BLAS/threading non-determinism). Contrasts use E6's own refits.
- Gap-closed intervals treat AEF as a constant and are very wide where AEF − Clay is small.

## Note — gap closed against E5's matched AEF values (added 2026-11-06, descriptive, CPU only; SUPERSEDED by the 2026-11-17 correction below: best-of-two pooling per task, E5 values on different rows)
Per E6 protocol, matched AEF values from E5 (same rows, heads; max budget) are used as an extra denominator. Gap closed = (best 4-date Clay − Clay 1-date)/(AEF matched − Clay 1-date), only where the gap > 0.02.

| AEF readout | task | Clay 1-date | best 4-date | AEF matched | gap | closed |
|---|---|---|---|---|---|---|
| aef_mean128 | africa_crop_mask | 0.83 | 0.83 | 0.83 | 0.0 |  |
| aef_mean128 | canada_crops_fine | 0.276 | 0.295 | 0.364 | 0.088 | 0.21 |
| aef_mean128 | descals | 0.78 | 0.767 | 0.78 | 0.0 |  |
| aef_mean128 | ethiopia_crops | 0.273 | 0.317 | 0.278 | 0.004 |  |
| aef_mean128 | glance | 0.538 | 0.598 | 0.638 | 0.1 | 0.6 |
| aef_mean128 | lcmap_lcc | 0.5 | 0.55 | 0.56 | 0.06 | 0.83 |
| aef_mean128 | lcmap_luc | 0.52 | 0.54 | 0.53 | 0.01 |  |
| aef_mean128 | openet_ensemble | 0.346 | -0.202 | -0.224 | -0.57 |  |
| aef_ov32_px | africa_crop_mask | 0.83 | 0.83 | 0.84 | 0.01 |  |
| aef_ov32_px | canada_crops_fine | 0.276 | 0.295 | 0.494 | 0.218 | 0.08 |
| aef_ov32_px | descals | 0.78 | 0.767 | 0.8 | 0.02 | -0.67 |
| aef_ov32_px | ethiopia_crops | 0.273 | 0.317 | 0.329 | 0.056 | 0.79 |
| aef_ov32_px | glance | 0.538 | 0.598 | 0.656 | 0.118 | 0.51 |
| aef_ov32_px | lcmap_lcc | 0.5 | 0.55 | 0.66 | 0.16 | 0.31 |
| aef_ov32_px | lcmap_luc | 0.52 | 0.54 | 0.62 | 0.1 | 0.2 |
| aef_ov32_px | openet_ensemble | 0.346 | -0.202 | -0.122 | -0.468 |  |

Median closed: 1.28 km mean 0.6 (3 tasks with gap>0.02); 320 m pixel 0.255 (6 tasks). Against matched AEF (320 m pixel), Clay's single-date gap at max is ≤0.10 on every task except Canada fine (0.22), LCMAP cover change (0.16) and GLanCE (0.12).

## Correction — gap closed recomputed with one readout, one comparator, matched rows (added 2026-11-17, issue #8; supersedes the 0.12, 0.255 and 0.60 medians above for any manuscript use)

**Why.** The 0.12 median divided by transcribed AEF numbers and took the better of the mean and concat arms per task; the 0.255 / 0.60 medians divided by E5 values computed on different rows and counted tasks whose gap was as small as 0.02. `code/gap_closed_matched.py` replaces them. Output: `experiments/results/e6_gap_closed_matched.json` (script, commit, arguments, versions, UTC time and digests of every selected training draw and test set are recorded in it; run on the compute host from cached E1 / E6 embeddings, 1,479 s).

**Definition (all fixed before reading the result).**
- Readout: Clay CLS, linear head (logistic C = 1 / RidgeCV), 5 label seeds. Mean-patch readouts are reported only as a secondary column in the JSON (`mean_patchreadout`).
- One-date arm: E1's single chip. Pooled arm: **mean** of the four quarterly CLS vectors, and, separately, the **concat** of the four. Never the better of the two.
- Comparator: AEF read at the row's label year; two named arms, `aef_ov32_px` (320 m pixel, the AEF column of Table 2) and `aef_mean128` (1.28 km mean). Every quantity in a (task, comparator, budget) cell comes from one row set (rows where E6 attempted the fetch and the comparator exists at the exact label year) and one set of training draws.
- Budgets: 10 per class (15 tasks) and max (only the tasks where E1's completeness rule still holds on the matched rows and E6 fetched the extra chips).
- Gap = comparator minus one-date CLS on the matched rows. A task is eligible for a ratio only if the gap is at least 0.05. Closed = (pooled minus one-date) / gap. Task-level absolute changes with paired 100 km block-bootstrap intervals are reported for every task, eligible or not.

**Result (max budget, comparator `aef_ov32_px`).**

| task | n_test | CLS 1 date | gap to AEF | mean-pool change [95% CI] | concat change [95% CI] | closed mean / concat | eligible |
|---|---|---|---|---|---|---|---|
| africa_crop_mask | 100 | 0.830 | +0.010 | -0.010 [-0.069, +0.055] | +0.000 [-0.063, +0.065] | -1.00 / 0.00 | no |
| aster_ged | 500 | not run | | | | | |
| canada_crops_coarse | 450 | not run | | | | | |
| canada_crops_fine | 1191 | 0.276 | +0.217 | +0.004 [-0.010, +0.019] | +0.018 [+0.003, +0.034] | 0.02 / 0.08 | yes |
| descals | 150 | 0.787 | +0.013 | -0.027 [-0.094, +0.042] | -0.020 [-0.086, +0.049] | -2.00 / -1.50 | no |
| ethiopia_crops | 150 | 0.273 | +0.056 | +0.021 [-0.104, +0.125] | +0.044 [-0.094, +0.165] | 0.37 / 0.79 | yes |
| glance | 550 | 0.540 | +0.116 | +0.007 [-0.033, +0.051] | +0.058 [+0.019, +0.102] | 0.06 / 0.50 | yes |
| lcmap_lc | 300 | not run | | | | | |
| lcmap_lcc | 100 | 0.500 | +0.160 | +0.050 [-0.070, +0.170] | +0.020 [-0.121, +0.161] | 0.31 / 0.13 | yes |
| lcmap_lu | 300 | not run | | | | | |
| lcmap_luc | 100 | 0.520 | +0.100 | -0.100 [-0.228, +0.029] | +0.020 [-0.087, +0.131] | -1.00 / 0.20 | yes |
| lucas_lc | 2000 | not run | | | | | |
| lucas_lu | 749 | not run | | | | | |
| openet_ensemble | 500 | 0.346 | -0.468 | -0.548 [-0.696, -0.415] | -0.667 [-0.852, -0.508] | 1.17 / 1.42 | no |
| us_trees | 1948 | not run | | | | | |

Medians over eligible tasks (`summarize_panelC.py`, section 7):

| comparator | pooling | budget | median closed | eligible tasks |
|---|---|---|---|---|
| 320 m (`aef_ov32_px`) | mean | max | 0.06 | 5 (Canada fine, Ethiopia, GLanCE, LCMAP cover change, LCMAP use change) |
| 320 m | concat | max | 0.20 | 5 |
| 1.28 km (`aef_mean128`) | mean | max | 0.07 | 3 (Canada fine, GLanCE, LCMAP cover change) |
| 1.28 km | concat | max | 0.33 | 3 |
| 320 m | mean / concat | 10 | 0.03 / 0.05 | 7 |
| 1.28 km | mean / concat | 10 | -0.00 / 0.02 | 3 |

**Reading.** Averaging four dates closes a median of 6-7% of the gap at max and essentially none at 10 labels; concatenation closes 20-33% at max (3-5 tasks, CIs for most tasks include 0). Mean pooling is significantly above one date on 0 of 8 tasks at max; concat on 2 of 8 (Canada fine, GLanCE) and mean-pool is significantly below on 1 (OpenET). At 10 labels both pooled arms have CI above 0 on 6 of 15 tasks and below on 1. The earlier "median 0.12" and "0.255" are not reproduced by a defined, matched analysis; no statement about crop tasks specifically is supported (the eligible set is mixed). The conclusion that the gap to AEF is not explained by date count alone stands.

**Africa crop mask and oil palm (starred cells).** With CLS fixed and the matched rows, four-date mean minus one date is +0.020 [-0.022, +0.062] at 10 labels and -0.010 [-0.069, +0.055] at max on Africa; +0.047 [+0.002, +0.089] at 10 and -0.027 [-0.094, +0.042] at max on oil palm. The 0.82 / 0.89 and 0.72 / 0.82 values quoted earlier combined different readouts (CLS at 10 labels, mean-patch at max); at max the CLS four-date arm is not above one date on either task. The CLS result is a gain only for oil palm at 10 labels.
