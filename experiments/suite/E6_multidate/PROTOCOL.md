# E6 — Multi-date Clay on the AEF suite (protocol, frozen 2026-11-05 before any E6 score)

Follows `../PROTOCOL-COMMON.md` and reuses E1 (`../E1_label_efficiency/PROTOCOL.md`, incl. amendments 1–4) for everything not stated here: same rows, AEF train/test split, test caps (E1 amendment 2: 50/class, 500 for regression), nested hash-ranked label budgets (seeds 0–4), heads, metrics and 1,000-resample 100 km block bootstrap (seed 1729), with all arms of a task sharing resamples (paired). Later changes are dated amendments appended at the end.

**Question.** Is Clay's gap to AlphaEarth (AEF) on annual/phenology targets due to E1's single-date observation, rather than the model? We give Clay four seasonal dates and measure how much of the gap closes.

## Observation
- Base chip: E1's chip for the row (one least-cloudy S2 L2A chip in the row's search window), with E1's stored embeddings.
- **Valid year.** For rows whose E1 window is annual (365/366 d): that window, [t0, t0 + 12 months). For rows with a 60-day window (canada_crops_coarse/fine, ethiopia_crops, lucas_lc, openet_ensemble): the calendar year containing the window's midpoint.
- **Seasons.** Four consecutive 3-month quarters of the valid year (for calendar years: JFM, AMJ, JAS, OND; not hemisphere-adjusted). E1's chip is assigned to its quarter by month offset from the valid-year start (mod 12; 149 of 30,424 E1 chips lie just outside the valid year of a 60-day-window row and are assigned by month). For each of the other three quarters, one additional chip is fetched with **E1's exact selection rule** (candidates by scene `eo:cloud_cover`; read SCL for up to 4; first with ≤5% SCL ∈ {0,1,3,8,9,10}, else the lowest seen), same chip geometry (128 px, 10 m, snapped to the 20 m grid), raw uint16, no harmonisation, same nodata rules. One STAC search per chip over the valid year (fields extension) supplies all candidates.
- Embedding per seasonal chip: Clay v1.5 CLS and mean patch token with the chip's **real acquisition date** and the point's lat/lon in Clay's **training order**; DOFA-L (as E1); per-band mean/std (20-D). Pixels are not kept (disk limit).
- Consistency check: a sample of E1's kept pixel chips is re-embedded with this runtime and compared with E1's stored Clay CLS (cosine), reported.

## Scope (declared before any score; driven by Planetary Computer throughput only)
- **Tier A (fetched first):** every E1 chip needed by E1's capped test sets (all 15 tasks) and by the union over seeds 0–4 of the n = 10 budget, i.e. the rows used by E1's n = 1 and n = 10 analyses: 17,846 chips.
- **Tier B:** the extra train chips needed for n = max in the 10 tasks where E1 could run n = max (africa, canada_fine, descals, ethiopia, glance, lcmap_lc, lcmap_lu, lcmap_lcc, lcmap_luc, openet): 12,578 chips, fetched task by task in ascending order of chips needed (ethiopia, africa, lcmap_luc, descals, lcmap_lcc, lcmap_lc, lcmap_lu, openet, glance, canada_fine). n = max is not run for aster, canada_coarse, lucas_lc/lu, us_trees (E1 could not).
- Hard stop 6 h of fetch wall-clock per job; resumable. A (task, budget) is scored only if E6 fetch was attempted for every row in its test set and in all its seed selections; otherwise it is listed as not run, with coverage.
- **Missing season** (no scene in the quarter, SCL unreadable, nodata > 0.5, or a fetch error that persists after one retry pass): for mean arms, average the available seasons; for concat arms, fill the missing slot with the E1 chip's vector. Per task, the fraction of rows with all 4 seasons is reported.

## Arms (all frozen, linear heads as E1; no tuning)
- (a) `clay1` — E1's single-date Clay CLS (E1's stored vector; identical to E1's `clay_cls`).
- (b) `clay4_mean` — mean of the 4 seasonal Clay CLS (1024-D).
- (c) `clay4_cat` — concat of the 4 seasonal CLS in quarter order (4096-D).
- (d) `stats4_cat` — concat of the 4 seasonal per-band mean/std (80-D), the matched non-learned control; `stats1` = E1's stats.
- (e) `dofa4_mean` — mean of 4 seasonal DOFA-L vectors; `dofa1` = E1's DOFA.
- Secondary readout: `claymean4_mean` (mean-patch readout, 4-date mean) vs E1 `clay_mean`.
- Change tasks (lcmap_lcc, lcmap_luc): each arm's feature is (arm at end year) − (arm at start year), each side built from its own 4 seasons; for `clay4_cat`/`stats4_cat` the slot-wise difference.

## Endpoints
- **Primary:** per task, `clay4_mean − clay1` and `clay4_cat − clay1` at n = 10 and n = max, linear head, paired 95% block-bootstrap CI (balanced accuracy; R² for aster/openet). Summary: tasks with CI above 0 / below 0 / including 0.
- **Gap closed** (at n = max, where AEF publishes a number: Fig. 1A BA for classification; text R² for aster/openet; text BA for change): (arm − clay1) / (AEF − clay1), with the bootstrap percentile interval of the numerator scaled by the same denominator (AEF treated as a constant). Not a matched protocol (AEF: best of 3 transfer methods, multi-source, 1,000 repeats). If E5's matched AEF numbers exist at scoring time they are reported as an extra denominator.
- **Secondary:** `clay4_cat − stats4_cat` and `clay4_mean − dofa4_mean` (is a multi-date advantage specific to Clay?); `stats4_cat − stats1` and `dofa4_mean − dofa1` (how much the extra dates help a control/comparator); n = 1; kNN-3 head; seasonal coverage and processing-baseline mix of the new chips.


## Amendment 1 — 2026-11-05 22:40 UTC (before any E6 score; fetch still running)
Reviewer-panel requirements, plus two implementation notes:
1. **Gap closed for controls.** Besides Clay, the gap-closed fraction is also reported for the matched control and comparator, each against its own single-date baseline: (stats4_cat − stats1)/(AEF − stats1) and (dofa4_mean − dofa1)/(AEF − dofa1). Raw deltas stats4_cat − stats1 and dofa4_mean − dofa1 are shown beside Clay's in the same table. A multi-date gain is called Clay-specific only if `clay4_cat − stats4_cat` (or `clay4_mean − dofa4_mean`) is also positive, with its CI above 0.
2. **Season availability.** Per task and split, report the fraction of rows with all four seasons actually fetched, as opposed to filled with the E1 vector (already in the protocol; now a required table column).
3. **LCMAP change.** The change arms are already 4 + 4-season differences: (end-year 4-season feature) − (start-year 4-season feature), for both the mean (`clay4_mean`) and the slot-wise concat (`clay4_cat`). No new arm.
4. **Pre-committed wording.** Phenology/annual tasks with n = max are africa_crop_mask, canada_crops_fine, ethiopia_crops, descals, glance, lcmap_lc, lcmap_lu, lcmap_lcc, lcmap_luc and openet_ensemble. For each, gap closed = max(clay4_mean, clay4_cat) − clay1, divided by (AEF − clay1). If the median of these is below 0.5, the conclusion is worded "**the gap to AEF is not explained by date count alone**". If it is at least 0.5, the wording is "most of the gap is explained by the observation window". Per-task values are shown either way.
5. **LUCAS LC/LU at n = max: not run.** It would need 13,885 rows that E1 never fetched (10,136 LC + 3,749 LU). Each needs a new E1-rule single-date chip plus 3 seasonal chips, about 55k chip reads (roughly 4–6 h more of Planetary Computer time at the current ~2 chips/s). This does not meet the "if cheap" bar, so it is declared not run.
6. Implementation: 92 chips whose valid year starts on day 29–31 errored in quarter arithmetic (month-end overflow). The fixed code clamps to the month's last day, and they are retried in the protocol's retry pass. Chips that still fail after that one retry pass count as missing seasons (`--errors-as-missing`).

## Amendment — 2026-11-17 (audit fix, issue #8; written after the E6 scores, so post hoc and labelled as such)
The "gap closed" statistic is redefined and recomputed by `code/gap_closed_matched.py` (results: `experiments/results/e6_gap_closed_matched.json`). The pre-committed rule above (best 4-date arm over transcribed AEF) is kept in RESULTS.md as the original result but is not used for manuscript claims. The new definition fixes: readout (Clay CLS, linear head), pooling (mean and concat reported separately, never best-of), comparator (AEF `aef_ov32_px` or `aef_mean128` at the exact label year, on the same rows and training draws as the Clay arms), budgets (10 and max; max only where E1's completeness rule holds on the matched rows), eligibility (gap >= 0.05; thresholds 0.02 and 0.10 are in the summary script), and reports task-level absolute changes with paired block-bootstrap intervals for every task.
