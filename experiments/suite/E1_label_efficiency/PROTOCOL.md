# E1 — Label efficiency of frozen Clay v1.5 (protocol, frozen 2026-11-05 before any downstream score)

Follows `../PROTOCOL-COMMON.md`. Anything not stated here defaults to it. Later changes are dated amendments appended at the end; this text is not edited.

**Question.** At low label budgets, how much do frozen Clay v1.5 readouts with fixed linear heads deliver on diverse public targets, against matched comparators run through the same pipeline?

## Part (a): AlphaEarth Foundations (AEF) evaluation suite

**Source.** Zenodo 16585402 v1.0.0 `evaluation.zip` (MD5 `dabf937b8b404ce399afc19fb041dcc4`), 15 CSVs. Targets, `split` (train/test), valid periods, `partition` bins (regression) and trial-group budgets are AEF's own. No relabelling. Row identity is `task|row index in the released CSV`.

**Tasks.** All 15. Eleven classification tasks, two regression tasks (`aster_ged`, `openet_ensemble`) and two change tasks (`lcmap_lcc`, `lcmap_luc`).

**Rows (declared caps).** Every train row is used, since the max-trial budget needs them. Test rows are capped:
- classification and change: at most **100 per class**, chosen by rank of `sha256("e1-testcap-v1|task|row")`;
- regression: **1,000** rows by the same hash.

Balanced accuracy is a mean of per-class recalls, so per-class capping does not change its expectation. R² is computed on the capped random subset. The cap gives 68,604 unique chips, because LCMAP LC/LU share plots and chips are deduplicated on (x, y, window). Rows whose chip cannot be fetched are dropped from both splits, and the counts are reported per task.

**Observation.** One Sentinel-2 L2A chip per row (per period for change), taken from Planetary Computer.
- Search window = the row's valid period. If that is shorter than 60 days (single-date and monthly labels), it is widened to 60 days centred on its midpoint.
- Candidates are sorted by scene `eo:cloud_cover`. For up to 4 candidates, read the chip's SCL. Take the first with ≤5% cloud, shadow, saturated or nodata pixels (SCL 0, 1, 3, 8, 9, 10); otherwise take the lowest seen.
- Chip: **128 × 128 px at 10 m (1.28 km)**, centred on the point and snapped to the item's 20 m grid. The 10 Clay bands are read as raw uint16, with no harmonisation. 20 m bands are nearest-upsampled.
- Nodata fraction > 0.5 means the row is dropped. Smaller nodata areas are filled with the per-band median, and the fraction is recorded.
- `s2:processing_baseline`, acquisition date, scene and chip cloud are recorded per chip.

**Arms (all frozen, one forward pass, no tuning).**
- `clay_cls` (primary readout), `clay_mean` (mean patch token), `clay_center` (mean of the 2 × 2 central patch tokens; secondary). Clay receives the real acquisition time and the point's lat/lon.
- `dinov2`: ViT-L/14-reg4 pooled output on display RGB (per-chip 2–98% stretch), 224 px. The timm model is rebuilt with `img_size=224`, with position embeddings interpolated at load.
- `dofa`: DOFA-L/16 (torchgeo `DOFALarge16_Weights.DOFA_MAE`), all 10 bands with their wavelengths, z-scored with Clay's S2 mean/std, bilinear to 224 px, `forward_features` output.
- `stats`: per-band mean and std of the raw chip (20-D).
- `coords`: [sin lat, cos lat, sin lon, cos lon] (AEF's XY control). Not used for change tasks, as AEF also omits it.
- Change tasks: every arm's feature is the difference of embeddings, after minus before.

**Budgets.** These are AEF's: n = 1, 10 and max samples per class (or per `partition` bin for regression). Max values are africa 200, aster 200, canada_coarse 75, canada_fine 68, descals 200, ethiopia 49, glance 300, lcmap_lc/lu/lcc 300, lcmap_luc 150, lucas_lc/lu 300, openet 300, us_trees 300.
- Selection uses `evalkit.label_budget` (nested, hash-ranked) on available train rows, with seeds 0–4.
- AEF uses 1,000 / 500 / k-fold repeats; we use 5 seeds as `PROTOCOL-COMMON` requires.

**Heads.**
- Primary: `evalkit.clf_head` (StandardScaler + LogisticRegression C = 1) for classification and change; `evalkit.reg_head` (StandardScaler + RidgeCV, train-only inner CV) for regression.
- Secondary: kNN with k = min(3, n_train), L2 on raw features (AEF's kNN-3 transfer).
- No best-of-transfer selection.

**Metrics.** These are AEF's: balanced accuracy (classification and change) and R² (regression). Macro-F1 is also recorded.

**Uncertainty.** 1,000 block-bootstrap resamples of the test rows over `evalkit.hblock` 100 km blocks (seed 1729), applied to the seed-mean. Contrasts are paired.

**Primary endpoint.** For each task: `clay_cls − stats` at **n = 10**, linear head, with paired 95% block-bootstrap CI. Summary: the number of tasks with a CI above 0, below 0, or including 0.

**Key secondary endpoints.**
- `clay_cls − dinov2` and `clay_cls − dofa` at n = 10.
- All arms at all budgets.
- `clay_cls − coords`.
- Readout sensitivity: cls vs mean vs center.
- The kNN-3 head.
- Processing-baseline distribution.

**AEF reference numbers (transcribed, not re-measured).**
- Max-trial values printed on the bars of AEF arXiv 2507.22291v2 Figure 1A: AEF on top, next-best below, best transfer method per evaluation, BA for classification and MAE for regression. Clay appears there only where it was next-best (LUCAS LU, LUCAS LC).
- Exact numbers stated in the text: ASTER R², OpenET R², change BA.
- Bars without printed numbers (Figures S16, S17, S19) are not digitised.
- AEF's Clay recipe (S19.3.3) is 768-D: patch tokens averaged over S1/S2/Landsat frames in the valid period, 256 px, training-split normalisation, checkpoint not bound. It is not this v1.5 1024-D single-chip configuration.

**Not the AEF protocol.** Differences that will be stated with every comparison:
- AEF embeds a multi-source, multi-date summary over the valid period. We use **one least-cloudy S2 date**, so seasonal and phenology information is lost, which matters most for crops and ET.
- Our chip context is 1.28 km, and Clay gets image-level CLS/mean readouts rather than AEF's per-pixel token at the point.
- Heads: LR C = 1 / RidgeCV, against AEF's OLS ridge-classifier and kNN.
- 5 seeds, against 500–1,000 repeats.
- Capped test sets.
- 95% block-bootstrap CIs, against AEF's 1σ.

## Part (b): EuroSAT pilot overlap and comparators

**Frozen inputs.** EuroSAT spatial-block manifest (`results/prepared/manifest.csv`), Clay embeddings (`results/features/embeddings.npz`), spectral features, and `evaluate_embeddings.py` (protocol v1.1). All 25,648 eligible IDs and the same split are used.

**Overlap.** Clay training locations are the MGRS tiles in `mgrs_sample_v02.fgb` (https://clay-mgrs-samples.s3.amazonaws.com/). A patch overlaps if its **centroid lies within** a tile polygon (primary); footprint intersection is also reported. We report the fraction of test (and train/val) patches that overlap. This is spatial overlap only: Clay's training chips were sampled in time and within tiles, so it does not establish image identity.

**Re-score.** Overlapping test patches get `split = excluded` in a copy of the manifest, and the embeddings NPZ is subset to the remaining eligible IDs. The unchanged script is rerun. The primary endpoint stays as before: CLS − spectral20 + xy2, macro-F1 at 50 labels/class, 5 seeds, paired 1,000-resample 100 km block bootstrap (seed 1729). The script's own viability check (≥ 20 test blocks) applies; if it fails, we report that and do not relax it.

**Comparators.** DINOv2-L/14-reg4 (display RGB from B04/B03/B02, stretched as above, 64 → 224 px bilinear) and DOFA-L (10 bands, Clay S2 z-score, 64 → 224 px), extracted on the same IDs from EuroSAT_MS.zip (Zenodo 7711810, MD5 `091174add3c8e680a49244acf185b9f0`, checked on download).
- Each runs through the unchanged script by placing its 1,024-D vector in the `cls` slot. DINOv2 patch-mean goes in `mean`; for DOFA, `mean` duplicates `cls` and is ignored.
- Paired contrasts (Clay CLS − DINOv2, Clay CLS − DOFA, at 50/class, macro-F1) are computed from the scripts' saved per-block confusion matrices with the script's own multiplicity matrix (`default_rng(1729)`, same test blocks).
- A convergence failure is reported, not worked around.

## Outputs
`results.json`, `RESULTS.md`, figure `figure/e1_label_efficiency.{png,pdf}` (label budget vs metric per task and arm), `manifest/` (requests, chip metadata, row → chip map), `code/`.


## Amendment 1 — 2026-11-05 (before any downstream score)
After 2,608 chips, Planetary Computer's STAC API began rate-limiting requests (`Request rate limited or blocked`; 47 rows errored), and the fetch stalled. Changes:
1. A shared adaptive STAC rate limiter (3 req/s start; halve and pause 60 s on rate-limit; +0.5 req/s per quiet 5 min, max 6). Rows that errored are retried on resume.
2. Fetch order is by priority. Priority 0: all capped test rows, plus train rows within the top ⌈1.1·10⌉+2 per class of `label_budget` for any seed 0–4. Priority 1: train rows within the top ⌈1.1·n_max⌉+2 for any seed. Within each priority, order is by sha256 hash. This makes the n = 1 / n = 10 analyses complete first.
3. Train rows that no budget can select for any seed (rank > ⌈1.1·n_max⌉+2 in every seed; 2,314 chips, all in classes with more train rows than n_max: Canada coarse/fine, Ethiopia) are not fetched. `label_budget` ranks only available rows, so selections are unchanged except where more than ~10% of a class's needed rows fail to fetch.
4. If the fetch cannot finish in the job's time window, the analysis covers what was fetched. Per task and budget, the coverage and any shortfall in n_max rows per class are reported, and those results are marked provisional.
Endpoints, arms, heads, metrics and caps are unchanged.


## Amendment 2 — 2026-11-05 12:05 UTC (before any downstream score)
Planetary Computer throttling is shared with a sibling group fetching from the same host. Cumulative throughput fell to about 1.4 chips/s, which puts the full 66k-chip set beyond 12 h. The scope is reduced as follows, decided only from fetch throughput, never from any score:
1. **Test set (all tasks):** per class, the first 50 rows of the frozen `e1-testcap-v1` hash ranking (a nested subset of the cap-100 set); for regression, the first 500. This replaces the 100 / 1,000 caps for scoring.
2. **Tier 0 fetch:** those test rows, plus exactly the union over seeds 0–4 of `label_budget(k=10)` train rows (17,850 chips). It is fetched first. Tier 1 holds all other needed rows (max-budget train rows, test rows 51–100), fetched in hash order until a fixed job limit of 5 h wall-clock from restart. The analysis then uses what exists.
3. **Budgets:** n = 1 and n = 10 are reported for every task. **n = max** is reported only for tasks whose max-budget selections are complete for every seed: each class contributes min(n_max, #train rows in class) available rows, so the selection is identical to full data. Otherwise it is listed as not run, with its coverage stated.
4. Rows of tier 0 whose chip fails stay missing. Their count is reported per task.
Endpoints (primary at n = 10), arms, heads and metrics are unchanged.


## Amendment 3 — 2026-11-05 13:15 UTC (before any AEF-suite score; after the EuroSAT re-scores)
1. **Lat/lon order.** Clay v1.0/v1.5 were trained with [sin lon, cos lon, sin lat, cos lat]; see the PROTOCOL-COMMON amendment of 2026-11-05. The earlier fetch embedded Clay in the documented order [sin lat, …]. Those Clay vectors are discarded unscored.
   - The AEF fetch is restarted from scratch into a new output directory with the updated `clayrt` (common_v2).
   - `clay_cls`, `clay_mean` and `clay_center` are now embedded in the **training order** (primary).
   - The same chips are also embedded in the documented order as a named sensitivity arm (`*_doc`): `clay_cls_doc − clay_cls` at n = 10, linear head, per task.
   - DINOv2, DOFA, stats and coords are unaffected (no metadata).
   - Pixel chips (uint16) are now kept, compressed, so metadata arms can be re-run without re-fetching.
2. **EuroSAT.** The frozen pilot embedded Clay in the documented order. The part (b) numbers computed so far are labelled **documented-order (out-of-distribution metadata)** and kept. The 25,648 pilot patches are re-embedded with Clay in the training order and re-scored with the same unchanged script: the primary endpoint on full and overlap-excluded test sets, plus contrasts vs DINOv2 and DOFA. Both are reported. A documented-order re-embedding is checked against the pilot's stored vectors to confirm the pipeline.
3. **Fetch robustness.** Per-request timeout of 180 s, GDAL HTTP timeout of 30 s and bounded task submission, after one hung HTTP read blocked the earlier run. Hard stop at 4.8 h of fetch wall-clock. Amendment 2 rules apply unchanged.


## Amendment 4 — 2026-11-05 14:45 UTC (before any AEF-suite score)
The tier-1 fetch order is changed from a cross-task hash order to **task by task, in ascending order of the chips each task still needs for a complete n = max**: ethiopia, africa, lcmap_luc, descals, lcmap_lcc, lcmap_lc, lcmap_lu, aster, canada_coarse, openet, glance, canada_fine, lucas_lu, us_trees, lucas_lc. Test rows 51–100 come last. The reason is that a hash order spread across tasks would likely leave no task with a complete max budget by the hard stop. The decision used chip counts only. The tier-0 order is unchanged. The hard stop is now 4 h after the 14:00 UTC restart (Ollama had reloaded and filled the GPU, and was unloaded per host rules).
