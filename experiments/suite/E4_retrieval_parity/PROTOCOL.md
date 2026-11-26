# E4 — Retrieval at a review budget, and parity of released embeddings
Frozen 2026-11-05, before any retrieval score or parity cosine was computed. It follows `../PROTOCOL-COMMON.md`. Deviations from that protocol are listed under each part.

## Part A — PV retrieval on full Sentinel-2 tiles

**Reference.** USGS USPVDB v4.0 (`uspvdb_v4_0_20260414.geojson`, downloaded from energy.usgs.gov, because eerscmap.usgs.gov did not answer from the host). Zip SHA-256: `ce5180752d80008c0ec99d67aa831564faf9c69dec859615722fc51f768fab1f`; 6,611 polygons.
- **Positive mask:** polygons with `p_year ≤ 2022`, which existed before the summer-2023 image.
- **Ambiguous mask:** polygons with `p_year ≥ 2023` or a missing year.

**Test regions.** Fixed by a rule that uses labels and scene metadata only (`select_regions.py` → `region_candidates.json`):
- In each of CA, TX and NC, take the S2 MGRS tile whose full-coverage scene (< 1% nodata) contains the largest area of `p_year ≤ 2022` PV (counted by site centroid).
- Use that tile's least-cloudy scene from Jun–Aug 2023 (Planetary Computer `sentinel-2-l2a`).
- Selected: CA `11SLU` (S2B 2023-07-27, baseline 05.10), TX `15STT` (S2B 2023-06-30, 05.10), NC `18STE` (S2A 2023-08-23, 05.10).

The CA tile sits at the southern end of the Central Valley and in the Mojave, not in the central Valley proper. It was chosen by the rule.

**Gallery.**
- All chips that tile the full 10,980 × 10,980 px tile from its top-left corner: 64 × 64 px (640 m; 171² = 29,241 chips) and 256 × 256 px (2.56 km; 42² = 1,764 chips).
- Pixels are raw L2A values with no harmonisation. 20 m bands are upsampled to 10 m by nearest neighbour, as stacchip does.
- Chips with any 0 (nodata) pixel are dropped.
- Chips whose ambiguous-mask fraction is ≥ 10% are excluded from the gallery and from the query pools.
- A chip is **positive** if ≥ 10% of its area is in the positive mask, rasterised at 10 m (pixel-centre rule). Everything else is negative, including chips with 0–10% PV.

**Queries (leave-one-region-out).**
- For test region R, query positives come only from the positive chips of the other two regions, at the same chip size.
- Query negatives come from chips in the other two regions with zero PV and zero ambiguous fraction.
- For seed s ∈ {0,…,4}, pools are ranked by sha256(`claypaper-e4-qpos|s|chip_id`). Positives are nested prefixes, k ∈ {1, 5, 20}. Negatives are the first m = 20 by sha256(`claypaper-e4-qneg|s|chip_id`).
- If a pool holds fewer than k positives, all of them are used, and the shortfall is reported.

**Scoring (no training, no tuning).**
- Vectors are L2-normalised.
- Mode `pos`: cosine to the mean of the query positives.
- Mode `pos_minus_neg`: cosine to (mean of positives − mean of negatives).

**Arms.**
1. `clay_cls`: Clay v1.5 class token. Real chip-centre lat/lon and scene date; platform mean/std from `metadata.yaml`.
2. `clay_mean`: mean of Clay patch tokens.
3. `dinov2_rgb`: DINOv2 ViT-L/14-reg4, pooled output, on per-chip 2–98% stretched RGB resized to 224.
4. `spectral_stats`: per-band mean and std of raw values (20-D), z-scored with statistics pooled over all three galleries (label-free), then cosine.
5. `random`: uniform scores, one draw per seed.

**Metrics.**
- Precision and recall at review budgets K ∈ {10, 100, 1,000} chips (the top K of the ranked gallery), plus average precision (AP), per region.
- Each point is the mean over the 5 seeds.
- "Pooled" is the unweighted mean over the 3 regions.
- At 256 px the gallery has about 1,700 chips per region, so K = 1,000 covers more than half of it. That budget is reported but not interpreted.

**Uncertainty.**
- 1,000 block-bootstrap resamples (seed 1729) of 10 km × 10 km UTM blocks within each tile; regions are resampled independently.
- Metrics are recomputed on the resampled gallery with the query vectors fixed.
- Paired contrasts share resamples.

**Primary endpoint.** Pooled AP at 64 px, k = 5, mode `pos`. Two contrasts: `clay_cls − dinov2_rgb` and `clay_cls − spectral_stats`, each with a 95% CI.

**Secondary.**
- All other k, both modes and every budget.
- `clay_cls − clay_mean`.
- The chip-size effect: each arm's AP and P@100 at 256 px versus 64 px. The two sizes have different galleries and prevalence, so the chip-size comparison is descriptive and is also reported relative to the random arm (prevalence).

**Deviations from the common protocol.**
- Retrieval uses no head or label budget.
- Blocks are 10 km, not 100 km, because each test region is a single ~110 km tile.
- Queries come from different states, which gives stronger-than-block spatial separation.

## Part B — parity of released embeddings

**Reference pipeline.** The released inference code: `Clay-foundation/model`, Docker checkout `ceecb6138705cb28a5f4d3f61f22b19a2f625edb`, `embeddings/{utils,all-naip,all-sentinel}.py`, stacchip 0.1.35, checkpoint `v1.5.0-no-mrl-dinov2/mae_v1.5.0_epoch-07_val-loss-0.1718.ckpt` (released as `v1.5/clay-v1.5.ckpt`).
- The `Encoder` class at that commit is byte-identical to our pinned `f14e698`.
- We stream-hash the public checkpoint and compare it with our pinned SHA-256.

The documented preprocessing is:
- stacchip 256 × 256 chip grid from the raster origin;
- per-band `metadata.yaml` mean/std;
- time = (sin, cos) of ISO-week·2π/52, then (sin, cos) of hour·2π/24. NAIP uses hour 12; S2 uses the item datetime;
- lat/lon from `normalize_latlon(*bbox.centroid.coords[0])` on a WGS84 box built with `always_xy`. That call passes (lon, lat) into a (lat, lon) signature. We reproduce this as written;
- the `gsd` from the path or 10;
- class token only.

**Arms (same pixels, local encoder, released checkpoint).**
- `as_shipped`: the code as written, including the lat/lon order. **This is primary.**
- `latlon_fixed`: lat/lon in model order.
- `no_meta`: time and lat/lon set to zero.
- `no_norm`: raw pixels without mean/std, the reported NAIP v1 bug.

**Sets and samples.** Selection is hash-ranked and fixed before encoding.
1. **NAIP v1.5 rev2** (`clay-v1-5-naip-2`).
   - 8 files: states hash-ranked; one year, resolution, quad and file per state by hash. 125 rows per file by hash, ≈ 1,000 vectors.
   - **Pixel substitution:** `naip-analytic` is requester-pays, so pixels come from Planetary Computer `naip`, which holds the same USDA DOQQs (item `{state}_{stem}`).
   - Provenance check: the released polygon must map onto the PC raster's 256 px grid. We report the residual in pixels and the raster-shape match.
   - The same footprints from the superseded `clay-v1-5-naip` are compared with `no_norm`.
2. **S2 v1.5 (deprecated `clay-v1-5-sentinel2`).** Up to 6 files from US MGRS zones, ≤ 300 rows. The item JSON and COGs come from public `sentinel-cogs`, as in the pipeline. Chips are those passing the pipeline's SCL cloud ≤ 0.1 and nodata ≤ 0.01 filters, matched by footprint.

**Metrics.**
- Cosine (released, recomputed): median, 5th percentile and the fraction > 0.99, per set and arm.
- The released/recomputed norm ratio.
- The vector dimension.
- The footprint residual.

If provenance cannot be reconstructed for a set, that is reported as a finding.

## Amendment 1 (2026-11-05, before any Part B cosine was computed)
Code reading of stacchip 0.1.35 `ChipIndexer._get_trsf_or_shape` found that when `proj:transform` is only on assets, it keeps the asset with the largest first element, i.e. the coarsest pixel size. For a multi-resolution S2 item that is the 60 m bands. Chip footprints would then be written as 15.36 km boxes, while the pixels are 256 px windows on the 10 m grid.
- Set 2 recovers chip indices under both hypotheses (10 m and 60 m footprint scale) and reports which one gives parity.

The primary arms and metrics are unchanged.

## Amendment 2 (2026-11-05, POST HOC, written after the Part A primary scores were seen)
At 64 px, Clay CLS ranked far below DINOv2-RGB. `pos_minus_neg` scoring, whose negatives come from the query regions, roughly tripled Clay's AP. That suggests Clay's lat/lon and time encodings add a region-specific offset that hurts cross-state queries.
- One **exploratory** arm is added: `clay_cls_nometa`, the Clay class token with time and lat/lon set to zero (the training convention for unknown metadata), on the same chips, queries and metrics.
- It is reported separately as post hoc and does not replace the pre-registered arms or the primary endpoint.
- The DINOv2 input is 224 px, built with timm `img_size=224`, as frozen above.

## Amendment 3 — lat/lon order (2026-11-05, written before any training-order score was computed)
The Clay v1.0/v1.5 training pipeline encoded location as [sin lon, cos lon, sin lat, cos lat] (stacchip `prechip.py` unpacks `normalize_latlon` in swapped order; see the dated amendment in `../PROTOCOL-COMMON.md`). The released embedding products use the same order; Part B confirms this exactly for NAIP rev2 and S2 v1.5. All Part A Clay metadata-ON arms above (`clay_cls`, `clay_mean`) used the *documented* order [sin lat, cos lat, sin lon, cos lon], which is out of distribution for the model. They are kept unchanged and are hereafter labelled **documented order (out-of-distribution metadata)**.

Added, on the same scenes, chip grid, chip ids (asserted equal to the cache), queries, seeds and bootstrap resamples:
- `clay_cls_train` and `clay_mean_train`: Clay class token and mean patch with real time and lat/lon in **training order**, via the updated `common/clayrt.py` (`latlon_order="training"`).
- Re-scored endpoints: the pre-registered primary (pooled AP, 64 px, k = 5, `pos`; `clay_cls_train − dinov2_rgb` and `clay_cls_train − spectral_stats`), the same at 256 px, and the metadata contrasts `clay_cls_train − clay_cls_nometa` and `clay_cls_train − clay_cls` (training minus documented order).
- `clay_cls_nometa` (metadata OFF) is unaffected and reused. DINOv2, spectral and random arms are unchanged.
- Results are written to new keys (`partA_trainorder_amendment`) in `results.json`; the original keys are not modified.
