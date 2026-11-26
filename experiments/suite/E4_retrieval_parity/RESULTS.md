# E4 — Retrieval at a review budget, and parity of released embeddings (results)
Protocol: `PROTOCOL.md`, frozen before scores. Amendment 1 was added before any Part B cosine. Amendment 2 is **post hoc** and adds one exploratory arm. CIs are 95% from 1,000 block-bootstrap resamples (10 km blocks, seed 1729) of the seed-mean over 5 query seeds.

**Part A — finding PV sites with few examples.**
- Reference: USPVDB v4.0 (sites built ≤ 2022). Galleries: three full Sentinel-2 tiles from summer 2023, CA 11SLU, TX 15STT and NC 18STE, each about 29k chips of 64 px (prevalence 0.5–1.6%) and about 1.7k chips of 256 px.
- Queries are k positives (and 20 negatives) from the *other two* states, scored by cosine.

At 64 px (640 m), the Clay v1.5 class token is clearly worse than DINOv2-L on RGB. Pooled AP at k = 5:

| Arm | AP |
|---|---|
| Clay v1.5 class token | 0.038 [0.018, 0.092] |
| DINOv2-L (RGB) | 0.216 [0.102, 0.307] |
| Band statistics | 0.016 [0.010, 0.027] |
| Random | 0.009 [0.006, 0.014] |

- Clay − DINOv2 = −0.178 [−0.261, −0.070]. Clay − band statistics = +0.022 [0.006, 0.072].
- At a review budget of 100 chips, Clay's precision is 0.053 [0.019, 0.096] against DINOv2's 0.313 [0.155, 0.430]. Recall at 1,000 chips is 0.21 against 0.39.
- The Clay mean-patch readout is close to random.

At 256 px (2.56 km), Clay rises to AP 0.268 [0.107, 0.638] and DINOv2 to 0.323 [0.155, 0.513]. The difference, −0.055 [−0.245, 0.266], is not resolved: there are only 6–39 positives per tile. Clay − band statistics is +0.246 [0.082, 0.608].
- The chip-size effect is large for Clay (AP lift over random: ≈ 4× at 64 px, ≈ 14× at 256 px) and small for DINOv2 (≈ 24× and ≈ 17×).
- Scoring against mean(pos) − mean(neg) helps both models (64 px: Clay 0.105, DINOv2 0.267) but does not change the ordering.

Post-hoc check: zeroing Clay's time and lat/lon inputs did not close the 64 px gap (Δ AP = −0.008 [−0.038, 0.023]). The gap is not a location-encoding artefact. It is most plausibly the short 8 × 8-token context at 64 px, relative to Clay's 256 px training chips.

**Part B — do released vectors match the encoder?** Each released vector was re-encoded locally with the released checkpoint. The public inference checkpoint has SHA-256 21432069…98d0, identical to the released one. Preprocessing followed the released inference code (`all-of-naip` branch @ ceecb61; its Encoder class is byte-identical to the pinned source). See `tableB_parity.md`.
- **NAIP v1.5 rev2:** 1,000 vectors, 8 states and 8 files, pixels from Planetary Computer's copy of the same DOQQs. 100% have cosine > 0.99 (median 1.0000). Footprints sit on the stacchip grid to within 4e−9 px.
  - Parity holds only with the pipeline's **lat/lon transposition**: it feeds (lon, lat) into Clay's (lat, lon) slots. Re-encoded in the model's convention, the median is 0.988 and only 43% exceed 0.99.
  - The superseded v1 release equals our no-normalisation re-encoding exactly (median cosine 1.0000), which confirms the reported bug. v1 against rev2 has median cosine 0.04.
- **Sentinel-2 v1.5 (deprecated `clay-v1-5-sentinel2`):** vectors reproduce exactly (300/300 > 0.99), but the metadata carries two defects.
  - Every footprint is 15.36 km wide instead of 2.56 km. stacchip 0.1.35 takes `proj:transform` from the 60 m bands. Read at face value, the matched cosine is 0.17.
  - Almost every chip is stored twice (e.g. 3,478 rows for 1,764 chips; the duplicate pairs have cosine 1.0). The cause is an `if start:` test that treats start = 0 as false, so the first batch embeds the whole scene.
- **LGND S2 1,024-D (aggregated):** not reconstructable from public documentation. The best guess (a 256 px crop centred on the MajorTOM cell, raw L2A, model-order lat/lon) gives median cosine 0.961, with 7% > 0.99 (n = 300). Harmonised inputs fit worse (0.73).
- **LGND 256-D @ 1,280 m:** these match the first 256 dimensions of a 128 px class token (median 0.970). v1.5 was trained without Matryoshka (MRL), so this truncation is undocumented and its quality is unvalidated.

**Caveats.**
- There are three test tiles, and the TX gallery has few PV sites (5). 256 px estimates rest on 60 positives in total.
- USPVDB polygons are rasterised at 10 m. PV built in 2023+ is excluded, not labelled.
- NAIP pixels come from Planetary Computer, not `naip-analytic`. The exact match argues they are the same data.
- Clay's pretraining sample may include these tiles. This was not checked.
- The Part B samples are hash-selected, not archive-wide estimates.

## Amendment — lat/lon order (2026-11-05)
Clay v1.0/v1.5 were trained with longitude in the first two location slots, [sin lon, cos lon, sin lat, cos lat], and the released embedding products use the same order (Part B). The Part A Clay arms above used the documented order [sin lat, cos lat, sin lon, cos lon], which is out of distribution for the model. They are kept and labelled as such.

The Clay class token and mean patch were re-embedded in **training order** on the same chips, queries and resamples. The original numbers reproduced exactly. All values are pooled AP at k = 5, `pos` mode.

| Arm | AP, 64 px | AP, 256 px |
|---|---|---|
| Clay CLS, documented order | 0.038 [0.018, 0.092] | 0.268 [0.107, 0.638] |
| Clay CLS, training order | 0.046 [0.022, 0.107] | 0.272 [0.111, 0.632] |
| Clay CLS, metadata off | 0.047 [0.022, 0.115] | 0.261 [0.107, 0.619] |
| Clay mean, documented / training order | 0.013 / 0.013 | 0.021 / 0.021 |
| Clay CLS − DINOv2, documented order | −0.178 [−0.261, −0.070] | −0.055 [−0.245, 0.266] |
| Clay CLS − DINOv2, training order | −0.170 [−0.255, −0.062] | −0.051 [−0.237, 0.264] |
| Clay CLS − band statistics, documented order | +0.022 [0.006, 0.072] | +0.246 [0.082, 0.608] |
| Clay CLS − band statistics, training order | +0.030 [0.010, 0.083] | +0.250 [0.087, 0.603] |
| Training-order CLS − metadata off | −0.001 [−0.025, 0.011] | +0.011 [0.001, 0.043] |
| Training − documented order | +0.008 [−0.025, 0.029] | +0.004 [−0.013, 0.019] |

At 64 px with training order, precision at a 100-chip budget is 0.069 [0.025, 0.119] (documented order 0.053; DINOv2 0.313).

The lat/lon order changes Part A by less than 0.01 AP, and no conclusion changes. At 64 px Clay remains far below DINOv2-RGB. At 256 px the Clay–DINOv2 gap remains unresolved. Metadata, in either order, contributes essentially nothing to cross-state PV retrieval.

Part B is unaffected. There, the "as_shipped" arm equals the training order, which explains why the released vectors reproduce exactly only with it.

## Audit reruns and artifacts (2026-11-17)
**Part A code and output are now in the repository.** `partA.py` is the script that produced the numbers above (stages `embed`, `nometa`, `train`, `score`; usage in its docstring), `regions.json` lists the three Sentinel-2 scenes, and `manifestA_chips.csv.gz` lists every chip with its label. The full score output is `../../results/e4_partA_results.json` (per-seed values, all contrasts, block-bootstrap intervals); the headline numbers are copied under key `partA` of `../../results/e4_retrieval_parity.json` by `summarize_partA.py`.

**What was re-run.** The `score` stage was re-run on 2026-11-17 on the cached float16 embeddings from the original embedding stage (CA/TX/NC, 64 and 256 px, documented-order, training-order and metadata-off Clay arms, DINOv2-L, band statistics). The chip manifest it writes is byte-identical to the committed one, and every pooled value quoted above reproduced to the printed precision (for example Clay CLS 0.038 [0.018, 0.092] at 64 px and 0.268 [0.107, 0.638] at 256 px; clay_mean 0.013 and 0.021). The embeddings themselves were first not regenerated; see the next paragraph.

**What the per-tile numbers add (key `partA.ap_k5_pos_by_region`).** The pooled 256 px AP of 0.27 is an unweighted mean of three tiles with very unequal evidence: CA 0.077 (39 positives; random 0.029), NC 0.219 (6 positives; random 0.013) and TX 0.508 (15 positive chips from 5 sites; random 0.013). DINOv2-L reads 0.064 / 0.245 / 0.659. The 256 px gallery holds 60 positives in total (789 at 64 px). Read 0.27 as a three-tile mean with a CI of [0.11, 0.64], not as a stable detection rate; DINOv2-L at 256 px is 0.323 [0.155, 0.513] and the Clay-minus-DINOv2 difference, -0.055 [-0.245, 0.266], is unresolved.

**Full regeneration from scratch (2026-11-17, same day).** `partA.py` stages `embed`, `nometa`, `train` and `score` were run end to end into an empty cache (Planetary Computer tile reads, DINOv2-L and Clay v1.5 on one RTX 4090, about 35 min plus the scoring stages). `../../results/e4_partA_regen_check.json` (written by `compare_partA.py`) puts the committed and regenerated pooled AP side by side: every arm agrees to 4e-4 or better at both chip sizes (Clay CLS 0.0381 and 0.2681, Clay mean-token 0.0130 and 0.0212, DINOv2-L 0.2157 and 0.3227 vs 0.3229, spectral statistics 0.0161 and 0.0221, random 0.0093 and 0.0185); individual per-tile P@10 / R@100 entries can move by up to 0.04 to 0.24 where only a handful of positives exist (GPU non-determinism). **Bug found and fixed on the way:** `clayrt.py` now defaults to the training lat/lon order, so a first regeneration with the unmodified `embed` stage produced documented-order arm values identical to the training-order arm (0.0458 and 0.2721 instead of 0.0381 and 0.2681). `partA.py` now passes `latlon_order="documented"` explicitly in the `embed` stage; with that the documented-order arm reproduces. Either order gives AP 0.27 at 256 px; the headline does not depend on the order.
