# E5 — Frozen Clay v1.5 vs state-of-the-art comparators on the AEF evaluation suite (protocol, 2026-11-05, frozen before any E5 score)

Follows `../PROTOCOL-COMMON.md` and reuses E1 part (a) exactly (`../E1_label_efficiency/PROTOCOL.md`, amendments 1–4): same rows (E1 `manifest/rows.csv`: all train rows with an E1 chip + tier-0 test rows, i.e. ≤50 test rows per class / 500 for regression), same AEF train/test split, same `evalkit.label_budget` draws (seeds 0–4), budgets n = 1, 10, max (max only where E1's completeness rule holds), same heads (LR C=1 / RidgeCV; kNN-3 secondary), metrics (balanced accuracy; R²), change-task features (after − before), and the same 1,000-resample 100 km block bootstrap (seed 1729) with paired contrasts. Clay arms are E1's stored vectors (training lat/lon order).

## Rows used by each arm
Each comparator is scored on E1's rows for which it has a feature. If an arm misses rows, the paired contrast re-scores Clay on exactly the same rows (label budgets re-drawn on available rows by the same hash, as in E1). Coverage per task and arm is reported. Row year = calendar year of the midpoint of the row's AEF `valid_time` period (per period for change tasks): 2017–2023 for all rows.

## Arms (what each receives)
1. **AEF released embeddings** (Source Cooperative mirror `tge-labs/aef/v1/annual`, CC-BY 4.0, "The AlphaEarth Foundations Satellite Embedding dataset is produced by Google and Google DeepMind"). Annual, 64-D, multi-source multi-date.
   - `aef_mean128` (primary AEF arm): mean of the dequantised 4×4 pixels of the COG's factor-32 overview (320 m px; the overview is the renormalised mean of the dequantised 10 m embeddings) nearest the point = a 1.28 km context like Clay's chip (centring error ≤160 m).
   - `aef_ov32_px`: the single 320 m overview pixel containing the point (secondary).
   - `aef_pixel` (AEF's own per-pixel protocol): the 10 m pixel at the point. Each point costs one 1024² block per band (~20–40 MB of transfer); it is fetched only if throughput allows, in priority order (tier-0 rows first), and is reported only for task/budget cells whose rows are complete. Otherwise reported as not run.
   - Dequantisation `((v/127.5)**2)*sign(v)`; −128 = nodata (row dropped for this arm).
2. **TESSERA v1.1** (geotessera 0.11.0, default dClimate Zarr store; annual 128-D, S1+S2 time series): `tessera_pixel` (primary, the 10 m pixel at the point, as TESSERA is used) and `tessera_mean128` (128×128 px mean; fetched in priority order until a fixed 2 h limit; reported where complete).
3. **OlmoEarth Base** (pre-registered as v1.2, `allenai/OlmoEarth-v1_2-Base`, olmoearth_pretrain 0.1.2; **the cached features behind Panel C were produced with v1.3**, `allenai/OlmoEarth-v1_3-Base`, see Amendment 4 and `code/embed_gpu_arms.py`): E1's single-date 128 px chip, 10 bands harmonised (−1000 where baseline ≥ 04.00), OlmoEarth computed normalisation, B01/B09 band set masked as missing, real acquisition timestamp, mean-pooled tokens; patch size 4 (primary, developers' eval default) and 8. Caveat: OlmoEarth is meant for multi-temporal input.
4. **Prithvi-EO-2.0-300M-TL**: 6 HLS-equivalent bands (B02,B03,B04,B8A,B11,B12) of the same chip, harmonised, Prithvi mean/std, bilinear 128→224 px, T=1 with time and location coordinates; mean of final-layer patch tokens (primary), CLS secondary.
5. **SatCLIP** (location only): ViT16-L40 (primary) and ViT16-L10 location encoders, 256-D, from the row's lon/lat.
Arms 3–5 depend on a GPU embedding script. Update 2026-11-17: OlmoEarth and Prithvi were later embedded (`code/embed_gpu_arms.py`; centre = mean of the central 4 x 4 tokens (OlmoEarth, 32 x 32 grid) or 2 x 2 tokens (Prithvi, 14 x 14 grid), mean = mean of all tokens; no CLS arm exists for either) and enter only the 15-task table produced by `run_e1_e5_complete.py`. SatCLIP and Clay + SatCLIP were not run.

## Endpoints
- **Primary:** per task, `clay_cls − X` at n = 10 and at n = max, linear head, paired 95% block-bootstrap CI; per arm a wins/ties/losses row (CI > 0 / includes 0 / < 0), for X in {aef_mean128, tessera_pixel, olmo_ps4, prithvi_mean, satclip_l40}.
- Secondary: all other readouts, kNN-3, n = 1.

## Amendment 1 — 2026-11-05 22:40 UTC (before any E5 score): reviewer-panel requirements
(a) For pixel-embedding comparators (AEF, TESSERA) also report `clay_center − X` (E1's 2×2 central patch-token arm). Clay CLS stays primary; centre is the matched-support secondary.
(b) Observation window per arm is recorded in the results table (Clay/OlmoEarth/Prithvi/DOFA: one S2 date in the valid period; AEF/TESSERA: calendar-year multi-sensor series; SatCLIP: none). Multi-temporal OlmoEarth input is not run (it would need new imagery beyond the ≤15 GB / PC rate limits) — stated as a limitation.
(c) DOFA-L with its own released normalisation at native 224 px: requires new GPU embedding; pending with arms 3–5. If not run, E1's DOFA (Clay S2 z-score, 128→224 bilinear) is reported as is, labelled.
(d) Location-only controls: SatCLIP (above), plus `sh10` = real spherical-harmonic basis of lat/lon up to degree L=10 (121 features) with the same linear head, plus `clay_cls ⊕ satclip_l40` concatenation.
(e) Per model, whether AEF-suite labels/locations entered training or model selection is reported from the papers.
(f) Embedding dimension and bytes per vector reported.
(g) Per-task row coverage vs AEF's released test set reported.

## Amendment 2 — 2026-11-05 23:30 UTC (before any E5 score)
1. GPU-embedded arms (OlmoEarth, Prithvi, SatCLIP, own-normalisation DOFA) are deferred (superseded: OlmoEarth and Prithvi were later run, see Amendment 4; SatCLIP and own-normalisation DOFA were not run). `clay_cls ⊕ satclip` is therefore also deferred. This round scores only the released-embedding arms (AEF, TESSERA) and the `sh10` coordinate control.
2. Point sampling is restricted to point-years used by some E1 evaluation cell: capped tier-0 test rows plus train rows selected by `label_budget` at n = 1/10/max for any seed (33,407 of 33,774 point-years; 367 never-selected train points are dropped). `label_budget` ranks only available rows, so selections are unchanged.
3. `aef_pixel` (AEF's 10 m per-pixel protocol) is read with 48 parallel requests in this order: points of test rows (9,260), then points of n ≤ 10 train rows (8,546), until a 3 h limit. It is scored only on the rows that complete. Comparisons are paired on rows that have both arms. If the train rows for n = 10 are incomplete, budgets are drawn on the available rows, and those cells are labelled provisional. `aef_mean128` stays the primary AEF arm.
4. `tessera_mean128` is read in priority order until a 2 h limit and scored on the rows that complete (labelled the same way). `tessera_pixel` stays primary.

## Amendment 3 — 2026-11-05 23:40 UTC (before any E5 score)
The bulk TESSERA `sample_points` call was too slow: it had not finished a single 2,000-point batch after 20 min. It was stopped. `tessera_pixel` is now taken as the centre pixel `[64, 64]` of the same `read_patch` call that gives `tessera_mean128`. geotessera documents that the point falls in that pixel. Both TESSERA arms are therefore read in one parallel pass (64 threads, priority order: test and n ≤ 10 train points first), stop at the 2 h limit, and are scored on the rows that complete, under the Amendment 2 rules. The decision was based on throughput only.

## Implementation note — 2026-11-06 02:40 UTC (after the first scoring run; no endpoint changed)
The first `eval_e5.py` run tested max-budget completeness against train rows that had already been filtered to E1 chips. That deviates from E1's rule, which tests against all tier-filtered train rows. The bug was fixed and everything was re-scored. All reported numbers come from the corrected run (`results_e5_main.json`, job a66e75b6). The only effect was to remove max cells that E1's rule does not allow (for example LUCAS max).

## Amendment 4 — 2026-11-17 (audit fixes; written after the first E5 scores, so it is post hoc and labelled as such)
Context: an independent audit (issues #7, #16, #26) found that the supplemental 15-task table (`results/e1_e5_complete.json`) was produced by a script that was not in the repository and that scored each arm on different rows. The following changes apply to that table. The original `eval_e5.py` output (`results/e5_sota_comparators.json`) is unchanged and is kept as the cross-check.
1. **Producer released and made portable**: `code/run_e1_e5_complete.py` (CPU, from cached features), `code/embed_gpu_arms.py` (the GPU stage that creates the cached OlmoEarth / Prithvi features), `code/compare_gpu_cache.py` (checks a fresh embedding against the cache), `code/build_label_years.py`. All paths are CLI flags; the results JSON records script hash, git commit, arguments, package versions, UTC time and SHA-256 of every input file.
2. **Label year**: point features are joined at the row's label year(s) by exact `(x, y, year)`. No earliest / latest / nearest-year fallback. `build_label_years.py` has an exact mode (the AEF `evaluation.zip` valid periods) and a fallback mode that derives the years from `points.csv` (used for the committed numbers because `evaluation.zip` was not available on the compute host; see RESULTS.md). The fallback mode never guesses: static rows take the single sampled year at their location; change rows take the single consecutive pair sampled there, or, if several rows share the location, the pair of years of their own two chips when that pair is one of the sampled consecutive pairs (`points_chip_tiebreak`, flagged in the JSON).
3. **Matched rows per comparison**: each comparison is scored on a *set* of arms with identical rows and identical training draws (rows where every arm of the set has a feature; draws made on the set's own train rows by the same hash). Sets: `chip` (all chip arms), `clay+<point arm>` (three Clay readouts plus one AEF / TESSERA arm) and `common` (all chip arms plus `aef_mean128`, `aef_ov32_px`, `tessera_pixel`).
4. **Budget handling**: budgets 10 and max. Each cell records the nominal draw and the achieved draw and whether E1's completeness rule holds on the set (`budget_complete`). The summaries count max cells with `budget_complete = true` as the headline and list the incomplete ones separately (they ran on fewer labels than nominal; they are never relabelled as max).
5. **Paired uncertainty**: all arms of a set share the same 1,000 block-bootstrap resamples (100 km blocks, seed 1729, test rows of the set). Contrasts are differences of the seed-mean metric on the shared resamples. Heads: linear only (kNN-3 is in the original `e5_sota_comparators.json`).
6. **Published counts**: `code/summarize_panelC.py` prints every count the manuscript states (wins / intervals including 0 / losses and point-estimate wins, per Clay readout against each comparator and budget, centre versus pooled readouts with denominators, every Panel C cell with its JSON key) from the JSONs only.
7. **Readout status**: CLS is the pre-registered Clay readout. The centre-token (2 x 2 central patches) readout was added as a secondary, matched-support arm in Amendment 1 and is a post-hoc primary for Table 2. Both are reported side by side.
8. **Equivalence**: no non-inferiority margin was ever defined. "Interval includes 0" is reported as inconclusive; it is not evidence of equivalence.

## Reproduction (clean checkout, CPU)
```
# inputs (not in the repository): E1 chip embeddings shard_*.npz + meta.csv, e5_gpu_feats.npz (from embed_gpu_arms.py),
# e1_rows.csv, points.csv, aef_ov32.jsonl, aef_pixel.jsonl, tessera_patch.jsonl   -> DATA/
python experiments/suite/E5_sota_comparators/code/build_label_years.py --rows DATA/e1_rows.csv --points DATA/points.csv --meta DATA/e1/meta.csv --out DATA/label_years.csv
python experiments/suite/E5_sota_comparators/code/run_e1_e5_complete.py --data_root DATA --out OUT/e1_e5_complete.json   # about 40 min on a loaded 32-thread host
python experiments/suite/E5_sota_comparators/code/summarize_panelC.py --results OUT          # prints every count
```
The fetch scripts for AEF / TESSERA points (`sample_aef.py`, `sample_tessera.py`) need network access to Source Cooperative / the dClimate store; the cached JSONL files are the starting point of the CPU stage.
