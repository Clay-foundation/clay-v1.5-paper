# E3 — Cross-sensor transfer in Clay v1.5 token space

**Setup.** We built 6,592 co-located locations from 824 NAIP tiles (2021–2022) across CONUS, all sampled before scoring. Each location is a 320 m cell with ESA WorldCover 2021 majority purity ≥ 0.60, and has:
- Sentinel-2 L2A 32×32 at 10 m (median 2 days from the NAIP date);
- NAIP 256×256 at 1.25 m;
- Sentinel-1 RTC 32×32 in dB (6,494 locations);
- Landsat 8/9 L2 16×16 at 30 m over 480 m (6,386 locations).

Seven classes were retained; wetland had too few samples. The split is 100 km blocks: about 5,500 train and 1,000 test locations in 85 test blocks. The head is the fixed logistic regression from `evalkit`. Embeddings are Clay CLS with real lat/lon/time unless stated. CIs are 95 % from 1,000 block-bootstrap resamples. Protocol: PROTOCOL.md, frozen before any score.

**Primary endpoint (pre-registered).** A head trained on S2 and applied unchanged to NAIP of the same places reaches macro-F1 **0.26 [0.23, 0.28]**, against **0.83 [0.79, 0.86]** for a NAIP-trained head. Δ = **−0.58 [−0.61, −0.53]**, far outside the ±0.10 we pre-declared as "transfer works". Transfer efficiency over the majority baseline (0.075) is 0.24 [0.20, 0.28]. The result is above chance but useless in practice. This is consistent with the community report that an S2 head "returned nothing" on NAIP.

**Every pair fails the same way.** Zero-shot macro-F1 vs native:
- NAIP→S2: 0.20 vs 0.70;
- S2→S1: 0.15 vs 0.54;
- S1→S2: 0.16 vs 0.69;
- S2→Landsat: 0.20 vs 0.64;
- Landsat→S2: 0.30 vs 0.71.

Re-standardising the target with its own unlabeled statistics helps by ≤ 0.14. An orthogonal Procrustes map learned from 1,000 unlabeled paired chips lifts S2→NAIP only to 0.37 [0.33, 0.39]. It changes the other pairs by between −0.09 and +0.08, and lowers Landsat→S2. Adding source labels hurts. Pooling all S2 training labels with 50 NAIP labels/class gives 0.69, worse than 50 NAIP labels/class alone at 0.79 (Δ −0.10 [−0.13, −0.08]). The pooled arm is below native-k in every pair at both k = 10 and 50. Mean-pooled patch tokens behave the same (S2→NAIP zero-shot 0.15). DINOv2-L on RGB transfers somewhat better S2→NAIP (zero-shot 0.36; Procrustes 0.57), but it too is far below its native 0.74.

**Retrieval of the paired chip.** For each S2 chip we ranked its own NAIP chip among all 6,592 NAIP chips (chance R@10 = 0.15 %). Recall@10:
- Clay CLS: 1.4 % [1.1, 1.8] (median rank 1,919); 2.4 % after per-sensor centering;
- DINOv2-L RGB: 4.0 % (11.2 % centred);
- raw down-sampled RGBN pixels: 12.2 % [10.6, 13.9], R@1 8.7 %.

Paired S2–NAIP Clay CLS cosine is 0.49. Unpaired cross-sensor pairs score 0.45, while two random S2 chips score 0.63. Embeddings cluster by sensor more than by place.

**Metadata does not drive alignment, in either direction.** Turning metadata off leaves retrieval unchanged: R@10 1.55 % [1.18, 1.97] off vs 1.40 % on. It also leaves classification transfer unchanged (S2→NAIP zero-shot 0.25 off vs 0.26 on). Location and time encodings are not what little alignment exists.

**Reading.** In Clay v1.5, sensors are not interchangeable at the head level. Each sensor has a good token space of its own (native heads 0.54–0.83 macro-F1), but the spaces are offset and only weakly aligned. A linear head, or a single rotation, does not carry across sensors. The claim of a shared cross-sensor token space is not supported by this test. The usable claim is one encoder for many sensors, with one head per sensor; a few target labels beat any transfer from another sensor.

**Caveats.**
- S2/S1 chips are 32 px (16 tokens), well below Clay's 256 px training chips.
- NAIP was resampled to 1.25 m.
- Labels are WorldCover (noisy) at 60 % purity.
- Acquisitions differ by up to 60 days.
- Landsat chips come from 10 m nearest-neighbour resampling plus a 3×3 mean.
- Procrustes in 1,024-d with ≤ 1,000 pairs is under-determined.
- The S2 processing baseline is recorded per chip (03.00/04.00/05.10/02.12: 2,954/1,740/1,337/561) but not used for stratification.


## Amendment — lat/lon order (2026-11-05)

**What changed.** Clay v1.0/v1.5 were trained with longitude in the first two lat/lon slots. The original metadata-ON numbers above used the documented order [sin lat, …], which is out-of-distribution metadata for the model. They are kept, labelled *documented order*. All metadata-ON Clay embeddings were re-run in the training order on the same cached chips, and every metadata-ON arm was re-scored with the frozen code. Metadata-OFF numbers are unaffected. Re-embedding in the documented order reproduced the stored embeddings (min cosine 0.9999998). Switching to the training order moves the CLS token only a little: mean cosine to the old token is 0.97 for S2 and NAIP, 0.86 for S1 and 0.88 for Landsat.

**Primary endpoint, training order.** S2→NAIP zero-shot reaches 0.258 [0.221, 0.289], against native 0.833 [0.795, 0.859]. Δ = **-0.575 [-0.611, -0.537]**, against −0.575 [−0.612, −0.531] in the documented order. Transfer efficiency is 0.24 [0.19, 0.28] (was 0.24). The conclusion is unchanged: the head does not transfer.

| Pair (Clay CLS, macro-F1) | Native, documented / training | Zero-shot, documented / training | Zero-shot, metadata off | Procrustes 1,000, documented / training |
|---|---|---|---|---|
| S2→NAIP | 0.83 / 0.83 | 0.26 / 0.26 | 0.25 | 0.37 / 0.40 |
| NAIP→S2 | 0.70 / 0.73 | 0.20 / 0.21 | 0.20 | 0.20 / 0.21 |
| S2→S1 | 0.54 / 0.56 | 0.15 / 0.14 | 0.15 | 0.23 / 0.20 |
| S1→S2 | 0.69 / 0.73 | 0.16 / 0.27 | 0.12 | 0.19 / 0.19 |
| S2→Landsat | 0.64 / 0.62 | 0.19 / 0.22 | 0.14 | 0.22 / 0.21 |
| Landsat→S2 | 0.71 / 0.73 | 0.30 / 0.29 | 0.32 | 0.20 / 0.21 |

The training order raises native heads by 0.02–0.03 when S2 is the target (S2→Landsat native falls by 0.015), and S1→S2 zero-shot from 0.16 to 0.27. No pair comes within 0.40 of its native head. Pooled source labels still hurt in every pair (transfer + 50 minus native-50 ranges from −0.10 to −0.06).

**Metadata on/off, training order.** S2→NAIP recall@10 for Clay CLS (chance 0.15 %):
- metadata on, training order: 1.52 % [1.15, 1.93];
- metadata on, documented order: 1.40 % [1.05, 1.77];
- metadata off: 1.55 % [1.18, 1.97].

Centred, the three give 2.34 / 2.40 / 2.25 %. NAIP→S2 and S2→S1 behave the same, with every Clay variant between 0.6 % and 2.9 %. S2→NAIP zero-shot classification is 0.26 in the training order vs 0.25 with metadata off.

Metadata in either order neither creates nor explains cross-sensor alignment, and every conclusion above stands. The figure still shows the documented-order values. In panel a, the training order changes S2→NAIP by about 0.03 at most; the largest shift elsewhere is S1→S2 zero-shot, 0.16 → 0.27.

## Note — training-order values quoted in the paper (added 2026-11-05, read from results.json `amendment_latlon_2026_11_05`)
- Cosine, Clay CLS metadata-on training order: S2–NAIP paired 0.497, unpaired 0.461, within-S2 0.623, within-NAIP 0.592; S2–S1 paired 0.597, within-S2 0.626; S2–Landsat paired 0.633, within-S2 0.624 (so the "sensor before place" grouping holds for S2–NAIP but not S2–Landsat).
- Pooled S2 + 50 NAIP labels/class, training order: 0.685 [0.642, 0.712] vs NAIP-only 0.789 [0.750, 0.814]; Δ −0.104 [−0.127, −0.082].
- The paper's figure is `fig_e3_cross_sensor_trainorder` (training order); the sentence above saying the figure shows documented-order values refers to the original `fig_e3_cross_sensor`.

## Audit reruns and corrections (2026-11-17)
**Artifacts now in the repository.** `manifest.csv` (location id, centre, WorldCover label, STAC item ids and dates, 100 km block, split for the 6,592 locations), `code/rows_from_manifest.py` (manifest -> `rows.json`) and `code/evaluate.py --overlay-trainorder` (re-scores with the metadata-on Clay arms in the training lat/lon order). `../../results/e3_cross_sensor.json` keeps the documented-order output it always had and now also holds, under `amendment_latlon_2026_11_05`, the full training-order classification, retrieval and cosine tables (with per-pair n and a `clay_on_cls_summary` of ranges), plus `rerun_check_documented_order_2026_11_17`.

**Which order produces the numbers in the paper.** The ranges "40-57 points", "Procrustes 0.20-0.40" and "native 0.56-0.83" are the **training-order** values (Clay v1.5 CLS, metadata on): zero-shot minus native −0.41 to −0.575 (S2→NAIP −0.575 [−0.611, −0.536], S2→S1 −0.412, S2→Landsat −0.408, NAIP→S2 −0.524, S1→S2 −0.455, Landsat→S2 −0.430), native macro-F1 0.555–0.833, zero-shot 0.143–0.295, Procrustes (1,000 pairs) 0.193–0.399. The documented-order values committed in the main JSON give a drop of −0.39 to −0.575, native 0.54–0.83 and Procrustes 0.19–0.37. The conclusion does not depend on the order.

**Counts.** 6,592 is the size of the location pool. The usable per-pair sample (train + test, 7 classes) is 6,540 for S2↔NAIP, 6,442 for S2↔S1 and 6,338 for S2↔Landsat; train/test are 5,526/1,014, 5,468/974 and 5,337/1,001. Zero-shot and native heads are single fits (n_seeds = 1); the k-label and Procrustes arms average 5 label seeds.

**Footprints.** S2, S1 and NAIP chips cover 320 m (32 × 32 px at 10 m; NAIP 256 × 256 px at 1.25 m). Landsat chips are 16 × 16 px at 30 m = **480 m**, so "matched 320 m footprints" is true for three of the four sensors only.

**Reproduction check.** Re-scoring the cached embeddings with this repository's code (scikit-learn 1.9.1) reproduces the committed documented-order headline arms (native, zero-shot, majority, re-standardised zero-shot) to within 0.008 macro-F1 and the paired-retrieval R@10 to 5e−7. The Procrustes arms move by up to 0.026 (n = 100) and 0.023 (n = 1,000) between environments, so quote them to one decimal place.

**Chip size.** E3 uses 32 px S2/S1 chips (16 tokens), far below Clay's 256 px training chips. Matched-context numbers are in `../E8_reviewer_controls/RESULTS.md` item 1 and collected next to the E3 values in `../../results/e8_confirmed_numbers.json` (`chip_size_confound`): S2→NAIP Δ = −0.591 at 320 m / 32 px against −0.222 at 2.56 km (CLS) and −0.448 (centre 4 × 4 patch tokens), but the 2.56 km narrowing comes from the native head falling (0.833 → 0.481) while zero-shot stays at 0.24–0.26.
