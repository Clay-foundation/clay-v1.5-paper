# E2c_posemb_ablation — does Clay's GSD position code limit spatial information in patch tokens? (frozen 2026-11-05, before the main training run and before any score)

Follows `../PROTOCOL-COMMON.md` and reuses E2_tokens' chips, splits, token positions, labels, heads and bootstrap. This is a controlled retrain of a **small** Clay MAE, not of Clay v1.5. Results do not transfer automatically to the ViT-L.

## Question and arms
Causal question: with everything else fixed, does replacing Clay's implemented position code change how much absolute position (and land cover) a linear probe can read from patch tokens?
- **A — implemented.** `posemb_sincos_2d_with_gsd` from the pinned Clay source, verbatim, except that the exponent denominator is fixed at 1016 (the value the released ViT-L uses, dim 1024 − 8). All frequencies then lie within 1.8 % of `gsd` rad/token, as in Clay v1.5. (Called literally inside the small model, with dim 376, the spread would be 4.8 %.)
- **B — standard.** `posemb_sincos_2d` (simple-ViT/MAE 2-D sin-cos, temperature 10⁴), no GSD.
- **C — GSD-scaled standard (Scale-MAE style).** Standard frequencies, positions multiplied by gsd / 10 m. C is therefore identical to B on S2 and S1 (10 m) and differs only for Landsat (×3) and NAIP (×0.1).
The code is used in both encoder and decoder, as in Clay. The 8 metadata channels are kept as zeros (time and lat/lon unknown).

## Model and training (identical across arms)
- `clay_mae_small` dims from the pinned `common/claymodel/model.py`: encoder 384-d, depth 6, 6 heads; decoder 192-d, depth 4. Clay's `Encoder`, `Decoder` and `DynamicEmbedding` are used unchanged; 14.6 M parameters per arm.
- Declared simplifications versus Clay v1.5: (1) no teacher/distillation loss and no projection head, so the loss is reconstruction only (L1 on masked patches, `norm_pix_loss` False, mask ratio 0.75, patch 8, as in Clay's config); (2) time/lat/lon zeros, which also means Clay's channel dropout (it fires only when lat/lon ≠ 0) never triggers; (3) no MODIS, Sentinel-2 bands as stored by E2 (10 bands), Landsat 6 bands, NAIP 4, S1 VV/VH dB.
- Data: E2's chips not in E2's test split (train ∪ val; 100 km blocks, `evalkit.block_split`): S2 2,908, S1 2,790, Landsat 2,830, NAIP 1,691 (10,219 chips). Normalised with Clay's platform mean/std (`metadata.yaml`), raw S2 values without harmonisation (as E2). E2 test chips are never loaded by the training script. No extra chips are fetched (declared deviation).
- Full 256×256 chips (32×32 tokens, the grid E2 probes), with a random dihedral transform per batch (rot90 k∈0..3, optional horizontal flip). No random crops.
- Batches are single-sensor (as in Clay), sensor drawn with probability ∝ training chips; batch 64.
- AdamW (β 0.9/0.95, wd 0.05 on weight matrices), peak lr 2×10⁻⁴, linear warm-up over 5 % of steps then cosine to 0, gradient clip 1.0, bf16 autocast.
- **Steps: 80,000** per arm (5.12 M chip presentations, ≈ 500 epochs). A pilot (200 steps, used only to time the loop) gave 0.155 s per lock-step for three arms, so ≈ 3.4 h wall and ≈ 1.15 GPU-h per arm, below the 2–3 h per arm offered. More steps over 10 k chips would mostly add epochs on the same data.
- Lock-step: one process trains all three arms. Identical initial weights (`torch.manual_seed(0)` before each model), identical batch, augmentation and mask at every step (`torch.manual_seed(1e6 + step)` before each arm's forward). The arms differ only in the position code.
- Logged: training loss per arm every 50 steps (loss curves). Checkpoints at 40,000 (mid) and 80,000 (final) steps.
- Single training seed (declared). Run-to-run variance from the training seed is therefore not estimated. The CIs cover test-sample (block) uncertainty, and for the dense probe also token-draw seeds.

## Evaluation (E2's test chips, held out from pretraining)
Tokens: encoder output with no masking, eval mode. Patch tokens are taken at exactly E2's token positions (64 sha256-ranked per train chip; the stride-2 16×16 grid per test chip). E2's labels, blocks and split are read from E2's `emb_<sensor>.npz`, with pid order asserted equal.
- **(i) Position R².** E2's probe: `evalkit.reg_head` (RidgeCV) from patch tokens to row and to column (0–31), fit on train tokens, R² on test tokens. The mean of row and column R² is reported, with each separately.
- **(ii) Dense WorldCover.** E2's per-token head: `evalkit.clf_head` (StandardScaler + LogReg C=1), 16 tokens per train chip per seed for seeds 0–4. Token-level macro-F1 (and mIoU) over the classes present in the sensor's test labels, seed-mean.
- **(iii) Reconstruction.** Masked-patch L1 (mask 0.75) on each test chip, averaged over two fixed mask draws that are identical across arms. Reported as the mean over chips.
- Sensors: S2, S1, Landsat and NAIP (S2 and NAIP are the two GSD extremes required).
- Uncertainty: 1,000 block-bootstrap resamples (seed 1729) over E2's 100 km test blocks per sensor. Paired contrasts share resamples (E2's `boot_idx`/`paired` helpers, copied verbatim).
- Descriptive: each arm's code itself as a linear ceiling (in-sample and checkerboard R², dim 376, at each sensor's GSD), and the same endpoints from the 40k-step mid checkpoint (training-length sensitivity).
- Context only, not a contrast: E2's ViT-L numbers (S2 position R² 0.30, dense macro-F1 0.613).

## Endpoints
- **Primary (pre-registered):** S2 position R² (mean of row/col), **B − A**, paired block bootstrap 95 % CI.
- Key secondary: NAIP position R² B − A; S2 and NAIP dense macro-F1 B − A; S2 and NAIP reconstruction L1 B − A; C − A on Landsat and NAIP.
- Everything else is descriptive.

## Code validation
Before this freeze, a 100-step smoke run of train → embed → probe (`--quick`) was used only to check that the pipeline runs. Its numbers come from an essentially untrained model, are not used for any decision, and are not reported.

## Amendment 1 (2026-11-05, after the seed-0 results; before any seed-1/2 training or score)
Why: the seed-0 report could not separate training-seed variance from the arm effect. The lead asked for two more training seeds per arm. Nothing above changes; the seed-0 results stay as reported.
- **Seeds 1 and 2**, trained with exactly the frozen recipe (arms A/B/C in lock-step, 80,000 steps, batch 64, same lr schedule, same data, same evaluation). Training seed `k` sets three things: initial weights (`torch.manual_seed(k)`), the sensor/batch/augmentation stream (`default_rng(20261105 + k)`) and the per-step masks (`manual_seed(1e6 + step + 1e7·k)`). Seed 0 corresponds to the original run's settings. Within a seed all arms still share init, batches and masks. Evaluation masks, token positions, heads and bootstrap resamples are identical across seeds.
- Seeds 1 and 2 are trained one after the other in one job (≤ 6 GB GPU).
- Seed 0 is re-embedded from its saved final checkpoints so that per-resample bootstrap values exist for every seed. The original seed-0 numbers stay canonical, and any difference on re-extraction is reported.
- Only final (80k) checkpoints are evaluated for seeds 1–2. The 40k-step sensitivity stays seed-0 only.
- **Pooled estimate:** for each contrast (B−A, C−A, C−B) and sensor, the mean of the three per-seed point contrasts.
- **Pooled CIs, both reported:**
  1. Hierarchical bootstrap: for each of the 1,000 shared block resamples, draw 3 training seeds with replacement (RNG 1729) and average that resample's per-seed contrasts. 95 % percentile interval. This covers test-block variance plus a (with n = 3, under-estimated) seed variance.
  2. Seed-level t-interval: mean ± t₀.₉₇₅,₂ · sd / √3 over the three per-seed point contrasts. This covers between-seed variance only.
- A contrast is called robust only if **both** intervals exclude 0, and all three seeds' point contrasts are also reported.
- The primary endpoint is unchanged (S2 position R² B − A), now reported pooled over 3 seeds. The secondaries are dense macro-F1 and reconstruction L1 contrasts per sensor.
