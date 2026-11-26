# E7 — Fine-tuning vs frozen Clay on a dense task (Sen1Floods11) — PROTOCOL (frozen 2026-11-05, before any score)

Follows `../PROTOCOL-COMMON.md`. Written and saved before any training run or score. Changes only by dated amendment appended below.

## Question
What does fine-tuning add over frozen Clay v1.5 on dense flood-water segmentation, and how does fine-tuned Clay compare with a from-scratch U-Net and with one other foundation model (DOFA) fine-tuned the same way?

## Data
- Sen1Floods11 v1.1, hand-labelled subset, from the public bucket `storage.googleapis.com/sen1floods11/v1.1` (S2Hand + LabelHand + official split CSVs `flood_{train,valid,test,bolivia}_data.csv`). 446 chips of 512×512 px at 10 m: train 252, valid 89, test 90, Bolivia 15 (held-out event, geographic transfer test). All chips used; no subsampling.
- Input: Sentinel-2 **L1C** (TOA) as shipped by Sen1Floods11, int16 DN. Clay's 10 S2 bands are taken from the 13 (B2,B3,B4,B5,B6,B7,B8,B8A,B11,B12 → blue…swir22); B1, B9, B10 dropped. Raw DN normalised with Clay's `sentinel-2-l2a` mean/std (as Clay was trained); **L1C vs L2A mismatch is accepted and stated** (no harmonisation, no atmospheric correction). The same 10 normalised bands feed every arm.
- Nodata: pixels with any band = 0 are filled with the band mean (0 after normalisation). Labels: 1 water, 0 not water, −1 ignored (cloud/nodata) in loss and metrics.
- Clay metadata: chip-centre lat/lon in Clay's TRAINING order (clayrt default), acquisition date = the event date from the Sen1Floods11 STAC catalog, GSD 10 m, S2 wavelengths from metadata.yaml. DOFA gets the same wavelengths (µm).

## Arms (all 10 bands, same augmentation, loss, epochs, schedule, early-stopping rule)
1. `clay_frozen_linear`: frozen Clay, final-layer patch tokens (normed, 1024-d, stride 8) → 1×1 linear layer to 2 logits → bilinear upsample ×8.
2. `clay_frozen_upernet`: frozen Clay, taps after blocks 6/12/18/24 (each passed through the encoder's final LayerNorm) → simple feature-pyramid neck (1×1 conv to 256 ch; resampled to strides 4/8/16/32 by ConvTranspose ×2 / identity / maxpool ×2 / maxpool ×4) → UPerNet head (PPM 1/2/3/6 + FPN, 256 ch) → 1×1 conv → bilinear ×4.
3. `clay_finetune_upernet`: as arm 2 with the whole encoder trainable. Encoder LR = 0.1 × decoder LR. Gradient checkpointing, bf16 autocast.
4. `unet_scratch`: U-Net (4 down/4 up levels, base 64 ch, BatchNorm), random init.
5. `dofa_finetune_upernet`: DOFA ViT-L/16 (torchgeo `dofa_large_patch16_224` MAE weights), fixed sincos pos-embed bicubically interpolated to 16×16 for 256-px crops, taps after blocks 6/12/18/24 with the final norm, neck resampling ×4 / ×2 / identity / maxpool ×2 (strides 4/8/16/32 again), same UPerNet head, whole encoder trainable at 0.1 × decoder LR.

## Training (fixed a priori; no tuning on valid or test)
- Random 256×256 crop + random h/v flip + random 90° rotation; one crop per training chip per epoch; batch 8; **100 epochs** (32 steps/epoch, ≈3,200 steps).
- Loss: unweighted cross-entropy, ignore_index −1.
- AdamW (β 0.9/0.999, weight decay 0.05), linear warm-up for 5 epochs then cosine decay to 0.
- Peak LR: linear head 1e-3; UPerNet decoder 1e-4 (arms 2, 3, 5); encoder 1e-5 (arms 3, 5); U-Net 1e-3.
- Validation: every 5 epochs on the official valid split; the checkpoint with best valid water IoU is kept (early stopping by selection only). Test and Bolivia are evaluated once, with that checkpoint, after training ends.
- Seeds 0, 1, 2 for every arm (init of new layers, crop/flip order).
- Inference: each 512² chip is split into 4 non-overlapping 256² tiles (same size as training), predictions stitched.
- GPU budget: ≤ 10 GB per job; batch fixed at 8 for all arms.

## Metrics and statistics
- Primary: **water IoU** = TP/(TP+FP+FN) pooled over all labelled pixels of a split; secondary: mIoU = mean(IoU_water, IoU_notwater). Reported on test (primary) and Bolivia (geographic transfer). Valid reported for completeness.
- Point estimate = mean over the 3 seeds. Uncertainty: 1,000 bootstrap resamples (RNG seed 1729) **over chips (tiles)** of the seed-mean pooled metric (per-chip confusion counts re-pooled per resample); seed min/max also reported. Sensitivity: the same bootstrap over 100 km `evalkit.hblock` blocks of chip centres (for Bolivia, few blocks → chip bootstrap is primary).
- Pre-registered paired contrasts (shared resamples): (3)−(1), (3)−(2), (2)−(1), (3)−(4), (3)−(5), on test and Bolivia, water IoU and mIoU.
- Interpretation rule: a difference is called "resolved" only if its 95% CI excludes 0.

## Outputs
`results.json`, `RESULTS.md` (≤ 1 page), `figure/e7_finetune.{png,pdf}`, `manifest/chips.csv` (chip IDs, split, lat/lon, date), `code/`.

## Amendments
### A1 (2026-11-05, ~21:40 UTC) — reviewer-panel reporting additions
Timing disclosure: when this amendment was written, exactly one run had produced test/Bolivia scores (`clay_finetune_upernet` seed 0, printed in the job log). No other arm had been scored. Nothing in training, splits, arms, epochs, LR or the early-stopping rule changes; this amendment only adds derived reporting quantities and context.
1. Water IoU on test and Bolivia for arms 1–4 (frozen-linear, frozen+UPerNet, full FT, U-Net scratch) is reported with **paired seed-and-chip CIs**: each bootstrap resample (1,000, seed 1729) draws chips with replacement *and* seeds with replacement (3 of 3) independently per arm, then recomputes the pooled metric. Paired contrasts share the chip draw. The chip-only bootstrap of the seed-mean stays as the original specification and is also reported.
2. **Fraction of the full-FT gain over scratch retained** by each frozen readout X ∈ {linear, UPerNet}: R_X = (IoU_frozenX − IoU_scratch)/(IoU_fullFT − IoU_scratch), on test and Bolivia, with the same paired resamples as CI (percentile). If the denominator CI includes 0, R is flagged as ill-defined and reported without a CI interpretation.
3. Input handicap stated explicitly in RESULTS.md: Sen1Floods11 ships S2 **L1C TOA**, while Clay (and its normalisation stats) were trained on **L2A BOA** from Planetary Computer. This is a shift for the frozen arms in particular. B1/B9/B10 are dropped for all arms, and the U-Net sees the same 10 bands.
4. Published Sen1Floods11 numbers for Prithvi-EO-2.0 / TerraMind appear only as clearly labelled context (different protocols, bands and splits), with citations verified at the source. They are not part of any contrast.
