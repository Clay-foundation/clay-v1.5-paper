# E7 — Fine-tuning vs frozen Clay on Sen1Floods11 (hand-labelled) — RESULTS

**What was measured.** Water IoU (pooled over labelled pixels) and 2-class mIoU, on the official Sen1Floods11 test split (90 chips) and on the held-out Bolivia event (15 chips). There are five arms, each trained with 3 seeds under one fixed recipe (PROTOCOL.md, frozen before scoring; amendment A1 adds reporting only): 100 epochs, AdamW, cosine schedule, 256-px crops, and the checkpoint selected on valid water IoU. Every arm sees the same 10 Sentinel-2 bands. Point estimates are seed means. The 95% CIs resample chips and seeds jointly (1,000 draws, seed 1729). Paired contrasts share the chip draws. Chip-only and 100 km-block bootstraps are in results.json.

| Arm (trainable params) | Test water IoU | Bolivia water IoU | Test mIoU | Bolivia mIoU |
|---|---|---|---|---|
| Clay frozen + linear (2 k) | 0.748 [0.652, 0.820] | 0.727 [0.570, 0.812] | 0.855 | 0.837 |
| Clay frozen + UPerNet (8.9 M) | 0.834 [0.774, 0.879] | 0.815 [0.688, 0.875] | 0.905 | 0.889 |
| Clay full fine-tune + UPerNet (320 M) | **0.838** [0.780, 0.881] | **0.816** [0.688, 0.877] | 0.907 | 0.889 |
| U-Net from scratch (31 M) | 0.829 [0.745, 0.883] | 0.794 [0.664, 0.869] | 0.902 | 0.876 |
| DOFA-L full fine-tune + UPerNet (346 M) | 0.812 [0.742, 0.864] | 0.796 [0.649, 0.865] | 0.892 | 0.878 |

**Paired contrasts, water IoU (seed-and-chip CI).**
- Fine-tuning vs frozen + UPerNet: +0.004 [−0.002, 0.012] on test and +0.001 [−0.006, 0.008] on Bolivia. The test difference is resolved under the pre-registered chip-only bootstrap ([+0.0006, +0.0075]) but not once seed variance is included.
- The decoder matters far more than unfreezing. Frozen + UPerNet minus frozen + linear is +0.086 [0.051, 0.137] on test and +0.088 [0.058, 0.125] on Bolivia.
- Clay fine-tuned vs U-Net from scratch: +0.009 [−0.017, 0.049] on test and +0.022 [−0.011, 0.056] on Bolivia. Neither is resolved.
- Frozen + linear is clearly below the U-Net: −0.081 [−0.125, −0.049] on test.
- Clay fine-tuned vs DOFA-L fine-tuned under the identical recipe: +0.026 [0.012, 0.049] on test and +0.020 [0.007, 0.043] on Bolivia. Both are resolved.

**Fraction of the fine-tuning gain over scratch retained by a frozen readout** (amendment A1, R = (frozen − scratch)/(FT − scratch)). This ratio is **ill-defined here**, because the denominator (FT − scratch) has a CI that includes 0.
- Frozen + UPerNet: R = 0.59 [−3.4, 3.9] on test and 0.97 [−0.3, 2.1] on Bolivia.
- Frozen + linear: R = −9.0 on test and −3.1 on Bolivia, with uninformative CIs.
- A more stable descriptive figure: frozen + UPerNet reaches 99.6% of fine-tuned water IoU on test and 99.9% on Bolivia.

**What it means.** On this small dense benchmark (252 training chips), the frozen Clay encoder with a light multi-level decoder gets essentially all of full fine-tuning's performance, at 1/36 of the trainable parameters, about 40% of the training wall time (mean 866 vs 2,216 s, shared GPU) and about 1/3 of the peak GPU memory (2.5 vs 7.0 GB). A single linear layer on the last-layer tokens is not enough. Neither Clay arm resolvably beats a well-trained from-scratch U-Net on the same bands. That is a null result, reported as such: Sen1Floods11 water is spectrally easy (NIR/SWIR), and a CNN trained directly on it is a strong baseline. Under an identical recipe, Clay fine-tuned is ahead of DOFA-L fine-tuned. The test-to-Bolivia drop is small for every arm (0.016–0.035 water IoU; largest for the U-Net). The ranking matches test except that DOFA and the U-Net swap places, and their difference is within noise.

**Caveats.**
1. **Input handicap.** Sen1Floods11 ships S2 **L1C** (top-of-atmosphere). Clay and its normalisation statistics come from **L2A** (surface reflectance, PC). No harmonisation was applied, and B1/B9/B10 were dropped. This shift hits the frozen arms hardest, so the frozen numbers are conservative.
2. **One fixed recipe, not tuned per arm.** The recipe is plain cross-entropy with a fixed LR per arm type. DOFA got Clay's normalisation statistics and an interpolated positional embedding. Its number reflects this protocol, not DOFA's best.
3. Bolivia has only 15 chips in 8 blocks, so its CIs are wide.
4. Test and Bolivia were scored once per run. When amendment A1 was written, one run (Clay FT, seed 0) had already printed test scores. A1 changed reporting only.

**Context only (different protocols, not comparable with the numbers above).**
- Prithvi-EO-2.0 paper, Table IV (arXiv:2412.02732): test water IoU 82.2–83.1 and mIoU 89.7–90.3. That setup uses 6 HLS bands, a 448-px resize, an UPerNet head, 10-trial hyperparameter tuning and 10 seeds.
- TerraMind paper, Table 8 (arXiv:2504.11171): TerraMindv1-B S-2 full fine-tune, water IoU 82.26 and mIoU 89.70.

## Audit rerun and paired intervals (2026-11-17)
**What was re-run.** The per-seed outputs of three arms were lost, so Clay frozen + linear, Clay frozen + UPerNet and the from-scratch U-Net were re-trained (3 seeds each, same recipe, 100 epochs, `train_e7.py --max-gpu-gb 10`). The Clay fine-tune and DOFA-L arms are the original runs (their per-seed outputs survived). All 15 runs, with provenance, are in `../../results/e7_runs/` (`PROVENANCE.md`); `code/analyze_e7.py` regenerates `../../results/e7_finetune_results.json` from them (arms, paired contrasts with chip-only, 100 km-block and seed-and-chip intervals, retained fraction, per-run training cost).

**Water IoU, test / Bolivia (point = seed mean; interval = seed-and-chip bootstrap), regenerated.** Frozen + linear 0.748 [0.650, 0.819] / 0.727 [0.571, 0.812]; frozen + UPerNet 0.833 [0.772, 0.878] / 0.816 [0.689, 0.875]; Clay fine-tune 0.838 [0.780, 0.881] / 0.816 [0.688, 0.877]; U-Net from scratch 0.831 [0.747, 0.886] / 0.799 [0.678, 0.871]; DOFA-L fine-tune 0.812 [0.742, 0.864] / 0.796 [0.649, 0.865]. Against the earlier table, frozen + linear is unchanged (0.748), frozen + UPerNet moves 0.834 -> 0.833, and the U-Net 0.829 -> 0.831 (its three seeds now span 0.830-0.831; the original three spanned 0.821-0.834).

**Paired contrasts, water IoU (point, seed-and-chip CI; chip-only and 100 km-block CIs in the JSON).**
- Fine-tune minus frozen + UPerNet: +0.005 [-0.001, 0.015] test; +0.000 [-0.007, 0.008] Bolivia. The chip-only interval on test is now [-0.0000, 0.011], so the earlier statement that this difference is resolved under the chip-only bootstrap no longer holds (earlier [+0.0006, +0.0075]).
- Frozen + UPerNet minus frozen + linear: +0.085 [0.052, 0.135] test; +0.088 [0.060, 0.125] Bolivia.
- Fine-tune minus U-Net: +0.007 [-0.018, 0.047] test; +0.017 [-0.012, 0.046] Bolivia. Unresolved.
- Frozen + UPerNet minus U-Net: +0.002 [-0.021, 0.039] test; +0.017 [-0.012, 0.045] Bolivia. Unresolved, so "frozen ties from-scratch" is a statement of no detected difference, not equivalence.
- Frozen + linear minus U-Net: -0.083 [-0.128, -0.052] test.
- Fine-tune minus DOFA-L fine-tune: +0.026 [0.012, 0.049] test; +0.020 [0.007, 0.043] Bolivia. Resolved.
- Retained fraction R = (frozen - scratch)/(FT - scratch) stays ill-defined (denominator CI includes 0). The descriptive ratio frozen/FT water IoU is 99.4% on test and 99.96% on Bolivia.

**Training cost (key `training`; wall-clock on a GPU shared with other jobs, so not a controlled timing).** Trainable parameters: linear 2,050; frozen UPerNet 8,855,810; fine-tune 320,298,754 (ratio 1/36.2); U-Net 31,041,730; DOFA-L 346,330,927. Peak GPU memory: 1.66, 2.45, 6.98, 3.19 and 6.70 GB. Mean wall seconds: 522, 405, 2,216, 285 and 823. The earlier "about 40% of the wall time (866 vs 2,216 s)" is **not reproduced** in this rerun: frozen UPerNet took 405 s on average (18% of fine-tuning), and the from-scratch U-Net 285 s. Because the original and the rerun GPUs were shared, quote the parameter ratio (1/36) and the memory ratio (about 1/3) and either drop the wall-clock ratio or give it as "roughly 20-40%, shared GPU".
