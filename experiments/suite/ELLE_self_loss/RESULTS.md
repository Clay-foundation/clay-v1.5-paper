# ELLE check: can a Ridge probe on frozen CLS predict the chip's own reconstruction loss?

**What was measured.** 3,000 Sentinel-2 L2A 256 px chips of the E2 token study (`../E2_tokens/manifest.csv`, 1,908 blocks of 100 km), Clay v1.5, location slots in the training order. Target: per-chip mean absolute error between the normalised input patches and the decoder reconstruction. Probe: StandardScaler + RidgeCV (alpha chosen by inner CV inside the training folds), 5-fold GroupKFold over 100 km blocks, out-of-fold predictions, block-bootstrap intervals. Script `elle_probe.py`, output `../../results/elle_probe.json`; latency script `latency_bench.py`, output `../../results/elle_latency_microbench.json`.

| Target | CLS probe r [95% CI], R2 | mean-patch probe r | band mean/std probe r |
|---|---|---|---|
| Unmasked input (mask ratio 0) | 0.9955 [0.9947, 0.9962], 0.991 | 0.9967 | 0.9777 |
| 75% masked, fixed mask seed (the MAE training loss) | 0.848 [0.821, 0.873], 0.719 | 0.840 | 0.763 |

The two targets correlate at r = 0.685, so they are different quantities.

**What it means.** The manuscript's "r > 0.9" holds for the loss with all patches visible. It does not hold for the 75%-masked training loss (0.85). A Ridge on 20 band statistics already reaches r = 0.978 for the unmasked loss, so the class token adds little there; for the masked loss it adds 0.09 in r.

**Latency.** The probe is a 1,024-term dot product once the scaler is folded into the weights: 0.42 us per chip batched in float32 (0.82 us in float64, 2.3 us for one vector from numpy; 100,000 synthetic chips, 1 BLAS thread, the host was heavily loaded). scikit-learn's `predict` costs 27 us per chip batched and 182 us for a single vector. For comparison the encoder forward costs 67 ms and the decoder 27 ms per chip on an RTX 4090. The "about 1 us per chip" figure therefore means the batched dot product after the CLS vector exists; it excludes producing the CLS vector.

**Caveats.** One sensor (Sentinel-2), one chip size (256 px), CONUS-weighted sampling, one mask seed for the masked target. The probe fits a quantity of the released checkpoint on these chips; it does not show that the quantity is a calibrated out-of-distribution score.
