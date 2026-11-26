# FarSLIP in the SatML harness (our run)

`harness_farslip.csv` is the result file written by `torchgeo/geo-eval-harness` for a model that is not on the
leaderboard, copied unchanged from our local harness branch. All 15 core datasets, kNN-5 and linear probe.

| Field | Value |
|---|---|
| Harness base | `55ef3c3042e63dfdd3c8102e379148e3aad46a4c` (main, 2026-11-17), 8 commits after the leaderboard snapshot `83ca5cb` |
| Our branch | `add-farslip`, commit `1a301b83420747dd8eabaebd9777546fdf10a655` (not pushed, no pull request yet) |
| Model | FarSLIP stage-2 ViT-B/16 image tower, `FarSLIP2_ViT-B-16.pt` from `ZhenShiL/FarSLIP`, Hub commit `9742933cfc3e7e62f6674cdb15ead7d1a9d148f3`; 512-d projected embedding |
| Settings | RGB bands, dataset normalization only, 224 px bilinear, seed 0, 200 bootstraps, harness defaults for kNN-5 and linear |
| Not run | the ViT-B/32 and stage-1 checkpoints; no larger FarSLIP model is public |
| Copied here | 2026-11-19 |

Used by: Table 2, Best other EO on `aid` (97.95) and `resisc45` (95.22), marked with a dagger
(`experiments/table2/build_cells_csv.py`, `OWN_HARNESS`), and one sentence in Section 3.1. The leaderboard keeps the
better of two normalizations per model; this run has one, so it is not favoured.
