# EmbeddingGemma 2 in the SatML harness (our run)

`harness_embeddinggemma2.csv` (280 vision tokens, the model card's default; 12 of the 15 core datasets) and
`harness_embeddinggemma2_vt70.csv` (70 tokens; all 15) are the result files written by `torchgeo/geo-eval-harness`
for a model that is not on the leaderboard. They are copied unchanged from our local harness branch.

| Field | Value |
|---|---|
| Harness base | `55ef3c3042e63dfdd3c8102e379148e3aad46a4c` (main, 2026-11-17), 8 commits after the leaderboard snapshot `83ca5cb` |
| Our branch | `add-embeddinggemma2`, commit `9de7dd71cd29b25706b408877e82612d53fce729` (not pushed, no pull request yet) |
| Model | `google/embeddinggemma-2`, Hub commit `f9972ac22e696a105c482202646979e33e4b5a1e`, image only, 768-d output |
| Settings | RGB bands, dataset normalization only, 224 px bilinear, seed 0, 200 bootstraps, harness defaults for kNN-5 and linear |
| Not run at 280 tokens | `benv2`, `eurosat-spatial`, `resisc45` (the instance shut down); the text uses the 70-token value there |
| Our wrapper's choices | z-scored input clipped to mean +- 2 sigma per band and rounded to uint8 before the model's processor |
| Copied here | 2026-11-19 |

Used by: the `EG2` tag on `aid` in Table 2 (`experiments/table2/build_cells_csv.py`, `OWN_HARNESS`) and one sentence in
Section 3.1. The leaderboard keeps the better of two normalizations per model; this run has one, so it is not favoured.
