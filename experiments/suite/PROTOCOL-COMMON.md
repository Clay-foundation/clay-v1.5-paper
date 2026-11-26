# Common protocol for the Clay v1.5 paper experiments (frozen 2026-11-05)

Applies to every experiment group (E1–E5). Each group also writes its own `PROTOCOL.md`, frozen and saved **before any downstream score is computed**. Its splits, label budgets, metrics and primary endpoint must not change after scores are seen. If something has to change, append a dated amendment that says what changed and why. Never delete or overwrite the original protocol.

## Model and runtime
- Clay v1.5 encoder from `common/clayrt.py`. Pinned source `Clay-foundation/model@f14e698`. Released checkpoint SHA-256 `21432069…98d0`, loaded strictly: 265 tensors, 311,442,944 parameters. Eval mode, no masking.
- Readouts: `cls` (primary), `mean` of patch tokens, and the `patches` grid where needed.
- Inputs: raw sensor values with the platform's mean/std, wavelengths and GSD from `common/configs/metadata.yaml`. Time and location come from the real acquisition; zeros mean unknown. Any departure (a different GSD, band subset or metadata off) is an explicit, named arm.
- Sentinel-2 L2A from Planetary Computer: feed raw values as Clay's training pipeline did. stacchip applies no harmonisation, so 2022+ scenes (processing baseline ≥ 04.00) carry the +1000 BOA offset in training too. Record each scene's `s2:processing_baseline` and report it as a covariate. Harmonisation, if tried, is a named sensitivity arm.
- Runtime check (2026-11-05, 16 real S2 chips): the projected class token aligns with DINOv2 on the same chip (mean cosine 0.87 vs 0.72 off-diagonal). Non-finite input pixels raise an error; mask nodata (uint16, nodata = 0) before embedding.
- **Lat/lon order (amendment 2026-11-05, before E1/E2 scores):** Clay v1.0/v1.5 were trained with longitude in the first two "latlon" slots: [sin lon, cos lon, sin lat, cos lat]. This is a stacchip unpacking swap from 2024-04-19, and the released embedding products match it. `clayrt` now defaults to the training order. The documented order [sin lat, …] is a named sensitivity arm, `latlon_order="documented"`. Any arm already run with metadata on in the documented order must be labelled as such and re-run in the training order.
- Comparators run through the same pipeline:
  - DINOv2 ViT-L/14-reg4 on display RGB (`DinoV2Runtime`, 224 px unless stated);
  - DOFA (torchgeo) where multispectral;
  - non-learned controls: per-band mean/std statistics, coordinates, majority class.

## Splits and statistics
- Spatial block splits only (`evalkit.hblock`, 100 km unless stated), or the dataset's own published geographic split. Random splits appear only as a named contrast arm.
- Label budgets are nested and hash-ranked (`evalkit.label_budget`), with 5 seeds.
- Fixed heads with no test-set tuning: logistic regression (C=1) or RidgeCV with inner CV on train only.
- Uncertainty: 1,000 block-bootstrap resamples (seed 1729), with paired contrasts sharing resamples.

## Outputs per group (in `experiments/suite/<E>/`)
- `PROTOCOL.md`, frozen before scores.
- `results.json`: every arm, budget, seed and metric, plus point estimates and CIs.
- `RESULTS.md`: at most one page, plain prose. State what was measured, the numbers, what they mean, and the caveats.
- One figure (PNG + PDF).
- `manifest.*`: the exact sample IDs and coordinates used, and the code.
- Negative and null results are reported with the same prominence as positive ones.

## Compute
- Reference hardware for the published runs: one 24 GB GPU and a multi-core CPU host; every job fits in 10 GB of GPU memory. Imagery is streamed and chipped, not mirrored; intermediate rasters are deleted after use.
- Environment, commands, expected runtimes and input sources: see `REPRODUCING.md` at the repository root (pinned by `pyproject.toml` / `uv.lock`).
