# eurosat-v1.1 (carried-over scores, not regenerated)

`scores.csv` is the `appendixA_eurosat_repro.scores_rows` table that the 2026-11-03 run of
`run_clay_v15_experiments.py` stored in `v15_clean_metrics.json`. The cloud pipeline that produced it
(per-seed arms majority / spectral20 / geography_knn5 / spectral20_plus_xy2 / clay_cls / clay_mean on the
18,232 / 3,176 train / test split) is not part of this repository, and no `review-audit.json` was ever shipped.
`run_clay_v15_experiments.py` therefore records these rows with `source = "carried over, not regenerated"` and
`audit_passed = null` (the earlier flag defaulted to `true` when the file was absent).
The reproducible EuroSAT numbers are in `probe_and_lsvm_results.json` and in the E1 suite.
