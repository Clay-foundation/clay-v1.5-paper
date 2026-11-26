# Reproducing the Clay v1.5 paper

Everything runs from the repository root. Nothing here uses an absolute path; inputs and outputs are CLI flags or
environment variables with the defaults shown.

## 1. Environment (`make env`)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh     # uv (tested with 0.13.5)
make env                                            # = uv sync --frozen --extra viz  (Python 3.12, torch 2.14.1 + CUDA 13.0 wheels, umap-learn)
```

Versions are frozen so that results can be reproduced. Three pins were raised on 2026-11-25 to clear security advisories: torch 2.5.1 → 2.14.1 (torchvision 0.29.2), geopandas 1.1.1 → 1.1.2 and pyarrow 21 → 23.0.1. The published runs used torch 2.5.1. With 2.14.1, `make smoke` reproduces the published teacher-alignment cosines to six decimals (0.871665 and 0.718149), and `make tables` gives the same output as the old pins. The experiment runners were not re-run on 2.14.1. Requires an NVIDIA driver ≥ 580 for CUDA.

`pyproject.toml` pins every direct dependency and `uv.lock` pins the full resolution (Linux x86-64 and macOS arm64).
The default group runs every Clay, DINOv2, DOFA and TorchGeo-based experiment. The OlmoEarth comparator needs
`torch >= 2.9, < 2.10` and therefore lives in a separate, conflicting group:

```bash
make olmo-env        # uv sync --frozen --no-default-groups --group olmo
```

What was and was not verified (2026-11-17, one Linux host, RTX 4090 24 GB, driver 580, CUDA 12.x):

| Item | Status |
|---|---|
| `uv lock --check`, `uv sync --frozen` from a clean copy of the repo | verified |
| `make smoke` on CPU and `smoke_test.py --device cuda` | verified (below) |
| `inspect_remote_checkpoint.py` in the pinned environment | verified |
| Experiment runners (`make e1` ... `e8`, `standard-benchmarks`, ...) in this consolidated environment | **not re-run by the stream that wrote this file**; the published runs used three separate environments (a Python 3.11 conda env with torch 2.5.1 / numpy 1.26, a Python 3.12 env with torch 2.7.1 / numpy 2.3.1, and a Python 3.11 virtualenv for OlmoEarth with `olmoearth-pretrain` installed without its dependency pins). Library-version differences can move numbers in the last digit. |
| `olmo` group (torch 2.9) | resolves; **not run**. The original OlmoEarth runs used torch 2.5.1 plus two import shims in `run_geobench_and_e7_complete.py`. |
| macOS arm64 | resolves in the lock; **not run** |

## 2. Smoke test: reproduce one published number on CPU (`make smoke`)

```bash
make smoke      # downloads the 5.2 GB checkpoint on first use (pinned revision, SHA-256 checked), then runs on CPU
```

About one minute of compute on an idle 32-thread CPU, 4 min 43 s measured from a clean copy on a loaded host (checkpoint download excluded). It checks, from public inputs only
(Hugging Face checkpoint, Sentinel-2 L2A from the Planetary Computer, DINOv2 weights from Hugging Face):

* SHA-256 `21432069...98d0`, strict load of 265 encoder tensors and 311,442,944 parameters (paper Sec. 2.3);
* the projected class token aligns with DINOv2 on the same chip: mean cosine 0.87 on the diagonal against 0.72 off the
  diagonal, as quoted in `experiments/suite/PROTOCOL-COMMON.md`. The result is written to
  `experiments/results/smoke_test.json` (0.872 / 0.718 on CPU, 0.872 / 0.718 on GPU).

The alignment value depends on the scene. The pinned scene (`S2A_MSIL2A_20230827T184921_R113_T10SEH_20241024T140931`)
reproduces the published pair; another clear-sky summer 2023 Bay Area scene gives 0.82 / 0.72 and a water-heavy tile
gives 0.65 / 0.57. Pass `--stac-item` to try another scene; the pass/fail check applies to the pinned one only.

`make verify-vendored` checks the vendored Clay encoder code against `Clay-foundation/model` at commit `f14e698`
(see `experiments/suite/common/claymodel/NOTICE`).

## 3. Paper build (`make paper`)

```bash
make paper        # tectonic build, all gates, deterministic tarball, pdflatex check of the tarball
make paper-fast   # same, but the pdflatex check is recorded as skipped
```

Needs `tectonic`, `pdftotext` and `pdfinfo` (poppler) on PATH. The arXiv engine check needs `pdflatex` or `docker`
(`TEXLIVE_IMAGE`, default `texlive/texlive:latest`; `DOCKER_HOST=ssh://host` works because the tarball is streamed
through stdin). Gates (see the `build.py` docstring): placeholders, DRAFT.md-to-LaTeX parity including Table 2 row
labels, undefined references, overfull boxes above 2 pt, section order, ligatures, `??`, bibliography count
against the compiled `.bbl`, main-text length measured at the References heading, tarball compile under tectonic and
pdflatex. A missing tool is an error. `SOURCE_DATE_EPOCH` (default: last commit time) makes the PDF and the tarball
reproducible: two builds in different directories gave identical SHA-256 for both. The manifest is
`experiments/results/build_manifest.json`.

## 4. Tables (`make tables`)

* Table 1: `experiments/table1/build_table1.py` and `check_carbon.py` regenerate the corpus and carbon blocks of
  `experiments/results/corpus_and_carbon_provenance.json`.
* Table 2: `experiments/table2/table2_cells.csv` holds one row per cell (panel, task, model, slot, value, metric,
  source, result file and key, readout, head, seeds, notes); `experiments/table2/summarize.py` prints every aggregate
  the paper quotes (rule stated in its docstring). `experiments/table2/check_cells.py` resolves every linked result
  file and key and compares the JSON value with the printed value at the printed precision (102 linked cells and 49
  leaderboard cells, 0 failures in this revision). `make tables` runs all four scripts. See `experiments/table2/README.md`.

## 5. Experiment families

Reference host for the stored runs: 1x RTX 4090 24 GB, 32 threads, 128 GB RAM. "Not recorded" means the repository
holds no wall-clock measurement; do not read it as fast. Each family has a frozen `PROTOCOL.md` and a `RESULTS.md`
under `experiments/suite/<family>/`. Script flags are documented in each script's docstring (`make <t> ARGS="--help"`).

| Make target | What it produces | GPU | Input data (public source) | Runtime |
|---|---|---|---|---|
| `standard-benchmarks` | Table 2 panels A, B, D: 12 tasks, Clay, DOFA-L, DINOv2 etc. | yes | GEO-Bench 1.0 (HF `recursix/geo-bench-1.0`, revision pinned in the script), TorchGeo datasets (UCMerced, RESISC45, ...) downloaded on demand | stored run: 0.92 h of summed task time (`standard_benchmarks.json`) |
| `table2-peers` | peer-model cells on the same tasks | yes | same as above plus released peer weights (OlmoEarth v1.3, Panopticon, Copernicus-FM) from Hugging Face | not recorded |
| `table2-geobench` | GEO-Bench and Sen1Floods11 UPerNet cells | yes | GEO-Bench, Sen1Floods11 | not recorded |
| `probe-lsvm` | Lesson 4 probe family and concept-search noise | optional | EuroSAT features (`CLAY_EUROSAT_FEAT_DIR`), S2-NLCD chips (`CLAY_S2_NLCD_ROOT`) | not recorded |
| `perturbations` | DynamicEmbedding, Matryoshka variance, nodata/radiometric perturbations | yes | checkpoint metadata from `make inspect`, S2-NLCD chips | not recorded |
| `e1` | label efficiency on the AlphaEarth 15-task suite | yes | AEF evaluation release (Zenodo 16585402); Sentinel-2 L2A from the Planetary Computer | fetch step is capped at 4.8 h in the protocol; total not recorded |
| `e2` | token anatomy (position, sensor, dense land cover, leakage) | yes | Planetary Computer chips at 3,509 points, ESA WorldCover | not recorded |
| `e2c` | position-code ablation, small MAEs trained from scratch, 3 seeds | yes (6 GB) | co-registered chips built by `common_data.py` | protocol estimate of about 3.4 h wall, extrapolated from a 200-step pilot (not a measured total) |
| `e3` | cross-sensor transfer (S2 / NAIP / S1 / Landsat) | yes | 6,592 co-located locations from Planetary Computer (NAIP, S2 L2A, S1 RTC, Landsat C2 L2), ESA WorldCover | not recorded |
| `e4` | retrieval at a review budget (Part A), released-embedding parity (Part B) | yes | USPVDB v4.0, three S2 tiles, Source Cooperative embedding catalogs | not recorded |
| `e5` | frozen Clay vs AEF, TESSERA and other embeddings on the AEF suite | yes for GPU arms | AEF annual embeddings (Source Cooperative), TESSERA v1.1 zarr, `olmo` group for OlmoEarth | not recorded |
| `e6` | four-date inputs vs one | yes | Planetary Computer S2 (4 dates, 30,424 chips) | fetch capped at 6 h per job in the protocol |
| `e7` | Sen1Floods11 frozen vs fine-tuned, 3 seeds | yes (<= 10 GB) | Sen1Floods11 hand-labelled split | not recorded |
| `e8` | reviewer controls (context-matched cross-sensor, readouts) | yes | as E3 | not recorded |

Scripts added or changed for the rerun revision. All take their arguments from the command line (`--help` prints the
real flags; the commands below are the ones in each docstring). `make <target> ARGS="..."` runs the same command.

| Make target | Command | Purpose |
|---|---|---|
| `clay-missing-cells` | `python experiments/run_clay_missing_cells.py --cells lp_bigearthnet,lp_cashew --out OUT.json` (fine-tune cells add `--seeds 0,1,2`, `--epochs N`, `--tag _long`; `--cells finalize` runs the budget check last) | Clay probes and fine-tunes for the Table 2 cells that were transcribed from the OlmoEarth report; writes `clay_missing_cells.json`. Needs `CLAY_V15_CKPT`, `CLAY_SCRATCH`; GPU for `ft_*` and `oe_contract` |
| `clay-missing-cells` (fine-tunes printed in Table 2) | `python experiments/run_clay_missing_cells.py --cells ft_vessel-s1 --seeds 0 --epochs 48000 --eval-every 4000 --tag _48k --out OUT.json`; the same with `--cells ft_vessel-l8`; `--cells ft_m-cashew-plant --seeds 0 --epochs 100 --lr 0.0005 --tag _e100` | the three single-seed Clay fine-tune cells of Panel C updated on 2026-11-19 (`vessel-s1/FT_48k`, `vessel-l8/FT_48k`, `m-cashew-plant/FT_e100_lr0.0005` in `clay_missing_cells.json`; for the vessel cells `--epochs` counts steps). Training took 4.0 h, 6.0 h and 1.8 h on the reference host; each record stores these arguments and the SHA-256 of the script, which is the committed file |
| `merge-outputs` | `python experiments/merge_run_outputs.py --out experiments/results/all_doable_table2.json LABEL=run_a.json LABEL=run_b.json` | merges per-run JSONs of one runner without editing values; keeps each input's provenance under `_merged_from` |
| `e7-peer-seeds` | `python experiments/aggregate_e7_seeds.py --out OUT.json --seed-json run_s0.json --seed-json run_s1.json --seed-json run_s2.json` | per-seed Sen1Floods11 (S2) peer-arm JSONs from `run_geobench_and_e7_complete.py --parts e7` into the `remaining_all.json` layout |
| `lesson-inputs` | `python experiments/build_lesson_inputs.py s2-nlcd` and `python experiments/build_lesson_inputs.py eurosat --latlon-order documented` | the 223-scene NLCD chips and the EuroSAT class-token features that Lessons 3 to 5 read. `eurosat` needs `EUROSAT_ZIP` and `CLAY_V15_CKPT` |
| `lessons-summary` | `python experiments/summarize_lessons_345.py` | collects the Lesson 3 to 5 numbers next to the pre-audit values into `lessons_345_old_vs_new.json` (CPU, reads `experiments/results/`) |
| `lesson-controls` | `python experiments/issue22_s1_s2_cosine.py`; `python experiments/probe_equal_tuning_control.py --feat-dir data/eurosat/features --seeds 3`; `python experiments/fold_sensitivity_nlcd.py --n-assign 100` | S1/S2 kernel cosine definitions (about 1 min), equal-tuning control for RF / HistGBDT / k-NN (CPU), NLCD fold-assignment sensitivity (CPU, about 15 min) |
| `e4-summary` | `python experiments/suite/E4_retrieval_parity/summarize_partA.py --results-a resultsA.json [--write]` after `python partA.py --stage embed\|nometa\|train\|score` run from `experiments/suite/E4_retrieval_parity/` | USPVDB retrieval headline numbers (Part A) |
| `e5-panelc` | `python experiments/suite/E5_sota_comparators/code/run_e1_e5_complete.py --data_root DATA --out OUT/e1_e5_complete.json`, then `python experiments/suite/E5_sota_comparators/code/summarize_panelC.py --results OUT` | Panel C and OpenET on matched rows from cached features (CPU, about 40 min on a loaded 32-thread host). `--merge part_a.json part_b.json --out OUT.json` joins per-task-group runs |
| `e6-gap` | `python experiments/suite/E6_multidate/code/gap_closed_matched.py --data_root DATA --out OUT/e6_gap_closed_matched.json` | gap closed by four dates with one readout, a named comparator and matched rows (CPU) |
| `e7-analyze` | `python experiments/suite/E7_finetune/code/analyze_e7.py RES_DIR OUT.json CHIPS.json` | pooled water IoU, paired contrasts and training cost from the per-seed `.npz` files written by `train_e7.py` |
| `elle` | `python experiments/suite/ELLE_self_loss/elle_probe.py --chips DIR_WITH_s2_npz --out experiments/results/elle_probe.json` | ELLE probe and latency; `latency_bench.py` in the same folder writes `elle_latency_microbench.json` |
| `tables` (part) | `python experiments/table2/check_cells.py` | checks every linked Table 2 cell against its JSON leaf |

Inputs that are not redistributable here (catalog exports, label files) are fetched from the sources above by each
family's `fetch*` script; the repository ships only compact result JSONs under `experiments/results/`.

## 6. Where outputs go, and what is "fresh"

Every new results JSON written by the scripts touched in this release carries a `provenance` block: script name, git
commit, arguments (absolute paths scrubbed), package versions, UTC time (`experiments/suite/common/provenance.py`).
Compare that block with the file you expect before trusting a JSON as a fresh run. The checkpoint inspector writes to
`experiments/results/` by default (`CLAY_INSPECT_OUT` or `--out` to change it), which is where
`run_clay_v15_experiments.py` reads `clay-v1.5-saved-metadata.json`.

## 7. What was regenerated in this revision

The numbers added to the manuscript in this revision come from reruns on the reference host. Some of them reuse cached
inputs; the cached inputs themselves were not regenerated.

* Regenerated from cached embeddings or features: E3 (cross-sensor transfer) and E2 (token anatomy) were re-scored from
  their cached embeddings; the embeddings were not regenerated, because the Sentinel-2, NAIP, Landsat and Sentinel-1
  reads take hours. Panel C (`run_e1_e5_complete.py`, `gap_closed_matched.py`) was recomputed on matched rows from the
  cached E1 chip features and the cached OlmoEarth v1.3 and Prithvi-EO-2.0 features (`embed_gpu_arms.py` was not rerun); the
  AlphaEarth and TESSERA point features are joined at the sampled point-years, not at the exact label years from the
  AlphaEarth `evaluation.zip`.
* Regenerated from the inputs: the standard benchmarks, the peer cells in `all_doable_table2.json`, the Clay cells in
  `clay_missing_cells.json`, E4 Part A, the Lesson 3 to 5 suites and the ELLE probe. In E7 the Clay frozen linear
  head (seeds 1 and 2), the frozen UPerNet and the U-Net were retrained; the Clay fine-tune and DOFA-L arms are the
  original runs, the only per-seed outputs that survive.
* Table 2 as printed now (`table2_cells.csv`, 2026-11-20): 102 cells from our runs (`rerun_here`), 2 from our runs in the
  leaderboard harness (`own_harness`), and 74 third-party cells that were not re-run (49 read from the SatML leaderboard,
  24 from the OlmoEarth v1.3 report, 1 from the AlphaEarth paper). No printed cell is marked `not_regenerated`; the five
  DOFA-L cells that were (key `_not_regenerated/…` in `remaining_all.json`) are no longer in the table.
* Not run: Clay fine-tune seeds 1 and 2 for `m-forestnet`, `m-cashewplant` and `Sen1Floods11 (S1)`, seed 2 for
  `m-bigearthnet`, and the 8-epoch fine-tunes of `m-so2sat` and `m-bigearthnet`. Of these rows, Table 2 now
  prints only the `m-cashewplant` fine-tune, from one seed at 100 epochs. The other single-seed Clay fine-tunes it prints
  are `Vessel Detect. (S1)` and `(L8)` (48,000 steps) and `Solar Farm Detect.` (8,000 steps); `MADOS` uses 3 seeds.
* Results JSONs record `git_commit` as the base commit with a note where the run host held a plain copy of the tree. Rerun
  from the committed tree if exact commit provenance is needed.

## 8. Replication in a container

```bash
docker build -t clay-v15-paper .
docker run --gpus all clay-v15-paper make repro-smoke
```

The host needs an NVIDIA driver, the NVIDIA container toolkit, about 16 GB of RAM and about 60 GB of free disk (the image is
about 22 GB on disk: two Python environments, because the OlmoEarth comparator needs torch >= 2.9; the run itself adds about 20 GB
of downloads). Mount an empty directory at `/cache` (`-v "$PWD/cache:/cache"`) if you want the 5.2 GB checkpoint and the
datasets to survive the container; nothing else needs to be mounted. The run was tested on one AWS g5.xlarge (4 vCPU, 16 GB,
one NVIDIA A10G with 24 GB) from an empty cache: 54 min 28 s for the run (32 PASS, 8 SKIP, 2 FAIL: `perturbations`, which I aborted after 8 min, and `probe_lsvm`, which asserts the full EuroSAT feature set; the script now skips both unless `SMOKE_RUN_LESSONS=1`, see below). A cold `docker build` took 9 min 45 s.

`make repro-smoke` runs `experiments/repro_smoke.sh`: every experiment family end to end at the smallest budget that still
exercises the real code path (one small task, one seed, 1 to 20 training steps, 8 to 1,500 samples; the exceptions are the downloads and the 320-chip NLCD build). It prints PASS, FAIL or
SKIP with the wall time for each step and exits non-zero only on FAIL. The numbers it writes are **not results**: they only show
that the pipeline starts, runs and writes its outputs. Logs, outputs and `summary.json` go to
`experiments/runs/repro_smoke_<UTC>/` (git-ignored; `docker cp` them out or mount `/cache` and `experiments/runs`).

| Step(s) | What runs |
|---|---|
| `env_check`, `verify_vendored`, `ckpt_fetch`, `inspect_remote`, `smoke_test` | environment import and lock check, vendored Clay code check, checkpoint download with SHA-256, remote checkpoint inspection, `smoke_test.py --device cuda` (published 0.87 / 0.72 alignment pair) |
| `table1`, `table2_cells`, `lessons_summary`, `panelC_summary` | Table 1 and Table 2 regeneration and the cell check, Lesson 3 to 5 summary, Panel C summary; all write to the run directory, not to `experiments/results/` |
| `standard_benchmarks` | `m-eurosat` (GEO-Bench), `m-pv4ger-seg` (GEO-Bench segmentation), `eurosat-s1` (Copernicus-Bench), Clay, DOFA-L, Scale-MAE-L, DINOv2-L, band statistics, 128 samples per split (`--limit`) |
| `peers_table2`, `geobench_e7_complete` | OlmoEarth, Panopticon, Copernicus-FM on `m-eurosat`; DOFA-L fine-tune on `m-eurosat` and the DOFA-L frozen UPerNet arm on Sen1Floods11, 1 epoch (torch 2.9 environment) |
| `clay_missing_cells` | `lp_sen1f_s1` and `ft_m-eurosat` through the script's own `--smoke` flag |
| `e7_fetch`, `e7_train_*` | Sen1Floods11 download (public bucket); three of the five E7 arms (`clay_frozen_linear`, `clay_finetune_upernet`, `dofa_finetune_upernet`), 1 epoch, `--smoke` |
| `lesson_inputs_s2_nlcd`, `eurosat_zip`, `lesson_inputs_eurosat` | the inputs of the perturbation and probe suites: 320 NLCD chips from the Planetary Computer (hash-checked), EuroSAT-MS from Zenodo (2 GB, downloaded in the background from the start of the run) and 1,500 of its chips embedded with Clay |
| `e3_*`, `e8_build` | E3: 40 candidate points (about 22 complete tiles, 176 locations) built, embedded, evaluated (class thresholds lowered to 3 train / 1 test locations and 20 bootstrap draws through `E3_MIN_TRAIN`, `E3_MIN_TEST`, `CLAY_NBOOT`); E8: context-matched build for the first two location groups |
| `e2_*`, `e2c_*`, `elle_probe` | E2: 16 points fetched, embedded, probed (`--quick`); E2c: three small MAEs, 20 steps, re-embedded; ELLE probe on the 16 Sentinel-2 chips |
| `e1_fetch_embed`, `e6_fetch_embed` | the E1 and E6 fetch-and-embed code on 12 chips taken from the E2 point list (the original request files are not in the repository) |

What is skipped, and the exact missing input:

* `perturbations` (`run_clay_v15_experiments.py`) and `probe_lsvm` (`run_probe_and_lsvm_benchmark.py`): no budget flag. The stored runs took 2,825 s and 2,394 s on 32 threads; both are CPU-bound, and an 8 min trial of `perturbations` on the 4-vCPU test host printed nothing before it was aborted. The probe runner also asserts all 25,648 EuroSAT feature rows. Opt in with `SMOKE_RUN_LESSONS=1` (hours on a small host); the inputs above are built either way.
* `e7_analyze`: needs the full per-seed outputs (3 seeds, 100 epochs) of `train_e7.py`.
* `e8_embed_eval`: needs the cached E3 embeddings of all 6,592 locations (`E3_EMB`, `E3_EMB_TR`, `E3_SUB`), which come from a full E3 run.
* `e2c_pool_seeds`: needs `probe_s0..2.json` from three full E2c training seeds.
* `e1_eval_aef`, `e5_comparators`, `e6_eval`: need the AlphaEarth evaluation release (Zenodo 16585402), AlphaEarth and TESSERA embeddings, and the cached E1 / E6 chip features.
* `e4_retrieval`: `partA.py` has no budget flag (three full Sentinel-2 tiles) and needs the USPVDB v4.0 GeoJSON, which is not in the repository.
* `paper_build`: needs tectonic, poppler and TeX Live (`make paper`).

Changes that the smoke run needed, all inert unless the variable or flag is set: `--limit` / `CLAY_SMOKE_LIMIT` in
`run_standard_benchmarks.py` (also read by the runners that import its loaders), `--limit` in `build_lesson_inputs.py eurosat`,
`E3_MIN_TRAIN`, `E3_MIN_TEST` in the E3 evaluator, `CLAY_NBOOT` in `evalkit.py`. Two environment gaps found by the run are fixed
in `pyproject.toml`, `uv.lock` and the `env` target: `pyarrow` (pandas parquet engine, missing before) and `umap-learn`
(`--extra viz`, imported unconditionally by `run_probe_and_lsvm_benchmark.py`). `E2_tokens/code/fetch.py` now lists the WorldCover
tiles in 24 spatial queries because the Planetary Computer STAC API returned `upstream request timeout` for page 2 and later of the
single listing on the test day.

`make repro-full` chains the existing full targets (`smoke tables standard-benchmarks table2-peers table2-geobench probe-lsvm
perturbations e1 e2 e2c e3 e4 e5 e6 e7 e8`) in order. It is many GPU-hours (see the runtime column in section 5), several targets
need `ARGS` or cached inputs, and it was not run.
