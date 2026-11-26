# Reproduction entry points for the Clay v1.5 paper. See REPRODUCING.md for details.
#   make env      pinned environment (uv, Python 3.12, torch 2.14.1 / CUDA 13.0 wheels, umap-learn)
#   make smoke    CPU, minutes: verified checkpoint + published teacher-alignment numbers
#   make paper    build PDF + deterministic arXiv tarball with all gates
#   make tables   Table 1 / Table 2 summary statistics from results JSON / CSV, and the Table 2 cell check
#   make e1 ... e8, standard-benchmarks, ...   experiment families (GPU, hours; see table below)
#   make repro-smoke   every family end to end on a tiny budget (GPU, about an hour; PASS/FAIL/SKIP per step)
#   make repro-full    chains the full targets above (many GPU-hours; see REPRODUCING.md)
# Extra arguments go through ARGS, e.g.  make e5 ARGS="--help"
SHELL := /bin/bash
UV     ?= uv
PY     ?= $(UV) run --frozen python
CKPT   ?= checkpoints/v1.5/clay-v1.5.ckpt
export CLAY_V15_CKPT ?= $(CKPT)
ARGS   ?=

.PHONY: env ckpt smoke verify-vendored paper paper-fast tables inspect \
        standard-benchmarks table2-peers table2-geobench probe-lsvm perturbations e1 e2 e2c e3 e4 e5 e6 e7 e8 olmo-env \
        clay-missing-cells merge-outputs e7-analyze e7-peer-seeds lesson-inputs lessons-summary lesson-controls e4-summary e5-panelc e6-gap elle \
        repro-smoke repro-full

env:                                     ## ~2 GB download; CPU/GPU agnostic (CUDA wheels are bundled on Linux); --extra viz: run_probe_and_lsvm_benchmark.py imports umap
	$(UV) sync --frozen --extra viz

$(CKPT):                                 ## 5.2 GB from Hugging Face, pinned revision, SHA-256 checked
	$(PY) experiments/suite/common/fetch_checkpoint.py --dest $(patsubst %/v1.5/clay-v1.5.ckpt,%,$(CKPT))
ckpt: $(CKPT)

verify-vendored:                         ## vendored claymodel files vs MANIFEST.json (no network)
	python3 experiments/suite/common/claymodel/verify_vendored.py

# --- smoke: CPU only, about 1 minute on a 32-thread host plus the 5.2 GB checkpoint download.
#     Reproduces: checkpoint SHA-256 + 265 tensors / 311,442,944 params; DINOv2-alignment 0.87 vs 0.72 on the pinned scene.
smoke: $(CKPT) verify-vendored
	CUDA_VISIBLE_DEVICES= $(PY) experiments/suite/common/smoke_test.py --device cpu --ckpt $(CKPT)

inspect:                                 ## HTTP-Range inspection of the public checkpoints (no weights downloaded, ~1 MB)
	$(PY) experiments/inspect_remote_checkpoint.py

# --- paper
paper:                                   ## needs tectonic, poppler (pdftotext, pdfinfo) and pdflatex or docker
	python3 build.py $(ARGS)
paper-fast:                              ## same gates, but record (not run) the pdflatex tarball check
	python3 build.py --arxiv-check skip $(ARGS)

# --- tables (no GPU, no network except Table 1 fetching the pinned upstream document)
tables: verify-vendored
	$(PY) experiments/table1/build_table1.py
	$(PY) experiments/table1/check_carbon.py
	$(PY) experiments/table2/summarize.py --json experiments/results/table2_aggregates.json
	$(PY) experiments/table2/check_cells.py

# --- experiment families. Runtime/GPU/input columns are in REPRODUCING.md.
standard-benchmarks:                     ## Table 2 panels A/B/D, 12 tasks (GPU; 0.92 h of task time on an RTX 4090 in the stored run)
	$(PY) experiments/run_standard_benchmarks.py $(ARGS)
table2-peers:                            ## peer models on the same tasks (GPU; peer env, see olmo-env for OlmoEarth)
	$(PY) experiments/run_all_doable_table2.py $(ARGS)
table2-geobench:                         ## GEO-Bench + Sen1Floods11 UPerNet (needs the olmo group, see olmo-env)
	$(PY) experiments/run_geobench_and_e7_complete.py $(ARGS)
probe-lsvm:                              ## Lesson 4 probe family + concept-search noise (CPU/GPU)
	$(PY) experiments/run_probe_and_lsvm_benchmark.py $(ARGS)
perturbations:                           ## DynamicEmbedding / Matryoshka / nodata perturbations (GPU; needs inspect first)
	$(PY) experiments/run_clay_v15_experiments.py $(ARGS)
clay-missing-cells:                      ## Clay cells of Table 2 (GPU); needs ARGS="--cells <ids> --out <json>", see the script docstring
	$(PY) experiments/run_clay_missing_cells.py $(ARGS)
merge-outputs:                           ## merge per-run result JSONs without editing values; ARGS="--out OUT.json LABEL=in.json ..."
	$(PY) experiments/merge_run_outputs.py $(ARGS)
e7-peer-seeds:                           ## per-seed Sen1Floods11 peer-arm JSONs into the remaining_all.json layout; ARGS="--out OUT.json --seed-json s0.json ..."
	$(PY) experiments/aggregate_e7_seeds.py $(ARGS)
lesson-inputs:                           ## inputs of Lessons 3-5; ARGS="s2-nlcd" or ARGS="eurosat --latlon-order documented"
	$(PY) experiments/build_lesson_inputs.py $(ARGS)
lessons-summary:                         ## Lesson 3-5 numbers next to pre-audit values (CPU, reads experiments/results/)
	$(PY) experiments/summarize_lessons_345.py
lesson-controls:                         ## issue 22 cosine definitions, Lesson 4 equal-tuning control, NLCD fold sensitivity (CPU)
	$(PY) experiments/issue22_s1_s2_cosine.py
	$(PY) experiments/probe_equal_tuning_control.py $(ARGS)
	$(PY) experiments/fold_sensitivity_nlcd.py
e4-summary:                              ## merge Part A headline numbers into e4_retrieval_parity.json; ARGS="--results-a resultsA.json [--write]"
	$(PY) experiments/suite/E4_retrieval_parity/summarize_partA.py $(ARGS)
e5-panelc:                               ## Panel C and OpenET from cached features (CPU); ARGS="--data_root DATA --out OUT.json"
	$(PY) experiments/suite/E5_sota_comparators/code/run_e1_e5_complete.py $(ARGS)
	$(PY) experiments/suite/E5_sota_comparators/code/summarize_panelC.py
e6-gap:                                  ## gap closed by four dates on matched rows (CPU); ARGS="--data_root DATA --out OUT.json"
	$(PY) experiments/suite/E6_multidate/code/gap_closed_matched.py $(ARGS)
elle:                                    ## ELLE probe and latency; ARGS="--chips DIR_WITH_s2_npz"
	$(PY) experiments/suite/ELLE_self_loss/elle_probe.py $(ARGS)
olmo-env:                                ## separate environment (torch >= 2.9) for the OlmoEarth comparator only
	$(UV) sync --frozen --no-default-groups --group olmo

e1:                                      ## label efficiency, AEF 15-task suite
	$(PY) experiments/suite/E1_label_efficiency/code/fetch_embed.py $(ARGS)
	$(PY) experiments/suite/E1_label_efficiency/code/eval_aef.py
e2:                                      ## token anatomy
	$(PY) experiments/suite/E2_tokens/code/fetch.py $(ARGS)
	$(PY) experiments/suite/E2_tokens/code/embed.py
	$(PY) experiments/suite/E2_tokens/code/probe.py
e2c:                                     ## position-code ablation (trains small MAEs from scratch)
	$(PY) experiments/suite/E2c_posemb_ablation/code/train.py $(ARGS)
	$(PY) experiments/suite/E2c_posemb_ablation/code/embed_eval.py
	$(PY) experiments/suite/E2c_posemb_ablation/code/pool_seeds.py
e3:                                      ## cross-sensor transfer
	$(PY) experiments/suite/E3_cross_sensor/code/build.py $(ARGS)
	$(PY) experiments/suite/E3_cross_sensor/code/embed.py
	$(PY) experiments/suite/E3_cross_sensor/code/evaluate.py
e4:                                      ## retrieval at a review budget; parity of released embeddings
	$(PY) experiments/suite/E4_retrieval_parity/select_regions.py $(ARGS)
	$(PY) experiments/suite/E4_retrieval_parity/partA.py
	$(PY) experiments/suite/E4_retrieval_parity/partB.py
e5:                                      ## SOTA comparators on the AEF suite
	$(PY) experiments/suite/E5_sota_comparators/code/eval_e5.py $(ARGS)
e6:                                      ## multi-date
	$(PY) experiments/suite/E6_multidate/code/e6_fetch.py $(ARGS)
	$(PY) experiments/suite/E6_multidate/code/eval_e6.py
e7:                                      ## Sen1Floods11 fine-tuning vs frozen
	$(PY) experiments/suite/E7_finetune/code/fetch_data.py $(ARGS)
	$(PY) experiments/suite/E7_finetune/code/train_e7.py
e7-analyze:                              ## paired contrasts and training cost; ARGS="RES_DIR OUT.json CHIPS.json" (positional, see the script docstring)
	$(PY) experiments/suite/E7_finetune/code/analyze_e7.py $(ARGS)
e8:                                      ## reviewer controls
	$(PY) experiments/suite/E8_reviewer_controls/code/e8_build.py $(ARGS)
	$(PY) experiments/suite/E8_reviewer_controls/code/e8_embed.py
	$(PY) experiments/suite/E8_reviewer_controls/code/e8_eval.py

# --- containerized replication (see REPRODUCING.md, "Replication in a container")
repro-smoke:                             ## tiny end-to-end run of every family, PASS/FAIL/SKIP per step (GPU; about an hour)
	PY="$(PY)" bash experiments/repro_smoke.sh
repro-full: smoke tables standard-benchmarks table2-peers table2-geobench probe-lsvm perturbations e1 e2 e2c e3 e4 e5 e6 e7 e8  ## chains the full targets, in this order: many GPU-hours, not run in CI
