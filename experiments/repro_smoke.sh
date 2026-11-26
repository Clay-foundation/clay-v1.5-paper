#!/usr/bin/env bash
# Smoke replication: run every experiment family end to end on a tiny budget and report PASS / FAIL / SKIP.
#
#   make repro-smoke                (inside the container: docker run --gpus all <image> make repro-smoke)
#
# Each step prints its status and wall time. The script exits non-zero only if a step FAILs. A SKIP names the exact
# input that is missing (not public, or not in the repository); nothing is faked. Numbers written by these runs
# (one small task, one seed, 1-2 epochs, a few hundred samples) are NOT results; they only show the code path runs.
#
# Outputs: $SMOKE_DIR (default experiments/runs/repro_smoke_<UTC>/) with logs/<step>.log, out/<family>/..., and
# summary.json (also copied to experiments/runs/repro_smoke_latest.json).
# Environment: PY (interpreter command, default `uv run --frozen python`), OLMO_PY (interpreter of the torch>=2.9
# environment, default .venv-olmo/bin/python), CKPT, CLAY_SCRATCH, SMOKE_DIR.
set -u
cd "$(dirname "${BASH_SOURCE[0]}")/.."
ROOT=$PWD
PY=${PY:-uv run --frozen python}
OLMO_PY=${OLMO_PY:-$ROOT/.venv-olmo/bin/python}
export CKPT=${CKPT:-checkpoints/v1.5/clay-v1.5.ckpt}
export CLAY_V15_CKPT=${CLAY_V15_CKPT:-$CKPT}
SMOKE_DIR=${SMOKE_DIR:-experiments/runs/repro_smoke_$(date -u +%Y%m%dT%H%M%SZ)}
case $SMOKE_DIR in /*) ;; *) SMOKE_DIR=$ROOT/$SMOKE_DIR ;; esac
OUT=$SMOKE_DIR/out
mkdir -p "$SMOKE_DIR/logs" "$OUT"
export CLAY_SCRATCH=${CLAY_SCRATCH:-$SMOKE_DIR/scratch}
export CLAY_E7_DATA=${CLAY_E7_DATA:-$CLAY_SCRATCH/sen1floods11}
export CLAY_RUNS_ROOT=$OUT
export PYTHONPATH=$ROOT/experiments/suite/common:$ROOT/experiments/suite/E2c_posemb_ablation/code${PYTHONPATH:+:$PYTHONPATH}
mkdir -p "$CLAY_SCRATCH"
TSV=$SMOKE_DIR/summary.tsv
: > "$TSV"
N_FAIL=0
declare -A STATUS

have_gpu() { $PY -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" >/dev/null 2>&1; }
if have_gpu; then GPU=1; else GPU=0; fi

record() { # name status seconds reason
  printf '%s\t%s\t%s\t%s\n' "$1" "$2" "$3" "$4" >> "$TSV"
  STATUS[$1]=$2
  printf '%-34s %-5s %6ss  %s\n' "$1" "$2" "$3" "$4"
}
# step NAME CMD...   : run CMD in bash, log to logs/NAME.log, PASS on exit 0
step() {
  local name=$1; shift
  local t0=$SECONDS rc
  ( cd "$ROOT" && bash -c "$*" ) > "$SMOKE_DIR/logs/$name.log" 2>&1
  rc=$?
  if [ $rc -eq 0 ]; then record "$name" PASS $((SECONDS - t0)) ""
  else
    N_FAIL=$((N_FAIL + 1)); record "$name" FAIL $((SECONDS - t0)) "exit $rc; see logs/$name.log: $(tail -n 1 "$SMOKE_DIR/logs/$name.log" | cut -c1-160)"
  fi
}
skip() { record "$1" SKIP 0 "$2"; }
# gpu_step NAME CMD... : like step, SKIP when no CUDA device is visible
gpu_step() {
  local name=$1; shift
  if [ $GPU -eq 0 ]; then skip "$name" "no CUDA device visible (docker run --gpus all)"; else step "$name" "$@"; fi
}
# after NAME DEP... CMD: run only if every DEP passed, else SKIP naming the failed dependency
needs() { # needs NAME DEP : returns 0 when DEP passed
  [ "${STATUS[$2]:-}" = PASS ] || { skip "$1" "blocked: step $2 did not pass"; return 1; }
}
olmo_ok() { [ -x "$OLMO_PY" ] || { skip "$1" "olmo environment not built (UV_PROJECT_ENVIRONMENT=.venv-olmo make olmo-env; then OLMO_PY=.venv-olmo/bin/python)"; return 1; }; }

echo "== repro-smoke: output in $SMOKE_DIR (GPU visible: $GPU)"

# The EuroSAT-MS archive (2 GB, Zenodo 7711810) is slow to download; start it now and collect it at step eurosat_zip.
EUROSAT_ZIP=$CLAY_SCRATCH/EuroSAT_MS.zip
if [ ! -s "$EUROSAT_ZIP" ]; then
  ( curl -fsSL --retry 3 -o "$EUROSAT_ZIP.part" https://zenodo.org/records/7711810/files/EuroSAT_MS.zip && mv "$EUROSAT_ZIP.part" "$EUROSAT_ZIP" ) > "$SMOKE_DIR/logs/eurosat_zip_download.log" 2>&1 &
  ZIP_PID=$!
else ZIP_PID=""; fi

# ---------------------------------------------------------------- 0. environment, vendored code, checkpoint
step env_check "$PY -c \"import torch, torchgeo, timm, rasterio, sklearn, pandas; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'gpu', torch.cuda.is_available())\" && uv lock --check"
[ $GPU -eq 1 ] || skip cuda_available "no CUDA device visible (docker run --gpus all)"
step verify_vendored "make verify-vendored"
step ckpt_fetch "make ckpt"
if needs inspect_remote ckpt_fetch; then step inspect_remote "CLAY_INSPECT_OUT=$OUT/inspect $PY experiments/inspect_remote_checkpoint.py"; fi
if needs smoke_test ckpt_fetch; then
  gpu_step smoke_test "$PY experiments/suite/common/smoke_test.py --device cuda --ckpt $CKPT --out $OUT/smoke_test.json"
fi

# ---------------------------------------------------------------- 1. tables and cell checks (CPU, no committed file touched)
step table1 "mkdir -p $OUT/tables && cp experiments/results/corpus_and_carbon_provenance.json $OUT/tables/corpus_and_carbon.json && $PY experiments/table1/build_table1.py --json $OUT/tables/corpus_and_carbon.json && $PY experiments/table1/check_carbon.py --json $OUT/tables/corpus_and_carbon.json"
step table2_cells "$PY experiments/table2/summarize.py --json $OUT/tables/table2_aggregates.json > /dev/null && $PY experiments/table2/check_cells.py"
step zeroshot_scores "$PY experiments/zeroshot/zs.py score --task all > /dev/null && for m in claude-fable-5-2 gpt-6.2-astra gemini-4-carbon; do $PY experiments/zeroshot/score_full.py \$m > /dev/null || exit 1; done && $PY experiments/table2/refresh_zeroshot.py --check"
step lessons_summary "$PY experiments/summarize_lessons_345.py --results experiments/results"
step panelC_summary "$PY experiments/suite/E5_sota_comparators/code/summarize_panelC.py --json_out $OUT/tables/e5_panelC_summary.json > /dev/null"

# ---------------------------------------------------------------- 2. GPU families
SB_LIMIT=${SB_LIMIT:-128}
# standard benchmarks: one GEO-Bench classification task, one GEO-Bench segmentation task, one Copernicus-Bench task
if needs standard_benchmarks ckpt_fetch; then
  gpu_step standard_benchmarks "mkdir -p $OUT/std && $PY experiments/run_standard_benchmarks.py --output-dir $OUT/std --limit $SB_LIMIT --tasks m-eurosat,m-pv4ger-seg,eurosat-s1 && test -s $OUT/std/standard_benchmarks.json"
fi

# Table 2 peer models (OlmoEarth v1.3, Panopticon, Copernicus-FM, DOFA-L, DINOv2-L): torch >= 2.9 environment
if olmo_ok peers_table2; then
  gpu_step peers_table2 "mkdir -p $OUT/peers && CLAY_SMOKE_LIMIT=$SB_LIMIT $OLMO_PY experiments/run_all_doable_table2.py --output-dir $OUT/peers --parts tasks --tasks m-eurosat && test -s $OUT/peers/all_doable_table2.json"
fi

# Sen1Floods11 (public GCS bucket): input of E7, the GEO-Bench/E7 runner, and the Clay missing cells
step e7_fetch "$PY experiments/suite/E7_finetune/code/fetch_data.py"

# GEO-Bench fine-tune cell and Sen1Floods11 UPerNet arm (OlmoEarth/DOFA/DINOv2/Prithvi runner)
if olmo_ok geobench_e7_complete; then
  if needs geobench_e7_complete e7_fetch; then
    gpu_step geobench_e7_complete "mkdir -p $OUT/gbe7 && CLAY_SMOKE_LIMIT=$SB_LIMIT $OLMO_PY experiments/run_geobench_and_e7_complete.py --output-dir $OUT/gbe7 --parts gb,e7 --gb-cells m-eurosat --e7-epochs 1 --e7-seeds 0 --e7-val-every 1 --e7-arms dofa_frozen_upernet && test -s $OUT/gbe7/*.json"
  fi
fi

# Clay cells of Table 2 (its own --smoke flag truncates data and budget)
if needs clay_missing_cells e7_fetch; then
  gpu_step clay_missing_cells "$PY experiments/run_clay_missing_cells.py --cells lp_sen1f_s1,ft_m-eurosat --seeds 0 --epochs 1 --smoke --out $OUT/clay_missing_cells_smoke.json && test -s $OUT/clay_missing_cells_smoke.json"
fi

# Sen1Floods11 frozen vs fine-tuned (E7): three of the five arms, one seed, --smoke (3 batches per epoch), 1 epoch
if needs e7_train e7_fetch; then
  for arm in clay_frozen_linear clay_finetune_upernet dofa_finetune_upernet; do
    gpu_step e7_train_$arm "mkdir -p $OUT/e7 && $PY experiments/suite/E7_finetune/code/train_e7.py --arm $arm --seed 0 --epochs 1 --smoke --out $OUT/e7"
  done
fi
skip e7_analyze "full per-seed outputs (3 seeds x 100 epochs) of train_e7.py are the input; the smoke run writes 1-epoch files only"

# Lesson inputs (public: Planetary Computer chips with hash check; EuroSAT-MS zip from Zenodo) and the probe / LSVM / perturbation runs
step lesson_inputs_s2_nlcd "$PY experiments/build_lesson_inputs.py s2-nlcd --out $CLAY_SCRATCH/lessons"
step eurosat_zip "${ZIP_PID:+while kill -0 $ZIP_PID 2>/dev/null; do sleep 5; done; }test -s $EUROSAT_ZIP && echo \"EuroSAT_MS.zip ready (background download started at the top of the run)\""
if needs lesson_inputs_eurosat eurosat_zip && needs lesson_inputs_eurosat ckpt_fetch; then
  gpu_step lesson_inputs_eurosat "$PY experiments/build_lesson_inputs.py eurosat --out $CLAY_SCRATCH/lessons --zip $CLAY_SCRATCH/EuroSAT_MS.zip --limit 1500 --latlon-order documented"
fi
LESSON_ENV="CLAY_S2_NLCD_ROOT=$CLAY_SCRATCH/lessons/s2_nlcd_320 CLAY_EUROSAT_DIR=$CLAY_SCRATCH/lessons/eurosat CLAY_EUROSAT_ZIP=$CLAY_SCRATCH/EuroSAT_MS.zip CLAY_EUROSAT_FEAT_DIR=$CLAY_SCRATCH/lessons/eurosat/features"
# These two runners have no budget flag: the stored runs took 2,825 s (perturbations) and 2,394 s (probe / LSVM) on 32 threads, and
# are CPU-bound (spatial cross-validation of sklearn models on 320 + 1,500 chips); on the 4-vCPU test host a 8 min trial of
# `perturbations` produced no output and was aborted. They are opt-in: SMOKE_RUN_LESSONS=1 runs them in full (hours on 4 vCPUs).
if [ "${SMOKE_RUN_LESSONS:-0}" = 1 ]; then
  if needs perturbations lesson_inputs_s2_nlcd && needs perturbations inspect_remote; then
    gpu_step perturbations "$LESSON_ENV $PY experiments/run_clay_v15_experiments.py --out-name smoke_v15_clean_metrics.json && mv experiments/results/smoke_v15_clean_metrics.json $OUT/"
  fi
  # the probe runner asserts all 25,648 EuroSAT feature rows, so the 1,500-chip subset above is not enough for it
  if needs probe_lsvm lesson_inputs_s2_nlcd && needs probe_lsvm lesson_inputs_eurosat; then
    gpu_step lesson_inputs_eurosat_full "$PY experiments/build_lesson_inputs.py eurosat --out $CLAY_SCRATCH/lessons --zip $EUROSAT_ZIP --latlon-order documented"
    gpu_step probe_lsvm "$LESSON_ENV $PY experiments/run_probe_and_lsvm_benchmark.py --out-name smoke_probe_and_lsvm_results.json && mv experiments/results/smoke_probe_and_lsvm_results.json $OUT/"
  fi
else
  skip perturbations "no budget flag (stored run 2,825 s on 32 threads, CPU-bound spatial CV on 320 chips; opt in with SMOKE_RUN_LESSONS=1; inputs lesson_inputs_s2_nlcd are built above)"
  skip probe_lsvm "no budget flag (stored run 2,394 s on 32 threads, CPU-bound) and the runner asserts all 25,648 EuroSAT feature rows (the smoke builds a 1,500-chip subset); opt in with SMOKE_RUN_LESSONS=1"
fi

# ---------------------------------------------------------------- 3. Planetary Computer families (public STAC, tiny point counts)
MAN_E2=experiments/suite/E2_tokens/manifest.csv
# E3 cross-sensor transfer: 40 candidate points (about 22 yield a complete S2/NAIP/S1/Landsat tile), embed, evaluate.
# The evaluator keeps classes with >= 100 train / >= 20 test locations in the full study; the smoke run lowers that to 3 / 1
# and the bootstrap to 20 resamples (E3_MIN_TRAIN, E3_MIN_TEST, CLAY_NBOOT).
step e3_build "$PY experiments/suite/E3_cross_sensor/code/build.py 40 $OUT/e3/build 8 && test -s $OUT/e3/build/rows.json"
if needs e3_embed e3_build && needs e3_embed ckpt_fetch; then
  gpu_step e3_embed "$PY experiments/suite/E3_cross_sensor/code/embed.py $OUT/e3/build $OUT/e3/emb.npz"
  if needs e3_evaluate e3_embed; then
    step e3_evaluate "E3_MIN_TRAIN=3 E3_MIN_TEST=1 CLAY_NBOOT=20 NJ=4 $PY experiments/suite/E3_cross_sensor/code/evaluate.py $OUT/e3/build/rows.json $OUT/e3/emb.npz $OUT/e3/eval && ls $OUT/e3/eval | grep -q json"
  fi
fi
# E8 reviewer controls: context-matched S2/NAIP build for the first 2 location groups of the shipped E3 manifest
if needs e8_build e3_build; then
  step e8_build "$PY experiments/suite/E8_reviewer_controls/code/e8_build.py experiments/suite/E3_cross_sensor/manifest.csv $OUT/e3/build/tiles $OUT/e8/build 4 2 && test -s $OUT/e8/build/e8_chips.csv"
fi
skip e8_embed_eval "inputs not in the repo: cached E3 embeddings of all 6,592 locations (E3_EMB, E3_EMB_TR, E3_SUB) from a full E3 run (hours of Planetary Computer reads)"

# E2 token anatomy: 8 + 8 accepted points, embed, probe (--quick)
step e2_fetch "$PY experiments/suite/E2_tokens/code/fetch.py $OUT/e2/data --n-global 8 --n-conus 8 --workers 4"
if needs e2_embed e2_fetch && needs e2_embed ckpt_fetch; then
  gpu_step e2_embed "$PY experiments/suite/E2_tokens/code/embed.py $OUT/e2/data $OUT/e2/emb"
  if needs e2_probe e2_embed; then step e2_probe "$PY experiments/suite/E2_tokens/code/probe.py $OUT/e2/emb $OUT/e2/probe.json --quick && test -s $OUT/e2/probe.json"; fi
  # E2c position-code ablation: 3 small MAEs trained 20 steps on the E2 chips, then re-embedded
  gpu_step e2c_train "$PY experiments/suite/E2c_posemb_ablation/code/train.py $OUT/e2/data $OUT/e2c/ckpt 20 ABC 4"
  if needs e2c_embed_eval e2c_train && needs e2c_embed_eval e2_embed; then
    gpu_step e2c_embed_eval "$PY experiments/suite/E2c_posemb_ablation/code/embed_eval.py $OUT/e2/data $OUT/e2/emb $OUT/e2c/ckpt $OUT/e2c/eval"
  fi
  # ELLE probe on the E2 Sentinel-2 chips
  gpu_step elle_probe "$PY experiments/suite/ELLE_self_loss/elle_probe.py --chips $OUT/e2/data/chips/s2 --out $OUT/e2/elle_probe.json --n 16"
fi
skip e2c_pool_seeds "needs probe_s0..2.json from three full training seeds (E2C_SEED=0,1,2) of the 3.4 h E2c run"

# E1 label efficiency and E6 multi-date: fetch + embed 12 chips from the shipped E2 point list (the original request
# files come from the AlphaEarth evaluation release and are not in the repo)
python3 - "$MAN_E2" "$OUT" <<'PYEOF'
import csv, os, sys
rows = [r for r in csv.DictReader(open(sys.argv[1])) if r["sensor"] == "s2"][:12]
out = sys.argv[2]; os.makedirs(out + "/e1", exist_ok=True); os.makedirs(out + "/e6", exist_ok=True)
with open(out + "/e1/requests.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["chip_id", "lon", "lat", "t0", "t1"])
    for r in rows: w.writerow([r["pid"], r["lon"], r["lat"], "2021-05-01", "2021-09-30"])
with open(out + "/e6/requests.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["chip_id", "lon", "lat", "y0", "e1_q", "tier", "order"])
    for i, r in enumerate(rows): w.writerow([r["pid"], r["lon"], r["lat"], "2021-01-01", 2, 0, i])
PYEOF
if needs e1_fetch_embed ckpt_fetch; then
  gpu_step e1_fetch_embed "$PY experiments/suite/E1_label_efficiency/code/fetch_embed.py --requests $OUT/e1/requests.csv --out $OUT/e1/out --workers 4 --batch 8 --limit 12 && ls $OUT/e1/out | grep -q shard"
  gpu_step e6_fetch_embed "$PY experiments/suite/E6_multidate/code/e6_fetch.py --requests $OUT/e6/requests.csv --out $OUT/e6/out --workers 4 --batch 8 --limit 12 && ls $OUT/e6/out | grep -q shard"
fi
skip e1_eval_aef "inputs not in the repo: row-to-chip map and labels of the 15 AlphaEarth evaluation tasks (AEF evaluation release, Zenodo 16585402) and the AEF embedding shards"
skip e5_comparators "inputs not in the repo: AEF annual embeddings (Source Cooperative), TESSERA zarr, cached E1 chip features and the E5 point lists; panel C summary checked in panelC_summary from the committed JSON"
skip e6_eval "inputs not in the repo: cached E1 + E6 embedding shards for 30,424 chips and the AEF evaluation labels"
skip e4_retrieval "partA.py has no budget flag (3 full Sentinel-2 tiles at about 140 s read each plus embedding) and needs the USPVDB v4.0 GeoJSON (energy.usgs.gov, not in the repo); partB reads Source Cooperative embedding catalogs"
skip paper_build "needs tectonic, poppler and TeX Live; not part of the experiment smoke (make paper)"

# ---------------------------------------------------------------- summary
python3 - "$TSV" "$SMOKE_DIR/summary.json" <<'PYEOF'
import json, sys
rows = [l.rstrip("\n").split("\t") for l in open(sys.argv[1])]
steps = [{"step": r[0], "status": r[1], "seconds": int(r[2]), "reason": r[3]} for r in rows]
counts = {k: sum(s["status"] == k for s in steps) for k in ("PASS", "FAIL", "SKIP")}
json.dump({"counts": counts, "total_seconds": sum(s["seconds"] for s in steps), "steps": steps}, open(sys.argv[2], "w"), indent=1)
print("== summary:", counts, "total", sum(s["seconds"] for s in steps), "s ->", sys.argv[2])
PYEOF
mkdir -p "$ROOT/experiments/runs" && cp "$SMOKE_DIR/summary.json" "$ROOT/experiments/runs/repro_smoke_latest.json"
[ $N_FAIL -eq 0 ]
