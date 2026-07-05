#!/usr/bin/env bash
# Online cache replay with periodic scorer retraining: retrain vs no-retrain
# baseline on the same stream. CPU-only (labels from NPZ ground truth).
#
# Submit one ordering per job:
#   ORDERING=natural       OUTPUT_DIR=runs_adaptive/retrain_natural  sbatch run_retrain_replay.sbatch
#   ORDERING=cluster_block OUTPUT_DIR=runs_adaptive/retrain_cluster  sbatch run_retrain_replay.sbatch

#SBATCH --job-name=mlcache_retrain
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=8:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -euo pipefail
REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
cd "$REPO_ROOT"
PYTHON_BIN="${REPO_ROOT}/.conda/bin/python"
export LD_LIBRARY_PATH="${REPO_ROOT}/.conda/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${REPO_ROOT}/.cache}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-8}}"
mkdir -p "${XDG_CACHE_HOME}"

NPZ="${NPZ:-data/h1h0_final.npz}"
ORDERING="${ORDERING:-natural}"
OUTPUT_DIR="${OUTPUT_DIR:-runs_adaptive/retrain_${ORDERING}}"
MAX_ROWS="${MAX_ROWS:-100000}"
ALPHA="${ALPHA:-0.05}"
MEMBERS="${MEMBERS:-cosine,lda,pca_whitened_cosine,xgboost,mlp}"
RETRAIN_EVERY_X="${RETRAIN_EVERY_X:-2000}"
NEW_DATA_PCT="${NEW_DATA_PCT:-0.5}"
CHANGE_UNIT="${CHANGE_UNIT:-miss}"
WARMUP_FRAC="${WARMUP_FRAC:-0.3}"
POOL_CAP="${POOL_CAP:-40000}"
CALIB_WINDOW="${CALIB_WINDOW:-6000}"
SEED="${SEED:-42}"

echo "Host: $(hostname)"; echo "ORDERING=${ORDERING} X=${RETRAIN_EVERY_X} Y=${NEW_DATA_PCT} unit=${CHANGE_UNIT} MEMBERS=${MEMBERS}"
date --iso-8601=seconds

"${PYTHON_BIN}" scripts/run_retrain_replay.py \
    --npz "${NPZ}" --output-dir "${OUTPUT_DIR}" \
    --max-rows "${MAX_ROWS}" --alpha "${ALPHA}" \
    --ensemble-members "${MEMBERS}" --ordering "${ORDERING}" \
    --retrain-every-x "${RETRAIN_EVERY_X}" --new-data-pct "${NEW_DATA_PCT}" \
    --change-unit "${CHANGE_UNIT}" --warmup-frac "${WARMUP_FRAC}" \
    --pool-cap "${POOL_CAP}" --calib-window "${CALIB_WINDOW}" \
    --seed "${SEED}" --progress-every 10000

date --iso-8601=seconds
echo "Retrain replay finished -> ${OUTPUT_DIR}"
