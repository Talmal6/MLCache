#!/usr/bin/env bash
# Offline adaptive-admission replay: fixed threshold vs Global ACI, for cosine
# AND WeightedEnsemble scorers, on the H1/H0 selected-candidate stream. CPU-only
# (labels come from the NPZ ground truth, not an LLM -- no vLLM/GPU needed).
#
# Submit with: sbatch run_adaptive_replay.sbatch
# Override scale/knobs via env, e.g.:  MAX_ROWS=60000 sbatch run_adaptive_replay.sbatch

#SBATCH --job-name=mlcache_adaptive_replay
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=6:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
cd "$REPO_ROOT"

PYTHON_BIN="${REPO_ROOT}/.conda/bin/python"

# Prefer the conda env's libstdc++ and keep all caches off the quota-limited home.
export LD_LIBRARY_PATH="${REPO_ROOT}/.conda/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${REPO_ROOT}/.cache}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-8}}"
mkdir -p "${XDG_CACHE_HOME}"

# ── Experiment knobs ─────────────────────────────────────────────────────
NPZ="${NPZ:-data/h1h0_final.npz}"
OUTPUT_DIR="${OUTPUT_DIR:-runs_adaptive/sbatch_progress_100}"
# Full labeled corpus is ~97k rows; cap here bounds record-build memory + time.
# Omit / set MAX_ROWS=0 to use the whole file.
MAX_ROWS="${MAX_ROWS:-100000}"
ALPHA="${ALPHA:-0.05}"
TOP_K="${TOP_K:-5}"
# Full member set so the ensemble is NOT cosine-equivalent (LDA/MLP/XGB pull it
# away from plain cosine ranking).
ENSEMBLE_MEMBERS="${ENSEMBLE_MEMBERS:-cosine,lda,pca_whitened_cosine,xgboost,mlp}"
ORDERING="${ORDERING:-random,natural,cluster_block}"
FIT_FRACTION="${FIT_FRACTION:-0.3}"
CALIB_FRACTION="${CALIB_FRACTION:-0.3}"
# ACI knobs
GAMMA="${GAMMA:-0.01}"
BUFFER_SIZE="${BUFFER_SIZE:-3000}"
MIN_BUFFER_SIZE="${MIN_BUFFER_SIZE:-300}"
P_CONTROL="${P_CONTROL:-0.04}"
SEED="${SEED:-42}"
SNAPSHOT_EVERY="${SNAPSHOT_EVERY:-100}"
PRINT_SNAPSHOTS="${PRINT_SNAPSHOTS:-1}"

echo "Host: $(hostname)"
echo "Python: ${PYTHON_BIN}"
echo "MAX_ROWS=${MAX_ROWS} ALPHA=${ALPHA} ORDERING=${ORDERING} MEMBERS=${ENSEMBLE_MEMBERS}"
echo "SNAPSHOT_EVERY=${SNAPSHOT_EVERY} PRINT_SNAPSHOTS=${PRINT_SNAPSHOTS}"
date --iso-8601=seconds

EXTRA=()
if [ "${MAX_ROWS}" != "0" ]; then
    EXTRA+=(--max-rows "${MAX_ROWS}")
fi
if [ "${PRINT_SNAPSHOTS}" != "0" ]; then
    EXTRA+=(--print-snapshots)
fi

"${PYTHON_BIN}" scripts/run_adaptive_replay.py \
    --npz "${NPZ}" \
    --output-dir "${OUTPUT_DIR}" \
    --alpha "${ALPHA}" \
    --top-k "${TOP_K}" \
    --ensemble-members "${ENSEMBLE_MEMBERS}" \
    --ordering "${ORDERING}" \
    --fit-fraction "${FIT_FRACTION}" \
    --calib-fraction "${CALIB_FRACTION}" \
    --gamma "${GAMMA}" \
    --buffer-size "${BUFFER_SIZE}" \
    --min-buffer-size "${MIN_BUFFER_SIZE}" \
    --p-control "${P_CONTROL}" \
    --seed "${SEED}" \
    --snapshot-every "${SNAPSHOT_EVERY}" \
    --progress-every 20000 \
    "${EXTRA[@]}"

date --iso-8601=seconds
echo "Adaptive replay finished successfully -> ${OUTPUT_DIR}"
