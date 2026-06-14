#!/usr/bin/env bash
# Boot a vLLM OpenAI-compatible server on the allocated GPU node, wait for it
# to become ready, run the MLCache LLM-judge replay experiment against it,
# and shut the server down cleanly on exit (success or failure).
#
# Submit with: sbatch run_llm_replay.sbatch

#SBATCH --job-name=mlcache_llm_replay
#SBATCH --gres=gpu:rtx_4090:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=48:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

# NOTE: job 18123438 was CANCELLED at exactly 08:00:00 by the OLD self-imposed
# time limit -- the GPU did NOT die; it was at request 39700/40000 of the
# ensemble system (the first of two) when killed, so write_artifacts never ran
# and nothing was saved. The partition max is 7 days, so we now request 48h.
# The experiment ALSO checkpoints (stable OUTPUT_DIR + judge cache below), so a
# future timeout/GPU death resumes cheaply instead of restarting from zero.

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
cd "$REPO_ROOT"

PYTHON_BIN="${REPO_ROOT}/.conda/bin/python"

# ── Experiment knobs ─────────────────────────────────────────────────────
NPZ="data/h1h0_final.npz"
# Stable output dir so a resubmitted job RESUMES (per-system result checkpoints
# + the persistent LLM-judge cache live here and are reused on restart). Do NOT
# key this on ${SLURM_JOB_ID} -- that changes every submission and defeats
# resume. Override by exporting OUTPUT_DIR to run an independent fresh experiment.
OUTPUT_DIR="${OUTPUT_DIR:-experiments/llm_judge_replay_v2}"
MODE="compare"
SEED=42
MAX_REQUESTS=40000
CACHE_SIZE=40000
TARGET_FPR=0.05
SELECTION="uniform"
SCORERS="cosine,lda,pca_whitened_cosine,xgboost,mlp"
PROGRESS_EVERY=100

# ── vLLM server config ───────────────────────────────────────────────────
VLLM_PORT=8000
export VLLM_BASE_URL="${VLLM_BASE_URL:-http://localhost:${VLLM_PORT}/v1}"
export VLLM_MODEL="${VLLM_MODEL:-Qwen/Qwen3-8B-AWQ}"
export VLLM_API_KEY="${VLLM_API_KEY:-local-vllm-token}"
VLLM_STARTUP_TIMEOUT=600

# Ensure the conda env's libstdc++ is preferred over the system one.
CONDA_LIB="${REPO_ROOT}/.conda/lib"
export LD_LIBRARY_PATH="${CONDA_LIB}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

echo "Host: $(hostname)"
echo "Python: ${PYTHON_BIN}"
date --iso-8601=seconds

# ── vLLM server guard ────────────────────────────────────────────────────
_vllm_ready() {
    curl -sf --max-time 5 "${VLLM_BASE_URL}/models" \
        -H "Authorization: Bearer ${VLLM_API_KEY}" > /dev/null 2>&1
}

if _vllm_ready; then
    echo "vLLM server already up at ${VLLM_BASE_URL}"
else
    echo "vLLM server not reachable at ${VLLM_BASE_URL}; starting (model=${VLLM_MODEL}, port=${VLLM_PORT})..."
    "${PYTHON_BIN}" -m vllm.entrypoints.openai.api_server \
        --model "${VLLM_MODEL}" \
        --port "${VLLM_PORT}" \
        --api-key "${VLLM_API_KEY}" &
    _VLLM_PID=$!
    trap 'echo "Stopping vLLM (pid=${_VLLM_PID})"; kill "${_VLLM_PID}" 2>/dev/null' EXIT

    _vllm_elapsed=0
    until _vllm_ready; do
        if [ "$_vllm_elapsed" -ge "$VLLM_STARTUP_TIMEOUT" ]; then
            echo "ERROR: vLLM did not start within ${VLLM_STARTUP_TIMEOUT}s" >&2
            exit 1
        fi
        sleep 10
        _vllm_elapsed=$((_vllm_elapsed + 10))
        echo "  waiting for vLLM... ${_vllm_elapsed}s elapsed"
    done
    echo "vLLM server ready after ${_vllm_elapsed}s"
fi
# ─────────────────────────────────────────────────────────────────────────

# ── Run the experiment ───────────────────────────────────────────────────
# Resume + judge cache are ON by default: the script skips any system already
# checkpointed under ${OUTPUT_DIR} and reuses ${OUTPUT_DIR}/judge_cache.jsonl,
# so a resubmission after a timeout/GPU death continues instead of restarting.
echo "Running online_replay_h1h0_final.py -> ${OUTPUT_DIR}"
"${PYTHON_BIN}" experiments/online_replay_h1h0_final.py \
    --npz "${NPZ}" \
    --output-dir "${OUTPUT_DIR}" \
    --mode "${MODE}" \
    --seed "${SEED}" \
    --max-requests "${MAX_REQUESTS}" \
    --cache-size "${CACHE_SIZE}" \
    --target-fpr "${TARGET_FPR}" \
    --selection "${SELECTION}" \
    --scorers "${SCORERS}" \
    --progress-every "${PROGRESS_EVERY}" \
    --llm-base-url "${VLLM_BASE_URL}" \
    --llm-model "${VLLM_MODEL}" \
    --llm-api-key "${VLLM_API_KEY}" \
    --warmup-requests 4000

date --iso-8601=seconds
echo "Experiment finished successfully"

