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
# NOTE: this dir is config-specific (inf cache, fpr 0.05, warmup 2000, check
# 3000, frozen); a NEW dir is REQUIRED -- result checkpoints are keyed by dir,
# so reusing an old dir would load stale results instead of replaying.
OUTPUT_DIR="${OUTPUT_DIR:-experiments/llm_judge_replay_infcache_fpr05_wu2000_check3000_frozen}"
MODE="compare"
SEED=42
# Measured ("run cache") replay length: number of queries served + audited after
# warm-up.
MAX_REQUESTS=3000
# "Infinite" cache: load_rows selects ALL rows when --cache-size >= the
# selectable-row count (97082 rows have label in {0,1}; the rest are uncertain
# and excluded), and the vector store capacity equals that universe -- so the
# cache never evicts. 200000 comfortably exceeds 97082 -> full selectable corpus.
CACHE_SIZE=200000
TARGET_FPR=0.05
SELECTION="uniform"
SCORERS="cosine,lda,pca_whitened_cosine,xgboost,mlp"
PROGRESS_EVERY=100
WARMUP_REQUESTS="${WARMUP_REQUESTS:-3000}"
# After warm-up, freeze the scorer + NP threshold and only MEASURE: the measured
# replay serves on the fixed post-warm-up model instead of retraining online.
# Set FREEZE_AFTER_WARMUP=0 to keep the online refit loop running.
FREEZE_AFTER_WARMUP="${FREEZE_AFTER_WARMUP:-1}"
# LLM judge labels are keyed only by (query_row_id, candidate_row_id) -- config
# independent -- so reuse the 177k labels already computed by the old v2 run
# instead of re-paying for those judge calls. Override/point elsewhere as needed.
JUDGE_CACHE_PATH="${JUDGE_CACHE_PATH:-experiments/judge_cache_shared.jsonl}"

# ── vLLM server config ───────────────────────────────────────────────────
VLLM_PORT=8000
export VLLM_BASE_URL="${VLLM_BASE_URL:-http://localhost:${VLLM_PORT}/v1}"
export VLLM_MODEL="${VLLM_MODEL:-Qwen/Qwen3-8B-AWQ}"
export VLLM_API_KEY="${VLLM_API_KEY:-local-vllm-token}"
VLLM_STARTUP_TIMEOUT=600

# Ensure the conda env's libstdc++ is preferred over the system one.
CONDA_LIB="${REPO_ROOT}/.conda/lib"
export LD_LIBRARY_PATH="${CONDA_LIB}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

# Put the conda env's bin on PATH. The server is launched via the absolute
# ${PYTHON_BIN} (env never activated), so without this FlashInfer's JIT compile
# step -- which shells out to `ninja` -- fails with FileNotFoundError: 'ninja',
# the vLLM EngineCore never starts, and the job stalls until the startup
# timeout (job 18304535 died exactly this way).
export PATH="${REPO_ROOT}/.conda/bin${PATH:+:${PATH}}"

# ── Caches OFF the (quota-limited) home filesystem ───────────────────────
# The /home quota is small; jobs 18269819 / 18271042 both died with
# "OSError: [Errno 122] Disk quota exceeded" while vLLM cached the model
# weights into ~/.cache/huggingface. Point ALL caches at the roomy group
# storage (REPO_ROOT) and silence the telemetry/usage-stats writes that also
# hit the quota, so the server can never fill home again.
export HF_HOME="${HF_HOME:-${REPO_ROOT}/.hf_cache}"
export HF_HUB_DISABLE_TELEMETRY=1
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${REPO_ROOT}/.cache}"
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-${REPO_ROOT}/.cache/vllm}"
export VLLM_NO_USAGE_STATS=1
export DO_NOT_TRACK=1
mkdir -p "${HF_HOME}" "${XDG_CACHE_HOME}" "${VLLM_CACHE_ROOT}"

echo "Host: $(hostname)"
echo "Python: ${PYTHON_BIN}"
date --iso-8601=seconds

# ── vLLM server guard ────────────────────────────────────────────────────
# Reachable AND serving the model we want. A stale server from a previous job
# can squat port 8000 with a DIFFERENT model id: its /models returns 200 (so a
# bare reachability check passes), but the first chat request then 404s with
# "The model ... does not exist" and set -e kills the job after ~20s. So we
# require ${VLLM_MODEL} to appear in the served model list.
_vllm_ready() {
    curl -sf --max-time 5 "${VLLM_BASE_URL}/models" \
        -H "Authorization: Bearer ${VLLM_API_KEY}" 2>/dev/null \
        | grep -qF "\"${VLLM_MODEL}\""
}

# Reachable at all (any model) -- used to detect a stale wrong-model server.
_vllm_port_busy() {
    curl -sf --max-time 5 "${VLLM_BASE_URL}/models" \
        -H "Authorization: Bearer ${VLLM_API_KEY}" > /dev/null 2>&1
}

if _vllm_ready; then
    echo "vLLM server already up at ${VLLM_BASE_URL} serving ${VLLM_MODEL}"
else
    if _vllm_port_busy; then
        echo "WARNING: a vLLM server is up at ${VLLM_BASE_URL} but does NOT serve ${VLLM_MODEL}." >&2
        echo "  Served models: $(curl -sf --max-time 5 "${VLLM_BASE_URL}/models" -H "Authorization: Bearer ${VLLM_API_KEY}" 2>/dev/null)" >&2
        echo "  Killing the stale server squatting port ${VLLM_PORT} and starting the correct one." >&2
        # Only kill listeners on our port owned by this user (won't touch other jobs' ports).
        for _pid in $(fuser "${VLLM_PORT}/tcp" 2>/dev/null); do
            kill "${_pid}" 2>/dev/null || true
        done
        sleep 5
    fi
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
# checkpointed under ${OUTPUT_DIR} and reuses the persistent LLM-judge cache,
# so a resubmission after a timeout/GPU death continues instead of restarting.
#
# Seed the shared judge cache from the old v2 run (177k pre-computed labels) the
# first time, so this fresh-config run does not re-pay for those judge calls.
# Labels are keyed by (query_row_id, candidate_row_id) only, so reuse is valid.
_OLD_JUDGE_CACHE="experiments/llm_judge_replay_v2/judge_cache.jsonl"
if [ ! -f "${JUDGE_CACHE_PATH}" ] && [ -f "${_OLD_JUDGE_CACHE}" ]; then
    echo "Seeding shared judge cache ${JUDGE_CACHE_PATH} from ${_OLD_JUDGE_CACHE}"
    mkdir -p "$(dirname "${JUDGE_CACHE_PATH}")"
    cp "${_OLD_JUDGE_CACHE}" "${JUDGE_CACHE_PATH}"
fi

echo "Running online_replay_h1h0_final.py -> ${OUTPUT_DIR}"
# store_true flags are passed only when enabled (argparse has no --flag=value form).
EXTRA_ARGS=()
if [ "${FREEZE_AFTER_WARMUP}" != "0" ]; then
    EXTRA_ARGS+=(--freeze-after-warmup)
fi
# Force a fresh replay (ignore per-system result checkpoints) so warm-up reruns
# and the end-of-warm-up training FPR/TPR logging fires. Set NO_RESUME=1 for that;
# leave unset for normal resume-after-timeout behaviour.
if [ "${NO_RESUME:-0}" != "0" ]; then
    EXTRA_ARGS+=(--no-resume)
fi
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
    --warmup-requests "${WARMUP_REQUESTS}" \
    --judge-cache-path "${JUDGE_CACHE_PATH}" \
    "${EXTRA_ARGS[@]}"

date --iso-8601=seconds
echo "Experiment finished successfully"

