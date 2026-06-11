# Real LLM API: vLLM + MLCache gateway

This sets up `SemanticCacheSystem` behind an OpenAI-compatible
`/v1/chat/completions` gateway, backed by a real model served by
[vLLM](https://github.com/vllm-project/vllm).

```
Client
  -> MLCache FastAPI Gateway (port 9000)
      -> SemanticCacheSystem.handle(prompt)
          -> cache HIT: return cached response
          -> cache MISS: OpenAICompatibleLLM.generate(prompt)
              -> vLLM /v1/chat/completions
              -> write response into cache
              -> return response
```

The gateway is a thin proxy: all embedding, retrieval, scoring,
thresholding, HIT/MISS decisions, LLM fallback, write-through, shadow
judging, and training/calibration remain owned by `SemanticCacheSystem` /
`MLCache`. `OpenAICompatibleLLM` (in `src/mlcache/llm_providers/`) is the only
vLLM/OpenAI-specific code, and it implements the existing `LLMClient`
interface, so it is reusable with any OpenAI-compatible backend.

## Install

```bash
python -m pip install -e ".[ml,embeddings,llm,api,dev]"
```

## Run vLLM

```bash
export VLLM_API_KEY="local-vllm-token"

CUDA_VISIBLE_DEVICES=0 vllm serve Qwen/Qwen3-8B-AWQ \
  --host 0.0.0.0 \
  --port 8000 \
  --api-key "$VLLM_API_KEY" \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.88 \
  --generation-config vllm
```

## Run the MLCache gateway

```bash
export VLLM_BASE_URL="http://localhost:8000/v1"
export VLLM_API_KEY="local-vllm-token"
export VLLM_MODEL="Qwen/Qwen3-8B-AWQ"
export MLCACHE_ROOT="cache_state/qwen3_8b_awq"
export MLCACHE_TARGET_FPR="0.05"
export MLCACHE_TOP_K="5"

python scripts/serve_mlcache_gateway.py
```

## Test

```bash
curl -X POST "http://localhost:9000/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mlcache-qwen3-8b-awq",
    "messages": [
      {"role": "user", "content": "Explain semantic caching in one paragraph."}
    ],
    "temperature": 0.2,
    "max_tokens": 256
  }'
```

The response includes an `mlcache` block reporting whether the answer came
from the cache or the LLM:

```json
{
  "...": "...",
  "mlcache": {
    "source": "cache",
    "cache_key": "...",
    "score": 0.97,
    "threshold": 0.81,
    "policy": {
      "scorer_name": "EnsembleScorer",
      "scorer_version": 3,
      "threshold": 0.81,
      "threshold_version": 2,
      "calibrated": true,
      "trained": true,
      "frozen": false
    }
  }
}
```

## Configuration reference

| Variable | Default | Meaning |
| --- | --- | --- |
| `VLLM_BASE_URL` | `http://localhost:8000/v1` | OpenAI-compatible base URL |
| `VLLM_API_KEY` | `local-vllm-token` | API key sent to the backend |
| `VLLM_MODEL` | `Qwen/Qwen3-8B-AWQ` | Model name passed to the backend |
| `MLCACHE_ROOT` | `cache_state/production` | `SemanticCacheSystem` persistence root |
| `MLCACHE_TARGET_FPR` | `0.05` | Target false-accept rate for calibration |
| `MLCACHE_TOP_K` | `5` | Retrieval top-k |
| `MLCACHE_SCORER` | `ensemble` | Active scorer preset |
| `MLCACHE_SCORERS` | `cosine,lda,pca_whitened_cosine,xgboost,mlp` | Ensemble member scorers |
| `MLCACHE_PERSISTENCE` | `true` | Whether `SemanticCacheSystem` persists cache state to disk |
| `MLCACHE_API_PORT` | `9000` | Port for `scripts/serve_mlcache_gateway.py` |

## Known limitations

- `stream=True` requests return HTTP 400 (`"streaming is not implemented yet"`).
  Streaming is not yet supported.
- Per-request `temperature` / `top_p` / `max_tokens` are accepted for
  OpenAI-API compatibility but are **not** forwarded to the backend on a
  cache MISS: `SemanticCacheSystem.handle(prompt)` does not currently accept
  generation kwargs. The LLM call instead uses `OpenAICompatibleLLM`'s
  configured defaults (`default_temperature`, `default_top_p`,
  `default_max_tokens`). Plumbing per-request overrides through would require
  extending `SemanticCacheSystem`'s public interface.
- `usage` token counts are reported as zeros (see `TODO` in
  `src/mlcache/api/server.py`); the backend's real token usage is not yet
  surfaced through `LLMResponse`.
