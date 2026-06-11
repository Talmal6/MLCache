"""FastAPI gateway exposing `SemanticCacheSystem` as an OpenAI-compatible API.

This module is a thin gateway/proxy: every semantic-matching decision (embed,
retrieve, score, threshold, HIT/MISS, LLM fallback, write-through, shadow
judging, training/calibration) is owned by `SemanticCacheSystem`. The HTTP
layer only converts between the OpenAI chat-completions wire format and
`SemanticCacheSystem.handle(prompt: str)`.
"""

from __future__ import annotations

import os
import time
import uuid
from typing import Any, Iterable

API_DEPENDENCIES_ERROR = "Install API dependencies with: pip install -e '.[api,dev]'"

try:
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel
except ImportError as exc:  # pragma: no cover - exercised via API_DEPENDENCIES_ERROR
    raise ImportError(API_DEPENDENCIES_ERROR) from exc

from mlcache.llm_providers import DEFAULT_ASSISTANT_SYSTEM_PROMPT, OpenAICompatibleLLM
from mlcache.system import ActivePolicy, SemanticCacheSystem, SystemResponse


# -- configuration -----------------------------------------------------------

def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off", ""}


def _env_list(name: str, default: str) -> list[str]:
    raw = os.environ.get(name, default)
    return [item.strip() for item in raw.split(",") if item.strip()]


def build_default_llm() -> OpenAICompatibleLLM:
    """Build the `OpenAICompatibleLLM` adapter from `VLLM_*` environment variables."""

    return OpenAICompatibleLLM(
        base_url=os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1"),
        api_key=os.environ.get("VLLM_API_KEY", "local-vllm-token"),
        model=os.environ.get("VLLM_MODEL", "Qwen/Qwen3-8B-AWQ"),
        system_prompt=DEFAULT_ASSISTANT_SYSTEM_PROMPT,
    )


def build_default_system(llm: Any | None = None) -> SemanticCacheSystem:
    """Build the production `SemanticCacheSystem` from `MLCACHE_*` environment variables.

    `SemanticCacheSystem` (not bare `CachedLLM`) is used here so the gateway
    gets the full serving + learning lifecycle (shadow judging, training,
    calibration, convergence freezing) -- `CachedLLM` only does cache-first
    orchestration with no learning loop, so it would not be a faithful
    "production path" for this gateway.
    """

    return SemanticCacheSystem(
        llm=llm if llm is not None else build_default_llm(),
        stream=None,
        scorer=os.environ.get("MLCACHE_SCORER", "ensemble"),
        scorers=_env_list("MLCACHE_SCORERS", "cosine,lda,pca_whitened_cosine,xgboost,mlp"),
        target_fpr=float(os.environ.get("MLCACHE_TARGET_FPR", "0.05")),
        top_k=int(os.environ.get("MLCACHE_TOP_K", "5")),
        root_dir=os.environ.get("MLCACHE_ROOT", "cache_state/production"),
        persistence=_env_bool("MLCACHE_PERSISTENCE", True),
    )


# -- request / response schemas ----------------------------------------------

class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str | None = None
    messages: list[ChatMessage]
    temperature: float = 0.2
    top_p: float = 0.95
    max_tokens: int = 512
    stream: bool = False


# -- prompt conversion --------------------------------------------------------

_ROLE_LABELS = {"system": "System", "user": "User", "assistant": "Assistant"}


def messages_to_prompt(messages: Iterable[ChatMessage]) -> str:
    """Convert OpenAI-style chat messages into a single prompt string.

    `SemanticCacheSystem.handle`/`LLMClient.generate` are prompt-based, so the
    chat history must be flattened. Role labels are preserved (rather than
    concatenating raw `content` values) so a "system: X / user: Y" exchange
    embeds and matches differently than a bare "Y" query would.
    """

    lines: list[str] = []
    for message in messages:
        label = _ROLE_LABELS.get(message.role.strip().lower(), message.role.strip() or "user")
        lines.append(f"{label}: {message.content}")
    lines.append("Assistant:")
    return "\n".join(lines)


def _policy_to_dict(policy: ActivePolicy) -> dict[str, Any]:
    return {
        "scorer_name": policy.scorer_name,
        "scorer_version": policy.scorer_version,
        "threshold": policy.threshold,
        "threshold_version": policy.threshold_version,
        "calibrated": policy.calibrated,
        "trained": policy.trained,
        "frozen": policy.frozen,
    }


# -- app factory ---------------------------------------------------------------

def create_app(system: SemanticCacheSystem | None = None, *, model_name: str | None = None) -> FastAPI:
    """Build the gateway app.

    `system` can be injected (e.g. a fake/mocked `SemanticCacheSystem` in
    tests); when omitted, a production `SemanticCacheSystem` is built from
    `MLCACHE_*`/`VLLM_*` environment variables via `build_default_system`.
    """

    if system is None:
        system = build_default_system()
    if model_name is None:
        model_name = os.environ.get("VLLM_MODEL", "Qwen/Qwen3-8B-AWQ")

    app = FastAPI(title="MLCache Gateway")

    @app.get("/health")
    def health() -> dict[str, Any]:
        payload: dict[str, Any] = {
            "status": "ok",
            "model": model_name,
            "vllm_base_url": getattr(system.llm, "base_url", None),
        }
        try:
            payload["policy"] = _policy_to_dict(system.policy)
        except Exception:
            payload["policy"] = None
        try:
            payload["report"] = system.report()
        except Exception:
            payload["report"] = None
        return payload

    @app.post("/v1/chat/completions")
    def chat_completions(request: ChatCompletionRequest) -> dict[str, Any]:
        if request.stream:
            raise HTTPException(status_code=400, detail="streaming is not implemented yet")

        prompt = messages_to_prompt(request.messages)

        # NOTE: `SemanticCacheSystem.handle(prompt)` does not accept
        # generation kwargs (temperature/top_p/max_tokens), so on a cache
        # MISS the LLM call uses the OpenAICompatibleLLM's configured
        # defaults, not these per-request overrides. Plumbing per-request
        # generation kwargs through `handle()` would require changing
        # `SemanticCacheSystem`'s public interface, which is out of scope
        # here (see Part 3 notes).
        response: SystemResponse = system.handle(prompt)

        return {
            "id": f"chatcmpl-{uuid.uuid4().hex}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": request.model or model_name,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": response.text},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                # TODO: populate from the backend's real token usage once
                # OpenAICompatibleLLM surfaces `completion.usage` through
                # LLMResponse.metadata for cache-MISS responses. Cache HITs
                # have no LLM call to report usage for.
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            },
            "mlcache": {
                "source": response.source,
                "cache_key": response.cache_key,
                "score": response.score,
                "threshold": response.threshold,
                "policy": _policy_to_dict(response.policy),
            },
        }

    return app


__all__ = [
    "API_DEPENDENCIES_ERROR",
    "ChatCompletionRequest",
    "ChatMessage",
    "build_default_llm",
    "build_default_system",
    "create_app",
    "messages_to_prompt",
]
