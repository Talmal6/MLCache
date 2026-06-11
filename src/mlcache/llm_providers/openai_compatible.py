"""Adapter from an OpenAI-compatible HTTP server (e.g. vLLM) to `LLMClient`.

This is the only place vLLM/OpenAI-specific request/response shapes are
handled. `SemanticCacheSystem`, `MLCache`, scorers, retrieval, the oracle, and
calibration are all unaware of this adapter -- they only see the `LLMClient`
protocol (`generate(prompt: str, **kwargs) -> LLMResponse`), exactly as they
do for `MockLLM`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from mlcache.llm_wrapper import LLMResponse

LLM_DEPENDENCIES_ERROR = "Install LLM provider dependencies with: pip install -e '.[llm,dev]'"

try:
    import openai
except ImportError:  # pragma: no cover - exercised via LLM_DEPENDENCIES_ERROR
    openai = None  # type: ignore[assignment]


@dataclass
class OpenAICompatibleLLM:
    """`LLMClient` backed by any OpenAI-compatible `/chat/completions` server.

    Reusable with vLLM, the real OpenAI API, or any other server that speaks
    the same protocol -- nothing here is vLLM-specific beyond the default
    `model` value, which callers override anyway.
    """

    base_url: str
    api_key: str
    model: str
    system_prompt: str | None = None
    timeout: float = 120.0
    default_temperature: float = 0.2
    default_top_p: float = 0.95
    default_max_tokens: int = 512
    extra_body: dict[str, Any] = field(default_factory=dict)
    client: Any | None = None

    def __post_init__(self) -> None:
        if self.client is not None:
            return
        if openai is None:
            raise ImportError(LLM_DEPENDENCIES_ERROR)
        self.client = openai.OpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
            timeout=self.timeout,
        )

    def generate(self, prompt: str, **kwargs: Any) -> LLMResponse:
        messages: list[dict[str, str]] = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": prompt})

        temperature = kwargs.pop("temperature", self.default_temperature)
        top_p = kwargs.pop("top_p", self.default_top_p)
        max_tokens = kwargs.pop("max_tokens", self.default_max_tokens)
        extra_body = {**self.extra_body, **kwargs.pop("extra_body", {})}

        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                extra_body=extra_body or None,
            )
        except Exception as exc:
            raise RuntimeError(
                f"OpenAI-compatible backend at {self.base_url!r} (model={self.model!r}) "
                f"request failed: {exc}"
            ) from exc

        if not completion.choices:
            raise RuntimeError(
                f"OpenAI-compatible backend at {self.base_url!r} (model={self.model!r}) "
                "returned no choices"
            )

        choice = completion.choices[0]
        text = choice.message.content
        finish_reason = getattr(choice, "finish_reason", None)
        if text is None:
            raise RuntimeError(
                f"OpenAI-compatible backend at {self.base_url!r} (model={self.model!r}) "
                f"returned empty content (finish_reason={finish_reason!r})"
            )

        return LLMResponse(
            text=text,
            raw=completion,
            metadata={
                "provider": "openai-compatible",
                "base_url": self.base_url,
                "model": self.model,
                "finish_reason": finish_reason,
            },
        )


__all__ = ["LLM_DEPENDENCIES_ERROR", "OpenAICompatibleLLM"]
