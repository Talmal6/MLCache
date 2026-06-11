"""Real-LLM provider adapters for `LLMClient`.

These adapters translate between a specific backend's wire protocol and the
repository's `LLMClient`/`LLMResponse` interfaces. They contain no semantic
matching, retrieval, scoring, thresholding, or calibration logic -- that all
remains owned by `MLCache`/`SemanticCacheSystem`.
"""

from mlcache.llm_providers.openai_compatible import LLM_DEPENDENCIES_ERROR, OpenAICompatibleLLM
from mlcache.llm_providers.prompts import (
    DEFAULT_ASSISTANT_SYSTEM_PROMPT,
    STRICT_REUSE_JUDGE_PROMPT,
    StrictLLMJudge,
)

__all__ = [
    "DEFAULT_ASSISTANT_SYSTEM_PROMPT",
    "LLM_DEPENDENCIES_ERROR",
    "OpenAICompatibleLLM",
    "STRICT_REUSE_JUDGE_PROMPT",
    "StrictLLMJudge",
]
