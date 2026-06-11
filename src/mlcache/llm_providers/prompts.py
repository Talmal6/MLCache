"""Recommended prompts for real-LLM integrations.

These constants are plain strings deliberately kept separate from the
provider adapter (`OpenAICompatibleLLM`) and from `LLMJudge`/`StrictLLMJudge`
so callers can reuse or override them independently.
"""

from __future__ import annotations

from mlcache.llm_wrapper import LLMClient, LLMJudge

DEFAULT_ASSISTANT_SYSTEM_PROMPT = ( "You are a concise, accurate assistant. Answer the user directly. "
                                    "Use the information provided in the request. If you must make an assumption, "
                                      "state it briefly before answering. Do not mention the cache, cache hits, " 
                                      "cache misses, retrieval, or internal system behavior." ) 

STRICT_REUSE_JUDGE_PROMPT = ( "You are a strict semantic cache reuse judge.\n\n" "Your task is to decide whether the cached response can be safely reused " 
                            "as the answer to the new query.\n\n" "Return exactly one token:\n" "REUSABLE\n" "NOT_REUSABLE\n" "UNCERTAIN\n\n" "Core rule:\n" 
                            "- Judge answer equivalence, not topic similarity. Return REUSABLE only if " "the cached response would be a complete, correct, and non-misleading answer "
                            "to the new query.\n\n" "Return REUSABLE only when:\n" "- The new query asks for the same information, task, or intent as the cached query.\n"
                            "- The cached response fully satisfies the new query without needing changes.\n" 
                            "- Any differences in wording are superficial and do not change the required answer.\n\n" "Return NOT_REUSABLE when:\n" 
                            "- The new query asks for different facts, constraints, code, numbers, dates, " 
                            "versions, files, paths, APIs, models, hardware, or user intent.\n" 
                            "- The new query is more specific than the cached query and the cached response " 
                            "does not cover the added constraint.\n" "- The cached response is more specific than the new query in a way that could " 
                            "mislead the user.\n" "- The new query depends on current, recent, private, user-specific, or " 
                            "environment-specific information that is not guaranteed by the cached response.\n" 
                            "- Reusing the cached response could be incomplete, stale, unsafe, or misleading.\n\n" 
                            "Return UNCERTAIN when:\n" 
                            "- There is not enough information to decide safely.\n" 
                            "- The cached response might answer the new query, but correctness depends on " 
                            "unstated assumptions.\n\n" 
                            "Output rules:\n" 
                            "- Do not explain your answer.\n" 
                            "- Do not output punctuation.\n" 
                            "- Do not output anything except the single token." )

class StrictLLMJudge(LLMJudge):
    """`LLMJudge` with the stricter `STRICT_REUSE_JUDGE_PROMPT`.

    Added alongside `LLMJudge` (rather than replacing its prompt) to avoid
    changing the behavior of any existing caller that already depends on
    `LLMJudge`'s default prompt/parsing. Reuses `LLMJudge.judge` and
    `LLMJudge._parse_label` (REUSABLE/NOT_REUSABLE exact match, anything else
    -> UNCERTAIN), which already matches this prompt's required output.
    """

    _PROMPT_TEMPLATE = (
        STRICT_REUSE_JUDGE_PROMPT + "\n\n"
        "New query: {query}\n"
        "Cached query: {candidate_query}\n"
        "Cached response: {candidate_response}\n"
    )

    def __init__(self, llm: LLMClient, *, name: str = "strict_llm_judge") -> None:
        super().__init__(llm, name=name)


__all__ = [
    "DEFAULT_ASSISTANT_SYSTEM_PROMPT",
    "STRICT_REUSE_JUDGE_PROMPT",
    "StrictLLMJudge",
]
