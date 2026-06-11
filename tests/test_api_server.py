"""Tests for the FastAPI gateway (src/mlcache/api/server.py).

No real `SemanticCacheSystem`, vLLM server, or network access required: a
small fake system stands in for `SemanticCacheSystem`, exposing only the
attributes/methods the gateway actually uses (`llm.base_url`, `.policy`,
`.report()`, `.handle(prompt)`).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

fastapi = pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from mlcache.api.server import ChatMessage, create_app, messages_to_prompt  # noqa: E402
from mlcache.system import ActivePolicy, SystemResponse  # noqa: E402


def _policy(**overrides) -> ActivePolicy:
    base = dict(
        scorer=None,
        scorer_name="cosine",
        scorer_version=1,
        threshold=0.8,
        threshold_version=1,
        calibrated=True,
        trained=True,
        frozen=False,
        metadata={},
    )
    base.update(overrides)
    return ActivePolicy(**base)


class _FakeLLM:
    base_url = "http://localhost:8000/v1"
    model = "Qwen/Qwen3-8B-AWQ"


class _FakeSystem:
    """Stands in for `SemanticCacheSystem` in API tests.

    `handle()` is scripted: the first call returns a cache MISS (LLM-served)
    response, every later call returns a cache HIT, so tests can assert both
    `mlcache.source` values without depending on real semantic matching.
    """

    def __init__(self) -> None:
        self.llm = _FakeLLM()
        self.calls: list[str] = []
        self._handled_once = False

    def handle(self, prompt: str) -> SystemResponse:
        self.calls.append(prompt)
        if not self._handled_once:
            self._handled_once = True
            return SystemResponse(
                text=f"llm-answer:{prompt}",
                source="llm",
                cache_key="abc123",
                score=None,
                threshold=0.8,
                policy=_policy(trained=False, calibrated=False, threshold=None),
            )
        return SystemResponse(
            text=f"cached-answer:{prompt}",
            source="cache",
            cache_key="abc123",
            score=0.97,
            threshold=0.8,
            policy=_policy(),
        )

    @property
    def policy(self) -> ActivePolicy:
        return _policy()

    def report(self) -> dict:
        return {
            "requests": len(self.calls),
            "cache_hits": max(0, len(self.calls) - 1),
            "llm_calls": 1 if self.calls else 0,
            "hit_rate": 0.0,
            "scorer_name": "cosine",
            "scorer_version": 1,
            "threshold": 0.8,
            "threshold_version": 1,
            "calibrated": True,
            "trained": True,
            "frozen": False,
            "freeze_reason": None,
            "stopping_updates_observed": 0,
        }


@pytest.fixture
def client() -> TestClient:
    app = create_app(system=_FakeSystem(), model_name="mlcache-qwen3-8b-awq")
    return TestClient(app)


# ── 1. prompt conversion preserves roles ────────────────────────────────────

def test_messages_to_prompt_preserves_roles():
    messages = [
        ChatMessage(role="system", content="be concise"),
        ChatMessage(role="user", content="hello"),
        ChatMessage(role="assistant", content="hi there"),
        ChatMessage(role="user", content="follow up"),
    ]
    prompt = messages_to_prompt(messages)

    assert "System: be concise" in prompt
    assert "User: hello" in prompt
    assert "Assistant: hi there" in prompt
    assert "User: follow up" in prompt
    # Roles must not be collapsed: the raw content alone must not appear
    # without its role label preceding it.
    lines = prompt.splitlines()
    assert lines[0] == "System: be concise"
    assert lines[1] == "User: hello"
    assert lines[2] == "Assistant: hi there"
    assert lines[3] == "User: follow up"


def test_messages_to_prompt_handles_unknown_role():
    messages = [ChatMessage(role="tool", content="result data")]
    prompt = messages_to_prompt(messages)
    assert "tool: result data" in prompt


# ── 2. /health ────────────────────────────────────────────────────────────

def test_health_ok(client: TestClient):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["model"] == "mlcache-qwen3-8b-awq"
    assert body["vllm_base_url"] == "http://localhost:8000/v1"
    assert body["policy"]["scorer_name"] == "cosine"
    assert body["report"]["requests"] == 0


# ── 3. /v1/chat/completions ─────────────────────────────────────────────────

def test_chat_completions_llm_then_cache_source(client: TestClient):
    payload = {
        "model": "mlcache-qwen3-8b-awq",
        "messages": [{"role": "user", "content": "Explain semantic caching."}],
        "temperature": 0.2,
        "max_tokens": 64,
    }

    resp1 = client.post("/v1/chat/completions", json=payload)
    assert resp1.status_code == 200
    body1 = resp1.json()
    assert body1["object"] == "chat.completion"
    assert body1["model"] == "mlcache-qwen3-8b-awq"
    assert body1["choices"][0]["message"]["role"] == "assistant"
    assert body1["choices"][0]["message"]["content"].startswith("llm-answer:")
    assert body1["choices"][0]["finish_reason"] == "stop"
    assert body1["usage"] == {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    assert body1["mlcache"]["source"] == "llm"
    assert body1["mlcache"]["cache_key"] == "abc123"
    assert body1["mlcache"]["policy"]["scorer_name"] == "cosine"

    resp2 = client.post("/v1/chat/completions", json=payload)
    body2 = resp2.json()
    assert body2["mlcache"]["source"] == "cache"
    assert body2["choices"][0]["message"]["content"].startswith("cached-answer:")
    assert body2["mlcache"]["score"] == 0.97
    assert body2["mlcache"]["threshold"] == 0.8


# ── 4. stream=True -> 400 ───────────────────────────────────────────────────

def test_chat_completions_stream_returns_400(client: TestClient):
    payload = {
        "messages": [{"role": "user", "content": "hi"}],
        "stream": True,
    }
    resp = client.post("/v1/chat/completions", json=payload)
    assert resp.status_code == 400
    assert "streaming is not implemented" in resp.json()["detail"]


# ── 5. default model name when request omits it ────────────────────────────

def test_chat_completions_defaults_model_from_app(client: TestClient):
    payload = {"messages": [{"role": "user", "content": "hi"}]}
    resp = client.post("/v1/chat/completions", json=payload)
    assert resp.json()["model"] == "mlcache-qwen3-8b-awq"
