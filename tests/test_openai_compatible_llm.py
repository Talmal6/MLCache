"""Tests for `OpenAICompatibleLLM` (src/mlcache/llm_providers/openai_compatible.py).

No network access and no real OpenAI/vLLM server required: a fake client
object (matching the small subset of `openai.OpenAI`'s shape this adapter
uses) is injected via the `client=` constructor field.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mlcache.llm_providers.openai_compatible import OpenAICompatibleLLM
from mlcache.llm_wrapper import LLMResponse


class _FakeMessage:
    def __init__(self, content: str | None) -> None:
        self.content = content


class _FakeChoice:
    def __init__(self, content: str | None, finish_reason: str = "stop") -> None:
        self.message = _FakeMessage(content)
        self.finish_reason = finish_reason


class _FakeCompletion:
    def __init__(self, choices: list[_FakeChoice]) -> None:
        self.choices = choices


class _FakeCompletions:
    def __init__(self, completion: _FakeCompletion) -> None:
        self._completion = completion
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self._completion


class _FakeChat:
    def __init__(self, completion: _FakeCompletion) -> None:
        self.completions = _FakeCompletions(completion)


class _FakeClient:
    def __init__(self, completion: _FakeCompletion) -> None:
        self.chat = _FakeChat(completion)


def _make_llm(completion: _FakeCompletion, **overrides) -> tuple[OpenAICompatibleLLM, _FakeClient]:
    fake_client = _FakeClient(completion)
    kwargs = dict(
        base_url="http://localhost:8000/v1",
        api_key="local-vllm-token",
        model="Qwen/Qwen3-8B-AWQ",
        client=fake_client,
    )
    kwargs.update(overrides)
    return OpenAICompatibleLLM(**kwargs), fake_client


def test_generate_builds_correct_request_and_maps_response():
    completion = _FakeCompletion([_FakeChoice("hello there", finish_reason="stop")])
    llm, fake_client = _make_llm(completion, system_prompt="be concise")

    response = llm.generate("hi")

    assert isinstance(response, LLMResponse)
    assert response.text == "hello there"
    assert response.metadata["provider"] == "openai-compatible"
    assert response.metadata["base_url"] == "http://localhost:8000/v1"
    assert response.metadata["model"] == "Qwen/Qwen3-8B-AWQ"
    assert response.metadata["finish_reason"] == "stop"

    [call] = fake_client.chat.completions.calls
    assert call["model"] == "Qwen/Qwen3-8B-AWQ"
    assert call["messages"] == [
        {"role": "system", "content": "be concise"},
        {"role": "user", "content": "hi"},
    ]
    assert call["temperature"] == 0.2
    assert call["top_p"] == 0.95
    assert call["max_tokens"] == 512


def test_generate_without_system_prompt_omits_system_message():
    completion = _FakeCompletion([_FakeChoice("ok")])
    llm, fake_client = _make_llm(completion)

    llm.generate("hi")

    [call] = fake_client.chat.completions.calls
    assert call["messages"] == [{"role": "user", "content": "hi"}]


def test_generate_allows_per_request_overrides():
    completion = _FakeCompletion([_FakeChoice("ok")])
    llm, fake_client = _make_llm(completion)

    llm.generate("hi", temperature=0.9, top_p=0.5, max_tokens=64, extra_body={"foo": "bar"})

    [call] = fake_client.chat.completions.calls
    assert call["temperature"] == 0.9
    assert call["top_p"] == 0.5
    assert call["max_tokens"] == 64
    assert call["extra_body"] == {"foo": "bar"}


def test_generate_merges_constructor_and_per_request_extra_body():
    completion = _FakeCompletion([_FakeChoice("ok")])
    llm, fake_client = _make_llm(completion, extra_body={"a": 1})

    llm.generate("hi", extra_body={"b": 2})

    [call] = fake_client.chat.completions.calls
    assert call["extra_body"] == {"a": 1, "b": 2}


def test_generate_raises_on_empty_choices():
    completion = _FakeCompletion([])
    llm, _ = _make_llm(completion)

    with pytest.raises(RuntimeError, match="no choices"):
        llm.generate("hi")


def test_generate_raises_on_none_content():
    completion = _FakeCompletion([_FakeChoice(None, finish_reason="length")])
    llm, _ = _make_llm(completion)

    with pytest.raises(RuntimeError, match="empty content"):
        llm.generate("hi")


def test_generate_raises_clear_error_on_provider_failure():
    class _BoomCompletions:
        def create(self, **kwargs):
            raise ConnectionError("boom")

    class _BoomChat:
        completions = _BoomCompletions()

    class _BoomClient:
        chat = _BoomChat()

    llm = OpenAICompatibleLLM(
        base_url="http://localhost:8000/v1",
        api_key="k",
        model="m",
        client=_BoomClient(),
    )

    with pytest.raises(RuntimeError, match="request failed"):
        llm.generate("hi")


def test_missing_openai_dependency_raises_clear_error(monkeypatch):
    import mlcache.llm_providers.openai_compatible as oc

    monkeypatch.setattr(oc, "openai", None)

    with pytest.raises(ImportError, match="pip install"):
        OpenAICompatibleLLM(base_url="http://x", api_key="k", model="m")
