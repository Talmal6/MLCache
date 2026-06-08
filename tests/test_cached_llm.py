from __future__ import annotations

import socket
import tempfile
import unittest
from pathlib import Path

from mlcache import (
    CachedLLM,
    CachedLLMResponse,
    HashingEmbeddingProvider,
    MLCache,
    MockLLM,
)


def _build_cache(tmp: str) -> MLCache:
    return MLCache.from_preset(
        root_dir=Path(tmp) / "state",
        scorer="cosine",
        threshold=0.05,
        persistence=False,
    )


class CachedLLMBehaviorTests(unittest.TestCase):
    def test_first_call_is_a_cache_miss_served_by_the_mock_llm(self) -> None:
        with tempfile.TemporaryDirectory(prefix="cached-llm-") as tmp:
            cached_llm = CachedLLM(llm=MockLLM(), cache=_build_cache(tmp), namespace="mock-llm")

            result = cached_llm.generate("Explain Byzantine broadcast in one paragraph.")

            self.assertEqual(result.source, "llm")
            self.assertEqual(result.text, "mock response for: Explain Byzantine broadcast in one paragraph.")

    def test_first_call_writes_the_response_into_the_cache(self) -> None:
        with tempfile.TemporaryDirectory(prefix="cached-llm-") as tmp:
            cache = _build_cache(tmp)
            cached_llm = CachedLLM(llm=MockLLM(), cache=cache, namespace="mock-llm")

            first = cached_llm.generate("Explain Byzantine broadcast in one paragraph.")

            self.assertIsNotNone(first.cache_key)
            embedding = HashingEmbeddingProvider().embed("Explain Byzantine broadcast in one paragraph.")
            from mlcache import CacheLookup, Query

            stored = cache.lookup(CacheLookup(query=Query("Explain Byzantine broadcast in one paragraph."), embedding=embedding))
            self.assertEqual(stored, first.text)

    def test_second_identical_call_is_served_from_the_cache(self) -> None:
        with tempfile.TemporaryDirectory(prefix="cached-llm-") as tmp:
            cached_llm = CachedLLM(llm=MockLLM(), cache=_build_cache(tmp), namespace="mock-llm")

            first = cached_llm.generate("Explain semantic caching.")
            second = cached_llm.generate("Explain semantic caching.")

            self.assertEqual(first.source, "llm")
            self.assertEqual(second.source, "cache")
            self.assertEqual(second.text, first.text)
            self.assertIsNotNone(second.score)
            self.assertIsNotNone(second.threshold)

    def test_cache_reads_disabled_forces_the_llm_even_with_a_cache_entry(self) -> None:
        with tempfile.TemporaryDirectory(prefix="cached-llm-") as tmp:
            cache = _build_cache(tmp)
            warm = CachedLLM(llm=MockLLM(), cache=cache, namespace="mock-llm")
            warm.generate("Explain semantic caching.")

            no_read_llm = CachedLLM(llm=MockLLM(), cache=cache, namespace="mock-llm", cache_reads=False)
            result = no_read_llm.generate("Explain semantic caching.")

            self.assertEqual(result.source, "llm")

    def test_cache_writes_disabled_prevents_storing_the_response(self) -> None:
        with tempfile.TemporaryDirectory(prefix="cached-llm-") as tmp:
            cache = _build_cache(tmp)
            cached_llm = CachedLLM(llm=MockLLM(), cache=cache, namespace="mock-llm", cache_writes=False)

            first = cached_llm.generate("Explain semantic caching.")
            second = cached_llm.generate("Explain semantic caching.")

            self.assertEqual(first.source, "llm")
            self.assertEqual(second.source, "llm")
            self.assertIsNone(first.cache_key)

    def test_response_is_a_structured_cached_llm_response(self) -> None:
        with tempfile.TemporaryDirectory(prefix="cached-llm-") as tmp:
            cached_llm = CachedLLM(llm=MockLLM(), cache=_build_cache(tmp), namespace="mock-llm")

            result = cached_llm.generate("Explain semantic caching.")

            self.assertIsInstance(result, CachedLLMResponse)
            self.assertEqual(result.source, "llm")
            self.assertTrue(result.text)
            self.assertEqual(result.metadata, {"llm": "mock", "generation_kwargs": {}})

    def test_wrapper_requires_no_api_key(self) -> None:
        with tempfile.TemporaryDirectory(prefix="cached-llm-") as tmp:
            llm = MockLLM()
            cached_llm = CachedLLM(llm=llm, cache=_build_cache(tmp), namespace="mock-llm")

            self.assertFalse(hasattr(llm, "api_key"))
            result = cached_llm.generate("Explain semantic caching.")
            self.assertEqual(result.source, "llm")

    def test_wrapper_makes_no_network_calls(self) -> None:
        def _blocked(*args: object, **kwargs: object) -> None:
            raise AssertionError("CachedLLM must not open network sockets")

        original_socket = socket.socket
        socket.socket = _blocked  # type: ignore[assignment]
        try:
            with tempfile.TemporaryDirectory(prefix="cached-llm-") as tmp:
                cached_llm = CachedLLM(llm=MockLLM(), cache=_build_cache(tmp), namespace="mock-llm")
                first = cached_llm.generate("Explain semantic caching.")
                second = cached_llm.generate("Explain semantic caching.")
        finally:
            socket.socket = original_socket

        self.assertEqual(first.source, "llm")
        self.assertEqual(second.source, "cache")


if __name__ == "__main__":
    unittest.main()
