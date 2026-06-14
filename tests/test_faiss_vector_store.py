from __future__ import annotations

import importlib.util
import tempfile
from pathlib import Path
import unittest

from mlcache.retrieval import FaissVectorStore, InMemoryVectorStore
from mlcache.semantic_types import CacheEntry, CacheKey, CacheMetadata, Query, Response


def entry(key: str, embedding: tuple[float, ...]) -> CacheEntry:
    return CacheEntry(
        cache_key=CacheKey(key),
        query=Query(f"query {key}"),
        response=Response(f"response {key}"),
        embedding=embedding,
        metadata=CacheMetadata(),
    )


@unittest.skipUnless(importlib.util.find_spec("faiss") is not None, "requires faiss; install with .[faiss]")
class FaissVectorStoreTests(unittest.TestCase):
    def test_faiss_exact_flat_ip_matches_in_memory_top1(self) -> None:
        rows = (
            entry("a", (1.0, 0.0, 0.0)),
            entry("b", (0.0, 1.0, 0.0)),
            entry("c", (0.9, 0.1, 0.0)),
        )
        memory = InMemoryVectorStore()
        faiss_store = FaissVectorStore(dim=3)
        for row in rows:
            memory.upsert(row)
            faiss_store.upsert(row)

        expected = memory.search((1.0, 0.0, 0.0), top_k=1)[0]
        actual = faiss_store.search((1.0, 0.0, 0.0), top_k=1)[0]

        self.assertEqual(actual.cache_key, expected.cache_key)

    def test_faiss_snapshot_round_trips_index(self) -> None:
        with tempfile.TemporaryDirectory(prefix="mlcache-faiss-") as tmp:
            path = Path(tmp) / "index.faiss"
            first = FaissVectorStore(dim=2, index_path=path)
            first.upsert(entry("a", (1.0, 0.0)))
            snapshot = first.save_snapshot()

            second = FaissVectorStore(index_path=path, autoload=False)
            second.load_snapshot(path, expected_checksum=snapshot["checksum"])

            self.assertEqual(second.size(), 1)


if __name__ == "__main__":
    unittest.main()
