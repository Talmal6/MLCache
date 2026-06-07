import tempfile
import unittest
from pathlib import Path

from cache import FileKVStore as OldFileKVStore
from cache import InMemoryKVStore as OldInMemoryKVStore
from thresholds import FileThresholdProvider as OldFileThresholdProvider
from thresholds import InMemoryThresholdProvider as OldInMemoryThresholdProvider
from vector_store import FileVectorStore as OldFileVectorStore
from vector_store import InMemoryVectorStore as OldInMemoryVectorStore

from mlcache.cache import FileKVStore, InMemoryKVStore
from mlcache.calibration import (
    FileQueryCalibrationRecordStore,
    FileThresholdProvider,
    InMemoryThresholdProvider,
    QueryCalibrationCandidate,
    QueryCalibrationRecord,
    ThresholdScope,
)
from mlcache.policies import FileQueryLevelShadowDecisionStore, QueryLevelPolicyDecision
from mlcache.retrieval import FileVectorStore, InMemoryVectorStore
from mlcache.semantic_types import (
    CacheEntry,
    CacheKey,
    CacheMetadata,
    OracleDecisionStatus,
    Query,
    Response,
    Score,
    ScorerName,
    Threshold,
)


def entry(
    key: str,
    embedding: tuple[float, ...],
    *,
    namespace: str | None = None,
) -> CacheEntry:
    return CacheEntry(
        cache_key=CacheKey(key),
        query=Query(f"query {key}"),
        response=Response(f"response {key}"),
        embedding=embedding,
        metadata=CacheMetadata(namespace=namespace, attributes={"key": key}),
    )


def query_record(query_id: str = "q1", score: float = 0.4) -> QueryCalibrationRecord:
    return QueryCalibrationRecord(
        query_id=query_id,
        query=Query("incoming"),
        candidates=(
            QueryCalibrationCandidate(
                score=Score(score),
                label=0,
                candidate_rank=1,
                candidate_key=CacheKey(f"candidate-{query_id}"),
                metadata={"source": "test"},
            ),
        ),
        metadata={"source": "file_test"},
    )


class FilePersistenceTests(unittest.TestCase):
    def test_file_kv_store_persists_across_new_instance(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "kv.json"
            first = FileKVStore(path)
            first.set(CacheKey("a"), Response("response a"))

            second = FileKVStore(path)

            self.assertEqual(second.get(CacheKey("a")), Response("response a"))
            self.assertTrue(second.contains(CacheKey("a")))

    def test_file_vector_store_persists_across_new_instance(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "vectors.json"
            first = FileVectorStore(path)
            first.upsert(entry("a", (1.0, 0.0), namespace="n"))

            second = FileVectorStore(path)
            result = second.get(CacheKey("a"))

            self.assertIsNotNone(result)
            self.assertEqual(result.query, Query("query a"))
            self.assertEqual(result.metadata.namespace, "n")

    def test_file_vector_store_search_after_reload_preserves_order(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "vectors.json"
            first = FileVectorStore(path)
            first.upsert(entry("b", (1.0, 0.0)))
            first.upsert(entry("a", (1.0, 0.0)))
            first.upsert(entry("c", (0.0, 1.0)))

            second = FileVectorStore(path)
            results = second.search((1.0, 0.0), top_k=3)

            self.assertEqual([str(result.cache_key) for result in results], ["a", "b", "c"])

    def test_file_vector_store_namespace_filtering_after_reload(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "vectors.json"
            first = FileVectorStore(path)
            first.upsert(entry("a", (1.0, 0.0), namespace="one"))
            first.upsert(entry("b", (1.0, 0.0), namespace="two"))

            second = FileVectorStore(path)
            results = second.search((1.0, 0.0), namespace="two")

            self.assertEqual([result.cache_key for result in results], [CacheKey("b")])

    def test_file_threshold_provider_persists_global_threshold(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "thresholds.json"
            first = FileThresholdProvider(path)
            first.set_threshold(Threshold(0.8), scorer=ScorerName("s"), scope=ThresholdScope.GLOBAL)

            second = FileThresholdProvider(path)

            self.assertEqual(second.get_threshold(scorer=ScorerName("s")), Threshold(0.8))

    def test_file_query_calibration_record_store_persists_records(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "records.json"
            first = FileQueryCalibrationRecordStore(path, max_records=10)
            first.add(query_record("q1", score=0.2))

            second = FileQueryCalibrationRecordStore(path, max_records=10)
            records = second.records()

            self.assertEqual(len(records), 1)
            self.assertEqual(records[0].query_id, "q1")
            self.assertEqual(records[0].candidates[0].score, Score(0.2))

    def test_file_query_calibration_record_store_fifo_eviction(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "records.json"
            first = FileQueryCalibrationRecordStore(path, max_records=2)
            first.add(query_record("q1"))
            first.add(query_record("q2"))
            first.add(query_record("q3"))

            second = FileQueryCalibrationRecordStore(path, max_records=2)

            self.assertEqual([record.query_id for record in second.records()], ["q2", "q3"])

    def test_file_query_level_shadow_decision_store_persists_decisions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "shadow.json"
            first = FileQueryLevelShadowDecisionStore(path, max_decisions=10)
            first.add(
                QueryLevelPolicyDecision(
                    status=OracleDecisionStatus.HIT,
                    accepted=True,
                    selected_candidate_key=CacheKey("a"),
                    selected_candidate_rank=1,
                    selected_score=Score(0.9),
                    threshold=Threshold(0.8),
                    reason=None,
                    metadata={"source": "test"},
                )
            )

            second = FileQueryLevelShadowDecisionStore(path, max_decisions=10)
            decisions = second.decisions()

            self.assertEqual(len(decisions), 1)
            self.assertEqual(decisions[0].selected_candidate_key, CacheKey("a"))
            self.assertEqual(decisions[0].threshold, Threshold(0.8))

    def test_old_and_new_imports_work(self) -> None:
        self.assertIs(OldInMemoryKVStore, InMemoryKVStore)
        self.assertIs(OldFileKVStore, FileKVStore)
        self.assertIs(OldInMemoryVectorStore, InMemoryVectorStore)
        self.assertIs(OldFileVectorStore, FileVectorStore)
        self.assertIs(OldInMemoryThresholdProvider, InMemoryThresholdProvider)
        self.assertIs(OldFileThresholdProvider, FileThresholdProvider)


if __name__ == "__main__":
    unittest.main()
