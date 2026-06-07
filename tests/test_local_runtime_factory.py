import tempfile
import unittest
from pathlib import Path

from mlcache.cache import FileKVStore, InMemoryKVStore
from mlcache.calibration import (
    FileQueryCalibrationRecordStore,
    FileThresholdProvider,
    InMemoryThresholdProvider,
    QueryCalibrationCandidate,
    QueryCalibrationRecord,
    ThresholdScope,
)
from mlcache.features import NormalizedHadamardFeatureBuilder
from mlcache.retrieval import FileVectorStore, InMemoryVectorStore
from mlcache.runtime import MLCacheRuntimeConfig, QueryLevelRuntimeConfig, build_local_mlcache_runtime
from mlcache.scorers import SemanticScorer
from mlcache.semantic_types import (
    CacheEntry,
    CacheKey,
    CacheLookup,
    CacheMetadata,
    Embedding,
    InputSpace,
    LabeledPairBatch,
    Query,
    Response,
    Score,
    ScorerName,
    Threshold,
    TieMode,
)
from mlcache.policies import QueryLevelPolicyMode


class DeterministicScorer(SemanticScorer):
    @property
    def name(self) -> ScorerName:
        return ScorerName("local_runtime_scorer")

    @property
    def input_space(self) -> InputSpace:
        return InputSpace.EMBEDDING

    def fit(self, batch: LabeledPairBatch, **kwargs: object) -> None:
        del batch, kwargs

    def copy_for_refit(self) -> "DeterministicScorer":
        return type(self)()

    def score(self, features) -> Score:
        if features.cosine is not None:
            return Score(float(features.cosine))
        if features.hadamard:
            return Score(float(sum(features.hadamard)))
        return Score(0.0)

    def calibrate(self, request) -> Threshold:
        del request
        return Threshold(0.5)

    def predict(self, features, threshold: Threshold, *, tie_mode: TieMode = TieMode.GE) -> bool:
        score = float(self.score(features))
        return score > float(threshold) if tie_mode == TieMode.GT else score >= float(threshold)


def entry(key: str, embedding: Embedding, response: str) -> CacheEntry:
    return CacheEntry(
        cache_key=CacheKey(key),
        query=Query(f"query {key}"),
        response=Response(response),
        embedding=embedding,
        metadata=CacheMetadata(),
    )


def lookup(query_id: str = "q-local") -> CacheLookup:
    return CacheLookup(
        query=Query("incoming local"),
        embedding=(1.0, 0.0),
        metadata=CacheMetadata(attributes={"query_id": query_id}),
    )


def query_record() -> QueryCalibrationRecord:
    return QueryCalibrationRecord(
        query_id="q-local",
        query=Query("incoming local"),
        candidates=(
            QueryCalibrationCandidate(
                score=Score(0.9),
                label=None,
                candidate_rank=1,
                candidate_key=CacheKey("ql"),
                metadata={"source": "local_test"},
            ),
        ),
        metadata={"source": "local_test"},
    )


def active_config(threshold: Threshold | None = Threshold(0.8)) -> MLCacheRuntimeConfig:
    return MLCacheRuntimeConfig(
        query_level=QueryLevelRuntimeConfig(
            enabled=True,
            mode=QueryLevelPolicyMode.ACTIVE,
            threshold=threshold,
        )
    )


class LocalRuntimeFactoryTests(unittest.TestCase):
    def test_build_local_runtime_memory_mode_works(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            runtime = build_local_mlcache_runtime(
                root_dir=tmp,
                feature_builder=NormalizedHadamardFeatureBuilder(),
                scorer=DeterministicScorer(),
                use_file_persistence=False,
            )
            runtime.put(entry("a", (1.0, 0.0), "response a"))
            runtime.oracle._threshold = Threshold(0.5)

            result = runtime.lookup_with_decision(lookup())

            self.assertIsInstance(runtime.kv_store, InMemoryKVStore)
            self.assertIsInstance(runtime.vector_store, InMemoryVectorStore)
            self.assertIsInstance(runtime.oracle.threshold_provider, InMemoryThresholdProvider)
            self.assertEqual(result.response, Response("response a"))

    def test_build_local_runtime_file_persistence_mode_works(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            runtime = build_local_mlcache_runtime(
                root_dir=tmp,
                feature_builder=NormalizedHadamardFeatureBuilder(),
                scorer=DeterministicScorer(),
                use_file_persistence=True,
            )
            runtime.put(entry("a", (1.0, 0.0), "response a"))
            runtime.oracle._threshold = Threshold(0.5)

            result = runtime.lookup_with_decision(lookup())

            self.assertIsInstance(runtime.kv_store, FileKVStore)
            self.assertIsInstance(runtime.vector_store, FileVectorStore)
            self.assertIsInstance(runtime.oracle.threshold_provider, FileThresholdProvider)
            self.assertEqual(result.response, Response("response a"))

    def test_runtime_put_lookup_works_after_file_restart(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            first = build_local_mlcache_runtime(
                root_dir=tmp,
                feature_builder=NormalizedHadamardFeatureBuilder(),
                scorer=DeterministicScorer(),
            )
            first.put(entry("a", (1.0, 0.0), "response a"))
            first.oracle.threshold_provider.set_threshold(
                Threshold(0.5),
                scorer=first.oracle.scorer.name,
                scope=ThresholdScope.GLOBAL,
            )

            second = build_local_mlcache_runtime(
                root_dir=tmp,
                feature_builder=NormalizedHadamardFeatureBuilder(),
                scorer=DeterministicScorer(),
            )
            result = second.lookup_with_decision(lookup())

            self.assertEqual(result.response, Response("response a"))

    def test_query_calibration_records_survive_file_restart(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            first = build_local_mlcache_runtime(
                root_dir=tmp,
                feature_builder=NormalizedHadamardFeatureBuilder(),
                scorer=DeterministicScorer(),
                config=active_config(),
            )
            first.query_record_store.add(query_record())

            second = build_local_mlcache_runtime(
                root_dir=tmp,
                feature_builder=NormalizedHadamardFeatureBuilder(),
                scorer=DeterministicScorer(),
                config=active_config(threshold=None),
            )

            self.assertIsInstance(second.query_record_store, FileQueryCalibrationRecordStore)
            self.assertEqual(second.query_record_store.records()[0].query_id, "q-local")

    def test_active_query_level_threshold_survives_file_restart(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            first = build_local_mlcache_runtime(
                root_dir=tmp,
                feature_builder=NormalizedHadamardFeatureBuilder(),
                scorer=DeterministicScorer(),
                config=active_config(Threshold(0.8)),
            )
            first.put(entry("pair", (1.0, 0.0), "pair response"))
            first.put(entry("ql", (0.0, 1.0), "query-level response"))
            first.query_record_store.add(query_record())

            second = build_local_mlcache_runtime(
                root_dir=tmp,
                feature_builder=NormalizedHadamardFeatureBuilder(),
                scorer=DeterministicScorer(),
                config=active_config(threshold=None),
            )
            result = second.lookup_with_decision(lookup())

            self.assertEqual(second.config.query_level.threshold, Threshold(0.8))
            self.assertEqual(result.response, Response("query-level response"))
            self.assertEqual(result.metadata["final_decision_source"], "query_level_active")

    def test_new_imports_work(self) -> None:
        self.assertIsNotNone(InMemoryKVStore)
        self.assertIsNotNone(FileKVStore)
        self.assertIsNotNone(InMemoryVectorStore)
        self.assertIsNotNone(FileVectorStore)
        self.assertIsNotNone(InMemoryThresholdProvider)
        self.assertIsNotNone(FileThresholdProvider)
        self.assertIsNotNone(FileQueryCalibrationRecordStore)
        self.assertIsNotNone(build_local_mlcache_runtime)


if __name__ == "__main__":
    unittest.main()
