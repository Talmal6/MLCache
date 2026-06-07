import unittest

from thresholds import InMemoryQueryCalibrationRecordStore as OldInMemoryQueryCalibrationRecordStore
from thresholds import QueryCalibrationRecordBuilder as OldQueryCalibrationRecordBuilder
from thresholds import QueryCalibrationRecordStore as OldQueryCalibrationRecordStore

from mlcache.cache import KVStore, SemanticCacheGateway
from mlcache.calibration import (
    DefaultQueryLevelCalibrationBuilder,
    InMemoryQueryCalibrationRecordStore,
    QueryCalibrationRecordBuilder,
    QueryCalibrationRecordStore,
    ThresholdCalibrationRequest,
)
from mlcache.features import NormalizedHadamardFeatureBuilder
from mlcache.feedback import (
    DefaultShadowTopKCollector,
    InMemorySplitJudgeTrainingStore,
    JudgeDecision,
    JudgeLabel,
    JudgeRequest,
    JudgeResult,
    SemanticReuseJudge,
    ShadowCollectionConfig,
)
from mlcache.oracle import TrainableSemanticCacheOracle
from mlcache.retrieval import VectorSearchResult, VectorStore
from mlcache.scorers import SemanticScorer
from mlcache.semantic_types import (
    CacheEntry,
    CacheKey,
    CacheLookup,
    CacheMetadata,
    Embedding,
    InputSpace,
    LabeledPairBatch,
    OracleDecision,
    OracleDecisionStatus,
    Query,
    Response,
    Score,
    ScorerName,
    Threshold,
    TieMode,
)


class DeterministicJudge(SemanticReuseJudge):
    def __init__(self, labels: dict[str, JudgeLabel], *, failures: set[str] | None = None) -> None:
        self.labels = labels
        self.failures = failures or set()

    @property
    def name(self) -> str:
        return "query-record-judge"

    def judge(self, request: JudgeRequest) -> JudgeResult:
        key = str(request.candidate_key)
        if key in self.failures:
            raise RuntimeError(f"judge failed for {key}")
        return JudgeResult(request=request, decision=JudgeDecision(label=self.labels.get(key, JudgeLabel.UNCERTAIN)))


class InMemoryKVStore(KVStore):
    def __init__(self) -> None:
        self.values: dict[CacheKey, Response] = {}

    def get(self, cache_key: CacheKey) -> Response | None:
        return self.values.get(cache_key)

    def set(self, cache_key: CacheKey, response: Response) -> None:
        self.values[cache_key] = response

    def delete(self, cache_key: CacheKey) -> None:
        self.values.pop(cache_key, None)

    def contains(self, cache_key: CacheKey) -> bool:
        return cache_key in self.values


class InMemoryVectorStore(VectorStore):
    def __init__(self, entries: list[CacheEntry] | None = None) -> None:
        self.entries = entries or []

    def upsert(self, entry: CacheEntry) -> None:
        self.entries.append(entry)

    def delete(self, cache_key: CacheKey) -> None:
        self.entries = [entry for entry in self.entries if entry.cache_key != cache_key]

    def get(self, cache_key: CacheKey) -> VectorSearchResult | None:
        for entry in self.entries:
            if entry.cache_key == cache_key:
                return result(entry, 1.0)
        return None

    def search(
        self,
        embedding: Embedding,
        *,
        namespace: str | None = None,
        top_k: int = 10,
    ) -> list[VectorSearchResult]:
        del embedding, namespace
        return [result(entry, float(idx + 1) / 10.0) for idx, entry in enumerate(self.entries[:top_k])]


class CosineScorer(SemanticScorer):
    @property
    def name(self) -> ScorerName:
        return ScorerName("query_record_test")

    @property
    def input_space(self) -> InputSpace:
        return InputSpace.EMBEDDING

    def fit(self, batch: LabeledPairBatch, **kwargs: object) -> None:
        del batch, kwargs

    def copy_for_refit(self) -> "CosineScorer":
        return type(self)()

    def score(self, features) -> Score:
        return Score(float(features.cosine or 0.0))

    def calibrate(self, request: ThresholdCalibrationRequest) -> Threshold:
        del request
        return Threshold(0.5)

    def predict(self, features, threshold: Threshold, *, tie_mode: TieMode = TieMode.GE) -> bool:
        score = float(self.score(features))
        return score > float(threshold) if tie_mode == TieMode.GT else score >= float(threshold)


def candidate(
    key: str,
    score: float,
    *,
    query: str | None = None,
    embedding: tuple[float, float] = (1.0, 0.0),
    metadata: CacheMetadata | None = None,
) -> VectorSearchResult:
    return VectorSearchResult(
        cache_key=CacheKey(key),
        embedding=embedding,
        score=Score(score),
        query=Query(query or f"cached {key}"),
        metadata=metadata or CacheMetadata(attributes={"candidate": key}),
    )


def entry(key: str, embedding: tuple[float, float] = (1.0, 0.0)) -> CacheEntry:
    return CacheEntry(
        cache_key=CacheKey(key),
        query=Query(f"cached {key}"),
        response=Response(f"response {key}"),
        embedding=embedding,
        metadata=CacheMetadata(),
    )


def result(cached: CacheEntry, score: float) -> VectorSearchResult:
    return VectorSearchResult(
        cache_key=cached.cache_key,
        embedding=cached.embedding,
        score=Score(score),
        query=cached.query,
        metadata=cached.metadata,
    )


def request(query_id: str = "q-1") -> CacheLookup:
    return CacheLookup(
        query=Query("incoming query"),
        embedding=(1.0, 0.0),
        metadata=CacheMetadata(attributes={"query_id": query_id}),
    )


def split_store() -> InMemorySplitJudgeTrainingStore:
    return InMemorySplitJudgeTrainingStore(
        max_h0_train=20,
        max_h1_train=20,
        max_h0_calibration=20,
        max_h1_calibration=20,
    )


def served_decision() -> OracleDecision:
    return OracleDecision(
        status=OracleDecisionStatus.HIT,
        accepted=True,
        cache_key=CacheKey("c1"),
        score=Score(0.9),
        threshold=Threshold(0.5),
        scorer=ScorerName("test"),
        candidate_count=3,
    )


class QueryCalibrationRecordTests(unittest.TestCase):
    def test_builder_creates_one_query_calibration_record_per_request(self) -> None:
        builder = QueryCalibrationRecordBuilder()

        record = builder.build_record(
            query_id="q-1",
            request=request(),
            candidates=[candidate("c1", 0.1), candidate("c2", 0.2)],
        )

        self.assertEqual(record.query_id, "q-1")
        self.assertEqual(record.query, Query("incoming query"))
        self.assertEqual(len(record.candidates), 2)

    def test_candidate_ranks_are_one_based_and_preserve_top_k_order(self) -> None:
        record = QueryCalibrationRecordBuilder().build_record(
            query_id="q-1",
            request=request(),
            candidates=[candidate("first", 0.3), candidate("second", 0.9), candidate("third", 0.1)],
        )

        self.assertEqual([item.candidate_rank for item in record.candidates], [1, 2, 3])
        self.assertEqual([item.candidate_key for item in record.candidates], [
            CacheKey("first"),
            CacheKey("second"),
            CacheKey("third"),
        ])

    def test_candidate_score_uses_explicit_candidate_scores_when_provided(self) -> None:
        record = QueryCalibrationRecordBuilder().build_record(
            query_id="q-1",
            request=request(),
            candidates=[candidate("c1", 0.1)],
            candidate_scores={CacheKey("c1"): Score(0.77)},
        )

        self.assertEqual(record.candidates[0].score, Score(0.77))
        self.assertEqual(record.candidates[0].metadata["score_source"], "scorer_score")
        self.assertEqual(record.candidates[0].metadata["scorer_score"], 0.77)

    def test_candidate_score_falls_back_to_vector_score(self) -> None:
        record = QueryCalibrationRecordBuilder().build_record(
            query_id="q-1",
            request=request(),
            candidates=[candidate("c1", 0.42)],
        )

        self.assertEqual(record.candidates[0].score, Score(0.42))
        self.assertEqual(record.candidates[0].metadata["score_source"], "vector_score")

    def test_candidate_labels_are_attached_from_mapping(self) -> None:
        record = QueryCalibrationRecordBuilder().build_record(
            query_id="q-1",
            request=request(),
            candidates=[candidate("h0", 0.1), candidate("h1", 0.9)],
            candidate_labels={CacheKey("h0"): 0, CacheKey("h1"): 1},
        )

        self.assertEqual([item.label for item in record.candidates], [0, 1])

    def test_missing_labels_become_none(self) -> None:
        record = QueryCalibrationRecordBuilder().build_record(
            query_id="q-1",
            request=request(),
            candidates=[candidate("labeled", 0.1), candidate("missing", 0.9)],
            candidate_labels={CacheKey("labeled"): 0},
        )

        self.assertEqual([item.label for item in record.candidates], [0, None])

    def test_metadata_preserves_source_scores_and_candidate_metadata(self) -> None:
        metadata = CacheMetadata(attributes={"payload": "kept"})
        builder = QueryCalibrationRecordBuilder(source="custom_source")

        record = builder.build_record(
            query_id="q-1",
            request=request(),
            candidates=[candidate("c1", 0.9, metadata=metadata)],
        )

        self.assertEqual(record.metadata["source"], "custom_source")
        self.assertEqual(record.candidates[0].metadata["source"], "custom_source")
        self.assertEqual(record.candidates[0].metadata["vector_score"], 0.9)
        self.assertEqual(record.candidates[0].metadata["score_source"], "vector_score")
        self.assertEqual(record.candidates[0].metadata["candidate_metadata"], metadata)

    def test_in_memory_store_stores_records_and_returns_tuple(self) -> None:
        store = InMemoryQueryCalibrationRecordStore(max_records=2)
        record = QueryCalibrationRecordBuilder().build_record(
            query_id="q-1",
            request=request(),
            candidates=[candidate("c1", 0.1)],
        )

        store.add(record)
        records = store.records()

        self.assertIsInstance(records, tuple)
        self.assertEqual(len(records), 1)
        record.metadata["mutated"] = True
        self.assertNotIn("mutated", store.records()[0].metadata)

    def test_store_evicts_fifo_when_capacity_is_exceeded(self) -> None:
        store = InMemoryQueryCalibrationRecordStore(max_records=2)
        builder = QueryCalibrationRecordBuilder()
        for idx in range(3):
            store.add(
                builder.build_record(
                    query_id=f"q-{idx}",
                    request=request(f"q-{idx}"),
                    candidates=[candidate(f"c{idx}", 0.1)],
                )
            )

        self.assertEqual([item.query_id for item in store.records()], ["q-1", "q-2"])

    def test_shadow_collector_does_not_create_query_records_by_default(self) -> None:
        record_store = InMemoryQueryCalibrationRecordStore(max_records=10)
        collector = DefaultShadowTopKCollector(
            feature_builder=NormalizedHadamardFeatureBuilder(),
            judge=DeterministicJudge({"c1": JudgeLabel.REUSABLE}),
            store=split_store(),
            query_record_builder=QueryCalibrationRecordBuilder(),
            query_record_store=record_store,
        )

        collector.collect(request(), [candidate("c1", 0.9)], served_decision())

        self.assertEqual(record_store.records(), ())

    def test_shadow_collector_stores_one_query_record_when_enabled(self) -> None:
        record_store = InMemoryQueryCalibrationRecordStore(max_records=10)
        collector = DefaultShadowTopKCollector(
            feature_builder=NormalizedHadamardFeatureBuilder(),
            judge=DeterministicJudge({"c1": JudgeLabel.REUSABLE, "c2": JudgeLabel.NOT_REUSABLE}),
            store=split_store(),
            query_record_builder=QueryCalibrationRecordBuilder(),
            query_record_store=record_store,
            record_query_calibration=True,
            config=ShadowCollectionConfig(top_k=2),
        )

        collector.collect(request("runtime-q"), [candidate("c1", 0.9), candidate("c2", 0.8)], served_decision())

        records = record_store.records()
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].query_id, "runtime-q")
        self.assertEqual([item.label for item in records[0].candidates], [1, 0])

    def test_shadow_record_contains_all_top_k_candidates_with_unlabeled_failures(self) -> None:
        record_store = InMemoryQueryCalibrationRecordStore(max_records=10)
        collector = DefaultShadowTopKCollector(
            feature_builder=NormalizedHadamardFeatureBuilder(),
            judge=DeterministicJudge(
                {"c1": JudgeLabel.REUSABLE, "c3": JudgeLabel.NOT_REUSABLE},
                failures={"c2"},
            ),
            store=split_store(),
            query_record_builder=QueryCalibrationRecordBuilder(),
            query_record_store=record_store,
            record_query_calibration=True,
            config=ShadowCollectionConfig(top_k=3),
        )

        collector.collect(
            request(),
            [candidate("c1", 0.9), candidate("c2", 0.8), candidate("c3", 0.7)],
            served_decision(),
        )

        record = record_store.records()[0]
        self.assertEqual([item.candidate_key for item in record.candidates], [
            CacheKey("c1"),
            CacheKey("c2"),
            CacheKey("c3"),
        ])
        self.assertEqual([item.label for item in record.candidates], [1, None, 0])

    def test_record_creation_does_not_change_serving_decision(self) -> None:
        cached = entry("c1")
        record_store = InMemoryQueryCalibrationRecordStore(max_records=10)
        collector = DefaultShadowTopKCollector(
            feature_builder=NormalizedHadamardFeatureBuilder(),
            judge=DeterministicJudge({"c1": JudgeLabel.REUSABLE}),
            store=split_store(),
            query_record_builder=QueryCalibrationRecordBuilder(),
            query_record_store=record_store,
            record_query_calibration=True,
        )
        oracle = TrainableSemanticCacheOracle(
            vector_store=InMemoryVectorStore([cached]),
            feature_builder=NormalizedHadamardFeatureBuilder(),
            scorer=CosineScorer(),
            shadow_collector=collector,
            shadow_collection_enabled=True,
            auto_refit=False,
        )
        oracle._threshold = Threshold(0.5)
        gateway = SemanticCacheGateway(kv_store=InMemoryKVStore(), oracle=oracle)
        gateway.kv_store.set(cached.cache_key, cached.response)

        result = gateway.lookup_with_decision(request())

        self.assertEqual(result.decision.status, OracleDecisionStatus.HIT)
        self.assertEqual(result.response, cached.response)
        self.assertEqual(len(record_store.records()), 1)

    def test_old_and_new_imports_work(self) -> None:
        self.assertIs(OldQueryCalibrationRecordBuilder, QueryCalibrationRecordBuilder)
        self.assertIs(OldQueryCalibrationRecordStore, QueryCalibrationRecordStore)
        self.assertIs(OldInMemoryQueryCalibrationRecordStore, InMemoryQueryCalibrationRecordStore)

    def test_shadow_records_feed_query_level_calibration_builder(self) -> None:
        record_store = InMemoryQueryCalibrationRecordStore(max_records=10)
        collector = DefaultShadowTopKCollector(
            feature_builder=NormalizedHadamardFeatureBuilder(),
            judge=DeterministicJudge(
                {
                    "h0-selected": JudgeLabel.NOT_REUSABLE,
                    "h1": JudgeLabel.REUSABLE,
                    "h0-lower": JudgeLabel.NOT_REUSABLE,
                }
            ),
            store=split_store(),
            query_record_builder=QueryCalibrationRecordBuilder(),
            query_record_store=record_store,
            record_query_calibration=True,
            config=ShadowCollectionConfig(top_k=3),
        )

        collector.collect(
            request(),
            [
                candidate("h0-selected", 0.9),
                candidate("h1", 0.8),
                candidate("h0-lower", 0.7),
            ],
            served_decision(),
        )
        dataset = DefaultQueryLevelCalibrationBuilder().build_calibration_decisions(record_store.records())

        self.assertEqual(len(dataset.decisions), 1)
        self.assertEqual(dataset.decisions[0].candidate_key, CacheKey("h0-selected"))
        self.assertEqual(tuple(float(score) for score in dataset.h0_scores), (0.9,))
        self.assertEqual(tuple(float(score) for score in dataset.all_pair_h0_scores), (0.9, 0.7))


if __name__ == "__main__":
    unittest.main()
