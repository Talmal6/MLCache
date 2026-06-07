import unittest

from policy import (
    InMemoryQueryLevelShadowDecisionStore as OldInMemoryQueryLevelShadowDecisionStore,
)
from policy import QueryLevelLearnedPolicy as OldQueryLevelLearnedPolicy
from policy import QueryLevelPolicyMode as OldQueryLevelPolicyMode
from runtime import QueryLevelRuntimeConfig as OldQueryLevelRuntimeConfig

from mlcache.cache import KVStore
from mlcache.calibration import InMemoryQueryCalibrationRecordStore, QueryCalibrationCandidate, QueryCalibrationRecord
from mlcache.features import NormalizedHadamardFeatureBuilder
from mlcache.observability import MetricsSink
from mlcache.policies import (
    InMemoryQueryLevelShadowDecisionStore,
    QueryLevelLearnedPolicy,
    QueryLevelPolicyConfig,
    QueryLevelPolicyMode,
)
from mlcache.retrieval import VectorSearchResult, VectorStore
from mlcache.runtime import MLCacheRuntimeConfig, QueryLevelRuntimeConfig, build_mlcache_runtime
from mlcache.scorers import SemanticScorer
from mlcache.semantic_types import (
    CacheEntry,
    CacheKey,
    CacheLookup,
    CacheMetadata,
    Embedding,
    InputSpace,
    LabeledPairBatch,
    OracleDecisionStatus,
    Query,
    Response,
    Score,
    ScorerName,
    Threshold,
    TieMode,
)


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
    def __init__(self) -> None:
        self.entries: dict[CacheKey, CacheEntry] = {}

    def upsert(self, entry: CacheEntry) -> None:
        self.entries[entry.cache_key] = entry

    def delete(self, cache_key: CacheKey) -> None:
        self.entries.pop(cache_key, None)

    def get(self, cache_key: CacheKey) -> VectorSearchResult | None:
        entry = self.entries.get(cache_key)
        if entry is None:
            return None
        return self._result(entry, Score(1.0))

    def search(
        self,
        embedding: Embedding,
        *,
        namespace: str | None = None,
        top_k: int = 10,
    ) -> list[VectorSearchResult]:
        scored = []
        for entry in self.entries.values():
            if namespace is not None and entry.metadata.namespace != namespace:
                continue
            scored.append((self._cosine(embedding, entry.embedding), entry))
        scored.sort(key=lambda item: (-item[0], str(item[1].cache_key)))
        return [self._result(entry, Score(score)) for score, entry in scored[:top_k]]

    @staticmethod
    def _result(entry: CacheEntry, score: Score) -> VectorSearchResult:
        return VectorSearchResult(
            cache_key=entry.cache_key,
            embedding=entry.embedding,
            score=score,
            query=entry.query,
            metadata=entry.metadata,
        )

    @staticmethod
    def _cosine(left: Embedding, right: Embedding) -> float:
        left_values = tuple(float(value) for value in left)
        right_values = tuple(float(value) for value in right)
        left_norm = sum(value * value for value in left_values) ** 0.5
        right_norm = sum(value * value for value in right_values) ** 0.5
        if left_norm <= 0.0 or right_norm <= 0.0:
            return 0.0
        return sum(l * r for l, r in zip(left_values, right_values)) / (left_norm * right_norm)


class DeterministicScorer(SemanticScorer):
    @property
    def name(self) -> ScorerName:
        return ScorerName("query_level_shadow_test")

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


class RecordingMetricsSink(MetricsSink):
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.records: list[tuple[str, float, dict | None]] = []

    def record(self, name: str, value: float, metadata: dict | None = None) -> None:
        if self.fail:
            raise RuntimeError("metrics failed")
        self.records.append((name, value, metadata))


def candidate(
    key: str,
    score: float,
    *,
    rank: int | None = None,
    label: int | None = None,
) -> QueryCalibrationCandidate:
    return QueryCalibrationCandidate(
        score=Score(score),
        label=label,
        candidate_rank=rank,
        candidate_key=CacheKey(key),
        metadata={"candidate_metadata": {"key": key}},
    )


def record(
    candidates: tuple[QueryCalibrationCandidate, ...],
    *,
    query_id: str = "q1",
    query: Query = Query("incoming"),
) -> QueryCalibrationRecord:
    return QueryCalibrationRecord(
        query_id=query_id,
        query=query,
        candidates=candidates,
        metadata={"source": "test"},
    )


def policy(
    *,
    threshold: float | None = 0.5,
    tie_mode: TieMode = TieMode.GE,
    require_threshold: bool = True,
) -> QueryLevelLearnedPolicy:
    return QueryLevelLearnedPolicy(
        threshold=None if threshold is None else Threshold(threshold),
        config=QueryLevelPolicyConfig(
            mode=QueryLevelPolicyMode.SHADOW,
            tie_mode=tie_mode,
            require_threshold=require_threshold,
        ),
    )


def entry(key: str = "cached", embedding: tuple[float, float] = (1.0, 0.0)) -> CacheEntry:
    return CacheEntry(
        cache_key=CacheKey(key),
        query=Query(f"query {key}"),
        response=Response(f"response {key}"),
        embedding=embedding,
        metadata=CacheMetadata(),
    )


def lookup(query_id: str = "q1") -> CacheLookup:
    return CacheLookup(
        query=Query("incoming"),
        embedding=(1.0, 0.0),
        metadata=CacheMetadata(attributes={"query_id": query_id}),
    )


def shadow_runtime(
    *,
    metrics_sink: MetricsSink | None = None,
    shadow_store: InMemoryQueryLevelShadowDecisionStore | None = None,
    active: bool = False,
) -> tuple[object, InMemoryQueryCalibrationRecordStore, InMemoryQueryLevelShadowDecisionStore]:
    query_record_store = InMemoryQueryCalibrationRecordStore(max_records=10)
    query_record_store.add(record((candidate("cached", 0.9, rank=1, label=None),)))
    decision_store = shadow_store or InMemoryQueryLevelShadowDecisionStore(max_decisions=10)
    config = MLCacheRuntimeConfig(
        query_level=QueryLevelRuntimeConfig(
            enabled=True,
            mode=QueryLevelPolicyMode.ACTIVE if active else QueryLevelPolicyMode.SHADOW,
            threshold=Threshold(0.8),
        )
    )
    runtime = build_mlcache_runtime(
        kv_store=InMemoryKVStore(),
        vector_store=InMemoryVectorStore(),
        feature_builder=NormalizedHadamardFeatureBuilder(),
        scorer=DeterministicScorer(),
        metrics_sink=metrics_sink,
        query_record_store=query_record_store,
        query_level_shadow_store=decision_store,
        config=config,
    )
    runtime.put(entry())
    runtime.oracle._threshold = Threshold(0.5)
    return runtime, query_record_store, decision_store


class QueryLevelPolicyShadowTests(unittest.TestCase):
    def test_policy_abstains_when_threshold_missing_and_required(self) -> None:
        decision = policy(threshold=None).evaluate(record((candidate("a", 0.9, rank=1),)))

        self.assertEqual(decision.status, OracleDecisionStatus.ABSTAIN)
        self.assertFalse(decision.accepted)
        self.assertEqual(decision.reason, "query_level_threshold_missing")

    def test_policy_abstains_when_no_candidates_exist(self) -> None:
        decision = policy().evaluate(record(()))

        self.assertEqual(decision.status, OracleDecisionStatus.ABSTAIN)
        self.assertFalse(decision.accepted)
        self.assertEqual(decision.reason, "no_candidates")

    def test_policy_selects_max_score_candidate_by_default(self) -> None:
        decision = policy(threshold=0.5).evaluate(
            record((candidate("low", 0.4, rank=1), candidate("high", 0.8, rank=2)))
        )

        self.assertTrue(decision.accepted)
        self.assertEqual(decision.selected_candidate_key, CacheKey("high"))
        self.assertEqual(decision.selected_candidate_rank, 2)

    def test_ge_tie_mode_accepts_score_equal_to_threshold(self) -> None:
        decision = policy(threshold=0.5, tie_mode=TieMode.GE).evaluate(record((candidate("a", 0.5, rank=1),)))

        self.assertEqual(decision.status, OracleDecisionStatus.HIT)
        self.assertTrue(decision.accepted)

    def test_gt_tie_mode_rejects_score_equal_to_threshold(self) -> None:
        decision = policy(threshold=0.5, tie_mode=TieMode.GT).evaluate(record((candidate("a", 0.5, rank=1),)))

        self.assertEqual(decision.status, OracleDecisionStatus.MISS)
        self.assertFalse(decision.accepted)
        self.assertEqual(decision.reason, "selected_score_below_threshold")

    def test_accepted_decision_contains_selected_candidate_fields(self) -> None:
        decision = policy(threshold=0.7).evaluate(record((candidate("a", 0.9, rank=3, label=None),)))

        self.assertEqual(decision.selected_candidate_key, CacheKey("a"))
        self.assertEqual(decision.selected_candidate_rank, 3)
        self.assertEqual(decision.selected_score, Score(0.9))
        self.assertEqual(decision.threshold, Threshold(0.7))
        self.assertIsNone(decision.metadata["label"])

    def test_rejected_decision_returns_miss_with_below_threshold_reason(self) -> None:
        decision = policy(threshold=0.7).evaluate(record((candidate("a", 0.6, rank=1),)))

        self.assertEqual(decision.status, OracleDecisionStatus.MISS)
        self.assertFalse(decision.accepted)
        self.assertEqual(decision.reason, "selected_score_below_threshold")

    def test_shadow_decision_store_evicts_fifo(self) -> None:
        store = InMemoryQueryLevelShadowDecisionStore(max_decisions=2)
        store.add(policy(threshold=0.5).evaluate(record((candidate("a", 0.9, rank=1),))))
        store.add(policy(threshold=0.5).evaluate(record((candidate("b", 0.9, rank=1),))))
        store.add(policy(threshold=0.5).evaluate(record((candidate("c", 0.9, rank=1),))))

        decisions = store.decisions()
        self.assertEqual([str(item.selected_candidate_key) for item in decisions], ["b", "c"])

    def test_runtime_shadow_mode_stores_decision_after_lookup(self) -> None:
        runtime, _, decision_store = shadow_runtime()

        result = runtime.lookup_with_decision(lookup())

        self.assertEqual(result.decision.status, OracleDecisionStatus.HIT)
        decisions = decision_store.decisions()
        self.assertEqual(len(decisions), 1)
        self.assertEqual(decisions[0].selected_candidate_key, CacheKey("cached"))
        self.assertEqual(decisions[0].status, OracleDecisionStatus.HIT)

    def test_runtime_shadow_decision_does_not_change_gateway_result(self) -> None:
        runtime, _, _ = shadow_runtime()

        direct = runtime.gateway.lookup_with_decision(lookup())
        shadowed = runtime.lookup_with_decision(lookup())

        self.assertEqual(shadowed.response, direct.response)
        self.assertEqual(shadowed.decision.status, direct.decision.status)
        self.assertEqual(shadowed.decision.cache_key, direct.decision.cache_key)

    def test_active_query_level_mode_constructs_with_required_dependencies(self) -> None:
        runtime, _, _ = shadow_runtime(active=True)

        self.assertEqual(runtime.config.query_level.mode, QueryLevelPolicyMode.ACTIVE)

    def test_query_level_shadow_metrics_are_recorded(self) -> None:
        metrics = RecordingMetricsSink()
        runtime, _, _ = shadow_runtime(metrics_sink=metrics)

        runtime.lookup_with_decision(lookup())

        names = [name for name, _, _ in metrics.records]
        self.assertIn("cache.query_level_shadow.evaluated", names)
        self.assertIn("cache.query_level_shadow.accepted", names)
        self.assertIn("cache.query_level_shadow.selected_rank", names)
        shadow_metadata = next(
            metadata for name, _, metadata in metrics.records if name == "cache.query_level_shadow.evaluated"
        )
        self.assertEqual(shadow_metadata["serving_status"], "hit")
        self.assertEqual(shadow_metadata["shadow_status"], "hit")
        self.assertEqual(shadow_metadata["selected_candidate_key"], "cached")

    def test_observability_failure_does_not_change_serving_result(self) -> None:
        metrics = RecordingMetricsSink(fail=True)
        runtime, _, _ = shadow_runtime(metrics_sink=metrics)

        result = runtime.lookup_with_decision(lookup())

        self.assertEqual(result.decision.status, OracleDecisionStatus.HIT)
        self.assertEqual(result.response, Response("response cached"))
        self.assertTrue(any(item["event"] == "observability_failure" for item in runtime.diagnostics))

    def test_old_and_new_imports_work(self) -> None:
        self.assertIs(OldQueryLevelLearnedPolicy, QueryLevelLearnedPolicy)
        self.assertIs(OldQueryLevelPolicyMode, QueryLevelPolicyMode)
        self.assertIs(OldInMemoryQueryLevelShadowDecisionStore, InMemoryQueryLevelShadowDecisionStore)
        self.assertIs(OldQueryLevelRuntimeConfig, QueryLevelRuntimeConfig)


if __name__ == "__main__":
    unittest.main()
