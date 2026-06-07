import unittest

from policy import QueryLevelLearnedPolicy as OldQueryLevelLearnedPolicy
from policy import QueryLevelPolicyMode as OldQueryLevelPolicyMode
from runtime import QueryLevelRuntimeConfig as OldQueryLevelRuntimeConfig

from mlcache.cache import KVStore
from mlcache.calibration import InMemoryQueryCalibrationRecordStore, QueryCalibrationCandidate, QueryCalibrationRecord
from mlcache.features import NormalizedHadamardFeatureBuilder
from mlcache.observability import MetricsSink
from mlcache.policies import QueryLevelLearnedPolicy, QueryLevelPolicyConfig, QueryLevelPolicyMode
from mlcache.policies import InMemoryQueryLevelShadowDecisionStore
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
        self.removed: list[CacheKey] = []

    def upsert(self, entry: CacheEntry) -> None:
        self.entries[entry.cache_key] = entry

    def delete(self, cache_key: CacheKey) -> None:
        self.removed.append(cache_key)
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
        return ScorerName("query_level_active_test")

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


def candidate(key: str, score: float, *, rank: int = 1, label: int | None = None) -> QueryCalibrationCandidate:
    return QueryCalibrationCandidate(
        score=Score(score),
        label=label,
        candidate_rank=rank,
        candidate_key=CacheKey(key),
        metadata={"candidate_metadata": {"key": key}},
    )


def query_record(
    candidates: tuple[QueryCalibrationCandidate, ...],
    *,
    query_id: str = "q-active",
) -> QueryCalibrationRecord:
    return QueryCalibrationRecord(
        query_id=query_id,
        query=Query("incoming active"),
        candidates=candidates,
        metadata={"source": "active_test"},
    )


def entry(key: str, embedding: tuple[float, float], response: str) -> CacheEntry:
    return CacheEntry(
        cache_key=CacheKey(key),
        query=Query(f"cached {key}"),
        response=Response(response),
        embedding=embedding,
        metadata=CacheMetadata(),
    )


def lookup(query_id: str = "q-active") -> CacheLookup:
    return CacheLookup(
        query=Query("incoming active"),
        embedding=(1.0, 0.0),
        metadata=CacheMetadata(attributes={"query_id": query_id}),
    )


def active_config(
    *,
    mode: QueryLevelPolicyMode = QueryLevelPolicyMode.ACTIVE,
    threshold: float | None = 0.8,
    fallback_on_abstain: bool = True,
    fallback_on_missing_record: bool = True,
    fallback_on_kv_miss: bool = True,
    active_requires_threshold: bool = True,
) -> MLCacheRuntimeConfig:
    return MLCacheRuntimeConfig(
        query_level=QueryLevelRuntimeConfig(
            enabled=True,
            mode=mode,
            threshold=None if threshold is None else Threshold(threshold),
            fallback_to_pair_level_on_abstain=fallback_on_abstain,
            fallback_to_pair_level_on_missing_record=fallback_on_missing_record,
            fallback_to_pair_level_on_kv_miss=fallback_on_kv_miss,
            active_requires_threshold=active_requires_threshold,
        )
    )


def build_runtime(
    *,
    config: MLCacheRuntimeConfig | None = None,
    record_candidates: tuple[QueryCalibrationCandidate, ...] | None = None,
    include_record: bool = True,
    include_query_candidate_in_kv: bool = True,
    query_level_policy: QueryLevelLearnedPolicy | None = None,
    metrics_sink: MetricsSink | None = None,
    shadow_store: InMemoryQueryLevelShadowDecisionStore | None = None,
):
    kv = InMemoryKVStore()
    vectors = InMemoryVectorStore()
    record_store = InMemoryQueryCalibrationRecordStore(max_records=10)
    if include_record:
        candidates = record_candidates if record_candidates is not None else (candidate("ql", 0.9),)
        record_store.add(query_record(candidates))

    runtime = build_mlcache_runtime(
        kv_store=kv,
        vector_store=vectors,
        feature_builder=NormalizedHadamardFeatureBuilder(),
        scorer=DeterministicScorer(),
        query_record_store=record_store,
        query_level_policy=query_level_policy,
        query_level_shadow_store=shadow_store,
        metrics_sink=metrics_sink,
        config=config,
    )
    runtime.put(entry("pair", (1.0, 0.0), "pair response"))
    query_candidate = entry("ql", (0.0, 1.0), "query-level response")
    if include_query_candidate_in_kv:
        runtime.put(query_candidate)
    else:
        vectors.upsert(query_candidate)
    runtime.oracle._threshold = Threshold(0.5)
    return runtime, kv, vectors, record_store


class QueryLevelActiveServingTests(unittest.TestCase):
    def test_default_runtime_behavior_remains_pair_level_serving(self) -> None:
        runtime, _, _, _ = build_runtime(config=MLCacheRuntimeConfig(), include_record=False)

        result = runtime.lookup_with_decision(lookup())

        self.assertEqual(result.response, Response("pair response"))
        self.assertEqual(result.decision.cache_key, CacheKey("pair"))
        self.assertNotIn("query_level_active", result.metadata)

    def test_shadow_mode_still_does_not_alter_cache_gateway_result(self) -> None:
        shadow_store = InMemoryQueryLevelShadowDecisionStore()
        runtime, _, _, _ = build_runtime(
            config=active_config(mode=QueryLevelPolicyMode.SHADOW),
            shadow_store=shadow_store,
        )

        result = runtime.lookup_with_decision(lookup())

        self.assertEqual(result.response, Response("pair response"))
        self.assertEqual(result.decision.cache_key, CacheKey("pair"))
        self.assertEqual(len(shadow_store.decisions()), 1)

    def test_active_mode_requires_threshold_when_configured(self) -> None:
        with self.assertRaisesRegex(ValueError, "requires a query-level threshold"):
            build_runtime(config=active_config(threshold=None))

    def test_active_mode_requires_query_record_store(self) -> None:
        config = active_config()

        with self.assertRaisesRegex(ValueError, "QueryCalibrationRecordStore"):
            build_mlcache_runtime(
                kv_store=InMemoryKVStore(),
                vector_store=InMemoryVectorStore(),
                feature_builder=NormalizedHadamardFeatureBuilder(),
                scorer=DeterministicScorer(),
                config=config,
            )

    def test_active_mode_accepts_selected_candidate_under_ge(self) -> None:
        runtime, _, _, _ = build_runtime(config=active_config(threshold=0.8))

        result = runtime.lookup_with_decision(lookup())

        self.assertEqual(result.response, Response("query-level response"))
        self.assertEqual(result.decision.status, OracleDecisionStatus.HIT)
        self.assertEqual(result.decision.cache_key, CacheKey("ql"))
        self.assertEqual(result.metadata["final_decision_source"], "query_level_active")

    def test_active_mode_rejects_selected_candidate_below_threshold(self) -> None:
        runtime, _, _, _ = build_runtime(
            config=active_config(threshold=0.8, fallback_on_abstain=False),
            record_candidates=(candidate("ql", 0.7),),
        )

        result = runtime.lookup_with_decision(lookup())

        self.assertIsNone(result.response)
        self.assertEqual(result.decision.status, OracleDecisionStatus.MISS)
        self.assertEqual(result.metadata["query_level_reason"], "selected_score_below_threshold")

    def test_gt_tie_mode_rejects_score_equal_to_threshold(self) -> None:
        query_level_policy = QueryLevelLearnedPolicy(
            threshold=Threshold(0.8),
            config=QueryLevelPolicyConfig(mode=QueryLevelPolicyMode.ACTIVE, tie_mode=TieMode.GT),
        )
        runtime, _, _, _ = build_runtime(
            config=active_config(threshold=0.8, fallback_on_abstain=False),
            record_candidates=(candidate("ql", 0.8),),
            query_level_policy=query_level_policy,
        )

        result = runtime.lookup_with_decision(lookup())

        self.assertIsNone(result.response)
        self.assertEqual(result.decision.status, OracleDecisionStatus.MISS)
        self.assertEqual(result.metadata["query_level_reason"], "selected_score_below_threshold")

    def test_active_accepted_candidate_fetches_response_from_kv_store(self) -> None:
        runtime, kv, _, _ = build_runtime(config=active_config(threshold=0.8))

        result = runtime.lookup_with_decision(lookup())

        self.assertEqual(kv.get(CacheKey("ql")), Response("query-level response"))
        self.assertEqual(result.response, Response("query-level response"))

    def test_active_selected_candidate_kv_miss_returns_no_response_when_fallback_disabled(self) -> None:
        runtime, _, vectors, _ = build_runtime(
            config=active_config(threshold=0.8, fallback_on_kv_miss=False),
            include_query_candidate_in_kv=False,
        )

        result = runtime.lookup_with_decision(lookup())

        self.assertIsNone(result.response)
        self.assertEqual(result.decision.status, OracleDecisionStatus.MISS)
        self.assertEqual(result.metadata["query_level_reason"], "query_level_kv_key_missing_or_expired")
        self.assertIn(CacheKey("ql"), vectors.removed)

    def test_active_selected_candidate_kv_miss_falls_back_when_configured(self) -> None:
        runtime, _, _, _ = build_runtime(
            config=active_config(threshold=0.8, fallback_on_kv_miss=True),
            include_query_candidate_in_kv=False,
        )

        result = runtime.lookup_with_decision(lookup())

        self.assertEqual(result.response, Response("pair response"))
        self.assertEqual(result.metadata["fallback_used"], True)
        self.assertEqual(result.metadata["final_decision_source"], "query_level_active_fallback_pair_level")

    def test_missing_query_record_falls_back_when_configured(self) -> None:
        runtime, _, _, _ = build_runtime(config=active_config(), include_record=False)

        result = runtime.lookup_with_decision(lookup())

        self.assertEqual(result.response, Response("pair response"))
        self.assertEqual(result.metadata["query_level_reason"], "query_level_record_missing")
        self.assertTrue(result.metadata["fallback_used"])

    def test_missing_query_record_returns_no_response_when_fallback_disabled(self) -> None:
        runtime, _, _, _ = build_runtime(
            config=active_config(fallback_on_missing_record=False),
            include_record=False,
        )

        result = runtime.lookup_with_decision(lookup())

        self.assertIsNone(result.response)
        self.assertEqual(result.decision.status, OracleDecisionStatus.ABSTAIN)
        self.assertEqual(result.metadata["final_decision_source"], "query_level_active_no_fallback")

    def test_query_level_reject_falls_back_when_configured(self) -> None:
        runtime, _, _, _ = build_runtime(
            config=active_config(fallback_on_abstain=True),
            record_candidates=(candidate("ql", 0.1),),
        )

        result = runtime.lookup_with_decision(lookup())

        self.assertEqual(result.response, Response("pair response"))
        self.assertTrue(result.metadata["fallback_used"])

    def test_query_level_abstain_returns_no_response_when_fallback_disabled(self) -> None:
        runtime, _, _, _ = build_runtime(
            config=active_config(fallback_on_abstain=False),
            record_candidates=(),
        )

        result = runtime.lookup_with_decision(lookup())

        self.assertIsNone(result.response)
        self.assertEqual(result.decision.status, OracleDecisionStatus.ABSTAIN)
        self.assertEqual(result.metadata["query_level_reason"], "no_candidates")

    def test_result_metadata_contains_pair_query_and_final_decision_fields(self) -> None:
        runtime, _, _, _ = build_runtime(config=active_config(threshold=0.8))

        result = runtime.lookup_with_decision(lookup())

        for key in (
            "pair_level_status",
            "pair_level_accepted",
            "query_level_status",
            "query_level_accepted",
            "query_level_selected_candidate_key",
            "query_level_selected_candidate_rank",
            "query_level_selected_score",
            "query_level_threshold",
            "query_level_reason",
            "fallback_used",
            "final_decision_source",
        ):
            self.assertIn(key, result.metadata)
        self.assertEqual(result.metadata["query_level_selected_candidate_key"], "ql")

    def test_observability_hooks_receive_query_level_active_metrics(self) -> None:
        metrics = RecordingMetricsSink()
        runtime, _, _, _ = build_runtime(config=active_config(threshold=0.8), metrics_sink=metrics)

        runtime.lookup_with_decision(lookup())

        names = [name for name, _, _ in metrics.records]
        self.assertIn("cache.query_level_active.evaluated", names)
        self.assertIn("cache.query_level_active.accepted", names)
        self.assertIn("cache.query_level_active.selected_rank", names)
        metadata = next(metadata for name, _, metadata in metrics.records if name == "cache.query_level_active.evaluated")
        self.assertEqual(metadata["final_decision_source"], "query_level_active")
        self.assertEqual(metadata["query_level_status"], "hit")

    def test_observability_failure_does_not_change_serving_result(self) -> None:
        runtime, _, _, _ = build_runtime(
            config=active_config(threshold=0.8),
            metrics_sink=RecordingMetricsSink(fail=True),
        )

        result = runtime.lookup_with_decision(lookup())

        self.assertEqual(result.response, Response("query-level response"))
        self.assertEqual(result.decision.cache_key, CacheKey("ql"))
        self.assertTrue(any(item["event"] == "observability_failure" for item in runtime.diagnostics))

    def test_old_and_new_imports_still_work(self) -> None:
        self.assertIs(OldQueryLevelLearnedPolicy, QueryLevelLearnedPolicy)
        self.assertIs(OldQueryLevelPolicyMode, QueryLevelPolicyMode)
        self.assertIs(OldQueryLevelRuntimeConfig, QueryLevelRuntimeConfig)


if __name__ == "__main__":
    unittest.main()
