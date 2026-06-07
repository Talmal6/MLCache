import unittest

from runtime import MLCacheRuntime as OldMLCacheRuntime

from mlcache.cache import KVStore, SemanticCacheGateway
from mlcache.calibration import ThresholdCalibrationRequest
from mlcache.features import NormalizedHadamardFeatureBuilder
from mlcache.feedback import (
    DefaultShadowTopKCollector,
    InMemorySplitJudgeTrainingStore,
    JudgeDecision,
    JudgeLabel,
    JudgeRequest,
    JudgeResult,
    SemanticReuseJudge,
)
from mlcache.observability import AuditEvent, AuditLogger, DiagnosticsReporter, MetricsSink
from mlcache.retrieval import VectorSearchResult, VectorStore
from mlcache.runtime import (
    MLCacheRuntime,
    MLCacheRuntimeConfig,
    ShadowRuntimeConfig,
    build_mlcache_runtime,
)
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
        return ScorerName("runtime_scorer")

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

    def calibrate(self, request: ThresholdCalibrationRequest) -> Threshold:
        del request
        return Threshold(0.5)

    def predict(self, features, threshold: Threshold, *, tie_mode: TieMode = TieMode.GE) -> bool:
        score = float(self.score(features))
        if tie_mode == TieMode.GT:
            return score > float(threshold)
        return score >= float(threshold)


class DeterministicJudge(SemanticReuseJudge):
    @property
    def name(self) -> str:
        return "runtime_judge"

    def judge(self, request: JudgeRequest) -> JudgeResult:
        return JudgeResult(request=request, decision=JudgeDecision(label=JudgeLabel.REUSABLE))


class RecordingAuditLogger(AuditLogger):
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.events: list[AuditEvent] = []

    def log(self, event: AuditEvent) -> None:
        if self.fail:
            raise RuntimeError("audit failed")
        self.events.append(event)


class RecordingMetricsSink(MetricsSink):
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.records: list[tuple[str, float, dict | None]] = []

    def record(self, name: str, value: float, metadata: dict | None = None) -> None:
        if self.fail:
            raise RuntimeError("metrics failed")
        self.records.append((name, value, metadata))


class RecordingDiagnosticsReporter(DiagnosticsReporter):
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.events: list[tuple[str, dict | None]] = []

    def report(self, event: str, metadata: dict | None = None) -> None:
        if self.fail:
            raise RuntimeError("diagnostics failed")
        self.events.append((event, metadata))


def entry(key: str = "cached", embedding: tuple[float, float] = (1.0, 0.0)) -> CacheEntry:
    return CacheEntry(
        cache_key=CacheKey(key),
        query=Query(f"query {key}"),
        response=Response(f"response {key}"),
        embedding=embedding,
        metadata=CacheMetadata(),
    )


def lookup(embedding: tuple[float, float] = (1.0, 0.0)) -> CacheLookup:
    return CacheLookup(query=Query("incoming"), embedding=embedding)


def runtime(
    *,
    config: MLCacheRuntimeConfig | None = None,
    judge: SemanticReuseJudge | None = None,
    store=None,
    audit_logger: AuditLogger | None = None,
    metrics_sink: MetricsSink | None = None,
    diagnostics_reporter: DiagnosticsReporter | None = None,
):
    kv = InMemoryKVStore()
    vectors = InMemoryVectorStore()
    instance = build_mlcache_runtime(
        kv_store=kv,
        vector_store=vectors,
        feature_builder=NormalizedHadamardFeatureBuilder(),
        scorer=DeterministicScorer(),
        judge=judge,
        judge_training_store=store,
        audit_logger=audit_logger,
        metrics_sink=metrics_sink,
        diagnostics_reporter=diagnostics_reporter,
        config=config,
    )
    return instance, kv, vectors


class RuntimeOrchestratorTests(unittest.TestCase):
    def test_build_mlcache_runtime_creates_gateway_and_oracle(self) -> None:
        instance, _, _ = runtime()

        self.assertIsInstance(instance, MLCacheRuntime)
        self.assertIsInstance(instance.gateway, SemanticCacheGateway)
        self.assertEqual(instance.components["gateway"], "SemanticCacheGateway")
        self.assertEqual(instance.components["feature_builder"], "NormalizedHadamardFeatureBuilder")

    def test_runtime_put_stores_response_and_indexes_vector_entry(self) -> None:
        instance, kv, vectors = runtime()
        cached = entry()

        key = instance.put(cached)

        self.assertEqual(key, cached.cache_key)
        self.assertEqual(kv.get(cached.cache_key), cached.response)
        self.assertIsNotNone(vectors.get(cached.cache_key))

    def test_lookup_with_decision_delegates_through_gateway(self) -> None:
        instance, _, _ = runtime()
        cached = entry()
        instance.put(cached)
        instance.oracle._threshold = Threshold(0.5)

        direct = instance.gateway.lookup_with_decision(lookup())
        delegated = instance.lookup_with_decision(lookup())

        self.assertEqual(delegated.response, direct.response)
        self.assertEqual(delegated.decision.status, direct.decision.status)
        self.assertEqual(delegated.decision.cache_key, direct.decision.cache_key)

    def test_shadow_disabled_factory_does_not_create_or_enable_shadow_collection(self) -> None:
        instance, _, _ = runtime(config=MLCacheRuntimeConfig(shadow=ShadowRuntimeConfig(enabled=False)))

        self.assertIsNone(instance.shadow_collector)
        self.assertFalse(instance.oracle.shadow_collection_enabled)

    def test_shadow_enabled_missing_dependencies_raises_value_error(self) -> None:
        config = MLCacheRuntimeConfig(shadow=ShadowRuntimeConfig(enabled=True))

        with self.assertRaisesRegex(ValueError, "SemanticReuseJudge"):
            runtime(config=config)

        with self.assertRaisesRegex(ValueError, "SplitJudgeTrainingStore"):
            runtime(config=config, judge=DeterministicJudge())

    def test_shadow_enabled_with_valid_dependencies_creates_default_collector(self) -> None:
        store = InMemorySplitJudgeTrainingStore(
            max_h0_train=10,
            max_h1_train=10,
            max_h0_calibration=10,
            max_h1_calibration=10,
        )
        config = MLCacheRuntimeConfig(shadow=ShadowRuntimeConfig(enabled=True, top_k=3, calibration_every_n=2))

        instance, _, _ = runtime(config=config, judge=DeterministicJudge(), store=store)

        self.assertIsInstance(instance.shadow_collector, DefaultShadowTopKCollector)
        self.assertTrue(instance.oracle.shadow_collection_enabled)
        self.assertEqual(instance.shadow_collector.config.top_k, 3)
        self.assertEqual(instance.shadow_collector.config.calibration_every_n, 2)

    def test_shadow_snapshot_is_none_without_shadow_collector(self) -> None:
        instance, _, _ = runtime()

        self.assertIsNone(instance.shadow_snapshot)

    def test_shadow_snapshot_delegates_to_collector_snapshot(self) -> None:
        store = InMemorySplitJudgeTrainingStore(
            max_h0_train=10,
            max_h1_train=10,
            max_h0_calibration=10,
            max_h1_calibration=10,
        )
        config = MLCacheRuntimeConfig(shadow=ShadowRuntimeConfig(enabled=True))
        instance, _, _ = runtime(config=config, judge=DeterministicJudge(), store=store)

        snapshot = instance.shadow_snapshot

        self.assertIsNotNone(snapshot)
        self.assertEqual(snapshot.pairs_observed, 0)

    def test_activation_status_and_threshold_delegate_to_oracle(self) -> None:
        instance, _, _ = runtime()
        instance.oracle._threshold = Threshold(0.7)

        self.assertEqual(instance.threshold, Threshold(0.7))
        self.assertEqual(instance.activation_status["threshold"], Threshold(0.7))

    def test_observability_hooks_are_called_after_lookup(self) -> None:
        audit = RecordingAuditLogger()
        metrics = RecordingMetricsSink()
        diagnostics = RecordingDiagnosticsReporter()
        instance, _, _ = runtime(audit_logger=audit, metrics_sink=metrics, diagnostics_reporter=diagnostics)

        result = instance.lookup_with_decision(lookup())

        self.assertEqual(result.decision.status, OracleDecisionStatus.MISS)
        self.assertEqual(len(audit.events), 1)
        self.assertEqual([name for name, _, _ in metrics.records], [
            "cache.lookup.accepted",
            "cache.lookup.hit",
            "cache.lookup.miss",
            "cache.lookup.abstain",
            "cache.lookup.candidate_count",
        ])
        self.assertEqual(diagnostics.events[0][0], "cache.lookup")
        self.assertEqual(metrics.records[0][2]["decision_status"], "miss")

    def test_observability_failures_do_not_change_lookup_result(self) -> None:
        instance, _, _ = runtime(
            audit_logger=RecordingAuditLogger(fail=True),
            metrics_sink=RecordingMetricsSink(fail=True),
            diagnostics_reporter=RecordingDiagnosticsReporter(fail=True),
        )

        result = instance.lookup_with_decision(lookup())

        self.assertEqual(result.decision.status, OracleDecisionStatus.MISS)
        self.assertIsNone(result.response)
        self.assertEqual([item["hook"] for item in instance.diagnostics], [
            "audit_logger",
            "metrics_sink",
            "diagnostics_reporter",
        ])

    def test_old_imports_work(self) -> None:
        self.assertIs(OldMLCacheRuntime, MLCacheRuntime)

    def test_new_imports_work(self) -> None:
        self.assertEqual(MLCacheRuntime.__name__, "MLCacheRuntime")
        self.assertEqual(MLCacheRuntimeConfig.__name__, "MLCacheRuntimeConfig")
        self.assertTrue(callable(build_mlcache_runtime))


if __name__ == "__main__":
    unittest.main()
