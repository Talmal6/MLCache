import unittest

from cache import SemanticCacheGateway
from features import NormalizedHadamardFeatureBuilder
from oracle import TrainableSemanticCacheOracle

from mlcache.cache import KVStore, SemanticCacheGateway as NewSemanticCacheGateway
from mlcache.calibration import ThresholdCalibrationRequest
from mlcache.features import NormalizedHadamardFeatureBuilder as NewNormalizedHadamardFeatureBuilder
from mlcache.feedback import (
    DefaultShadowTopKCollector,
    InMemorySplitJudgeTrainingStore,
    JudgeDecision,
    JudgeLabel,
    JudgeRequest,
    JudgeResult,
    JudgedPairExample,
    SemanticReuseJudge,
    ShadowCollectionConfig,
)
from mlcache.oracle import TrainableSemanticCacheOracle as NewTrainableSemanticCacheOracle
from mlcache.policies.refit import (
    ConservativeRefitConfig,
    RefitAction,
    RefitPolicy,
    RefitPolicyContext,
    RefitPolicyDecision,
)
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
        self._responses: dict[CacheKey, Response] = {}

    def get(self, cache_key: CacheKey) -> Response | None:
        return self._responses.get(cache_key)

    def set(self, cache_key: CacheKey, response: Response) -> None:
        self._responses[cache_key] = response

    def delete(self, cache_key: CacheKey) -> None:
        self._responses.pop(cache_key, None)

    def contains(self, cache_key: CacheKey) -> bool:
        return cache_key in self._responses


class InMemoryVectorStore(VectorStore):
    def __init__(self) -> None:
        self._entries: dict[CacheKey, CacheEntry] = {}
        self.removed: list[CacheKey] = []

    def upsert(self, entry: CacheEntry) -> None:
        self._entries[entry.cache_key] = entry

    def delete(self, cache_key: CacheKey) -> None:
        self.removed.append(cache_key)
        self._entries.pop(cache_key, None)

    def get(self, cache_key: CacheKey) -> VectorSearchResult | None:
        entry = self._entries.get(cache_key)
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
        for entry in self._entries.values():
            if namespace is not None and entry.metadata.namespace != namespace:
                continue
            scored.append((self._cosine(embedding, entry.embedding), entry))
        scored.sort(key=lambda item: (-item[0], str(item[1].cache_key)))
        return [self._result(entry, Score(float(score))) for score, entry in scored[:top_k]]

    @staticmethod
    def _result(entry: CacheEntry, score: Score) -> VectorSearchResult:
        return VectorSearchResult(
            cache_key=entry.cache_key,
            embedding=entry.embedding,
            score=score,
            query=entry.query,
            metadata=entry.metadata,
            payload={"response": entry.response},
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
    def __init__(self, *, threshold: float = 0.5) -> None:
        self.threshold = float(threshold)
        self.fit_calls: list[LabeledPairBatch] = []
        self.calibration_calls: list[ThresholdCalibrationRequest] = []

    @property
    def name(self) -> ScorerName:
        return ScorerName("deterministic")

    @property
    def input_space(self) -> InputSpace:
        return InputSpace.EMBEDDING

    def fit(self, batch: LabeledPairBatch, **kwargs: object) -> None:
        del kwargs
        self.fit_calls.append(batch)

    def copy_for_refit(self) -> "DeterministicScorer":
        return type(self)(threshold=self.threshold)

    def score(self, features) -> Score:
        if features.cosine is not None:
            return Score(float(features.cosine))
        if features.hadamard:
            return Score(float(sum(features.hadamard)))
        return Score(0.0)

    def calibrate(self, request: ThresholdCalibrationRequest) -> Threshold:
        self.calibration_calls.append(request)
        return Threshold(self.threshold)

    def predict(self, features, threshold: Threshold, *, tie_mode: TieMode = TieMode.GE) -> bool:
        score = float(self.score(features))
        if tie_mode == TieMode.GT:
            return score > float(threshold)
        return score >= float(threshold)


class RecordingScorer(DeterministicScorer):
    def __init__(self, *, recorder: dict[str, object] | None = None, threshold: float = 0.5) -> None:
        super().__init__(threshold=threshold)
        self.recorder = recorder if recorder is not None else {}
        self._fitted = False

    def copy_for_refit(self) -> "RecordingScorer":
        return RecordingScorer(recorder=self.recorder, threshold=self.threshold)

    def fit(self, batch: LabeledPairBatch, **kwargs: object) -> None:
        super().fit(batch, **kwargs)
        self._fitted = True
        self.recorder["fit_h0"] = tuple(tuple(float(value) for value in row) for row in batch.h0)
        self.recorder["fit_h1"] = tuple(tuple(float(value) for value in row) for row in batch.h1)

    def score(self, features) -> Score:
        if self._fitted:
            rows = self.recorder.setdefault("calibration_score_rows", [])
            rows.append(self._features_to_tuple(features))
        return super().score(features)

    def calibrate(self, request: ThresholdCalibrationRequest) -> Threshold:
        self.recorder["calibration_scores"] = tuple(float(score) for score in request.h0_scores)
        self.recorder["calibration_context"] = dict(request.context)
        return super().calibrate(request)

    @staticmethod
    def _features_to_tuple(features) -> tuple[float, ...]:
        if features.hadamard:
            return tuple(float(value) for value in features.hadamard)
        if features.concat:
            return tuple(float(value) for value in features.concat)
        if features.abs_diff:
            return tuple(float(value) for value in features.abs_diff)
        if features.cosine is not None:
            return (float(features.cosine),)
        return ()


class DeterministicJudge(SemanticReuseJudge):
    def __init__(self, labels: dict[str, JudgeLabel], *, default: JudgeLabel = JudgeLabel.UNCERTAIN) -> None:
        self.labels = labels
        self.default = default
        self.requests: list[JudgeRequest] = []

    @property
    def name(self) -> str:
        return "deterministic-judge"

    def judge(self, request: JudgeRequest) -> JudgeResult:
        self.requests.append(request)
        candidate_key = str(request.candidate_key) if request.candidate_key is not None else ""
        label = self.labels.get(candidate_key, self.labels.get(str(request.candidate_query), self.default))
        return JudgeResult(request=request, decision=JudgeDecision(label=label))


class AlwaysRefitPolicy(RefitPolicy):
    def __init__(self, config: ConservativeRefitConfig | None = None) -> None:
        self.config = config or relaxed_activation_config()

    def decide(self, context: RefitPolicyContext) -> RefitPolicyDecision:
        del context
        return RefitPolicyDecision(action=RefitAction.REFIT_SCORER, reason="test_forced_refit")


def relaxed_activation_config() -> ConservativeRefitConfig:
    return ConservativeRefitConfig(
        min_train_total=2,
        min_train_h0=1,
        min_train_h1=1,
        min_calibration_h0=1,
        min_calibration_h1=0,
        fpr_wilson_margin=0.8,
        min_decisions_between_refits=0,
        min_new_h0_for_refit=0,
        min_new_h1_for_refit=0,
        min_new_fraction_for_refit=0.0,
    )


def split_store() -> InMemorySplitJudgeTrainingStore:
    return InMemorySplitJudgeTrainingStore(
        max_h0_train=20,
        max_h1_train=20,
        max_h0_calibration=20,
        max_h1_calibration=20,
    )


def entry(key: str, embedding: Embedding, *, query: str | None = None, response: str | None = None) -> CacheEntry:
    return CacheEntry(
        cache_key=CacheKey(key),
        query=Query(query or f"cached {key}"),
        response=Response(response or f"response {key}"),
        embedding=embedding,
        metadata=CacheMetadata(),
    )


def lookup(embedding: Embedding = (1.0, 0.0), *, query: str = "incoming") -> CacheLookup:
    return CacheLookup(query=Query(query), embedding=embedding)


def oracle(
    vector_store: InMemoryVectorStore,
    *,
    scorer: SemanticScorer | None = None,
    top_k: int = 1,
    threshold: float | None = 0.5,
    judge_training_store=None,
    shadow_collector=None,
    shadow_collection_enabled: bool = False,
    auto_refit: bool = False,
    refit_policy: RefitPolicy | None = None,
) -> TrainableSemanticCacheOracle:
    instance = TrainableSemanticCacheOracle(
        vector_store=vector_store,
        feature_builder=NormalizedHadamardFeatureBuilder(),
        scorer=scorer or DeterministicScorer(),
        judge_training_store=judge_training_store,
        shadow_collector=shadow_collector,
        shadow_collection_enabled=shadow_collection_enabled,
        auto_refit=auto_refit,
        refit_policy=refit_policy,
        top_k=top_k,
    )
    instance._threshold = None if threshold is None else Threshold(float(threshold))
    return instance


class E2ECachePipelineTests(unittest.TestCase):
    def test_old_and_new_imports_work_in_e2e_module(self) -> None:
        self.assertIs(SemanticCacheGateway, NewSemanticCacheGateway)
        self.assertIs(TrainableSemanticCacheOracle, NewTrainableSemanticCacheOracle)
        self.assertIs(NormalizedHadamardFeatureBuilder, NewNormalizedHadamardFeatureBuilder)
        self.assertEqual(DefaultShadowTopKCollector.__name__, "DefaultShadowTopKCollector")
        self.assertEqual(InMemorySplitJudgeTrainingStore.__name__, "InMemorySplitJudgeTrainingStore")

    def test_empty_cache_lookup_returns_miss_no_response(self) -> None:
        kv = InMemoryKVStore()
        vectors = InMemoryVectorStore()
        gateway = SemanticCacheGateway(kv_store=kv, oracle=oracle(vectors))

        result = gateway.lookup_with_decision(lookup())

        self.assertIsNone(result.response)
        self.assertFalse(result.decision.accepted)
        self.assertEqual(result.decision.status, OracleDecisionStatus.MISS)
        self.assertEqual(result.decision.reason, "no_neighbors")

    def test_put_then_lookup_similar_query_returns_hit_and_cached_response(self) -> None:
        kv = InMemoryKVStore()
        vectors = InMemoryVectorStore()
        gateway = SemanticCacheGateway(kv_store=kv, oracle=oracle(vectors, threshold=0.8))
        cached = entry("a", (1.0, 0.0), response="cached response")
        gateway.put(cached)

        result = gateway.lookup_with_decision(lookup((0.99, 0.05)))

        self.assertEqual(result.decision.status, OracleDecisionStatus.HIT)
        self.assertTrue(result.decision.accepted)
        self.assertEqual(result.response, Response("cached response"))
        self.assertEqual(result.decision.cache_key, cached.cache_key)

    def test_kv_missing_invalidates_semantic_index(self) -> None:
        kv = InMemoryKVStore()
        vectors = InMemoryVectorStore()
        cache_oracle = oracle(vectors, threshold=0.5)
        gateway = SemanticCacheGateway(kv_store=kv, oracle=cache_oracle)
        cached = entry("missing", (1.0, 0.0))
        vectors.upsert(cached)

        result = gateway.lookup_with_decision(lookup())

        self.assertIsNone(result.response)
        self.assertEqual(result.metadata["reason"], "kv_key_missing_or_expired")
        self.assertEqual(vectors.removed, [cached.cache_key])
        self.assertEqual(vectors.search((1.0, 0.0)), [])

    def test_unfitted_oracle_abstains(self) -> None:
        kv = InMemoryKVStore()
        vectors = InMemoryVectorStore()
        gateway = SemanticCacheGateway(kv_store=kv, oracle=oracle(vectors, threshold=None))
        cached = entry("a", (1.0, 0.0), response="cached response")
        gateway.put(cached)

        result = gateway.lookup_with_decision(lookup())

        self.assertIsNone(result.response)
        self.assertEqual(result.decision.status, OracleDecisionStatus.ABSTAIN)
        self.assertEqual(result.decision.reason, "oracle_not_fitted")

    def test_shadow_collection_is_disabled_by_default(self) -> None:
        kv = InMemoryKVStore()
        vectors = InMemoryVectorStore()
        store = split_store()
        judge = DeterministicJudge({"a": JudgeLabel.NOT_REUSABLE})
        shadow = DefaultShadowTopKCollector(
            feature_builder=NormalizedHadamardFeatureBuilder(),
            judge=judge,
            store=store,
        )
        gateway = SemanticCacheGateway(
            kv_store=kv,
            oracle=oracle(
                vectors,
                threshold=0.5,
                shadow_collector=shadow,
                shadow_collection_enabled=False,
            ),
        )
        gateway.put(entry("a", (1.0, 0.0), response="cached response"))

        result = gateway.lookup_with_decision(lookup())
        snapshot = shadow.snapshot()

        self.assertEqual(result.decision.status, OracleDecisionStatus.HIT)
        self.assertEqual(result.response, Response("cached response"))
        self.assertEqual(store.h0(), ())
        self.assertEqual(store.h1(), ())
        self.assertEqual(snapshot.pairs_observed, 0)
        self.assertEqual(snapshot.judge_calls, 0)

    def test_shadow_collection_enabled_judges_top_k_candidates(self) -> None:
        kv = InMemoryKVStore()
        vectors = InMemoryVectorStore()
        store = split_store()
        labels = {
            "c1": JudgeLabel.REUSABLE,
            "c2": JudgeLabel.NOT_REUSABLE,
            "c3": JudgeLabel.REUSABLE,
            "c4": JudgeLabel.NOT_REUSABLE,
            "c5": JudgeLabel.NOT_REUSABLE,
            "c6": JudgeLabel.REUSABLE,
        }
        judge = DeterministicJudge(labels)
        shadow = DefaultShadowTopKCollector(
            feature_builder=NormalizedHadamardFeatureBuilder(),
            judge=judge,
            store=store,
            config=ShadowCollectionConfig(top_k=5, calibration_every_n=2),
        )
        gateway = SemanticCacheGateway(
            kv_store=kv,
            oracle=oracle(
                vectors,
                threshold=0.5,
                top_k=5,
                shadow_collector=shadow,
                shadow_collection_enabled=True,
            ),
        )
        for idx, x in enumerate((1.0, 0.9, 0.8, 0.7, 0.6, 0.1), start=1):
            gateway.put(entry(f"c{idx}", (x, 1.0 - x), response=f"response c{idx}"))

        result = gateway.lookup_with_decision(lookup())
        snapshot = shadow.snapshot()

        self.assertEqual(result.decision.status, OracleDecisionStatus.HIT)
        self.assertEqual([str(req.candidate_key) for req in judge.requests], ["c1", "c2", "c3", "c4", "c5"])
        self.assertEqual(snapshot.pairs_observed, 5)
        self.assertEqual(snapshot.judge_calls, 5)
        self.assertEqual(len(store.h0_train()), 2)
        self.assertEqual(len(store.h0_calibration()), 1)
        self.assertEqual(len(store.h1_train()), 1)
        self.assertEqual(len(store.h1_calibration()), 1)

    def test_shadow_labels_do_not_affect_request_t_decision(self) -> None:
        kv = InMemoryKVStore()
        vectors = InMemoryVectorStore()
        store = split_store()
        judge = DeterministicJudge({"c1": JudgeLabel.REUSABLE, "c2": JudgeLabel.REUSABLE})
        shadow = DefaultShadowTopKCollector(
            feature_builder=NormalizedHadamardFeatureBuilder(),
            judge=judge,
            store=store,
            config=ShadowCollectionConfig(top_k=2),
        )
        gateway = SemanticCacheGateway(
            kv_store=kv,
            oracle=oracle(
                vectors,
                threshold=None,
                top_k=2,
                shadow_collector=shadow,
                shadow_collection_enabled=True,
                auto_refit=True,
                refit_policy=AlwaysRefitPolicy(),
            ),
        )
        gateway.put(entry("c1", (1.0, 0.0)))
        gateway.put(entry("c2", (0.9, 0.1)))

        result = gateway.lookup_with_decision(lookup())

        self.assertIsNone(result.response)
        self.assertEqual(result.decision.status, OracleDecisionStatus.ABSTAIN)
        self.assertEqual(result.decision.reason, "oracle_not_fitted")
        self.assertEqual(len(store.h1_train()), 2)

    def test_future_request_can_observe_previous_shadow_refit(self) -> None:
        kv = InMemoryKVStore()
        vectors = InMemoryVectorStore()
        store = split_store()
        judge = DeterministicJudge(
            {
                "c1": JudgeLabel.REUSABLE,
                "c2": JudgeLabel.NOT_REUSABLE,
                "c3": JudgeLabel.NOT_REUSABLE,
            }
        )
        shadow = DefaultShadowTopKCollector(
            feature_builder=NormalizedHadamardFeatureBuilder(),
            judge=judge,
            store=store,
            config=ShadowCollectionConfig(top_k=3, calibration_every_n=2),
        )
        cache_oracle = oracle(
            vectors,
            scorer=RecordingScorer(threshold=0.5),
            threshold=None,
            top_k=3,
            judge_training_store=store,
            shadow_collector=shadow,
            shadow_collection_enabled=True,
            auto_refit=True,
            refit_policy=AlwaysRefitPolicy(),
        )
        gateway = SemanticCacheGateway(kv_store=kv, oracle=cache_oracle)
        gateway.put(entry("c1", (1.0, 0.0), response="future hit"))
        gateway.put(entry("c2", (0.0, 1.0)))
        gateway.put(entry("c3", (0.0, 1.0)))

        first = gateway.lookup_with_decision(lookup())
        cache_oracle.wait_for_fit(timeout=2.0)
        second = gateway.lookup_with_decision(lookup())

        self.assertEqual(first.decision.status, OracleDecisionStatus.ABSTAIN)
        self.assertIsNone(first.response)
        self.assertGreater(len(store.h0()), 0)
        self.assertGreater(len(store.h1()), 0)
        self.assertEqual(second.decision.status, OracleDecisionStatus.HIT)
        self.assertEqual(second.response, Response("future hit"))

    def test_split_aware_refit_uses_explicit_train_and_calibration_buckets(self) -> None:
        kv = InMemoryKVStore()
        vectors = InMemoryVectorStore()
        store = split_store()
        store.add_train(self._judged(JudgeLabel.NOT_REUSABLE, "h0-train", (0.10,)))
        store.add_train(self._judged(JudgeLabel.REUSABLE, "h1-train", (0.90,)))
        store.add_calibration(self._judged(JudgeLabel.NOT_REUSABLE, "h0-calib-1", (0.20,)))
        store.add_calibration(self._judged(JudgeLabel.NOT_REUSABLE, "h0-calib-2", (0.30,)))
        store.add_calibration(self._judged(JudgeLabel.REUSABLE, "h1-calib", (0.80,)))
        recorder: dict[str, object] = {}
        cache_oracle = oracle(
            vectors,
            scorer=RecordingScorer(recorder=recorder, threshold=0.5),
            threshold=None,
            judge_training_store=store,
            auto_refit=True,
            refit_policy=AlwaysRefitPolicy(),
        )
        gateway = SemanticCacheGateway(kv_store=kv, oracle=cache_oracle)
        gateway.put(entry("served", (1.0, 0.0)))

        gateway.lookup_with_decision(lookup())
        cache_oracle.wait_for_fit(timeout=2.0)

        self.assertEqual(recorder["fit_h0"], ((0.10,),))
        self.assertEqual(recorder["fit_h1"], ((0.90,),))
        self.assertEqual(recorder["calibration_score_rows"], [(0.20,), (0.30,)])
        self.assertEqual(recorder["calibration_context"]["split_source"], "split_judge_training_store")

    @staticmethod
    def _judged(label: JudgeLabel, key: str, features: tuple[float, ...]) -> JudgedPairExample:
        return JudgedPairExample(
            features=features,
            request=JudgeRequest(query=Query("training"), candidate_key=CacheKey(key)),
            decision=JudgeDecision(label=label),
        )


if __name__ == "__main__":
    unittest.main()
