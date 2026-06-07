import unittest

from mlcache.cache import SemanticCacheGateway
from mlcache.calibration import ThresholdCalibrationRequest, ThresholdProvider, ThresholdScope
from mlcache.features import NormalizedHadamardFeatureBuilder
from mlcache.feedback import (
    DefaultShadowTopKCollector,
    InMemoryJudgeTrainingStore,
    InMemorySplitJudgeTrainingStore,
    JudgeDecision,
    JudgeLabel,
    JudgeRequest,
    JudgeResult,
    JudgedPairExample,
    SemanticReuseJudge,
    ShadowCollectionConfig,
)
from mlcache.policies.refit import ConservativeRefitConfig, ConservativeRefitPolicy, RefitAction
from mlcache.policies.refit import RefitPolicy, RefitPolicyContext, RefitPolicyDecision
from mlcache.oracle import TrainableSemanticCacheOracle
from mlcache.retrieval import VectorSearchResult, VectorStore
from mlcache.scorers import SemanticScorer
from mlcache.semantic_types import (
    CacheEntry,
    CacheKey,
    CacheLookup,
    CacheMetadata,
    InputSpace,
    LabeledPairBatch,
    OracleDecisionStatus,
    Query,
    Response,
    Score,
    ScorerName,
    Threshold,
    TieMode,
    TrainCalibEvalSplit,
)


class GateScorer(SemanticScorer):
    def __init__(self, *, threshold: float = 0.5, name: str = "gate") -> None:
        self.threshold = float(threshold)
        self._name = ScorerName(name)
        self.fit_batches: list[LabeledPairBatch] = []
        self.calibration_requests: list[ThresholdCalibrationRequest] = []

    @property
    def name(self) -> ScorerName:
        return self._name

    @property
    def input_space(self) -> InputSpace:
        return InputSpace.EMBEDDING

    def fit(self, batch: LabeledPairBatch, **kwargs: object) -> None:
        del kwargs
        self.fit_batches.append(batch)

    def copy_for_refit(self) -> "GateScorer":
        return type(self)(threshold=self.threshold, name=f"{self._name}-candidate")

    def score(self, features) -> Score:
        if features.hadamard:
            return Score(float(sum(features.hadamard)))
        if features.cosine is not None:
            return Score(float(features.cosine))
        return Score(0.0)

    def calibrate(self, request: ThresholdCalibrationRequest) -> Threshold:
        self.calibration_requests.append(request)
        return Threshold(self.threshold)

    def predict(self, features, threshold: Threshold, *, tie_mode: TieMode = TieMode.GE) -> bool:
        score = float(self.score(features))
        return score > float(threshold) if tie_mode == TieMode.GT else score >= float(threshold)


class RecordingGateScorer(GateScorer):
    def __init__(self, *, recorder: dict[str, object], threshold: float = 0.5, name: str = "recording") -> None:
        super().__init__(threshold=threshold, name=name)
        self.recorder = recorder
        self._fitted = False

    def copy_for_refit(self) -> "RecordingGateScorer":
        return RecordingGateScorer(
            recorder=self.recorder,
            threshold=self.threshold,
            name=f"{self.name}-candidate",
        )

    def fit(self, batch: LabeledPairBatch, **kwargs: object) -> None:
        super().fit(batch, **kwargs)
        self._fitted = True
        self.recorder["fit_h0"] = tuple(tuple(row) for row in batch.h0)
        self.recorder["fit_h1"] = tuple(tuple(row) for row in batch.h1)

    def score(self, features) -> Score:
        if self._fitted:
            rows = self.recorder.setdefault("calibration_rows", [])
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
        if features.cosine is not None:
            return (float(features.cosine),)
        return ()


class AlwaysRefitPolicy(RefitPolicy):
    def __init__(self, config: ConservativeRefitConfig | None = None) -> None:
        self.config = config or passing_config()

    def decide(self, context: RefitPolicyContext) -> RefitPolicyDecision:
        del context
        return RefitPolicyDecision(action=RefitAction.REFIT_SCORER, reason="test_forced_refit")


class AlwaysRecalibratePolicy(RefitPolicy):
    def __init__(self, config: ConservativeRefitConfig | None = None) -> None:
        self.config = config or passing_config()

    def decide(self, context: RefitPolicyContext) -> RefitPolicyDecision:
        del context
        return RefitPolicyDecision(action=RefitAction.RECALIBRATE_THRESHOLD, reason="test_forced_recalibration")


class StaticVectorStore(VectorStore):
    def __init__(self, entries: list[CacheEntry] | None = None) -> None:
        self.entries = entries or []

    def upsert(self, entry: CacheEntry) -> None:
        self.entries.append(entry)

    def delete(self, cache_key: CacheKey) -> None:
        self.entries = [entry for entry in self.entries if entry.cache_key != cache_key]

    def get(self, cache_key: CacheKey) -> VectorSearchResult | None:
        for entry in self.entries:
            if entry.cache_key == cache_key:
                return self._result(entry, Score(1.0))
        return None

    def search(self, embedding, *, namespace: str | None = None, top_k: int = 10) -> list[VectorSearchResult]:
        del embedding, namespace
        return [self._result(entry, Score(1.0)) for entry in self.entries[:top_k]]

    @staticmethod
    def _result(entry: CacheEntry, score: Score) -> VectorSearchResult:
        return VectorSearchResult(
            cache_key=entry.cache_key,
            embedding=entry.embedding,
            score=score,
            query=entry.query,
            metadata=entry.metadata,
        )


class DictKVStore:
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


class MappingJudge(SemanticReuseJudge):
    def __init__(self, labels: dict[str, JudgeLabel]) -> None:
        self.labels = labels

    @property
    def name(self) -> str:
        return "mapping"

    def judge(self, request: JudgeRequest) -> JudgeResult:
        key = str(request.candidate_key)
        return JudgeResult(request=request, decision=JudgeDecision(label=self.labels[key]))


class RecordingThresholdProvider(ThresholdProvider):
    def __init__(self) -> None:
        self.calls: list[tuple[Threshold, ScorerName, ThresholdScope, dict[str, object] | None]] = []
        self.current = Threshold(0.0)

    def get_threshold(
        self,
        *,
        scorer: ScorerName,
        scope: ThresholdScope = ThresholdScope.GLOBAL,
        region_id=None,
        cluster_id=None,
        context: dict[str, object] | None = None,
    ) -> Threshold:
        del scorer, scope, region_id, cluster_id, context
        return self.current

    def set_threshold(
        self,
        threshold: Threshold,
        *,
        scorer: ScorerName,
        scope: ThresholdScope = ThresholdScope.GLOBAL,
        region_id=None,
        cluster_id=None,
        context: dict[str, object] | None = None,
    ) -> None:
        del region_id, cluster_id
        self.current = threshold
        self.calls.append((threshold, scorer, scope, context))


def passing_config(**overrides: object) -> ConservativeRefitConfig:
    values = {
        "min_train_total": 1200,
        "min_train_h0": 100,
        "min_train_h1": 50,
        "min_calibration_h0": 500,
        "min_calibration_h1": 50,
        "fpr_wilson_margin": 0.03,
        "wilson_confidence_z": 1.96,
        "min_decisions_between_refits": 0,
        "min_decisions_between_calibrations": 0,
        "min_new_h0_for_refit": 0,
        "min_new_h1_for_refit": 0,
        "min_new_h0_for_calibration": 0,
    }
    values.update(overrides)
    return ConservativeRefitConfig(**values)


def row(value: float) -> tuple[float, ...]:
    return (float(value),)


def rows(count: int, value: float) -> tuple[tuple[float, ...], ...]:
    return tuple(row(value) for _ in range(count))


def h0_calibration_scores(*, total: int = 500, accepts: int = 25, threshold: float = 0.5) -> tuple[tuple[float, ...], ...]:
    return rows(accepts, threshold) + rows(total - accepts, 0.0)


def valid_split(
    *,
    h0_train: int = 1150,
    h1_train: int = 50,
    h0_calib: int = 500,
    h1_calib: int = 50,
    h0_accepts: int = 25,
) -> TrainCalibEvalSplit:
    return TrainCalibEvalSplit(
        h0_train=rows(h0_train, 0.0),
        h1_train=rows(h1_train, 1.0),
        h0_calib=h0_calibration_scores(total=h0_calib, accepts=h0_accepts),
        h1_calib=rows(h1_calib, 1.0),
        h0_eval=(),
        h1_eval=(),
        metadata={"test": "activation"},
    )


def oracle(
    *,
    scorer: GateScorer | None = None,
    config: ConservativeRefitConfig | None = None,
    threshold: Threshold | None = None,
    vector_store: VectorStore | None = None,
    refit_policy: RefitPolicy | None = None,
    auto_refit: bool = False,
    judge_training_store=None,
    shadow_collector=None,
    shadow_collection_enabled: bool = False,
    threshold_provider: ThresholdProvider | None = None,
    top_k: int = 1,
) -> TrainableSemanticCacheOracle:
    instance = TrainableSemanticCacheOracle(
        vector_store=vector_store or StaticVectorStore(),
        feature_builder=NormalizedHadamardFeatureBuilder(),
        scorer=scorer or GateScorer(),
        refit_policy=refit_policy or ConservativeRefitPolicy(config or passing_config()),
        auto_refit=auto_refit,
        judge_training_store=judge_training_store,
        shadow_collector=shadow_collector,
        shadow_collection_enabled=shadow_collection_enabled,
        threshold_provider=threshold_provider,
        top_k=top_k,
    )
    instance._threshold = threshold
    return instance


def entry(key: str, embedding: tuple[float, float] = (1.0, 0.0)) -> CacheEntry:
    return CacheEntry(
        cache_key=CacheKey(key),
        query=Query(f"query {key}"),
        response=Response(f"response {key}"),
        embedding=embedding,
        metadata=CacheMetadata(),
    )


def lookup() -> CacheLookup:
    return CacheLookup(query=Query("incoming"), embedding=(1.0, 0.0))


def judged(label: JudgeLabel, key: str, features: tuple[float, ...]) -> JudgedPairExample:
    return JudgedPairExample(
        features=features,
        request=JudgeRequest(query=Query("training"), candidate_key=CacheKey(key)),
        decision=JudgeDecision(label=label),
    )


class ActivationGateTests(unittest.TestCase):
    def test_gate_fails_when_train_total_below_minimum(self) -> None:
        cache_oracle = oracle()
        cache_oracle.fit(valid_split(h0_train=100, h1_train=50))

        status = cache_oracle.activation_status
        self.assertFalse(status["activation_gate_passed"])
        self.assertEqual(status["activation_gate_reason"], "min_train_total_not_met")
        self.assertIsNone(cache_oracle.threshold)

    def test_gate_fails_when_h0_train_below_minimum(self) -> None:
        cache_oracle = oracle()
        cache_oracle.fit(valid_split(h0_train=99, h1_train=1101))

        self.assertEqual(cache_oracle.activation_status["activation_gate_reason"], "min_train_h0_not_met")

    def test_gate_fails_when_h1_train_below_minimum(self) -> None:
        cache_oracle = oracle()
        cache_oracle.fit(valid_split(h0_train=1151, h1_train=49))

        self.assertEqual(cache_oracle.activation_status["activation_gate_reason"], "min_train_h1_not_met")

    def test_gate_fails_when_h0_calibration_below_minimum(self) -> None:
        cache_oracle = oracle()
        cache_oracle.fit(valid_split(h0_calib=499, h0_accepts=24))

        self.assertEqual(cache_oracle.activation_status["activation_gate_reason"], "min_calibration_h0_not_met")

    def test_gate_fails_when_h1_calibration_below_minimum(self) -> None:
        cache_oracle = oracle()
        cache_oracle.fit(valid_split(h1_calib=49))

        self.assertEqual(cache_oracle.activation_status["activation_gate_reason"], "min_calibration_h1_not_met")

    def test_gate_fails_when_threshold_is_non_finite(self) -> None:
        cache_oracle = oracle(scorer=GateScorer(threshold=float("inf")))
        cache_oracle.fit(valid_split())

        status = cache_oracle.activation_status
        self.assertEqual(status["activation_gate_reason"], "threshold_not_finite")
        self.assertFalse(status["threshold_is_finite"])

    def test_gate_passes_for_synthetic_wilson_case(self) -> None:
        cache_oracle = oracle()
        cache_oracle.fit(valid_split())

        status = cache_oracle.activation_status
        self.assertTrue(status["activation_gate_passed"])
        self.assertIsNone(status["activation_gate_reason"])
        self.assertAlmostEqual(status["wilson_upper_fpr"], 0.07277, places=5)
        self.assertAlmostEqual(status["allowed_fpr_bound"], 0.08)
        self.assertEqual(status["calibration_h0_accepts"], 25)
        self.assertEqual(cache_oracle.threshold, Threshold(0.5))

    def test_gate_fails_when_wilson_upper_exceeds_bound(self) -> None:
        cache_oracle = oracle()
        cache_oracle.fit(valid_split(h0_accepts=100))

        status = cache_oracle.activation_status
        self.assertFalse(status["activation_gate_passed"])
        self.assertEqual(status["activation_gate_reason"], "wilson_upper_fpr_exceeds_bound")
        self.assertGreater(status["wilson_upper_fpr"], status["allowed_fpr_bound"])

    def test_threshold_provider_updates_only_when_gate_passes(self) -> None:
        provider = RecordingThresholdProvider()

        failed_refit = oracle(threshold_provider=provider)
        failed_refit.fit(valid_split(h0_accepts=100))
        self.assertEqual(provider.calls, [])

        passed_refit = oracle(threshold_provider=provider)
        passed_refit.fit(valid_split())
        self.assertEqual(len(provider.calls), 1)
        self.assertEqual(provider.calls[-1][0], Threshold(0.5))

        failed_recalibration = oracle(threshold=Threshold(0.4), threshold_provider=provider)
        failed_recalibration._recalibrate_threshold_from_rows(rows(499, 0.0), metadata={})
        self.assertEqual(len(provider.calls), 1)

        passed_recalibration = oracle(
            threshold=Threshold(0.4),
            scorer=GateScorer(threshold=0.6),
            threshold_provider=provider,
        )
        passed_recalibration._recalibrate_threshold_from_rows(
            h0_calibration_scores(total=500, accepts=25, threshold=0.6),
            metadata={},
        )
        self.assertEqual(len(provider.calls), 2)
        self.assertEqual(provider.calls[-1][0], Threshold(0.6))

    def test_failed_refit_does_not_replace_existing_scorer_or_threshold(self) -> None:
        active_scorer = GateScorer(threshold=0.4, name="active")
        cache_oracle = oracle(scorer=active_scorer, threshold=Threshold(0.4))
        cache_oracle._threshold_version = 3

        cache_oracle.fit(valid_split(h0_accepts=100))

        self.assertIs(cache_oracle.scorer, active_scorer)
        self.assertEqual(cache_oracle.threshold, Threshold(0.4))
        self.assertEqual(cache_oracle.activation_status["threshold_version"], 3)
        self.assertFalse(cache_oracle.activation_status["activation_gate_passed"])

    def test_no_previous_threshold_failed_gate_keeps_abstain_behavior(self) -> None:
        vectors = StaticVectorStore([entry("candidate")])
        cache_oracle = oracle(vector_store=vectors, threshold=None)
        cache_oracle.fit(valid_split(h0_accepts=100))

        decision = cache_oracle.decide(lookup())

        self.assertIsNone(cache_oracle.threshold)
        self.assertEqual(decision.status, OracleDecisionStatus.ABSTAIN)
        self.assertEqual(decision.reason, "oracle_not_fitted")

    def test_recalibration_failure_does_not_replace_old_threshold(self) -> None:
        cache_oracle = oracle(threshold=Threshold(0.4))
        cache_oracle._threshold_version = 7

        result = cache_oracle._recalibrate_threshold_from_rows(rows(499, 0.0), metadata={})

        self.assertIsNone(result)
        self.assertEqual(cache_oracle.threshold, Threshold(0.4))
        self.assertEqual(cache_oracle.activation_status["threshold_version"], 7)
        self.assertEqual(cache_oracle.last_refit_decision.reason, "threshold_activation_gate_failed")

    def test_recalibration_success_updates_threshold(self) -> None:
        cache_oracle = oracle(threshold=Threshold(0.4), scorer=GateScorer(threshold=0.6))
        cache_oracle._threshold_version = 7

        result = cache_oracle._recalibrate_threshold_from_rows(
            h0_calibration_scores(total=500, accepts=25, threshold=0.6),
            metadata={},
        )

        self.assertEqual(result, Threshold(0.6))
        self.assertEqual(cache_oracle.threshold, Threshold(0.6))
        self.assertEqual(cache_oracle.activation_status["threshold_version"], 8)

    def test_split_aware_store_uses_explicit_buckets_and_records_source(self) -> None:
        store = InMemorySplitJudgeTrainingStore(
            max_h0_train=1200,
            max_h1_train=100,
            max_h0_calibration=600,
            max_h1_calibration=100,
        )
        for idx in range(1150):
            store.add_train(judged(JudgeLabel.NOT_REUSABLE, f"h0-train-{idx}", row(0.0)))
        for idx in range(50):
            store.add_train(judged(JudgeLabel.REUSABLE, f"h1-train-{idx}", row(1.0)))
        for idx, value in enumerate(h0_calibration_scores()):
            store.add_calibration(judged(JudgeLabel.NOT_REUSABLE, f"h0-calib-{idx}", value))
        for idx in range(50):
            store.add_calibration(judged(JudgeLabel.REUSABLE, f"h1-calib-{idx}", row(1.0)))
        recorder: dict[str, object] = {}
        cache_oracle = oracle(
            scorer=RecordingGateScorer(recorder=recorder),
            judge_training_store=store,
            refit_policy=AlwaysRefitPolicy(),
            auto_refit=True,
        )

        decision = cache_oracle._maybe_auto_refit(current_threshold=None)
        cache_oracle.wait_for_fit(timeout=2.0)

        self.assertEqual(decision.action, RefitAction.REFIT_SCORER)
        self.assertEqual(recorder["fit_h0"], rows(1150, 0.0))
        self.assertEqual(recorder["fit_h1"], rows(50, 1.0))
        self.assertEqual(cache_oracle.activation_status["split_source"], "split_judge_training_store")
        self.assertTrue(cache_oracle.activation_status["activation_gate_passed"])

    def test_legacy_store_still_resplits_and_records_source(self) -> None:
        store = InMemoryJudgeTrainingStore(max_h0=10, max_h1=10)
        for idx in range(5):
            store.add(judged(JudgeLabel.NOT_REUSABLE, f"h0-{idx}", row(float(idx))))
            store.add(judged(JudgeLabel.REUSABLE, f"h1-{idx}", row(float(idx + 10))))
        cache_oracle = oracle(judge_training_store=store)

        refit_rows = cache_oracle._refit_rows_from_training_store(store)
        split = cache_oracle._build_refit_split(refit_rows.h0_rows, refit_rows.h1_rows, metadata={})

        self.assertIsNotNone(split)
        self.assertEqual(split.metadata["split_source"], "legacy_resplit")
        self.assertEqual(split.h0_train, (row(0), row(1), row(2)))
        self.assertEqual(split.h0_calib, (row(3),))

    def test_request_t_shadow_labels_can_only_affect_future_requests(self) -> None:
        config = passing_config(
            min_train_total=2,
            min_train_h0=1,
            min_train_h1=1,
            min_calibration_h0=1,
            min_calibration_h1=0,
            fpr_wilson_margin=0.8,
        )
        vectors = StaticVectorStore([entry("h1"), entry("h0-train", (0.0, 1.0)), entry("h0-calib", (0.0, 1.0))])
        kv = DictKVStore()
        store = InMemorySplitJudgeTrainingStore(
            max_h0_train=10,
            max_h1_train=10,
            max_h0_calibration=10,
            max_h1_calibration=10,
        )
        judge = MappingJudge(
            {
                "h1": JudgeLabel.REUSABLE,
                "h0-train": JudgeLabel.NOT_REUSABLE,
                "h0-calib": JudgeLabel.NOT_REUSABLE,
            }
        )
        shadow = DefaultShadowTopKCollector(
            feature_builder=NormalizedHadamardFeatureBuilder(),
            judge=judge,
            store=store,
            config=ShadowCollectionConfig(top_k=3, calibration_every_n=2),
        )
        cache_oracle = oracle(
            scorer=GateScorer(threshold=0.5),
            config=config,
            threshold=None,
            vector_store=vectors,
            judge_training_store=store,
            shadow_collector=shadow,
            shadow_collection_enabled=True,
            auto_refit=True,
            refit_policy=AlwaysRefitPolicy(config),
            top_k=3,
        )
        gateway = SemanticCacheGateway(kv_store=kv, oracle=cache_oracle)
        for cached in vectors.entries:
            kv.set(cached.cache_key, cached.response)

        first = gateway.lookup_with_decision(lookup())
        cache_oracle.wait_for_fit(timeout=2.0)
        second = gateway.lookup_with_decision(lookup())

        self.assertEqual(first.decision.status, OracleDecisionStatus.ABSTAIN)
        self.assertIsNone(first.response)
        self.assertTrue(cache_oracle.activation_status["activation_gate_passed"])
        self.assertEqual(second.decision.status, OracleDecisionStatus.HIT)
        self.assertEqual(second.response, Response("response h1"))


if __name__ == "__main__":
    unittest.main()
