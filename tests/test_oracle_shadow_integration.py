import unittest

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
from mlcache.oracle import TrainableSemanticCacheOracle
from mlcache.retrieval import VectorSearchResult, VectorStore
from mlcache.scorers import SemanticScorer
from mlcache.semantic_types import (
    CacheEntry,
    CacheKey,
    CacheLookup,
    InputSpace,
    LabeledPairBatch,
    OracleDecisionStatus,
    Query,
    Score,
    ScorerName,
    Threshold,
    TieMode,
)
from mlcache.calibration import ThresholdCalibrationRequest


class StaticVectorStore(VectorStore):
    def __init__(self, candidates: list[VectorSearchResult]) -> None:
        self.candidates = candidates

    def upsert(self, entry: CacheEntry) -> None:
        del entry

    def delete(self, cache_key: CacheKey) -> None:
        del cache_key

    def get(self, cache_key: CacheKey) -> VectorSearchResult | None:
        del cache_key
        return None

    def search(
        self,
        embedding,
        *,
        namespace: str | None = None,
        top_k: int = 10,
    ) -> list[VectorSearchResult]:
        del embedding, namespace
        return self.candidates[:top_k]


class FixedScorer(SemanticScorer):
    def __init__(self, score: float) -> None:
        self.score_value = float(score)

    @property
    def name(self) -> ScorerName:
        return ScorerName("fixed")

    @property
    def input_space(self) -> InputSpace:
        return InputSpace.EMBEDDING

    def fit(self, batch: LabeledPairBatch, **kwargs: object) -> None:
        del batch, kwargs

    def copy_for_refit(self) -> "FixedScorer":
        return FixedScorer(self.score_value)

    def score(self, features) -> Score:
        del features
        return Score(self.score_value)

    def calibrate(self, request: ThresholdCalibrationRequest) -> Threshold:
        del request
        return Threshold(0.5)

    def predict(self, features, threshold: Threshold, *, tie_mode: TieMode = TieMode.GE) -> bool:
        del features
        if tie_mode == TieMode.GT:
            return self.score_value > float(threshold)
        return self.score_value >= float(threshold)


class ConstantJudge(SemanticReuseJudge):
    def __init__(self, label: JudgeLabel) -> None:
        self.label = label

    @property
    def name(self) -> str:
        return "constant"

    def judge(self, request: JudgeRequest) -> JudgeResult:
        return JudgeResult(request=request, decision=JudgeDecision(label=self.label))


def candidate(key: str) -> VectorSearchResult:
    return VectorSearchResult(
        cache_key=CacheKey(key),
        embedding=(1.0, 0.0),
        score=Score(0.99),
        query=Query("cached"),
    )


def request() -> CacheLookup:
    return CacheLookup(query=Query("incoming"), embedding=(1.0, 0.0))


def split_store() -> InMemorySplitJudgeTrainingStore:
    return InMemorySplitJudgeTrainingStore(
        max_h0_train=10,
        max_h1_train=10,
        max_h0_calibration=10,
        max_h1_calibration=10,
    )


class OracleShadowIntegrationTests(unittest.TestCase):
    def test_shadow_collection_does_not_change_serving_decision(self) -> None:
        store = split_store()
        shadow = DefaultShadowTopKCollector(
            feature_builder=NormalizedHadamardFeatureBuilder(),
            judge=ConstantJudge(JudgeLabel.NOT_REUSABLE),
            store=store,
        )
        oracle = TrainableSemanticCacheOracle(
            vector_store=StaticVectorStore([candidate("hit")]),
            feature_builder=NormalizedHadamardFeatureBuilder(),
            scorer=FixedScorer(1.0),
            judge_training_store=store,
            shadow_collector=shadow,
            shadow_collection_enabled=True,
            auto_refit=False,
        )
        oracle._threshold = Threshold(0.5)

        decision = oracle.decide(request())

        self.assertEqual(decision.status, OracleDecisionStatus.HIT)
        self.assertTrue(decision.accepted)
        self.assertEqual(decision.cache_key, CacheKey("hit"))
        self.assertEqual(len(store.h0_train()), 1)

    def test_shadow_labels_collected_on_request_t_do_not_affect_decision_t(self) -> None:
        store = split_store()
        shadow = DefaultShadowTopKCollector(
            feature_builder=NormalizedHadamardFeatureBuilder(),
            judge=ConstantJudge(JudgeLabel.REUSABLE),
            store=store,
        )
        oracle = TrainableSemanticCacheOracle(
            vector_store=StaticVectorStore([candidate("miss")]),
            feature_builder=NormalizedHadamardFeatureBuilder(),
            scorer=FixedScorer(0.0),
            judge_training_store=store,
            shadow_collector=shadow,
            shadow_collection_enabled=True,
            auto_refit=True,
        )
        oracle._threshold = Threshold(0.5)

        decision = oracle.decide(request())

        self.assertEqual(decision.status, OracleDecisionStatus.MISS)
        self.assertFalse(decision.accepted)
        self.assertIsNone(decision.cache_key)
        self.assertEqual(len(store.h1_train()), 1)


if __name__ == "__main__":
    unittest.main()

