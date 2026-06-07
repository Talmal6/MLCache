import unittest

from mlcache.calibration import (
    DefaultQueryLevelCalibrationBuilder,
    QueryCalibrationCandidate,
    QueryCalibrationDataset,
    QueryCalibrationRecord,
    QueryLevelCalibrationConfig,
    ThresholdCalibrationRequest,
)
from mlcache.features import NormalizedHadamardFeatureBuilder
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
    OracleDecisionStatus,
    Query,
    Response,
    Score,
    ScorerName,
    Threshold,
    TieMode,
)


def candidate(
    score: float,
    label: int | None,
    *,
    rank: int,
    key: str,
    metadata: dict | None = None,
) -> QueryCalibrationCandidate:
    return QueryCalibrationCandidate(
        score=Score(score),
        label=label,
        candidate_rank=rank,
        candidate_key=CacheKey(key),
        metadata=metadata or {},
    )


def record(query_id: str, candidates: list[QueryCalibrationCandidate], metadata: dict | None = None):
    return QueryCalibrationRecord(query_id=query_id, candidates=candidates, metadata=metadata or {})


def calibration_dataset(selected_h0_scores, all_pair_h0_scores=None) -> QueryCalibrationDataset:
    return QueryCalibrationDataset(
        decisions=(),
        h0_scores=tuple(Score(float(score)) for score in selected_h0_scores),
        all_pair_h0_scores=tuple(Score(float(score)) for score in (all_pair_h0_scores or selected_h0_scores)),
    )


class StaticVectorStore(VectorStore):
    def __init__(self, entries: list[CacheEntry]) -> None:
        self.entries = entries

    def upsert(self, entry: CacheEntry) -> None:
        self.entries.append(entry)

    def delete(self, cache_key: CacheKey) -> None:
        self.entries = [entry for entry in self.entries if entry.cache_key != cache_key]

    def get(self, cache_key: CacheKey) -> VectorSearchResult | None:
        for entry in self.entries:
            if entry.cache_key == cache_key:
                return self._result(entry)
        return None

    def search(
        self,
        embedding: Embedding,
        *,
        namespace: str | None = None,
        top_k: int = 10,
    ) -> list[VectorSearchResult]:
        del embedding, namespace
        return [self._result(entry) for entry in self.entries[:top_k]]

    @staticmethod
    def _result(entry: CacheEntry) -> VectorSearchResult:
        return VectorSearchResult(
            cache_key=entry.cache_key,
            embedding=entry.embedding,
            score=Score(1.0),
            query=entry.query,
            metadata=entry.metadata,
        )


class CosineOnlyScorer(SemanticScorer):
    @property
    def name(self) -> ScorerName:
        return ScorerName("query_level_test")

    @property
    def input_space(self) -> InputSpace:
        return InputSpace.EMBEDDING

    def fit(self, batch: LabeledPairBatch, **kwargs: object) -> None:
        del batch, kwargs

    def copy_for_refit(self) -> "CosineOnlyScorer":
        return type(self)()

    def score(self, features) -> Score:
        return Score(float(features.cosine or 0.0))

    def calibrate(self, request: ThresholdCalibrationRequest) -> Threshold:
        del request
        return Threshold(0.5)

    def predict(self, features, threshold: Threshold, *, tie_mode: TieMode = TieMode.GE) -> bool:
        score = float(self.score(features))
        if tie_mode == TieMode.GT:
            return score > float(threshold)
        return score >= float(threshold)


class QueryLevelCalibrationTests(unittest.TestCase):
    def test_builder_selects_one_candidate_per_query(self) -> None:
        builder = DefaultQueryLevelCalibrationBuilder()
        dataset = builder.build_calibration_decisions(
            [
                record("q1", [candidate(0.1, 0, rank=1, key="q1-a"), candidate(0.9, 1, rank=2, key="q1-b")]),
                record("q2", [candidate(0.2, 0, rank=1, key="q2-a"), candidate(0.3, 0, rank=2, key="q2-b")]),
            ]
        )

        self.assertEqual(len(dataset.decisions), 2)
        self.assertEqual([decision.query_id for decision in dataset.decisions], ["q1", "q2"])
        self.assertEqual([decision.candidate_key for decision in dataset.decisions], [CacheKey("q1-b"), CacheKey("q2-b")])

    def test_selected_h0_scores_differ_from_all_pair_h0_scores(self) -> None:
        builder = DefaultQueryLevelCalibrationBuilder()
        dataset = builder.build_calibration_decisions(
            [
                record("q1", [candidate(0.1, 0, rank=1, key="q1-h0"), candidate(0.9, 1, rank=2, key="q1-h1")]),
                record("q2", [candidate(0.8, 0, rank=1, key="q2-h0"), candidate(0.2, 0, rank=2, key="q2-low")]),
            ]
        )

        self.assertEqual(tuple(float(score) for score in dataset.h0_scores), (0.8,))
        self.assertEqual(tuple(float(score) for score in dataset.all_pair_h0_scores), (0.1, 0.8, 0.2))

    def test_empty_top_k_query_can_be_skipped_or_marked_abstain(self) -> None:
        skipped = DefaultQueryLevelCalibrationBuilder().build_calibration_decisions([record("empty", [])])
        self.assertEqual(skipped.total_queries, 1)
        self.assertEqual(skipped.skipped_queries, 1)
        self.assertEqual(skipped.decisions, ())

        abstained = DefaultQueryLevelCalibrationBuilder(empty_query_policy="abstain").build_calibration_decisions(
            [record("empty", [])]
        )
        self.assertEqual(abstained.abstained_queries, 1)
        self.assertEqual(abstained.decisions[0].status, OracleDecisionStatus.ABSTAIN)
        self.assertEqual(abstained.decisions[0].metadata["reason"], "empty_top_k")

    def test_labels_metadata_and_candidate_ranks_are_preserved(self) -> None:
        builder = DefaultQueryLevelCalibrationBuilder()
        dataset = builder.build_calibration_decisions(
            [
                record(
                    "q1",
                    [
                        candidate(
                            0.7,
                            0,
                            rank=7,
                            key="candidate",
                            metadata={"candidate_source": "shadow"},
                        )
                    ],
                    metadata={"tenant": "test"},
                )
            ]
        )

        decision = dataset.decisions[0]
        self.assertEqual(decision.label, 0)
        self.assertEqual(decision.candidate_rank, 7)
        self.assertEqual(decision.candidate_key, CacheKey("candidate"))
        self.assertEqual(decision.query_metadata["tenant"], "test")
        self.assertEqual(decision.candidate_metadata["candidate_source"], "shadow")

    def test_query_level_builder_does_not_change_serving_behavior(self) -> None:
        cached = CacheEntry(
            cache_key=CacheKey("cached"),
            query=Query("cached query"),
            response=Response("cached response"),
            embedding=(1.0, 0.0),
            metadata=CacheMetadata(),
        )
        oracle = TrainableSemanticCacheOracle(
            vector_store=StaticVectorStore([cached]),
            feature_builder=NormalizedHadamardFeatureBuilder(),
            scorer=CosineOnlyScorer(),
            auto_refit=False,
        )
        oracle._threshold = Threshold(0.5)
        request = CacheLookup(query=Query("incoming"), embedding=(1.0, 0.0))

        before = oracle.decide(request)
        DefaultQueryLevelCalibrationBuilder().build_calibration_decisions(
            [record("q1", [candidate(0.1, 0, rank=1, key="unserved-h0")])]
        )
        after = oracle.decide(request)

        self.assertEqual(before.status, OracleDecisionStatus.HIT)
        self.assertEqual(after.status, before.status)
        self.assertEqual(after.cache_key, before.cache_key)
        self.assertEqual(after.threshold, before.threshold)

    def test_calibrate_uses_selected_h0_scores_not_all_pair_h0_scores(self) -> None:
        dataset = calibration_dataset(
            selected_h0_scores=[0.1] * 500,
            all_pair_h0_scores=([0.1] * 500) + ([0.99] * 500),
        )

        result = DefaultQueryLevelCalibrationBuilder().calibrate(dataset)

        self.assertIsNotNone(result.threshold)
        self.assertLess(float(result.threshold), 0.99)
        self.assertEqual(result.metadata["selected_h0_accepts"], 0)
        self.assertEqual(result.metadata["all_pair_h0_accepts_at_threshold"], 500)
        self.assertEqual(result.metadata["selected_vs_all_pair_note"], "threshold_calibrated_on_selected_h0_scores_only")

    def test_calibration_fails_when_selected_h0_count_below_minimum(self) -> None:
        builder = DefaultQueryLevelCalibrationBuilder(
            config=QueryLevelCalibrationConfig(min_selected_h0=11)
        )

        result = builder.calibrate(calibration_dataset([0.1] * 10))

        self.assertIsNone(result.threshold)
        self.assertFalse(result.metadata["calibration_gate_passed"])
        self.assertEqual(result.metadata["calibration_gate_reason"], "min_selected_h0_not_met")

    def test_calibration_passes_with_enough_selected_h0_and_wilson_bound(self) -> None:
        dataset = calibration_dataset(([0.9] * 25) + ([0.1] * 475))

        result = DefaultQueryLevelCalibrationBuilder().calibrate(dataset)

        self.assertEqual(result.threshold, Threshold(0.9))
        self.assertTrue(result.metadata["calibration_gate_passed"])
        self.assertIsNone(result.metadata["calibration_gate_reason"])
        self.assertEqual(result.metadata["selected_h0_count"], 500)
        self.assertEqual(result.metadata["selected_h0_accepts"], 25)
        self.assertAlmostEqual(result.metadata["empirical_selected_fpr"], 0.05)
        self.assertAlmostEqual(result.metadata["wilson_upper_selected_fpr"], 0.07277, places=5)
        self.assertAlmostEqual(result.metadata["allowed_fpr_bound"], 0.08)
        self.assertTrue(result.metadata["threshold_is_finite"])

    def test_calibration_fails_when_wilson_upper_exceeds_alpha_plus_margin(self) -> None:
        builder = DefaultQueryLevelCalibrationBuilder(
            config=QueryLevelCalibrationConfig(fpr_wilson_margin=0.0)
        )
        dataset = calibration_dataset(([0.9] * 25) + ([0.1] * 475))

        result = builder.calibrate(dataset)

        self.assertIsNone(result.threshold)
        self.assertFalse(result.metadata["calibration_gate_passed"])
        self.assertEqual(result.metadata["calibration_gate_reason"], "wilson_upper_selected_fpr_exceeds_bound")
        self.assertGreater(result.metadata["wilson_upper_selected_fpr"], result.metadata["allowed_fpr_bound"])

    def test_all_pair_h0_scores_are_diagnostics_only(self) -> None:
        selected = ([0.9] * 25) + ([0.1] * 475)
        builder = DefaultQueryLevelCalibrationBuilder()

        with_low_all_pair = builder.calibrate(calibration_dataset(selected, all_pair_h0_scores=selected))
        with_high_all_pair = builder.calibrate(
            calibration_dataset(selected, all_pair_h0_scores=selected + ([0.99] * 50))
        )

        self.assertEqual(with_low_all_pair.threshold, with_high_all_pair.threshold)
        self.assertEqual(with_low_all_pair.metadata["selected_h0_count"], 500)
        self.assertEqual(with_high_all_pair.metadata["selected_h0_count"], 500)
        self.assertEqual(with_low_all_pair.metadata["all_pair_h0_count"], 500)
        self.assertEqual(with_high_all_pair.metadata["all_pair_h0_count"], 550)

    def test_ge_and_gt_tie_modes_are_deterministic(self) -> None:
        dataset = calibration_dataset([0.9, 0.8, 0.1, 0.0])
        ge_builder = DefaultQueryLevelCalibrationBuilder(
            config=QueryLevelCalibrationConfig(
                target_false_accept_rate=0.25,
                min_selected_h0=4,
                fpr_wilson_margin=0.8,
                tie_mode=TieMode.GE,
            )
        )
        gt_builder = DefaultQueryLevelCalibrationBuilder(
            config=QueryLevelCalibrationConfig(
                target_false_accept_rate=0.25,
                min_selected_h0=4,
                fpr_wilson_margin=0.8,
                tie_mode=TieMode.GT,
            )
        )

        ge = ge_builder.calibrate(dataset)
        gt = gt_builder.calibrate(dataset)

        self.assertEqual(ge.threshold, Threshold(0.9))
        self.assertLess(float(gt.threshold), 0.9)
        self.assertEqual(gt.metadata["tie_mode"], "gt")
        self.assertEqual(ge.metadata["selected_h0_accepts"], 1)
        self.assertEqual(gt.metadata["selected_h0_accepts"], 1)

    def test_query_level_calibration_result_preserves_dataset(self) -> None:
        dataset = calibration_dataset(([0.9] * 25) + ([0.1] * 475))

        result = DefaultQueryLevelCalibrationBuilder().calibrate(dataset)

        self.assertIs(result.dataset, dataset)


if __name__ == "__main__":
    unittest.main()
