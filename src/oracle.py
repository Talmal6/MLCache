"""Semantic cache oracle contract."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from semantic_desider.online_properties import OnlineBatch, StopStatus
from semantic_desider.features import PairFeatureBuilder, PairFeatures
from semantic_desider.judges import JudgeLabel, JudgeRequest, JudgedPairExample, JudgeTrainingStore, SemanticReuseJudge
from semantic_desider.online_properties import OnlineUpdater
from semantic_desider.scorers import SemanticScorer
from semantic_desider.thresholds import ThresholdCalibrationRequest, ThresholdProvider, ThresholdScope
from semantic_desider.types import (
    CacheEntry,
    CacheKey,
    CacheLookup,
    OracleDecision,
    OracleDecisionStatus,
    Score,
    Threshold,
    TieMode,
    TrainCalibEvalSplit,
)
from semantic_desider.vector_store import VectorStore


class SemanticCacheOracle(ABC):
    """Owns all semantic correctness logic for cache reuse."""

    @abstractmethod
    def decide(self, request: CacheLookup) -> OracleDecision:
        raise NotImplementedError

    @abstractmethod
    def fit(self, split: TrainCalibEvalSplit) -> None:
        raise NotImplementedError

    @abstractmethod
    def update_online(self, batch: OnlineBatch) -> StopStatus:
        raise NotImplementedError

    @abstractmethod
    def index(self, entry: CacheEntry) -> None:
        raise NotImplementedError

    @abstractmethod
    def remove(self, cache_key: CacheKey) -> None:
        raise NotImplementedError


@dataclass(frozen=True, slots=True)
class OracleFitResult:
    scorer: str
    threshold: Threshold
    target_false_accept_rate: float
    n_h0_train: int
    n_h1_train: int
    n_h0_calib: int
    n_h1_calib: int
    metadata: dict[str, Any] = field(default_factory=dict)


class TrainableSemanticCacheOracle(SemanticCacheOracle):
    """Trainable oracle that owns scorer fitting, NP thresholding, and candidate decisions."""

    def __init__(
        self,
        *,
        vector_store: VectorStore,
        feature_builder: PairFeatureBuilder,
        scorer: SemanticScorer,
        threshold_provider: ThresholdProvider | None = None,
        online_updater: OnlineUpdater | None = None,
        judge: SemanticReuseJudge | None = None,
        judge_training_store: JudgeTrainingStore | None = None,
        target_false_accept_rate: float = 0.05,
        tie_mode: TieMode = TieMode.GE,
        top_k: int = 1,
    ) -> None:
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        if not 0.0 < target_false_accept_rate < 1.0:
            raise ValueError("target_false_accept_rate must be in (0,1)")

        self.vector_store = vector_store
        self.feature_builder = feature_builder
        self.scorer = scorer
        self.threshold_provider = threshold_provider
        self.online_updater = online_updater
        self.judge = judge
        self.judge_training_store = judge_training_store
        self.target_false_accept_rate = float(target_false_accept_rate)
        self.tie_mode = tie_mode
        self.top_k = int(top_k)
        self._threshold: Threshold | None = None
        self._fit_result: OracleFitResult | None = None

    @property
    def fit_result(self) -> OracleFitResult | None:
        return self._fit_result

    @property
    def threshold(self) -> Threshold | None:
        return self._threshold

    def decide(self, request: CacheLookup) -> OracleDecision:
        candidates = self.vector_store.search(
            request.embedding,
            namespace=request.namespace,
            top_k=self._resolve_top_k(request),
        )
        if not candidates:
            threshold = self._resolve_threshold(request)
            return OracleDecision(
                status=OracleDecisionStatus.MISS,
                accepted=False,
                cache_key=None,
                score=None,
                threshold=threshold,
                scorer=self.scorer.name,
                reason="no_neighbors",
            )

        threshold = self._resolve_threshold(request)
        if threshold is None:
            return self._judge_nearest_candidate(request, candidates[0])

        best_key: CacheKey | None = None
        best_score: Score | None = None
        best_rank: int | None = None
        scored: list[dict[str, Any]] = []

        for rank, candidate in enumerate(candidates, start=1):
            features = self.feature_builder.build(request.embedding, candidate.embedding)
            score = self.scorer.score(features)
            accepted = self.scorer.predict(features, threshold, tie_mode=self.tie_mode)
            scored.append(
                {
                    "rank": rank,
                    "cache_key": str(candidate.cache_key),
                    "score": float(score),
                    "accepted": accepted,
                    "vector_score": float(candidate.score),
                }
            )
            if accepted and (best_score is None or float(score) > float(best_score)):
                best_key = candidate.cache_key
                best_score = score
                best_rank = rank

        if best_key is None:
            return OracleDecision(
                status=OracleDecisionStatus.MISS,
                accepted=False,
                cache_key=None,
                score=max((Score(item["score"]) for item in scored), default=None),
                threshold=threshold,
                scorer=self.scorer.name,
                candidate_count=len(candidates),
                reason="no_candidate_passed_threshold",
                evidence={"candidates": scored},
            )

        return OracleDecision(
            status=OracleDecisionStatus.HIT,
            accepted=True,
            cache_key=best_key,
            score=best_score,
            threshold=threshold,
            scorer=self.scorer.name,
            candidate_count=len(candidates),
            accepted_candidate_rank=best_rank,
            evidence={"candidates": scored},
        )

    def fit(self, split: TrainCalibEvalSplit) -> None:
        from semantic_desider.types import LabeledPairBatch

        self.scorer.fit(
            LabeledPairBatch(
                h0=split.h0_train,
                h1=split.h1_train,
                weights=split.metadata.get("weights"),
                metadata=split.metadata,
            ),
            alpha=self.target_false_accept_rate,
        )

        h0_scores = [self.scorer.score(self._row_features(row)) for row in split.h0_calib]
        threshold = self.scorer.calibrate(
            ThresholdCalibrationRequest(
                h0_scores=h0_scores,
                target_false_accept_rate=self.target_false_accept_rate,
                tie_mode=self.tie_mode,
                context=split.metadata,
            )
        )
        self._threshold = threshold

        if self.threshold_provider is not None:
            self.threshold_provider.set_threshold(
                threshold,
                scorer=self.scorer.name,
                scope=ThresholdScope.GLOBAL,
                context=split.metadata,
            )

        self._fit_result = OracleFitResult(
            scorer=str(self.scorer.name),
            threshold=threshold,
            target_false_accept_rate=self.target_false_accept_rate,
            n_h0_train=len(split.h0_train),
            n_h1_train=len(split.h1_train),
            n_h0_calib=len(split.h0_calib),
            n_h1_calib=len(split.h1_calib),
            metadata=split.metadata,
        )

    def update_online(self, batch: OnlineBatch) -> StopStatus:
        if self.online_updater is None:
            return StopStatus(
                stopped=False,
                reason="online_updater_not_configured",
                metadata={"batch_size": len(batch.features)},
            )
        self.online_updater.update_batch(batch)
        return StopStatus(
            stopped=False,
            reason="online_batch_applied",
            metadata={"batch_size": len(batch.features)},
        )

    def index(self, entry: CacheEntry) -> None:
        self.vector_store.upsert(entry)

    def remove(self, cache_key: CacheKey) -> None:
        self.vector_store.delete(cache_key)

    def _resolve_threshold(self, request: CacheLookup) -> Threshold | None:
        if self.threshold_provider is not None:
            try:
                return self.threshold_provider.get_threshold(
                    scorer=self.scorer.name,
                    scope=ThresholdScope.GLOBAL,
                    context={"request": request},
                )
            except Exception:
                pass
        return self._threshold

    def _resolve_top_k(self, request: CacheLookup) -> int:
        raw_top_k = request.metadata.attributes.get("top_k")
        if raw_top_k is None:
            return self.top_k
        top_k = int(raw_top_k)
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        return top_k

    @staticmethod
    def _row_features(row: Any) -> PairFeatures:
        return PairFeatures(hadamard=tuple(float(value) for value in row))

    def _judge_nearest_candidate(self, request: CacheLookup, candidate: Any) -> OracleDecision:
        features = self.feature_builder.build(request.embedding, candidate.embedding)
        score = self._safe_score(features)

        if self.judge is None:
            return OracleDecision(
                status=OracleDecisionStatus.ABSTAIN,
                accepted=False,
                cache_key=None,
                score=score,
                threshold=None,
                scorer=self.scorer.name,
                candidate_count=1,
                reason="oracle_not_fitted_and_judge_not_configured",
                evidence={"candidate_key": str(candidate.cache_key), "vector_score": float(candidate.score)},
            )

        judge_request = JudgeRequest(
            query=request.query,
            candidate_query=candidate.query,
            candidate_key=candidate.cache_key,
            metadata=candidate.metadata,
            context={
                "request_metadata": request.metadata,
                "vector_score": float(candidate.score),
            },
        )
        judge_result = self.judge.judge(judge_request)

        if self.judge_training_store is not None:
            self.judge_training_store.add(
                JudgedPairExample(
                    features=self._features_to_tuple(features),
                    request=judge_request,
                    decision=judge_result.decision,
                    metadata={"oracle_score": float(score) if score is not None else None},
                )
            )

        if judge_result.decision.label == JudgeLabel.REUSABLE:
            decision_score = judge_result.decision.confidence if judge_result.decision.confidence is not None else score
            return OracleDecision(
                status=OracleDecisionStatus.HIT,
                accepted=True,
                cache_key=candidate.cache_key,
                score=decision_score,
                threshold=None,
                scorer=self.scorer.name,
                candidate_count=1,
                accepted_candidate_rank=1,
                reason="judge_labeled_h1",
                evidence={
                    "candidate_key": str(candidate.cache_key),
                    "judge": self.judge.name,
                    "judge_label": judge_result.decision.label.value,
                    "judge_rationale": judge_result.decision.rationale,
                    "vector_score": float(candidate.score),
                },
            )

        if judge_result.decision.label == JudgeLabel.NOT_REUSABLE:
            decision_score = judge_result.decision.confidence if judge_result.decision.confidence is not None else score
            return OracleDecision(
                status=OracleDecisionStatus.MISS,
                accepted=False,
                cache_key=None,
                score=decision_score,
                threshold=None,
                scorer=self.scorer.name,
                candidate_count=1,
                reason="judge_labeled_h0",
                evidence={
                    "candidate_key": str(candidate.cache_key),
                    "judge": self.judge.name,
                    "judge_label": judge_result.decision.label.value,
                    "judge_rationale": judge_result.decision.rationale,
                    "vector_score": float(candidate.score),
                },
            )

        decision_score = judge_result.decision.confidence if judge_result.decision.confidence is not None else score
        return OracleDecision(
            status=OracleDecisionStatus.ABSTAIN,
            accepted=False,
            cache_key=None,
            score=decision_score,
            threshold=None,
            scorer=self.scorer.name,
            candidate_count=1,
            reason="judge_uncertain",
            evidence={
                "candidate_key": str(candidate.cache_key),
                "judge": self.judge.name,
                "judge_label": judge_result.decision.label.value,
                "judge_rationale": judge_result.decision.rationale,
                "vector_score": float(candidate.score),
            },
        )

    @staticmethod
    def _features_to_tuple(features: PairFeatures) -> tuple[float, ...]:
        if features.concat:
            return tuple(float(v) for v in features.concat)
        if features.hadamard:
            return tuple(float(v) for v in features.hadamard)
        if features.abs_diff:
            return tuple(float(v) for v in features.abs_diff)
        if features.cosine is not None:
            return (float(features.cosine),)
        main = features.values.get("main")
        if main is not None:
            return tuple(float(v) for v in main)
        return ()

    def _safe_score(self, features: PairFeatures) -> Score | None:
        try:
            return self.scorer.score(features)
        except Exception:
            return None
