"""Oracle runtime decision types."""

from __future__ import annotations

from dataclasses import dataclass

from mlcache.features import PairFeatures
from mlcache.feedback import JudgeResult
from mlcache.retrieval import VectorSearchResult
from mlcache.scorers import SemanticScorer
from mlcache.semantic_types import OracleDecision, Score, Threshold, TieMode


@dataclass(frozen=True, slots=True)
class OracleRuntimeSnapshot:
    scorer: SemanticScorer
    threshold: Threshold | None
    top_k: int
    tie_mode: TieMode
    scorer_name: str
    threshold_version: str | None = None
    semantic_hits_disabled: bool = False


@dataclass(frozen=True, slots=True)
class OracleScoredResult:
    decision: OracleDecision
    feedback_candidate: VectorSearchResult | None = None
    feedback_candidate_rank: int | None = None
    feedback_features: PairFeatures | None = None
    feedback_score: Score | None = None


@dataclass(frozen=True, slots=True)
class OracleJudgeFeedback:
    result: JudgeResult
    candidate: VectorSearchResult
    features: PairFeatures
    score: Score | None
    scorer_decision: OracleDecision
    candidate_rank: int | None = None

