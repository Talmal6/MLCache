"""Compatibility wrapper for the old flat semantic_types module."""

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
    RegionId,
    Response,
    Score,
    ScorerName,
    Threshold,
    TieMode,
    TrainCalibEvalSplit,
)

__all__ = [
    "CacheEntry",
    "CacheKey",
    "CacheLookup",
    "CacheMetadata",
    "Embedding",
    "InputSpace",
    "LabeledPairBatch",
    "OracleDecision",
    "OracleDecisionStatus",
    "Query",
    "RegionId",
    "Response",
    "Score",
    "ScorerName",
    "Threshold",
    "TieMode",
    "TrainCalibEvalSplit",
]

