"""Compatibility wrapper for the old flat oracle module."""

from mlcache.oracle import (
    OracleFitResult,
    OracleJudgeFeedback,
    OracleRuntimeSnapshot,
    OracleScoredResult,
    SemanticCacheOracle,
    TrainableSemanticCacheOracle,
)

__all__ = [
    "OracleFitResult",
    "OracleJudgeFeedback",
    "OracleRuntimeSnapshot",
    "OracleScoredResult",
    "SemanticCacheOracle",
    "TrainableSemanticCacheOracle",
]

