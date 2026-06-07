"""Compatibility wrapper for the old flat oracle module."""

from mlcache.oracle import (
    ActivationGateResult,
    OracleFitResult,
    OracleJudgeFeedback,
    OracleRuntimeSnapshot,
    OracleScoredResult,
    SemanticCacheOracle,
    TrainableSemanticCacheOracle,
)

__all__ = [
    "ActivationGateResult",
    "OracleFitResult",
    "OracleJudgeFeedback",
    "OracleRuntimeSnapshot",
    "OracleScoredResult",
    "SemanticCacheOracle",
    "TrainableSemanticCacheOracle",
]
