"""Semantic cache oracle contracts and implementations."""

from mlcache.oracle.base import SemanticCacheOracle
from mlcache.oracle.fit import ActivationGateResult, OracleFitResult
from mlcache.oracle.runtime import OracleJudgeFeedback, OracleRuntimeSnapshot, OracleScoredResult
from mlcache.oracle.trainable import TrainableSemanticCacheOracle

__all__ = [
    "OracleFitResult",
    "ActivationGateResult",
    "OracleJudgeFeedback",
    "OracleRuntimeSnapshot",
    "OracleScoredResult",
    "SemanticCacheOracle",
    "TrainableSemanticCacheOracle",
]
