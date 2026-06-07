"""Compatibility wrapper for the old flat thresholds module."""

from mlcache.calibration import (
    CalibrationExample,
    DefaultQueryLevelCalibrationBuilder,
    NPThresholdCalibrator,
    QueryCalibrationCandidate,
    QueryCalibrationDataset,
    QueryCalibrationDecision,
    QueryCalibrationRecord,
    QueryLevelCalibrationResult,
    QueryLevelCalibrator,
    ThresholdCalibrationRequest,
    ThresholdProvider,
    ThresholdScope,
    wilson_upper_bound,
)

__all__ = [
    "CalibrationExample",
    "DefaultQueryLevelCalibrationBuilder",
    "NPThresholdCalibrator",
    "QueryCalibrationCandidate",
    "QueryCalibrationDataset",
    "QueryCalibrationDecision",
    "QueryCalibrationRecord",
    "QueryLevelCalibrationResult",
    "QueryLevelCalibrator",
    "ThresholdCalibrationRequest",
    "ThresholdProvider",
    "ThresholdScope",
    "wilson_upper_bound",
]
