"""Compatibility wrapper for the old flat thresholds module."""

from mlcache.calibration import (
    CalibrationExample,
    NPThresholdCalibrator,
    ThresholdCalibrationRequest,
    ThresholdProvider,
    ThresholdScope,
    wilson_upper_bound,
)

__all__ = [
    "CalibrationExample",
    "NPThresholdCalibrator",
    "ThresholdCalibrationRequest",
    "ThresholdProvider",
    "ThresholdScope",
    "wilson_upper_bound",
]

