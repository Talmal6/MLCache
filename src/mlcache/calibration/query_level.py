"""Query-level calibration interfaces."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class QueryLevelCalibrator(ABC):
    """Calibrates decisions selected per query rather than independent pair scores."""

    @abstractmethod
    def build_calibration_decisions(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    @abstractmethod
    def calibrate(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

