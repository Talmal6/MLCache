"""Observability metric interfaces."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class MetricsSink(ABC):
    """Receives semantic cache metrics."""

    @abstractmethod
    def record(self, name: str, value: float, metadata: dict[str, Any] | None = None) -> None:
        raise NotImplementedError

