"""Diagnostic reporting interfaces."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class DiagnosticsReporter(ABC):
    """Receives diagnostic events for semantic cache decisions."""

    @abstractmethod
    def report(self, event: str, metadata: dict[str, Any] | None = None) -> None:
        raise NotImplementedError

