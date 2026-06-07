"""Shadow top-k feedback collection interfaces."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class ShadowTopKCollector(ABC):
    """Collects labels for top-k candidates independently of serving decisions."""

    @abstractmethod
    def collect(self, request: Any, candidates: Any, served_decision: Any) -> Any:
        raise NotImplementedError

