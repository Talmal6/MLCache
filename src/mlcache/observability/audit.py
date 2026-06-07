"""Audit logging contracts."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from mlcache.semantic_types import CacheKey, OracleDecision, Query


@dataclass(frozen=True, slots=True)
class AuditEvent:
    timestamp: datetime
    query: Query
    decision: OracleDecision
    candidate_keys: tuple[CacheKey, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


class AuditLogger(ABC):
    """Records oracle decisions and evidence."""

    @abstractmethod
    def log(self, event: AuditEvent) -> None:
        raise NotImplementedError
