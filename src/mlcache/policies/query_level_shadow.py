"""Stores query-level shadow policy decisions."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import replace
from pathlib import Path

from mlcache.persistence import (
    atomic_write_json,
    decode_query_level_policy_decision,
    encode_query_level_policy_decision,
    read_json_or_default,
)
from mlcache.policies.query_level import QueryLevelPolicyDecision


class QueryLevelShadowDecisionStore(ABC):
    """Stores query-level policy decisions produced in shadow mode."""

    @abstractmethod
    def add(self, decision: QueryLevelPolicyDecision) -> None:
        raise NotImplementedError

    @abstractmethod
    def decisions(self) -> tuple[QueryLevelPolicyDecision, ...]:
        raise NotImplementedError

    @abstractmethod
    def clear(self) -> None:
        raise NotImplementedError


class InMemoryQueryLevelShadowDecisionStore(QueryLevelShadowDecisionStore):
    """Bounded FIFO in-memory store for query-level shadow decisions."""

    def __init__(self, *, max_decisions: int = 100_000) -> None:
        if int(max_decisions) <= 0:
            raise ValueError("max_decisions must be positive")
        self.max_decisions = int(max_decisions)
        self._decisions: list[QueryLevelPolicyDecision] = []

    def add(self, decision: QueryLevelPolicyDecision) -> None:
        if len(self._decisions) >= self.max_decisions:
            self._decisions.pop(0)
        self._decisions.append(self._copy_decision(decision))

    def decisions(self) -> tuple[QueryLevelPolicyDecision, ...]:
        return tuple(self._copy_decision(decision) for decision in self._decisions)

    def clear(self) -> None:
        self._decisions.clear()

    @staticmethod
    def _copy_decision(decision: QueryLevelPolicyDecision) -> QueryLevelPolicyDecision:
        return replace(decision, metadata=dict(decision.metadata))


class FileQueryLevelShadowDecisionStore(InMemoryQueryLevelShadowDecisionStore):
    """Bounded FIFO JSON-backed query-level shadow decision store."""

    def __init__(self, path: str | Path, *, max_decisions: int = 100_000) -> None:
        self.path = Path(path)
        super().__init__(max_decisions=max_decisions)
        data = read_json_or_default(self.path, {"decisions": []})
        decoded = tuple(decode_query_level_policy_decision(item) for item in data.get("decisions", ()))
        self._decisions = [
            self._copy_decision(decision)
            for decision in decoded[-self.max_decisions :]
        ]

    def add(self, decision: QueryLevelPolicyDecision) -> None:
        super().add(decision)
        self._persist()

    def clear(self) -> None:
        super().clear()
        self._persist()

    def _persist(self) -> None:
        atomic_write_json(
            self.path,
            {
                "format": "mlcache.file_query_level_shadow_decision_store.v1",
                "max_decisions": self.max_decisions,
                "decisions": [
                    encode_query_level_policy_decision(decision)
                    for decision in self._decisions
                ],
            },
        )


__all__ = [
    "FileQueryLevelShadowDecisionStore",
    "InMemoryQueryLevelShadowDecisionStore",
    "QueryLevelShadowDecisionStore",
]
