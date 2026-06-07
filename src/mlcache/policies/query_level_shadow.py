"""Stores query-level shadow policy decisions."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import replace

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


__all__ = ["InMemoryQueryLevelShadowDecisionStore", "QueryLevelShadowDecisionStore"]
