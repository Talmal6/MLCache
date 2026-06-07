"""Judge-labeled training stores."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Sequence

from mlcache.feedback.types import JudgeDecision, JudgeLabel, JudgeRequest


@dataclass(frozen=True, slots=True)
class JudgedPairExample:
    features: tuple[float, ...]
    request: JudgeRequest
    decision: JudgeDecision
    created_at: datetime = field(default_factory=datetime.utcnow)
    metadata: dict[str, Any] = field(default_factory=dict)


class JudgeTrainingStore(ABC):
    """Stores judge-labeled H0/H1 examples for later oracle training."""

    @abstractmethod
    def add(self, example: JudgedPairExample) -> None:
        raise NotImplementedError

    @abstractmethod
    def h0(self) -> tuple[JudgedPairExample, ...]:
        raise NotImplementedError

    @abstractmethod
    def h1(self) -> tuple[JudgedPairExample, ...]:
        raise NotImplementedError


class TrainingExampleEvictionPolicy(ABC):
    """Chooses which training example to evict when a label store is full."""

    @abstractmethod
    def choose_eviction_index(
        self,
        *,
        label: JudgeLabel,
        existing: Sequence[JudgedPairExample],
        incoming: JudgedPairExample,
    ) -> int:
        raise NotImplementedError


class FIFOTrainingExampleEvictionPolicy(TrainingExampleEvictionPolicy):
    def choose_eviction_index(
        self,
        *,
        label: JudgeLabel,
        existing: Sequence[JudgedPairExample],
        incoming: JudgedPairExample,
    ) -> int:
        del label, existing, incoming
        return 0


class InMemoryJudgeTrainingStore(JudgeTrainingStore):
    """Two bounded in-memory stores: one for H0 and one for H1."""

    def __init__(
        self,
        *,
        max_h0: int,
        max_h1: int,
        eviction_policy: TrainingExampleEvictionPolicy | None = None,
    ) -> None:
        if max_h0 <= 0 or max_h1 <= 0:
            raise ValueError("max_h0 and max_h1 must be positive")
        self.max_h0 = int(max_h0)
        self.max_h1 = int(max_h1)
        self.eviction_policy = eviction_policy or FIFOTrainingExampleEvictionPolicy()
        self._h0: list[JudgedPairExample] = []
        self._h1: list[JudgedPairExample] = []

    def add(self, example: JudgedPairExample) -> None:
        if example.decision.label == JudgeLabel.UNCERTAIN:
            return
        bucket = self._h1 if example.decision.label == JudgeLabel.REUSABLE else self._h0
        capacity = self.max_h1 if example.decision.label == JudgeLabel.REUSABLE else self.max_h0
        if len(bucket) >= capacity:
            idx = self.eviction_policy.choose_eviction_index(
                label=example.decision.label,
                existing=tuple(bucket),
                incoming=example,
            )
            if idx < 0 or idx >= len(bucket):
                raise IndexError(f"eviction index out of range: {idx}")
            bucket.pop(idx)
        bucket.append(example)

    def h0(self) -> tuple[JudgedPairExample, ...]:
        return tuple(self._h0)

    def h1(self) -> tuple[JudgedPairExample, ...]:
        return tuple(self._h1)
