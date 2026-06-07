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


class SplitJudgeTrainingStore(JudgeTrainingStore):
    """Judge training store with disjoint train and calibration buckets."""

    def add(self, example: JudgedPairExample) -> None:
        self.add_train(example)

    @abstractmethod
    def add_train(self, example: JudgedPairExample) -> None:
        raise NotImplementedError

    @abstractmethod
    def add_calibration(self, example: JudgedPairExample) -> None:
        raise NotImplementedError

    @abstractmethod
    def h0_train(self) -> tuple[JudgedPairExample, ...]:
        raise NotImplementedError

    @abstractmethod
    def h1_train(self) -> tuple[JudgedPairExample, ...]:
        raise NotImplementedError

    @abstractmethod
    def h0_calibration(self) -> tuple[JudgedPairExample, ...]:
        raise NotImplementedError

    @abstractmethod
    def h1_calibration(self) -> tuple[JudgedPairExample, ...]:
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


class InMemorySplitJudgeTrainingStore(SplitJudgeTrainingStore):
    """Bounded in-memory H0/H1 stores split into train and calibration buckets."""

    def __init__(
        self,
        *,
        max_h0_train: int,
        max_h1_train: int,
        max_h0_calibration: int,
        max_h1_calibration: int,
        eviction_policy: TrainingExampleEvictionPolicy | None = None,
    ) -> None:
        capacities = {
            "max_h0_train": max_h0_train,
            "max_h1_train": max_h1_train,
            "max_h0_calibration": max_h0_calibration,
            "max_h1_calibration": max_h1_calibration,
        }
        for name, value in capacities.items():
            if int(value) <= 0:
                raise ValueError(f"{name} must be positive")

        self.max_h0_train = int(max_h0_train)
        self.max_h1_train = int(max_h1_train)
        self.max_h0_calibration = int(max_h0_calibration)
        self.max_h1_calibration = int(max_h1_calibration)
        self.eviction_policy = eviction_policy or FIFOTrainingExampleEvictionPolicy()
        self._h0_train: list[JudgedPairExample] = []
        self._h1_train: list[JudgedPairExample] = []
        self._h0_calibration: list[JudgedPairExample] = []
        self._h1_calibration: list[JudgedPairExample] = []

    def add_train(self, example: JudgedPairExample) -> None:
        if example.decision.label == JudgeLabel.UNCERTAIN:
            return
        if self._is_in_split(example, split="calibration"):
            return
        if example.decision.label == JudgeLabel.REUSABLE:
            self._add_to_bucket(
                bucket=self._h1_train,
                capacity=self.max_h1_train,
                label=JudgeLabel.REUSABLE,
                example=example,
            )
        else:
            self._add_to_bucket(
                bucket=self._h0_train,
                capacity=self.max_h0_train,
                label=JudgeLabel.NOT_REUSABLE,
                example=example,
            )

    def add_calibration(self, example: JudgedPairExample) -> None:
        if example.decision.label == JudgeLabel.UNCERTAIN:
            return
        if self._is_in_split(example, split="train"):
            return
        if example.decision.label == JudgeLabel.REUSABLE:
            self._add_to_bucket(
                bucket=self._h1_calibration,
                capacity=self.max_h1_calibration,
                label=JudgeLabel.REUSABLE,
                example=example,
            )
        else:
            self._add_to_bucket(
                bucket=self._h0_calibration,
                capacity=self.max_h0_calibration,
                label=JudgeLabel.NOT_REUSABLE,
                example=example,
            )

    def h0(self) -> tuple[JudgedPairExample, ...]:
        return self.h0_train() + self.h0_calibration()

    def h1(self) -> tuple[JudgedPairExample, ...]:
        return self.h1_train() + self.h1_calibration()

    def h0_train(self) -> tuple[JudgedPairExample, ...]:
        return tuple(self._h0_train)

    def h1_train(self) -> tuple[JudgedPairExample, ...]:
        return tuple(self._h1_train)

    def h0_calibration(self) -> tuple[JudgedPairExample, ...]:
        return tuple(self._h0_calibration)

    def h1_calibration(self) -> tuple[JudgedPairExample, ...]:
        return tuple(self._h1_calibration)

    def _add_to_bucket(
        self,
        *,
        bucket: list[JudgedPairExample],
        capacity: int,
        label: JudgeLabel,
        example: JudgedPairExample,
    ) -> None:
        if len(bucket) >= capacity:
            idx = self.eviction_policy.choose_eviction_index(
                label=label,
                existing=tuple(bucket),
                incoming=example,
            )
            if idx < 0 or idx >= len(bucket):
                raise IndexError(f"eviction index out of range: {idx}")
            bucket.pop(idx)
        bucket.append(example)

    def _is_in_split(self, example: JudgedPairExample, *, split: str) -> bool:
        key = self._example_key(example)
        buckets = (
            (self._h0_train, self._h1_train)
            if split == "train"
            else (self._h0_calibration, self._h1_calibration)
        )
        return any(self._example_key(existing) == key for bucket in buckets for existing in bucket)

    @staticmethod
    def _example_key(example: JudgedPairExample) -> tuple[str, str | None, tuple[float, ...], str]:
        candidate_key = example.request.candidate_key
        return (
            str(example.request.query),
            str(candidate_key) if candidate_key is not None else None,
            tuple(float(value) for value in example.features),
            example.decision.label.value,
        )
