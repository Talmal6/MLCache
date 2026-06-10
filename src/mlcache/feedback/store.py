"""Judge-labeled training stores."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Sequence

from mlcache.feedback.types import JudgeDecision, JudgeLabel, JudgeRequest


@dataclass(frozen=True, slots=True)
class JudgedPairExample:
    features: tuple[float, ...]
    request: JudgeRequest
    decision: JudgeDecision
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
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
        self._h0_train_keys: list[tuple] = []
        self._h1_train_keys: list[tuple] = []
        self._h0_calibration_keys: list[tuple] = []
        self._h1_calibration_keys: list[tuple] = []
        self._train_keys: Counter[tuple] = Counter()
        self._calibration_keys: Counter[tuple] = Counter()

    def add_train(self, example: JudgedPairExample) -> None:
        if example.decision.label == JudgeLabel.UNCERTAIN:
            return
        key = self._example_key(example)
        if key in self._calibration_keys:
            return
        if example.decision.label == JudgeLabel.REUSABLE:
            self._add_to_bucket(
                bucket=self._h1_train,
                bucket_keys=self._h1_train_keys,
                split_keys=self._train_keys,
                capacity=self.max_h1_train,
                label=JudgeLabel.REUSABLE,
                example=example,
                key=key,
            )
        else:
            self._add_to_bucket(
                bucket=self._h0_train,
                bucket_keys=self._h0_train_keys,
                split_keys=self._train_keys,
                capacity=self.max_h0_train,
                label=JudgeLabel.NOT_REUSABLE,
                example=example,
                key=key,
            )

    def add_calibration(self, example: JudgedPairExample) -> None:
        if example.decision.label == JudgeLabel.UNCERTAIN:
            return
        key = self._example_key(example)
        if key in self._train_keys:
            return
        if example.decision.label == JudgeLabel.REUSABLE:
            self._add_to_bucket(
                bucket=self._h1_calibration,
                bucket_keys=self._h1_calibration_keys,
                split_keys=self._calibration_keys,
                capacity=self.max_h1_calibration,
                label=JudgeLabel.REUSABLE,
                example=example,
                key=key,
            )
        else:
            self._add_to_bucket(
                bucket=self._h0_calibration,
                bucket_keys=self._h0_calibration_keys,
                split_keys=self._calibration_keys,
                capacity=self.max_h0_calibration,
                label=JudgeLabel.NOT_REUSABLE,
                example=example,
                key=key,
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
        bucket_keys: list[tuple],
        split_keys: Counter[tuple],
        capacity: int,
        label: JudgeLabel,
        example: JudgedPairExample,
        key: tuple,
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
            evicted_key = bucket_keys.pop(idx)
            split_keys[evicted_key] -= 1
            if split_keys[evicted_key] <= 0:
                del split_keys[evicted_key]
        bucket.append(example)
        bucket_keys.append(key)
        split_keys[key] += 1

    @staticmethod
    def _example_key(example: JudgedPairExample) -> tuple[str, str | None, tuple[float, ...], str]:
        candidate_key = example.request.candidate_key
        return (
            str(example.request.query),
            str(candidate_key) if candidate_key is not None else None,
            tuple(float(value) for value in example.features),
            example.decision.label.value,
        )
