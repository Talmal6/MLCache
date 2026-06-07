"""Pair feature construction contracts."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from mlcache.semantic_types import Embedding, Score


class PairFeatureKind(StrEnum):
    HADAMARD = "hadamard"
    ABS_DIFF = "abs_diff"
    CONCAT = "concat"
    COSINE = "cosine"


@dataclass(frozen=True, slots=True)
class PairFeatures:
    cosine: Score | None = None
    hadamard: tuple[float, ...] = ()
    abs_diff: tuple[float, ...] = ()
    concat: tuple[float, ...] = ()
    values: dict[str, Any] = field(default_factory=dict)


class PairFeatureBuilder(ABC):
    """Builds features for query-candidate semantic scoring."""

    @abstractmethod
    def build(self, query_embedding: Embedding, candidate_embedding: Embedding) -> PairFeatures:
        raise NotImplementedError

    @abstractmethod
    def default_kind(self) -> PairFeatureKind:
        raise NotImplementedError
