"""Judge interfaces."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable

from mlcache.feedback.types import JudgeRequest, JudgeResult


class SemanticReuseJudge(ABC):
    """Judges whether a cached response is reusable for a new query."""

    @property
    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def judge(self, request: JudgeRequest) -> JudgeResult:
        raise NotImplementedError

    def judge_batch(self, requests: Iterable[JudgeRequest]) -> list[JudgeResult]:
        return [self.judge(request) for request in requests]
