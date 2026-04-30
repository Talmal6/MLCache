"""Base scorer interface and shared scorer behavior."""

from __future__ import annotations

from abc import ABC, abstractmethod

from semantic_desider.features import PairFeatures
from semantic_desider.thresholds import ThresholdCalibrationRequest
from semantic_desider.types import InputSpace, LabeledPairBatch, ScorerName, Score, Threshold, TieMode


class SemanticScorer(ABC):
    """Scores whether a cached response is semantically correct for a query."""

    @property
    @abstractmethod
    def name(self) -> ScorerName:
        raise NotImplementedError

    @property
    @abstractmethod
    def input_space(self) -> InputSpace:
        raise NotImplementedError

    @abstractmethod
    def fit(self, batch: LabeledPairBatch, **kwargs: object) -> None:
        raise NotImplementedError

    @abstractmethod
    def score(self, features: PairFeatures) -> Score:
        raise NotImplementedError

    @abstractmethod
    def calibrate(self, request: ThresholdCalibrationRequest) -> Threshold:
        raise NotImplementedError

    @abstractmethod
    def predict(
        self,
        features: PairFeatures,
        threshold: Threshold,
        *,
        tie_mode: TieMode = TieMode.GE,
    ) -> bool:
        raise NotImplementedError


class BaseScorer(SemanticScorer):
    """Shared implementation for calibration and threshold prediction."""

    _name: ScorerName
    _input_space: InputSpace = InputSpace.EMBEDDING

    @property
    def name(self) -> ScorerName:
        return self._name

    @property
    def input_space(self) -> InputSpace:
        return self._input_space

    def calibrate(self, request: ThresholdCalibrationRequest) -> Threshold:
        from semantic_desider.scorers.utils import scores_to_threshold

        return scores_to_threshold(request)

    def predict(
        self,
        features: PairFeatures,
        threshold: Threshold,
        *,
        tie_mode: TieMode = TieMode.GE,
    ) -> bool:
        score = float(self.score(features))
        tau = float(threshold)
        if tie_mode == TieMode.GT:
            return score > tau
        return score >= tau


class ScorerRegistry(ABC):
    """Registry of available semantic scorers."""

    @abstractmethod
    def register(self, scorer: SemanticScorer) -> None:
        raise NotImplementedError

    @abstractmethod
    def get(self, name: ScorerName) -> SemanticScorer:
        raise NotImplementedError

    @abstractmethod
    def default(self) -> SemanticScorer:
        raise NotImplementedError

    @abstractmethod
    def list(self) -> list[ScorerName]:
        raise NotImplementedError
