"""High-level one-call construction of an MLCache runtime.

`MLCache.from_preset` (and the `build_mlcache` factory function it wraps) is the
normal way to assemble a working cache: it owns wiring the feature builder,
scorer(s), threshold, and persistence that would otherwise need to be composed
by hand via `build_local_mlcache_runtime`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from mlcache.calibration import ThresholdScope
from mlcache.feedback import SemanticReuseJudge
from mlcache.features import NormalizedHadamardFeatureBuilder, PairFeatureBuilder
from mlcache.runtime.config import MLCacheRuntimeConfig, RuntimeRefitConfig
from mlcache.runtime.local import build_local_mlcache_runtime
from mlcache.runtime.runtime import MLCacheRuntime
from mlcache.scorers import (
    CosineScorer,
    EnsembleScorer,
    LDAScorer,
    PCAWhitenedCosineScorer,
    SemanticScorer,
    TinyMLPScorer,
    XGBoostScorer,
)
from mlcache.semantic_types import LabeledPairBatch, Threshold

ML_DEPENDENCIES_ERROR = "Install ML dependencies with: pip install -e '.[ml,dev]'"
EMBEDDINGS_DEPENDENCIES_ERROR = "Install embedding dependencies with: pip install -e '.[embeddings,dev]'"

DEFAULT_ENSEMBLE_MEMBERS: tuple[str, ...] = ("cosine", "lda", "pca_whitened_cosine", "xgboost", "mlp")


def _cosine_factory(**kwargs: object) -> SemanticScorer:
    return CosineScorer(**kwargs)


def _lda_factory(**kwargs: object) -> SemanticScorer:
    return LDAScorer(**kwargs)


def _pca_whitened_cosine_factory(**kwargs: object) -> SemanticScorer:
    return PCAWhitenedCosineScorer(**kwargs)


def _xgboost_factory(**kwargs: object) -> SemanticScorer:
    return XGBoostScorer(**kwargs)


def _mlp_factory(**kwargs: object) -> SemanticScorer:
    return TinyMLPScorer(**kwargs)


_SCORER_FACTORIES: dict[str, Any] = {
    "cosine": _cosine_factory,
    "lda": _lda_factory,
    "pca_whitened_cosine": _pca_whitened_cosine_factory,
    "xgboost": _xgboost_factory,
    "mlp": _mlp_factory,
}

SCORER_PRESET_NAMES: tuple[str, ...] = (*_SCORER_FACTORIES, "ensemble")


def _normalize_preset_name(name: str) -> str:
    return str(name).strip().lower().replace("-", "_")


def _build_single_scorer(name: str, **kwargs: object) -> SemanticScorer:
    key = _normalize_preset_name(name)
    factory = _SCORER_FACTORIES.get(key)
    if factory is None:
        raise ValueError(
            f"Unknown scorer preset {name!r}; available presets: {', '.join(SCORER_PRESET_NAMES)}"
        )
    return factory(**kwargs)


def build_scorer(
    scorer: str | SemanticScorer,
    *,
    scorers: Iterable[str] | None = None,
    **scorer_kwargs: object,
) -> SemanticScorer:
    """Build a scorer from a preset name, or pass an existing scorer through.

    `scorer="ensemble"` builds an `EnsembleScorer` whose members are given by
    `scorers` (defaulting to cosine, lda, pca_whitened_cosine, xgboost, mlp).
    """

    if isinstance(scorer, SemanticScorer):
        return scorer

    key = _normalize_preset_name(scorer)
    if key == "ensemble":
        member_names = tuple(scorers) if scorers is not None else DEFAULT_ENSEMBLE_MEMBERS
        if not member_names:
            raise ValueError("ensemble scorer preset requires at least one member in 'scorers'")
        judges = [_build_single_scorer(member_name) for member_name in member_names]
        return EnsembleScorer(judges=judges)

    return _build_single_scorer(key, **scorer_kwargs)


def _set_pair_threshold(runtime: MLCacheRuntime, *, scorer_name: Any, threshold: Threshold) -> None:
    runtime.oracle._threshold = threshold
    provider = runtime.oracle.threshold_provider
    if provider is not None:
        provider.set_threshold(
            threshold,
            scorer=scorer_name,
            scope=ThresholdScope.GLOBAL,
            context={"source": "mlcache_from_preset"},
        )


class MLCache:
    """Ergonomic facade around `MLCacheRuntime` built from a single constructor."""

    def __init__(self, runtime: MLCacheRuntime, *, scorer: SemanticScorer) -> None:
        self.runtime = runtime
        self.scorer = scorer

    @classmethod
    def from_preset(
        cls,
        *,
        root_dir: str | Path,
        scorer: str | SemanticScorer = "cosine",
        scorers: Iterable[str] | None = None,
        threshold: float | Threshold | None = None,
        top_k: int = 5,
        persistence: bool = True,
        feature_builder: PairFeatureBuilder | None = None,
        judge: SemanticReuseJudge | None = None,
        config: MLCacheRuntimeConfig | None = None,
        scorer_kwargs: dict[str, object] | None = None,
    ) -> "MLCache":
        """Build a ready-to-use `MLCache` without manually wiring its components.

        This is the normal way to construct the cache: it picks a feature
        builder, resolves the scorer preset(s), wires file or in-memory
        persistence, and applies the pair-level threshold in one call.
        """

        scorer_obj = build_scorer(scorer, scorers=scorers, **(scorer_kwargs or {}))
        builder = feature_builder or NormalizedHadamardFeatureBuilder()
        runtime_config = config or MLCacheRuntimeConfig(refit=RuntimeRefitConfig(auto_refit=False, top_k=int(top_k)))

        runtime = build_local_mlcache_runtime(
            root_dir=root_dir,
            feature_builder=builder,
            scorer=scorer_obj,
            judge=judge,
            config=runtime_config,
            use_file_persistence=bool(persistence),
        )

        if threshold is not None:
            _set_pair_threshold(runtime, scorer_name=scorer_obj.name, threshold=Threshold(float(threshold)))

        return cls(runtime, scorer=scorer_obj)

    def prefit(self, batch: LabeledPairBatch, **kwargs: object) -> None:
        """Fit the underlying scorer (and its ensemble members, if any) on labeled pairs."""

        try:
            self.scorer.fit(batch, **kwargs)
        except ImportError as exc:
            raise ImportError(ML_DEPENDENCIES_ERROR) from exc

    def index(self, entries: Iterable[Any]) -> None:
        """Index cache candidates (CacheEntry instances) ahead of serving lookups."""

        for entry in entries:
            self.runtime.put(entry)

    def put(self, entry: Any) -> Any:
        return self.runtime.put(entry)

    def lookup(self, lookup: Any) -> Any:
        return self.runtime.lookup(lookup)

    def lookup_with_decision(self, lookup: Any) -> Any:
        return self.runtime.lookup_with_decision(lookup)

    def invalidate(self, cache_key: Any) -> None:
        self.runtime.invalidate(cache_key)

    def set_threshold(self, threshold: float | Threshold) -> None:
        _set_pair_threshold(self.runtime, scorer_name=self.scorer.name, threshold=Threshold(float(threshold)))

    @property
    def threshold(self) -> Threshold | None:
        return self.runtime.threshold

    @property
    def components(self) -> dict[str, str | None]:
        return self.runtime.components


def build_mlcache(
    *,
    root_dir: str | Path,
    scorer: str | SemanticScorer = "cosine",
    scorers: Iterable[str] | None = None,
    threshold: float | Threshold | None = None,
    top_k: int = 5,
    persistence: bool = True,
    feature_builder: PairFeatureBuilder | None = None,
    judge: SemanticReuseJudge | None = None,
    config: MLCacheRuntimeConfig | None = None,
    scorer_kwargs: dict[str, object] | None = None,
) -> MLCache:
    """Factory-function form of `MLCache.from_preset`."""

    return MLCache.from_preset(
        root_dir=root_dir,
        scorer=scorer,
        scorers=scorers,
        threshold=threshold,
        top_k=top_k,
        persistence=persistence,
        feature_builder=feature_builder,
        judge=judge,
        config=config,
        scorer_kwargs=scorer_kwargs,
    )


__all__ = [
    "EMBEDDINGS_DEPENDENCIES_ERROR",
    "ML_DEPENDENCIES_ERROR",
    "MLCache",
    "SCORER_PRESET_NAMES",
    "build_mlcache",
    "build_scorer",
]
