"""Build query-level calibration records from top-k runtime observations."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import Any

from mlcache.calibration.query_level import QueryCalibrationCandidate, QueryCalibrationRecord
from mlcache.retrieval import VectorSearchResult
from mlcache.semantic_types import CacheKey, CacheLookup, Score


class QueryCalibrationRecordBuilder:
    """Converts one request's top-k candidates into a query calibration record."""

    def __init__(self, *, source: str = "runtime_top_k") -> None:
        if not source:
            raise ValueError("source must be non-empty")
        self.source = str(source)

    def build_record(
        self,
        *,
        query_id: str,
        request: CacheLookup,
        candidates: Sequence[VectorSearchResult],
        candidate_scores: Mapping[CacheKey, Score] | None = None,
        candidate_labels: Mapping[CacheKey, int | None] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> QueryCalibrationRecord:
        if not query_id:
            raise ValueError("query_id must be non-empty")

        built_candidates = []
        for rank, candidate in enumerate(candidates, start=1):
            explicit_score = self._lookup_by_cache_key(candidate_scores, candidate.cache_key)
            score = Score(float(explicit_score)) if explicit_score is not None else Score(float(candidate.score))
            label = self._lookup_by_cache_key(candidate_labels, candidate.cache_key)
            score_source = "scorer_score" if explicit_score is not None else "vector_score"
            built_candidates.append(
                QueryCalibrationCandidate(
                    score=score,
                    label=label,
                    candidate_rank=rank,
                    candidate_key=candidate.cache_key,
                    metadata={
                        "source": self.source,
                        "vector_score": float(candidate.score),
                        "score_source": score_source,
                        "candidate_metadata": candidate.metadata,
                        **({"scorer_score": float(explicit_score)} if explicit_score is not None else {}),
                    },
                )
            )

        record_metadata = {
            "source": self.source,
            "namespace": request.namespace,
            "request_metadata": request.metadata,
        }
        if metadata:
            record_metadata.update(metadata)

        return QueryCalibrationRecord(
            query_id=str(query_id),
            query=request.query,
            candidates=tuple(built_candidates),
            metadata=record_metadata,
        )

    @staticmethod
    def copy_record(record: QueryCalibrationRecord) -> QueryCalibrationRecord:
        return QueryCalibrationRecord(
            query_id=str(record.query_id),
            query=record.query,
            candidates=tuple(
                replace(candidate, metadata=dict(candidate.metadata))
                for candidate in record.candidates
            ),
            metadata=dict(record.metadata),
        )

    @staticmethod
    def _lookup_by_cache_key(
        mapping: Mapping[CacheKey, Any] | None,
        cache_key: CacheKey,
    ) -> Any | None:
        if mapping is None:
            return None
        if cache_key in mapping:
            return mapping[cache_key]
        key_as_string = str(cache_key)
        for key, value in mapping.items():
            if str(key) == key_as_string:
                return value
        return None


__all__ = ["QueryCalibrationRecordBuilder"]
