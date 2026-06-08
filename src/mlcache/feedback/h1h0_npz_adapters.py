"""Adapters for H1/H0 NPZ datasets."""

from __future__ import annotations

import hashlib
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from mlcache.feedback.judges import SemanticReuseJudge
from mlcache.feedback.types import JudgeDecision, JudgeLabel, JudgeRequest, JudgeResult
from mlcache.semantic_types import CacheEntry, CacheKey, CacheLookup, CacheMetadata, Query, Response, Score


@dataclass(frozen=True, slots=True)
class H1H0NPZRecord:
    row_id: int
    query_id: str
    query: Query
    query_embedding: tuple[float, ...]
    anchor_key: CacheKey
    anchor_query: Query
    anchor_embedding: tuple[float, ...]
    label: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class H1H0NPZSchema:
    query_field: str
    anchor_field: str
    label_field: str
    query_embedding_field: str
    anchor_embedding_field: str | None
    available_fields: tuple[str, ...]
    row_count: int
    metadata: dict[str, Any] = field(default_factory=dict)


class H1H0NPZDataset:
    """Loads and validates H1/H0 pair data from an NPZ file."""

    QUERY_FIELD_CANDIDATES = ("query", "text", "query_text", "request", "prompt")
    ANCHOR_FIELD_CANDIDATES = (
        "anchor",
        "anchor_query",
        "anchor_text",
        "candidate",
        "candidate_query",
        "nearest_anchor",
        "global_cluster",
    )
    LABEL_FIELD_CANDIDATES = ("h0h1", "label")
    QUERY_EMBEDDING_FIELD_CANDIDATES = ("query_embedding", "query_emb", "embedding", "emb", "x", "query_vec")
    ANCHOR_EMBEDDING_FIELD_CANDIDATES = (
        "anchor_embedding",
        "anchor_emb",
        "candidate_embedding",
        "candidate_emb",
        "anchor_vec",
    )
    ANCHOR_ID_CANDIDATES = ("anchor_id", "candidate_id", "nearest_anchor_id", "global_cluster")

    def __init__(
        self,
        path: str | Path,
        *,
        query_field: str | None = None,
        anchor_field: str | None = None,
        label_field: str = "h0h1",
        query_embedding_field: str | None = None,
        anchor_embedding_field: str | None = None,
        max_rows: int | None = None,
        allow_pickle: bool = True,
    ) -> None:
        self.path = Path(path)
        self.max_rows = max_rows
        self.allow_pickle = allow_pickle
        self._data = self._load_npz()
        try:
            self.available_fields = tuple(self._data.files)
            self._schema = self._detect_schema(
                query_field=query_field,
                anchor_field=anchor_field,
                label_field=label_field,
                query_embedding_field=query_embedding_field,
                anchor_embedding_field=anchor_embedding_field,
            )
            self._materialize_selected_rows()
            self._records = self._build_records()
        finally:
            self.close()

    @property
    def schema(self) -> H1H0NPZSchema:
        return self._schema

    def records(self) -> tuple[H1H0NPZRecord, ...]:
        return tuple(self._records)

    def schema_report(self) -> dict[str, Any]:
        labels: dict[str, int] = {}
        for record in self._records:
            labels[str(record.label)] = labels.get(str(record.label), 0) + 1
        unique_anchors = len({record.anchor_key for record in self._records})
        embedding_dim = len(self._records[0].query_embedding) if self._records else None
        return {
            "path": str(self.path),
            "available_fields": list(self._schema.available_fields),
            "chosen_fields": {
                "query_field": self._schema.query_field,
                "anchor_field": self._schema.anchor_field,
                "label_field": self._schema.label_field,
                "query_embedding_field": self._schema.query_embedding_field,
                "anchor_embedding_field": self._schema.anchor_embedding_field,
            },
            "row_count": self._schema.row_count,
            "embedding_dim": embedding_dim,
            "label_distribution": labels,
            "unique_anchors": unique_anchors,
            "metadata": dict(self._schema.metadata),
        }

    def close(self) -> None:
        data = getattr(self, "_data", None)
        if data is not None:
            data.close()
            self._data = None

    def _load_npz(self):
        try:
            import numpy as np
        except Exception as exc:
            raise ImportError("H1H0NPZDataset requires numpy") from exc
        return np.load(self.path, allow_pickle=self.allow_pickle)

    def _detect_schema(
        self,
        *,
        query_field: str | None,
        anchor_field: str | None,
        label_field: str,
        query_embedding_field: str | None,
        anchor_embedding_field: str | None,
    ) -> H1H0NPZSchema:
        label = self._resolve_label_field(label_field)
        query = self._field_or_detect(query_field, self.QUERY_FIELD_CANDIDATES, "query text")
        anchor = self._resolve_anchor_field(anchor_field, query_field=query)
        query_embedding = self._field_or_detect(
            query_embedding_field,
            self.QUERY_EMBEDDING_FIELD_CANDIDATES,
            "query embedding",
        )
        anchor_embedding = self._resolve_anchor_embedding_field(
            anchor_embedding_field,
            anchor_field=anchor,
            query_embedding_field=query_embedding,
        )

        row_count = self._selected_row_count(query)
        return H1H0NPZSchema(
            query_field=query,
            anchor_field=anchor,
            label_field=label,
            query_embedding_field=query_embedding,
            anchor_embedding_field=anchor_embedding,
            available_fields=self.available_fields,
            row_count=row_count,
            metadata={"source": "h1h0_npz", "max_rows": self.max_rows},
        )

    def _explicit_field(self, field_name: str, purpose: str) -> str:
        if field_name not in self.available_fields:
            raise ValueError(
                f"NPZ missing required {purpose}={field_name!r}; available fields: {self.available_fields}"
            )
        return field_name

    def _resolve_label_field(self, explicit_label_field: str) -> str:
        if explicit_label_field in self.available_fields:
            return explicit_label_field
        # Backward-compatible fallback for datasets that use "label" instead of "h0h1".
        if explicit_label_field == "h0h1" and "label" in self.available_fields:
            return "label"
        return self._explicit_field(explicit_label_field, "label_field")

    def _resolve_anchor_field(self, explicit_anchor_field: str | None, *, query_field: str) -> str:
        if explicit_anchor_field is not None:
            return self._explicit_field(explicit_anchor_field, "anchor text")
        matches = tuple(field for field in self.ANCHOR_FIELD_CANDIDATES if field in self.available_fields)
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            raise ValueError(
                f"Ambiguous anchor text fields {matches}; provide an explicit field name. "
                f"Available fields: {self.available_fields}"
            )
        # If no dedicated anchor field exists, reuse the query field.
        if query_field in self.available_fields:
            return query_field
        raise ValueError(f"Could not detect anchor text field; available fields: {self.available_fields}")

    def _resolve_anchor_embedding_field(
        self,
        explicit_anchor_embedding_field: str | None,
        *,
        anchor_field: str,
        query_embedding_field: str,
    ) -> str | None:
        if explicit_anchor_embedding_field is not None:
            return self._explicit_field(explicit_anchor_embedding_field, "anchor embedding")
        matches = tuple(field for field in self.ANCHOR_EMBEDDING_FIELD_CANDIDATES if field in self.available_fields)
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            raise ValueError(
                f"Ambiguous anchor embedding fields {matches}; provide an explicit field name. "
                f"Available fields: {self.available_fields}"
            )
        if anchor_field == "global_cluster":
            return None
        # If no dedicated anchor embedding field exists, reuse query embeddings.
        if query_embedding_field in self.available_fields:
            return query_embedding_field
        raise ValueError(f"Could not detect anchor embedding field; available fields: {self.available_fields}")

    def _field_or_detect(self, explicit: str | None, candidates: Sequence[str], purpose: str) -> str:
        if explicit is not None:
            return self._explicit_field(explicit, purpose)
        matches = tuple(field for field in candidates if field in self.available_fields)
        if len(matches) == 1:
            return matches[0]
        if not matches:
            raise ValueError(f"Could not detect {purpose} field; available fields: {self.available_fields}")
        raise ValueError(
            f"Ambiguous {purpose} fields {matches}; provide an explicit field name. "
            f"Available fields: {self.available_fields}"
        )

    def _selected_row_count(self, query_field: str) -> int:
        total = len(self._data[query_field])
        if self.max_rows is None:
            return int(total)
        if int(self.max_rows) < 0:
            raise ValueError("max_rows must be non-negative when set")
        return int(min(int(self.max_rows), total))

    def _materialize_selected_rows(self) -> None:
        schema = self._schema
        limit = schema.row_count
        loaded: dict[str, Any] = {}

        def head(field_name: str) -> Any:
            if field_name not in loaded:
                loaded[field_name] = self._data[field_name][:limit]
            return loaded[field_name]

        self._query_values = head(schema.query_field)
        self._anchor_values = head(schema.anchor_field)
        self._labels = head(schema.label_field)
        self._query_embeddings = head(schema.query_embedding_field)
        self._anchor_embeddings = (
            self._query_embeddings
            if schema.anchor_embedding_field is None
            else head(schema.anchor_embedding_field)
        )

        for field_name, array in (
            (schema.query_field, self._query_values),
            (schema.anchor_field, self._anchor_values),
            (schema.label_field, self._labels),
            (schema.query_embedding_field, self._query_embeddings),
        ):
            if len(array) < limit:
                raise ValueError(f"Field {field_name!r} has {len(array)} rows but expected at least {limit}")
        if len(self._anchor_embeddings) < limit:
            raise ValueError(f"Anchor embeddings have {len(self._anchor_embeddings)} rows but expected at least {limit}")

        if getattr(self._query_embeddings, "ndim", None) != 2:
            raise ValueError(f"Query embedding field {schema.query_embedding_field!r} must be a 2D array")
        if getattr(self._anchor_embeddings, "ndim", None) != 2:
            raise ValueError("Anchor embeddings must be a 2D array")
        if self._query_embeddings.shape[1] != self._anchor_embeddings.shape[1]:
            raise ValueError(
                "query and anchor embeddings must have the same dimension, "
                f"got {self._query_embeddings.shape[1]} and {self._anchor_embeddings.shape[1]}"
            )

        self._validate_labels(self._labels)
        self._validate_nonzero_embeddings(self._query_embeddings, schema.query_embedding_field)
        self._validate_nonzero_embeddings(self._anchor_embeddings, schema.anchor_embedding_field or schema.query_embedding_field)

    @staticmethod
    def _validate_labels(labels: Any) -> None:
        bad = sorted({int(value) for value in labels if int(value) not in {-1, 0, 1}})
        if bad:
            raise ValueError(f"H1/H0 labels must be -1, 0, or 1; found invalid labels: {bad}")

    @staticmethod
    def _validate_nonzero_embeddings(embeddings: Any, field_name: str) -> None:
        import numpy as np

        norms = np.linalg.norm(embeddings, axis=1)
        if bool(np.any(norms <= 0.0)):
            raise ValueError(f"Embedding field {field_name!r} contains zero vectors")

    def _build_records(self) -> tuple[H1H0NPZRecord, ...]:
        schema = self._schema
        anchor_id_field = self._detect_anchor_id_field()
        records = []
        for row_id in range(schema.row_count):
            query = Query(str(self._query_values[row_id]))
            anchor_value = self._anchor_values[row_id]
            label = int(self._labels[row_id])
            anchor_key = self._anchor_key(anchor_value, anchor_id_field=anchor_id_field)
            anchor_query = Query(str(anchor_key) if schema.anchor_field == "global_cluster" else str(anchor_value))
            records.append(
                H1H0NPZRecord(
                    row_id=row_id,
                    query_id=f"row:{row_id}",
                    query=query,
                    query_embedding=tuple(float(value) for value in self._query_embeddings[row_id]),
                    anchor_key=anchor_key,
                    anchor_query=anchor_query,
                    anchor_embedding=tuple(float(value) for value in self._anchor_embeddings[row_id]),
                    label=label,
                    metadata={
                        "row_id": row_id,
                        "source": "h1h0_npz",
                        "query_field": schema.query_field,
                        "anchor_field": schema.anchor_field,
                        "label_field": schema.label_field,
                        "query_embedding_field": schema.query_embedding_field,
                        "anchor_embedding_field": schema.anchor_embedding_field,
                        "raw_label": label,
                        **({"anchor_id_field": anchor_id_field} if anchor_id_field is not None else {}),
                    },
                )
            )
        return tuple(records)

    def _detect_anchor_id_field(self) -> str | None:
        if self._schema.anchor_field == "global_cluster":
            return "global_cluster"
        matches = tuple(field for field in self.ANCHOR_ID_CANDIDATES if field in self.available_fields)
        return matches[0] if len(matches) == 1 else None

    @staticmethod
    def _anchor_key(anchor_value: Any, *, anchor_id_field: str | None) -> CacheKey:
        if anchor_id_field is not None:
            return CacheKey(f"anchor:{anchor_value}")
        return CacheKey(f"anchor:{_stable_hash(str(anchor_value))}")


class H1H0NPZStreamAdapter:
    """Converts H1/H0 NPZ records into cache stream inputs."""

    def __init__(self, dataset: H1H0NPZDataset) -> None:
        self.dataset = dataset

    def records(self) -> tuple[H1H0NPZRecord, ...]:
        return self.dataset.records()

    def anchor_entries(self, *, response_template: str = "cached_response_for:{anchor_key}") -> tuple[CacheEntry, ...]:
        anchor_rows: dict[CacheKey, list[H1H0NPZRecord]] = {}
        for record in self.records():
            anchor_rows.setdefault(record.anchor_key, []).append(record)

        anchors: dict[CacheKey, CacheEntry] = {}
        for anchor_key, records in anchor_rows.items():
            first = records[0]
            response = response_template.format(anchor_key=str(anchor_key))
            anchors[anchor_key] = CacheEntry(
                cache_key=anchor_key,
                query=first.anchor_query,
                response=Response(response),
                embedding=self._mean_embedding(records),
                metadata=CacheMetadata(
                    attributes={
                        "anchor_key": str(anchor_key),
                        "source": "h1h0_npz_anchor",
                        "anchor_observation_count": len(records),
                    }
                ),
            )
        return tuple(anchors[key] for key in sorted(anchors, key=str))

    @staticmethod
    def _mean_embedding(records: Sequence[H1H0NPZRecord]) -> tuple[float, ...]:
        if not records:
            return ()
        dim = len(records[0].anchor_embedding)
        sums = [0.0 for _ in range(dim)]
        for record in records:
            for idx, value in enumerate(record.anchor_embedding):
                sums[idx] += float(value)
        count = float(len(records))
        return tuple(value / count for value in sums)

    def lookups(self) -> Iterator[CacheLookup]:
        for _, lookup in self.iter_records_and_lookups():
            yield lookup

    def iter_records_and_lookups(self) -> Iterator[tuple[H1H0NPZRecord, CacheLookup]]:
        for record in self.records():
            yield record, CacheLookup(
                query=record.query,
                embedding=record.query_embedding,
                metadata=CacheMetadata(
                    attributes={
                        "query_id": record.query_id,
                        "row_id": record.row_id,
                        "expected_anchor_key": str(record.anchor_key),
                        "h0h1": record.label,
                        "source": "h1h0_npz",
                    }
                ),
            )


class H1H0NPZJudgeAdapter(SemanticReuseJudge):
    """Deterministic judge backed by H1/H0 labels."""

    def __init__(self, records: Sequence[H1H0NPZRecord]) -> None:
        self._by_query_id_anchor: dict[tuple[str, str], int] = {}
        self._by_query_text_anchor_key: dict[tuple[str, str], int] = {}
        self._by_query_anchor_text: dict[tuple[str, str], int] = {}
        for record in records:
            anchor_key = str(record.anchor_key)
            self._by_query_id_anchor[(record.query_id, anchor_key)] = record.label
            self._by_query_text_anchor_key[(str(record.query), anchor_key)] = record.label
            self._by_query_anchor_text[(str(record.query), str(record.anchor_query))] = record.label

    @property
    def name(self) -> str:
        return "h1h0_npz_judge"

    def judge(self, request: JudgeRequest) -> JudgeResult:
        label = self._lookup_label(request)
        if label is None:
            return JudgeResult(
                request=request,
                decision=JudgeDecision(
                    label=JudgeLabel.UNCERTAIN,
                    confidence=Score(0.0),
                    rationale="h1h0_pair_not_found",
                ),
            )
        if label == 1:
            return JudgeResult(
                request=request,
                decision=JudgeDecision(
                    label=JudgeLabel.REUSABLE,
                    confidence=Score(1.0),
                    rationale="h1h0_label_reusable",
                ),
            )
        if label == -1:
            return JudgeResult(
                request=request,
                decision=JudgeDecision(
                    label=JudgeLabel.UNCERTAIN,
                    confidence=Score(0.0),
                    rationale="h1h0_label_unknown",
                ),
            )
        return JudgeResult(
            request=request,
            decision=JudgeDecision(
                label=JudgeLabel.NOT_REUSABLE,
                confidence=Score(1.0),
                rationale="h1h0_label_not_reusable",
            ),
        )

    def _lookup_label(self, request: JudgeRequest) -> int | None:
        candidate_key = str(request.candidate_key) if request.candidate_key is not None else None
        query_id = self._query_id_from_request(request)
        if query_id is not None and candidate_key is not None:
            label = self._by_query_id_anchor.get((query_id, candidate_key))
            if label is not None:
                return label
        query_text = str(request.query)
        if candidate_key is not None:
            label = self._by_query_text_anchor_key.get((query_text, candidate_key))
            if label is not None:
                return label
        if request.candidate_query is not None:
            return self._by_query_anchor_text.get((query_text, str(request.candidate_query)))
        return None

    @staticmethod
    def _query_id_from_request(request: JudgeRequest) -> str | None:
        context = request.context
        query_id = context.get("query_id") or context.get("request_id")
        if query_id is not None:
            return str(query_id)
        request_metadata = context.get("request_metadata")
        attributes = getattr(request_metadata, "attributes", None)
        if isinstance(attributes, dict):
            query_id = attributes.get("query_id") or attributes.get("request_id")
            if query_id is not None:
                return str(query_id)
        return None


def _stable_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]


__all__ = [
    "H1H0NPZDataset",
    "H1H0NPZJudgeAdapter",
    "H1H0NPZRecord",
    "H1H0NPZSchema",
    "H1H0NPZStreamAdapter",
]
