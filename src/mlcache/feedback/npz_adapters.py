"""Adapters for reading a dataset stored in an .npz file.

This module provides:
- `NPZStreamAdapter`: iterate over records in an .npz file as judge requests
- `NPZJudgeAdapter`: an implementation of `SemanticReuseJudge` that uses
  a per-query `global_cluster` value to decide reuse (same cluster -> reusable).

The adapters are intentionally lightweight and configurable with the
names of the keys inside the `.npz` so they can be used with different
file layouts.
"""

from __future__ import annotations

import os
from typing import Iterator, Mapping

import numpy as np

from mlcache.feedback.judges import SemanticReuseJudge
from mlcache.feedback.types import JudgeDecision, JudgeLabel, JudgeRequest, JudgeResult
from mlcache.semantic_types import Query, CacheMetadata, Score


class NPZStreamAdapter:
    """Stream adapter that yields `JudgeRequest` objects from an `.npz` file.

    Parameters
    - path: path to the .npz file
    - query_key: key in the archive holding a textual query identifier (str)
    - cluster_key: key holding the global cluster / anchor value
    - label_key: optional key holding a boolean/int label (kept in metadata)
    - allow_pickle: passed to `np.load`
    """

    def __init__(
        self,
        path: str,
        *,
        query_key: str = "query",
        cluster_key: str = "global_cluster",
        label_key: str | None = "label",
        allow_pickle: bool = True,
    ) -> None:
        self.path = path
        self.query_key = query_key
        self.cluster_key = cluster_key
        self.label_key = label_key
        self.allow_pickle = allow_pickle

    def __iter__(self) -> Iterator[JudgeRequest]:
        p = os.fspath(self.path)
        with np.load(p, allow_pickle=self.allow_pickle) as d:
            # determine length from one of the arrays
            if self.query_key in d:
                length = len(d[self.query_key])
            elif self.cluster_key in d:
                length = len(d[self.cluster_key])
            else:
                raise ValueError("npz does not contain query nor cluster arrays")

            for i in range(length):
                q = None
                if self.query_key in d:
                    q = d[self.query_key][i]
                # normalize to string Query
                q_str = Query(str(q)) if q is not None else Query(str(i))
                metadata = CacheMetadata()
                if self.cluster_key in d:
                    metadata.attributes["global_cluster"] = d[self.cluster_key][i]
                if self.label_key and self.label_key in d:
                    metadata.attributes["label"] = d[self.label_key][i]

                yield JudgeRequest(query=q_str, metadata=metadata)

    def stream_requests(self) -> Iterator[JudgeRequest]:
        return iter(self)


class NPZJudgeAdapter(SemanticReuseJudge):
    """A SemanticReuseJudge backed by an .npz file.

    The adapter loads a mapping from stringified queries to a cluster id
    (from `cluster_key`). Two queries are considered `REUSABLE` when their
    cluster ids are equal. When a query is missing from the dataset the
    adapter returns `UNCERTAIN`.

    Parameters
    - path: path to the .npz file
    - query_key / cluster_key: names of the arrays inside the archive
    - preload: if True, load the index into memory at init (default True)
    """

    def __init__(
        self,
        path: str,
        *,
        query_key: str = "query",
        cluster_key: str = "global_cluster",
        allow_pickle: bool = True,
        preload: bool = True,
    ) -> None:
        self._path = os.fspath(path)
        self._query_key = query_key
        self._cluster_key = cluster_key
        self._allow_pickle = allow_pickle
        self._index: dict[str, object] = {}
        if preload:
            self._build_index()

    @property
    def name(self) -> str:
        return f"npz_judge:{os.path.basename(self._path)}"

    def _build_index(self) -> None:
        with np.load(self._path, allow_pickle=self._allow_pickle) as d:
            if self._query_key not in d or self._cluster_key not in d:
                # nothing to index
                return
            queries = d[self._query_key]
            clusters = d[self._cluster_key]
            if len(queries) != len(clusters):
                # still build index for min length
                n = min(len(queries), len(clusters))
            else:
                n = len(queries)
            self._index = {str(queries[i]): clusters[i] for i in range(n)}

    def _lookup_cluster(self, q: Query) -> object | None:
        # ensure index built lazily
        if not self._index:
            try:
                self._build_index()
            except Exception:
                return None
        return self._index.get(str(q))

    def judge(self, request: JudgeRequest) -> JudgeResult:
        # prefer explicit candidate_query; fall back to candidate_response or candidate_key
        q = request.query
        cq = request.candidate_query

        if cq is None:
            return JudgeResult(
                request,
                JudgeDecision(label=JudgeLabel.UNCERTAIN, rationale="no candidate_query"),
            )

        a = self._lookup_cluster(q)
        b = self._lookup_cluster(cq)

        if a is None or b is None:
            return JudgeResult(
                request,
                JudgeDecision(label=JudgeLabel.UNCERTAIN, rationale="query not found in npz"),
            )

        if a == b:
            decision = JudgeDecision(label=JudgeLabel.REUSABLE, confidence=Score(1.0))
        else:
            decision = JudgeDecision(label=JudgeLabel.NOT_REUSABLE, confidence=Score(1.0))

        return JudgeResult(request=request, decision=decision)
