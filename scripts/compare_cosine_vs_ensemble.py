"""Dataset-backed ONLINE cosine-vs-full-ensemble comparison.

This script intentionally uses `SemanticCacheSystem` directly.

It is NOT an offline `run_cache.py` wrapper:
- no import of run_cache.py
- no cache.prefit_and_calibrate(...)
- no synthetic topics / generated prompts
- no hashing embeddings

Traffic comes from an H1/H0 NPZ dataset. Each prompt is encoded as:

    row:{row_id}\t{original_text}

The row id lets the dataset-backed embedding provider and judge recover the
exact NPZ row for both the incoming query and any cached candidate query.

Pairwise judge rule:
    same anchor/global_cluster AND both rows label == 1 -> REUSABLE / H1
    label == -1 on either side                         -> UNCERTAIN
    otherwise                                          -> NOT_REUSABLE / H0

Instrumentation (online, no offline prefit): console + run.log logging,
per-method progress CSVs with running FPR/TPR/precision, a lifecycle JSONL,
and final per-method + comparison summaries. Accept/reject labels are exact:
the dataset judge records every judged pair, and the serve loop joins each
judged candidate against the SystemResponse accepted cache_key (shadow
collection runs synchronously inside handle(), so all judge calls for a
request have completed by the time handle() returns).
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import shutil
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock
from typing import Any, Iterable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mlcache import (  # noqa: E402
    JudgeDecision,
    JudgeLabel,
    JudgeRequest,
    JudgeResult,
    MockLLM,
    SemanticCacheSystem,
    SemanticReuseJudge,
)
from mlcache.embeddings import EmbeddingProvider  # noqa: E402
from mlcache.persistence import json_safe  # noqa: E402


FULL_ENSEMBLE = ["cosine", "lda", "pca_whitened_cosine", "xgboost", "mlp"]

EXPECTED_EMB_DIM_FOR_EMB_FIELD = 1024

PROGRESS_CSV_FIELDS = [
    "method",
    "request_index",
    "total_requests",
    "hits",
    "misses",
    "hit_rate",
    "empirical_fpr",
    "tpr",
    "precision",
    "true_accepts",
    "false_accepts",
    "true_rejects",
    "false_rejects",
    "trained",
    "calibrated",
    "threshold",
    "finite_threshold",
    "scorer_version",
    "threshold_version",
    "judged_pair_count",
    "training_store_h0",
    "training_store_h1",
    "first_calibrated_at_request",
    "first_hit_at_request",
]


# ── Phase 2: minimal instrumentation helpers ───────────────────────────────

def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def setup_logging(output_dir: str | Path) -> logging.Logger:
    """Create a logger that writes ASCII-only timestamped lines to stdout and run.log."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "run.log"

    logger = logging.getLogger(f"compare_cosine_vs_ensemble::{log_path}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    # Reset handlers so repeated calls (e.g. tests) do not duplicate output.
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        try:
            handler.close()
        except Exception:
            pass

    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s", datefmt="%Y-%m-%dT%H:%M:%S")
    file_handler = logging.FileHandler(log_path, mode="w", encoding="ascii", errors="backslashreplace")
    file_handler.setFormatter(fmt)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(fmt)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    logger.info("logging initialized; run.log at %s", str(log_path))
    flush_logger(logger)
    return logger


def flush_logger(logger: logging.Logger) -> None:
    for handler in logger.handlers:
        try:
            handler.flush()
        except Exception:
            pass


def write_json(path: str | Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(payload), indent=2, sort_keys=True), encoding="utf-8")


def write_csv_rows(path: str | Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def append_jsonl(path: str | Path, row: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(json_safe(row), sort_keys=True))
        f.write("\n")


def safe_rate(num: float | int | None, den: float | int | None) -> float | None:
    if num is None or den is None:
        return None
    den = float(den)
    if den == 0.0:
        return None
    return float(num) / den


def snapshot_policy(system: SemanticCacheSystem) -> dict[str, Any]:
    p = system.policy
    threshold = p.threshold
    finite = threshold is not None and math.isfinite(float(threshold))
    return {
        "scorer_name": p.scorer_name,
        "scorer_version": int(p.scorer_version),
        "threshold": (float(threshold) if threshold is not None else None),
        "threshold_version": int(p.threshold_version),
        "trained": bool(p.trained),
        "calibrated": bool(p.calibrated),
        "frozen": bool(p.frozen),
        "finite_threshold": bool(finite),
    }


def snapshot_training_counts(system: SemanticCacheSystem) -> dict[str, int]:
    breakdown = store_breakdown(system)
    return {
        "h0_total": int(breakdown["h0_total"]),
        "h1_total": int(breakdown["h1_total"]),
        "h0_train": int(breakdown["h0_train"]),
        "h1_train": int(breakdown["h1_train"]),
        "h0_calibration": int(breakdown["h0_calibration"]),
        "h1_calibration": int(breakdown["h1_calibration"]),
        "total": int(breakdown["total"]),
    }


def snapshot_judged_pair_count(system: SemanticCacheSystem) -> int:
    return int(store_breakdown(system)["total"])


def _fmt_float(value: float | None, *, digits: int = 4) -> str:
    if value is None:
        return "None"
    return f"{float(value):.{digits}f}"


def _truncate_for_log(text: object, *, max_len: int = 80) -> str:
    s = str(text).replace("\n", "\\n")
    if len(s) <= max_len:
        return s
    return s[:max_len] + "...(truncated)"


# ── Phase 3/4: judge-sourced metrics recorder with exact accept/reject join ─

class MetricsRecorder:
    """Records judged pairs from the dataset judge and joins them against the
    served decision (SystemResponse) to compute exact serving FPR/TPR.

    The dataset judge runs concurrently across top-k candidates on the system
    executor, so `record_judge` is lock-guarded.  `resolve_request` is called
    from the single-threaded serve loop after `handle()` returns (all judge
    calls for the request have completed by then).
    """

    def __init__(self, method: str, target_fpr: float, logger: logging.Logger) -> None:
        self.method = method
        self.target_fpr = float(target_fpr)
        self.logger = logger
        self._lock = RLock()
        self._current_request_index = 0
        self._pending: dict[int, list[dict[str, Any]]] = {}
        self.true_accepts = 0
        self.false_accepts = 0
        self.true_rejects = 0
        self.false_rejects = 0
        self.judge_events = 0
        self.uncertain_events = 0
        self.unjudged_accepts = 0
        self._missing_accept_warned = False

    def set_request_index(self, request_index: int) -> None:
        with self._lock:
            self._current_request_index = int(request_index)

    def record_judge(
        self,
        *,
        candidate_key: Any,
        judge_label: JudgeLabel,
        query_row_id: Any = None,
        candidate_row_id: Any = None,
        query_label: Any = None,
        candidate_label: Any = None,
        query_anchor: Any = None,
        candidate_anchor: Any = None,
        same_anchor: Any = None,
        reason: str | None = None,
    ) -> None:
        with self._lock:
            self.judge_events += 1
            if judge_label == JudgeLabel.UNCERTAIN:
                self.uncertain_events += 1
            self._pending.setdefault(self._current_request_index, []).append(
                {
                    "candidate_key": (str(candidate_key) if candidate_key is not None else None),
                    "label": judge_label,
                }
            )

    def resolve_request(self, request_index: int, *, source: str, accepted_key: Any) -> None:
        """Mark which judged pair (if any) was the accepted served candidate.

        accept = source == "cache" AND candidate_key == accepted_key.
        Everything else retrieved-and-judged for this request was rejected.
        """
        with self._lock:
            pairs = self._pending.pop(int(request_index), [])
            accepted_key = str(accepted_key) if accepted_key is not None else None
            accept_matched = False
            for pair in pairs:
                label = pair["label"]
                if label == JudgeLabel.UNCERTAIN:
                    continue
                is_accepted = (
                    source == "cache"
                    and accepted_key is not None
                    and pair["candidate_key"] == accepted_key
                )
                if is_accepted:
                    accept_matched = True
                    if label == JudgeLabel.REUSABLE:
                        self.true_accepts += 1
                    else:
                        self.false_accepts += 1
                else:
                    if label == JudgeLabel.REUSABLE:
                        self.false_rejects += 1
                    else:
                        self.true_rejects += 1
            if source == "cache" and not accept_matched:
                # A cache HIT whose served candidate had no usable judged label
                # (exact-duplicate self-pair skipped by the shadow collector, or
                # judged UNCERTAIN). Honest accounting: exclude from FPR/TPR.
                self.unjudged_accepts += 1
                if not self._missing_accept_warned:
                    self.logger.warning(
                        "[%s] cache HIT had no judged served-candidate label "
                        "(self-pair skip or uncertain); excluded from FPR/TPR. "
                        "Expected for exact-duplicate hits.",
                        self.method,
                    )
                    self._missing_accept_warned = True

    def reset(self) -> None:
        """Zero all accumulated counters and pending judged pairs.

        Used after a warm-up phase so that shadow-view metrics (FPR/TPR/etc.)
        reflect only the measured replay, not the warm-up traffic.
        """
        with self._lock:
            self._pending.clear()
            self.true_accepts = 0
            self.false_accepts = 0
            self.true_rejects = 0
            self.false_rejects = 0
            self.judge_events = 0
            self.uncertain_events = 0
            self.unjudged_accepts = 0

    def metrics(self) -> dict[str, Any]:
        return self.metrics_since(None)

    def count_snapshot(self) -> dict[str, int]:
        """Return an atomic confusion-count snapshot for phase deltas."""
        with self._lock:
            return {
                "true_accepts": int(self.true_accepts),
                "false_accepts": int(self.false_accepts),
                "true_rejects": int(self.true_rejects),
                "false_rejects": int(self.false_rejects),
                "unjudged_accepts": int(self.unjudged_accepts),
                "judge_events": int(self.judge_events),
                "uncertain_events": int(self.uncertain_events),
            }

    def metrics_since(self, baseline: dict[str, int] | None) -> dict[str, Any]:
        """Compute metrics from counts observed after ``baseline``.

        This is used to exclude cold-start requests from active-session FPR:
        before a threshold exists, every candidate is necessarily rejected and
        counting those rejections as active true negatives biases FPR downward.
        """
        current = self.count_snapshot()
        base = baseline or {key: 0 for key in current}
        counts = {key: current[key] - int(base.get(key, 0)) for key in current}
        ta = counts["true_accepts"]
        fa = counts["false_accepts"]
        tr = counts["true_rejects"]
        fr = counts["false_rejects"]
        unjudged = counts["unjudged_accepts"]
        judge_events = counts["judge_events"]
        uncertain = counts["uncertain_events"]
        empirical_fpr = safe_rate(fa, fa + tr)
        tpr = safe_rate(ta, ta + fr)
        precision = safe_rate(ta, ta + fa)
        target_fpr_respected: bool | None = None
        if empirical_fpr is not None:
            target_fpr_respected = bool(empirical_fpr <= self.target_fpr)
        return {
            "true_accepts": int(ta),
            "false_accepts": int(fa),
            "true_rejects": int(tr),
            "false_rejects": int(fr),
            "empirical_fpr": empirical_fpr,
            "tpr": tpr,
            "precision": precision,
            "target_fpr": self.target_fpr,
            "target_fpr_respected": target_fpr_respected,
            "unjudged_accepts": int(unjudged),
            "judge_events": int(judge_events),
            "uncertain_events": int(uncertain),
            "fpr_tpr_computable": bool(empirical_fpr is not None or tpr is not None),
        }


# ── Phase 7: lifecycle event tracker ───────────────────────────────────────

class LifecycleTracker:
    """Emits observable lifecycle transitions to a shared JSONL file."""

    def __init__(self, method: str, jsonl_path: str | Path, logger: logging.Logger) -> None:
        self.method = method
        self.path = Path(jsonl_path)
        self.logger = logger
        self._prev: dict[str, Any] | None = None
        self._first_hit_emitted = False
        self._first_pair_emitted = False

    def _emit(self, request_index: int, event: str, old: Any, new: Any, details: dict | None = None) -> None:
        row = {
            "timestamp": _utc_now(),
            "method": self.method,
            "request_index": int(request_index),
            "event": event,
            "old_value": old,
            "new_value": new,
            "details": details or {},
        }
        append_jsonl(self.path, row)
        self.logger.info("[%s] lifecycle %s: %s -> %s", self.method, event, old, new)

    def update(self, *, request_index: int, policy: dict[str, Any], judged_pairs: int, hit_count: int) -> None:
        cur = {
            "trained": bool(policy["trained"]),
            "calibrated": bool(policy["calibrated"]),
            "threshold": policy["threshold"],
            "finite_threshold": bool(policy["finite_threshold"]),
            "threshold_version": int(policy["threshold_version"]),
            "scorer_version": int(policy["scorer_version"]),
        }
        if self._prev is None:
            self._prev = cur
        else:
            if cur["trained"] != self._prev["trained"]:
                self._emit(request_index, "trained_changed", self._prev["trained"], cur["trained"])
            if cur["calibrated"] != self._prev["calibrated"]:
                self._emit(request_index, "calibrated_changed", self._prev["calibrated"], cur["calibrated"])
            if cur["threshold"] != self._prev["threshold"]:
                self._emit(request_index, "threshold_changed", self._prev["threshold"], cur["threshold"])
            if cur["finite_threshold"] and not self._prev["finite_threshold"]:
                self._emit(request_index, "threshold_became_finite", False, True)
            if cur["threshold_version"] != self._prev["threshold_version"]:
                self._emit(
                    request_index, "threshold_version_changed",
                    self._prev["threshold_version"], cur["threshold_version"],
                )
            if cur["scorer_version"] != self._prev["scorer_version"]:
                self._emit(
                    request_index, "scorer_version_changed",
                    self._prev["scorer_version"], cur["scorer_version"],
                )
            self._prev = cur

        if not self._first_pair_emitted and judged_pairs > 0:
            self._first_pair_emitted = True
            self._emit(request_index, "first_judged_pair_observed", 0, int(judged_pairs))
        if not self._first_hit_emitted and hit_count > 0:
            self._first_hit_emitted = True
            self._emit(request_index, "first_cache_hit", 0, int(hit_count))


@dataclass(frozen=True)
class DatasetRow:
    row_id: int
    text: str
    label: int
    group: str
    embedding: np.ndarray
    prompt: str


@dataclass(frozen=True)
class OnlineRunSpec:
    name: str
    scorer: str
    scorers: list[str] | None


def encode_prompt(row_id: int, text: str) -> str:
    # Keep row id as the first token so candidate_query can be mapped back
    # exactly even when text is duplicated in the NPZ.
    return f"row:{int(row_id)}\t{text}"


def row_id_from_prompt(prompt: object) -> int:
    text = str(prompt)
    head = text.split("\t", 1)[0]
    if not head.startswith("row:"):
        raise ValueError(f"Bad dataset prompt key; expected 'row:<id>\\t...', got {text[:120]!r}")
    try:
        return int(head[4:])
    except ValueError as exc:
        raise ValueError(f"Bad dataset row id in prompt: {text[:120]!r}") from exc


class DatasetEmbeddingProvider(EmbeddingProvider):
    """EmbeddingProvider backed by the selected NPZ embedding field."""

    def __init__(self, rows_by_id: dict[int, DatasetRow]) -> None:
        self._rows_by_id = rows_by_id

    def embed(self, text: str) -> tuple[float, ...]:
        row_id = row_id_from_prompt(text)
        row = self._rows_by_id[row_id]
        arr = np.asarray(row.embedding, dtype=np.float64).reshape(-1)
        return tuple(float(x) for x in arr.tolist())


class DatasetH1H0Judge(SemanticReuseJudge):
    """Pairwise judge using NPZ labels and groups.

    When a `MetricsRecorder` is attached, every judged pair is recorded at the
    source of truth (this judge sees every query-candidate pair).  The judge
    semantics are unchanged by recording.
    """

    def __init__(self, rows_by_id: dict[int, DatasetRow], *, recorder: MetricsRecorder | None = None, logger: logging.Logger | None = None) -> None:
        self._rows_by_id = rows_by_id
        self._recorder = recorder
        self._logger = logger

    @property
    def name(self) -> str:
        return "dataset-h1h0-pairwise-judge"

    def judge(self, request: JudgeRequest) -> JudgeResult:
        start_time = time.time()
        if self._logger:
            self._logger.info(f"DatasetH1H0Judge: Starting judge for query '{_truncate_for_log(request.query)}'")

        if request.candidate_query is None:
            if self._recorder is not None:
                self._recorder.record_judge(
                    candidate_key=request.candidate_key,
                    judge_label=JudgeLabel.UNCERTAIN,
                    reason="missing candidate_query",
                )
            return JudgeResult(
                request=request,
                decision=JudgeDecision(
                    label=JudgeLabel.UNCERTAIN,
                    rationale="missing candidate_query",
                    metadata={"judge": self.name},
                ),
            )

        try:
            q = self._rows_by_id[row_id_from_prompt(request.query)]
            c = self._rows_by_id[row_id_from_prompt(request.candidate_query)]
        except Exception as exc:
            if self._recorder is not None:
                self._recorder.record_judge(
                    candidate_key=request.candidate_key,
                    judge_label=JudgeLabel.UNCERTAIN,
                    reason=f"could not decode row ids: {exc}",
                )
            if self._logger:
                end_time = time.time()
                self._logger.warning(f"DatasetH1H0Judge: Error judging query '{_truncate_for_log(request.query)}'. Took {end_time - start_time:.4f}s. Error: {exc}")
            return JudgeResult(
                request=request,
                decision=JudgeDecision(
                    label=JudgeLabel.UNCERTAIN,
                    rationale=f"could not decode row ids: {exc}",
                    metadata={"judge": self.name, "error": str(exc)},
                ),
            )

        same_group = q.group == c.group

        if q.label == -1 or c.label == -1:
            label = JudgeLabel.UNCERTAIN
            rationale = "query or candidate has label=-1"
        elif q.label == 1 and c.label == 1 and same_group:
            label = JudgeLabel.REUSABLE
            rationale = "same group and both rows are H1"
        else:
            label = JudgeLabel.NOT_REUSABLE
            rationale = "different group or at least one row is H0"

        if self._recorder is not None:
            self._recorder.record_judge(
                candidate_key=request.candidate_key,
                judge_label=label,
                query_row_id=q.row_id,
                candidate_row_id=c.row_id,
                query_label=q.label,
                candidate_label=c.label,
                query_anchor=q.group,
                candidate_anchor=c.group,
                same_anchor=same_group,
                reason=rationale,
            )

        if self._logger:
            end_time = time.time()
            self._logger.info(f"DatasetH1H0Judge: Finished judge for query '{_truncate_for_log(request.query)}'. Label: {label.name}. Took {end_time - start_time:.4f}s")

        return JudgeResult(
            request=request,
            decision=JudgeDecision(
                label=label,
                rationale=rationale,
                metadata={
                    "judge": self.name,
                    "query_row_id": q.row_id,
                    "candidate_row_id": c.row_id,
                    "query_label": q.label,
                    "candidate_label": c.label,
                    "query_group": q.group,
                    "candidate_group": c.group,
                    "same_group": same_group,
                },
            ),
        )


def inspect_npz_schema(npz_path: str | Path, *, allow_pickle: bool) -> dict[str, Any]:
    npz = np.load(npz_path, allow_pickle=allow_pickle)
    try:
        fields: list[dict[str, Any]] = []
        for name in npz.files:
            arr = npz[name]
            field: dict[str, Any] = {
                "name": name,
                "shape": list(arr.shape),
                "dtype": str(arr.dtype),
                "ndim": int(arr.ndim),
            }
            if arr.ndim == 1 and name in {"label", "h0h1"}:
                values, counts = np.unique(arr, return_counts=True)
                field["distribution"] = {
                    str(int(v)): int(c)
                    for v, c in zip(values.tolist(), counts.tolist())
                }
            fields.append(field)
        return {"path": str(npz_path), "available_fields": list(npz.files), "fields": fields}
    finally:
        npz.close()


def _require_field(npz: Any, field: str, *, role: str) -> np.ndarray:
    if field not in npz.files:
        raise ValueError(f"NPZ missing required {role} field {field!r}; available fields: {tuple(npz.files)}")
    return npz[field]


def _select_indices(
    labels: np.ndarray,
    *,
    max_rows: int | None,
    seed: int,
    include_uncertain: bool,
    selection: str,
) -> list[int]:
    labels = np.asarray(labels).reshape(-1)

    if include_uncertain:
        candidate_indices = np.arange(len(labels), dtype=np.int64)
    else:
        candidate_indices = np.flatnonzero(np.isin(labels, [0, 1]))

    if candidate_indices.size == 0:
        raise ValueError("No selectable rows. Check --label-field and --include-uncertain.")

    if max_rows is None or int(max_rows) >= int(candidate_indices.size):
        selected = candidate_indices.copy()
    elif int(max_rows) < 0:
        raise ValueError("--max-rows must be non-negative")
    elif selection == "head":
        selected = candidate_indices[: int(max_rows)]
    elif selection == "random":
        rng = np.random.default_rng(int(seed))
        selected = rng.choice(candidate_indices, size=int(max_rows), replace=False)
    else:
        raise ValueError(f"Unknown selection mode: {selection!r}")

    if selection == "random":
        # Random traffic order is more realistic for online cache learning than
        # sorting back to corpus order. It is still deterministic via --seed.
        return [int(i) for i in selected.tolist()]

    return [int(i) for i in selected.tolist()]


def load_rows(
    npz_path: str | Path,
    *,
    query_field: str,
    label_field: str,
    anchor_field: str,
    embedding_field: str,
    max_rows: int | None,
    seed: int,
    include_uncertain: bool,
    selection: str,
    allow_pickle: bool,
) -> tuple[list[DatasetRow], dict[str, Any]]:
    raw_schema = inspect_npz_schema(npz_path, allow_pickle=allow_pickle)
    npz = np.load(npz_path, allow_pickle=allow_pickle)
    try:
        text_arr = _require_field(npz, query_field, role="query")
        label_arr = _require_field(npz, label_field, role="label")
        group_arr = _require_field(npz, anchor_field, role="anchor/group")
        emb_arr = _require_field(npz, embedding_field, role="embedding")

        n = int(len(text_arr))
        for field_name, arr in (
            (label_field, label_arr),
            (anchor_field, group_arr),
            (embedding_field, emb_arr),
        ):
            if int(arr.shape[0]) != n:
                raise ValueError(
                    f"Field {field_name!r} has first dimension {arr.shape[0]}, expected {n} from {query_field!r}"
                )

        if emb_arr.ndim != 2:
            raise ValueError(f"Embedding field {embedding_field!r} must be 2D, got shape {emb_arr.shape}")

        label_dist_before = {
            str(int(v)): int(c)
            for v, c in zip(*np.unique(np.asarray(label_arr).reshape(-1), return_counts=True))
        }

        indices = _select_indices(
            label_arr,
            max_rows=max_rows,
            seed=seed,
            include_uncertain=include_uncertain,
            selection=selection,
        )

        rows: list[DatasetRow] = []
        for row_id in indices:
            text = str(text_arr[row_id])
            label = int(label_arr[row_id])
            group = str(group_arr[row_id])
            emb = np.asarray(emb_arr[row_id], dtype=np.float32).reshape(-1)
            rows.append(
                DatasetRow(
                    row_id=int(row_id),
                    text=text,
                    label=label,
                    group=group,
                    embedding=emb,
                    prompt=encode_prompt(int(row_id), text),
                )
            )

        label_values, label_counts = np.unique([row.label for row in rows], return_counts=True)
        group_values = {row.group for row in rows}
        schema_report = {
            "schema_valid": True,
            "raw_npz_schema": raw_schema,
            "chosen_fields": {
                "query_field": query_field,
                "label_field": label_field,
                "anchor_field": anchor_field,
                "query_embedding_field": embedding_field,
            },
            "row_count_total": n,
            "row_count_selected": len(rows),
            "selected_original_row_min": min((row.row_id for row in rows), default=None),
            "selected_original_row_max": max((row.row_id for row in rows), default=None),
            "selection": {
                "mode": selection,
                "seed": seed,
                "max_rows": max_rows,
                "include_uncertain": include_uncertain,
            },
            "embedding_dim": int(emb_arr.shape[1]),
            "label_distribution_before_filtering": label_dist_before,
            "label_distribution_selected": {
                str(int(v)): int(c) for v, c in zip(label_values.tolist(), label_counts.tolist())
            },
            "unique_groups_selected": len(group_values),
        }
        return rows, schema_report
    finally:
        npz.close()


def _safe_len_call(obj: Any, method_name: str) -> int | None:
    method = getattr(obj, method_name, None)
    if not callable(method):
        return None
    try:
        return int(len(method()))
    except Exception:
        return None


def store_breakdown(system: SemanticCacheSystem) -> dict[str, Any]:
    store = getattr(getattr(system.cache, "runtime", None), "judge_training_store", None)
    if store is None:
        return {
            "present": False,
            "h0_total": 0,
            "h1_total": 0,
            "total": 0,
            "h0_train": 0,
            "h1_train": 0,
            "h0_calibration": 0,
            "h1_calibration": 0,
        }

    h0_total = _safe_len_call(store, "h0")
    h1_total = _safe_len_call(store, "h1")
    h0_train = _safe_len_call(store, "h0_train")
    h1_train = _safe_len_call(store, "h1_train")
    h0_calibration = _safe_len_call(store, "h0_calibration")
    h1_calibration = _safe_len_call(store, "h1_calibration")

    # Legacy non-split stores only have h0()/h1().
    if h0_train is None:
        h0_train = h0_total or 0
    if h1_train is None:
        h1_train = h1_total or 0
    if h0_calibration is None:
        h0_calibration = 0
    if h1_calibration is None:
        h1_calibration = 0
    if h0_total is None:
        h0_total = h0_train + h0_calibration
    if h1_total is None:
        h1_total = h1_train + h1_calibration

    return {
        "present": True,
        "h0_total": int(h0_total),
        "h1_total": int(h1_total),
        "total": int(h0_total + h1_total),
        "h0_train": int(h0_train),
        "h1_train": int(h1_train),
        "h0_calibration": int(h0_calibration),
        "h1_calibration": int(h1_calibration),
    }


def policy_snapshot(system: SemanticCacheSystem) -> dict[str, Any]:
    p = system.policy
    return {
        "scorer_name": p.scorer_name,
        "scorer_version": p.scorer_version,
        "threshold": p.threshold,
        "threshold_version": p.threshold_version,
        "trained": p.trained,
        "calibrated": p.calibrated,
        "frozen": p.frozen,
        "metadata": dict(p.metadata),
    }


def _trace_row(
    *,
    phase: str,
    request_index: int,
    row: DatasetRow,
    response: Any,
) -> dict[str, Any]:
    policy = response.policy
    return {
        "phase": phase,
        "request_index": int(request_index),
        "row_id": row.row_id,
        "label": row.label,
        "group": row.group,
        "source": response.source,
        "cache_key": response.cache_key,
        "score": response.score,
        "threshold": response.threshold,
        "trained": bool(policy.trained),
        "calibrated": bool(policy.calibrated),
        "scorer_version": int(policy.scorer_version),
        "threshold_version": int(policy.threshold_version),
    }


def _write_trace_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "phase",
        "request_index",
        "row_id",
        "label",
        "group",
        "source",
        "cache_key",
        "score",
        "threshold",
        "trained",
        "calibrated",
        "scorer_version",
        "threshold_version",
    ]
    write_csv_rows(path, rows, fields)


def _write_json(path: Path, payload: Any) -> None:
    write_json(path, payload)


def _shutdown_system(system: SemanticCacheSystem) -> None:
    executor = getattr(system, "_executor", None)
    if executor is not None and hasattr(executor, "shutdown"):
        try:
            executor.shutdown(wait=True)
        except Exception:
            pass


def _progress_row(
    *,
    method: str,
    request_index: int,
    total_requests: int,
    report: dict[str, Any],
    policy: dict[str, Any],
    counts: dict[str, int],
    metrics: dict[str, Any],
    judged_pairs: int,
    first_calibrated_at: int | None,
    first_hit_at: int | None,
) -> dict[str, Any]:
    return {
        "method": method,
        "request_index": int(request_index),
        "total_requests": int(total_requests),
        "hits": int(report.get("cache_hits", 0)),
        "misses": int(report.get("llm_calls", 0)),
        "hit_rate": float(report.get("hit_rate", 0.0)),
        "empirical_fpr": metrics["empirical_fpr"],
        "tpr": metrics["tpr"],
        "precision": metrics["precision"],
        "true_accepts": metrics["true_accepts"],
        "false_accepts": metrics["false_accepts"],
        "true_rejects": metrics["true_rejects"],
        "false_rejects": metrics["false_rejects"],
        "trained": bool(policy["trained"]),
        "calibrated": bool(policy["calibrated"]),
        "threshold": policy["threshold"],
        "finite_threshold": bool(policy["finite_threshold"]),
        "scorer_version": int(policy["scorer_version"]),
        "threshold_version": int(policy["threshold_version"]),
        "judged_pair_count": int(judged_pairs),
        "training_store_h0": int(counts["h0_total"]),
        "training_store_h1": int(counts["h1_total"]),
        "first_calibrated_at_request": first_calibrated_at,
        "first_hit_at_request": first_hit_at,
    }


def _log_progress_line(logger: logging.Logger, row: dict[str, Any]) -> None:
    logger.info(
        "[%s][request=%d/%d] hits=%d misses=%d hit_rate=%s fpr=%s tpr=%s precision=%s "
        "trained=%s calibrated=%s threshold=%s finite_threshold=%s scorer_v=%d threshold_v=%d "
        "judged_pairs=%d train_h0=%d train_h1=%d first_calibrated=%s first_hit=%s",
        row["method"], row["request_index"], row["total_requests"],
        row["hits"], row["misses"], _fmt_float(row["hit_rate"], digits=3),
        _fmt_float(row["empirical_fpr"], digits=3), _fmt_float(row["tpr"], digits=3),
        _fmt_float(row["precision"], digits=3),
        row["trained"], row["calibrated"], _fmt_float(row["threshold"]),
        row["finite_threshold"], row["scorer_version"], row["threshold_version"],
        row["judged_pair_count"], row["training_store_h0"], row["training_store_h1"],
        str(row["first_calibrated_at_request"]), str(row["first_hit_at_request"]),
    )


def _wait_for_policy_logged(
    system: SemanticCacheSystem,
    *,
    fit_wait_secs: float,
    method: str,
    logger: logging.Logger,
    phase: str,
) -> None:
    start = time.monotonic()
    next_log = start
    logger.info("[%s] %s: waiting up to %.1fs for background training/calibration", method, phase, float(fit_wait_secs))
    while time.monotonic() - start < float(fit_wait_secs):
        if system.policy.calibrated:
            logger.info("[%s] %s: policy calibrated after %.1fs", method, phase, time.monotonic() - start)
            break
        now = time.monotonic()
        if now >= next_log:
            policy = snapshot_policy(system)
            counts = snapshot_training_counts(system)
            logger.info(
                "[%s] %s wait: trained=%s calibrated=%s threshold=%s finite_threshold=%s "
                "scorer_v=%d threshold_v=%d judged_pairs=%d train_h0=%d train_h1=%d",
                method, phase, policy["trained"], policy["calibrated"], _fmt_float(policy["threshold"]),
                policy["finite_threshold"], policy["scorer_version"], policy["threshold_version"],
                counts["total"], counts["h0_total"], counts["h1_total"],
            )
            flush_logger(logger)
            next_log = now + 5.0
        time.sleep(0.2)


def run_one_online(
    *,
    spec: OnlineRunSpec,
    rows: list[DatasetRow],
    rows_by_id: dict[int, DatasetRow],
    output_dir: Path,
    progress_csv_path: Path,
    lifecycle_path: Path,
    logger: logging.Logger,
    target_fpr: float,
    top_k: int,
    batch_size: int,
    min_h0: int,
    min_h1: int,
    parallelism: int,
    fit_wait_secs: float,
    warmup_rows: int,
    persistence: bool,
    progress_every: int,
    namespace_prefix: str,
    embedding_dim: int,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)

    # Clear any prior persisted state so a new run starts from an empty cache.
    # With --persist, the cache still persists *within* this run (normal cache
    # behavior); we only wipe leftovers from earlier runs to avoid a reused
    # --output-dir contaminating this run with stale vectors / thresholds.
    state_dir = output_dir / "state"
    if state_dir.exists():
        shutil.rmtree(state_dir, ignore_errors=True)
        logger.info("[%s] cleared prior persisted state at %s (fresh run)", spec.name, state_dir)

    logger.info("=== METHOD START ===")
    logger.info("method: %s", spec.name)
    logger.info("scorers: %s", ",".join(spec.scorers) if spec.scorers else spec.scorer)
    logger.info("stream_rows: %d", len(rows))
    logger.info("target_fpr: %s", target_fpr)
    logger.info("top_k: %d", top_k)
    logger.info("batch_size: %d", batch_size)
    logger.info("min_h0: %d", min_h0)
    logger.info("min_h1: %d", min_h1)
    logger.info("parallelism: %d", parallelism)
    logger.info("persistence: %s", persistence)
    logger.info("embedding_dim: %d", embedding_dim)
    flush_logger(logger)

    recorder = MetricsRecorder(method=spec.name, target_fpr=float(target_fpr), logger=logger)
    lifecycle = LifecycleTracker(method=spec.name, jsonl_path=lifecycle_path, logger=logger)

    system = SemanticCacheSystem(
        llm=MockLLM(response_template="answer: {prompt}"),
        stream=None,
        scorer=spec.scorer,
        scorers=spec.scorers if spec.scorer == "ensemble" else None,
                judge=DatasetH1H0Judge(rows_by_id, recorder=recorder, logger=logger),
        embedding_provider=DatasetEmbeddingProvider(rows_by_id),
        target_fpr=target_fpr,
        top_k=top_k,
        root_dir=output_dir / "state",
        batch_size=batch_size,
        min_h0=min_h0,
        min_h1=min_h1,
        persistence=persistence,
        parallelism=parallelism,
        namespace=f"{namespace_prefix}-{spec.name}",
    )

    trace: list[dict[str, Any]] = []
    progress_rows: list[dict[str, Any]] = []
    first_hit_at: int | None = None
    first_calibrated_at: int | None = None
    request_index = 0

    def serve(row: DatasetRow, *, phase: str) -> None:
        nonlocal first_hit_at, first_calibrated_at, request_index
        request_index += 1
        recorder.set_request_index(request_index)
        response = system.handle(row.prompt)

        # Exact accept/reject join: shadow collection (and thus all judge calls)
        # for this request have completed synchronously inside handle().
        recorder.resolve_request(
            request_index,
            source=response.source,
            accepted_key=response.cache_key if response.source == "cache" else None,
        )

        if first_hit_at is None and response.source == "cache":
            first_hit_at = request_index
        if first_calibrated_at is None and response.policy.calibrated:
            first_calibrated_at = request_index

        trace.append(_trace_row(phase=phase, request_index=request_index, row=row, response=response))

        report = system.report()
        policy = snapshot_policy(system)
        counts = snapshot_training_counts(system)
        lifecycle.update(
            request_index=request_index,
            policy=policy,
            judged_pairs=counts["total"],
            hit_count=int(report.get("cache_hits", 0)),
        )

        if request_index % max(1, batch_size) == 0:
            system._maybe_check_stopping()

        if progress_every > 0 and request_index % progress_every == 0:
            metrics = recorder.metrics()
            judged_pairs = counts["total"]
            row_out = _progress_row(
                method=spec.name, request_index=request_index, total_requests=len(rows),
                report=report, policy=policy, counts=counts, metrics=metrics,
                judged_pairs=judged_pairs, first_calibrated_at=first_calibrated_at,
                first_hit_at=first_hit_at,
            )
            progress_rows.append(row_out)
            write_csv_rows(progress_csv_path, progress_rows, PROGRESS_CSV_FIELDS)
            _log_progress_line(logger, row_out)
            flush_logger(logger)

    started_at = time.time()
    logger.info("[%s] starting online stream over %d dataset rows", spec.name, len(rows))
    for row in rows:
        serve(row, phase="main")

    system._maybe_check_stopping()
    logger.info("[%s] main stream finished (%d requests)", spec.name, request_index)
    _wait_for_policy_logged(
        system, fit_wait_secs=fit_wait_secs, method=spec.name, logger=logger, phase="post-stream",
    )

    # If calibration finishes after the main stream, replay real dataset rows so
    # the active policy can actually serve cache hits. This is still online
    # traffic, not offline prefit/replay.
    if warmup_rows > 0 and (not system.policy.calibrated or first_hit_at is None):
        limit = min(int(warmup_rows), len(rows))
        logger.info("[%s] warmup: replaying up to %d dataset rows as online traffic", spec.name, limit)
        for row in rows[:limit]:
            serve(row, phase="warmup")
            if system.policy.calibrated and first_hit_at is not None:
                break
        system._maybe_check_stopping()
        _wait_for_policy_logged(
            system, fit_wait_secs=fit_wait_secs, method=spec.name, logger=logger, phase="post-warmup",
        )

    final_report = system.report()
    final_policy = policy_snapshot(system)
    final_policy_snap = snapshot_policy(system)
    final_store = store_breakdown(system)
    final_metrics = recorder.metrics()

    runtime_seconds = time.time() - started_at
    summary = {
        "name": spec.name,
        "scorer": spec.scorer,
        "scorers": spec.scorers,
        "target_fpr": float(target_fpr),
        "top_k": int(top_k),
        "batch_size": int(batch_size),
        "min_h0": int(min_h0),
        "min_h1": int(min_h1),
        "parallelism": int(parallelism),
        "runtime_seconds": runtime_seconds,
        "first_calibrated_at_request": first_calibrated_at,
        "first_hit_at_request": first_hit_at,
        "report": final_report,
        "policy": final_policy,
        "store": final_store,
        "metrics": final_metrics,
        "trained": bool(final_report.get("trained")),
        "calibrated": bool(final_report.get("calibrated")),
        "threshold": final_report.get("threshold"),
        "finite_threshold": bool(final_policy_snap["finite_threshold"]),
        "threshold_version": final_report.get("threshold_version"),
        "scorer_version": final_report.get("scorer_version"),
        "hit_count": int(final_report.get("cache_hits", 0)),
        "miss_count": int(final_report.get("llm_calls", 0)),
        "hit_rate": float(final_report.get("hit_rate", 0.0)),
        "judged_pair_count": int(final_store.get("total", 0)),
        "empirical_fpr": final_metrics["empirical_fpr"],
        "tpr": final_metrics["tpr"],
        "precision": final_metrics["precision"],
        "true_accepts": final_metrics["true_accepts"],
        "false_accepts": final_metrics["false_accepts"],
        "true_rejects": final_metrics["true_rejects"],
        "false_rejects": final_metrics["false_rejects"],
        "target_fpr_respected": final_metrics["target_fpr_respected"],
    }

    # Final per-method progress row so the CSV ends with the converged state.
    progress_rows.append(
        _progress_row(
            method=spec.name, request_index=request_index, total_requests=len(rows),
            report=final_report, policy=final_policy_snap, counts=snapshot_training_counts(system),
            metrics=final_metrics, judged_pairs=int(final_store.get("total", 0)),
            first_calibrated_at=first_calibrated_at, first_hit_at=first_hit_at,
        )
    )
    write_csv_rows(progress_csv_path, progress_rows, PROGRESS_CSV_FIELDS)
    _write_trace_csv(output_dir / f"{spec.name}_trace.csv", trace)
    _write_json(output_dir / f"{spec.name}_report.json", summary)

    _log_method_final_summary(logger, summary)
    _shutdown_system(system)
    return summary


def _log_method_final_summary(logger: logging.Logger, summary: dict[str, Any]) -> None:
    m = summary["metrics"]
    logger.info("=== METHOD FINAL SUMMARY ===")
    logger.info("method: %s", summary["name"])
    logger.info("requests: %d", summary["hit_count"] + summary["miss_count"])
    logger.info("hits: %d", summary["hit_count"])
    logger.info("misses: %d", summary["miss_count"])
    logger.info("hit_rate: %s", _fmt_float(summary["hit_rate"]))
    logger.info("trained: %s", summary["trained"])
    logger.info("calibrated: %s", summary["calibrated"])
    logger.info("threshold: %s", _fmt_float(summary["threshold"]))
    logger.info("finite_threshold: %s", summary["finite_threshold"])
    logger.info("threshold_version: %s", summary["threshold_version"])
    logger.info("scorer_version: %s", summary["scorer_version"])
    logger.info("judged_pair_count: %d", summary["judged_pair_count"])
    logger.info("training_store_h0: %d", summary["store"]["h0_total"])
    logger.info("training_store_h1: %d", summary["store"]["h1_total"])
    logger.info("first_calibrated_at_request: %s", summary["first_calibrated_at_request"])
    logger.info("first_hit_at_request: %s", summary["first_hit_at_request"])
    logger.info("target_fpr: %s", summary["target_fpr"])
    logger.info("empirical_fpr: %s", _fmt_float(m["empirical_fpr"]))
    logger.info("tpr: %s", _fmt_float(m["tpr"]))
    logger.info("precision: %s", _fmt_float(m["precision"]))
    logger.info("true_accepts: %d", m["true_accepts"])
    logger.info("false_accepts: %d", m["false_accepts"])
    logger.info("true_rejects: %d", m["true_rejects"])
    logger.info("false_rejects: %d", m["false_rejects"])
    logger.info("target_fpr_respected: %s", m["target_fpr_respected"])
    flush_logger(logger)


def determine_winner(cosine: dict[str, Any], ensemble: dict[str, Any]) -> dict[str, str]:
    """FPR/TPR-first winner logic with calibrated+hit_rate fallback.

    1. If only one method has a computable FPR that respects target_fpr, it wins.
    2. If both respect target_fpr, the higher TPR wins.
    3. If TPR is tied/unavailable, the higher hit_rate wins.
    4. If neither has computable FPR/TPR, fall back to calibrated + hit_rate and
       state explicitly that FPR/TPR were not computable.
    """
    c_fpr = cosine.get("empirical_fpr")
    e_fpr = ensemble.get("empirical_fpr")
    c_resp = cosine.get("target_fpr_respected")
    e_resp = ensemble.get("target_fpr_respected")
    c_tpr = cosine.get("tpr")
    e_tpr = ensemble.get("tpr")
    c_hit = float(cosine.get("hit_rate") or 0.0)
    e_hit = float(ensemble.get("hit_rate") or 0.0)

    c_ok = c_fpr is not None and bool(c_resp)
    e_ok = e_fpr is not None and bool(e_resp)

    def by_hit_rate(reason_prefix: str) -> dict[str, str]:
        if abs(c_hit - e_hit) <= 1e-12:
            return {"winner": "tie", "basis": "hit_rate", "reason": f"{reason_prefix}; hit rates equal"}
        winner = "ensemble" if e_hit > c_hit else "cosine"
        return {
            "winner": winner, "basis": "hit_rate",
            "reason": f"{reason_prefix}; higher hit_rate wins (cosine={c_hit:.4f}, ensemble={e_hit:.4f})",
        }

    if c_ok and not e_ok:
        return {"winner": "cosine", "basis": "only_cosine_respects_fpr",
                "reason": "only cosine has a computable FPR that respects target_fpr"}
    if e_ok and not c_ok:
        return {"winner": "ensemble", "basis": "only_ensemble_respects_fpr",
                "reason": "only ensemble has a computable FPR that respects target_fpr"}

    if c_ok and e_ok:
        if c_tpr is not None and e_tpr is not None and abs(c_tpr - e_tpr) > 1e-12:
            winner = "ensemble" if e_tpr > c_tpr else "cosine"
            return {
                "winner": winner, "basis": "tpr_among_fpr_respecting",
                "reason": f"both respect target_fpr; higher TPR wins (cosine={c_tpr:.4f}, ensemble={e_tpr:.4f})",
            }
        return by_hit_rate("both respect target_fpr; TPR tied or unavailable")

    # Neither has computable, respected FPR. If FPR/TPR not computable at all,
    # fall back to calibrated + hit_rate and say so explicitly.
    if c_fpr is None and e_fpr is None and c_tpr is None and e_tpr is None:
        c_cal = bool(cosine.get("calibrated"))
        e_cal = bool(ensemble.get("calibrated"))
        if c_cal and not e_cal:
            return {"winner": "cosine", "basis": "calibrated_fallback",
                    "reason": "FPR/TPR not computable; only cosine calibrated"}
        if e_cal and not c_cal:
            return {"winner": "ensemble", "basis": "calibrated_fallback",
                    "reason": "FPR/TPR not computable; only ensemble calibrated"}
        if not c_cal and not e_cal:
            return {"winner": "undetermined", "basis": "calibrated_fallback",
                    "reason": "FPR/TPR not computable and neither calibrated"}
        return by_hit_rate("FPR/TPR not computable; both calibrated")

    # Computable but neither respects target_fpr -> defer to hit_rate.
    return by_hit_rate("neither method respects target_fpr")


def build_comparison_summary(
    *,
    config: dict[str, Any],
    dataset: dict[str, Any],
    cosine: dict[str, Any],
    ensemble: dict[str, Any],
) -> dict[str, Any]:
    def public(method: dict[str, Any]) -> dict[str, Any]:
        return {k: v for k, v in method.items() if k not in {"report", "policy", "store"}}

    winner = determine_winner(cosine, ensemble)
    return {
        "experiment_type": "dataset_backed_online_semantic_cache_system",
        "offline_prefit_used": False,
        "config": config,
        "dataset": dataset,
        "cosine": public(cosine),
        "ensemble": public(ensemble),
        "cosine_lifecycle": {
            "report": cosine.get("report"),
            "policy": cosine.get("policy"),
            "store": cosine.get("store"),
        },
        "ensemble_lifecycle": {
            "report": ensemble.get("report"),
            "policy": ensemble.get("policy"),
            "store": ensemble.get("store"),
        },
        "delta": {
            "hit_rate": float(ensemble.get("hit_rate") or 0.0) - float(cosine.get("hit_rate") or 0.0),
            "hit_count": int(ensemble.get("hit_count") or 0) - int(cosine.get("hit_count") or 0),
            "miss_count": int(ensemble.get("miss_count") or 0) - int(cosine.get("miss_count") or 0),
            "judged_pair_count": int(ensemble.get("judged_pair_count") or 0) - int(cosine.get("judged_pair_count") or 0),
            "empirical_fpr": _delta(ensemble.get("empirical_fpr"), cosine.get("empirical_fpr")),
            "tpr": _delta(ensemble.get("tpr"), cosine.get("tpr")),
            "precision": _delta(ensemble.get("precision"), cosine.get("precision")),
        },
        "winner": winner,
        # Top-level metric keys so consumers/tests can find them directly.
        "empirical_fpr": {"cosine": cosine.get("empirical_fpr"), "ensemble": ensemble.get("empirical_fpr")},
        "tpr": {"cosine": cosine.get("tpr"), "ensemble": ensemble.get("tpr")},
        "precision": {"cosine": cosine.get("precision"), "ensemble": ensemble.get("precision")},
    }


def _delta(a: float | None, b: float | None) -> float | None:
    if a is None or b is None:
        return None
    return float(a) - float(b)


def write_markdown_report(path: Path, summary: dict[str, Any]) -> None:
    c = summary["cosine"]
    e = summary["ensemble"]
    w = summary["winner"]
    d = summary["dataset"]
    lines = [
        "# Dataset-backed ONLINE SemanticCacheSystem comparison",
        "",
        "This is an online lifecycle experiment. It does not call `run_cache.py` and does not call `prefit_and_calibrate` before the stream.",
        "",
        "## Dataset",
        f"- path: `{d.get('path')}`",
        f"- rows selected: {d.get('row_count_selected')} / {d.get('row_count_total')}",
        f"- embedding field: `{d.get('chosen_fields', {}).get('query_embedding_field')}`",
        f"- label field: `{d.get('chosen_fields', {}).get('label_field')}`",
        f"- anchor/group field: `{d.get('chosen_fields', {}).get('anchor_field')}`",
        f"- selected label distribution: `{d.get('label_distribution_selected')}`",
        "",
        "## Results",
        "",
        "| metric | cosine | ensemble |",
        "|---|---:|---:|",
        f"| trained | {c['trained']} | {e['trained']} |",
        f"| calibrated | {c['calibrated']} | {e['calibrated']} |",
        f"| threshold | {c['threshold']} | {e['threshold']} |",
        f"| hit_rate | {c['hit_rate']} | {e['hit_rate']} |",
        f"| empirical_fpr | {c['empirical_fpr']} | {e['empirical_fpr']} |",
        f"| tpr | {c['tpr']} | {e['tpr']} |",
        f"| precision | {c['precision']} | {e['precision']} |",
        f"| target_fpr_respected | {c['target_fpr_respected']} | {e['target_fpr_respected']} |",
        f"| judged_pair_count | {c['judged_pair_count']} | {e['judged_pair_count']} |",
        f"| first_calibrated_at_request | {c['first_calibrated_at_request']} | {e['first_calibrated_at_request']} |",
        f"| first_hit_at_request | {c['first_hit_at_request']} | {e['first_hit_at_request']} |",
        "",
        "## Winner",
        f"- winner: `{w['winner']}`",
        f"- basis: `{w['basis']}`",
        f"- reason: {w['reason']}",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _log_comparison_summary(logger: logging.Logger, cosine: dict[str, Any], ensemble: dict[str, Any], winner: dict[str, str]) -> None:
    logger.info("=== COMPARISON SUMMARY ===")
    logger.info("cosine_hit_rate: %s", _fmt_float(cosine.get("hit_rate")))
    logger.info("ensemble_hit_rate: %s", _fmt_float(ensemble.get("hit_rate")))
    logger.info("delta_hit_rate: %s", _fmt_float(_delta(ensemble.get("hit_rate"), cosine.get("hit_rate"))))
    logger.info("cosine_empirical_fpr: %s", _fmt_float(cosine.get("empirical_fpr")))
    logger.info("ensemble_empirical_fpr: %s", _fmt_float(ensemble.get("empirical_fpr")))
    logger.info("delta_empirical_fpr: %s", _fmt_float(_delta(ensemble.get("empirical_fpr"), cosine.get("empirical_fpr"))))
    logger.info("cosine_tpr: %s", _fmt_float(cosine.get("tpr")))
    logger.info("ensemble_tpr: %s", _fmt_float(ensemble.get("tpr")))
    logger.info("delta_tpr: %s", _fmt_float(_delta(ensemble.get("tpr"), cosine.get("tpr"))))
    logger.info("cosine_precision: %s", _fmt_float(cosine.get("precision")))
    logger.info("ensemble_precision: %s", _fmt_float(ensemble.get("precision")))
    logger.info("delta_precision: %s", _fmt_float(_delta(ensemble.get("precision"), cosine.get("precision"))))
    logger.info("cosine_calibrated: %s", cosine.get("calibrated"))
    logger.info("ensemble_calibrated: %s", ensemble.get("calibrated"))
    logger.info("cosine_trained: %s", cosine.get("trained"))
    logger.info("ensemble_trained: %s", ensemble.get("trained"))
    logger.info("cosine_first_calibrated_at_request: %s", cosine.get("first_calibrated_at_request"))
    logger.info("ensemble_first_calibrated_at_request: %s", ensemble.get("first_calibrated_at_request"))
    logger.info("cosine_first_hit_at_request: %s", cosine.get("first_hit_at_request"))
    logger.info("ensemble_first_hit_at_request: %s", ensemble.get("first_hit_at_request"))
    logger.info("winner: %s (%s) -- %s", winner["winner"], winner["basis"], winner["reason"])
    flush_logger(logger)


def run_comparison(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(output_dir)

    scorers = [s.strip() for s in args.scorers.split(",") if s.strip()]

    logger.info("=== EXPERIMENT CONFIG ===")
    logger.info("mode: online SemanticCacheSystem")
    logger.info("offline_prefit_used: false")
    logger.info("npz: %s", args.npz)
    logger.info("output_dir: %s", str(output_dir))
    logger.info("query_field: %s", args.query_field)
    logger.info("label_field: %s", args.label_field)
    logger.info("anchor_field: %s", args.anchor_field)
    logger.info("query_embedding_field: %s", args.query_embedding_field)
    logger.info("target_fpr: %s", args.target_fpr)
    logger.info("top_k: %s", args.top_k)
    logger.info("max_rows: %s", args.max_rows)
    logger.info("batch_size: %s", args.batch_size)
    logger.info("min_h0: %s", args.min_h0)
    logger.info("min_h1: %s", args.min_h1)
    logger.info("parallelism: %s", args.parallelism)
    logger.info("fit_wait_secs: %s", args.fit_wait_secs)
    logger.info("warmup_rows: %s", args.warmup_rows)
    logger.info("progress_every: %s", args.progress_every)
    logger.info("persistence: %s", "enabled" if bool(args.persist) else "disabled (in-memory)")
    logger.info("ensemble_scorers: %s", ",".join(scorers))
    flush_logger(logger)

    rows, schema_report = load_rows(
        args.npz,
        query_field=args.query_field,
        label_field=args.label_field,
        anchor_field=args.anchor_field,
        embedding_field=args.query_embedding_field,
        max_rows=args.max_rows,
        seed=args.seed,
        include_uncertain=args.include_uncertain,
        selection=args.selection,
        allow_pickle=not args.no_allow_pickle,
    )

    label_dist = schema_report["label_distribution_selected"]
    if not args.include_uncertain:
        if int(label_dist.get("0", 0)) <= 0 or int(label_dist.get("1", 0)) <= 0:
            raise ValueError(
                "Selected rows must contain both label=0 and label=1 for online H0/H1 training. "
                "Try --selection random --seed 42 or increase --max-rows."
            )

    # Hard embedding validation: every selected embedding must share one dim.
    dims = {int(row.embedding.reshape(-1).shape[0]) for row in rows}
    if len(dims) != 1:
        logger.error("mixed embedding dimensions detected: %s", sorted(dims))
        raise ValueError(f"All selected embeddings must share one dimension; got {sorted(dims)}")
    embedding_dim = next(iter(dims))

    unique_anchors = len({row.group for row in rows})

    logger.info("=== DATASET SUMMARY ===")
    logger.info("total_rows: %s", schema_report["row_count_total"])
    logger.info("rows_selected: %s", schema_report["row_count_selected"])
    logger.info("rows_after_filtering: %s", len(rows))
    logger.info("label_distribution_before_filtering: %s", schema_report["label_distribution_before_filtering"])
    logger.info("label_distribution_after_filtering: %s", label_dist)
    logger.info("unique_anchors: %s", unique_anchors)
    logger.info("embedding_field: %s", args.query_embedding_field)
    logger.info("embedding_shape: [%d, %d]", len(rows), embedding_dim)
    logger.info("embedding_dtype: %s", str(rows[0].embedding.dtype) if rows else "none")
    logger.info("embedding_dim: %d", embedding_dim)
    logger.info("filtered_uncertain: %s", (not args.include_uncertain))
    if str(args.query_embedding_field) == "emb" and embedding_dim != EXPECTED_EMB_DIM_FOR_EMB_FIELD:
        logger.warning(
            "embedding field 'emb' has dim %d, expected %d", embedding_dim, EXPECTED_EMB_DIM_FOR_EMB_FIELD,
        )
    for row in rows[:3]:
        logger.info(
            "selected_row row_id=%d label=%d anchor=%s embedding_dim=%d text=%s",
            row.row_id, row.label, row.group, int(row.embedding.reshape(-1).shape[0]),
            row.text[:120].replace("\n", " "),
        )
    flush_logger(logger)

    rows_by_id = {row.row_id: row for row in rows}

    config = {
        "npz": str(args.npz),
        "output_dir": str(output_dir),
        "offline_prefit_used": False,
        "max_rows": args.max_rows,
        "seed": args.seed,
        "selection": args.selection,
        "include_uncertain": args.include_uncertain,
        "query_field": args.query_field,
        "label_field": args.label_field,
        "anchor_field": args.anchor_field,
        "query_embedding_field": args.query_embedding_field,
        "embedding_dim": embedding_dim,
        "target_fpr": args.target_fpr,
        "top_k": args.top_k,
        "batch_size": args.batch_size,
        "min_h0": args.min_h0,
        "min_h1": args.min_h1,
        "parallelism": args.parallelism,
        "fit_wait_secs": args.fit_wait_secs,
        "warmup_rows": args.warmup_rows,
        "progress_every": args.progress_every,
        "scorers": scorers,
        "persistence": bool(args.persist),
    }

    _write_json(output_dir / "schema_report.json", schema_report)

    lifecycle_path = output_dir / "lifecycle_events.jsonl"
    # Start the shared lifecycle JSONL clean for this run.
    if lifecycle_path.exists():
        lifecycle_path.unlink()

    runs = [
        OnlineRunSpec(name="cosine", scorer="cosine", scorers=None),
        OnlineRunSpec(name="ensemble", scorer="ensemble", scorers=scorers),
    ]

    results: dict[str, dict[str, Any]] = {}
    for spec in runs:
        results[spec.name] = run_one_online(
            spec=spec,
            rows=rows,
            rows_by_id=rows_by_id,
            output_dir=output_dir / spec.name,
            progress_csv_path=output_dir / f"{spec.name}_progress.csv",
            lifecycle_path=lifecycle_path,
            logger=logger,
            target_fpr=float(args.target_fpr),
            top_k=int(args.top_k),
            batch_size=int(args.batch_size),
            min_h0=int(args.min_h0),
            min_h1=int(args.min_h1),
            parallelism=int(args.parallelism),
            fit_wait_secs=float(args.fit_wait_secs),
            warmup_rows=int(args.warmup_rows),
            persistence=bool(args.persist),
            progress_every=int(args.progress_every),
            namespace_prefix=str(args.namespace_prefix),
            embedding_dim=embedding_dim,
        )

        # Copy the key per-method artifacts to the top-level output dir.
        src_report = output_dir / spec.name / f"{spec.name}_report.json"
        src_trace = output_dir / spec.name / f"{spec.name}_trace.csv"
        (output_dir / f"{spec.name}_report.json").write_text(src_report.read_text(encoding="utf-8"), encoding="utf-8")
        (output_dir / f"{spec.name}_trace.csv").write_text(src_trace.read_text(encoding="utf-8"), encoding="utf-8")

    summary = build_comparison_summary(
        config=config,
        dataset={"path": str(args.npz), **schema_report},
        cosine=results["cosine"],
        ensemble=results["ensemble"],
    )
    _write_json(output_dir / "comparison_summary.json", summary)
    write_markdown_report(output_dir / "comparison_report.md", summary)

    _log_comparison_summary(logger, results["cosine"], results["ensemble"], summary["winner"])
    logger.info("artifacts -> %s", str(output_dir.resolve()))
    flush_logger(logger)
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--npz", required=True, help="Path to data/h1h0_final.npz.")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--max-rows", type=int, default=2000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--selection", choices=["head", "random"], default="random")
    p.add_argument("--include-uncertain", action="store_true", help="Include label=-1 rows in the traffic stream.")
    p.add_argument("--query-field", default="text")
    p.add_argument("--label-field", default="label")
    p.add_argument("--anchor-field", default="global_cluster")
    p.add_argument("--query-embedding-field", default="emb")
    p.add_argument("--target-fpr", type=float, default=0.10)
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=20)
    p.add_argument("--min-h0", type=int, default=15)
    p.add_argument("--min-h1", type=int, default=8)
    p.add_argument("--parallelism", type=int, default=4)
    p.add_argument("--fit-wait-secs", type=float, default=60.0)
    p.add_argument("--warmup-rows", type=int, default=500)
    p.add_argument("--progress-every", type=int, default=100)
    p.add_argument("--scorers", default=",".join(FULL_ENSEMBLE))
    p.add_argument("--namespace-prefix", default="h1h0-online")
    # Persistence is OFF by default: each run starts from a clean in-memory
    # state so a reused --output-dir can never contaminate a run with a prior
    # run's cached vectors / calibrated threshold. Opt back in with --persist.
    p.add_argument("--persist", action="store_true",
                   help="Persist cache/vector state to disk (default: in-memory only).")
    p.add_argument("--no-file-persistence", action="store_true",
                   help="Deprecated no-op; persistence is already off unless --persist is given.")
    p.add_argument("--no-allow-pickle", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        run_comparison(args)
    except Exception as exc:
        print(f"compare_cosine_vs_ensemble.py failed: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
