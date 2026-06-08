"""Construct an MLCache with one constructor call and replay an H1/H0 NPZ stream.

This is the normal way to exercise the cache end to end: build it with
`MLCache.from_preset`, prefit any trainable scorers on H0/H1 examples, index
the anchors, and replay the incoming-query stream while writing the standard
experiment artifacts (summary_metrics.json, per_request_decisions.csv,
runtime_config.json, schema_report.json).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mlcache import (  # noqa: E402
    EMBEDDINGS_DEPENDENCIES_ERROR,
    ML_DEPENDENCIES_ERROR,
    MLCache,
    H1H0NPZDataset,
    H1H0NPZJudgeAdapter,
    H1H0NPZStreamAdapter,
    JudgeLabel,
    JudgeRequest,
    LabeledPairBatch,
    SCORER_PRESET_NAMES,
)
from mlcache.persistence import json_safe  # noqa: E402


class RunCacheError(Exception):
    """Raised for user-correctable input or environment problems."""


CSV_FIELDS = [
    "row_id",
    "query_id",
    "expected_anchor_key",
    "h0h1",
    "status",
    "accepted",
    "accepted_cache_key",
    "accepted_candidate_rank",
    "score",
    "threshold",
    "response_returned",
    "judge_label_for_accepted_candidate",
    "is_true_accept",
    "is_false_accept",
    "is_false_reject",
    "is_true_reject",
    "reason",
]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--npz", required=True, help="Path to the H1/H0 NPZ dataset.")
    parser.add_argument("--output-dir", required=True, help="Directory for experiment artifacts.")
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--query-field", default=None)
    parser.add_argument("--anchor-field", default=None)
    parser.add_argument("--label-field", default="label")
    parser.add_argument("--query-embedding-field", default=None)
    parser.add_argument("--anchor-embedding-field", default=None)
    parser.add_argument("--scorer", default="cosine", choices=SCORER_PRESET_NAMES)
    parser.add_argument(
        "--scorers",
        default=None,
        help="Comma-separated scorer presets used as ensemble members (only with --scorer ensemble).",
    )
    parser.add_argument("--pair-threshold", type=float, default=0.0)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--no-file-persistence", action="store_true")
    parser.add_argument("--no-allow-pickle", action="store_true")
    parser.add_argument("--progress-every", type=int, default=100)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        run(args)
        return 0
    except RunCacheError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    except ImportError as exc:
        print(str(exc), file=sys.stderr)
        return 2


def run(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = _load_dataset(args, output_dir)
    try:
        schema_report = {"schema_valid": True, "detected_schema": dataset.schema_report()}
        _write_json(output_dir / "schema_report.json", schema_report)

        records = dataset.records()
        stream = H1H0NPZStreamAdapter(dataset)
        anchors = stream.anchor_entries()
        judge = H1H0NPZJudgeAdapter(records)

        scorers = args.scorers.split(",") if args.scorers else None
        cache = _build_cache(args, output_dir, scorers=scorers)
        _prefit(cache, records)
        cache.set_threshold(args.pair_threshold)
        cache.index(anchors)

        decision_rows, counters = _replay(cache, stream, judge, progress_every=int(args.progress_every))
        _write_csv(output_dir / "per_request_decisions.csv", decision_rows)

        runtime_config = {
            "scorer": args.scorer,
            "scorers": scorers,
            "pair_threshold": float(args.pair_threshold),
            "top_k": int(args.top_k),
            "persistence": not args.no_file_persistence,
            "components": cache.components,
        }
        _write_json(output_dir / "runtime_config.json", runtime_config)

        summary = _summary_metrics(records=records, counters=counters, cache=cache)
        _write_json(output_dir / "summary_metrics.json", summary)
        print(json.dumps(json_safe(summary), indent=2, sort_keys=True))
        return summary
    finally:
        dataset.close()


def _load_dataset(args: argparse.Namespace, output_dir: Path) -> H1H0NPZDataset:
    try:
        return H1H0NPZDataset(
            args.npz,
            query_field=args.query_field,
            anchor_field=args.anchor_field,
            label_field=args.label_field,
            query_embedding_field=args.query_embedding_field,
            anchor_embedding_field=args.anchor_embedding_field,
            max_rows=args.max_rows,
            allow_pickle=not args.no_allow_pickle,
        )
    except Exception as exc:
        available = _available_npz_fields(args.npz, allow_pickle=not args.no_allow_pickle)
        message = str(exc)
        if "label" in message.lower() or args.label_field not in available:
            message = (
                f"{message}\nAvailable NPZ fields: {sorted(available)}. "
                f"If the dataset stores labels under 'label', pass --label-field label."
            )
        schema_report = {"schema_valid": False, "available_fields": sorted(available), "error": str(exc)}
        _write_json(output_dir / "schema_report.json", schema_report)
        _write_json(output_dir / "summary_metrics.json", {"status": "failed", "error": str(exc)})
        raise RunCacheError(message) from exc


def _available_npz_fields(path: str, *, allow_pickle: bool) -> set[str]:
    try:
        import numpy as np
    except ImportError as exc:
        raise ImportError(EMBEDDINGS_DEPENDENCIES_ERROR) from exc

    try:
        npz = np.load(path, allow_pickle=allow_pickle)
    except Exception:
        return set()
    try:
        return set(npz.files)
    finally:
        npz.close()


def _build_cache(args: argparse.Namespace, output_dir: Path, *, scorers: list[str] | None) -> MLCache:
    try:
        return MLCache.from_preset(
            root_dir=output_dir / "local_runtime_state",
            scorer=args.scorer,
            scorers=scorers,
            top_k=int(args.top_k),
            persistence=not args.no_file_persistence,
        )
    except ImportError as exc:
        raise RunCacheError(str(exc) or ML_DEPENDENCIES_ERROR) from exc


def _prefit(cache: MLCache, records: tuple[Any, ...]) -> None:
    h0 = [record.query_embedding for record in records if int(record.label) == 0]
    h1 = [record.query_embedding for record in records if int(record.label) == 1]
    if not h0 or not h1:
        return
    try:
        cache.prefit(LabeledPairBatch(h0=h0, h1=h1), alpha=0.05, seed=42)
    except ImportError as exc:
        raise RunCacheError(str(exc) or ML_DEPENDENCIES_ERROR) from exc


def _replay(
    cache: MLCache,
    stream: H1H0NPZStreamAdapter,
    judge: H1H0NPZJudgeAdapter,
    *,
    progress_every: int,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    rows: list[dict[str, Any]] = []
    counters = {
        "n_accepts": 0,
        "n_responses": 0,
        "n_true_accepts": 0,
        "n_false_accepts": 0,
        "n_false_rejects": 0,
        "n_true_rejects": 0,
    }
    for idx, (record, lookup) in enumerate(stream.iter_records_and_lookups(), start=1):
        result = cache.lookup_with_decision(lookup)
        accepted_label = _judge_accepted_candidate(cache, judge, record.query_id, lookup, result.decision.cache_key)
        row = _decision_row(record, result, accepted_label)
        rows.append(row)

        if row["accepted"]:
            counters["n_accepts"] += 1
        if result.response is not None:
            counters["n_responses"] += 1
        if row["is_true_accept"]:
            counters["n_true_accepts"] += 1
        if row["is_false_accept"]:
            counters["n_false_accepts"] += 1
        if row["is_false_reject"]:
            counters["n_false_rejects"] += 1
        if row["is_true_reject"]:
            counters["n_true_rejects"] += 1
        if progress_every > 0 and idx % progress_every == 0:
            print(f"replayed {idx} rows", flush=True)
    return rows, counters


def _judge_accepted_candidate(
    cache: MLCache,
    judge: H1H0NPZJudgeAdapter,
    query_id: str,
    lookup: Any,
    cache_key: Any,
) -> JudgeLabel | None:
    if cache_key is None:
        return None
    candidate = cache.runtime.vector_store.get(cache_key)
    result = judge.judge(
        JudgeRequest(
            query=lookup.query,
            candidate_key=cache_key,
            candidate_query=candidate.query if candidate is not None else None,
            context={"query_id": query_id, "request_metadata": lookup.metadata},
        )
    )
    return result.decision.label


def _decision_row(record: Any, result: Any, accepted_label: JudgeLabel | None) -> dict[str, Any]:
    decision = result.decision
    accepted = bool(decision.accepted)
    true_accept = accepted and accepted_label == JudgeLabel.REUSABLE
    false_accept = accepted and accepted_label == JudgeLabel.NOT_REUSABLE
    false_reject = (not accepted) and record.label == 1
    true_reject = (not accepted) and record.label == 0
    return {
        "row_id": record.row_id,
        "query_id": record.query_id,
        "expected_anchor_key": str(record.anchor_key),
        "h0h1": record.label,
        "status": decision.status.value,
        "accepted": accepted,
        "accepted_cache_key": str(decision.cache_key) if decision.cache_key is not None else "",
        "accepted_candidate_rank": decision.accepted_candidate_rank or "",
        "score": "" if decision.score is None else float(decision.score),
        "threshold": "" if decision.threshold is None else float(decision.threshold),
        "response_returned": result.response is not None,
        "judge_label_for_accepted_candidate": accepted_label.value if accepted_label is not None else "",
        "is_true_accept": bool(true_accept),
        "is_false_accept": bool(false_accept),
        "is_false_reject": bool(false_reject),
        "is_true_reject": bool(true_reject),
        "reason": decision.reason or "",
    }


def _summary_metrics(*, records: tuple[Any, ...], counters: dict[str, int], cache: MLCache) -> dict[str, Any]:
    n_requests = len(records)
    n_h0 = sum(1 for record in records if record.label == 0)
    n_h1 = sum(1 for record in records if record.label == 1)
    n_accepts = counters["n_accepts"]
    n_true_accepts = counters["n_true_accepts"]
    n_false_accepts = counters["n_false_accepts"]
    n_false_rejects = counters["n_false_rejects"]
    n_true_rejects = counters["n_true_rejects"]
    rejected = max(n_requests - n_accepts, 0)
    return {
        "status": "ok",
        "n_requests": n_requests,
        "n_h0": n_h0,
        "n_h1": n_h1,
        "n_accepts": n_accepts,
        "n_responses": counters["n_responses"],
        "n_true_accepts": n_true_accepts,
        "n_false_accepts": n_false_accepts,
        "n_false_rejects": n_false_rejects,
        "n_true_rejects": n_true_rejects,
        "stream_error_rate": _safe_rate(n_false_accepts, n_requests),
        "accept_error_rate": _safe_rate(n_false_accepts, max(n_accepts, 1)),
        "hit_rate": _safe_rate(n_accepts, n_requests),
        "h1_recall": _safe_rate(n_true_accepts, max(n_h1, 1)),
        "h0_rejection_rate": _safe_rate(n_true_rejects, max(n_h0, 1)),
        "abstain_or_miss_rate": _safe_rate(rejected, n_requests),
        "threshold": None if cache.threshold is None else float(cache.threshold),
    }


def _safe_rate(numer: int, denom: int) -> float:
    if denom <= 0:
        return 0.0
    return float(numer) / float(denom)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"{json.dumps(json_safe(data), indent=2, sort_keys=True)}\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
