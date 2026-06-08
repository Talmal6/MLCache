"""Replay an H1/H0 NPZ dataset through a local MLCache runtime."""

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

from mlcache.calibration import (  # noqa: E402
    DefaultQueryLevelCalibrationBuilder,
    QueryLevelCalibrationConfig,
    ThresholdScope,
)
from mlcache.features import NormalizedHadamardFeatureBuilder  # noqa: E402
from mlcache.feedback import (  # noqa: E402
    H1H0NPZDataset,
    H1H0NPZJudgeAdapter,
    H1H0NPZStreamAdapter,
    JudgeLabel,
    JudgeRequest,
)
from mlcache.persistence import json_safe  # noqa: E402
from mlcache.policies import QueryLevelPolicyMode  # noqa: E402
from mlcache.runtime import (  # noqa: E402
    MLCacheRuntime,
    MLCacheRuntimeConfig,
    QueryLevelRuntimeConfig,
    RuntimeRefitConfig,
    ShadowRuntimeConfig,
    build_local_mlcache_runtime,
)
from mlcache.scorers import CosineScorer, EnsembleScorer, LDAScorer, PCAWhitenedCosineScorer  # noqa: E402
from mlcache.semantic_types import CacheLookup, ScorerName, Threshold  # noqa: E402
from mlcache.semantic_types import LabeledPairBatch  # noqa: E402


class ExperimentInputError(Exception):
    """Raised when the experiment input data cannot be adapted."""


CSV_FIELDS = [
    "row_id",
    "query_id",
    "expected_anchor_key",
    "h0h1",
    "pair_level_status",
    "pair_level_accepted",
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
    "final_decision_source",
    "query_level_status",
    "query_level_accepted",
    "query_level_selected_candidate_key",
    "query_level_selected_candidate_rank",
    "query_level_selected_score",
    "query_level_threshold",
    "fallback_used",
]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--npz", required=True, help="Path to the H1/H0 NPZ dataset.")
    parser.add_argument("--output-dir", required=True, help="Directory for experiment artifacts.")
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--mode",
        choices=("pair_level", "shadow_query_level", "active_query_level"),
        default="pair_level",
    )
    parser.add_argument("--query-field", default=None)
    parser.add_argument("--anchor-field", default=None)
    parser.add_argument("--label-field", default="h0h1")
    parser.add_argument("--query-embedding-field", default=None)
    parser.add_argument("--anchor-embedding-field", default=None)
    parser.add_argument("--pair-threshold", type=float, default=0.0)
    parser.add_argument("--query-level-threshold", type=float, default=None)
    parser.add_argument("--no-file-persistence", action="store_true")
    parser.add_argument("--no-allow-pickle", action="store_true")
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--schema-only", action="store_true")
    parser.add_argument(
        "--compare-scorers",
        action="store_true",
        help="Replay cosine first, then ensemble, and write a comparison report.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        run_experiment(args)
        return 0
    except (ExperimentInputError, NotImplementedError) as exc:
        print(str(exc), file=sys.stderr)
        return 2


def run_experiment(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_schema = inspect_npz(args.npz, allow_pickle=not args.no_allow_pickle)
    _log("loaded schema")

    if args.schema_only:
        schema_report = _schema_only_report(args, raw_schema)
        _write_json(output_dir / "schema_report.json", schema_report)
        _write_json(output_dir / "summary_metrics.json", {"status": "schema_only"})
        print(json.dumps(json_safe(schema_report), indent=2, sort_keys=True))
        _log("wrote summary")
        return schema_report

    try:
        dataset = H1H0NPZDataset(
            args.npz,
            query_field=args.query_field,
            anchor_field=args.anchor_field,
            label_field=args.label_field,
            query_embedding_field=args.query_embedding_field,
            anchor_embedding_field=args.anchor_embedding_field,
            max_rows=args.max_rows,
            allow_pickle=not args.no_allow_pickle,
        )
        _log(f"materialized {len(dataset.records())} records")
    except Exception as exc:
        schema_report = {
            "schema_valid": False,
            "raw_npz_schema": raw_schema,
            "error": str(exc),
        }
        _write_json(output_dir / "schema_report.json", schema_report)
        _write_json(
            output_dir / "summary_metrics.json",
            {
                "status": "failed",
                "reason": "dataset_schema_invalid",
                "error": str(exc),
            },
        )
        print(json.dumps(json_safe(schema_report), indent=2, sort_keys=True))
        raise ExperimentInputError(str(exc)) from exc

    try:
        schema_report = {
            "schema_valid": True,
            "raw_npz_schema": raw_schema,
            "detected_schema": dataset.schema_report(),
        }
        _write_json(output_dir / "schema_report.json", schema_report)
        print(json.dumps(json_safe(schema_report), indent=2, sort_keys=True))

        if args.mode == "active_query_level":
            raise NotImplementedError(
                "active_query_level runner mode is not implemented in this experiment runner yet; "
                "pair_level and shadow_query_level are supported."
            )

        stream = H1H0NPZStreamAdapter(dataset)
        records = stream.records()
        anchors = stream.anchor_entries()
        _log(f"built {len(anchors)} unique anchors")
        if args.compare_scorers:
            comparison = _run_scorer_comparison(
                args=args,
                output_dir=output_dir,
                dataset=dataset,
                records=records,
                anchors=anchors,
            )
            _write_json(output_dir / "comparison_report.json", comparison)
            _write_json(output_dir / "summary_metrics.json", comparison)
            print(json.dumps(json_safe(comparison), indent=2, sort_keys=True))
            _log("wrote comparison")
            return comparison

        summary = _run_single_scorer_experiment(
            args=args,
            output_dir=output_dir,
            dataset=dataset,
            records=records,
            anchors=anchors,
            scorer=CosineScorer(),
        )
        print(json.dumps(json_safe(summary), indent=2, sort_keys=True))
        _log("wrote summary")
        return summary
    finally:
        dataset.close()


def _run_single_scorer_experiment(
    *,
    args: argparse.Namespace,
    output_dir: Path,
    dataset: H1H0NPZDataset,
    records: tuple[Any, ...],
    anchors: tuple[Any, ...],
    scorer: Any,
) -> dict[str, Any]:
    judge = H1H0NPZJudgeAdapter(records)
    stream = H1H0NPZStreamAdapter(dataset)
    _prefit_scorer(scorer, records, seed=42, alpha=0.05)
    runtime_config = _runtime_config(args)
    runtime = build_local_mlcache_runtime(
        root_dir=output_dir / "local_runtime_state",
        feature_builder=NormalizedHadamardFeatureBuilder(),
        scorer=scorer,
        judge=judge,
        config=runtime_config,
        use_file_persistence=not args.no_file_persistence,
    )
    _clear_runtime_state(runtime)
    _set_pair_threshold(runtime, scorer.name, Threshold(float(args.pair_threshold)))
    _index_anchors(runtime, anchors)
    _log(f"indexed anchors for {scorer.name}")

    decision_rows, counters = _replay(runtime, stream, judge, progress_every=int(args.progress_every))
    _write_csv(output_dir / "per_request_decisions.csv", decision_rows)
    calibration_report = _query_level_calibration_report(runtime)
    _write_json(output_dir / "query_level_calibration_report.json", calibration_report)
    _write_json(output_dir / "runtime_config.json", runtime_config)
    summary = _summary_metrics(
        records=records,
        counters=counters,
        runtime=runtime,
        query_level_calibration_report=calibration_report,
    )
    _write_json(output_dir / "summary_metrics.json", summary)
    return {
        "scorer": str(scorer.name),
        "output_dir": str(output_dir),
        "summary": summary,
        "decision_rows": decision_rows,
    }


def _prefit_scorer(scorer: Any, records: tuple[Any, ...], *, seed: int, alpha: float) -> None:
    h0 = [record.query_embedding for record in records if int(record.label) == 0]
    h1 = [record.query_embedding for record in records if int(record.label) == 1]
    if not h0 or not h1:
        return
    scorer.fit(
        LabeledPairBatch(
            h0=h0,
            h1=h1,
        ),
        alpha=alpha,
        seed=seed,
    )


def _build_compare_ensemble_scorer() -> EnsembleScorer:
    return EnsembleScorer(
        judges=[
            CosineScorer(),
            PCAWhitenedCosineScorer(),
            LDAScorer(),
        ]
    )


def _run_scorer_comparison(
    *,
    args: argparse.Namespace,
    output_dir: Path,
    dataset: H1H0NPZDataset,
    records: tuple[Any, ...],
    anchors: tuple[Any, ...],
) -> dict[str, Any]:
    runs = [
        _run_single_scorer_experiment(
            args=args,
            output_dir=output_dir / "cosine",
            dataset=dataset,
            records=records,
            anchors=anchors,
            scorer=CosineScorer(),
        ),
        _run_single_scorer_experiment(
            args=args,
            output_dir=output_dir / "ensemble",
            dataset=dataset,
            records=records,
            anchors=anchors,
            scorer=_build_compare_ensemble_scorer(),
        ),
    ]

    baseline, challenger = runs
    return {
        "status": "compared",
        "order": [run["scorer"] for run in runs],
        "runs": [
            {
                "scorer": run["scorer"],
                "output_dir": run["output_dir"],
                "summary": run["summary"],
            }
            for run in runs
        ],
        "summary_deltas": _compare_summaries(baseline["summary"], challenger["summary"]),
        "decision_comparison": _compare_decisions(baseline["decision_rows"], challenger["decision_rows"]),
    }


def _compare_summaries(first: dict[str, Any], second: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "n_accepts",
        "n_true_accepts",
        "n_false_accepts",
        "n_false_rejects",
        "n_true_rejects",
        "stream_error_rate",
        "accept_error_rate",
        "hit_rate",
        "h1_recall",
        "h0_rejection_rate",
        "abstain_or_miss_rate",
    )
    return {key: _numeric_delta(second.get(key), first.get(key)) for key in keys}


def _compare_decisions(first_rows: list[dict[str, Any]], second_rows: list[dict[str, Any]]) -> dict[str, Any]:
    row_count = min(len(first_rows), len(second_rows))
    if row_count <= 0:
        return {"row_count": 0, "agreement_rate": 0.0, "disagreement_count": 0}

    agreement_count = 0
    disagreement_count = 0
    for first_row, second_row in zip(first_rows, second_rows):
        if bool(first_row["pair_level_accepted"]) == bool(second_row["pair_level_accepted"]):
            agreement_count += 1
        else:
            disagreement_count += 1

    return {
        "row_count": row_count,
        "agreement_rate": _safe_rate(agreement_count, row_count),
        "disagreement_count": disagreement_count,
    }


def _numeric_delta(second: Any, first: Any) -> float | None:
    try:
        if second is None or first is None:
            return None
        return float(second) - float(first)
    except Exception:
        return None


def inspect_npz(path: str | Path, *, allow_pickle: bool = True) -> dict[str, Any]:
    try:
        import numpy as np
    except Exception as exc:
        raise ImportError("run_h1h0_local_runtime.py requires numpy") from exc

    fields = _npz_header_fields(path)
    label_fields = {field["name"] for field in fields if field["name"] in {"h0h1", "label"}}
    if label_fields:
        npz = np.load(path, allow_pickle=allow_pickle)
        try:
            for field in fields:
                if field["name"] not in label_fields or field["ndim"] != 1:
                    continue
                labels = npz[field["name"]]
                values, counts = np.unique(labels, return_counts=True)
                field["distribution"] = {
                    str(_json_scalar(value)): int(count)
                    for value, count in zip(values, counts)
                }
        finally:
            npz.close()

    return {
        "path": str(path),
        "available_fields": [field["name"] for field in fields],
        "fields": fields,
    }


def _schema_only_report(args: argparse.Namespace, raw_schema: dict[str, Any]) -> dict[str, Any]:
    fields = {field["name"]: field for field in raw_schema["fields"]}
    available = tuple(raw_schema["available_fields"])
    query_field = _choose_field(args.query_field, H1H0NPZDataset.QUERY_FIELD_CANDIDATES, available, "query field")
    label_field = _choose_label_field(args.label_field, available)
    anchor_field = _choose_field(
        args.anchor_field,
        H1H0NPZDataset.ANCHOR_FIELD_CANDIDATES,
        available,
        "anchor field",
        fallback=query_field,
    )
    query_embedding_field = _choose_field(
        args.query_embedding_field,
        H1H0NPZDataset.QUERY_EMBEDDING_FIELD_CANDIDATES,
        available,
        "query embedding field",
    )
    anchor_embedding_field = args.anchor_embedding_field
    if anchor_embedding_field is not None and anchor_embedding_field not in available:
        raise ExperimentInputError(
            f"NPZ missing required anchor_embedding_field={anchor_embedding_field!r}; available fields: {available}"
        )

    total_rows = int(fields[query_field]["shape"][0])
    selected_rows = total_rows if args.max_rows is None else min(int(args.max_rows), total_rows)
    embedding_shape = fields[query_embedding_field]["shape"]
    embedding_dim = embedding_shape[1] if len(embedding_shape) > 1 else None
    anchor_counts = _anchor_counts(
        args.npz,
        anchor_field=anchor_field,
        selected_rows=selected_rows,
        allow_pickle=not args.no_allow_pickle,
    )
    return {
        "schema_valid": True,
        "schema_only": True,
        "raw_npz_schema": raw_schema,
        "chosen_fields": {
            "query_field": query_field,
            "anchor_field": anchor_field,
            "label_field": label_field,
            "query_embedding_field": query_embedding_field,
            "anchor_embedding_field": anchor_embedding_field,
        },
        "row_count": total_rows,
        "selected_row_count": selected_rows,
        "embedding_dim": embedding_dim,
        **anchor_counts,
    }


def _choose_field(
    explicit: str | None,
    candidates: tuple[str, ...],
    available: tuple[str, ...],
    purpose: str,
    *,
    fallback: str | None = None,
) -> str:
    if explicit is not None:
        if explicit not in available:
            raise ExperimentInputError(f"NPZ missing required {purpose}={explicit!r}; available fields: {available}")
        return explicit
    matches = tuple(field for field in candidates if field in available)
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise ExperimentInputError(f"Ambiguous {purpose} fields {matches}; provide an explicit field name")
    if fallback is not None:
        return fallback
    raise ExperimentInputError(f"Could not detect {purpose}; available fields: {available}")


def _choose_label_field(explicit: str, available: tuple[str, ...]) -> str:
    if explicit in available:
        return explicit
    if explicit == "h0h1" and "label" in available:
        return "label"
    raise ExperimentInputError(f"NPZ missing required label_field={explicit!r}; available fields: {available}")


def _anchor_counts(
    path: str | Path,
    *,
    anchor_field: str,
    selected_rows: int,
    allow_pickle: bool,
) -> dict[str, Any]:
    try:
        import numpy as np
    except Exception as exc:
        raise ImportError("run_h1h0_local_runtime.py requires numpy") from exc

    npz = np.load(path, allow_pickle=allow_pickle)
    try:
        values = npz[anchor_field]
        return {
            "unique_anchor_count": int(len(np.unique(values))),
            "selected_unique_anchor_count": int(len(np.unique(values[:selected_rows]))),
        }
    finally:
        npz.close()


def _npz_header_fields(path: str | Path) -> list[dict[str, Any]]:
    import zipfile

    import numpy as np

    fields: list[dict[str, Any]] = []
    with zipfile.ZipFile(path) as archive:
        for member in archive.namelist():
            if not member.endswith(".npy"):
                continue
            name = member[:-4]
            with archive.open(member) as handle:
                version = np.lib.format.read_magic(handle)
                if version == (1, 0):
                    shape, _fortran_order, dtype = np.lib.format.read_array_header_1_0(handle)
                else:
                    shape, _fortran_order, dtype = np.lib.format.read_array_header_2_0(handle)
            fields.append(
                {
                    "name": name,
                    "shape": list(shape),
                    "dtype": str(dtype),
                    "ndim": len(shape),
                }
            )
    return fields


def _log(message: str) -> None:
    print(message, flush=True)


def _runtime_config(args: argparse.Namespace) -> MLCacheRuntimeConfig:
    query_level_threshold = (
        Threshold(float(args.query_level_threshold)) if args.query_level_threshold is not None else None
    )
    query_level = QueryLevelRuntimeConfig()
    shadow = ShadowRuntimeConfig(enabled=False)
    if args.mode == "shadow_query_level":
        shadow = ShadowRuntimeConfig(
            enabled=True,
            top_k=int(args.top_k),
            max_pairs_per_request=int(args.top_k),
            calibration_every_n=5,
            collect_uncertain=False,
        )
        query_level = QueryLevelRuntimeConfig(
            enabled=True,
            mode=QueryLevelPolicyMode.SHADOW,
            threshold=query_level_threshold,
            require_threshold=True,
        )

    return MLCacheRuntimeConfig(
        shadow=shadow,
        refit=RuntimeRefitConfig(auto_refit=False, top_k=int(args.top_k)),
        query_level=query_level,
        metadata={
            "source": "h1h0_local_runtime_runner",
            "mode": args.mode,
            "top_k": int(args.top_k),
        },
    )


def _clear_runtime_state(runtime: MLCacheRuntime) -> None:
    for component in (
        runtime.kv_store,
        runtime.vector_store,
        runtime.query_record_store,
        runtime.query_level_shadow_store,
    ):
        if component is not None and hasattr(component, "clear"):
            component.clear()


def _set_pair_threshold(runtime: MLCacheRuntime, scorer_name: ScorerName, threshold: Threshold) -> None:
    runtime.oracle._threshold = threshold
    provider = getattr(runtime.oracle, "threshold_provider", None)
    if provider is not None:
        provider.set_threshold(
            threshold,
            scorer=scorer_name,
            scope=ThresholdScope.GLOBAL,
            context={"source": "h1h0_local_runtime_pair_threshold"},
        )


def _index_anchors(runtime: MLCacheRuntime, anchors: tuple[Any, ...]) -> None:
    for entry in anchors:
        runtime.put(entry)


def _replay(
    runtime: MLCacheRuntime,
    stream: H1H0NPZStreamAdapter,
    judge: H1H0NPZJudgeAdapter,
    *,
    progress_every: int = 100,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    rows: list[dict[str, Any]] = []
    counters = {
        "n_accepts": 0,
        "n_responses": 0,
        "n_true_accepts": 0,
        "n_false_accepts": 0,
        "n_false_rejects": 0,
        "n_true_rejects": 0,
        "n_uncertain_accepts": 0,
    }
    for idx, (record, lookup) in enumerate(stream.iter_records_and_lookups(), start=1):
        result = runtime.lookup_with_decision(lookup)
        accepted_label = _judge_accepted_candidate(runtime, judge, record.query_id, lookup, result.decision.cache_key)
        row = _decision_row(record, result, accepted_label)
        rows.append(row)

        accepted = bool(result.decision.accepted)
        if accepted:
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
        if accepted and accepted_label == JudgeLabel.UNCERTAIN:
            counters["n_uncertain_accepts"] += 1
        if progress_every > 0 and idx % progress_every == 0:
            _log(f"replayed {idx} rows")
    if rows and (progress_every <= 0 or len(rows) % progress_every != 0):
        _log(f"replayed {len(rows)} rows")
    return rows, counters


def _judge_accepted_candidate(
    runtime: MLCacheRuntime,
    judge: H1H0NPZJudgeAdapter,
    query_id: str,
    lookup: CacheLookup,
    cache_key: Any,
) -> JudgeLabel | None:
    if cache_key is None:
        return None
    candidate = runtime.vector_store.get(cache_key)
    result = judge.judge(
        JudgeRequest(
            query=lookup.query,
            candidate_key=cache_key,
            candidate_query=candidate.query if candidate is not None else None,
            context={
                "query_id": query_id,
                "request_metadata": lookup.metadata,
            },
        )
    )
    return result.decision.label


def _decision_row(record: Any, result: Any, accepted_label: JudgeLabel | None) -> dict[str, Any]:
    decision = result.decision
    metadata = result.metadata
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
        "pair_level_status": metadata.get("pair_level_status", decision.status.value),
        "pair_level_accepted": metadata.get("pair_level_accepted", accepted),
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
        "reason": metadata.get("reason", decision.reason) or "",
        "final_decision_source": metadata.get("final_decision_source", "pair_level"),
        "query_level_status": metadata.get("query_level_status", ""),
        "query_level_accepted": metadata.get("query_level_accepted", ""),
        "query_level_selected_candidate_key": metadata.get("query_level_selected_candidate_key", ""),
        "query_level_selected_candidate_rank": metadata.get("query_level_selected_candidate_rank", ""),
        "query_level_selected_score": metadata.get("query_level_selected_score", ""),
        "query_level_threshold": metadata.get("query_level_threshold", ""),
        "fallback_used": metadata.get("fallback_used", ""),
    }


def _query_level_calibration_report(runtime: MLCacheRuntime) -> dict[str, Any]:
    store = runtime.query_record_store
    if store is None:
        return {"status": "not_run", "reason": "query_record_store_missing", "record_count": 0}
    records = store.records()
    if not records:
        return {"status": "not_run", "reason": "no_query_records", "record_count": 0}

    builder = DefaultQueryLevelCalibrationBuilder(
        config=QueryLevelCalibrationConfig(
            target_false_accept_rate=0.05,
            min_selected_h0=500,
        )
    )
    dataset = builder.build_calibration_decisions(records, metadata={"source": "h1h0_local_runtime_runner"})
    result = builder.calibrate(dataset, metadata={"source": "h1h0_local_runtime_runner"})
    return {
        "status": "calibrated" if result.threshold is not None else "gate_failed",
        "record_count": len(records),
        "decision_count": len(result.dataset.decisions),
        "selected_h0_count": len(result.dataset.h0_scores),
        "all_pair_h0_count": len(result.dataset.all_pair_h0_scores),
        "threshold": None if result.threshold is None else float(result.threshold),
        "target_false_accept_rate": result.target_false_accept_rate,
        "metadata": result.metadata,
    }


def _summary_metrics(
    *,
    records: tuple[Any, ...],
    counters: dict[str, int],
    runtime: MLCacheRuntime,
    query_level_calibration_report: dict[str, Any],
) -> dict[str, Any]:
    n_requests = len(records)
    n_h0 = sum(1 for record in records if record.label == 0)
    n_h1 = sum(1 for record in records if record.label == 1)
    n_accepts = counters["n_accepts"]
    n_false_accepts = counters["n_false_accepts"]
    n_true_accepts = counters["n_true_accepts"]
    n_false_rejects = counters["n_false_rejects"]
    n_true_rejects = counters["n_true_rejects"]
    rejected = max(n_requests - n_accepts, 0)
    query_record_count = (
        len(runtime.query_record_store.records()) if runtime.query_record_store is not None else None
    )
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
        "n_uncertain_accepts": counters["n_uncertain_accepts"],
        "stream_error_rate": _safe_rate(n_false_accepts, n_requests),
        "accept_error_rate": _safe_rate(n_false_accepts, max(n_accepts, 1)),
        "hit_rate": _safe_rate(n_accepts, n_requests),
        "h1_recall": _safe_rate(n_true_accepts, max(n_h1, 1)),
        "h0_rejection_rate": _safe_rate(n_true_rejects, max(n_h0, 1)),
        "abstain_or_miss_rate": _safe_rate(rejected, n_requests),
        "runtime_activation_status": runtime.activation_status,
        "shadow_snapshot": runtime.shadow_snapshot,
        "query_record_count": query_record_count,
        "query_level_threshold": query_level_calibration_report.get("threshold"),
        "query_level_calibration_metadata": query_level_calibration_report.get("metadata"),
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


def _json_scalar(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    return value


if __name__ == "__main__":
    raise SystemExit(main())
