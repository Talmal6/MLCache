"""Compare cold-start TPR/FPR of two cache systems calibrated to the same FPR budget.

System A ("hadamard_ensemble") is the existing `MLCache.from_preset(scorer="ensemble", ...)`
pipeline: a `NormalizedHadamardFeatureBuilder` feeding a weighted `EnsembleScorer`
of cosine, LDA, PCA-whitened-cosine, XGBoost, and a tiny MLP.

System B ("cosine_only") is the same pipeline with `scorer="cosine"` — a single
unlearned cosine-similarity judge over the same Hadamard pair features.

Both systems are built from the same H1/H0 dataset split:

  1. A *calibration* slice is adapted into judged pairs (the dataset's
     ground-truth H0/H1 labels stand in for a live judge) and handed to
     `cache.prefit_and_calibrate(...)`, which owns the entire cold-start
     sequence — fit trainable scorers, score H0 pairs, and pick a pair-level
     threshold via Neyman-Pearson calibration that targets a shared
     false-accept-rate budget (`--target-fpr`). This experiment configures and
     reports; it does not implement calibration or training itself.
  2. A disjoint *cold-start* slice — the system's first live traffic right
     after calibration, with no online refit in between — is replayed through
     `cache.lookup_with_decision`. Empirical TPR (fraction of H1 "should reuse"
     pairs the cache accepts) and FPR (fraction of H0 "should not reuse" pairs
     the cache accepts) are measured directly from those decisions.

The report compares both systems' calibrated thresholds and cold-start
TPR/FPR so you can see how much the richer Hadamard+ensemble scorer buys you
over plain cosine when both are held to the same FPR budget at cold start.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mlcache import (  # noqa: E402
    CacheLookup,
    EMBEDDINGS_DEPENDENCIES_ERROR,
    H1H0NPZDataset,
    H1H0NPZStreamAdapter,
    JudgeDecision,
    JudgedPairExample,
    JudgeLabel,
    JudgeRequest,
    ML_DEPENDENCIES_ERROR,
    MLCache,
    NormalizedHadamardFeatureBuilder,
)
from mlcache.persistence import json_safe  # noqa: E402


class ExperimentError(Exception):
    """Raised for user-correctable input or environment problems."""


@dataclass(frozen=True)
class SystemSpec:
    name: str
    scorer: str
    scorers: tuple[str, ...] | None = None


SYSTEMS: tuple[SystemSpec, ...] = (
    SystemSpec(
        name="hadamard_ensemble",
        scorer="ensemble",
        scorers=("cosine", "lda", "pca_whitened_cosine", "xgboost", "mlp"),
    ),
    SystemSpec(name="cosine_only", scorer="cosine"),
)

COLD_START_CSV_FIELDS = [
    "system",
    "row_id",
    "query_id",
    "h0h1",
    "status",
    "accepted",
    "score",
    "threshold",
]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--npz", default=str(ROOT.parent / "data" / "h1h0_final.npz"))
    parser.add_argument("--output-dir", default=str(ROOT / "experiments" / "cold_start_tpr_fpr"))
    parser.add_argument("--label-field", default="label")
    parser.add_argument("--query-field", default="text")
    parser.add_argument("--anchor-field", default="global_cluster")
    parser.add_argument("--query-embedding-field", default="emb")
    parser.add_argument("--target-fpr", type=float, default=0.05)
    parser.add_argument("--calibration-rows", type=int, default=400)
    parser.add_argument("--cold-start-rows", type=int, default=400)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--no-file-persistence", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        run(args)
        return 0
    except ExperimentError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    except ImportError as exc:
        print(str(exc), file=sys.stderr)
        return 2


def run(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = _load_dataset(args)
    try:
        records = dataset.records()
        calibration_records, cold_start_records = _split_records(
            records,
            calibration_rows=int(args.calibration_rows),
            cold_start_rows=int(args.cold_start_rows),
        )
        anchors = H1H0NPZStreamAdapter(dataset).anchor_entries()

        feature_builder = NormalizedHadamardFeatureBuilder()
        cold_start_rows: list[dict[str, Any]] = []
        systems: dict[str, Any] = {}
        for spec in SYSTEMS:
            system_report, decision_rows = _run_system(
                spec,
                args=args,
                output_dir=output_dir,
                feature_builder=feature_builder,
                calibration_records=calibration_records,
                cold_start_records=cold_start_records,
                anchors=anchors,
            )
            systems[spec.name] = system_report
            cold_start_rows.extend(decision_rows)

        report = {
            "status": "ok",
            "target_fpr": float(args.target_fpr),
            "calibration_rows": len(calibration_records),
            "cold_start_rows": len(cold_start_records),
            "systems": systems,
            "comparison": _compare(systems),
        }

        _write_csv(output_dir / "cold_start_decisions.csv", cold_start_rows)
        _write_json(output_dir / "cold_start_tpr_fpr_report.json", report)
        print(json.dumps(json_safe(report), indent=2, sort_keys=True))
        return report
    finally:
        dataset.close()


def _load_dataset(args: argparse.Namespace) -> H1H0NPZDataset:
    try:
        return H1H0NPZDataset(
            args.npz,
            query_field=args.query_field,
            anchor_field=args.anchor_field,
            label_field=args.label_field,
            query_embedding_field=args.query_embedding_field,
            max_rows=args.max_rows,
        )
    except KeyError as exc:
        try:
            available = _available_npz_fields(args.npz)
        except ImportError as inner_exc:
            raise ImportError(EMBEDDINGS_DEPENDENCIES_ERROR) from inner_exc
        raise ExperimentError(
            f"Could not find label field {args.label_field!r} in {args.npz!r}. "
            f"Available fields: {sorted(available)}. "
            "If the dataset stores labels under 'label', pass --label-field label."
        ) from exc


def _available_npz_fields(path: str) -> set[str]:
    try:
        import numpy as np
    except ImportError as exc:
        raise ImportError(EMBEDDINGS_DEPENDENCIES_ERROR) from exc

    npz = np.load(path, allow_pickle=True)
    try:
        return set(npz.files)
    finally:
        npz.close()


def _split_records(
    records: tuple[Any, ...],
    *,
    calibration_rows: int,
    cold_start_rows: int,
) -> tuple[tuple[Any, ...], tuple[Any, ...]]:
    calibration = records[:calibration_rows]
    cold_start = records[calibration_rows : calibration_rows + cold_start_rows]
    if not calibration or not cold_start:
        raise ExperimentError(
            f"Dataset has only {len(records)} rows; need at least "
            f"{calibration_rows} calibration rows followed by {cold_start_rows} "
            "cold-start rows. Lower --calibration-rows/--cold-start-rows or use "
            "--max-rows to confirm dataset size."
        )
    if not any(int(r.label) == 0 for r in calibration) or not any(int(r.label) == 1 for r in calibration):
        raise ExperimentError(
            "Calibration split must contain both H0 and H1 examples to calibrate "
            "a Neyman-Pearson threshold; widen --calibration-rows."
        )
    return calibration, cold_start


def _run_system(
    spec: SystemSpec,
    *,
    args: argparse.Namespace,
    output_dir: Path,
    feature_builder: NormalizedHadamardFeatureBuilder,
    calibration_records: tuple[Any, ...],
    cold_start_records: tuple[Any, ...],
    anchors: tuple[Any, ...],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    try:
        cache = MLCache.from_preset(
            root_dir=output_dir / spec.name / "local_runtime_state",
            scorer=spec.scorer,
            scorers=list(spec.scorers) if spec.scorers else None,
            top_k=int(args.top_k),
            persistence=not args.no_file_persistence,
        )
        # The cache owns the whole cold-start sequence — fit (if trainable),
        # score H0 pairs, and pick a Neyman-Pearson threshold for the shared
        # FPR budget — from judged pairs alone; this experiment never scores
        # pairs or calls `scorer.calibrate(...)` itself.
        judged_pairs = _judged_pairs(calibration_records, feature_builder=feature_builder)
        calibration_report = cache.prefit_and_calibrate(
            judged_pairs,
            target_fpr=float(args.target_fpr),
            fit_kwargs={"alpha": 0.05, "seed": 42},
        )
    except ImportError as exc:
        raise ExperimentError(str(exc) or ML_DEPENDENCIES_ERROR) from exc

    cache.index(anchors)

    cold_start_report, decision_rows = _replay_cold_start(cache, cold_start_records, system_name=spec.name)

    system_report = {
        "scorer": spec.scorer,
        "scorers": list(spec.scorers) if spec.scorers else None,
        "calibrated_threshold": float(calibration_report["threshold"]),
        "calibration": calibration_report,
        "cold_start": cold_start_report,
    }
    return system_report, decision_rows


def _judged_pairs(
    records: tuple[Any, ...],
    *,
    feature_builder: NormalizedHadamardFeatureBuilder,
) -> list[JudgedPairExample]:
    """Adapt H1/H0 dataset rows into judged pairs for `cache.prefit_and_calibrate`.

    The dataset's ground-truth labels stand in for a live judge's verdicts —
    there is no judge to call here, only a labeled offline dataset to replay.
    """

    pairs: list[JudgedPairExample] = []
    for record in records:
        features = feature_builder.build(record.query_embedding, record.anchor_embedding)
        label = JudgeLabel.REUSABLE if int(record.label) == 1 else JudgeLabel.NOT_REUSABLE
        request = JudgeRequest(
            query=record.query,
            candidate_query=record.anchor_query,
            candidate_key=record.anchor_key,
        )
        pairs.append(
            JudgedPairExample(
                features=tuple(float(value) for value in features.hadamard),
                request=request,
                decision=JudgeDecision(label=label),
                metadata={"row_id": record.row_id, "query_id": record.query_id},
            )
        )
    return pairs


def _replay_cold_start(
    cache: MLCache,
    records: tuple[Any, ...],
    *,
    system_name: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Replay the system's first live queries and score TPR/FPR against ground truth.

    Each record's `label` is the ground truth for whether reusing a cached
    response is semantically correct (H1=1) or not (H0=0); the cache's
    `accepted` decision is the prediction. No online feedback or refit happens
    between calibration and this replay — this *is* cold start.
    """

    n_h0 = n_h1 = 0
    n_h0_accept = n_h1_accept = 0
    rows: list[dict[str, Any]] = []
    for record in records:
        lookup = CacheLookup(query=record.query, embedding=record.query_embedding)
        result = cache.lookup_with_decision(lookup)
        decision = result.decision
        accepted = bool(decision.accepted)

        if int(record.label) == 1:
            n_h1 += 1
            if accepted:
                n_h1_accept += 1
        else:
            n_h0 += 1
            if accepted:
                n_h0_accept += 1

        rows.append(
            {
                "system": system_name,
                "row_id": record.row_id,
                "query_id": record.query_id,
                "h0h1": record.label,
                "status": decision.status.value,
                "accepted": accepted,
                "score": "" if decision.score is None else float(decision.score),
                "threshold": "" if decision.threshold is None else float(decision.threshold),
            }
        )

    report = {
        "n_h0": n_h0,
        "n_h1": n_h1,
        "n_h0_accepts": n_h0_accept,
        "n_h1_accepts": n_h1_accept,
        "tpr": _safe_rate(n_h1_accept, n_h1),
        "fpr": _safe_rate(n_h0_accept, n_h0),
    }
    return report, rows


def _compare(systems: dict[str, Any]) -> dict[str, Any]:
    if "hadamard_ensemble" not in systems or "cosine_only" not in systems:
        return {}
    ensemble = systems["hadamard_ensemble"]["cold_start"]
    cosine = systems["cosine_only"]["cold_start"]
    return {
        "tpr_delta_ensemble_minus_cosine": ensemble["tpr"] - cosine["tpr"],
        "fpr_delta_ensemble_minus_cosine": ensemble["fpr"] - cosine["fpr"],
        "threshold_delta_ensemble_minus_cosine": (
            systems["hadamard_ensemble"]["calibrated_threshold"] - systems["cosine_only"]["calibrated_threshold"]
        ),
    }


def _safe_rate(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return float(numerator) / float(denominator)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLD_START_CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_json(path: Path, data: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(json_safe(data), handle, indent=2, sort_keys=True)


if __name__ == "__main__":
    raise SystemExit(main())
