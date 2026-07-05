"""Offline replay comparing fixed thresholds vs Global ACI admission.

Builds a frozen selected-candidate table once per scorer, then runs the
baselines on the *same* stream:

    A. Fixed cosine threshold
    B. Cosine + Global ACI
    C. Fixed WeightedEnsemble threshold
    D. WeightedEnsemble + Global ACI

The headline question: does WeightedEnsemble + Global ACI keep FPR <= alpha
while giving higher hit-rate / TPR / correct-hit utility than a fixed
WeightedEnsemble threshold, judging only a small audited fraction?

Example:
    .conda/bin/python scripts/run_adaptive_replay.py \
        --npz data/h1h0_final.npz --output-dir runs/adaptive_smoke \
        --max-rows 20000 --alpha 0.05 --ordering random
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mlcache.adaptive.aci_controller import GlobalACIController  # noqa: E402
from mlcache.adaptive.audit_policy import MarginBandAuditConfig, MarginBandAuditPolicy  # noqa: E402
from mlcache.adaptive.replay import (  # noqa: E402
    FixedThresholdAdmission,
    GlobalACIAdmission,
    OfflineReplayRunner,
    ScoredSelection,
    build_frozen_table,
    calibrate_selected_h0,
    order_stream,
    score_percentiles,
    split_calibration_serving,
    table_separation_diagnostics,
)
from mlcache.builder import build_scorer  # noqa: E402
from mlcache.features.hadamard import NormalizedHadamardFeatureBuilder  # noqa: E402
from mlcache.feedback.h1h0_npz_adapters import (  # noqa: E402
    H1H0NPZDataset,
    H1H0NPZJudgeAdapter,
    H1H0NPZStreamAdapter,
)
from mlcache.semantic_types import LabeledPairBatch, TieMode  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--npz", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--label-field", default="label")
    p.add_argument("--max-rows", type=int, default=20000)
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--alpha", type=float, default=0.05, help="FPR budget alpha_target")
    p.add_argument("--ensemble-members", default="cosine,lda,pca_whitened_cosine")
    p.add_argument("--fit-fraction", type=float, default=0.3, help="rows used to train the ensemble scorer")
    p.add_argument("--calib-fraction", type=float, default=0.3, help="prefix of served stream used for batch calibration")
    p.add_argument("--ordering", default="random", help="comma list: random,natural,cluster_block")
    p.add_argument("--selection", default="argmax", choices=["argmax", "labeled_pair"])
    # ACI knobs
    p.add_argument("--gamma", type=float, default=0.005)
    p.add_argument("--alpha-min", type=float, default=0.001)
    p.add_argument("--alpha-max", type=float, default=0.20)
    p.add_argument("--buffer-size", type=int, default=20000)
    p.add_argument("--min-buffer-size", type=int, default=200)
    p.add_argument("--wilson-z", type=float, default=1.96)
    p.add_argument("--wilson-window", type=int, default=1000)
    p.add_argument("--judge-cost-ratio", type=float, default=0.1,
                   help="cost of one judge call relative to a saved provider call")
    # audit knobs
    p.add_argument("--p-control", type=float, default=0.04)
    p.add_argument("--accept-margin", type=float, default=0.05)
    p.add_argument("--reject-margin", type=float, default=0.05)
    p.add_argument("--window", type=int, default=2000)
    p.add_argument("--snapshot-every", type=int, default=1000)
    p.add_argument(
        "--print-snapshots",
        action="store_true",
        help="print rolling and cumulative FPR/TPR snapshots during each replay",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--progress-every", type=int, default=5000)
    return p.parse_args(argv)


def _fit_ensemble(records, n_fit: int, members: list[str], alpha: float, anchor_embedding: dict):
    """Fit the ensemble on (query, cluster-mean anchor) pairs.

    IMPORTANT: use the cluster-mean anchor embedding (the same object the frozen
    table scores against at serving), NOT ``record.anchor_embedding`` -- the H1/H0
    adapter aliases the latter to the row's own query embedding, so training on it
    would fit self-pairs (cosine==1.0) instead of real query-vs-anchor pairs.
    """

    feature_builder = NormalizedHadamardFeatureBuilder()
    h0: list[tuple[float, ...]] = []
    h1: list[tuple[float, ...]] = []
    for record in records:
        if record.row_id >= n_fit:
            continue
        if record.label not in (0, 1):
            continue
        anchor_emb = anchor_embedding.get(str(record.anchor_key))
        if anchor_emb is None:
            continue
        features = feature_builder.build(record.query_embedding, anchor_emb)
        if record.label == 0:
            h0.append(features.hadamard)
        else:
            h1.append(features.hadamard)
    scorer = build_scorer("ensemble", scorers=members)
    scorer.fit(LabeledPairBatch(h0=h0, h1=h1), alpha=alpha)
    return scorer, len(h0), len(h1)


def _make_controller(args: argparse.Namespace, calib, *, wilson_gate: bool) -> GlobalACIController:
    controller = GlobalACIController(
        alpha_target=args.alpha,
        gamma=args.gamma,
        alpha_min=args.alpha_min,
        alpha_max=args.alpha_max,
        buffer_size=args.buffer_size,
        min_buffer_size=args.min_buffer_size,
        fallback_threshold=calib.threshold,
        tie_mode=TieMode.GE,
        wilson_gate=wilson_gate,
        wilson_z=args.wilson_z,
        wilson_window=args.wilson_window,
    )
    controller.seed(calib.h0_scores, fallback_threshold=calib.threshold)
    return controller


def _controller_diagnostics(controller: GlobalACIController) -> dict[str, Any]:
    return {
        "aci_alpha_t_final": controller.alpha_t,
        "aci_tau_final": controller.threshold(),
        "aci_control_h0_count": controller.control_h0_count,
        "aci_post_seed_buffer_fraction": controller.post_seed_buffer_fraction,
        "aci_wilson_ucb_final": controller.wilson_ucb(),
        "aci_blocked_loosen_count": controller.blocked_loosen_count,
    }


def _run_pair(
    args: argparse.Namespace,
    label: str,
    table: tuple[ScoredSelection, ...],
    ordering: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    ordered = order_stream(table, ordering, seed=args.seed)
    calib_stream, serve_stream = split_calibration_serving(ordered, calib_fraction=args.calib_fraction)
    calib = calibrate_selected_h0(calib_stream, alpha_target=args.alpha, tie_mode=TieMode.GE)
    calibration_meta = {
        "ordering": ordering,
        "n_calib": len(calib_stream),
        "n_serve": len(serve_stream),
        "calib_n_h0": calib.n_h0,
        "calib_n_h1": calib.n_h1,
        "batch_threshold": calib.threshold,
    }

    audit_config = MarginBandAuditConfig(
        accept_margin=args.accept_margin,
        reject_margin=args.reject_margin,
        p_control=args.p_control,
    )
    rows: list[dict[str, Any]] = []
    snapshots: list[dict[str, Any]] = []

    if calib.threshold is None:
        print(f"  [{label}] no finite batch threshold (calib H0={calib.n_h0}); skipping", flush=True)
        return rows, snapshots

    # fixed baseline + ungated ACI + Wilson-gated ACI, on the same serve stream.
    # Each gets a fresh RNG seeded identically so audit draws are comparable.
    def _fixed():
        return FixedThresholdAdmission(calib.threshold, name=f"fixed_{label}"), None

    def _aci(gate: bool, name: str):
        controller = _make_controller(args, calib, wilson_gate=gate)
        return GlobalACIAdmission(controller, name=name), controller

    builders = [
        (f"fixed_{label}", _fixed),
        (f"aci_{label}", lambda: _aci(False, f"aci_{label}")),
        (f"aci_wilson_{label}", lambda: _aci(True, f"aci_wilson_{label}")),
    ]

    for method_name, builder in builders:
        admission, controller = builder()
        def _snapshot_callback(snapshot: dict[str, Any], method_name: str = method_name) -> None:
            enriched = {"method": method_name, "calib_ordering": ordering, **snapshot}
            print(_fmt_snapshot(enriched), flush=True)

        runner = OfflineReplayRunner(
            alpha_target=args.alpha,
            audit_policy=MarginBandAuditPolicy(audit_config, rng=random.Random(args.seed)),
            window=args.window,
            snapshot_every=args.snapshot_every,
            tie_mode=TieMode.GE,
            judge_cost_ratio=args.judge_cost_ratio,
        )
        result, _ = runner.run(
            admission,
            serve_stream,
            calibration_meta=calibration_meta,
            snapshot_callback=_snapshot_callback if args.print_snapshots else None,
        )
        snapshots.extend(
            {"method": method_name, "calib_ordering": ordering, **snapshot}
            for snapshot in result.snapshots
        )
        row = {"method": method_name, **result.summary, **{f"calib_{k}": v for k, v in calibration_meta.items()}}
        if controller is not None:
            row.update(_controller_diagnostics(controller))
        rows.append(row)
        print(f"  [{method_name}] {_fmt_row(row)}", flush=True)
    return rows, snapshots


def _fmt_row(row: dict[str, Any]) -> str:
    def g(k: str) -> str:
        v = row.get(k)
        return "  n/a" if v is None else f"{v:.4f}"

    base = (
        f"FPR={g('achieved_fpr')} TPR={g('tpr')} hit={g('hit_rate')} "
        f"nojudge_hit={g('no_judge_hit_rate')} judged={g('judged_fraction')} "
        f"NP={g('np_score')} ctrl_fpr={g('control_audit_fpr')} cost_util={g('cost_adjusted_utility_per_request')}"
    )
    if row.get("aci_post_seed_buffer_fraction") is not None:
        base += f" postseed={g('aci_post_seed_buffer_fraction')} blocked={row.get('aci_blocked_loosen_count')}"
    return base


def _fmt_snapshot(row: dict[str, Any]) -> str:
    def g(k: str) -> str:
        v = row.get(k)
        if v is None:
            return "n/a"
        value = float(v)
        return "n/a" if not math.isfinite(value) else f"{value:.4f}"

    return (
        f"  [snapshot {row['calib_ordering']}/{row['method']}] "
        f"seen={row['total_seen']} stream={row['stream_index']} "
        f"cum_FPR={g('cumulative_fpr')} cum_TPR={g('cumulative_tpr')} "
        f"win_FPR={g('window_fpr')} win_TPR={g('window_tpr')} "
        f"hit={g('cumulative_hit_rate')} judged={g('cumulative_judged_fraction')} "
        f"alpha_t={g('alpha_t')} tau={g('tau')}"
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    dataset = H1H0NPZDataset(args.npz, label_field=args.label_field, max_rows=args.max_rows)
    records = dataset.records()
    judge = H1H0NPZJudgeAdapter(records)
    n_fit = max(1, int(round(args.fit_fraction * len(records))))
    members = [m.strip() for m in args.ensemble_members.split(",") if m.strip()]

    # Cluster-mean anchor embeddings -- the same objects the frozen table scores
    # against. Used to fit the ensemble on real query-vs-anchor pairs.
    anchors = H1H0NPZStreamAdapter(dataset).anchor_entries()
    anchor_embedding = {str(a.cache_key): a.embedding for a in anchors}

    print(f"loaded {len(records)} rows; fitting ensemble on first {n_fit} rows", flush=True)
    ensemble, fit_h0, fit_h1 = _fit_ensemble(records, n_fit, members, args.alpha, anchor_embedding)
    ensemble_weights = {
        str(member.name): float(weight)
        for member, weight in zip(ensemble.members(), ensemble.weights)
    }
    print(f"ensemble fit on H0={fit_h0} H1={fit_h1}; weights={ensemble_weights}", flush=True)
    print("scoring frozen tables ...", flush=True)

    feature_builder = NormalizedHadamardFeatureBuilder()
    cosine = build_scorer("cosine")

    cosine_table_all = build_frozen_table(
        dataset, cosine, feature_builder=feature_builder, judge=judge,
        top_k=args.top_k, selection=args.selection, progress_every=args.progress_every,
    )
    ensemble_table_all = build_frozen_table(
        dataset, ensemble, feature_builder=feature_builder, judge=judge,
        top_k=args.top_k, selection=args.selection, progress_every=args.progress_every,
    )
    # Evaluate only on rows the ensemble scorer was NOT trained on.
    cosine_table = tuple(s for s in cosine_table_all if s.stream_index >= n_fit)
    ensemble_table = tuple(s for s in ensemble_table_all if s.stream_index >= n_fit)

    # Verification diagnostics (checks 2 & 3): are the two tables genuinely
    # different admission rules, and is the ensemble threshold on a sane scale?
    separation = table_separation_diagnostics(
        cosine_table, ensemble_table, alpha_target=args.alpha, tie_mode=TieMode.GE
    )
    cosine_pct = score_percentiles(cosine_table)
    ensemble_pct = score_percentiles(ensemble_table)
    print(f"[diag] table separation: {separation}", flush=True)
    print(f"[diag] cosine best_score pct:   {cosine_pct}", flush=True)
    print(f"[diag] ensemble best_score pct: {ensemble_pct}", flush=True)

    orderings = [o.strip() for o in args.ordering.split(",") if o.strip()]
    all_rows: list[dict[str, Any]] = []
    all_snapshots: list[dict[str, Any]] = []
    for ordering in orderings:
        print(f"== ordering={ordering} ==", flush=True)
        rows, snapshots = _run_pair(args, "cosine", cosine_table, ordering)
        all_rows.extend(rows)
        all_snapshots.extend(snapshots)
        rows, snapshots = _run_pair(args, "ensemble", ensemble_table, ordering)
        all_rows.extend(rows)
        all_snapshots.extend(snapshots)

    report = {
        "config": vars(args),
        "n_rows": len(records),
        "n_fit": n_fit,
        "eval_n_cosine": len(cosine_table),
        "eval_n_ensemble": len(ensemble_table),
        "fit_h0": fit_h0,
        "fit_h1": fit_h1,
        "ensemble_members": members,
        "ensemble_weights": ensemble_weights,
        "table_separation": separation,
        "cosine_score_percentiles": cosine_pct,
        "ensemble_score_percentiles": ensemble_pct,
        "rows": all_rows,
        "snapshots": all_snapshots,
    }
    (out / "adaptive_replay_report.json").write_text(json.dumps(report, indent=2, default=str))
    _write_table_csv(out / "comparison_table.csv", all_rows)
    _write_snapshots_csv(out / "snapshots.csv", all_snapshots)
    print(f"\nwrote {out / 'adaptive_replay_report.json'}", flush=True)
    return report


_TABLE_FIELDS = [
    "method", "calib_ordering", "achieved_fpr", "tpr", "hit_rate", "no_judge_hit_rate",
    "judged_fraction", "provider_call_rate", "correct_hit_utility",
    "cost_adjusted_utility_per_request", "false_hits", "np_score", "fpr_ok",
    "control_audit_fpr", "iw_audit_fpr", "aci_alpha_t_final",
    "aci_post_seed_buffer_fraction", "aci_blocked_loosen_count",
    "gt_n_h0", "gt_n_h1", "total_requests",
]


def _write_table_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    import csv

    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=_TABLE_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in _TABLE_FIELDS})


_SNAPSHOT_FIELDS = [
    "calib_ordering", "method", "total_seen", "stream_index", "alpha_t", "tau",
    "cumulative_fpr", "cumulative_tpr", "cumulative_hit_rate",
    "cumulative_no_judge_hit_rate", "cumulative_judged_fraction",
    "cumulative_n_h0", "cumulative_n_h1", "cumulative_false_hits",
    "cumulative_true_hits", "window_fpr", "window_tpr", "window_hit_rate",
    "window_no_judge_hit_rate", "window_judged_fraction", "window_n_h0",
    "window_n_h1", "window_false_hits", "window_true_hits",
]


def _write_snapshots_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    import csv

    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=_SNAPSHOT_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in _SNAPSHOT_FIELDS})


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
