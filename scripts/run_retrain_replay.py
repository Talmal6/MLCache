"""Online cache replay with periodic scorer retraining.

Models a growing semantic cache: queries arrive in stream order, each is scored
against the retrieved anchors by the *current* ensemble and admitted against the
*current* NP threshold. Every time the cache DB changes X times (a change =
a cache write, i.e. a MISS/insert by default) the scorer is **retrained** and the
threshold **recalibrated** on Y% of the new cached data accumulated since the last
retrain. A no-retrain baseline (fit once on warm-up, then frozen) is run over the
exact same stream so the retrain effect is isolated.

No leakage: every admission decision uses only the scorer/threshold trained on
*past* data; the label is revealed after the decision and only past labels ever
enter a retrain.

Example:
    .conda/bin/python scripts/run_retrain_replay.py \
        --npz data/h1h0_final.npz --output-dir runs_adaptive/retrain \
        --max-rows 24000 --ordering cluster_block \
        --retrain-every-x 2000 --new-data-pct 0.5
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mlcache.adaptive.audit_policy import MarginBandAuditConfig, MarginBandAuditPolicy  # noqa: E402
from mlcache.adaptive.metrics import AdmissionEvent, AdmissionMetrics  # noqa: E402
from mlcache.adaptive.replay import ScoredSelection, calibrate_selected_h0  # noqa: E402
from mlcache.builder import build_scorer  # noqa: E402
from mlcache.features.hadamard import NormalizedHadamardFeatureBuilder  # noqa: E402
from mlcache.feedback.h1h0_npz_adapters import (  # noqa: E402
    H1H0NPZDataset,
    H1H0NPZJudgeAdapter,
    H1H0NPZStreamAdapter,
)
from mlcache.feedback.types import JudgeLabel, JudgeRequest  # noqa: E402
from mlcache.retrieval.in_memory import InMemoryVectorStore  # noqa: E402
from mlcache.semantic_types import LabeledPairBatch, TieMode  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--npz", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--label-field", default="label")
    p.add_argument("--max-rows", type=int, default=24000)
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--ensemble-members", default="cosine,lda,pca_whitened_cosine,xgboost,mlp")
    p.add_argument("--ordering", default="cluster_block", help="natural | random | cluster_block")
    p.add_argument("--warmup-frac", type=float, default=0.3, help="prefix used for the initial fit + calibration")
    # retrain knobs
    p.add_argument("--retrain-every-x", type=int, default=2000, help="X: cache changes between retrains")
    p.add_argument("--new-data-pct", type=float, default=0.5, help="Y in (0,1]: fraction of new cached data used to retrain")
    p.add_argument("--change-unit", default="miss", choices=["miss", "query", "labeled"],
                   help="what counts as one cache change (default: a MISS = a cache write)")
    p.add_argument("--pool-cap", type=int, default=40000, help="max labeled pairs kept for refitting (recency-capped)")
    p.add_argument("--calib-window", type=int, default=6000, help="recent labeled records re-scored to recalibrate tau")
    # audit (metrics only; training uses known labels)
    p.add_argument("--p-control", type=float, default=0.04)
    p.add_argument("--judge-cost-ratio", type=float, default=0.1)
    p.add_argument("--window", type=int, default=2000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--progress-every", type=int, default=5000)
    return p.parse_args(argv)


def order_records(records, ordering: str, seed: int):
    items = list(records)
    if ordering == "natural":
        return sorted(items, key=lambda r: r.row_id)
    if ordering == "random":
        rng = random.Random(seed)
        rng.shuffle(items)
        return items
    if ordering == "cluster_block":
        rng = random.Random(seed)
        blocks: dict[str, list] = {}
        for r in sorted(items, key=lambda r: r.row_id):
            blocks.setdefault(str(r.anchor_key), []).append(r)
        keys = list(blocks)
        rng.shuffle(keys)
        out = []
        for k in keys:
            out.extend(blocks[k])
        return out
    raise ValueError(f"unknown ordering {ordering!r}")


def _label_to_int(label: JudgeLabel) -> int | None:
    if label == JudgeLabel.REUSABLE:
        return 1
    if label == JudgeLabel.NOT_REUSABLE:
        return 0
    return None


class OnlineCache:
    """Holds the mutable scorer + threshold and the retrieval index."""

    def __init__(self, members, anchors, anchor_embedding, feature_builder, alpha, top_k):
        self.members = members
        self.anchor_embedding = anchor_embedding
        self.fb = feature_builder
        self.alpha = alpha
        self.top_k = top_k
        self.store = InMemoryVectorStore(similarity="cosine")
        for a in anchors:
            self.store.upsert(a)
        self.scorer = None
        self.tau = float("inf")
        self.n_fits = 0

    def score_pair(self, q_emb, anchor_emb) -> float:
        return float(self.scorer.score(self.fb.build(q_emb, anchor_emb)))

    def select(self, q_emb):
        """Retrieve top-k, score each with the current scorer, return argmax."""
        results = self.store.search(q_emb, top_k=self.top_k)
        best_key, best_score = None, float("-inf")
        for c in results:
            s = self.score_pair(q_emb, c.embedding)
            if s > best_score:
                best_score, best_key = s, str(c.cache_key)
        return best_key, best_score

    def fit(self, pool: list[tuple[tuple, int]]):
        """Refit the ensemble on the labeled-pair pool (hadamard, label)."""
        h0 = [f for f, y in pool if y == 0]
        h1 = [f for f, y in pool if y == 1]
        if len(h0) < 8 or len(h1) < 8:
            return False
        scorer = build_scorer("ensemble", scorers=self.members)
        scorer.fit(LabeledPairBatch(h0=h0, h1=h1), alpha=self.alpha)
        self.scorer = scorer
        self.n_fits += 1
        return True

    def recalibrate(self, calib_recs, amap):
        """Recompute the NP threshold from selected-H0 scores under the new scorer."""
        sels = []
        for r in calib_recs:
            if r.label not in (0, 1):
                continue
            key, score = self.select(r.query_embedding)
            lab = r.label if key == str(r.anchor_key) else None
            sels.append(ScoredSelection(r.row_id, r.query_id, key or "", score, lab, key, key == str(r.anchor_key)))
        res = calibrate_selected_h0(sels, alpha_target=self.alpha, tie_mode=TieMode.GE)
        if res.threshold is not None:
            self.tau = float(res.threshold)
        return res.threshold


@dataclass
class RunResult:
    summary: dict
    n_retrains: int


def serve(cache: OnlineCache, serve_recs, amap, judge, args, *, retrain: bool, label: str) -> RunResult:
    fb = cache.fb
    audit = MarginBandAuditPolicy(MarginBandAuditConfig(p_control=args.p_control), rng=random.Random(args.seed))
    metrics = AdmissionMetrics(alpha_target=args.alpha, window=args.window,
                               judge_cost_ratio=args.judge_cost_ratio)
    pool: list[tuple[tuple, int]] = list(cache._pool_seed)  # start from warm-up pool
    new_since: list[tuple[tuple, int]] = []
    calib_recs: list = list(cache._calib_seed)
    changes = 0
    n_retrains = 0

    for i, r in enumerate(serve_recs):
        key, score = cache.select(r.query_embedding)
        accepted = score >= cache.tau
        a = audit.decide(score, cache.tau)
        # reveal label (own-cluster pair only, as the judge would know)
        lab = r.label if (key == str(r.anchor_key) and r.label in (0, 1)) else None
        metrics.record(AdmissionEvent(
            stream_index=i, query_id=r.query_id, best_score=score, tau=cache.tau,
            alpha_t=float("nan"), accepted=accepted, label=lab, judged=a.judged,
            control_audit=a.control_audit, diagnostic_audit=a.diagnostic_audit,
            p_control=a.p_control, p_diagnostic=a.p_diagnostic, zone=str(a.zone),
            threshold_source="retrain" if retrain else "fixed", region_id=key,
        ))

        # accumulate training data from known-label own-cluster pairs
        if r.label in (0, 1) and str(r.anchor_key) in amap:
            feat = fb.build(r.query_embedding, amap[str(r.anchor_key)]).hadamard
            new_since.append((feat, r.label))
            calib_recs.append(r)
            if len(calib_recs) > args.calib_window:
                calib_recs = calib_recs[-args.calib_window:]

        # count a cache change
        is_change = (args.change_unit == "query"
                     or (args.change_unit == "miss" and not accepted)
                     or (args.change_unit == "labeled" and lab is not None))
        if is_change:
            changes += 1

        # trigger a retrain
        if retrain and changes >= args.retrain_every_x:
            k = max(1, int(round(args.new_data_pct * len(new_since))))
            rng = random.Random(args.seed + n_retrains + 1)
            sample = new_since if k >= len(new_since) else rng.sample(new_since, k)
            pool.extend(sample)
            if len(pool) > args.pool_cap:
                pool = pool[-args.pool_cap:]
            tau_before = cache.tau
            if cache.fit(pool):
                cache.recalibrate(calib_recs, amap)
                n_retrains += 1
                if args.progress_every:
                    print(f"    · retrain #{n_retrains} @serve {i+1}: pool={len(pool)} "
                          f"new_used={len(sample)} tau {tau_before:.4f}->{cache.tau:.4f}", flush=True)
            new_since = []
            changes = 0

        if args.progress_every and (i + 1) % args.progress_every == 0:
            s = metrics.summary()
            print(f"  [{label}] {i+1}/{len(serve_recs)} cum_FPR={s['achieved_fpr']} "
                  f"cum_TPR={s['tpr']} tau={cache.tau:.4f} retrains={n_retrains}", flush=True)

    return RunResult(summary=metrics.summary(), n_retrains=n_retrains)


def run(args: argparse.Namespace) -> dict:
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    members = [m.strip() for m in args.ensemble_members.split(",") if m.strip()]

    ds = H1H0NPZDataset(args.npz, label_field=args.label_field, max_rows=args.max_rows)
    records = ds.records()
    judge = H1H0NPZJudgeAdapter(records)
    anchors = H1H0NPZStreamAdapter(ds).anchor_entries()
    amap = {str(a.cache_key): a.embedding for a in anchors}
    fb = NormalizedHadamardFeatureBuilder()

    ordered = order_records(records, args.ordering, args.seed)
    n_warm = max(1, int(round(args.warmup_frac * len(ordered))))
    warm, serve_recs = ordered[:n_warm], ordered[n_warm:]
    print(f"loaded {len(records)} rows; ordering={args.ordering}; warmup={n_warm} serve={len(serve_recs)}", flush=True)

    # warm-up training pool + calibration seed (shared by both methods)
    warm_pool = [(fb.build(r.query_embedding, amap[str(r.anchor_key)]).hadamard, r.label)
                 for r in warm if r.label in (0, 1) and str(r.anchor_key) in amap]

    results = {}
    for retrain in (False, True):
        label = "retrain" if retrain else "fixed_noretrain"
        cache = OnlineCache(members, anchors, amap, fb, args.alpha, args.top_k)
        cache._pool_seed = warm_pool
        cache._calib_seed = warm[-args.calib_window:]
        print(f"== {label}: initial fit on {len(warm_pool)} warm-up pairs ==", flush=True)
        cache.fit(list(warm_pool))
        cache.recalibrate(cache._calib_seed, amap)
        print(f"   initial tau={cache.tau:.4f}", flush=True)
        res = serve(cache, serve_recs, amap, judge, args, retrain=retrain, label=label)
        results[label] = {"summary": res.summary, "n_retrains": res.n_retrains}
        s = res.summary
        print(f"   [{label}] FPR={s['achieved_fpr']:.4f} TPR={s['tpr']:.4f} NP={s['np_score']:.4f} "
              f"hit={s['hit_rate']:.4f} judged={s['judged_fraction']:.4f} retrains={res.n_retrains}", flush=True)

    report = {"config": vars(args), "n_rows": len(records), "n_warm": n_warm,
              "n_serve": len(serve_recs), "results": results}
    (out / "retrain_report.json").write_text(json.dumps(report, indent=2, default=str))
    print(f"\nwrote {out / 'retrain_report.json'}", flush=True)
    return report


def main(argv: list[str] | None = None) -> int:
    run(parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
