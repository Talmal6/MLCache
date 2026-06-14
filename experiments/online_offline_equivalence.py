"""Online<->Offline cache decision-equivalence protocol.

Goal
----
Prove that, on a FROZEN MLCache state, an offline re-score of the EXACT pairs the
online runtime produced reproduces MLCache's HIT/MISS decision up to numerical
tolerance -- given the same fitted scorer, the same literal threshold, and the
same score orientation. This is an equivalence / regression test of the runtime
decision path (retrieve -> feature pipeline -> score -> threshold), NOT a policy
comparison: retrieval, state, tau and params are pinned IDENTICAL on both sides
by construction, so the test is blind to them on purpose (see the protocol's
"Fairness invariants").

What is pinned (invariants)
---------------------------
 1. Fitted scorer params: the SAME frozen `cache.scorer` object scores both sides.
 2. Literal threshold tau: `cache.threshold` is read once and never recomputed.
 3. Orientation: HIT iff score >= tau on both sides.
 4. Exact pairs: the offline side scores the embeddings the online retriever
    actually returned for that request (read back from the runtime vector store),
    it does not retrieve its own.
 5. Feature pipeline: the offline side rebuilds the normalized-Hadamard
    PairFeatures the same way `NormalizedHadamardFeatureBuilder` does (L2-norm
    then elementwise product, cosine = float64 sum), in float64.
 7. Frozen state: `MLCache.from_preset` wires `auto_refit=False`; we never `put`
    an eval query, so there is no write-through and no online recalibration.

Run:
  .conda/bin/python experiments/online_offline_equivalence.py \
      --npz data/h1h0_final.npz --scorer ensemble \
      --scorers cosine,lda,pca_whitened_cosine,xgboost,mlp \
      --target-fpr 0.05 --n-anchors 800 --n-eval 1200 --seed 42 \
      --output-dir experiments/equivalence_out
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mlcache.builder import MLCache  # noqa: E402
from mlcache.features import NormalizedHadamardFeatureBuilder, PairFeatures  # noqa: E402
from mlcache.feedback import JudgedPairExample  # noqa: E402
from mlcache.feedback.types import JudgeDecision, JudgeLabel, JudgeRequest  # noqa: E402
from mlcache.semantic_types import (  # noqa: E402
    CacheEntry,
    CacheLookup,
    CacheMetadata,
    Query,
    Response,
)

NS = "equivalence-test"
EPS = 1e-6
_FB = NormalizedHadamardFeatureBuilder()


# ── feature pipeline (invariant #5): rebuild PairFeatures exactly like the
#    runtime's NormalizedHadamardFeatureBuilder, in float64. Setting `cosine`
#    matters: CosineScorer.score prefers features.cosine (float64) and only
#    falls back to a float32 Hadamard sum when it is absent. ───────────────────
def offline_pair_features(q: np.ndarray, c: np.ndarray) -> PairFeatures:
    q = np.asarray(q, dtype=np.float64)
    c = np.asarray(c, dtype=np.float64)
    qn = q / np.linalg.norm(q)
    cn = c / np.linalg.norm(c)
    hadamard = tuple(float(x) for x in (qn * cn))
    return PairFeatures(cosine=float(sum(hadamard)), hadamard=hadamard)


def hadamard_tuple(q: np.ndarray, c: np.ndarray) -> tuple[float, ...]:
    return offline_pair_features(q, c).hadamard


# ── dataset ──────────────────────────────────────────────────────────────────
def load_npz(path: Path, *, max_rows: int, seed: int):
    d = np.load(path, allow_pickle=True)
    emb = np.asarray(d["emb"], dtype=np.float64)
    label = np.asarray(d["label"]).astype(int)
    cluster = np.asarray(d["global_cluster"]).astype(str)
    n = emb.shape[0]
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)[: min(max_rows, n)]
    return emb[idx], label[idx], cluster[idx]


def build_calibration_pairs(
    emb: np.ndarray, label: np.ndarray, cluster: np.ndarray, *, seed: int, max_pairs: int
) -> list[JudgedPairExample]:
    """H1 = same-cluster label==1 pairs (reusable); H0 = a label==1 row paired
    with a label==0 row (not reusable). Mirrors the dataset H1/H0 convention so
    the scorer fits and the NP threshold calibrates on meaningful pairs."""
    rng = np.random.default_rng(seed + 7)
    by_cluster_pos: dict[str, list[int]] = defaultdict(list)
    pos_rows, neg_rows = [], []
    for i in range(len(emb)):
        if label[i] == 1:
            by_cluster_pos[cluster[i]].append(i)
            pos_rows.append(i)
        elif label[i] == 0:
            neg_rows.append(i)

    pairs: list[JudgedPairExample] = []

    def add(a: int, b: int, lab: JudgeLabel) -> None:
        feats = hadamard_tuple(emb[a], emb[b])
        pairs.append(
            JudgedPairExample(
                features=feats,
                request=JudgeRequest(query=Query(f"row:{a}")),
                decision=JudgeDecision(label=lab),
            )
        )

    # H1: same-cluster positive pairs
    for rows in by_cluster_pos.values():
        if len(rows) < 2:
            continue
        for _ in range(min(len(rows), 6)):
            a, b = rng.choice(rows, size=2, replace=False)
            add(int(a), int(b), JudgeLabel.REUSABLE)
    # H0: positive query against a random negative anchor
    if neg_rows and pos_rows:
        for _ in range(len(pairs)):  # roughly class-balanced
            a = int(rng.choice(pos_rows))
            b = int(rng.choice(neg_rows))
            add(a, b, JudgeLabel.NOT_REUSABLE)

    rng.shuffle(pairs)
    return pairs[:max_pairs]


# ── online side: freeze + replay, dump per served pair ───────────────────────
def run_equivalence(args: argparse.Namespace) -> int:
    out = Path(args.output_dir)
    (out / "pairs").mkdir(parents=True, exist_ok=True)

    emb, label, cluster = load_npz(Path(args.npz), max_rows=args.max_rows, seed=args.seed)
    print(f"loaded {len(emb)} rows  dim={emb.shape[1]}  "
          f"label_dist={{0:{int((label==0).sum())},1:{int((label==1).sum())},-1:{int((label==-1).sum())}}}")

    pairs = build_calibration_pairs(emb, label, cluster, seed=args.seed, max_pairs=args.max_pairs)
    n_h1 = sum(1 for p in pairs if p.decision.label == JudgeLabel.REUSABLE)
    print(f"calibration pairs: {len(pairs)}  (H1={n_h1}, H0={len(pairs)-n_h1})")

    scorers = [s.strip() for s in args.scorers.split(",") if s.strip()] if args.scorer == "ensemble" else None
    tmp_root = tempfile.mkdtemp(prefix="equiv_")
    cache = MLCache.from_preset(
        root_dir=tmp_root, scorer=args.scorer, scorers=scorers,
        top_k=int(args.top_k), persistence=False,
    )
    info = cache.prefit_and_calibrate(pairs, target_fpr=float(args.target_fpr))
    tau = cache.threshold
    if tau is None:
        print("FAIL: calibration did not produce a threshold")
        return 2
    TAU = float(tau)
    print(f"frozen policy: scorer={info['scorer']}  tau={TAU:.10f}  "
          f"target_fpr={info['target_fpr']}  n_h0={info['n_h0']} n_h1={info['n_h1']}")

    # invariant #7: from_preset wires auto_refit=False -> no online learning.
    assert cache.runtime.oracle.auto_refit is False, "runtime is not frozen (auto_refit on)"

    # anchors (cache state) and eval queries: disjoint row partitions
    perm = np.random.default_rng(args.seed + 99).permutation(len(emb))
    anchor_idx = perm[: int(args.n_anchors)]
    eval_idx = perm[int(args.n_anchors): int(args.n_anchors) + int(args.n_eval)]

    anchor_emb: dict[str, np.ndarray] = {}
    for i in anchor_idx:
        key = f"row:{int(i)}"
        cache.put(CacheEntry(
            cache_key=key, query=Query(key), response=Response(f"resp:{int(i)}"),
            embedding=tuple(float(x) for x in emb[i]),
            metadata=CacheMetadata(namespace=NS),
        ))
        anchor_emb[key] = emb[i]
    print(f"indexed {len(anchor_emb)} anchors; replaying {len(eval_idx)} eval queries (write-through OFF)")

    vstore = cache.runtime.oracle.vector_store
    scorer = cache.scorer
    online_rows: list[dict[str, Any]] = []
    offline_rows: list[dict[str, Any]] = []
    skipped_no_candidate = 0

    # `OracleDecision.score` is the MAX scorer-score over the retrieved top-k and
    # `accepted` is `max >= tau`; the served key is that argmax candidate. To
    # compare the SAME quantity, the offline side scores the EXACT top-k the
    # runtime retrieved (read back from the same vector store, invariant #4),
    # takes the max, and reproduces HIT iff max >= tau. Comparing the served
    # rank-1 alone would compare different candidates on a MISS.
    for req_id, i in enumerate(eval_idx):
        q_emb = emb[i]
        lookup = CacheLookup(
            query=Query(f"row:{int(i)}"),
            embedding=tuple(float(x) for x in q_emb), namespace=NS,
        )
        cands = vstore.search(lookup.embedding, namespace=NS, top_k=int(args.top_k))
        cand_keys = [str(c.cache_key) for c in cands if str(c.cache_key) in anchor_emb]
        if not cand_keys:
            skipped_no_candidate += 1
            continue

        # offline: score every retrieved candidate, take argmax
        offline_scores = [
            float(scorer.score(offline_pair_features(q_emb, anchor_emb[k]))) for k in cand_keys
        ]
        off_arg = int(np.argmax(offline_scores))
        offline_max = offline_scores[off_arg]
        offline_served = cand_keys[off_arg]

        # online: the runtime's actual decision over the same candidates
        dec = cache.lookup_with_decision(lookup).decision
        online_max = float(dec.score) if dec.score is not None else float("nan")
        online_served = str(dec.cache_key) if dec.cache_key is not None else None

        np.savez(
            out / "pairs" / f"req_{req_id}.npz",
            q=q_emb.astype(np.float64),
            cands=np.stack([anchor_emb[k] for k in cand_keys]).astype(np.float64),
            cand_ids=np.array([int(k.split(":")[1]) for k in cand_keys]),
        )
        online_rows.append({
            "request_id": req_id, "query_id": int(i),
            "served_key": online_served if online_served else "",
            "score": online_max, "threshold": TAU,
            "decision": "HIT" if dec.accepted else "MISS",
        })
        offline_rows.append({
            "request_id": req_id, "query_id": int(i),
            "served_key": offline_served,
            "score": offline_max, "threshold": TAU,
            "decision": "HIT" if offline_max >= TAU else "MISS",
        })

    for name, rows in (("online_decisions.csv", online_rows), ("offline_decisions.csv", offline_rows)):
        with (out / name).open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
    print(f"online: {len(online_rows)} requests compared"
          f"  (skipped {skipped_no_candidate} with no retrievable candidate)")

    # served-key agreement on mutual HITs (argmax identity, not just the rate)
    served_mismatch = sum(
        1 for a, b in zip(online_rows, offline_rows)
        if a["decision"] == "HIT" and b["decision"] == "HIT" and a["served_key"] != b["served_key"]
    )
    print(f"served argmax-key mismatches on mutual HITs: {served_mismatch}")

    # ── compare (step 5) ─────────────────────────────────────────────────────
    on = {r["request_id"]: r for r in online_rows}
    off = {r["request_id"]: r for r in offline_rows}
    keys = sorted(set(on) & set(off))
    agree = 0
    score_gaps = []
    bug_decision_path = []      # scores agree but decisions differ -> orientation/tau BUG
    param_feature_mismatch = []  # scores differ beyond tol -> feature/param MISMATCH
    boundary_flips = []          # |score - tau| < EPS -> benign numerical
    for k in keys:
        a, b = on[k], off[k]
        gap = abs(a["score"] - b["score"])
        score_gaps.append(gap)
        same = a["decision"] == b["decision"]
        near_tau = abs(a["score"] - TAU) < EPS or abs(b["score"] - TAU) < EPS
        if same:
            agree += 1
        else:
            if near_tau:
                boundary_flips.append(k)
            elif gap < EPS:
                bug_decision_path.append(k)
            else:
                param_feature_mismatch.append(k)

    n = len(keys)
    non_boundary = n - len(boundary_flips)
    non_boundary_agree = sum(
        1 for k in keys
        if on[k]["decision"] == off[k]["decision"] and k not in boundary_flips
    )
    summary = {
        "scorer": info["scorer"], "tau": TAU, "target_fpr": float(args.target_fpr),
        "n_compared": n,
        "decision_agreement": agree / n if n else None,
        "non_boundary_agreement": non_boundary_agree / non_boundary if non_boundary else None,
        "max_score_gap": max(score_gaps) if score_gaps else None,
        "mean_score_gap": float(np.mean(score_gaps)) if score_gaps else None,
        "bug_in_decision_path": len(bug_decision_path),
        "param_or_feature_mismatch": len(param_feature_mismatch),
        "boundary_flips": len(boundary_flips),
    }
    (out / "equivalence_summary.json").write_text(json.dumps(summary, indent=2))

    passed = (
        n > 0
        and len(bug_decision_path) == 0
        and len(param_feature_mismatch) == 0
        and (non_boundary_agree == non_boundary)
    )
    print("\n=== EQUIVALENCE RESULT ===")
    print(f"compared pairs            : {n}")
    print(f"decision agreement        : {summary['decision_agreement']:.6f}")
    print(f"max score gap             : {summary['max_score_gap']:.3e}")
    print(f"orientation/tau BUGs      : {len(bug_decision_path)}")
    print(f"feature/param MISMATCHes  : {len(param_feature_mismatch)}")
    print(f"benign boundary flips     : {len(boundary_flips)}")
    print(f"\nPASS: {passed}")
    return 0 if passed else 1


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--npz", default="data/h1h0_final.npz")
    p.add_argument("--scorer", default="ensemble")
    p.add_argument("--scorers", default="cosine,lda,pca_whitened_cosine,xgboost,mlp")
    p.add_argument("--target-fpr", type=float, default=0.05)
    p.add_argument("--max-rows", type=int, default=6000)
    p.add_argument("--max-pairs", type=int, default=4000)
    p.add_argument("--n-anchors", type=int, default=800)
    p.add_argument("--n-eval", type=int, default=1200)
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", default="experiments/equivalence_out")
    return p


def main() -> int:
    return run_equivalence(build_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
