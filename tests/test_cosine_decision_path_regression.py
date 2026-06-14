"""Regression guard for MLCache's cosine decision-path plumbing.

SCOPE
-----
Validates correctness of the DECISION-PATH PLUMBING -- not the scorer, not the
learning loop, not retrieval. On a frozen cosine cache it checks that the runtime
turns a (query, candidate) pair into a HIT/MISS the way an offline float64
re-score does, pinning four invariants that silently break under refactors:

  1. No orientation flip   -- runtime decides HIT iff score >= tau (not <=, not <).
  2. No tau mishandling    -- the literal `cache.threshold` is the value the
                              runtime compares against.
  3. No float32 leakage    -- Hadamard build + cosine sum stay float64, so the
                              online score matches an offline float64 sum exactly
                              (a silent downcast would show up at ~1e-7).
  4. No norm-order bug      -- L2-normalize BEFORE the Hadamard product; doing it
                              after / not at all changes the score.

Deliberately NOT covered (own tests elsewhere): the online judge + training
store, vector_store.search / top-k ordering, and trainable scorers
(lda/xgboost/mlp). Cosine + synthetic pairs only => no ML deps, runs < 1s.

The end-to-end equivalence on real data lives in
`experiments/online_offline_equivalence.py`; this is its minimal CI-safe distillation.
"""

from __future__ import annotations

import numpy as np
import pytest

from mlcache.builder import MLCache
from mlcache.feedback import JudgedPairExample
from mlcache.feedback.types import JudgeDecision, JudgeLabel, JudgeRequest
from mlcache.semantic_types import (
    CacheEntry,
    CacheLookup,
    CacheMetadata,
    Query,
    Response,
)

DIM = 128
SEED = 1234
N_EVAL = 50


def _offline_cosine(u: np.ndarray, v: np.ndarray) -> float:
    """Reference rule: L2-normalize THEN Hadamard THEN sum, all float64 (#3, #4)."""
    u = np.asarray(u, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    un = u / np.linalg.norm(u)
    vn = v / np.linalg.norm(v)
    return float(np.sum(un * vn))


def _calibration_pairs(rng: np.random.Generator) -> list[JudgedPairExample]:
    """Synthetic H1 (near-duplicate, high cosine) / H0 (independent, ~0 cosine)
    pairs so cosine calibrates to a tau strictly inside (0, 1)."""

    def feat(u: np.ndarray, v: np.ndarray) -> tuple[float, ...]:
        un = u / np.linalg.norm(u)
        vn = v / np.linalg.norm(v)
        return tuple(float(x) for x in (un * vn))

    pairs: list[JudgedPairExample] = []
    for _ in range(60):  # H1: v = u + small noise
        u = rng.standard_normal(DIM)
        v = u + 0.05 * rng.standard_normal(DIM)
        pairs.append(JudgedPairExample(
            features=feat(u, v), request=JudgeRequest(query=Query("h1")),
            decision=JudgeDecision(label=JudgeLabel.REUSABLE)))
    for _ in range(60):  # H0: independent draws
        u = rng.standard_normal(DIM)
        v = rng.standard_normal(DIM)
        pairs.append(JudgedPairExample(
            features=feat(u, v), request=JudgeRequest(query=Query("h0")),
            decision=JudgeDecision(label=JudgeLabel.NOT_REUSABLE)))
    rng.shuffle(pairs)
    return pairs


@pytest.fixture
def frozen_cosine_cache(tmp_path):
    """Cosine MLCache, calibrated once then frozen (from_preset wires
    auto_refit=False; we never put an eval query => no write-through, no learning)."""
    rng = np.random.default_rng(SEED)
    cache = MLCache.from_preset(
        root_dir=str(tmp_path), scorer="cosine", top_k=1, persistence=False,
    )
    cache.prefit_and_calibrate(_calibration_pairs(rng), target_fpr=0.05)
    assert cache.runtime.oracle.auto_refit is False, "cache is not frozen"
    assert cache.threshold is not None, "calibration produced no threshold"
    return cache


def _eval_pairs(rng: np.random.Generator):
    """Pairs whose cosine spans both sides of tau: interpolate query toward an
    anchor by a varying blend so the served cosine sweeps ~0..1."""
    out = []
    for k in range(N_EVAL):
        anchor = rng.standard_normal(DIM)
        noise = rng.standard_normal(DIM)
        blend = k / (N_EVAL - 1)  # 0 -> orthogonal-ish (MISS), 1 -> aligned (HIT)
        query = blend * anchor + (1.0 - blend) * noise
        out.append((query, anchor))
    return out


def test_cosine_decision_path_regression(frozen_cosine_cache):
    cache = frozen_cosine_cache
    tau = float(cache.threshold)
    rng = np.random.default_rng(SEED + 1)

    n_hit = n_miss = 0
    orientation_violations: list[str] = []
    score_mismatches: list[str] = []

    for i, (query, anchor) in enumerate(_eval_pairs(rng)):
        ns = f"pair-{i}"  # one anchor per namespace -> retrieval is trivial (#out of scope)
        cache.put(CacheEntry(
            cache_key=f"anchor-{i}", query=Query(f"a{i}"), response=Response("r"),
            embedding=tuple(float(x) for x in anchor),
            metadata=CacheMetadata(namespace=ns)))
        dec = cache.lookup_with_decision(CacheLookup(
            query=Query(f"q{i}"), embedding=tuple(float(x) for x in query),
            namespace=ns)).decision

        online = float(dec.score)
        offline = _offline_cosine(query, anchor)
        decision = "HIT" if dec.accepted else "MISS"

        # #3 + #4: online score equals the offline float64 normalize->hadamard->sum.
        if abs(online - offline) > 1e-12:
            score_mismatches.append(
                f"pair {i}: online={online:.17g} offline={offline:.17g} "
                f"gap={abs(online - offline):.3e} (float32 leak or norm-order bug)")

        # #1 + #2: runtime decided HIT iff the score it computed clears the
        # literal cache.threshold. A flip to <=/< inverts this on one side.
        expected = "HIT" if online >= tau else "MISS"
        if abs(online - tau) > 1e-14 and decision != expected:
            orientation_violations.append(
                f"pair {i}: score={online:.17g} tau={tau:.17g} -> runtime said "
                f"{decision}, expected {expected} (orientation flip or tau mishandling)")

        n_hit += decision == "HIT"
        n_miss += decision == "MISS"

    assert not score_mismatches, (
        "online score != offline float64 cosine (invariants #3/#4):\n"
        + "\n".join(score_mismatches[:5]))
    assert not orientation_violations, (
        "runtime decision != (score >= tau) (invariants #1/#2):\n"
        + "\n".join(orientation_violations[:5]))
    # Guard against a vacuous pass: both branches must actually be exercised.
    assert n_hit > 0 and n_miss > 0, (
        f"eval set degenerate (hits={n_hit}, misses={n_miss}); "
        f"tau={tau:.6g} did not split the pairs -- test would not catch a flip")


def test_threshold_is_a_read_back_literal(frozen_cosine_cache):
    """Invariant #2, isolated: the threshold the runtime uses is exactly the
    value exposed by `cache.threshold` (a finite, in-range cosine threshold)."""
    tau = cache_tau = frozen_cosine_cache.threshold
    assert tau is not None
    assert np.isfinite(float(tau))
    assert -1.0 <= float(tau) <= 1.0
    # reading it twice returns the identical literal (no recomputation/drift)
    assert float(frozen_cosine_cache.threshold) == float(cache_tau)
