"""Replay integrity: no label leakage, preserved order, shared candidates."""

from __future__ import annotations

import random

from mlcache.adaptive.aci_controller import GlobalACIController
from mlcache.adaptive.audit_policy import MarginBandAuditConfig, MarginBandAuditPolicy
from mlcache.adaptive.replay import (
    FixedThresholdAdmission,
    GlobalACIAdmission,
    OfflineReplayRunner,
    ScoredSelection,
    order_stream,
)


def _synthetic_table(n=400, seed=0):
    rng = random.Random(seed)
    rows = []
    for i in range(n):
        # H0 scores centered lower, H1 higher, some unknowns.
        r = rng.random()
        if r < 0.5:
            label, score = 0, rng.gauss(0.3, 0.15)
        elif r < 0.8:
            label, score = 1, rng.gauss(0.7, 0.15)
        else:
            label, score = None, rng.gauss(0.5, 0.2)
        rows.append(
            ScoredSelection(
                stream_index=i,
                query_id=f"q{i}",
                anchor_id=f"a{i % 10}",
                best_score=score,
                label=label,
                region_id=f"a{i % 10}",
                own_cluster=True,
            )
        )
    return rows


def _runner():
    return OfflineReplayRunner(
        alpha_target=0.1,
        audit_policy=MarginBandAuditPolicy(
            MarginBandAuditConfig(p_control=0.1), rng=random.Random(123)
        ),
        window=100,
    )


def _controller():
    c = GlobalACIController(
        alpha_target=0.1,
        gamma=0.02,
        alpha_min=0.001,
        alpha_max=0.3,
        buffer_size=1000,
        min_buffer_size=20,
        fallback_threshold=0.5,
    )
    c.seed([i / 200 for i in range(200)], fallback_threshold=0.5)
    return c


def test_order_preserved():
    table = _synthetic_table()
    ordered = order_stream(table, "natural")
    _, events = _runner().run(
        FixedThresholdAdmission(0.5), ordered, collect_events=True
    )
    assert [e.stream_index for e in events] == list(range(len(events)))
    assert [e.query_id for e in events] == [s.query_id for s in ordered]


def test_fixed_admission_ignores_labels_entirely():
    table = _synthetic_table()
    ordered = order_stream(table, "natural")

    _, events_a = _runner().run(FixedThresholdAdmission(0.5), ordered, collect_events=True)

    flipped = [
        ScoredSelection(
            stream_index=s.stream_index,
            query_id=s.query_id,
            anchor_id=s.anchor_id,
            best_score=s.best_score,
            label=(None if s.label is None else 1 - s.label),
            region_id=s.region_id,
            own_cluster=s.own_cluster,
        )
        for s in ordered
    ]
    _, events_b = _runner().run(FixedThresholdAdmission(0.5), flipped, collect_events=True)

    # A fixed threshold cannot depend on labels: accept sequence is identical.
    assert [e.accepted for e in events_a] == [e.accepted for e in events_b]


def test_aci_decision_does_not_peek_at_current_label():
    """Changing the label at position i must not change the decision at i.

    (Later decisions may legitimately change, because ACI learns from the
    post-decision label -- but never the current one.)
    """

    table = _synthetic_table(n=300, seed=7)
    ordered = order_stream(table, "natural")
    flip_at = 150

    runner_a = _runner()
    _, events_a = runner_a.run(GlobalACIAdmission(_controller()), ordered, collect_events=True)

    modified = list(ordered)
    s = modified[flip_at]
    modified[flip_at] = ScoredSelection(
        stream_index=s.stream_index,
        query_id=s.query_id,
        anchor_id=s.anchor_id,
        best_score=s.best_score,
        label=(0 if s.label != 0 else 1),
        region_id=s.region_id,
        own_cluster=s.own_cluster,
    )
    runner_b = _runner()
    _, events_b = runner_b.run(GlobalACIAdmission(_controller()), modified, collect_events=True)

    # Decisions up to and including flip_at are unaffected by the label at flip_at.
    for i in range(flip_at + 1):
        assert events_a[i].accepted == events_b[i].accepted
        assert events_a[i].tau == events_b[i].tau
        assert events_a[i].alpha_t == events_b[i].alpha_t


def test_fixed_and_aci_run_on_same_selected_candidates():
    table = _synthetic_table()
    ordered = order_stream(table, "natural")
    _, fixed_events = _runner().run(FixedThresholdAdmission(0.5), ordered, collect_events=True)
    _, aci_events = _runner().run(GlobalACIAdmission(_controller()), ordered, collect_events=True)
    assert [e.best_score for e in fixed_events] == [e.best_score for e in aci_events]
    assert [e.query_id for e in fixed_events] == [e.query_id for e in aci_events]
    assert [e.label for e in fixed_events] == [e.label for e in aci_events]


def test_ordering_is_a_permutation():
    table = _synthetic_table()
    for ordering in ("natural", "random", "cluster_block"):
        ordered = order_stream(table, ordering, seed=1)
        assert sorted(s.stream_index for s in ordered) == list(range(len(table)))
