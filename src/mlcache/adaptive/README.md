# Adaptive Neyman-Pearson cache admission (Global ACI + sparse auditing)

This subpackage adds an **online admission layer** on top of MLCache's existing
scorers and calibration. It is a thin vertical slice, not the full architecture:
Global ACI controller, sparse two-channel auditing, admission metrics, and an
offline replay harness for the A–D baselines.

## Objective

The goal is **NP cache admission**, not average accuracy and not drift
detection:

> maximize hit-rate / TPR / correct-hit utility **subject to FPR ≤ α**, while
> judging as few queries as possible.

* **H1 / reusable** — the cached response for the selected anchor `a*` is
  acceptable for query `q_t`.
* **H0 / non-reusable** — it is not.
* **False hit** — accepting a HIT on an H0 pair. **FPR** = `P(accept | H0)`,
  **TPR** = `P(accept | H1)`.
* **NP score** — TPR (equivalently correct-hit utility) at `FPR ≤ α`.

All FPR/TPR are measured over **selected-candidate, deployment-shaped events**:
`retrieve top-k → score → select argmax a* → decide → (later) label a*`. We never
calibrate on random H0 pairs.

## What each piece does

| Module | Responsibility |
|---|---|
| `aci_controller.py` | `GlobalACIController` — adapts one global threshold `τ_t` so long-run FPR tracks α. |
| `audit_policy.py` | `MarginBandAuditPolicy` — two independent audit channels (control + diagnostic). |
| `metrics.py` | `AdmissionMetrics` — ground-truth + raw/control/IW audited metrics, Kish ESS, sliding windows. |
| `replay.py` | Frozen scored-candidate table, stream orderings, admission methods, `OfflineReplayRunner`. |

### 1. Global ACI is a threshold controller, not a scorer

The scorer still owns low-FPR ranking; it produces `best_score = s(q_t, a*)`.
The controller only decides where the accept line sits:

```
τ_t = Quantile(recent selected-H0 scores, 1 − α_t)     # tie-exact, reused from
                                                        # calibration.query_level
accept  ⇔  best_score ≥ τ_t
```

`α_t` is nudged by the ACI update on H0 events:

```
α_t ← clip(α_t + γ · (α_target − err_t),  α_min, α_max)
err_t = 1  iff  the H0 was accepted (a false hit)
```

Smaller `α_t` → higher quantile → stricter gate → fewer future false hits.
The buffer and fallback are seeded from the batch NP calibration
(`threshold_from_selected_h0_scores`) so cold-start uses the real threshold.

**Where adaptation actually comes from (be precise).** The recent-H0 buffer is a
FIFO window, but at realistic audit rates it barely turns over: with
`p_control≈0.04` and ~40% H0 density the buffer refreshes at ~1.6% of the stream,
so over a ~70k serve stream a 3k-entry buffer stays ~90% seeded calibration
scores (`controller.post_seed_buffer_fraction` reports this per run, ~0.10 in
practice). **So the drift adaptation is carried by `α_t`, not by buffer
turnover** — the quantile shape is roughly fixed by calibration, and ACI slides
the operating point on it. Treat the buffer as "which score distribution the
quantile is read from," not as a fast drift tracker. If you need the quantile
*shape* to track drift too, raise `p_control`/shrink `buffer_size`, or decouple
the buffer from the seed.

**Wilson gate on the hard budget.** Vanilla ACI drives FPR toward α from both
sides, so on sparse updates it can loosen a conservative threshold until FPR
*overshoots* α — and our NP score is a hard constraint (0 if FPR > α). With
`wilson_gate=True`, an *upward* (loosening) `α_t` move is applied only while
`wilson_upper_bound(control-audit FP, control-audit H0) ≤ α_target`; *downward*
(tightening) moves are always allowed. The replay runs `fixed_*`, `aci_*`
(ungated), and `aci_wilson_*` (gated) so the TPR cost of the gate is visible.

### 2. Sparse auditing — and why two channels

Production does not label every hit. We judge only a small fraction, split into:

* **control-audit** — uniform, **margin-independent** `Bernoulli(p_control)`
  (~3–5%). **Only these events feed the ACI update.** Because the draw does not
  depend on the score margin, the controller sees an unbiased sample of served
  H0 outcomes and targets the *served* FPR — not the FPR of the near-threshold
  subpopulation that a margin-dependent audit would over-sample.
* **diagnostic-audit** — margin-dependent (heavier near `τ_t`). Feeds
  hard-example mining, region diagnostics, local-separator candidates, and
  importance-weighted diagnostic metrics. It must **not** drive the controller.

This split is the one correctness change from the original spec: feeding raw,
margin-biased audit outcomes into ACI would make it control the wrong FPR.

### 3. Metrics: ground-truth vs what production can see

* **Ground-truth (full-label)** — FPR/TPR/hit-rate/utility over *every* labeled
  selected event. Legitimate offline: the label is revealed only *after* the
  decision. This is the true achieved performance and the headline number.
* **Audited estimates** — what a deployment could measure from sparse judging:
  `raw` (all judged, biased), `control` (unbiased slice), `iw`
  (Horvitz–Thompson over diagnostic audits, weighted by `1/p_diagnostic`, with a
  Kish effective sample size so you know when to trust it).

FPR is computed only over H0 events, TPR only over H1; unknown labels never
enter either.

**Cost-adjusted utility.** A cache hit is only a win if the judge bill it implies
is smaller than the provider calls it saves. `metrics.py` reports
`cost_adjusted_utility = true_hits − judge_cost_ratio · judged` (CLI
`--judge-cost-ratio`, default 0.1). This matters because judged fraction (~5%) is
larger than correct-hit utility (~1.5% of requests) on this data, so the raw
utility story is net-negative unless a judge call is much cheaper than a provider
call. Report it explicitly rather than quoting hit-rate alone.

### Anchor contamination (a known, bounded caveat)

Anchors are **cluster-mean embeddings computed over the whole corpus**, including
eval queries, so an eval query technically contributed to the mean of the anchor
it is later scored against. The fit/eval split is on *queries* (rows `< n_fit`
train the ensemble; rows `≥ n_fit` are scored), not on the anchor statistics.
The leakage is bounded and small: each query contributes weight `1/|cluster|` to
exactly one anchor mean, so for clusters of size ≫1 an eval query's own influence
on "its" anchor is negligible, and it has zero influence on every other anchor.
It does not touch the labels (labels come from the judge, revealed post-decision)
and it is identical across all methods, so it cannot explain a fixed-vs-ACI or
cosine-vs-ensemble *difference*. A fully clean setup would compute anchor means on
a held-out anchor corpus; that is a straightforward follow-up, not a correctness
bug in the comparison.

### 4. Offline replay with no leakage

Each scorer is scored **once** into a frozen table of selected candidates; every
method runs over the *same* table, so fixed-vs-ACI and cosine-vs-ensemble differ
only in the admission rule. At decision time a method sees only `best_score`; the
label is revealed afterwards. `order_stream` provides `natural`, `random`, and
`cluster_block` orderings to test whether fixed calibration breaks under
workload shift while ACI holds FPR.

## Running the baselines (A–D)

```bash
.conda/bin/python scripts/run_adaptive_replay.py \
    --npz data/h1h0_final.npz --output-dir runs_adaptive/shift \
    --max-rows 24000 --alpha 0.05 \
    --ordering random,cluster_block --buffer-size 3000
```

For each scorer the runner emits three methods — `fixed_*`, `aci_*` (ungated),
`aci_wilson_*` (Wilson-gated) — so baselines A/C (fixed), B/D (ACI), plus the
gated variants are all in one table. It also records verification diagnostics:
`table_separation` (Spearman + accept-disagreement between the cosine and
ensemble tables) and per-scorer `*_score_percentiles`, so a degenerate scorer or
a collapsed-to-identical pair of tables is caught in the report. Writes
`adaptive_replay_report.json` and `comparison_table.csv`.

Expect ACI's benefit to show under **shift** (`cluster_block`): a fixed threshold
calibrated on early clusters drifts off-budget on later traffic, while ACI
adapts `α_t`/`τ_t` from the control-audit slice to hold FPR ≤ α. Under
`random` order there is no shift, so a well-calibrated fixed threshold is already
near-optimal and ACI has little to correct — that is the expected null result,
not a failure.

## Reuse (not rebuilt here)

* Selected-H0 tie-exact quantile threshold — `calibration.query_level.threshold_from_selected_h0_scores` (public; shared by batch calibration and the ACI controller).
* Wilson bounds — `calibration.wilson` (drives the ACI loosening gate).
* Judge / labels — `feedback.h1h0_npz_adapters`.
* Sliding-window monitoring & refit gating already exist in `online/` and
  `policies/refit.py`.

## Deliberately postponed

Local-separator training, full retraining automation, per-region ACI, weighted
(DtACI-style) ACI updates, and the plotting script. Region diagnostics and
retraining triggers are next; the `snapshots` in `AdmissionMetrics` already carry
the sliding-window series those plots need.
