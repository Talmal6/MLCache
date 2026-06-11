# Retrieved-Anchor Diagnostics (real online replay vs gold NP anchor)

- npz: `C:/Work/MLCache/data/h1h0_final.npz`  scorer: `ensemble`  seed: 42
- cache_universe: 4000  stream_length: 1000  top_k: 5  target_fpr: 0.05  selection: mixed
- gold anchor rule: np_compat centroid_nearest over the cache universe (== np_compat_pair_eval anchor restricted to retrievable rows)

## Raw counts (every rate below prints numerator/denominator)

| count | value |
|---|---:|
| total_requests | 1000 |
| gold_anchor_available_count | 779 |
| gold_anchor_missing_count | 221 |
| top1_is_gold_anchor_count | 109 |
| top1_wrong_anchor_count | 891 |
| topk_contains_gold_anchor_count | 557 |
| topk_misses_gold_anchor_count | 443 |
| accepted_gold_anchor | 111 |
| rejected_gold_anchor | 20 |
| accepted_wrong_anchor | 179 |
| rejected_wrong_anchor | 370 |
| same_cluster_wrong_anchor | 154 |
| different_cluster_wrong_anchor | 25 |
| active_judged_hits | 198 |
| active_unjudged_hits | 386 |
| active_tp | 138 |
| active_fp | 60 |
| served_hits | 584 |
| misses | 416 |

## Derived rates

| rate | value (num/den) |
|---|---|
| gold_anchor_cache_availability_rate | 0.7790 (779/1000) |
| top1_gold_match_rate | 0.1090 (109/1000) |
| topk_gold_recall | 0.5570 (557/1000) |
| scorer_tpr_given_gold_anchor | 0.8473 (111/131) |
| wrong_anchor_accept_rate | 0.3260 (179/549) |
| active_precision | 0.6970 (138/198) |
| active_fp_rate | 0.3030 (60/198) |
| active_hit_rate | 0.5840 (584/1000) |

## Active false-positive attribution

- gold_anchor_absent: 19
- gold_present_not_in_topk: 12
- gold_in_topk_not_selected: 27
- other: 2
- same_cluster_confusion: 35
- different_cluster_confusion: 25
- **dominant active-FP bucket: gold_in_topk_not_selected**

## Direct answers

1. Gold anchor available in cache: 0.7790 (779/1000).
2. Top-1 retrieval returns the gold anchor: 0.1090 (109/1000).
3. Gold anchor present in top-k: 0.5570 (557/1000).
4. When online serves a false positive (60 FPs), the dominant cause is `gold_in_topk_not_selected` (absent=19, present-but-not-in-topk=12, in-topk-not-selected=27).
5. Wrong accepted anchors by cluster: same-cluster=35, different-cluster=25.
6. If the gold anchor were forced, the scorer accepts the H1 gold pair at 0.8473 (111/131) (forced gold-anchor acceptance among judged H1 gold pairs).
7. This is a RETRIEVAL / anchor-selection problem, NOT a scorer/calibration problem: the scorer accepts the forced gold anchor at a high rate (0.8473 (111/131)) -- consistent with the passing np_compat / forced-gold modes -- yet online serves 60 active FPs because the dominant active-FP bucket is `gold_in_topk_not_selected` and top-1 retrieval returns the gold anchor only 0.1090 (109/1000) of the time (gold available 0.7790 (779/1000)).

## Interpretation rule

np_compat_pair_eval and forced_gold_anchor_eval already pass, so the scorer + NP threshold are not the cause unless `scorer_tpr_given_gold_anchor` is low (the scorer rejects even the correct, forced gold anchor). If gold-anchor acceptance is high but online active FPs are high, the failure is retrieval / anchor selection / cache state, per the attribution above -- NOT the scorer or calibration.
