# Retrieved-Anchor Diagnostics (real online replay vs gold NP anchor)

- npz: `C:/Work/MLCache/data/h1h0_final.npz`  scorer: `cosine`  seed: 42
- cache_universe: 4000  stream_length: 2000  top_k: 5  target_fpr: 0.05  selection: mixed
- gold anchor rule: np_compat centroid_nearest over the cache universe (== np_compat_pair_eval anchor restricted to retrievable rows)

## Raw counts (every rate below prints numerator/denominator)

| count | value |
|---|---:|
| total_requests | 2000 |
| gold_anchor_available_count | 466 |
| gold_anchor_missing_count | 1534 |
| top1_is_gold_anchor_count | 118 |
| top1_wrong_anchor_count | 1881 |
| topk_contains_gold_anchor_count | 401 |
| topk_misses_gold_anchor_count | 1599 |
| accepted_gold_anchor | 188 |
| rejected_gold_anchor | 49 |
| accepted_wrong_anchor | 345 |
| rejected_wrong_anchor | 857 |
| same_cluster_wrong_anchor | 296 |
| different_cluster_wrong_anchor | 49 |
| active_judged_hits | 395 |
| active_unjudged_hits | 713 |
| active_tp | 81 |
| active_fp | 314 |
| served_hits | 1108 |
| misses | 892 |

## Derived rates

| rate | value (num/den) |
|---|---|
| gold_anchor_cache_availability_rate | 0.2330 (466/2000) |
| top1_gold_match_rate | 0.0590 (118/2000) |
| topk_gold_recall | 0.2005 (401/2000) |
| scorer_tpr_given_gold_anchor | 0.7932 (188/237) |
| wrong_anchor_accept_rate | 0.2870 (345/1202) |
| active_precision | 0.2051 (81/395) |
| active_fp_rate | 0.7949 (314/395) |
| active_hit_rate | 0.5540 (1108/2000) |

## Active false-positive attribution

- gold_anchor_absent: 260
- gold_present_not_in_topk: 7
- gold_in_topk_not_selected: 17
- other: 30
- same_cluster_confusion: 265
- different_cluster_confusion: 49
- **dominant active-FP bucket: gold_anchor_absent**

## Direct answers

1. Gold anchor available in cache: 0.2330 (466/2000).
2. Top-1 retrieval returns the gold anchor: 0.0590 (118/2000).
3. Gold anchor present in top-k: 0.2005 (401/2000).
4. When online serves a false positive (314 FPs), the dominant cause is `gold_anchor_absent` (absent=260, present-but-not-in-topk=7, in-topk-not-selected=17).
5. Wrong accepted anchors by cluster: same-cluster=265, different-cluster=49.
6. If the gold anchor were forced, the scorer accepts the H1 gold pair at 0.7932 (188/237) (forced gold-anchor acceptance among judged H1 gold pairs).
7. This is a RETRIEVAL / anchor-selection problem, NOT a scorer/calibration problem: the scorer accepts the forced gold anchor at a high rate (0.7932 (188/237)) -- consistent with the passing np_compat / forced-gold modes -- yet online serves 314 active FPs because the dominant active-FP bucket is `gold_anchor_absent` and top-1 retrieval returns the gold anchor only 0.0590 (118/2000) of the time (gold available 0.2330 (466/2000)).

## Interpretation rule

np_compat_pair_eval and forced_gold_anchor_eval already pass, so the scorer + NP threshold are not the cause unless `scorer_tpr_given_gold_anchor` is low (the scorer rejects even the correct, forced gold anchor). If gold-anchor acceptance is high but online active FPs are high, the failure is retrieval / anchor selection / cache state, per the attribution above -- NOT the scorer or calibration.
