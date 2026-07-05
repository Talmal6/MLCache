"""Weighted ensemble scorer."""

from __future__ import annotations

from typing import Iterable

from mlcache.calibration import ThresholdCalibrationRequest
from mlcache.features import PairFeatures
from mlcache.scorers.base import BaseScorer, SemanticScorer
from mlcache.scorers.cosine import CosineScorer
from mlcache.scorers.lda import LDAScorer
from mlcache.scorers.mlp import TinyMLPScorer
from mlcache.scorers.pca_whitened import PCAWhitenedCosineScorer
from mlcache.scorers.utils import as_matrix, np_module, scores_to_threshold
from mlcache.semantic_types import InputSpace, LabeledPairBatch, ScorerName, Score


_WEIGHT_HOLDOUT_FRACTION = 0.3
_MIN_FIT_SAMPLES = 4
_WEIGHT_RANDOM_CANDIDATES = 768

# Score-normalization modes for mixing members onto a common scale.
#   "zscore"  : (score - mean_ref) / std_ref. Unbounded above, so H1 pairs can
#               score strictly higher than the densest H0 region instead of
#               everything piling onto a shared maximum. This is the default
#               because the bounded "h0_cdf" map crushes the upper tail into an
#               atom at 1.0 that no threshold can split (see git history /
#               the FPR-overshoot diagnosis).
#   "h0_cdf"  : smoothed empirical-H0 percentile (rank + 0.5)/(n + 1) in (0,1).
#   "none"    : raw member scores (only sensible when members already share a
#               scale, e.g. the unit tests' column scorers).
_NORMALIZATION_MODES = ("zscore", "h0_cdf", "none")
_NORM_EPS = 1e-12
_WEIGHT_TIE_TOL = 1e-4


def _holdout_split(x):
    """Split rows into (fit_subset, weight_selection_subset).

    Falls back to using all rows for both subsets when there isn't enough
    data to carve out a non-degenerate holdout (e.g. tiny test fixtures);
    in that case weight selection is in-sample, same as before this fix.
    """
    np = np_module()
    n = x.shape[0]
    if n < _MIN_FIT_SAMPLES * 2:
        return x, x
    rng = np.random.default_rng(42)
    order = rng.permutation(n)
    n_holdout = max(1, int(round(n * _WEIGHT_HOLDOUT_FRACTION)))
    n_holdout = min(n_holdout, n - _MIN_FIT_SAMPLES)
    holdout_idx = order[:n_holdout]
    fit_idx = order[n_holdout:]
    return x[fit_idx], x[holdout_idx]


class EnsembleScorer(BaseScorer):
    """Non-negative weighted ensemble over fitted scorer judges."""

    _name = ScorerName("Ensemble")
    _input_space = InputSpace.MIXED

    def __init__(
        self,
        judges: Iterable[SemanticScorer] | None = None,
        *,
        normalization: str = "zscore",
    ) -> None:
        self._judges = list(judges) if judges is not None else [
            CosineScorer(),
            PCAWhitenedCosineScorer(),
            LDAScorer(),
            TinyMLPScorer(),
        ]
        if not self._judges:
            raise ValueError("EnsembleScorer requires at least one judge")
        normalization = str(normalization).lower()
        if normalization not in _NORMALIZATION_MODES:
            raise ValueError(
                f"normalization must be one of {_NORMALIZATION_MODES}, got {normalization!r}"
            )
        self._normalization = normalization
        self._weights = tuple(1.0 / len(self._judges) for _ in self._judges)
        # Per-mode normalization state (only the active mode's fields are set).
        self._h0_score_reference = None  # "h0_cdf": sorted H0 scores per member
        self._norm_mu = None             # "zscore": per-member mean
        self._norm_std = None            # "zscore": per-member std

    @property
    def normalization(self) -> str:
        return self._normalization

    def members(self) -> Iterable[SemanticScorer]:
        return tuple(self._judges)

    @property
    def weights(self) -> tuple[float, ...]:
        return tuple(float(weight) for weight in self._weights)

    def copy_for_refit(self):
        return type(self)(
            judges=[judge.copy_for_refit() for judge in self._judges],
            normalization=self._normalization,
        )

    def fit(self, batch: LabeledPairBatch, **kwargs: object) -> None:
        h0 = as_matrix(batch.h0, name="h0")
        h1 = as_matrix(batch.h1, name="h1")

        # Hold out a slice of each class for weight selection so that flexible,
        # *fitted* judges (lda/pca_whitened/xgboost/mlp) don't get to vote on
        # their own fitting data. Scored in-sample, those judges can partially
        # memorize this batch and look stronger than the unfit `cosine` judge,
        # biasing the weight search away from the component that actually
        # generalizes best. `cosine` is unaffected either way since it never
        # fits anything.
        h0_fit, h0_weight = _holdout_split(h0)
        h1_fit, h1_weight = _holdout_split(h1)

        fit_batch = LabeledPairBatch(h0=h0_fit, h1=h1_fit, weights=None, metadata=batch.metadata)
        for judge in self._judges:
            judge.fit(fit_batch, **kwargs)

        z0_raw = self._score_matrix(h0_weight)
        z1_raw = self._score_matrix(h1_weight)

        # Members use incompatible score scales (e.g. cosine similarities,
        # LDA margins, and classifier probabilities). Put every member onto a
        # common scale before mixing so that a large numerical range cannot
        # dominate the convex weights. The transform is fit on the held-out
        # weight-selection slice and reused at serving time.
        self._fit_normalization(z0_raw, z1_raw)
        z0 = self._normalize_score_matrix(z0_raw)
        z1 = self._normalize_score_matrix(z1_raw)
        self._weights = self._fit_np_weights(z0, z1, alpha=float(kwargs.get("alpha", 0.05)))

    def _score_matrix(self, x):
        from mlcache.features import PairFeatures

        np = np_module()
        rows = []
        for row in x:
            features = PairFeatures(hadamard=tuple(float(v) for v in row))
            rows.append([float(judge.score(features)) for judge in self._judges])
        return np.asarray(rows, dtype=np.float64)

    def _fit_normalization(self, z0_raw, z1_raw) -> None:
        np = np_module()
        self._h0_score_reference = None
        self._norm_mu = None
        self._norm_std = None

        if self._normalization == "none":
            return
        if self._normalization == "h0_cdf":
            self._h0_score_reference = np.sort(z0_raw, axis=0)
            return
        if self._normalization == "zscore":
            # Centre/scale on BOTH classes' weight-selection scores so the
            # transform reflects the full score range, not just H0. Affine with
            # positive scale, so member rescaling leaves normalized scores (and
            # therefore the fitted weights) invariant.
            stacked = np.vstack([z0_raw, z1_raw])
            self._norm_mu = stacked.mean(axis=0)
            self._norm_std = np.maximum(stacked.std(axis=0), _NORM_EPS)
            return

    def _normalize_score_matrix(self, scores):
        np = np_module()
        if self._normalization == "none":
            return np.asarray(scores, dtype=np.float64)

        if self._normalization == "zscore":
            if self._norm_mu is None or self._norm_std is None:
                return np.asarray(scores, dtype=np.float64)
            return (np.asarray(scores, dtype=np.float64) - self._norm_mu) / self._norm_std

        # h0_cdf
        if self._h0_score_reference is None:
            return np.asarray(scores, dtype=np.float64)
        scores = np.asarray(scores, dtype=np.float64)
        normalized = np.empty_like(scores, dtype=np.float64)
        for idx in range(scores.shape[1]):
            reference = self._h0_score_reference[:, idx]
            n = float(reference.size)
            if n == 0:
                normalized[:, idx] = scores[:, idx]
                continue
            # Smoothed empirical-H0 CDF: (rank + 0.5)/(n + 1) keeps the output
            # strictly inside (0, 1) so the densest H0 region does not land on a
            # hard 1.0 that the threshold cannot sit above.
            rank = np.searchsorted(reference, scores[:, idx], side="right").astype(np.float64)
            normalized[:, idx] = (rank + 0.5) / (n + 1.0)
        return normalized

    def _fit_np_weights(self, z0, z1, *, alpha: float):
        np = np_module()
        m = z0.shape[1]
        candidates = []
        for idx in range(m):
            w = np.zeros(m, dtype=np.float64)
            w[idx] = 1.0
            candidates.append(w)
        uniform = np.ones(m, dtype=np.float64) / m
        candidates.append(uniform)

        # Explicit pairwise mixtures cover the edges of the simplex, where a
        # small contribution from a complementary judge is often most useful.
        for left in range(m):
            for right in range(left + 1, m):
                for left_weight in (0.25, 0.5, 0.75):
                    w = np.zeros(m, dtype=np.float64)
                    w[left] = left_weight
                    w[right] = 1.0 - left_weight
                    candidates.append(w)

        rng = np.random.default_rng(42)
        concentrations = (0.25, 1.0, 4.0)
        for idx in range(_WEIGHT_RANDOM_CANDIDATES):
            concentration = concentrations[idx % len(concentrations)]
            candidates.append(rng.dirichlet(np.full(m, concentration, dtype=np.float64)))

        best_w = uniform
        best_tpr = -1.0
        best_entropy = -1.0
        best_fpr = 2.0
        for w in candidates:
            s0 = z0 @ w
            tau = scores_to_threshold(
                ThresholdCalibrationRequest(
                    h0_scores=[Score(float(s)) for s in s0],
                    target_false_accept_rate=alpha,
                )
            )
            if not np.isfinite(float(tau)):
                # Infeasible: this weight vector's H0 cannot be thresholded to
                # the FPR budget at all. Never deploy it.
                continue
            s1 = z1 @ w
            tpr = float(np.mean(s1 >= float(tau)))
            fpr = float(np.mean(s0 >= float(tau)))
            entropy = self._weight_entropy(w)
            # Maximize TPR; break near-ties toward the MORE SPREAD weight vector
            # (higher entropy) and then toward LOWER FPR. The old rule preferred
            # higher FPR, which collapsed onto a single saturating judge and left
            # an H0/H1 score atom no threshold could separate.
            if tpr > best_tpr + _WEIGHT_TIE_TOL:
                better = True
            elif tpr >= best_tpr - _WEIGHT_TIE_TOL:
                if entropy > best_entropy + _WEIGHT_TIE_TOL:
                    better = True
                elif entropy >= best_entropy - _WEIGHT_TIE_TOL and fpr < best_fpr:
                    better = True
                else:
                    better = False
            else:
                better = False
            if better:
                best_tpr = tpr
                best_entropy = entropy
                best_fpr = fpr
                best_w = w
        return best_w / max(float(best_w.sum()), 1e-12)

    @staticmethod
    def _weight_entropy(w) -> float:
        np = np_module()
        ww = np.maximum(np.asarray(w, dtype=np.float64).reshape(-1), 0.0)
        total = float(ww.sum())
        if total <= _NORM_EPS:
            return 0.0
        ww = ww / total
        return float(-np.sum(ww * np.log(np.maximum(ww, _NORM_EPS))))

    def score(self, features: PairFeatures) -> Score:
        np = np_module()
        scores = np.asarray([float(judge.score(features)) for judge in self._judges], dtype=np.float64)
        scores = self._normalize_score_matrix(scores.reshape(1, -1))[0]
        return Score(float(scores @ self._weights))
