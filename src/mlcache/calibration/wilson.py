"""Wilson confidence interval helpers."""

from __future__ import annotations

from math import sqrt


def wilson_upper_bound(successes: int, n: int, z: float = 1.96) -> float | None:
    """Return the Wilson score upper bound for a binomial proportion."""

    if successes < 0:
        raise ValueError("successes must be non-negative")
    if n < 0:
        raise ValueError("n must be non-negative")
    if successes > n:
        raise ValueError("successes must be less than or equal to n")
    if z <= 0.0:
        raise ValueError("z must be positive")
    if n == 0:
        return None

    phat = float(successes) / float(n)
    z2 = float(z) * float(z)
    denom = 1.0 + z2 / float(n)
    center = phat + z2 / (2.0 * float(n))
    radius = float(z) * sqrt((phat * (1.0 - phat) / float(n)) + (z2 / (4.0 * float(n) * float(n))))
    return min(1.0, (center + radius) / denom)

