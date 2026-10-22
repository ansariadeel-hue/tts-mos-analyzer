"""Small, dependency-free statistics toolkit used by the MOS analyzer.

Everything here is pure standard library: Student's t quantiles are computed
by numerically inverting the t CDF (regularized incomplete beta function,
evaluated with Lentz's continued fraction), so no scipy is required.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Sequence

__all__ = [
    "CI",
    "betainc",
    "holm",
    "mean",
    "mean_ci",
    "normal_cdf",
    "normal_ppf",
    "paired_bootstrap",
    "pearson",
    "stdev",
    "t_cdf",
    "t_critical",
    "t_ppf",
]


# --------------------------------------------------------------------------
# Basic descriptive statistics
# --------------------------------------------------------------------------

def mean(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("mean() of empty sequence")
    return math.fsum(values) / len(values)


def stdev(values: Sequence[float]) -> float:
    """Sample standard deviation (n - 1 denominator). Returns 0.0 for n < 2."""
    n = len(values)
    if n < 2:
        return 0.0
    m = mean(values)
    return math.sqrt(math.fsum((v - m) ** 2 for v in values) / (n - 1))


def pearson(x: Sequence[float], y: Sequence[float]) -> float | None:
    """Pearson correlation; ``None`` when undefined (n < 2 or zero variance)."""
    if len(x) != len(y):
        raise ValueError("pearson() needs sequences of equal length")
    n = len(x)
    if n < 2:
        return None
    mx, my = mean(x), mean(y)
    sxx = math.fsum((a - mx) ** 2 for a in x)
    syy = math.fsum((b - my) ** 2 for b in y)
    if sxx == 0 or syy == 0:
        return None
    sxy = math.fsum((a - mx) * (b - my) for a, b in zip(x, y))
    return max(-1.0, min(1.0, sxy / math.sqrt(sxx * syy)))


# --------------------------------------------------------------------------
# Distributions
# --------------------------------------------------------------------------

def _betacf(a: float, b: float, x: float) -> float:
    """Continued fraction for the incomplete beta function (modified Lentz)."""
    tiny = 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < tiny:
        d = tiny
    d = 1.0 / d
    h = d
    for m in range(1, 1000):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-15:
            break
    return h


def betainc(a: float, b: float, x: float) -> float:
    """Regularized incomplete beta function I_x(a, b)."""
    if not 0.0 <= x <= 1.0:
        raise ValueError("x must be in [0, 1]")
    if x == 0.0 or x == 1.0:
        return x
    log_front = (
        math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
        + a * math.log(x) + b * math.log1p(-x)
    )
    front = math.exp(log_front)
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def normal_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def _bisect(f, target: float, lo: float, hi: float) -> float:
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if f(mid) < target:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-13 * max(1.0, abs(mid)):
            break
    return 0.5 * (lo + hi)


def normal_ppf(p: float) -> float:
    """Inverse standard normal CDF (bisection on ``math.erf``)."""
    if not 0.0 < p < 1.0:
        raise ValueError("p must be in (0, 1)")
    if p == 0.5:
        return 0.0
    if p < 0.5:
        return -normal_ppf(1.0 - p)
    hi = 1.0
    while normal_cdf(hi) < p:
        hi *= 2.0
    return _bisect(normal_cdf, p, 0.0, hi)


def t_cdf(t: float, df: float) -> float:
    """CDF of Student's t distribution with ``df`` degrees of freedom."""
    if df <= 0:
        raise ValueError("df must be positive")
    if math.isinf(df):
        return normal_cdf(t)
    if t == 0:
        return 0.5
    x = df / (df + t * t)
    tail = 0.5 * betainc(df / 2.0, 0.5, x)
    return 1.0 - tail if t > 0 else tail


def t_ppf(p: float, df: float) -> float:
    """Inverse CDF (quantile) of Student's t, found by bisection on ``t_cdf``."""
    if not 0.0 < p < 1.0:
        raise ValueError("p must be in (0, 1)")
    if df <= 0:
        raise ValueError("df must be positive")
    if math.isinf(df):
        return normal_ppf(p)
    if p == 0.5:
        return 0.0
    if p < 0.5:
        return -t_ppf(1.0 - p, df)
    hi = 1.0
    while t_cdf(hi, df) < p:
        hi *= 2.0
    return _bisect(lambda t: t_cdf(t, df), p, 0.0, hi)


def t_critical(confidence: float, df: float) -> float:
    """Two-sided critical value, e.g. ``t_critical(0.95, 10) == 2.228...``."""
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be in (0, 1)")
    return t_ppf(1.0 - (1.0 - confidence) / 2.0, df)


# --------------------------------------------------------------------------
# Confidence intervals, bootstrap, multiple comparisons
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class CI:
    mean: float
    low: float | None
    high: float | None
    n: int
    sd: float

    @property
    def half_width(self) -> float | None:
        if self.low is None or self.high is None:
            return None
        return (self.high - self.low) / 2.0


def mean_ci(values: Sequence[float], confidence: float = 0.95) -> CI:
    """Mean with a t-distribution confidence interval (``None`` bounds if n < 2)."""
    n = len(values)
    m = mean(values)
    sd = stdev(values)
    if n < 2:
        return CI(m, None, None, n, sd)
    half = t_critical(confidence, n - 1) * sd / math.sqrt(n)
    return CI(m, m - half, m + half, n, sd)


@dataclass(frozen=True)
class BootstrapResult:
    mean_diff: float
    low: float
    high: float
    p_value: float
    n: int
    n_boot: int


def paired_bootstrap(
    diffs: Sequence[float],
    n_boot: int = 2000,
    seed: int = 0,
    confidence: float = 0.95,
) -> BootstrapResult:
    """Paired bootstrap test of H0: mean(diffs) == 0.

    ``diffs`` are per-pair differences (system A minus system B on the same
    rater/utterance). Pairs are resampled with replacement ``n_boot`` times
    using a ``random.Random(seed)`` generator, so results are reproducible.
    The two-sided p-value is ``2 * min(P(mean* <= 0), P(mean* >= 0))`` with a
    +1 correction so it is never exactly zero.
    """
    n = len(diffs)
    if n == 0:
        raise ValueError("paired_bootstrap() needs at least one pair")
    if n_boot < 1:
        raise ValueError("n_boot must be >= 1")
    rng = random.Random(seed)
    observed = mean(diffs)
    data = list(diffs)
    boot = []
    for _ in range(n_boot):
        sample = rng.choices(data, k=n)
        boot.append(math.fsum(sample) / n)
    boot.sort()
    le = sum(1 for b in boot if b <= 0)
    ge = sum(1 for b in boot if b >= 0)
    p = min(1.0, 2.0 * (min(le, ge) + 1) / (n_boot + 1))
    alpha = 1.0 - confidence
    lo_idx = max(0, int(math.floor(alpha / 2 * n_boot)))
    hi_idx = min(n_boot - 1, int(math.ceil((1 - alpha / 2) * n_boot)) - 1)
    return BootstrapResult(observed, boot[lo_idx], boot[hi_idx], p, n, n_boot)


def holm(p_values: Sequence[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values (same order as the input)."""
    m = len(p_values)
    order = sorted(range(m), key=lambda i: p_values[i])
    adjusted = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        value = min(1.0, (m - rank) * p_values[i])
        running = max(running, value)
        adjusted[i] = running
    return adjusted
