"""Core MOS analysis: normalisation, rater screening, MOS tables, ranking."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from itertools import combinations
from typing import Sequence

from .data import Rating
from .stats import CI, BootstrapResult, holm, mean, mean_ci, paired_bootstrap, pearson, stdev

OVERALL = "overall"


# --------------------------------------------------------------------------
# Normalisation
# --------------------------------------------------------------------------

def zscore_normalize(ratings: Sequence[Rating], rescale: bool = True) -> list[Rating]:
    """Per-rater z-score normalisation.

    Each rater's scores are transformed to ``z = (x - mean_r) / sd_r``, which
    removes individual leniency/severity and scale-usage differences. With
    ``rescale=True`` (the default) the z-scores are mapped back onto the
    original scale using the grand mean and the pooled standard deviation of
    all raw scores, so normalised MOS values remain readable on the 1-5 axis.
    Raters with zero variance get ``z = 0`` (i.e. the grand mean).
    """
    by_rater: dict[str, list[float]] = defaultdict(list)
    for r in ratings:
        by_rater[r.rater].append(r.score)
    params = {rater: (mean(v), stdev(v)) for rater, v in by_rater.items()}
    all_scores = [r.score for r in ratings]
    grand_mean = mean(all_scores)
    grand_sd = stdev(all_scores) or 1.0
    out = []
    for r in ratings:
        m, sd = params[r.rater]
        z = (r.score - m) / sd if sd > 0 else 0.0
        out.append(r.with_score(grand_mean + grand_sd * z if rescale else z))
    return out


# --------------------------------------------------------------------------
# Rater reliability
# --------------------------------------------------------------------------

@dataclass
class RaterStats:
    rater: str
    n: int
    mean: float
    sd: float
    corr: float | None
    n_overlap: int
    reasons: list[str] = field(default_factory=list)

    @property
    def unreliable(self) -> bool:
        return bool(self.reasons)


def rater_reliability(
    ratings: Sequence[Rating],
    min_corr: float = 0.3,
    min_sd: float = 0.5,
    min_overlap: int = 5,
) -> list[RaterStats]:
    """Screen raters against the leave-one-out consensus of the other raters.

    For every item (system, utterance, dimension) a rater scored, we take the
    mean score the *other* raters gave that item. A rater is flagged when the
    Pearson correlation between their scores and these leave-one-out means is
    below ``min_corr`` (or undefined), or when their own scores have a
    standard deviation below ``min_sd`` (e.g. straight-lining "4" on
    everything). Raters sharing fewer than ``min_overlap`` items with anyone
    else are reported but not flagged on correlation.
    """
    items: dict[tuple[str, str, str], dict[str, float]] = defaultdict(dict)
    by_rater: dict[str, list[Rating]] = defaultdict(list)
    for r in ratings:
        items[(r.system, r.utterance, r.dimension)][r.rater] = r.score
        by_rater[r.rater].append(r)

    results = []
    for rater in sorted(by_rater):
        own, others = [], []
        for r in by_rater[rater]:
            peer = [s for who, s in items[(r.system, r.utterance, r.dimension)].items() if who != rater]
            if peer:
                own.append(r.score)
                others.append(mean(peer))
        scores = [r.score for r in by_rater[rater]]
        sd = stdev(scores)
        corr = pearson(own, others) if len(own) >= 2 else None
        reasons = []
        if sd < min_sd:
            reasons.append(f"sd {sd:.2f} < {min_sd:.2f}")
        if len(own) >= min_overlap:
            if corr is None:
                if sd >= min_sd:
                    reasons.append("correlation undefined")
            elif corr < min_corr:
                reasons.append(f"r {corr:.2f} < {min_corr:.2f}")
        results.append(RaterStats(rater, len(scores), mean(scores), sd, corr, len(own), reasons))
    return results


def drop_raters(ratings: Sequence[Rating], raters: set[str]) -> list[Rating]:
    return [r for r in ratings if r.rater not in raters]


# --------------------------------------------------------------------------
# MOS tables
# --------------------------------------------------------------------------

def trial_scores(ratings: Sequence[Rating], dimension: str = OVERALL) -> dict[str, dict[tuple[str, str], float]]:
    """Per system, one score per (rater, utterance) trial.

    With ``dimension == "overall"`` the trial score is the mean over all
    dimensions the rater scored for that utterance; otherwise only that
    dimension is used.
    """
    acc: dict[str, dict[tuple[str, str], list[float]]] = defaultdict(lambda: defaultdict(list))
    for r in ratings:
        if dimension != OVERALL and r.dimension != dimension:
            continue
        acc[r.system][(r.rater, r.utterance)].append(r.score)
    return {
        system: {k: mean(v) for k, v in trials.items()}
        for system, trials in acc.items()
    }


def mos_table(ratings: Sequence[Rating], confidence: float = 0.95) -> dict[str, dict[str, CI]]:
    """MOS with t-based CI for every system x dimension, plus ``overall``.

    Per-dimension cells use the individual ratings. The ``overall`` column
    averages each (rater, utterance) trial over dimensions first so that the
    CI is not artificially narrowed by counting one trial several times.
    """
    cells: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for r in ratings:
        cells[r.system][r.dimension].append(r.score)
    overall = trial_scores(ratings, OVERALL)
    table: dict[str, dict[str, CI]] = {}
    for system in sorted(cells):
        row = {dim: mean_ci(vals, confidence) for dim, vals in sorted(cells[system].items())}
        row[OVERALL] = mean_ci(list(overall[system].values()), confidence)
        table[system] = row
    return table


# --------------------------------------------------------------------------
# Ranking and pairwise significance
# --------------------------------------------------------------------------

@dataclass
class Comparison:
    system_a: str
    system_b: str
    result: BootstrapResult
    p_holm: float = 1.0
    alpha: float = 0.05

    @property
    def significant(self) -> bool:
        return self.p_holm < self.alpha


def rank_systems(
    ratings: Sequence[Rating],
    dimension: str = OVERALL,
    confidence: float = 0.95,
) -> list[tuple[str, CI]]:
    trials = trial_scores(ratings, dimension)
    if not trials:
        raise ValueError(f"no ratings for dimension {dimension!r}")
    ranked = [(s, mean_ci(list(t.values()), confidence)) for s, t in trials.items()]
    ranked.sort(key=lambda item: (-item[1].mean, item[0]))
    return ranked


def pairwise_comparisons(
    ratings: Sequence[Rating],
    dimension: str = OVERALL,
    n_boot: int = 2000,
    seed: int = 0,
    alpha: float = 0.05,
    confidence: float = 0.95,
) -> list[Comparison]:
    """Paired bootstrap for every pair of systems, Holm-corrected.

    Pairs are (rater, utterance) trials that both systems received; the
    statistic is the mean of A - B over those trials. Systems are ordered by
    their MOS so ``system_a`` is always the higher-ranked one.
    """
    trials = trial_scores(ratings, dimension)
    order = [s for s, _ in rank_systems(ratings, dimension, confidence)]
    comparisons = []
    for i, (a, b) in enumerate(combinations(order, 2)):
        shared = sorted(set(trials[a]) & set(trials[b]))
        if not shared:
            continue
        diffs = [trials[a][k] - trials[b][k] for k in shared]
        # Derive a distinct but deterministic seed per pair.
        result = paired_bootstrap(diffs, n_boot=n_boot, seed=seed + i, confidence=confidence)
        comparisons.append(Comparison(a, b, result, alpha=alpha))
    adjusted = holm([c.result.p_value for c in comparisons])
    for c, p in zip(comparisons, adjusted):
        c.p_holm = p
    return comparisons
