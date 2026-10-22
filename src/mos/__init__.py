"""tts-mos-analyzer: MOS listening-test analysis for TTS and AI dubbing."""

from .analysis import (
    mos_table,
    pairwise_comparisons,
    rank_systems,
    rater_reliability,
    zscore_normalize,
)
from .data import Rating, RatingsError, load_ratings
from .report import ReportOptions, build_report

__version__ = "0.1.0"

__all__ = [
    "Rating",
    "RatingsError",
    "ReportOptions",
    "build_report",
    "load_ratings",
    "mos_table",
    "pairwise_comparisons",
    "rank_systems",
    "rater_reliability",
    "zscore_normalize",
    "__version__",
]
