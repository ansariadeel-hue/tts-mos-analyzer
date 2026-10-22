"""Command-line interface: ``mos report ratings.csv [--normalize] [--drop-unreliable]``."""

from __future__ import annotations

import argparse
import sys
from typing import Sequence

from . import __version__
from .analysis import OVERALL
from .data import RatingsError, load_ratings
from .report import ReportOptions, build_report, format_json, format_text


def _probability(value: str) -> float:
    x = float(value)
    if not 0.0 < x < 1.0:
        raise argparse.ArgumentTypeError("must be between 0 and 1 (exclusive)")
    return x


def _positive_int(value: str) -> int:
    x = int(value)
    if x < 1:
        raise argparse.ArgumentTypeError("must be >= 1")
    return x


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mos",
        description="Analyse Mean Opinion Score (MOS) listening tests for TTS / AI dubbing.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    rep = sub.add_parser("report", help="MOS per system/dimension, rater screening and ranking")
    rep.add_argument("csv", help="ratings CSV with columns rater,system,utterance,dimension,score")
    rep.add_argument("--normalize", action="store_true",
                     help="per-rater z-score normalisation (rescaled to the 1-5 axis)")
    rep.add_argument("--drop-unreliable", action="store_true",
                     help="exclude raters flagged by the screening step")
    rep.add_argument("--json", action="store_true", help="emit JSON instead of text")
    rep.add_argument("--confidence", type=_probability, default=0.95,
                     help="confidence level for intervals (default 0.95)")
    rep.add_argument("--alpha", type=_probability, default=0.05,
                     help="significance level after Holm correction (default 0.05)")
    rep.add_argument("--bootstrap", type=_positive_int, default=2000,
                     help="bootstrap resamples per pair (default 2000)")
    rep.add_argument("--seed", type=int, default=0, help="bootstrap random seed (default 0)")
    rep.add_argument("--rank-by", default=OVERALL,
                     help="dimension used for ranking/significance (default: overall)")
    rep.add_argument("--min-corr", type=float, default=0.3,
                     help="flag raters whose leave-one-out correlation is below this (default 0.3)")
    rep.add_argument("--min-sd", type=float, default=0.5,
                     help="flag raters whose score SD is below this (default 0.5)")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # pragma: no cover - exotic streams
        pass

    if args.command == "report":
        try:
            ratings = load_ratings(args.csv)
            options = ReportOptions(
                normalize=args.normalize,
                drop_unreliable=args.drop_unreliable,
                confidence=args.confidence,
                alpha=args.alpha,
                n_boot=args.bootstrap,
                seed=args.seed,
                rank_by=args.rank_by,
                min_corr=args.min_corr,
                min_sd=args.min_sd,
            )
            report = build_report(ratings, options)
        except (OSError, RatingsError, ValueError) as exc:
            print(f"mos: error: {exc}", file=sys.stderr)
            return 2
        sys.stdout.write(format_json(report) if args.json else format_text(report))
        return 0
    parser.error(f"unknown command {args.command!r}")  # pragma: no cover
    return 2  # pragma: no cover


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
