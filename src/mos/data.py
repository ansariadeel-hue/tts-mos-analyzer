"""Loading and validating MOS rating CSV files."""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable

REQUIRED_COLUMNS = ("rater", "system", "utterance", "dimension", "score")
SCORE_MIN = 1.0
SCORE_MAX = 5.0


class RatingsError(ValueError):
    """Raised for malformed rating files."""


@dataclass(frozen=True)
class Rating:
    rater: str
    system: str
    utterance: str
    dimension: str
    score: float

    @property
    def key(self) -> tuple[str, str, str, str]:
        return (self.rater, self.system, self.utterance, self.dimension)

    def with_score(self, score: float) -> "Rating":
        return replace(self, score=score)


def parse_rows(rows: Iterable[dict], source: str = "<input>") -> list[Rating]:
    ratings: list[Rating] = []
    seen: dict[tuple, int] = {}
    for lineno, row in enumerate(rows, start=2):
        if all((v or "").strip() == "" for v in row.values() if isinstance(v, str)):
            continue  # blank line
        values = {}
        for col in REQUIRED_COLUMNS:
            raw = row.get(col)
            if raw is None or str(raw).strip() == "":
                raise RatingsError(f"{source}:{lineno}: missing value for '{col}'")
            values[col] = str(raw).strip()
        try:
            score = float(values["score"])
        except ValueError:
            raise RatingsError(
                f"{source}:{lineno}: score {values['score']!r} is not a number"
            ) from None
        if math.isnan(score) or not SCORE_MIN <= score <= SCORE_MAX:
            raise RatingsError(
                f"{source}:{lineno}: score {values['score']} outside "
                f"{SCORE_MIN:g}-{SCORE_MAX:g}"
            )
        rating = Rating(
            values["rater"], values["system"], values["utterance"],
            values["dimension"], score,
        )
        if rating.key in seen:
            raise RatingsError(
                f"{source}:{lineno}: duplicate rating for "
                f"rater={rating.rater} system={rating.system} "
                f"utterance={rating.utterance} dimension={rating.dimension} "
                f"(first seen on line {seen[rating.key]})"
            )
        seen[rating.key] = lineno
        ratings.append(rating)
    if not ratings:
        raise RatingsError(f"{source}: no ratings found")
    return ratings


def load_ratings(path: str | Path) -> list[Rating]:
    """Read a CSV with columns rater, system, utterance, dimension, score."""
    path = Path(path)
    with path.open(newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None:
            raise RatingsError(f"{path}: empty file")
        normalized = [(name or "").strip().lower() for name in reader.fieldnames]
        missing = [c for c in REQUIRED_COLUMNS if c not in normalized]
        if missing:
            raise RatingsError(f"{path}: missing column(s): {', '.join(missing)}")
        reader.fieldnames = normalized
        return parse_rows(reader, source=str(path))
