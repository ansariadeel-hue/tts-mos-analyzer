import json
import random
from pathlib import Path

import pytest

from mos.analysis import (
    mos_table,
    pairwise_comparisons,
    rank_systems,
    rater_reliability,
    zscore_normalize,
)
from mos.data import Rating, RatingsError, load_ratings
from mos.report import ReportOptions, build_report, format_json, format_text
from mos.stats import mean, stdev

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "ratings.csv"


def synthetic(seed=7):
    """3 good raters, 1 random rater, 1 straight-liner; A > B > C."""
    rng = random.Random(seed)
    truth = {"A": 4.3, "B": 3.4, "C": 2.2}
    out = []
    for rater in ["g1", "g2", "g3", "rand", "flat"]:
        bias = {"g1": 0.3, "g2": -0.3, "g3": 0.0}.get(rater, 0.0)
        for system, mu in truth.items():
            for u in range(10):
                if rater == "rand":
                    s = rng.randint(1, 5)
                elif rater == "flat":
                    s = 3
                else:
                    s = max(1, min(5, round(mu + bias + rng.gauss(0, 0.4))))
                out.append(Rating(rater, system, f"u{u}", "naturalness", float(s)))
    return out


def write_csv(tmp_path, text):
    p = tmp_path / "r.csv"
    p.write_text(text, encoding="utf-8")
    return p


def test_load_example_file():
    ratings = load_ratings(EXAMPLE)
    assert len(ratings) == 1152
    assert {r.dimension for r in ratings} == {"naturalness", "prosody", "translation_accuracy"}


def test_loader_accepts_case_insensitive_headers_and_bom(tmp_path):
    p = write_csv(tmp_path, "﻿Rater,System,Utterance,Dimension,Score\nr1,a,u1,nat,4\n")
    assert load_ratings(p) == [Rating("r1", "a", "u1", "nat", 4.0)]


@pytest.mark.parametrize(
    "body,message",
    [
        ("rater,system,utterance,score\nr1,a,u1,4\n", "missing column"),
        ("rater,system,utterance,dimension,score\nr1,a,u1,nat,6\n", "outside"),
        ("rater,system,utterance,dimension,score\nr1,a,u1,nat,good\n", "not a number"),
        ("rater,system,utterance,dimension,score\nr1,a,u1,nat,4\nr1,a,u1,nat,5\n", "duplicate"),
        ("rater,system,utterance,dimension,score\nr1,,u1,nat,4\n", "missing value"),
    ],
)
def test_loader_rejects_bad_input(tmp_path, body, message):
    with pytest.raises(RatingsError, match=message):
        load_ratings(write_csv(tmp_path, body))


def test_zscore_normalize_equalises_rater_means():
    ratings = synthetic()
    norm = zscore_normalize([r for r in ratings if r.rater.startswith("g")])
    by_rater = {}
    for r in norm:
        by_rater.setdefault(r.rater, []).append(r.score)
    means = [mean(v) for v in by_rater.values()]
    sds = [stdev(v) for v in by_rater.values()]
    assert max(means) - min(means) == pytest.approx(0, abs=1e-9)
    assert max(sds) - min(sds) == pytest.approx(0, abs=1e-9)


def test_zscore_without_rescale_gives_unit_variance():
    ratings = [Rating("r", "s", f"u{i}", "d", float(v)) for i, v in enumerate([1, 2, 3, 4, 5])]
    z = [r.score for r in zscore_normalize(ratings, rescale=False)]
    assert mean(z) == pytest.approx(0)
    assert stdev(z) == pytest.approx(1)


def test_reliability_flags_random_and_flat_raters():
    stats = {s.rater: s for s in rater_reliability(synthetic())}
    assert not stats["g1"].unreliable and not stats["g2"].unreliable
    # LOO consensus includes the noisy raters, so good raters land well above 0.3
    assert stats["g1"].corr > 0.5 > stats["rand"].corr
    assert stats["rand"].unreliable and any(r.startswith("r ") for r in stats["rand"].reasons)
    assert stats["flat"].unreliable and stats["flat"].corr is None
    assert stats["flat"].reasons[0].startswith("sd 0.00")


def test_mos_table_values():
    ratings = [
        Rating("r1", "A", "u1", "nat", 4), Rating("r1", "A", "u1", "pros", 2),
        Rating("r2", "A", "u1", "nat", 5), Rating("r2", "A", "u1", "pros", 3),
    ]
    table = mos_table(ratings)
    assert table["A"]["nat"].mean == 4.5
    assert table["A"]["pros"].mean == 2.5
    # overall averages each (rater, utterance) trial first: (3 + 4) / 2
    assert table["A"]["overall"].mean == 3.5
    assert table["A"]["overall"].n == 2


def test_ranking_and_pairwise_significance():
    good = [r for r in synthetic() if r.rater.startswith("g")]
    ranking = rank_systems(good)
    assert [s for s, _ in ranking] == ["A", "B", "C"]
    comps = pairwise_comparisons(good, n_boot=500, seed=3)
    assert [(c.system_a, c.system_b) for c in comps] == [("A", "B"), ("A", "C"), ("B", "C")]
    assert all(c.significant for c in comps)
    assert all(c.result.mean_diff > 0 for c in comps)


def test_identical_systems_not_significant():
    rng = random.Random(0)
    ratings = []
    for rater in ["a", "b", "c"]:
        for u in range(15):
            ratings.append(Rating(rater, "X", f"u{u}", "nat", float(rng.randint(2, 5))))
            ratings.append(Rating(rater, "Y", f"u{u}", "nat", float(rng.randint(2, 5))))
    (comp,) = pairwise_comparisons(ratings, n_boot=1000)
    assert not comp.significant


def test_build_report_drop_unreliable_on_example():
    report = build_report(load_ratings(EXAMPLE), ReportOptions(drop_unreliable=True, n_boot=300))
    assert report["dropped_raters"] == ["r07", "r08"]
    assert report["input"]["n_ratings_used"] == 1152 - 2 * 144
    assert [r["system"] for r in report["ranking"]] == ["human_dub", "tts_v2", "tts_v1", "baseline"]
    pair = next(p for p in report["pairwise"] if (p["system_a"], p["system_b"]) == ("tts_v2", "tts_v1"))
    assert not pair["significant"]
    json.loads(format_json(report))
    text = format_text(report)
    assert "UNRELIABLE" in text and "[dropped]" in text


def test_build_report_unknown_rank_dimension():
    with pytest.raises(ValueError, match="unknown dimension"):
        build_report(synthetic(), ReportOptions(rank_by="clarity"))


def test_report_is_reproducible_with_seed():
    ratings = synthetic()
    a = build_report(ratings, ReportOptions(n_boot=200, seed=5))
    b = build_report(ratings, ReportOptions(n_boot=200, seed=5))
    assert a == b
