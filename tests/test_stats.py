import math

import pytest

from mos.stats import (
    betainc,
    holm,
    mean_ci,
    normal_ppf,
    paired_bootstrap,
    pearson,
    stdev,
    t_cdf,
    t_critical,
    t_ppf,
)

# Two-sided 95% critical values from standard t tables.
T_TABLE_95 = {
    1: 12.7062,
    2: 4.3027,
    3: 3.1824,
    4: 2.7764,
    5: 2.5706,
    10: 2.2281,
    20: 2.0860,
    30: 2.0423,
    60: 2.0003,
    120: 1.9799,
}


@pytest.mark.parametrize("df,expected", sorted(T_TABLE_95.items()))
def test_t_critical_95_matches_table(df, expected):
    assert t_critical(0.95, df) == pytest.approx(expected, abs=1e-4)


@pytest.mark.parametrize(
    "conf,df,expected",
    [(0.99, 10, 3.1693), (0.90, 5, 2.0150), (0.99, 1, 63.6567), (0.80, 30, 1.3104)],
)
def test_t_critical_other_levels(conf, df, expected):
    assert t_critical(conf, df) == pytest.approx(expected, abs=1e-4)


def test_t_approaches_normal_for_large_df():
    assert t_critical(0.95, 1_000_000) == pytest.approx(1.959964, abs=1e-4)
    assert t_critical(0.95, math.inf) == pytest.approx(1.959964, abs=1e-6)
    assert normal_ppf(0.975) == pytest.approx(1.959964, abs=1e-6)


def test_t_cdf_and_ppf_are_inverse_and_symmetric():
    for df in (1, 3, 7.5, 40):
        for p in (0.01, 0.2, 0.5, 0.9, 0.999):
            assert t_cdf(t_ppf(p, df), df) == pytest.approx(p, abs=1e-10)
        assert t_ppf(0.1, df) == pytest.approx(-t_ppf(0.9, df))
    # df = 1 is the Cauchy distribution: CDF(1) = 0.75
    assert t_cdf(1.0, 1) == pytest.approx(0.75, abs=1e-12)


def test_betainc_known_values():
    assert betainc(1, 1, 0.3) == pytest.approx(0.3)  # uniform
    assert betainc(2, 3, 0.4) == pytest.approx(0.5248, abs=1e-4)
    assert betainc(0.5, 0.5, 0.5) == pytest.approx(0.5)


def test_invalid_arguments():
    with pytest.raises(ValueError):
        t_ppf(1.0, 5)
    with pytest.raises(ValueError):
        t_critical(0.95, 0)
    with pytest.raises(ValueError):
        t_critical(1.5, 10)


def test_mean_ci_reference():
    ci = mean_ci([1, 2, 3, 4, 5], 0.95)
    # mean 3, sd sqrt(2.5), se = sqrt(0.5), t(4) = 2.776445
    assert ci.mean == 3
    assert ci.sd == pytest.approx(math.sqrt(2.5))
    assert ci.half_width == pytest.approx(2.776445 * math.sqrt(0.5), abs=1e-5)
    assert ci.low == pytest.approx(3 - 1.963243, abs=1e-5)


def test_mean_ci_single_value_has_no_interval():
    ci = mean_ci([4.0])
    assert ci.mean == 4.0 and ci.low is None and ci.half_width is None


def test_pearson_and_stdev():
    assert pearson([1, 2, 3], [2, 4, 6]) == pytest.approx(1.0)
    assert pearson([1, 2, 3], [3, 2, 1]) == pytest.approx(-1.0)
    assert pearson([1, 2, 3, 4], [1, 3, 2, 4]) == pytest.approx(0.8)
    assert pearson([1, 1, 1], [1, 2, 3]) is None
    assert stdev([2, 4, 4, 4, 5, 5, 7, 9]) == pytest.approx(2.13809, abs=1e-5)


def test_holm_reference():
    # Classic example: p = [0.01, 0.04, 0.03, 0.005] -> [0.03, 0.06, 0.06, 0.02]
    assert holm([0.01, 0.04, 0.03, 0.005]) == pytest.approx([0.03, 0.06, 0.06, 0.02])
    assert holm([0.9, 0.8]) == pytest.approx([1.0, 1.0])


def test_paired_bootstrap_is_deterministic_and_detects_effect():
    diffs = [0.5, 0.7, 0.2, 0.9, 0.4, 0.6, 0.3, 0.8, 0.5, 0.6]
    a = paired_bootstrap(diffs, n_boot=1000, seed=42)
    b = paired_bootstrap(diffs, n_boot=1000, seed=42)
    assert a == b
    assert a.mean_diff == pytest.approx(0.55)
    assert a.p_value == pytest.approx(2 / 1001)  # no resample <= 0
    assert a.low > 0


def test_paired_bootstrap_null_effect():
    diffs = [1, -1, 0.5, -0.5, 0.2, -0.2, 0, 0.1, -0.1, 0]
    res = paired_bootstrap(diffs, n_boot=2000, seed=1)
    assert res.p_value > 0.5
    assert res.low < 0 < res.high
