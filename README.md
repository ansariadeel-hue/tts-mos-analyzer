# tts-mos-analyzer

> Turn raw MOS listening-test ratings for TTS and AI dubbing into confidence intervals, rater screening and statistically tested system rankings.

[![CI](https://github.com/ansariadeel-hue/tts-mos-analyzer/actions/workflows/ci.yml/badge.svg)](https://github.com/ansariadeel-hue/tts-mos-analyzer/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://www.python.org/)

## Why

Mean Opinion Score (MOS) tests are the standard way to evaluate synthetic speech
and AI dubbing: listeners rate each clip 1–5 on dimensions such as *naturalness*,
*prosody* and *translation accuracy*. In practice the raw averages are not enough:

- a bare mean hides how uncertain it is — with a few raters and utterances two
  systems 0.1 apart are often indistinguishable;
- crowd raters differ in leniency and scale use, and some click randomly or
  give every clip the same score;
- "system A beats system B" needs a paired test, because both systems were
  heard by the same raters on the same sentences.

`tts-mos-analyzer` covers these steps in one dependency-free command, so a rating
export from an annotation tool can be turned into a defensible report.

## Features

- **MOS per system × dimension** with **95% t-distribution confidence intervals**.
  The t critical value is computed without scipy, by numerically inverting the
  Student-t CDF (regularized incomplete beta + bisection) — verified against
  standard t-table values in the tests.
- **Overall MOS** per system computed from per-(rater, utterance) trial averages,
  so one trial is not counted once per dimension.
- **Per-rater z-score normalisation** (`--normalize`), rescaled back onto the
  1–5 axis using the grand mean and SD so numbers stay readable.
- **Unreliable-rater detection**: Pearson correlation of each rater with the
  **leave-one-out mean of the other raters** on the same items, plus a minimum
  score SD to catch straight-liners. `--drop-unreliable` excludes them.
- **System ranking with pairwise significance** via a **paired bootstrap**
  (fixed seed, reproducible) over shared rater × utterance trials, with
  **Holm–Bonferroni** correction across all pairs.
- Text or **JSON** report (`--json`); rank by `overall` or a single dimension
  (`--rank-by prosody`).
- Pure Python standard library at runtime.

## Install

```bash
pip install git+https://github.com/ansariadeel-hue/tts-mos-analyzer
```

## Quickstart

```bash
mos report examples/ratings.csv --drop-unreliable
```

Output on the bundled example (8 raters × 4 systems × 12 utterances × 3 dimensions;
`r07` is a simulated random clicker and `r08` a straight-liner):

```text
MOS listening-test report
=========================
Ratings: 1152 | raters: 8 | systems: 4 | utterances: 12 | dimensions: naturalness, prosody, translation_accuracy
Options: normalize=no, drop_unreliable=yes, CI=95%, bootstrap=2000 (seed 0), rank_by=overall

Rater screening (r = correlation with leave-one-out mean of other raters; flag if r < 0.30 or sd < 0.50)
  rater    n  mean    sd     r  status
  -----  ---  ----  ----  ----  ------------------------------------
  r01    144  3.39  0.93  0.54  ok
  r02    144  3.68  0.88  0.59  ok
  r03    144  3.74  0.97  0.67  ok
  r04    144  4.04  0.89  0.70  ok
  r05    144  3.49  1.01  0.67  ok
  r06    144  3.48  0.96  0.61  ok
  r07    144  3.16  1.39  0.05  UNRELIABLE: r 0.05 < 0.30 [dropped]
  r08    144  4.00  0.00   n/a  UNRELIABLE: sd 0.00 < 0.50 [dropped]

MOS by system and dimension (mean ± 95% CI half-width)
  system     naturalness      prosody  translation_accuracy      overall
  ---------  -----------  -----------  --------------------  -----------
  human_dub  4.38 ± 0.15  4.21 ± 0.17           4.49 ± 0.12  4.36 ± 0.10
  tts_v2     3.68 ± 0.20  3.21 ± 0.17           4.12 ± 0.18  3.67 ± 0.14
  tts_v1     3.43 ± 0.18  3.22 ± 0.16           4.17 ± 0.18  3.61 ± 0.12
  baseline   2.62 ± 0.18  2.44 ± 0.16           3.65 ± 0.16  2.91 ± 0.12

Ranking by overall
  1. human_dub    4.36  [4.26, 4.46]  (n=72)
  2. tts_v2       3.67  [3.53, 3.81]  (n=72)
  3. tts_v1       3.61  [3.49, 3.72]  (n=72)
  4. baseline     2.91  [2.79, 3.03]  (n=72)

Pairwise differences (paired bootstrap over rater x utterance trials, Holm-adjusted, alpha=0.05)
  pair                    diff          95% CI       p  p_holm  significant
  ---------------------  -----  --------------  ------  ------  -----------
  human_dub vs tts_v2    +0.69  [+0.56, +0.81]  0.0010  0.0060          yes
  human_dub vs tts_v1    +0.75  [+0.62, +0.87]  0.0010  0.0060          yes
  human_dub vs baseline  +1.45  [+1.33, +1.56]  0.0010  0.0060          yes
  tts_v2 vs tts_v1       +0.06  [-0.06, +0.19]  0.2979  0.2979           no
  tts_v2 vs baseline     +0.76  [+0.63, +0.90]  0.0010  0.0060          yes
  tts_v1 vs baseline     +0.70  [+0.56, +0.83]  0.0010  0.0060          yes
```

Other useful flags:

```bash
mos report examples/ratings.csv --normalize          # per-rater z-scores
mos report examples/ratings.csv --json > report.json # machine-readable
mos report examples/ratings.csv --rank-by naturalness --bootstrap 5000 --seed 1
mos report examples/ratings.csv --min-corr 0.4 --min-sd 0.6 --confidence 0.99
```

Exit code is `0` on success and `2` on input errors (missing file, bad column,
score outside 1–5, duplicate rating).

### JSON output (excerpt)

Selected keys from `mos report examples/ratings.csv --drop-unreliable --json` (the full
document also contains `input`, `options`, `raters` and the full `mos` table):

```json
{
  "dropped_raters": [
    "r07",
    "r08"
  ],
  "ranking": [
    {
      "rank": 1,
      "system": "human_dub",
      "mos": 4.3565,
      "ci_low": 4.2561,
      "ci_high": 4.4569,
      "n": 72
    },
    {
      "rank": 2,
      "system": "tts_v2",
      "mos": 3.6713,
      "ci_low": 3.5344,
      "ci_high": 3.8082,
      "n": 72
    }
  ],
  "pairwise": [
    {
      "system_a": "tts_v2",
      "system_b": "tts_v1",
      "diff": 0.0648,
      "ci_low": -0.0556,
      "ci_high": 0.1898,
      "p": 0.2979,
      "p_holm": 0.2979,
      "significant": false,
      "n_pairs": 72
    }
  ]
}
```

## Methods

| Step | What is computed |
|---|---|
| CI | `mean ± t(1-α/2, n-1) · sd / √n` over individual ratings (per dimension) or per-trial averages (overall). |
| Normalisation | `z = (x − mean_rater) / sd_rater`, then `grand_mean + grand_sd · z`. Zero-variance raters map to the grand mean. |
| Rater screening | For each item a rater scored, mean of the *other* raters on that item; flag if Pearson r < `--min-corr` (default 0.3) or rater SD < `--min-sd` (default 0.5). Correlation needs ≥ 5 shared items. |
| Pairwise test | Differences A − B on shared (rater, utterance) trials; 2000 bootstrap resamples (`random.Random(seed)`); two-sided p = 2·min(P(mean* ≤ 0), P(mean* ≥ 0)) with +1 smoothing, so the smallest reportable p is ≈ 2/(B+1). Holm adjustment across all pairs. |

## Data format

A CSV with a header row containing (case-insensitive, any column order):

| column | meaning |
|---|---|
| `rater` | rater / listener id |
| `system` | TTS or dubbing system id |
| `utterance` | stimulus / sentence id |
| `dimension` | e.g. `naturalness`, `prosody`, `translation_accuracy` |
| `score` | number in 1–5 (decimals allowed) |

```csv
rater,system,utterance,dimension,score
r01,human_dub,utt01,naturalness,4
r01,human_dub,utt01,prosody,3
```

Each (rater, system, utterance, dimension) combination may appear once.

## Python API

```python
from mos import load_ratings, build_report, ReportOptions

ratings = load_ratings("examples/ratings.csv")
report = build_report(ratings, ReportOptions(drop_unreliable=True, normalize=True))
print(report["ranking"][0])
```

## Project structure

```
tts-mos-analyzer/
├── src/mos/
│   ├── stats.py      # t CDF/quantile, CIs, Pearson, paired bootstrap, Holm
│   ├── data.py       # CSV loading and validation
│   ├── analysis.py   # normalisation, rater screening, MOS table, ranking
│   ├── report.py     # report dict + text/JSON rendering
│   └── cli.py        # `mos report ...`
├── tests/            # pytest suite (reference values for t quantiles, Holm, CIs)
├── examples/ratings.csv
└── .github/workflows/ci.yml
```

Run the tests with `python -m pytest -q` from the project root.

## License

MIT — see [LICENSE](LICENSE).
