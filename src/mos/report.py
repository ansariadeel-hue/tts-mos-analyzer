"""Build a structured MOS report and render it as text or JSON."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Sequence

from .analysis import (
    OVERALL,
    drop_raters,
    mos_table,
    pairwise_comparisons,
    rank_systems,
    rater_reliability,
    zscore_normalize,
)
from .data import Rating


@dataclass
class ReportOptions:
    normalize: bool = False
    drop_unreliable: bool = False
    confidence: float = 0.95
    alpha: float = 0.05
    n_boot: int = 2000
    seed: int = 0
    rank_by: str = OVERALL
    min_corr: float = 0.3
    min_sd: float = 0.5
    min_overlap: int = 5


def _r(x: float | None, nd: int = 4) -> float | None:
    return None if x is None else round(x, nd)


def build_report(ratings: Sequence[Rating], options: ReportOptions | None = None) -> dict:
    opts = options or ReportOptions()
    dimensions = sorted({r.dimension for r in ratings})
    if opts.rank_by != OVERALL and opts.rank_by not in dimensions:
        raise ValueError(
            f"unknown dimension {opts.rank_by!r}; available: {', '.join(dimensions)}"
        )

    raters = rater_reliability(ratings, opts.min_corr, opts.min_sd, opts.min_overlap)
    flagged = sorted(r.rater for r in raters if r.unreliable)
    dropped: list[str] = []
    data = list(ratings)
    if opts.drop_unreliable and flagged:
        dropped = flagged
        data = drop_raters(data, set(flagged))
        if not data:
            raise ValueError("all raters were flagged as unreliable; nothing left to analyse")
    if opts.normalize:
        data = zscore_normalize(data)

    table = mos_table(data, opts.confidence)
    ranking = rank_systems(data, opts.rank_by, opts.confidence)
    comps = pairwise_comparisons(
        data, opts.rank_by, n_boot=opts.n_boot, seed=opts.seed,
        alpha=opts.alpha, confidence=opts.confidence,
    )

    return {
        "input": {
            "n_ratings": len(ratings),
            "n_raters": len({r.rater for r in ratings}),
            "n_systems": len({r.system for r in ratings}),
            "n_utterances": len({r.utterance for r in ratings}),
            "dimensions": dimensions,
            "n_ratings_used": len(data),
        },
        "options": {
            "normalize": opts.normalize,
            "drop_unreliable": opts.drop_unreliable,
            "confidence": opts.confidence,
            "alpha": opts.alpha,
            "bootstrap": opts.n_boot,
            "seed": opts.seed,
            "rank_by": opts.rank_by,
            "min_corr": opts.min_corr,
            "min_sd": opts.min_sd,
        },
        "raters": [
            {
                "rater": r.rater,
                "n": r.n,
                "mean": _r(r.mean),
                "sd": _r(r.sd),
                "corr_loo": _r(r.corr),
                "unreliable": r.unreliable,
                "reasons": r.reasons,
                "dropped": r.rater in dropped,
            }
            for r in raters
        ],
        "dropped_raters": dropped,
        "mos": {
            system: {
                dim: {
                    "mos": _r(ci.mean),
                    "ci_low": _r(ci.low),
                    "ci_high": _r(ci.high),
                    "n": ci.n,
                }
                for dim, ci in row.items()
            }
            for system, row in table.items()
        },
        "ranking": [
            {
                "rank": i,
                "system": system,
                "mos": _r(ci.mean),
                "ci_low": _r(ci.low),
                "ci_high": _r(ci.high),
                "n": ci.n,
            }
            for i, (system, ci) in enumerate(ranking, start=1)
        ],
        "pairwise": [
            {
                "system_a": c.system_a,
                "system_b": c.system_b,
                "diff": _r(c.result.mean_diff),
                "ci_low": _r(c.result.low),
                "ci_high": _r(c.result.high),
                "p": _r(c.result.p_value),
                "p_holm": _r(c.p_holm),
                "significant": c.significant,
                "n_pairs": c.result.n,
            }
            for c in comps
        ],
    }


# --------------------------------------------------------------------------
# Text rendering
# --------------------------------------------------------------------------

def _table(headers: list[str], rows: list[list[str]], left: tuple[int, ...] = (0,)) -> list[str]:
    widths = [max(len(h), *(len(row[i]) for row in rows)) if rows else len(h) for i, h in enumerate(headers)]

    def fmt(cells: list[str]) -> str:
        parts = []
        for i, cell in enumerate(cells):
            parts.append(cell.ljust(widths[i]) if i in left else cell.rjust(widths[i]))
        return "  " + "  ".join(parts).rstrip()

    lines = [fmt(headers), "  " + "  ".join("-" * w for w in widths)]
    lines.extend(fmt(row) for row in rows)
    return lines


def _cell(c: dict) -> str:
    if c["ci_low"] is None:
        return f"{c['mos']:.2f}"
    return f"{c['mos']:.2f} ± {(c['ci_high'] - c['ci_low']) / 2:.2f}"


def format_text(report: dict) -> str:
    inp, opt = report["input"], report["options"]
    conf = f"{opt['confidence'] * 100:g}%"
    out = ["MOS listening-test report", "=" * 25]
    out.append(
        f"Ratings: {inp['n_ratings']} | raters: {inp['n_raters']} | systems: {inp['n_systems']}"
        f" | utterances: {inp['n_utterances']} | dimensions: {', '.join(inp['dimensions'])}"
    )
    out.append(
        f"Options: normalize={'yes' if opt['normalize'] else 'no'}, "
        f"drop_unreliable={'yes' if opt['drop_unreliable'] else 'no'}, CI={conf}, "
        f"bootstrap={opt['bootstrap']} (seed {opt['seed']}), rank_by={opt['rank_by']}"
    )

    out += ["", f"Rater screening (r = correlation with leave-one-out mean of other raters;"
                f" flag if r < {opt['min_corr']:.2f} or sd < {opt['min_sd']:.2f})"]
    rows = []
    for r in report["raters"]:
        status = "ok"
        if r["unreliable"]:
            status = "UNRELIABLE: " + "; ".join(r["reasons"])
            if r["dropped"]:
                status += " [dropped]"
        corr = "n/a" if r["corr_loo"] is None else f"{r['corr_loo']:.2f}"
        rows.append([r["rater"], str(r["n"]), f"{r['mean']:.2f}", f"{r['sd']:.2f}", corr, status])
    out += _table(["rater", "n", "mean", "sd", "r", "status"], rows, left=(0, 5))

    dims = inp["dimensions"] + [OVERALL]
    out += ["", f"MOS by system and dimension (mean ± {conf} CI half-width"
                f"{', z-normalised' if opt['normalize'] else ''})"]
    rows = []
    for rank in report["ranking"]:
        system = rank["system"]
        rows.append([system] + [_cell(report["mos"][system][d]) if d in report["mos"][system] else "-" for d in dims])
    out += _table(["system"] + dims, rows)

    out += ["", f"Ranking by {opt['rank_by']}"]
    for r in report["ranking"]:
        ci = "" if r["ci_low"] is None else f"  [{r['ci_low']:.2f}, {r['ci_high']:.2f}]"
        out.append(f"  {r['rank']}. {r['system']:<12} {r['mos']:.2f}{ci}  (n={r['n']})")

    out += ["", f"Pairwise differences (paired bootstrap over rater x utterance trials,"
                f" Holm-adjusted, alpha={opt['alpha']:g})"]
    rows = []
    for c in report["pairwise"]:
        rows.append([
            f"{c['system_a']} vs {c['system_b']}",
            f"{c['diff']:+.2f}",
            f"[{c['ci_low']:+.2f}, {c['ci_high']:+.2f}]",
            f"{c['p']:.4f}",
            f"{c['p_holm']:.4f}",
            "yes" if c["significant"] else "no",
        ])
    out += _table(["pair", "diff", f"{conf} CI", "p", "p_holm", "significant"], rows)
    return "\n".join(out) + "\n"


def format_json(report: dict) -> str:
    return json.dumps(report, indent=2, ensure_ascii=False) + "\n"
