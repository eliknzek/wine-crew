"""EDA: eda_summary, make_charts, build_html_report.

Plain deterministic functions (no LLM). eda_summary produces every number the
Insights Writer may quote; the charts and the HTML report only display data,
they never compute anything the summary does not also contain.
"""

from __future__ import annotations

import base64
import html
import json
import re
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

CHART_NAMES = [
    "points_distribution",
    "price_distribution_log",
    "points_vs_log_price",
    "by_country",
    "by_variety",
    "points_by_vintage",
    "missing_values",
]
CHART_TITLES = {
    "points_distribution": "Points distribution",
    "price_distribution_log": "Price distribution (log scale)",
    "points_vs_log_price": "Points vs log price",
    "by_country": "Median points and price by country",
    "by_variety": "Median points and price by variety",
    "points_by_vintage": "Points by vintage",
    "missing_values": "Missing values by column",
}


def _r(x: float, nd: int = 4) -> float:
    return round(float(x), nd)


def _describe(s: pd.Series) -> dict[str, float]:
    s = s.dropna().astype("float64")
    q = s.quantile([0.05, 0.25, 0.5, 0.75, 0.95])
    return {
        "count": int(s.size), "mean": _r(s.mean()), "std": _r(s.std()),
        "min": _r(s.min()), "p05": _r(q[0.05]), "p25": _r(q[0.25]), "median": _r(q[0.5]),
        "p75": _r(q[0.75]), "p95": _r(q[0.95]), "max": _r(s.max()),
    }


def _group_table(df: pd.DataFrame, col: str, groups: list[str]) -> list[dict[str, Any]]:
    sub = df[df[col].astype(object).isin(groups)]
    g = sub.groupby(sub[col].astype(object), sort=False)
    stats = pd.DataFrame({
        "count": g.size(),
        "share_pct": g.size() / len(df) * 100,
        "median_points": g["points"].median(),
        "mean_points": g["points"].mean(),
        "median_price": g["price"].median(),
    }).reindex(groups)
    return [
        {"name": str(name), "count": int(r["count"]), "share_pct": _r(r["share_pct"], 2),
         "median_points": _r(r["median_points"], 2), "mean_points": _r(r["mean_points"], 2),
         "median_price": _r(r["median_price"], 2)}
        for name, r in stats.iterrows()
    ]


def _top_groups(df: pd.DataFrame, col: str, n: int) -> list[str]:
    counts = df[col].astype(object).value_counts()
    # Stable order: count desc, then name asc.
    counts = counts.sort_index().sort_values(ascending=False, kind="mergesort")
    return [str(x) for x in counts.index[:n]]


def _price_bands(df: pd.DataFrame, edges: list[float], plateau_gain: float) -> dict[str, Any]:
    """Mean points per price band, and where the gain per price doubling falls below plateau_gain.

    Bands have unequal widths, so the gain between consecutive bands is divided by
    the number of price doublings between their geometric midpoints.
    """
    codes = pd.cut(df["price"], bins=edges, right=True, include_lowest=True, labels=False)
    g = df.groupby(codes)["points"]
    size, mean, median = g.size(), g.mean(), g.median()
    rows = []
    for i, (low, high) in enumerate(zip(edges[:-1], edges[1:])):
        count = int(size.get(i, 0))
        rows.append({
            "band": f"${low:g}-${high:g}", "low": float(low), "high": float(high),
            "geo_mid": _r(np.sqrt(low * high), 2), "count": count,
            "mean_points": _r(mean[i], 2) if count else None,
            "median_points": _r(median[i], 2) if count else None,
        })
    prev = None
    for row in rows:
        row["gain_vs_previous"] = None
        row["gain_per_price_doubling"] = None
        if row["mean_points"] is None:
            continue
        if prev is not None:
            gain = row["mean_points"] - prev["mean_points"]
            doublings = np.log2(row["geo_mid"] / prev["geo_mid"])
            row["gain_vs_previous"] = _r(gain, 2)
            row["gain_per_price_doubling"] = _r(gain / doublings, 2)
        prev = row
    plateau = next(
        (r["band"] for r in rows
         if r["gain_per_price_doubling"] is not None and r["gain_per_price_doubling"] < plateau_gain),
        None,
    )
    return {
        "bands": rows,
        "plateau_gain_per_doubling_threshold": plateau_gain,
        "plateau_starts_at_band": plateau,
    }


def eda_summary(clean_df: pd.DataFrame, config: dict[str, Any]) -> dict[str, Any]:
    """Every number the Insights Writer may use. JSON-serializable and deterministic."""
    eda = config["eda"]
    df = clean_df
    top_countries = _top_groups(df, "country", eda["top_n"])
    top_varieties = _top_groups(df, "variety", eda["top_n"])
    by_country = _group_table(df, "country", top_countries)
    by_variety = _group_table(df, "variety", top_varieties)

    overall_median_points = float(df["points"].median())
    overall_median_price = float(df["price"].median())

    def value_groups(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        hits = [r for r in rows
                if r["median_points"] >= overall_median_points and r["median_price"] <= overall_median_price]
        return sorted(hits, key=lambda r: (-r["median_points"], r["median_price"], r["name"]))

    vint = df.dropna(subset=["vintage"])
    vg = vint.groupby(vint["vintage"].astype(int))["points"]
    vstats = pd.DataFrame({"count": vg.size(), "mean": vg.mean(), "median": vg.median()})
    vstats = vstats[vstats["count"] >= eda["vintage_min_count"]].sort_index()

    country_share = df["country"].astype(object).value_counts(normalize=True) * 100
    country_share = country_share.sort_index().sort_values(ascending=False, kind="mergesort")

    eda_price = _price_bands(df, eda["price_bands"], eda["plateau_gain_per_doubling"])
    return {
        "row_count": int(len(df)),
        "top_n": int(eda["top_n"]),
        "points": _describe(df["points"]),
        "price": _describe(df["price"]),
        "corr_points_log_price": _r(df["points"].corr(df["log_price"])),
        "price_bands": eda_price,
        "by_country": by_country,
        "by_variety": by_variety,
        "value_countries": value_groups(by_country),
        "value_varieties": value_groups(by_variety),
        "overall_median_points": _r(overall_median_points, 2),
        "overall_median_price": _r(overall_median_price, 2),
        "by_vintage": [
            {"vintage": int(y), "count": int(r["count"]), "mean_points": _r(r["mean"], 2),
             "median_points": _r(r["median"], 2)}
            for y, r in vstats.iterrows()
        ],
        "vintage_min_count": eda["vintage_min_count"],
        "missing_pct": {c: _r(df[c].isna().mean() * 100, 2) for c in df.columns},
        "country_bias": {
            "n_countries": int(df["country"].nunique()),
            "top_country": str(country_share.index[0]),
            "top_country_share_pct": _r(country_share.iloc[0], 2),
            "top3_share_pct": _r(country_share.iloc[:3].sum(), 2),
            "shares_pct": {str(k): _r(v, 2) for k, v in country_share.iloc[: eda["top_n"]].items()},
        },
    }


# --------------------------------------------------------------------------- insights check

_NUMBER = re.compile(r"(?<![\d.])\d[\d,]*(?:\.\d+)?")
_LIST_MARKER = re.compile(r"^\s*(?:#{1,6}\s*)?\d+[.)]\s", re.MULTILINE)


def _summary_numbers(obj: Any) -> list[float]:
    """Every number in the summary, including numbers inside strings such as "$4-$10"."""
    out: list[float] = []
    if isinstance(obj, bool) or obj is None:
        return out
    if isinstance(obj, (int, float)):
        out.append(float(obj))
    elif isinstance(obj, str):
        out.extend(float(m.replace(",", "")) for m in _NUMBER.findall(obj))
    elif isinstance(obj, dict):
        for k, v in obj.items():
            out.extend(_summary_numbers(k))
            out.extend(_summary_numbers(v))
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            out.extend(_summary_numbers(v))
    return out


def insight_number_violations(text: str, summary: dict[str, Any]) -> list[str]:
    """Numbers in text that are not a rounding of some number in summary.

    A number written with d decimals matches a summary value v when it is within
    half a unit of the d-th decimal (so 45% matches 45.04 and 88.6 matches 88.56).
    Numbered-list markers ("1. ") are not counted as numbers.
    """
    values = sorted(set(_summary_numbers(summary)))
    body = _LIST_MARKER.sub("", text)
    violations: list[str] = []
    for token in _NUMBER.findall(body):
        number = float(token.replace(",", ""))
        decimals = len(token.split(".")[1]) if "." in token else 0
        tolerance = 0.5 * 10 ** (-decimals) + 1e-9
        if not any(abs(v - number) <= tolerance for v in values):
            violations.append(token)
    return list(dict.fromkeys(violations))


INSIGHT_HEADINGS = [
    "## באיזה טווח מחיר העלייה בניקוד מאטה?",
    "## אילו מדינות וזנים נותנים ניקוד גבוה במחיר נמוך?",
    "## היכן הנתונים מוטים?",
    "## המלצות לקנייה",
]
FORBIDDEN_PHRASES = {
    "מפסיק לעלות": "the score keeps rising; say the gain slows ('מאטה')",
    "האזור הראשון": "region_1 is 'אזור'",
    "קטגוריית ההגדרה": "designation is 'שם הסדרה/היין'",
    "דירוג": "points is 'ניקוד'",
}
_TRAILING_ZERO = re.compile(r"(?<![\d.])\d+\.0(?!\d)")
_LONG_PERCENT = re.compile(r"(?<![\d.])\d+\.\d{2,}\s*%")


def _section(text: str, heading: str) -> str:
    start = text.find(heading)
    if start < 0:
        return ""
    rest = text[start + len(heading):]
    nxt = rest.find("\n## ")
    return rest if nxt < 0 else rest[:nxt]


def _names_in(text: str, names: list[str]) -> list[str]:
    """Names mentioned in text, longest first so "Red Blend" is not found inside "Bordeaux-style Red Blend"."""
    found = []
    for name in sorted(set(names), key=len, reverse=True):
        pattern = re.compile(rf"(?<!\w){re.escape(name)}(?!\w)")
        if pattern.search(text):
            found.append(name)
            text = pattern.sub(" ", text)
    return found


def insight_style_violations(text: str, summary: dict[str, Any]) -> list[str]:
    """Deterministic style rules for insights.md: headings, terms, rounding, value-only picks."""
    out: list[str] = []
    positions = [text.find(h) for h in INSIGHT_HEADINGS]
    for heading, pos in zip(INSIGHT_HEADINGS, positions):
        if pos < 0:
            out.append(f"missing heading: {heading}")
    found = [p for p in positions if p >= 0]
    if found != sorted(found):
        out.append("headings are not in the required order")
    for phrase, why in FORBIDDEN_PHRASES.items():
        if phrase in text:
            out.append(f"do not write '{phrase}': {why}")
    for token in dict.fromkeys(_TRAILING_ZERO.findall(text)):
        out.append(f"write {token} as a whole number ({token[:-2]})")
    for token in dict.fromkeys(_LONG_PERCENT.findall(text)):
        out.append(f"round the percentage {token.strip()} to a whole number")

    value_section = _section(text, INSIGHT_HEADINGS[1])
    allowed = {r["name"] for r in summary.get("value_countries", []) + summary.get("value_varieties", [])}
    candidates = [r["name"] for r in summary.get("by_country", []) + summary.get("by_variety", [])]
    for name in _names_in(value_section, candidates + sorted(allowed)):
        if name not in allowed:
            out.append(f"{name} is not in value_countries or value_varieties; name only those")
    return out


# --------------------------------------------------------------------------- charts


def _new_fig(config: dict[str, Any], ncols: int = 1):
    eda = config["eda"]
    return plt.subplots(1, ncols, figsize=tuple(eda["chart_size"]), dpi=eda["chart_dpi"])


def _save(fig, path: Path, config: dict[str, Any]) -> Path:
    fig.tight_layout()
    fig.savefig(path, dpi=config["eda"]["chart_dpi"], format="png", metadata={"Software": None})
    plt.close(fig)
    return path


def _group_chart(df: pd.DataFrame, col: str, n: int, title: str, config: dict[str, Any], path: Path) -> Path:
    groups = _top_groups(df, col, n)[::-1]  # largest at top of a horizontal bar chart
    rows = {r["name"]: r for r in _group_table(df, col, groups)}
    fig, (ax1, ax2) = _new_fig(config, ncols=2)
    ax1.barh(groups, [rows[g]["median_points"] for g in groups], color="#7b2d43")
    ax1.set_xlim(80, max(rows[g]["median_points"] for g in groups) + 2)
    ax1.set_xlabel("Median points")
    ax2.barh(groups, [rows[g]["median_price"] for g in groups], color="#c9a227")
    ax2.set_xlabel("Median price (USD)")
    ax2.set_yticklabels([])
    fig.suptitle(title)
    return _save(fig, path, config)


def make_charts(clean_df: pd.DataFrame, out_dir: str | Path, config: dict[str, Any]) -> dict[str, Path]:
    """Write the EDA charts from docs/spec.md as PNG files. Returns {name: path}."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = clean_df
    n = config["eda"]["top_n"]
    paths: dict[str, Path] = {}

    fig, ax = _new_fig(config)
    ax.hist(df["points"], bins=np.arange(79.5, 101.5, 1), color="#7b2d43", edgecolor="white")
    ax.set(xlabel="Points", ylabel="Wines", title=CHART_TITLES["points_distribution"])
    paths["points_distribution"] = _save(fig, out_dir / "points_distribution.png", config)

    fig, ax = _new_fig(config)
    ax.hist(df["price"], bins=np.logspace(np.log10(df["price"].min()), np.log10(df["price"].max()), 50),
            color="#c9a227", edgecolor="white")
    ax.set_xscale("log")
    ax.set(xlabel="Price (USD, log scale)", ylabel="Wines", title=CHART_TITLES["price_distribution_log"])
    paths["price_distribution_log"] = _save(fig, out_dir / "price_distribution_log.png", config)

    fig, ax = _new_fig(config)
    hb = ax.hexbin(df["log_price"], df["points"], gridsize=40, bins="log", cmap="viridis", mincnt=1)
    fig.colorbar(hb, ax=ax, label="Wines (log)")
    ax.set(xlabel="log(price)", ylabel="Points", title=CHART_TITLES["points_vs_log_price"])
    paths["points_vs_log_price"] = _save(fig, out_dir / "points_vs_log_price.png", config)

    paths["by_country"] = _group_chart(
        df, "country", n, f"{CHART_TITLES['by_country']} (top {n} by count)", config, out_dir / "by_country.png")
    paths["by_variety"] = _group_chart(
        df, "variety", n, f"{CHART_TITLES['by_variety']} (top {n} by count)", config, out_dir / "by_variety.png")

    vint = df.dropna(subset=["vintage"])
    vg = vint.groupby(vint["vintage"].astype(int))["points"]
    vstats = pd.DataFrame({"count": vg.size(), "mean": vg.mean()})
    vstats = vstats[vstats["count"] >= config["eda"]["vintage_min_count"]]
    fig, ax = _new_fig(config)
    ax.plot(vstats.index, vstats["mean"], marker="o", color="#7b2d43")
    ax.set(xlabel="Vintage", ylabel="Mean points",
           title=f"{CHART_TITLES['points_by_vintage']} (vintages with >= {config['eda']['vintage_min_count']} wines)")
    paths["points_by_vintage"] = _save(fig, out_dir / "points_by_vintage.png", config)

    missing = (df.isna().mean() * 100).sort_values(kind="mergesort")
    fig, ax = _new_fig(config)
    ax.barh(missing.index.astype(str), missing.values, color="#555555")
    for y, v in enumerate(missing.values):
        ax.text(v + 0.3, y, f"{v:.1f}%", va="center", fontsize=8)
    ax.set(xlabel="Missing (%)", title=CHART_TITLES["missing_values"])
    paths["missing_values"] = _save(fig, out_dir / "missing_values.png", config)

    return paths


# --------------------------------------------------------------------------- HTML report


def _table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    head = "".join(f"<th>{html.escape(c)}</th>" for c in columns)
    body = "".join(
        "<tr>" + "".join(f"<td>{html.escape(str(r.get(c, '')))}</td>" for c in columns) + "</tr>"
        for r in rows
    )
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def build_html_report(summary: dict[str, Any], chart_paths: dict[str, str | Path], out_path: str | Path) -> Path:
    """Self-contained eda_report.html: charts embedded as base64, no external resources."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pts, price, bias = summary["points"], summary["price"], summary["country_bias"]
    plateau = summary["price_bands"]["plateau_starts_at_band"]

    key_facts = [
        ("Wines", f"{summary['row_count']:,}"),
        ("Points mean / median", f"{pts['mean']:.2f} / {pts['median']:.0f}"),
        ("Price median (p05–p95)", f"${price['median']:.0f} (${price['p05']:.0f}–${price['p95']:.0f})"),
        ("Correlation points ~ log price", f"{summary['corr_points_log_price']:.3f}"),
        ("Score stops rising from", plateau or "not within the bands"),
        ("Largest country share", f"{bias['top_country']} {bias['top_country_share_pct']:.1f}%"),
    ]
    facts_html = "".join(f"<tr><th>{html.escape(k)}</th><td>{html.escape(v)}</td></tr>" for k, v in key_facts)

    charts_html = []
    for name, path in chart_paths.items():
        data = base64.b64encode(Path(path).read_bytes()).decode("ascii")
        title = html.escape(CHART_TITLES.get(name, name))
        charts_html.append(
            f'<figure id="chart-{html.escape(name)}"><img alt="{title}" '
            f'src="data:image/png;base64,{data}"><figcaption>{title}</figcaption></figure>'
        )

    sections = [
        "<h2>Key facts</h2>", f"<table class='facts'>{facts_html}</table>",
        "<h2>Charts</h2>", *charts_html,
        "<h2>Points by price band</h2>",
        _table(summary["price_bands"]["bands"],
               ["band", "count", "mean_points", "median_points", "gain_vs_previous", "gain_per_price_doubling"]),
        "<h2>Top countries</h2>",
        _table(summary["by_country"], ["name", "count", "share_pct", "median_points", "median_price"]),
        "<h2>Top varieties</h2>",
        _table(summary["by_variety"], ["name", "count", "share_pct", "median_points", "median_price"]),
        "<h2>Missing values (%)</h2>",
        _table([{"column": k, "missing_pct": v} for k, v in summary["missing_pct"].items()], ["column", "missing_pct"]),
        "<h2>Full summary (JSON)</h2>",
        f"<pre>{html.escape(json.dumps(summary, indent=2, ensure_ascii=False))}</pre>",
    ]
    doc = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Wine EDA Report</title>
<style>
body {{ font-family: system-ui, sans-serif; max-width: 960px; margin: 2rem auto; padding: 0 1rem; color: #222; background: #fff; }}
h1 {{ color: #7b2d43; }}
table {{ border-collapse: collapse; margin: 0.5rem 0 1.5rem; font-size: 0.9rem; }}
th, td {{ border: 1px solid #ddd; padding: 0.3rem 0.6rem; text-align: left; }}
thead th, table.facts th {{ background: #f6f0f2; }}
figure {{ margin: 1.5rem 0; }}
img {{ max-width: 100%; height: auto; }}
figcaption {{ color: #666; font-size: 0.85rem; }}
pre {{ background: #f7f7f7; padding: 1rem; overflow-x: auto; font-size: 0.75rem; }}
</style>
</head>
<body>
<h1>Wine Reviews – EDA Report</h1>
{''.join(sections)}
</body>
</html>
"""
    out_path.write_text(doc, encoding="utf-8")
    return out_path
