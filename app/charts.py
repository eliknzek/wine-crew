"""Interactive altair charts for the dashboard tabs. Data comes ready from wine_crew.app_data
(read from the artifacts, never computed here); colors come from app_data.CHART_COLORS.

RTL: axis titles and legends are Hebrew only, units come from number formats ("$,.0f"),
so Hebrew and Latin text are never mixed in one string inside the SVG.
"""

from __future__ import annotations

import altair as alt
import pandas as pd

from wine_crew import app_data as ad

HEIGHT = 280


def _body_font() -> str:
    """The body font name from .streamlit/config.toml (e.g. "Rubik"), so charts follow the theme."""
    import tomllib
    from pathlib import Path

    config = Path(__file__).resolve().parent.parent / ".streamlit" / "config.toml"
    try:
        return tomllib.loads(config.read_text(encoding="utf-8"))["theme"]["font"].split(":", 1)[0]
    except (OSError, KeyError, ValueError):
        return "sans-serif"


FONT = _body_font()


def _style(chart: alt.Chart) -> alt.Chart:
    return (chart.properties(height=HEIGHT, background="#FFFFFF")
            .configure_axis(labelFont=FONT, titleFont=FONT, labelColor="#4A403A", titleColor="#4A403A",
                            gridColor="#E8DFD2")
            .configure_legend(labelFont=FONT, titleFont=FONT, orient="top", direction="horizontal")
            .configure_view(stroke=None))


def price_curve(bands: pd.DataFrame, plateau_from: float) -> alt.Chart:
    """Median points per price band (log price axis), with the plateau line from eda_summary."""
    domain = [float(bands["geo_mid"].min()) / 1.3, float(bands["geo_mid"].max()) * 1.3]
    x = alt.X("geo_mid:Q", title="מחיר", scale=alt.Scale(type="log", domain=domain, nice=False),
              axis=alt.Axis(format="$,.0f", values=[5, 10, 20, 50, 100, 200, 500, 1000]))
    y = alt.Y("median_points:Q", title="ניקוד חציוני", scale=alt.Scale(zero=False, padding=12),
              axis=alt.Axis(format="d", tickMinStep=1))
    tooltip = [alt.Tooltip("band:N", title="טווח מחיר"), alt.Tooltip("median_points:Q", title="ניקוד חציוני"),
               alt.Tooltip("count:Q", title="יינות", format=",")]
    line = alt.Chart(bands).mark_line(color=ad.CHART_COLORS["main"], point=alt.OverlayMarkDef(
        color=ad.CHART_COLORS["main"], size=50)).encode(x=x, y=y, tooltip=tooltip)
    rule_df = pd.DataFrame({"geo_mid": [plateau_from], "text": ["מכאן העלייה מאטה"]})
    rule = alt.Chart(rule_df).mark_rule(color=ad.CHART_COLORS["reference"], strokeDash=[6, 4],
                                        strokeWidth=2).encode(x="geo_mid:Q")
    label = alt.Chart(rule_df).mark_text(align="right", dx=-6, color="#4A403A", font=FONT).encode(
        x="geo_mid:Q", y=alt.value(HEIGHT - 40), text="text:N")
    return _style(alt.layer(line, rule, label))


def value_scatter(groups: pd.DataFrame, median_price: float, median_points: float, kind: str) -> alt.Chart:
    """Median price vs median points per country/variety; the value list in deep red, the rest
    muted; gold reference lines at the overall medians."""
    name_title = "מדינה" if kind == "country" else "זן"
    scale = alt.Scale(domain=["שווים", "אחרים"], range=[ad.CHART_COLORS["main"], ad.CHART_COLORS["other"]])
    color = alt.Color("group:N", title=None, scale=scale)
    tooltip = [alt.Tooltip("label:N", title=name_title)]
    if kind == "country":
        tooltip.append(alt.Tooltip("name:N", title="Country"))
    tooltip += [alt.Tooltip("median_points:Q", title="ניקוד חציוני"),
                alt.Tooltip("median_price:Q", title="מחיר חציוני", format="$,.0f"),
                alt.Tooltip("count:Q", title="יינות", format=",")]
    x = alt.X("median_price:Q", title="מחיר חציוני", axis=alt.Axis(format="$,.0f"),
              scale=alt.Scale(zero=False, nice=True))
    y = alt.Y("median_points:Q", title="ניקוד חציוני", scale=alt.Scale(zero=False, nice=True, padding=14),
              axis=alt.Axis(format="d", tickMinStep=1))
    points = alt.Chart(groups).mark_circle(size=110, opacity=0.95).encode(x=x, y=y, color=color, tooltip=tooltip)
    # Labels only for the value list (the rest: tooltip), one label per position, so names
    # that share a point (e.g. two countries at $25 / 89) are joined instead of overlapping.
    value = groups[groups["group"] == "שווים"]
    merged = (value.groupby(["median_price", "median_points"], as_index=False)["label"]
              .agg(lambda names: ", ".join(names)))
    labels = alt.Chart(merged).mark_text(dy=-12, fontSize=11, font=FONT, fontWeight="bold",
                                         color=ad.CHART_COLORS["main"]).encode(x=x, y=y, text="label:N")
    ref_x = alt.Chart(pd.DataFrame({"v": [median_price]})).mark_rule(
        color=ad.CHART_COLORS["reference"], strokeDash=[4, 4]).encode(x="v:Q")
    ref_y = alt.Chart(pd.DataFrame({"v": [median_points]})).mark_rule(
        color=ad.CHART_COLORS["reference"], strokeDash=[4, 4]).encode(y="v:Q")
    return _style(alt.layer(ref_x, ref_y, points, labels))


def share_bars(groups: pd.DataFrame) -> alt.Chart:
    """Share of wines per country (horizontal bars, largest on top)."""
    tooltip = [alt.Tooltip("label:N", title="מדינה"), alt.Tooltip("name:N", title="Country"),
               alt.Tooltip("share_pct:Q", title="אחוז מהיינות", format=".1f"),
               alt.Tooltip("count:Q", title="יינות", format=",")]
    return _style(alt.Chart(groups).mark_bar(color=ad.CHART_COLORS["main"]).encode(
        x=alt.X("share_pct:Q", title="אחוז מהיינות", axis=alt.Axis(format=".0f")),
        y=alt.Y("label:N", title=None, sort="-x", axis=alt.Axis(labelOverlap=False)),
        tooltip=tooltip))


def points_hist(hist: pd.DataFrame) -> alt.Chart:
    """Number of wines per score, colored by the WineEnthusiast band (legend)."""
    color = alt.Color("label:N", title=None, sort=list(ad.SCORE_COLORS),
                      scale=alt.Scale(domain=list(ad.SCORE_COLORS), range=list(ad.SCORE_COLORS.values())),
                      # one column, top to bottom from "סביר" to "קלאסי": no left/right reading order
                      legend=alt.Legend(orient="right", direction="vertical", columns=1, symbolType="square"))
    return _style(alt.Chart(hist).mark_bar().encode(
        x=alt.X("points:O", title="ניקוד", axis=alt.Axis(labelAngle=0)),
        y=alt.Y("count:Q", title="יינות", axis=alt.Axis(format=",.0f")),
        color=color,
        tooltip=[alt.Tooltip("points:O", title="ניקוד"), alt.Tooltip("label:N", title="דרגה"),
                 alt.Tooltip("count:Q", title="יינות", format=",")]))


def price_bars(bands: pd.DataFrame) -> alt.Chart:
    """Number of wines per price band, horizontal so every band label fits (order from eda_summary)."""
    return _style(alt.Chart(bands).mark_bar(color=ad.CHART_COLORS["main"], height={"band": 0.75}).encode(
        y=alt.Y("band:N", title=None, sort=list(bands["band"]), axis=alt.Axis(labelOverlap=False, labelPadding=4)),
        x=alt.X("count:Q", title="יינות", axis=alt.Axis(format=",.0f", tickCount=4)),
        tooltip=[alt.Tooltip("band:N", title="טווח מחיר"), alt.Tooltip("count:Q", title="יינות", format=",")])
        ).properties(height=HEIGHT + 60)   # 13 band labels need more room than the other charts


def vintage_line(vintages: pd.DataFrame) -> alt.Chart:
    """Mean points per vintage (only years with enough wines, as in eda_summary)."""
    return _style(alt.Chart(vintages).mark_line(color=ad.CHART_COLORS["main"], point=alt.OverlayMarkDef(
        color=ad.CHART_COLORS["main"], size=40)).encode(
        x=alt.X("vintage:Q", title="בציר", scale=alt.Scale(zero=False, nice=False),
                axis=alt.Axis(format="d", tickCount=5)),
        y=alt.Y("mean_points:Q", title="ניקוד ממוצע", scale=alt.Scale(zero=False, padding=10)),
        tooltip=[alt.Tooltip("vintage:Q", title="בציר", format="d"),
                 alt.Tooltip("mean_points:Q", title="ניקוד ממוצע", format=".2f"),
                 alt.Tooltip("count:Q", title="יינות", format=",")]))
