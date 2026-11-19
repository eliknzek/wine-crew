import copy
import json
import re
from html.parser import HTMLParser

import pytest

from wine_crew.tools.eda_tools import CHART_NAMES, build_html_report, eda_summary, make_charts


@pytest.fixture
def eda_config(config):
    cfg = copy.deepcopy(config)
    cfg["eda"]["vintage_min_count"] = 5
    cfg["eda"]["chart_size"] = [6, 4]
    cfg["eda"]["chart_dpi"] = 50
    return cfg


@pytest.fixture
def charts(model_df, eda_config, tmp_path):
    return make_charts(model_df, tmp_path / "charts", eda_config)


def test_eda_summary_is_deterministic_and_json_serializable(model_df, eda_config):
    first = eda_summary(model_df, eda_config)
    second = eda_summary(model_df, eda_config)
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)
    assert json.loads(json.dumps(first)) == first


def test_eda_summary_contents(model_df, eda_config):
    s = eda_summary(model_df, eda_config)
    assert s["row_count"] == len(model_df)
    assert s["points"]["min"] >= 80 and s["points"]["max"] <= 100
    assert len(s["by_country"]) <= eda_config["eda"]["top_n"]
    assert [r["name"] for r in s["by_country"]][0] == model_df["country"].astype(str).value_counts().index[0]
    assert sum(r["count"] for r in s["price_bands"]["bands"]) == len(model_df)
    assert s["by_vintage"] and all(r["count"] >= 5 for r in s["by_vintage"])
    assert set(s["missing_pct"]) == set(model_df.columns)
    assert 0 < s["country_bias"]["top_country_share_pct"] <= 100
    assert s["corr_points_log_price"] > 0.5  # the sample is built with points rising in log price
    # Explicit fields the insights templates read.
    assert isinstance(s["overall_median_points"], float)
    assert isinstance(s["overall_median_price"], float)
    bands = s["price_bands"]
    if bands["plateau_starts_at_band"] is not None:
        row = next(b for b in bands["bands"] if b["band"] == bands["plateau_starts_at_band"])
        assert bands["plateau_from_price"] == row["low"]
        assert bands["plateau_gain_per_doubling"] == row["gain_per_price_doubling"]
    else:
        assert bands["plateau_from_price"] is None and bands["plateau_gain_per_doubling"] is None
    for r in s["value_countries"] + s["value_varieties"]:
        assert r["median_points"] >= s["overall_median_points"]
        assert r["median_price"] <= s["overall_median_price"]


def test_make_charts_creates_every_chart(charts, eda_config):
    assert list(charts) == CHART_NAMES
    for path in charts.values():
        data = path.read_bytes()
        assert data.startswith(b"\x89PNG\r\n\x1a\n")
        width, height = int.from_bytes(data[16:20], "big"), int.from_bytes(data[20:24], "big")
        assert (width, height) == (6 * 50, 4 * 50)  # fixed size from config


class _ImgCollector(HTMLParser):
    def __init__(self):
        super().__init__()
        self.srcs, self.external = [], []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "img":
            self.srcs.append(attrs.get("src", ""))
        for key in ("src", "href"):
            if attrs.get(key, "").startswith(("http://", "https://", "//")):
                self.external.append(attrs[key])


def test_html_report_opens_and_embeds_all_charts(model_df, eda_config, charts, tmp_path):
    summary = eda_summary(model_df, eda_config)
    path = build_html_report(summary, charts, tmp_path / "eda_report.html")
    text = path.read_text(encoding="utf-8")
    assert text.startswith("<!doctype html>") and text.rstrip().endswith("</html>")
    parser = _ImgCollector()
    parser.feed(text)
    assert len(parser.srcs) == len(CHART_NAMES)
    assert all(src.startswith("data:image/png;base64,") for src in parser.srcs)
    assert parser.external == []
    assert not re.search(r"<script|<link", text)
    for name in CHART_NAMES:
        assert f'id="chart-{name}"' in text


# --------------------------------------------------------------------------- chart axes


from wine_crew.tools.eda_tools import _chart_figures, _price_bands, dollar_label  # noqa: E402


def _xlabels(ax):
    ax.figure.canvas.draw()
    lo, hi = ax.get_xlim()
    return [t.get_text() for t, x in zip(ax.get_xticklabels(), ax.get_xticks()) if lo <= x <= hi]


@pytest.fixture
def figures(model_df, eda_config):
    import matplotlib.pyplot as plt

    figs = _chart_figures(model_df, eda_config)
    yield figs
    for fig in figs.values():
        plt.close(fig)


def test_dollar_label():
    assert [dollar_label(v) for v in (10, 100, 1000, 3300)] == ["$10", "$100", "$1,000", "$3,300"]


def test_price_axes_are_in_dollars(figures):
    for name in ("price_distribution_log", "points_vs_log_price"):
        ax = figures[name].axes[0]
        assert ax.get_xscale() == "log"
        labels = _xlabels(ax)
        assert labels and all(lab.startswith("$") for lab in labels), (name, labels)
    for name in ("by_country", "by_variety"):
        assert all(lab.startswith("$") for lab in _xlabels(figures[name].axes[1]))


def test_vintage_axis_has_whole_years(figures):
    labels = _xlabels(figures["points_by_vintage"].axes[0])
    assert labels and all(lab.isdigit() for lab in labels)


def test_plateau_line_on_points_vs_price(model_df, eda_config, figures):
    plateau = _price_bands(model_df, eda_config["eda"]["price_bands"],
                           eda_config["eda"]["plateau_gain_per_doubling"])["plateau_from_price"]
    ax = figures["points_vs_log_price"].axes[0]
    if plateau is None:
        assert len(ax.lines) == 0 and len(ax.texts) == 0
    else:
        assert [list(line.get_xdata()) for line in ax.lines] == [[plateau, plateau]]
        assert [t.get_text().strip() for t in ax.texts] == [f"Gains slow from {dollar_label(plateau)}"]


def test_plateau_line_appears_and_disappears(model_df, eda_config):
    import copy

    import matplotlib.pyplot as plt

    always = copy.deepcopy(eda_config)
    always["eda"]["plateau_gain_per_doubling"] = 100.0      # every band is a plateau -> first band
    never = copy.deepcopy(eda_config)
    never["eda"]["plateau_gain_per_doubling"] = -100.0      # no band is a plateau
    with_line = _chart_figures(model_df, always)
    ax = with_line["points_vs_log_price"].axes[0]
    expected = _price_bands(model_df, always["eda"]["price_bands"], 100.0)["plateau_from_price"]
    assert expected is not None and list(ax.lines[0].get_xdata()) == [expected, expected]
    without = _chart_figures(model_df, never)
    assert len(without["points_vs_log_price"].axes[0].lines) == 0
    for fig in list(with_line.values()) + list(without.values()):
        plt.close(fig)
