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
