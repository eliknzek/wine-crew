"""App data helpers against the committed artifacts (no Streamlit, no LLM)."""

import math

import pandas as pd
import pytest

from wine_crew import app_data as ad
from wine_crew.config import load_config, resolve_path
from wine_crew.tools.eda_tools import CHART_NAMES

CONFIG = load_config()
pytestmark = pytest.mark.skipif(not (resolve_path(CONFIG, "artifacts_scientist") / "model.pkl").exists(),
                                reason="artifacts not present")


@pytest.fixture(scope="module")
def model():
    return ad.load_model(CONFIG)


@pytest.fixture(scope="module")
def data():
    return ad.load_all(CONFIG)


@pytest.fixture(scope="module")
def ranking():
    return ad.load_ranking(CONFIG)


def test_chart_titles_cover_eda_charts():
    assert list(ad.CHART_TITLES_HE) == CHART_NAMES
    assert list(ad.chart_paths(CONFIG)) == CHART_NAMES


def test_model_options_are_the_learned_categories(model):
    sizes = {k: len(v) for k, v in ad.model_options(model).items()}
    assert sizes == {"country": 25, "province": 95, "variety": 106, "region_1": 255}


def test_ranges(data):
    assert ad.price_range(data["contract"]) == (4.0, 3300.0)
    assert ad.price_range(None) == (4.0, 3300.0)
    assert ad.vintage_range(CONFIG) == (1980, 2017)


def test_options_by_parent_only_known_categories(model, ranking):
    opts = ad.model_options(model)
    by_country = ad.options_by_parent(ranking, opts, "country", "province")
    assert "California" in by_country["US"] and "Tuscany" not in by_country["US"]
    assert set(by_country) <= set(opts["country"])
    assert all(set(v) <= set(opts["province"]) for v in by_country.values())
    flat = ad.options_by_parent(None, opts, "country", "province")
    assert flat["US"] == opts["province"]


def test_predict_points(model, data):
    row = {"price": 25.0, "vintage": 2012, "country": "France", "province": "Bordeaux",
           "variety": "Bordeaux-style Red Blend", "region_1": ad.UNKNOWN}
    value = ad.predict_points(model, data["metrics"], row)
    assert 80 <= value <= 100
    unknown = ad.predict_points(model, data["metrics"], {**row, "vintage": None})
    assert 80 <= unknown <= 100 and not math.isnan(unknown)
    pricier = ad.predict_points(model, data["metrics"], {**row, "price": 200.0})
    assert pricier > value


def test_filter_ranking(ranking):
    out = ad.filter_ranking(ranking, ["Italy"], ["Sangiovese"], 10, 30, top=20)
    assert len(out) == 20
    assert set(out["country"]) == {"Italy"} and set(out["variety"]) == {"Sangiovese"}
    assert out["price"].between(10, 30).all()
    assert out["value_score"].is_monotonic_decreasing
    assert len(ad.filter_ranking(ranking, top=50)) == 50


def test_missing_files_are_none(tmp_path):
    assert ad.read_json(tmp_path / "x.json") is None
    assert ad.read_text(tmp_path / "x.md") is None
    cfg = {**CONFIG, "paths": {**CONFIG["paths"], "artifacts_scientist": str(tmp_path),
                               "value_ranking": str(tmp_path / "v.csv")}}
    assert ad.load_model(cfg) is None and ad.load_ranking(cfg) is None
