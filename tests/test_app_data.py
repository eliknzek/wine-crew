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


def test_check_details_text():
    msg = 'ok {"rows": 111537, "sha256": "52af2643c8ac29f010f0cc", "mae": 1.6895, "columns": ["a", "b"]}'
    assert ad.check_details_text(msg) == "שורות: 111,537 · sha256: 52af2643c8ac… · MAE: 1.69 · עמודות: a, b"
    assert ad.check_details_text("Contract check failed: x") == "Contract check failed: x"
    assert ad.check_details_text("ok not-json") == "ok not-json"


def test_most_common(model, ranking):
    opts = ad.model_options(model)
    assert ad.most_common(ranking, "country", opts["country"]) == "US"
    assert ad.most_common(ranking, "province", opts["province"], {"country": "US"}) == "California"
    assert ad.most_common(ranking, "variety", opts["variety"]) == "Pinot Noir"
    assert ad.most_common(None, "country", opts["country"]) is None


def _ok(details):
    return {"passed": True, "message": "ok " + __import__("json").dumps(details)}


SUMMARY = {"rows": {"clean": 1234, "features": 1234}, "selected": "v1", "raw_sha256": "abc"}


def test_check_sentences_use_the_summary_numbers():
    contract = {"source": {"sha256": "abc"}}
    assert ad.check_sentence("preflight", _ok({"raw_csv": "x.csv", "sha256": "abc"}), SUMMARY, contract) == \
        "קובץ הנתונים הגולמי נמצא, וטביעת האצבע שלו (sha256) זהה לזו שנרשמה בחוזה הנתונים."
    assert "שונה מזו שנרשמה" in ad.check_sentence("preflight", _ok({"sha256": "abc"}), SUMMARY, {"source": {"sha256": "zzz"}})
    assert ad.check_sentence("contract", _ok({"rows": 1234, "sha256": "abc"}), SUMMARY) == \
        "הנתונים הנקיים עומדים בכל כללי החוזה: 1,234 שורות."
    assert ad.check_sentence("features", _ok({"rows": 1234, "columns": ["a", "b", "c"]}), SUMMARY) == \
        "נבנו 3 עמודות features לכל 1,234 השורות."
    assert ad.check_sentence("model", _ok({"selected": "v1", "mae": 1.8219, "baseline_mae": 2.538}), SUMMARY) == \
        "המודל שנבחר (v1) טועה בממוצע ב-1.82 נקודות, לעומת 2.54 בניחוש הבסיס."


@pytest.mark.parametrize("name, message, expected", [
    ("preflight", "Preflight check failed: raw file not found: data/raw/x.csv", "קובץ הנתונים הגולמי לא נמצא"),
    ("contract", "Contract check failed: price has 812 nulls (allowed 0)",
     "הנתונים הנקיים לא עומדים בחוזה: price has 812 nulls (allowed 0)"),
    ("features", "Feature check failed: taster_name is a leakage column",
     "ה-features לא תקינים: taster_name is a leakage column"),
    ("model", "Model check failed: MAE 2.61 >= baseline 2.54", "המודל לא עבר את הבדיקה: MAE 2.61 >= baseline 2.54"),
])
def test_failed_check_sentence_says_what_failed(name, message, expected):
    assert ad.check_sentence(name, {"passed": False, "message": message}, SUMMARY).startswith(expected)


def test_check_sentences_on_the_real_summary(data):
    s = data["summary"]
    sentences = {n: ad.check_sentence(n, c, s, data["contract"]) for n, c in s["checks"].items()}
    assert sentences["contract"] == f"הנתונים הנקיים עומדים בכל כללי החוזה: {s['rows']['clean']:,} שורות."
    assert f"({s['selected']})" in sentences["model"]
    assert "זהה" in sentences["preflight"]
    assert not any("{" in v or "sha256:" in v for v in sentences.values())
    assert ad.check_details(s["checks"]["features"]["message"])["columns"]
    assert ad.check_details("Contract check failed: x") is None


def test_rtl_columns_reverses_column_order_only():
    df = pd.DataFrame({"מודל": ["v2", "v1"], "CV MAE": [1.7, 1.83], "R² (test)": [0.52, 0.45]}, index=[5, 3])
    out = ad.rtl_columns(df)
    assert list(out.columns) == ["R² (test)", "CV MAE", "מודל"]
    assert list(out.index) == [5, 3]                       # row order unchanged
    pd.testing.assert_frame_equal(out[df.columns], df)     # same data, same names


def test_is_text_column():
    df = pd.DataFrame({"a": ["x"], "b": [1.5], "c": [3]})
    assert [ad.is_text_column(df[c]) for c in df] == [True, False, False]


def test_every_country_in_the_data_has_a_hebrew_name(ranking, model):
    in_data = set(ranking["country"].dropna()) | set(ad.model_options(model)["country"])
    assert in_data - set(ad.COUNTRY_HE) == set()
    assert ad.country_label("Israel") == "ישראל (Israel)"
    assert ad.country_label("US") == "ארצות הברית (US)"
    assert ad.country_label("Atlantis") == "Atlantis"          # no translation: English, no error


def test_countries_sorted_by_hebrew_name():
    ordered = ad.sort_countries(["US", "Israel", "Atlantis", "France", "Argentina", "Italy"])
    assert ordered == ["Italy", "Argentina", "US", "Israel", "France", "Atlantis"]
    hebrew = [ad.COUNTRY_HE[c] for c in ad.sort_countries(list(ad.COUNTRY_HE))]
    assert hebrew == sorted(hebrew)


def test_round_for_display_has_no_negative_zero():
    assert str(ad.round_for_display(-0.0001)) == "0.0" and f"{ad.round_for_display(-0.0001):.2f}" == "0.00"
    assert ad.round_for_display(0.5183) == 0.52 and ad.round_for_display(-0.126) == -0.13


def test_theme_uses_rubik():
    import tomllib

    from wine_crew.config import REPO_ROOT

    theme = tomllib.loads((REPO_ROOT / ".streamlit" / "config.toml").read_text())["theme"]
    for key in ("font", "headingFont"):
        assert theme[key].startswith("Rubik:https://fonts.googleapis.com/css2?family=Rubik:wght@400;500;700")
    assert "codeFont" not in theme                   # code and the run id stay monospace


@pytest.mark.parametrize("text, expected", [
    ("- **Austria**: ניקוד חציוני 90, מחיר חציוני $25", "- **אוסטריה (Austria)**: ניקוד חציוני 90, מחיר חציוני $25"),
    ("45% מ-US", "45% מ-ארצות הברית (US)"),
    ("prices in USD", "prices in USD"),
    ("South Africa ו-Israel", "דרום אפריקה (South Africa) ו-ישראל (Israel)"),
    ("ישראל (Israel) כבר מתורגם", "ישראל (Israel) כבר מתורגם"),
    ("Riesling ו-Portuguese Red", "Riesling ו-Portuguese Red"),
])
def test_localize_countries(text, expected):
    assert ad.localize_countries(text) == expected


def test_localize_countries_keeps_every_number(data):
    import re as _re

    text = data["insights"]
    numbers = _re.findall(r"\d[\d,.]*", text)
    assert _re.findall(r"\d[\d,.]*", ad.localize_countries(text)) == numbers
    assert ad.localize_countries(ad.localize_countries(text)) == ad.localize_countries(text)
