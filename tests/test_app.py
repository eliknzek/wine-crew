"""The Streamlit app loads and its prediction form works (streamlit.testing.v1.AppTest)."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from wine_crew import app_data as ad
from wine_crew.config import REPO_ROOT, load_config, resolve_path

APP = str(REPO_ROOT / "app" / "streamlit_app.py")
CONFIG = load_config()
pytestmark = pytest.mark.skipif(not resolve_path(CONFIG, "value_ranking").exists(),
                                reason="artifacts not present")


@pytest.fixture
def app(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)   # the app must not need it
    at = AppTest.from_file(APP, default_timeout=60)
    at.run()
    assert not at.exception, at.exception
    return at


def test_app_loads_with_five_tabs(app):
    assert [t.label for t in app.tabs] == ["סקירה", "ניתוח", "מודל", "מציאת שווים למחיר", "חיזוי"]
    assert any("passed" in s.value for s in app.success)
    assert len(app.dataframe) >= 3              # checks, model comparison, value wines
    assert any("ניקוד חציוני" in m.value for m in app.markdown)   # insights.md rendered


def test_overview_is_readable(app):
    assert any("20" in m.value and "מזהה ריצה" in m.value for m in app.markdown)   # full run id, not a metric
    checks = next(d for d in app.dataframe if "בדיקה" in d.value.columns)
    assert "שער 2 (features)" in set(checks.value["בדיקה"])
    details = list(checks.value["פרטים"])
    assert "הנתונים הנקיים עומדים בכל כללי החוזה: 111,537 שורות." in details
    assert not any("{" in v or "sha256:" in v for v in details)
    expander = next(e for e in app.expander if e.label == "פרטים טכניים")
    assert len(expander.json) == 4                 # the original details of each check


def test_form_defaults_are_most_common(app):
    assert app.selectbox(key="form_country").value == "US"
    assert app.selectbox(key="form_province").value == "California"
    assert app.selectbox(key="form_variety").value == "Pinot Noir"


def test_prediction_form(app):
    app.selectbox(key="form_country").select("France").run()
    assert "Bordeaux" in app.selectbox(key="form_province").options
    app.selectbox(key="form_province").select("Bordeaux")
    app.selectbox(key="form_variety").select("Bordeaux-style Red Blend")
    app.number_input(key="form_price").set_value(40.0)
    app.selectbox(key="form_vintage").select(2010)
    app.button(key="predict").click().run()
    assert not app.exception, app.exception
    metric = next(m for m in app.metric if m.label == "ניקוד צפוי")
    data = ad.load_all(CONFIG)
    expected = ad.predict_points(ad.load_model(CONFIG), data["metrics"], {
        "price": 40.0, "vintage": 2010, "country": "France", "province": "Bordeaux",
        "variety": "Bordeaux-style Red Blend", "region_1": ad.UNKNOWN})
    assert metric.value == f"{expected:.1f}"
    assert 80 <= float(metric.value) <= 100


def test_value_filter(app):
    app.multiselect(key="value_countries").select("Italy").run()
    assert not app.exception
    table = next(d for d in app.dataframe if "value_score" in d.value.columns)
    assert len(table.value) == 50 and set(table.value["country"]) == {"איטליה (Italy)"}
    assert app.number_input(key="value_price_max").value == 3300.0      # max price in the file
    assert app.multiselect(key="value_varieties").placeholder.startswith("כל הזנים")


def test_missing_vintage_shown_empty(app):
    app.slider(key="value_top").set_value(200).run()
    app.multiselect(key="value_countries").select("France").run()
    table = next(d for d in app.dataframe if "value_score" in d.value.columns)
    vintages = table.value["vintage"]
    assert "None" not in set(vintages) and "nan" not in set(vintages)
    assert all(v == "" or v.isdigit() for v in vintages)


def test_every_table_is_right_to_left(app):
    tables = {tuple(d.value.columns) for d in app.dataframe}
    checks = next(t for t in tables if "בדיקה" in t)
    models = next(t for t in tables if "מודל" in t)
    ranking = next(t for t in tables if "value_score" in t)
    assert checks == ("פרטים", "עבר", "בדיקה")
    assert models == ("R² (test)", "RMSE (test)", "MAE (test)", "CV MAE", "מודל")
    assert ranking[-1] == "rank" and ranking[-2] == "title" and ranking[0] == "value_score"


def test_ranking_sort_survives_rtl(app):
    table = next(d for d in app.dataframe if "value_score" in d.value.columns).value
    assert table["value_score"].is_monotonic_decreasing
    assert table["rank"].is_monotonic_increasing


def test_text_columns_are_right_aligned(app):
    import json

    for d in app.dataframe:
        config = json.loads(d.proto.columns)
        for col in d.value.columns:
            if ad.is_text_column(d.value[col]):
                assert config[col]["alignment"] == "right", col


def test_menus_are_sorted_and_send_english_names(app):
    countries = app.multiselect(key="value_countries").options
    hebrew_part = [c.split(" (")[0] for c in countries]
    assert hebrew_part == sorted(hebrew_part)
    assert "ישראל (Israel)" in countries
    form = app.selectbox(key="form_country")
    assert [o.split(" (")[0] for o in form.options] == sorted(o.split(" (")[0] for o in form.options)
    assert form.value == "US"                                   # the model still gets the English name
    for key in ("form_variety", "form_province"):
        options = app.selectbox(key=key).options
        assert options == sorted(options)
    regions = app.selectbox(key="form_region").options
    assert regions[0] == ad.UNKNOWN and regions[1:] == sorted(regions[1:])
    assert app.multiselect(key="value_varieties").options == sorted(app.multiselect(key="value_varieties").options)
