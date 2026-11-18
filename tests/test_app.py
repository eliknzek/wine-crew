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
    assert len(table.value) == 50 and set(table.value["country"]) == {"Italy"}
    assert (table.value["price"] <= 50).all()
