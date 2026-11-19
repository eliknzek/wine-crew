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


TABS = ["מצא יין שווה", "בדוק יין", "מה למדנו", "איך זה עובד"]


def _tab(app, label):
    return next(t for t in app.tabs if t.label == label)


def test_app_loads_with_four_tabs(app):
    assert [t.label for t in app.tabs] == TABS
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
    assert 80 <= expected <= 100
    assert metric.value == ad.score_text(expected)                      # "89: טוב מאוד"
    assert metric.proto.help == ad.GLOSSARY["ניקוד צפוי"]
    band = ad.residual_band(ad.load_ranking(CONFIG))                     # from value_ranking.csv
    assert any(m.value == ad.prediction_sentence(expected, band) for m in app.markdown)
    assert any(f"התחזית המדויקת: {expected:.1f}" in c.value for c in app.caption)
    assert any(ad.accuracy_sentence(data["metrics"]) in c.value for c in app.caption)


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
    assert ranking[:3] == ("value_score", "predicted", "label")


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


def test_model_table_shows_zero_not_negative_zero(app):
    table = next(d for d in app.dataframe if "מודל" in d.value.columns).value
    baseline = table[table["מודל"] == "baseline"].iloc[0]
    assert f"{baseline['R² (test)']:.2f}" == "0.00"


def test_title_is_hebrew_and_page_title_stays(app):
    assert app.title[0].value == "🍷 ניקוד יינות ויינות ששווים את מחירם"
    assert app.caption[0].value.startswith("wine-crew · כל המספרים")
    source = open(APP, encoding="utf-8").read()
    assert 'page_title="wine-crew"' in source
    assert '[data-testid="stHeading"]' in source and '[data-testid="stCaptionContainer"]' in source


def test_analysis_tab_shows_hebrew_country_names(app):
    assert any("אוסטריה (Austria)" in m.value for m in app.markdown)


def test_new_artifacts_load_without_clearing_the_cache(tmp_path, monkeypatch):
    """A changed run_summary.json shows up on the next run (the cache key is the file fingerprint)."""
    import json
    import shutil
    import sys
    import time

    import wine_crew.config as cfg_module

    sys.path.insert(0, str(REPO_ROOT / "app"))
    import code_reload

    # Record the current code fingerprint so the app does not evict our patched module.
    setattr(sys, code_reload._STATE_ATTR, code_reload.code_fingerprint(REPO_ROOT / "src" / "wine_crew"))
    art = tmp_path / "artifacts"
    shutil.copytree(REPO_ROOT / "artifacts", art)
    config = load_config()
    config["paths"] = {**config["paths"],
                       **{k: str(art / v.removeprefix("artifacts/")) for k, v in config["paths"].items()
                          if isinstance(v, str) and v.startswith("artifacts/")},
                       "artifacts_analyst": str(art / "analyst"), "artifacts_scientist": str(art / "scientist")}
    monkeypatch.setattr(cfg_module, "load_config", lambda path=None: config)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    at = AppTest.from_file(APP, default_timeout=60)
    at.run()
    assert not at.exception, at.exception
    old_id = json.loads((art / "run_summary.json").read_text())["run_id"]
    assert any(old_id in m.value for m in at.markdown)

    summary_path = art / "run_summary.json"
    summary = json.loads(summary_path.read_text())
    summary["run_id"] = "20990101-000000-abcdef"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False))
    later = time.time_ns() + 5_000_000_000
    import os

    os.utime(summary_path, ns=(later, later))
    at.run()
    assert not at.exception, at.exception
    assert any("20990101-000000-abcdef" in m.value for m in at.markdown)


def test_every_former_item_is_still_shown(app):
    """Everything the five old tabs showed is still shown, in the tab it moved to."""
    find, learned, how = (_tab(app, label) for label in ("מצא יין שווה", "מה למדנו", "איך זה עובד"))
    predict = _tab(app, "בדוק יין")
    data = ad.load_all(CONFIG)
    summary = data["summary"]
    # was "סקירה"
    assert any("passed" in s.value for s in how.success)
    assert any(summary["run_id"] in m.value for m in how.markdown)
    assert {"תאריך (UTC)", "משך", "עלות LLM", "בקובץ הגולמי", "אחרי ניקוי", "ב-features", "אימון",
            "בדיקה"} <= {m.label for m in how.metric}
    if summary.get("tokens"):
        assert any(c.value == ad.tokens_text(summary["tokens"]) for c in how.caption)
    assert any("בדיקה" in d.value.columns for d in how.dataframe)
    assert len(next(e for e in how.expander if e.label == "פרטים טכניים").json) == 4
    assert any(summary["raw_sha256"] in c.value for c in how.caption)
    # was "מודל"
    assert any("מודל" in d.value.columns and "CV MAE" in d.value.columns for d in how.dataframe)
    assert any(m.value.strip() == data["model_card"].strip() for m in how.markdown)
    assert any(e.label.startswith("דוח ההערכה המלא") for e in how.expander)
    # was "ניתוח"
    assert any("ניקוד חציוני" in m.value for m in learned.markdown)
    assert len(learned.get("image")) == len(ad.chart_paths(CONFIG)) == 7
    assert len(learned.get("download_button")) == 1
    # was "מציאת שווים למחיר"
    assert any("value_score" in d.value.columns for d in find.dataframe)
    keys = {w.key for kind in ("number_input", "multiselect", "slider") for w in find.get(kind)}
    assert keys == {"value_price_max", "value_price_min", "value_countries", "value_varieties", "value_top"}
    more = next(e for e in find.expander if e.label == "סינון נוסף")
    assert {w.key for kind in ("number_input", "slider") for w in more.get(kind)} == \
        {"value_price_min", "value_top"}
    # was "חיזוי"
    assert {w.key for w in predict.selectbox} == {"form_country", "form_province", "form_region",
                                                 "form_variety", "form_vintage"}
    assert [b.key for b in predict.button] == ["predict"]


def test_intro_above_tabs_and_one_line_per_tab(app):
    intro = list(app.main.children.values())[2]                  # title, caption, then the intro line
    assert all(f"**{label}**" in intro.value for label in TABS)
    for tab in app.tabs:
        assert tab.markdown[0].value.strip()                                     # first thing in each tab


def test_budget_comes_first(app):
    find = _tab(app, "מצא יין שווה")
    assert find.number_input[0].key == "value_price_max"
    assert find.number_input[0].label.startswith("תקציב")


def test_value_table_has_label_signed_score_and_help(app):
    import json

    table = next(d for d in app.dataframe if "value_score" in d.value.columns)
    assert list(table.value["label"]) == [ad.score_label(p) for p in table.value["points"]]
    assert "מה זה אומר" not in table.value.columns
    config = json.loads(table.proto.columns)
    assert config["value_score"]["type_config"]["format"] == "%+.1f"
    assert config["value_score"]["help"] == ad.GLOSSARY["ציון שווי"]
    assert config["predicted"]["help"] == ad.GLOSSARY["ניקוד צפוי"]
    first = table.value["value_score"].iloc[0]
    assert any(ad.value_sentence(first) in c.value for c in _tab(app, "מצא יין שווה").caption)


def test_flow_diagram(app):
    how = _tab(app, "איך זה עובד")
    summary = ad.load_all(CONFIG)["summary"]
    line = next(m.value for m in how.markdown if "←" in m.value)
    assert line == ad.flow_line(ad.flow_steps(summary)) and line.count("✅") == 6
    text = " ".join(m.value for m in how.markdown)
    for crew in ("crew1", "crew2"):
        for _, role in ad.agent_roles(crew):
            assert role in text
    assert any("שער 2 נבדק גם בתוך Crew 2" in c.value for c in how.caption)


def test_glossary_help_on_terms(app):
    how, find = _tab(app, "איך זה עובד"), _tab(app, "מצא יין שווה")
    learned = _tab(app, "מה למדנו")
    assert any(m.proto.help == ad.GLOSSARY["out-of-fold"] for m in find.markdown)
    assert any(ad.GLOSSARY["חציון"] in (h.proto.help or "") for h in learned.subheader)
    assert any(ad.GLOSSARY["MAE"] in (h.proto.help or "") for h in how.subheader)
