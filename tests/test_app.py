"""The Streamlit app loads and its prediction form works (streamlit.testing.v1.AppTest)."""

import html
import json
import re
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
    assert sum(ad.score_text(expected) in m.value for m in app.markdown) == 0   # the score is shown once, in the metric
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
    assert app.title[0].value == "ניקוד יינות ויינות ששווים את מחירם"   # no emoji: the glass is CSS
    assert app.caption[0].value.startswith("wine-crew · כל המספרים")
    source = open(APP, encoding="utf-8").read()
    assert 'page_title="wine-crew"' in source
    css = (REPO_ROOT / "app" / "style.css").read_text(encoding="utf-8")       # RTL rules live in style.css
    assert '[data-testid="stHeading"]' in css and '[data-testid="stCaptionContainer"]' in css


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
    setattr(sys, code_reload._STATE_ATTR, code_reload.current_fingerprint(REPO_ROOT / "src" / "wine_crew"))
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
    # The 7 EDA charts: each has an interactive version (by_country feeds two: the value scatter and
    # the share bars), except missing_values, which has no data in the artifacts and stays a PNG.
    assert len(ad.chart_paths(CONFIG)) == 7
    assert len(learned.get("vega_lite_chart")) == 7 and len(learned.get("image")) == 1
    assert len(learned.get("download_button")) == 1
    # was "מציאת שווים למחיר"
    assert any("value_score" in d.value.columns for d in find.dataframe)
    keys = {w.key for kind in ("number_input", "multiselect", "slider") for w in find.get(kind)}
    assert keys == {"value_price_max", "value_price_min", "value_countries", "value_varieties", "value_top"}
    # "סינון נוסף" is now a popover in the filter bar (was an expander); same two widgets inside
    more = _keyed(app, "מצא יין שווה", "value_more")
    assert more is not None and getattr(more, "type", None) == "popover"
    assert {w.key for kind in ("number_input", "slider") for w in more.get(kind)} == \
        {"value_price_min", "value_top"}
    # was "חיזוי"
    assert {w.key for w in predict.selectbox} == {"form_country", "form_province", "form_region",
                                                 "form_variety", "form_vintage"}
    assert [b.key for b in predict.button] == ["predict"]


def test_intro_above_tabs_and_one_line_per_tab(app):
    hero = list(app.main.children.values())[0]                   # the hero band comes first
    intro = list(hero.children.values())[2]                       # title, caption, then the intro line
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
    line = next(h.proto.body for h in how.get("html") if "flow-line" in h.proto.body)
    assert line == ad.flow_html(ad.flow_steps(summary))
    assert line.count('class="flow-step step-passed"') == 6
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


def _predict(app, country="US", province="California", variety="Pinot Noir", price=25.0, vintage=2014):
    app.selectbox(key="form_country").select(country).run()
    app.selectbox(key="form_province").select(province)
    app.selectbox(key="form_variety").select(variety)
    app.number_input(key="form_price").set_value(price)
    app.selectbox(key="form_vintage").select(vintage)
    app.button(key="predict").click().run()
    assert not app.exception, app.exception
    return _tab(app, "בדוק יין")


def test_price_chart_and_sentence_after_prediction(app):
    tab = _predict(app)
    assert len(tab.get("vega_lite_chart")) == 1
    assert any(m.value == "**הניקוד הצפוי ליין כזה לפי מחיר**" for m in tab.markdown)
    ranking, data = ad.load_ranking(CONFIG), ad.load_all(CONFIG)
    low, high, _ = ad.price_band(ranking, "US", "Pinot Noir")
    row = {"vintage": 2014, "country": "US", "province": "California", "variety": "Pinot Noir",
           "region_1": ad.UNKNOWN}
    curve = ad.price_curve(ad.load_model(CONFIG), data["metrics"], row, ad.price_grid(25.0, low, high))
    sentence = ad.curve_sentence(curve, 25.0, high)
    assert sentence and any(m.value == sentence for m in tab.markdown)
    assert not any(c.value == ad.RARE_PRICE_NOTE for c in tab.caption)          # $25 is common here


def test_rare_price_note_after_prediction(app):
    tab = _predict(app, price=1500.0)
    assert any(c.value == ad.RARE_PRICE_NOTE for c in tab.caption)


def test_similar_wines_after_prediction(app):
    """'אולי תאהבו גם': one card per wine from similar_wines, in the same order, with the same data."""
    tab = _predict(app)
    assert any(h.value == "אולי תאהבו גם" for h in tab.subheader)
    expected, _levels = ad.similar_wines(ad.load_ranking(CONFIG), "US", "Pinot Noir", 25.0)
    cards = _cards(app, "בדוק יין", "similar_wine_")
    assert 1 <= len(cards) == len(expected) <= 5
    for card, wine in zip(cards, expected.itertuples()):
        text = " ".join(m.value for m in card.markdown)
        assert html.escape(wine.title) in text and ad.money(wine.price) in text
        assert ad.score_text(wine.points) in text and ad.signed(wine.value_score) in text
    assert not any("title" in d.value.columns for d in tab.dataframe)     # cards, not a table


def test_wine_card_after_selecting_a_row(app):
    find = _tab(app, "מצא יין שווה")
    assert any(c.value == "בחרו שורה בטבלה כדי לראות את פרטי היין." for c in find.caption)
    app.session_state["value_table"] = {"selection": {"rows": [0], "columns": [], "cells": []}}
    app.run()
    assert not app.exception, app.exception
    ranking = ad.load_ranking(CONFIG)
    first = ad.filter_ranking(ranking, top=50).iloc[0]
    card = _tab(app, "מצא יין שווה")
    text = [m.value for m in card.markdown]
    header = next(h.proto.body for h in card.get("html") if "wine-card-title" in h.proto.body)
    assert f'<div class="wine-card-title">{html.escape(first["title"])}</div>' in header
    for label, value in ad.wine_props(first):                      # every property, label and value
        assert f"<dt>{label}</dt><dd>{html.escape(value)}</dd>" in header
    assert 'class="wine-price">' + ad.money(first["price"]) in header
    assert html.escape(ad.score_text(first["points"])) in header and ad.signed(first["value_score"]) in header
    assert f"ניקוד צפוי: {first['predicted']:.1f}" in header
    assert any(ad.value_sentence(first["value_score"]) in t for t in text)
    # The signed number next to the value sentence is isolated, in the card and in the example
    # caption under the table (otherwise the sign jumps to the wrong side in Hebrew text).
    isolated = f"ציון שווי {ad.signed_html(first['value_score'])}, כלומר {ad.value_sentence(first['value_score'])}."
    assert isolated in text
    assert any(isolated in c.value for c in card.caption)
    pct, count = ad.variety_percentile(ranking, first["variety"], first["value_score"])
    assert ad.variety_rank_sentence(pct, first["variety"], count) in text


def test_csv_download_button(app):
    buttons = _tab(app, "מצא יין שווה").get("download_button")
    assert [b.proto.label for b in buttons] == ["הורדת הרשימה (CSV)"]


def _cards(app, tab_label="מצא יין שווה", prefix="top_wine_"):
    blocks = []

    def walk(node):
        for child in getattr(node, "children", {}).values():
            key = getattr(getattr(child, "proto", None), "id", "") or ""
            if prefix in key:
                blocks.append(child)
            walk(child)

    walk(_tab(app, tab_label))
    return blocks


def test_tabs_carry_the_icons_and_the_guide_is_gone(app):
    """The four tabs get the drawn icons (CSS, in tab order); the former arch guide is removed."""
    css = next(h.proto.body for h in app.get("html") if "<style>" in h.proto.body)
    for n in range(1, 5):
        assert f'[data-testid="stTab"]:nth-child({n})::before {{ background-image: url("data:image/svg+xml,' in css
    assert not any("wine-guide" in h.proto.body or "wine-arch" in h.proto.body for h in app.get("html"))
    assert [t.label for t in app.tabs] == TABS


def test_top3_cards_match_the_first_table_rows(app):
    cards = _cards(app)
    first = ad.filter_ranking(ad.load_ranking(CONFIG), top=50).head(3)
    assert len(cards) == 3
    for card, wine in zip(cards, first.itertuples()):
        text = " ".join(m.value for m in card.markdown)
        assert html.escape(wine.title) in text
        assert ad.money(wine.price) in text and 'class="wine-price"' in text
        assert ad.score_text(wine.points) in text and ad.signed(wine.value_score) in text


def _pair_with_two_wines():
    ranking = ad.load_ranking(CONFIG)
    sizes = ranking.groupby(["country", "variety"]).size()
    country, variety = sizes[sizes == 2].index[0]
    return ranking, country, variety


def test_fewer_cards_when_the_filter_returns_fewer_wines(app):
    ranking, country, variety = _pair_with_two_wines()
    app.multiselect(key="value_countries").select(country)
    app.multiselect(key="value_varieties").select(variety).run()
    assert not app.exception, app.exception
    cards = _cards(app)
    expected = ad.filter_ranking(ranking, [country], [variety], top=50)
    assert len(cards) == len(expected) == 2
    assert all(html.escape(w) in " ".join(m.value for m in c.markdown) for c, w in zip(cards, expected["title"]))
    cheapest = float(expected["price"].min())
    app.number_input(key="value_price_max").set_value(max(4.0, cheapest - 1)).run()
    assert not app.exception, app.exception
    assert _cards(app) == []


def test_footer(app):
    footer = [h for h in app.get("html") if "<footer" in h.proto.body]
    assert len(footer) == 1
    body = footer[0].proto.body
    assert "אלי ומתנאל · פרויקט קורס" in body
    assert 'href="https://github.com/eliknzek/wine-crew"' in body
    assert "נבנה עם CrewAI ו-Streamlit" in body


def test_css_is_injected_once_from_style_css(app):
    styles = [h for h in app.get("html") if "<style>" in h.proto.body]
    assert len(styles) == 1
    css = styles[0].proto.body
    assert 'url("app/static/hero-cellar-barrels.webp")' in css and "data:image/svg+xml," in css
    assert "{{" not in css and "base64" not in css
    # Streamlit sanitizes st.html with DOMPurify, which drops a whole <style> whose text has
    # anything tag-like ("<key>" in a comment did that), so the stylesheet must have no "<".
    assert "<" not in (REPO_ROOT / "app" / "style.css").read_text(encoding="utf-8")
    source = open(APP, encoding="utf-8").read()
    assert "<style" not in source.replace('f"<style>{css}</style>"', "")   # no other CSS in the code


def _html_in(tab):
    return [h.proto.body for h in tab.get("html")]


def test_photo_strips_only_in_two_tabs(app):
    expected = {"מצא יין שווה": None, "בדוק יין": "band-check", "מה למדנו": "band-learned", "איך זה עובד": None}
    for label, band in expected.items():
        bands = [b for b in _html_in(_tab(app, label)) if "tab-band" in b]
        if band is None:
            assert bands == [], label
        else:
            assert len(bands) == 1 and band in bands[0] and 'role="img"' in bands[0], label


def test_no_results_state(app):
    find = _tab(app, "מצא יין שווה")
    assert not any("no-results" in b for b in _html_in(find))           # default: wines found
    ranking, country, variety = _pair_with_two_wines()
    cheapest = float(ad.filter_ranking(ranking, [country], [variety], top=50)["price"].min())
    app.multiselect(key="value_countries").select(country)
    app.multiselect(key="value_varieties").select(variety)
    app.number_input(key="value_price_max").set_value(max(4.0, cheapest - 1)).run()
    assert not app.exception, app.exception
    find = _tab(app, "מצא יין שווה")
    block = [b for b in _html_in(find) if "no-results" in b]
    assert len(block) == 1 and "לא נמצאו יינות" in block[0] and "no-results-img" in block[0]
    assert not any("value_score" in d.value.columns for d in find.dataframe)   # no empty table
    assert not find.get("download_button")


CHART_CARDS = {"price_curve": "vega", "countries": "vega", "varieties": "vega", "share": "vega",
               "missing": "image", "points_hist": "vega", "price_bars": "vega", "vintage": "vega"}


def _keyed(app, tab_label, key):
    found = []

    def walk(node):
        for child in getattr(node, "children", {}).values():
            if (getattr(getattr(child, "proto", None), "id", "") or "").endswith(key):
                found.append(child)
            walk(child)

    walk(_tab(app, tab_label))
    return found[0] if found else None


def _all(node, kind):
    out = []

    def walk(n):
        for child in getattr(n, "children", {}).values():
            if getattr(child, "type", None) == kind:
                out.append(child)
            walk(child)

    walk(node)
    return out


def test_learned_kpi_tiles_come_from_eda_summary(app):
    eda = ad.load_all(CONFIG)["eda_summary"]
    tiles = _keyed(app, "מה למדנו", "kpis_learned")
    assert [(m.label, m.value) for m in _all(tiles, "metric")] == \
        [(t["label"], t["value"]) for t in ad.learned_kpis(eda)]
    values = [m.value for m in _all(tiles, "metric")]
    assert values[0] == f"{eda['row_count']:,}" and values[1] == ad.money(eda["overall_median_price"])


def test_top3_share_tile_matches_insights(app):
    """One fact, one number: the tile shows the same whole percent as the bias line of insights.md."""
    data = ad.load_all(CONFIG)
    written = ad.bias_share_percent(ad.insight_sections(data["insights"])[2])
    tiles = _keyed(app, "מה למדנו", "kpis_learned")
    share = next(m.value for m in _all(tiles, "metric") if m.label == "משלוש המדינות המובילות")
    assert written is not None and share == f"{written}%"


def test_how_kpi_tiles(app):
    data = ad.load_all(CONFIG)
    tiles = _keyed(app, "איך זה עובד", "kpis_how")
    metrics = _all(tiles, "metric")
    expected = ad.how_kpis(data["summary"], data["metrics"])
    assert [(m.label, m.value) for m in metrics] == [(t["label"], t["value"]) for t in expected]
    assert len(metrics) == 6


def test_each_chart_card_has_one_chart(app):
    for key, kind in CHART_CARDS.items():
        card = _keyed(app, "מה למדנו", f"chart_{key}")
        assert card is not None, key
        vega, images = _all(card, "vega_lite_chart"), _all(card, "image")
        assert (len(vega), len(images)) == ((1, 0) if kind == "vega" else (0, 1)), key


def test_every_insight_is_still_shown(app):
    text = " ".join(m.value for m in _all(_tab(app, "מה למדנו"), "markdown"))
    for section in ad.insight_sections(ad.load_all(CONFIG)["insights"]):
        for line in section.splitlines():
            line = line.strip().lstrip("-# ").strip()
            if line:
                assert ad.localize_countries(line) in text, line


def _artifacts_copy(tmp_path, monkeypatch, drop=()):
    import shutil
    import sys

    import wine_crew.config as cfg_module

    sys.path.insert(0, str(REPO_ROOT / "app"))
    import code_reload

    setattr(sys, code_reload._STATE_ATTR, code_reload.current_fingerprint(REPO_ROOT / "src" / "wine_crew"))
    art = tmp_path / "artifacts"
    shutil.copytree(REPO_ROOT / "artifacts", art)
    for rel in drop:
        (art / rel).unlink()
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
    return at


def test_charts_fall_back_to_png_without_eda_summary(tmp_path, monkeypatch):
    at = _artifacts_copy(tmp_path, monkeypatch, drop=["analyst/eda_summary.json"])
    learned = _tab(at, "מה למדנו")
    assert any("eda_summary.json לא נמצא" in i.value for i in learned.info)
    assert _keyed(at, "מה למדנו", "kpis_learned") is None
    # every eda_summary chart is now its PNG; the score histogram still comes from value_ranking.csv
    assert len(learned.get("image")) == 6 and len(learned.get("vega_lite_chart")) == 1


def test_missing_files_show_messages_not_errors(tmp_path, monkeypatch):
    at = _artifacts_copy(tmp_path, monkeypatch, drop=["run_summary.json", "scientist/metrics.json"])
    how = _tab(at, "איך זה עובד")
    assert _keyed(at, "איך זה עובד", "kpis_how") is None
    assert len(how.warning) >= 2


EMOJI = re.compile("[\U0001F300-\U0001FAFF\u2600-\u27BF\u23E9-\u23FA\u2B50\u2B55\uFE0F]")


def _shown_text(app):
    """Every piece of text the app shows (all tabs), except image data URIs in table cells."""
    texts = []
    for kind in ("title", "header", "subheader", "markdown", "caption", "success", "info", "warning",
                 "error", "metric", "expander", "button", "download_button", "tabs"):
        for el in app.get(kind) if kind not in ("title", "markdown", "caption") else getattr(app, kind):
            for attr in ("value", "label"):
                value = getattr(el, attr, None)
                if isinstance(value, str):
                    texts.append(value)
    texts += [h.proto.body for h in app.get("html") if "<style>" not in h.proto.body]
    for frame in app.dataframe:
        for col in frame.value.columns:
            texts.append(str(col))
            texts += [str(v) for v in frame.value[col] if not str(v).startswith("data:image/")]
    return texts


def test_no_emoji_in_the_app(app):
    app.button(key="predict").click().run()          # also the prediction result and similar wines
    offending = [t for t in _shown_text(app) if EMOJI.search(t)]
    assert offending == []


def test_checks_table_shows_status_icons(app):
    how = _tab(app, "איך זה עובד")
    table = next(d for d in how.dataframe if "בדיקה" in d.value.columns)
    assert all(str(v).startswith("data:image/svg+xml;base64,") for v in table.value["עבר"])
    config = json.loads(table.proto.columns)
    assert config["עבר"]["type_config"]["type"] == "image"


def test_status_tile_has_its_icon_container(app):
    tile = _keyed(app, "איך זה עובד", "kpi_status_passed")
    assert tile is not None and _all(tile, "metric")[0].value == "עברה"


def test_filters_sit_in_one_bar(app):
    """Budget, country, variety and the "סינון נוסף" popover share one row inside one card."""
    bar = _keyed(app, "מצא יין שווה", "card_filters")
    assert bar is not None
    row = next(c for c in bar.children.values() if getattr(c, "type", None) == "flex_container")
    columns = list(row.children.values())
    assert len(columns) == 4
    assert [w.key for w in columns[0].get("number_input")] == ["value_price_max"]
    assert [w.key for w in columns[1].get("multiselect")] == ["value_countries"]
    assert [w.key for w in columns[2].get("multiselect")] == ["value_varieties"]
    assert _keyed(app, "מצא יין שווה", "value_more") is not None
