"""Numbers in words, glossary and Flow steps for the app (pure functions, no artifacts, no LLM)."""

import pandas as pd
import pytest

from wine_crew import app_data as ad


@pytest.mark.parametrize("points,label", [
    (80, "סביר"), (82, "סביר"), (83, "טוב"), (86, "טוב"), (87, "טוב מאוד"), (89, "טוב מאוד"),
    (90, "מצוין"), (93, "מצוין"), (94, "יוצא דופן"), (97, "יוצא דופן"), (98, "קלאסי"), (100, "קלאסי"),
    (89.4, "טוב מאוד"), (89.5, "מצוין"), (82.49, "סביר"), (82.5, "טוב"),
    (77, "סביר"), (103, "קלאסי"),
])
def test_score_label_boundaries(points, label):
    assert ad.score_label(points) == label


def test_score_text():
    assert ad.score_text(88.6) == "89: טוב מאוד"
    assert ad.score_text(89.5) == "90: מצוין"


def test_signed_keeps_sign_with_number():
    assert ad.signed(9.081) == "\u2066+9.1\u2069" and ad.signed(-0.42) == "\u2066-0.4\u2069"


def test_signed_html_isolates_the_signed_number():
    assert ad.signed_html(8.94) == '<bdi dir="ltr">\u2066+8.9\u2069</bdi>'
    assert ad.signed_html(-0.42) == '<bdi dir="ltr">\u2066-0.4\u2069</bdi>'


def test_value_sentence():
    assert ad.value_sentence(9.081) == "9.1 נקודות מעל הניקוד שצפוי ליין כזה במחיר הזה"
    assert ad.value_sentence(-0.42) == "0.4 נקודות מתחת לניקוד שצפוי ליין כזה במחיר הזה"
    assert ad.value_sentence(0.04) == ad.value_sentence(-0.04) == "בדיוק הניקוד שצפוי ליין כזה במחיר הזה"


def _metrics(mae):
    return {"selected": "v2", "models": {"v2": {"test": {"mae": mae}}, "v1": {"test": {"mae": 9.9}}}}


def test_accuracy_sentence_comes_from_metrics():
    assert ad.model_mae(_metrics(2.34)) == 2.34
    assert ad.accuracy_sentence(_metrics(2.34)) == "המודל טועה בממוצע בכ-2.3 נקודות"
    assert ad.accuracy_sentence(_metrics(1.6895)) == "המודל טועה בממוצע בכ-1.7 נקודות"


def test_residual_band_is_10th_and_90th_percentile():
    ranking = pd.DataFrame({"points": list(range(80, 101)) * 5, "predicted": [90.0] * 105})
    low, high = ad.residual_band(ranking)
    gaps = ranking["points"] - ranking["predicted"]
    assert (low, high) == (gaps.quantile(0.1), gaps.quantile(0.9))


def test_prediction_range_rounds_inward_and_clips():
    assert ad.prediction_range(89.0, (-2.78, 2.63)) == (87, 91)    # 86.22..91.63 holds 87..91
    assert ad.prediction_range(88.6, (-2.78, 2.63)) == (86, 91)
    assert ad.prediction_range(89.0, (-2.0, 2.0)) == (87, 91)      # exact whole numbers stay in
    assert ad.prediction_range(81.0, (-3.0, 2.0)) == (80, 83)
    assert ad.prediction_range(99.0, (-2.0, 3.0)) == (97, 100)


def test_prediction_sentence():
    assert ad.prediction_sentence(89.0, (-2.78, 2.63)) == \
        "ב-8 מתוך 10 יינות דומים, הניקוד האמיתי היה בין 87 ל-91."


def test_glossary_has_every_term():
    assert set(ad.GLOSSARY) == {"ציון שווי", "ניקוד צפוי", "חציון", "out-of-fold", "MAE"}
    assert all(text.strip() for text in ad.GLOSSARY.values())
    assert "ליין כזה במחיר הזה" in ad.GLOSSARY["ציון שווי"]


def _ok(passed=True):
    return {"passed": passed, "message": "ok {}" if passed else "failed"}


def test_flow_steps_passed_run():
    summary = {"checks": {k: _ok() for k in ("preflight", "contract", "features", "model")},
               "crews": {"crew1": {}, "crew2": {}}, "failed_check": None}
    steps = ad.flow_steps(summary)
    assert [s["key"] for s in steps] == ["preflight", "crew1", "contract", "crew2", "features", "model"]
    assert {s["status"] for s in steps} == {"passed"}
    assert ad.flow_line(steps).count("✅") == 6


def test_flow_steps_gate2_failed():
    summary = {"checks": {"preflight": _ok(), "contract": _ok(), "features": _ok(False)},
               "crews": {"crew1": {}, "crew2": {}}, "failed_check": "features"}
    status = {s["key"]: s["status"] for s in ad.flow_steps(summary)}
    assert status == {"preflight": "passed", "crew1": "passed", "contract": "passed",
                      "crew2": "passed", "features": "failed", "model": "not_run"}


def test_flow_steps_crew1_crashed():
    summary = {"checks": {"preflight": _ok()}, "crews": {}, "failed_check": "crew1"}
    status = {s["key"]: s["status"] for s in ad.flow_steps(summary)}
    assert status == {"preflight": "passed", "crew1": "failed", "contract": "not_run",
                      "crew2": "not_run", "features": "not_run", "model": "not_run"}


@pytest.mark.parametrize("crew", ["crew1", "crew2"])
def test_agent_roles_come_from_agents_yaml(crew):
    roles = ad.agent_roles(crew)
    assert len(roles) == 4
    assert all(key in ad.AGENT_HE for key, _ in roles)


# --------------------------------------------------------------------------- is it worth paying more?

def _ranking(rows):
    """rows: (country, variety, price, value_score)"""
    return pd.DataFrame([{"title": f"w{i}", "country": c, "variety": v, "price": float(p), "value_score": s,
                          "points": 90, "predicted": 89.0, "vintage": 2010, "province": "P", "region_1": None,
                          "rank": i + 1} for i, (c, v, p, s) in enumerate(rows)])


def test_price_band_country_and_variety():
    ranking = _ranking([("US", "Syrah", p, 0) for p in range(10, 50)] + [("FR", "Syrah", 500, 0)] * 40)
    low, high, level = ad.price_band(ranking, "US", "Syrah")
    prices = pd.Series(range(10, 50), dtype=float)
    assert level == "country_variety" and (low, high) == (prices.quantile(0.05), prices.quantile(0.95))


def test_price_band_falls_back_to_variety_then_all():
    ranking = _ranking([("US", "Syrah", 20, 0)] * 29 + [("FR", "Syrah", 100, 0)] * 10
                       + [("FR", "Merlot", 7, 0)] * 40)
    assert ad.price_band(ranking, "US", "Syrah")[2] == "variety"          # 29 US Syrah < 30, 39 Syrah
    assert ad.price_band(ranking, "IT", "Nebbiolo")[2] == "all"           # no such wines at all
    small = _ranking([("US", "Syrah", 20, 0)] * 5 + [("FR", "Merlot", 7, 0)] * 40)
    low, high, level = ad.price_band(small, "US", "Syrah")
    assert level == "all" and low == 7.0 and high == 20.0


def test_rare_price_note():
    assert ad.is_rare_price(300, (14.0, 52.0, "country_variety"))
    assert ad.is_rare_price(10, (14.0, 52.0))
    assert not ad.is_rare_price(30, (14.0, 52.0))
    assert ad.RARE_PRICE_NOTE == "המחיר שבחרת נדיר ליין כזה בנתונים, ולכן התחזית פחות בטוחה."


def test_price_grid():
    grid = ad.price_grid(25, 17, 75)
    assert grid[0] == 17 and grid[-1] == 75 and grid == sorted(grid)
    assert 25.0 in grid and 50.0 in grid
    ratios = [b / a for a, b in zip(grid, grid[1:]) if a not in (25.0, 50.0) and b not in (25.0, 50.0)]
    assert max(ratios) / min(ratios) < 1.01                                  # log-spaced
    out = ad.price_grid(300, 14, 52)
    assert out[-1] == 300.0 and 600.0 not in out                            # chosen price always included


def _curve(points):
    return pd.DataFrame(points, columns=["price", "predicted"])


def test_curve_sentence_doubling():
    curve = _curve([(10, 85.0), (20, 87.0), (40, 89.4), (80, 91.0)])
    assert ad.curve_sentence(curve, 20, 80) == "לפי המודל, יין כזה במחיר $40 צפוי לקבל בערך 2.4 נקודות יותר."


def test_curve_sentence_flat_from_60():
    curve = _curve([(15, 85.0), (30, 88.0), (60, 90.0), (90, 90.2), (120, 90.3), (150, 90.4)])
    assert ad.curve_sentence(curve, 15, 150) == (
        "לפי המודל, יין כזה במחיר $30 צפוי לקבל בערך 3.0 נקודות יותר. "
        "לפי המודל, מעל $60 הניקוד הצפוי כמעט לא עולה (פחות מ-0.5 נקודה עד $150).")
    assert ad.curve_sentence(curve, 60, 150) == \
        "לפי המודל, מעל $60 הניקוד הצפוי כמעט לא עולה (פחות מ-0.5 נקודה עד $150)."


def test_curve_sentence_beyond_band_and_falling():
    curve = _curve([(20, 88.0), (30, 88.5), (40, 89.0), (52, 89.3)])
    assert ad.curve_sentence(curve, 30, 52) == \
        "לפי המודל, יין כזה במחיר $52 (המחיר הגבוה בטווח הנתונים) צפוי לקבל בערך 0.8 נקודות יותר."
    assert ad.curve_sentence(curve, 300, 52) is None
    falling = _curve([(10, 88.0), (20, 87.5), (40, 87.0)])                  # falling counts as flat
    assert ad.curve_sentence(falling, 10, 40).startswith("לפי המודל, מעל $10 הניקוד הצפוי כמעט לא עולה")
    same_then_up = _curve([(10, 88.0), (20, 88.02), (40, 91.0)])
    assert ad.curve_sentence(same_then_up, 10, 40) == "לפי המודל, יין כזה במחיר $20 צפוי לקבל ניקוד כמעט זהה."


# --------------------------------------------------------------------------- similar wines

def test_similar_wines_same_country_and_variety():
    ranking = _ranking([("US", "Syrah", 25, s) for s in range(8)] + [("US", "Syrah", 100, 99)])
    found, levels = ad.similar_wines(ranking, "US", "Syrah", 25)
    assert levels == ["country_variety"] and list(found["value_score"]) == [7, 6, 5, 4, 3]
    assert ad.similar_note(levels, "US", "Syrah") is None


def test_similar_wines_widen_in_order():
    ranking = _ranking([("US", "Syrah", 20, 1), ("US", "Syrah", 30, 2),     # level 1
                        ("FR", "Syrah", 25, 9),                               # level 2
                        ("US", "Merlot", 25, 5), ("US", "Merlot", 22, 4), ("US", "Merlot", 24, 3),
                        ("US", "Syrah", 60, 50), ("FR", "Merlot", 25, 80)])   # out of price / other
    found, levels = ad.similar_wines(ranking, "US", "Syrah", 25)
    assert levels == ["country_variety", "variety", "country"]
    assert list(found["value_score"]) == [2, 1, 9, 5, 4]
    assert found.index.is_unique and found["price"].between(25 / 1.5, 25 * 1.5).all()
    assert ad.similar_note(levels, "US", "Syrah") == (
        "לא נמצאו 5 יינות מאותה מדינה ואותו זן בטווח המחיר, ולכן נוספו יינות Syrah ממדינות אחרות "
        "ואחר כך יינות מזנים אחרים מארצות הברית.")


def test_similar_wines_none():
    found, levels = ad.similar_wines(_ranking([("US", "Syrah", 500, 1)]), "US", "Syrah", 20)
    assert found.empty and levels == []
    assert ad.similar_note(levels, "US", "Syrah") == "לא נמצאו יינות דומים בטווח המחיר הזה."


# --------------------------------------------------------------------------- wine card and download

def test_variety_percentile():
    ranking = _ranking([("US", "Syrah", 20, s) for s in range(10)] + [("US", "Merlot", 20, 100)])
    assert ad.variety_percentile(ranking, "Syrah", 9) == (90, 10)
    assert ad.variety_percentile(ranking, "Syrah", 0) == (0, 10)
    assert ad.variety_percentile(ranking, "Syrah", 4.5) == (50, 10)
    assert ad.variety_percentile(ranking, "Nebbiolo", 1) == (0, 0)


def test_variety_rank_sentence():
    assert ad.variety_rank_sentence(97, "Pinot Noir", 9885) == \
        "ציון השווי שלו גבוה יותר מ-97% מיינות ה-Pinot Noir (9,885 יינות)."
    assert ad.variety_rank_sentence(50, "Tannat", 12).endswith("(12 יינות) (זן עם מעט יינות בנתונים).")


def test_ranking_csv():
    import io

    ranking = _ranking([("US", "Syrah", 20, 1.234), ("Israel", "Merlot", 30, -0.456)])
    ranking.loc[1, "vintage"] = None
    data = ad.ranking_csv(ranking)
    assert data.startswith("﻿".encode("utf-8"))
    back = pd.read_csv(io.BytesIO(data), encoding="utf-8-sig", dtype=str, keep_default_na=False)
    assert list(back.columns) == list(ad.CSV_COLUMNS.values())
    assert list(back["מדינה"]) == ["ארצות הברית (US)", "ישראל (Israel)"]
    assert list(back["בציר"]) == ["2010", ""]
    assert list(back["ציון שווי"]) == ["1.23", "-0.46"]
    assert list(back["דרגה"]) == ["מצוין", "מצוין"]


# --------------------------------------------------------------------------- design

def test_short_title_cuts_at_a_word():
    long = "Au Bon Climat 2007 Los Alamos Vineyard Chardonnay (Santa Barbara County)"
    short = ad.short_title(long)
    assert short == "Au Bon Climat 2007 Los Alamos Vineyard…"
    assert len(short) <= 41 and long.startswith(short[:-1])
    assert ad.short_title("Rulo 2007 Syrah (Columbia Valley (WA))") == "Rulo 2007 Syrah (Columbia Valley (WA))"
    assert ad.short_title("A" * 50).endswith("…")          # one long word: cut inside it, still marked


def _luminance(hex_color):
    channels = [int(hex_color[i:i + 2], 16) / 255 for i in (1, 3, 5)] if isinstance(hex_color, str) \
        else [c / 255 for c in hex_color]
    lin = [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in channels]
    return 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2]


def contrast_ratio(a, b):
    hi, lo = sorted([_luminance(a), _luminance(b)], reverse=True)
    return (hi + 0.05) / (lo + 0.05)


def _theme():
    import tomllib

    from wine_crew.config import REPO_ROOT
    return tomllib.loads((REPO_ROOT / ".streamlit" / "config.toml").read_text())["theme"]


def _css_value(name):
    import re

    from wine_crew.config import REPO_ROOT
    css = (REPO_ROOT / "app" / "style.css").read_text()
    return re.search(rf"--{name}:\s*([^;]+);", css).group(1).strip()


def test_theme_text_contrast_meets_wcag_aa():
    theme = _theme()
    assert contrast_ratio(theme["textColor"], theme["backgroundColor"]) >= 4.5
    assert contrast_ratio(theme["textColor"], theme["secondaryBackgroundColor"]) >= 4.5
    assert contrast_ratio(theme["primaryColor"], theme["backgroundColor"]) >= 4.5    # prices, button text
    assert contrast_ratio("#FFFFFF", theme["primaryColor"]) >= 4.5                    # hovered button


def test_deep_wine_red_headings_meet_wcag_aa():
    """The deep red used for the large headings and the guide arches, on cream and on white cards."""
    red = _css_value("wine-red").split()[0]
    theme = _theme()
    assert red.upper() == theme["primaryColor"].upper()
    assert contrast_ratio(red, theme["backgroundColor"]) >= 4.5
    assert contrast_ratio(red, "#FFFFFF") >= 4.5


def test_wine_props_lists_every_property():
    wine = {"country": "US", "province": "California", "region_1": None, "variety": "Syrah", "vintage": 2010.0}
    assert ad.wine_props(wine) == [("מדינה", "ארצות הברית (US)"), ("מחוז", "California"),
                                   ("אזור", ad.UNKNOWN), ("זן", "Syrah"), ("בציר", "2010")]
    assert ad.wine_props({**wine, "vintage": float("nan")})[-1] == ("בציר", ad.UNKNOWN)


def _hero_brightest_pixel():
    """The brightest pixel of the hero photo actually used (the url in style.css)."""
    import re

    from PIL import Image

    from wine_crew.config import REPO_ROOT
    css = (REPO_ROOT / "app" / "style.css").read_text()
    name = re.search(r'\.st-key-hero \{[^}]*url\("app/static/([^"]+)"\)', css, re.S).group(1)
    image = Image.open(REPO_ROOT / "app" / "static" / name).convert("RGB")
    pixels = list(image.getdata())
    return max(pixels, key=_luminance)


def _overlay_on(pixel, rgb, alpha):
    return tuple(alpha * c + (1 - alpha) * p for c, p in zip(rgb, pixel))


def test_hero_overlay_is_just_dark_enough_for_white_text():
    """Worst case: the brightest pixel of the hero photo under the overlay. The CSS alpha is the
    lowest that still gives white text 4.5:1 there, plus a small margin, so the photo stays visible."""
    rgb = tuple(int(v) for v in _css_value("hero-overlay-rgb").split(","))
    alpha = float(_css_value("hero-overlay-min"))
    brightest = _hero_brightest_pixel()
    lowest = next(a / 100 for a in range(101)
                  if contrast_ratio("#FFFFFF", _overlay_on(brightest, rgb, a / 100)) >= 4.5)
    # The cellar photo has pure white pixels (the light at the end of the arch), so this equals
    # the theoretical worst case of a white pixel.
    assert brightest == (255, 255, 255) and lowest == 0.60
    assert contrast_ratio("#FFFFFF", _overlay_on(brightest, rgb, alpha)) >= 4.5
    assert lowest < alpha <= lowest + 0.05
