"""Streamlit app for wine-crew, in Hebrew (RTL).

It only reads artifacts/ and model.pkl: it never runs the Flow and needs no API
key. Run with:
    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import html
import sys
from pathlib import Path

# Make wine_crew importable without installing the package (also needed to
# unpickle model.pkl, whose pipeline uses wine_crew.tools.model_tools).
APP_DIR = Path(__file__).resolve().parent
SRC = APP_DIR.parent / "src"
for folder in (SRC, APP_DIR):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))

import altair as alt  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from code_reload import reload_if_changed  # noqa: E402


def _clear_caches() -> None:
    st.cache_data.clear()
    st.cache_resource.clear()


# After an update, load the new src/wine_crew code (Streamlit does not reload it itself).
reload_if_changed(SRC / "wine_crew", clear_caches=_clear_caches)   # also app/charts.py

from wine_crew import app_data as ad  # noqa: E402
import charts  # noqa: E402  (app/charts.py: altair charts for the dashboard tabs)
from wine_crew.config import load_config, resolve_path  # noqa: E402

# Favicon: a line wine glass drawn for the app (PNG: page_icon does not take SVG). No emoji.
st.set_page_config(page_title="wine-crew", page_icon=str(APP_DIR / "assets" / "favicon.png"), layout="wide")

STYLE_FILE = APP_DIR / "style.css"
ASSETS = APP_DIR / "assets"


@st.cache_resource
def load_style(css_fingerprint: tuple) -> str:  # noqa: ARG001 - cache key only
    """app/style.css with each {{SVG:name}} replaced by app/assets/<name>.svg as a data URI.
    The SVGs are small line drawings; the hero photo is served as a static file (app/static/),
    not inlined, so the CSS stays small on every rerun."""
    import re
    from urllib.parse import quote

    def svg(match):
        text = (ASSETS / f"{match.group(1)}.svg").read_text(encoding="utf-8")
        return "data:image/svg+xml," + quote(text)

    css = re.sub(r"\{\{SVG:([a-z0-9-]+)\}\}", svg, STYLE_FILE.read_text(encoding="utf-8"))
    return f"<style>{css}</style>"


# The only CSS injection in the app.
st.html(load_style(ad.file_fingerprint([STYLE_FILE, *sorted(ASSETS.glob("*.svg"))])))

TABS = ["מצא יין שווה", "בדוק יין", "מה למדנו", "איך זה עובד"]
INTRO = ("האתר מנבא את הניקוד של יין לפי המחיר, המקור, הזן והבציר, ומוצא יינות שקיבלו ניקוד גבוה "
         "ממה שצפוי ליין כזה במחיר שלו. ב**מצא יין שווה** מחפשים לפי תקציב, ב**בדוק יין** מקבלים "
         "ניקוד צפוי ליין מסוים, ב**מה למדנו** מוצגים הממצאים מהנתונים, וב**איך זה עובד** כל הפרטים הטכניים.")
TAB_INTRO = {
    "value": "בחרו תקציב, ואם תרצו גם מדינה וזן, ותקבלו את היינות שקיבלו את הניקוד הגבוה ביותר "
             "ביחס למה שצפוי ליין כזה במחיר שלו.",
    "predict": "בחרו מקור, זן, מחיר ובציר, וקבלו את הניקוד שהמודל צופה ליין כזה.",
    "learned": "מה למדנו מביקורות היין עצמן: הממצאים המרכזיים והגרפים שמהם הם עולים.",
    "how": "איך הנתונים עוברים מהקובץ הגולמי למודל, מה נבדק בדרך, ועד כמה המודל מדויק.",
}
DIVIDER = '<div class="wine-divider" role="presentation"></div>'
FOOTER = ('<footer class="wine-footer">אלי ומתנאל · פרויקט קורס · '
          '<a href="https://github.com/eliknzek/wine-crew" target="_blank" rel="noopener">'
          'github.com/eliknzek/wine-crew</a> · נבנה עם CrewAI ו-Streamlit</footer>')
TITLE_HELP = "שמות ארוכים מקוצרים בטבלה. השם המלא בכרטיס היין (בחרו שורה) ובקובץ ה-CSV."
# Decorative photo strips at the top of two tabs (no text on them; images in app/static/, set in
# style.css). Only "בדוק יין" and "מה למדנו": the first tab is right under the hero, and the
# technical tab does not need decoration.
BANDS = {
    "predict": '<div class="tab-band band-check" role="img" aria-label="יין אדום נמזג לכוס"></div>',
    "learned": '<div class="tab-band band-learned" role="img" aria-label="אשכול ענבים על הגפן"></div>',
}
NO_RESULTS = ('<div class="no-results"><div class="no-results-text">'
              '<div class="no-results-title">לא נמצאו יינות</div>'
              '<p>נסו להגדיל את התקציב, או להסיר מדינה או זן.</p></div>'
              '<div class="no-results-img" role="img" aria-label="כוס יין על חבית בכרם"></div></div>')
CHART_TITLE = "הניקוד הצפוי ליין כזה לפי מחיר"   # shown as text above the chart (RTL-safe)
CREW_TITLES = {"crew1": "Crew 1: Data Analyst", "crew2": "Crew 2: Data Scientist"}
CHECK_NAMES = {"preflight": "בדיקה מקדימה (קובץ גולמי)", "contract": "שער 1 (חוזה נתונים)",
               "features": "שער 2 (features)", "model": "שער 3 (מודל)"}


# Each loader takes the fingerprint (mtime, size) of the files it reads as its cache key,
# so a new artifact (for example after a git pull) is loaded on the next rerun. The
# argument must not start with "_": Streamlit leaves such arguments out of the key.
@st.cache_data(max_entries=2)
def _data(fingerprint: tuple):  # noqa: ARG001 - cache key only
    return ad.load_all(load_config())


@st.cache_resource(max_entries=2)
def _model(fingerprint: tuple):  # noqa: ARG001 - cache key only
    return ad.load_model(load_config())


@st.cache_data(max_entries=2)
def _ranking(fingerprint: tuple):  # noqa: ARG001 - cache key only
    return ad.load_ranking(load_config())


@st.cache_data(max_entries=2)
def _band(fingerprint: tuple):  # noqa: ARG001 - cache key only
    """10th-90th percentile of out-of-fold gaps (points - predicted), computed once per file."""
    ranking = ad.load_ranking(load_config())
    return None if ranking is None else ad.residual_band(ranking)


@st.cache_data(max_entries=64)
def _curve(model_fingerprint: tuple, row: tuple, prices: tuple, _model, _metrics):  # noqa: ARG001
    """Predicted points on the price grid, once per input (the key: model file + the form values)."""
    return ad.price_curve(_model, _metrics, dict(row), prices)


def price_chart(curve: pd.DataFrame, chosen: float, chosen_points: float):
    """Predicted points by price (log scale, $ on the axis), with a dot on the chosen price."""
    domain = [float(curve["price"].min()), float(curve["price"].max())]   # the band only, no rounding out
    x = alt.X("price:Q", title="מחיר", scale=alt.Scale(type="log", domain=domain, nice=False),
              axis=alt.Axis(format="$,.0f"))
    y = alt.Y("predicted:Q", title="ניקוד צפוי", scale=alt.Scale(zero=False))
    tooltip = [alt.Tooltip("price:Q", title="מחיר", format="$,.0f"),
               alt.Tooltip("predicted:Q", title="ניקוד צפוי", format=".1f")]
    line = alt.Chart(curve).mark_line().encode(x=x, y=y, tooltip=tooltip)
    dot = alt.Chart(pd.DataFrame({"price": [chosen], "predicted": [chosen_points]})).mark_point(
        size=120, filled=True).encode(x=x, y=y, tooltip=tooltip)
    return (line + dot).properties(height=260)


def show_table_rtl(df: pd.DataFrame, column_config: dict | None = None, **kwargs):
    """The one way to show a table: columns reversed (first column on the right), index hidden,
    text columns aligned right. Data, rounding, row order and column names are unchanged."""
    config = dict(column_config or {})
    for col in df.columns:
        if not ad.is_text_column(df[col]):
            continue
        current = config.get(col)
        if current is None or isinstance(current, str):
            config[col] = st.column_config.TextColumn(current, alignment="right")
        elif isinstance(current, dict):
            config[col] = {**current, "alignment": "right"}
    return st.dataframe(ad.rtl_columns(df), hide_index=True, width="stretch", column_config=config,
                        **kwargs)


def kpi_row(tiles: list[dict], key: str) -> None:
    """A row of KPI tiles (st.metric with a border); 2 per row on narrow screens (style.css)."""
    with st.container(key=key):
        for col, tile in zip(st.columns(len(tiles)), tiles):
            # a tile with a "key" (the run status) gets its own container, so CSS can add its icon
            with col, (st.container(key=f"kpi_{tile['key']}") if tile.get("key") else st.container()):
                st.metric(tile["label"], tile["value"], delta=tile.get("delta"),
                          delta_color=tile.get("delta_color", "normal"), help=tile.get("help"), border=True)


@st.cache_resource
def _status_uris() -> dict[str, str]:
    """The status icons (app/assets/status-*.svg) as data URIs, for image cells in tables."""
    import base64

    return {status: "data:image/svg+xml;base64," + base64.b64encode(
        (ASSETS / f"status-{status.replace('_', '-')}.svg").read_bytes()).decode()
        for status in ad.STATUS_HE}


def status_icon(status: str) -> str:
    return _status_uris()[status]


def wine_cards(wines: pd.DataFrame, key_prefix: str) -> None:
    """Small cards side by side (one per column; stacked on narrow screens): full name, price in
    wine red, score with its word and the signed value score. Data as given, no new computation."""
    for col, (i, wine) in zip(st.columns(len(wines)), enumerate(wines.itertuples())):
        with col, st.container(border=True, key=f"{key_prefix}_{i}"):
            st.markdown(f'<span class="wine-name">{html.escape(wine.title)}</span>', unsafe_allow_html=True)
            st.markdown(f'<span class="wine-price">{ad.money(wine.price)}</span>', unsafe_allow_html=True)
            st.markdown(f"{ad.score_text(wine.points)} · ציון שווי {ad.signed_html(wine.value_score)}",
                        unsafe_allow_html=True)


def wine_card_html(wine) -> str:
    """The product-style header of the wine card: big name, two-column properties, highlight row."""
    props = "".join(f"<div><dt>{html.escape(label)}</dt><dd>{html.escape(value)}</dd></div>"
                    for label, value in ad.wine_props(wine))
    return (f'<div class="wine-card-title">{html.escape(wine["title"])}</div>'
            f'<dl class="wine-props">{props}</dl>'
            f'<div class="wine-highlight"><span class="wine-price">{ad.money(wine["price"])}</span>'
            f'<span>ניקוד בפועל: <strong>{html.escape(ad.score_text(wine["points"]))}</strong></span>'
            f'<span>ניקוד צפוי: {wine["predicted"]:.1f}</span>'
            f'<span>ציון שווי <strong>{ad.signed_html(wine["value_score"])}</strong></span></div>')


def _missing(name: str) -> None:
    st.warning(f"הקובץ {name} לא נמצא. הריצו `python -m wine_crew.main` כדי לייצר את התוצרים.")


config = load_config()
data = _data(ad.file_fingerprint(ad.data_files(config).values()))

ranking_fp = ad.file_fingerprint([resolve_path(config, "value_ranking")])
ranking = _ranking(ranking_fp)
band = _band(ranking_fp)
metrics = data["metrics"]

with st.container(key="hero"):
    st.title("ניקוד יינות ויינות ששווים את מחירם")   # the glass icon is added in style.css
    st.caption("wine-crew · כל המספרים מגיעים מתוצרי הריצה האחרונה של ה-Flow. האפליקציה לא מריצה מודלי שפה.")
    st.markdown(INTRO)
    st.html(DIVIDER)

value_tab, predict_tab, learned_tab, how_tab = st.tabs(TABS)

# --------------------------------------------------------------------------- find a value wine
with value_tab:
    st.markdown(TAB_INTRO["value"])
    if ranking is None:
        _missing("artifacts/scientist/value_ranking.csv")
    else:
        lo, hi = ad.price_range(data["contract"])
        file_max = min(hi, float(ranking["price"].max()))
        price_max = st.number_input("תקציב: מחיר מקסימלי לבקבוק ($)", lo, hi, file_max, step=1.0,
                                    key="value_price_max")
        countries = ad.sort_countries(ranking["country"].dropna().unique().tolist())
        varieties = sorted(ranking["variety"].dropna().unique().tolist())
        f1, f2 = st.columns(2)
        chosen_countries = f1.multiselect("מדינה", countries, key="value_countries",
                                          format_func=ad.country_label,
                                          placeholder="כל המדינות (אפשר לבחור כמה)")
        chosen_varieties = f2.multiselect("זן", varieties, key="value_varieties",
                                          placeholder="כל הזנים (אפשר לבחור כמה)")
        with st.expander("סינון נוסף"):
            p1, p2 = st.columns(2)
            price_min = p1.number_input("מחיר מינימלי ($)", lo, hi, lo, step=1.0, key="value_price_min")
            top = p2.slider("מספר יינות", 10, 200, 50, step=10, key="value_top")
        result = ad.filter_ranking(ranking, chosen_countries, chosen_varieties, price_min, price_max, top)
        st.markdown("ציון שווי = הניקוד בפועל פחות הניקוד שהמודל צפה ליין כזה (מחיר, מקור, זן ובציר). "
                    "התחזית לכל יין מגיעה ממודל שלא ראה אותו באימון (out-of-fold).",
                    help=ad.GLOSSARY["out-of-fold"])
        if metrics is not None:
            st.caption(f"{len(result)} יינות מוצגים · {ad.accuracy_sentence(metrics)}.")
        else:
            st.caption(f"{len(result)} יינות מוצגים")
        top3 = result.head(3)   # the same rows as the top of the table below, no new computation
        if len(top3):
            with st.container(key="top3"):
                st.subheader("3 היינות השווים ביותר")
                wine_cards(top3, "top_wine")
        if not len(result):
            # Display only: the filter and the result are unchanged; no empty table.
            st.html(NO_RESULTS)
        else:
            shown = result.assign(vintage=result["vintage"].map(lambda v: "" if pd.isna(v) else str(int(v))),
                                  country=result["country"].map(ad.country_label),
                                  label=result["points"].map(ad.score_label),
                                  title=result["title"].map(ad.short_title))
            event = show_table_rtl(
                shown[["rank", "title", "country", "province", "variety", "vintage", "price", "points",
                       "label", "predicted", "value_score"]],
                key="value_table", on_select="rerun", selection_mode="single-row",
                column_config={
                    "rank": "דירוג כללי", "title": st.column_config.TextColumn("יין", help=TITLE_HELP),
                    "country": "מדינה", "province": "מחוז",
                    # 150px: auto width cut long names such as "Cabernet Sauvignon" inside 1200px
                    "variety": st.column_config.TextColumn("זן", width=150),
                    "vintage": st.column_config.TextColumn("בציר"),
                    "price": st.column_config.NumberColumn("מחיר", format="$%.0f"),
                    "points": "ניקוד", "label": "דרגה",
                    "predicted": st.column_config.NumberColumn("ניקוד צפוי", format="%.1f",
                                                               help=ad.GLOSSARY["ניקוד צפוי"]),
                    "value_score": st.column_config.NumberColumn("ציון שווי", format="%+.1f",
                                                                 help=ad.GLOSSARY["ציון שווי"]),
                })
            first = result.iloc[0]
            st.caption(f"לדוגמה, ליין הראשון ציון שווי {ad.signed_html(first['value_score'])}, כלומר "
                       f"{ad.value_sentence(first['value_score'])}.", unsafe_allow_html=True)
            st.download_button("הורדת הרשימה (CSV)", ad.ranking_csv(result), file_name="value_wines.csv",
                               mime="text/csv", key="value_csv")
            rows = event.selection.rows if event is not None else []
            if not rows:
                st.caption("בחרו שורה בטבלה כדי לראות את פרטי היין.")
            else:
                wine = result.iloc[rows[0]]
                pct, count = ad.variety_percentile(ranking, wine["variety"], wine["value_score"])
                with st.container(border=True, key="wine_card"):
                    st.html(wine_card_html(wine))
                    st.markdown(f"ציון שווי {ad.signed_html(wine['value_score'])}, כלומר "
                                f"{ad.value_sentence(wine['value_score'])}.", unsafe_allow_html=True)
                    st.markdown(ad.variety_rank_sentence(pct, wine["variety"], count))

# --------------------------------------------------------------------------- check a wine
with predict_tab:
    st.html(BANDS["predict"])
    st.markdown(TAB_INTRO["predict"])
    model = _model(ad.file_fingerprint([ad.model_file(config)]))
    if model is None or metrics is None:
        _missing("artifacts/scientist/model.pkl")
    else:
        st.caption("האפשרויות הן הקטגוריות שהמודל מכיר; קטגוריות נדירות מטופלות כ-\"Other\".")
        opts = ad.model_options(model)
        provinces = ad.options_by_parent(ranking, opts, "country", "province")
        regions = ad.options_by_parent(ranking, opts, "province", "region_1")
        lo, hi = ad.price_range(data["contract"])
        vmin, vmax = ad.vintage_range(config)
        with st.container(border=True):
            def _index(options, column, where=None):
                # Default to the most common option in the data, not the first alphabetically.
                top = ad.most_common(ranking, column, options, where)
                return options.index(top) if top in options else 0

            c1, c2 = st.columns(2)
            country_opts = ad.sort_countries(opts["country"])
            country = c1.selectbox("מדינה", country_opts, index=_index(country_opts, "country"),
                                   format_func=ad.country_label, key="form_country")
            province_opts = provinces.get(country) or sorted(opts["province"])
            province = c2.selectbox("מחוז", province_opts,
                                    index=_index(province_opts, "province", {"country": country}),
                                    key="form_province")
            region_opts = [ad.UNKNOWN] + sorted(regions.get(province, []))  # "לא ידוע" stays first
            region = c1.selectbox("אזור (region_1)", region_opts, key="form_region")
            variety_opts = sorted(opts["variety"])
            variety = c2.selectbox("זן", variety_opts, index=_index(variety_opts, "variety"),
                                   key="form_variety")
            price = c1.number_input("מחיר ($)", lo, hi, 25.0, step=1.0, key="form_price")
            vintage = c2.selectbox("בציר", [ad.UNKNOWN] + list(range(vmax, vmin - 1, -1)), index=4,
                                   key="form_vintage")
            submitted = st.button("חזה ניקוד", type="primary", key="predict")
        if submitted:
            row = {"price": price, "vintage": None if vintage == ad.UNKNOWN else vintage,
                   "country": country, "province": province, "variety": variety, "region_1": region}
            points = ad.predict_points(model, metrics, row)
            st.metric("ניקוד צפוי", ad.score_text(points), help=ad.GLOSSARY["ניקוד צפוי"])
            if band is not None:
                st.markdown(ad.prediction_sentence(points, band))
            st.caption(f"התחזית המדויקת: {points:.1f} · {ad.accuracy_sentence(metrics)} "
                       f"(המודל {metrics['selected']}).")

            if ranking is not None:
                st.subheader("האם שווה לשלם יותר?")
                low, high, _level = ad.price_band(ranking, country, variety)
                grid = ad.price_grid(price, low, high)
                curve = _curve(ad.file_fingerprint([ad.model_file(config)]),
                               tuple(sorted((k, v) for k, v in row.items() if k != "price")),
                               tuple(grid), model, metrics)
                st.markdown(f"**{CHART_TITLE}**")
                st.altair_chart(price_chart(curve, price, points), width="stretch", key="price_chart")
                sentence = ad.curve_sentence(curve, price, high)
                if sentence:
                    st.markdown(sentence)
                if ad.is_rare_price(price, (low, high)):
                    st.caption(ad.RARE_PRICE_NOTE)

                st.subheader("אולי תאהבו גם")
                similar, levels = ad.similar_wines(ranking, country, variety, price)
                st.caption(f"עד 5 יינות במחיר בין {ad.money(price / 1.5)} ל-{ad.money(price * 1.5)}, "
                           "עם ציון השווי הגבוה ביותר.")
                if len(similar):
                    with st.container(key="similar_cards"):
                        wine_cards(similar, "similar_wine")
                note = ad.similar_note(levels, country, variety)
                if note:
                    st.caption(note)

# --------------------------------------------------------------------------- what we learned
with learned_tab:
    st.html(BANDS["learned"])
    st.markdown(TAB_INTRO["learned"])
    eda = data["eda_summary"]
    if eda is not None:
        kpi_row(ad.learned_kpis(eda), "kpis_learned")
    else:
        st.info("הקובץ artifacts/analyst/eda_summary.json לא נמצא, ולכן אין אריחי מדדים, "
                "והגרפים מוצגים כתמונות.")

    sections = ad.insight_sections(data["insights"])
    if data["insights"] is None:
        _missing("artifacts/analyst/insights.md")
    elif sections is None:   # headings changed: show the whole text as before
        st.markdown(ad.localize_countries(data["insights"]))
    pngs = ad.chart_paths(config)

    def chart_card(key: str, title: str, chart, png: str, text: str | None = None,
                   beside: bool = False) -> None:
        """A card with an interactive chart (or the PNG when the artifacts have no data for it,
        or a message when neither exists) and, if given, the finding it supports."""
        with st.container(border=True, key=f"chart_{key}"):
            st.markdown(f'<div class="card-title">{html.escape(title)}</div>', unsafe_allow_html=True)
            # Text first (on the right, where a Hebrew reader starts), then the chart, with a wide
            # gap: the chart's Y-axis title then sits on the far side, away from the bullets.
            text_col, graph_col = st.columns([2, 3], gap="large") if (beside and text) else (None, st.container())
            with graph_col:
                if chart is not None:
                    st.altair_chart(chart, width="stretch", key=f"altair_{key}")
                elif png in pngs:
                    st.image(str(pngs[png]), width="stretch")   # the card title already names it
                else:
                    st.info("אין נתונים לגרף הזה בתוצרים.")
            if text:
                with (text_col or st.container()):
                    st.markdown(ad.localize_countries(text))

    s1, s2, s3, s4 = sections or ("", "", "", "")
    st.subheader("הממצאים והגרפים", help=f"חציון: {ad.GLOSSARY['חציון']}")
    # 1. price and the plateau
    chart_card("price_curve", s1.split("\n", 1)[0].removeprefix("## ") or "ניקוד לפי מחיר",
               charts.price_curve(ad.price_band_data(eda), eda["price_bands"]["plateau_from_price"])
               if eda else None, "points_vs_log_price", ad.section_body(s1) or None, beside=True)
    # 2. value countries and varieties
    if s2:
        st.markdown(f'<div class="section-title">{html.escape(s2.split(chr(10), 1)[0].removeprefix("## "))}</div>',
                    unsafe_allow_html=True)
        st.markdown(ad.section_body(s2).split("\n###", 1)[0])
    with st.container(key="dash_row_value"):
        c1, c2 = st.columns(2)
        with c1:
            chart_card("countries", "מדינות: מחיר חציוני מול ניקוד חציוני",
                       charts.value_scatter(ad.group_chart_data(eda["by_country"], eda["value_countries"], True),
                                            eda["overall_median_price"], eda["overall_median_points"], "country")
                       if eda else None, "by_country", ad.subsection(s2, "### מדינות") or None)
        with c2:
            chart_card("varieties", "זנים: מחיר חציוני מול ניקוד חציוני",
                       charts.value_scatter(ad.group_chart_data(eda["by_variety"], eda["value_varieties"]),
                                            eda["overall_median_price"], eda["overall_median_points"], "variety")
                       if eda else None, "by_variety", ad.subsection(s2, "### זנים") or None)
    # 3. bias and missing values
    bias_lines = [line for line in ad.section_body(s3).splitlines() if line.strip()]
    missing_text = "\n".join(line for line in bias_lines if "ערכים חסרים" in line)
    bias_text = "\n".join(line for line in bias_lines if "ערכים חסרים" not in line)
    if s3:
        st.markdown(f'<div class="section-title">{html.escape(s3.split(chr(10), 1)[0].removeprefix("## "))}</div>',
                    unsafe_allow_html=True)
    with st.container(key="dash_row_bias"):
        c1, c2 = st.columns(2)
        with c1:
            chart_card("share", "חלק היינות לפי מדינה",
                       charts.share_bars(ad.group_chart_data(eda["by_country"], [], True)) if eda else None,
                       "", bias_text or None)   # by_country.png is already in the countries card
        with c2:
            chart_card("missing", "ערכים חסרים לפי עמודה", None, "missing_values", missing_text or None)
    # 4. distributions
    with st.container(key="dash_row_dist"):
        c1, c2, c3 = st.columns(3)
        with c1:
            chart_card("points_hist", "התפלגות הניקוד",
                       charts.points_hist(ad.points_histogram(ranking)) if ranking is not None else None,
                       "points_distribution")
        with c2:
            chart_card("price_bars", "התפלגות המחיר",
                       charts.price_bars(ad.price_band_data(eda)) if eda else None, "price_distribution_log")
        with c3:
            chart_card("vintage", "ניקוד ממוצע לפי בציר",
                       charts.vintage_line(ad.vintage_data(eda)) if eda else None, "points_by_vintage")
    # 5. recommendations
    if s4:
        with st.container(border=True, key="card_recommendations"):
            st.markdown(ad.localize_countries(s4))
    report = resolve_path(config, "eda_report")
    if report.exists():
        st.download_button("הורדת דוח ה-EDA המלא (HTML)", report.read_bytes(),
                           file_name="eda_report.html", mime="text/html")

# --------------------------------------------------------------------------- how it works
with how_tab:
    st.markdown(TAB_INTRO["how"])
    summary = data["summary"]
    if summary is None:
        _missing("artifacts/run_summary.json")
    else:
        kpi_row(ad.how_kpis(summary, metrics), "kpis_how")
        with st.container(border=True, key="card_flow"):
            st.subheader("איך ה-Flow רץ")
            st.html(ad.flow_html(ad.flow_steps(summary)))
            st.caption("שער 2 נבדק גם בתוך Crew 2, לפני האימון (guardrail של משימת ה-features), "
                       "ושוב כשער של ה-Flow אחרי ש-Crew 2 מסיים.")
            crew_cols = st.columns(2)
            for col, (crew, title) in zip(crew_cols, CREW_TITLES.items()):
                with col:
                    st.markdown(f"**{title}**")
                    requests = summary.get("crews", {}).get(crew, {}).get("requests")
                    st.markdown("\n".join(f"- **{role}**: {ad.AGENT_HE.get(key, '')}"
                                          for key, role in ad.agent_roles(crew)))
                    if requests is not None:
                        st.caption(f"{requests} בקשות למודל השפה בריצה האחרונה")

        # The checks table needs the full width for its sentences.
        with st.container(border=True, key="card_checks"):
            st.subheader("בדיקות")
            checks = pd.DataFrame([
                {"בדיקה": CHECK_NAMES.get(name, name), "עבר": status_icon("passed" if c["passed"] else "failed"),
                 "פרטים": ad.check_sentence(name, c, summary, data["contract"])}
                for name, c in summary["checks"].items()
            ])
            show_table_rtl(checks, column_config={"עבר": st.column_config.ImageColumn("עבר", width="small")})
            with st.expander("פרטים טכניים"):
                for name, c in summary["checks"].items():
                    st.markdown(f"**{CHECK_NAMES.get(name, name)}**")
                    details = ad.check_details(c["message"])
                    if details is not None:
                        st.json(details, expanded=True)
                    else:
                        st.code(c["message"], language=None)

    def run_card() -> None:
        with st.container(border=True, key="card_run"):
            st.subheader("הריצה האחרונה")
            passed = summary["status"] == "passed"
            (st.success if passed else st.error)(
                f"סטטוס הריצה האחרונה: **{summary['status']}**"
                + ("" if passed else f" (נעצרה בבדיקה {summary.get('failed_check')}: {summary.get('error')})"))
            st.markdown(f"**מזהה ריצה:** `{summary['run_id']}`")
            st.metric("תאריך (UTC)", summary["started_at"][:16].replace("T", " "))
            tokens = summary.get("tokens")
            if tokens:   # counted from OpenAI's per-call usage (runs before this field: not shown)
                st.caption(ad.tokens_text(tokens))
            st.markdown('<div class="card-title">שורות</div>', unsafe_allow_html=True)
            rows = summary.get("rows", {})
            labels = {"raw": "בקובץ הגולמי", "clean": "אחרי ניקוי", "features": "ב-features",
                      "train": "אימון", "test": "בדיקה"}
            items = list(labels.items())
            for chunk in (items[:3], items[3:]):
                for col, (key, label) in zip(st.columns(3), chunk):
                    col.metric(label, f"{rows[key]:,}" if key in rows else "—")
            st.caption(f"sha256 של הקובץ הגולמי: `{summary.get('raw_sha256')}`")

    def models_card() -> None:
        with st.container(border=True, key="card_models"):
            st.subheader("השוואת מודלים", help=f"MAE: {ad.GLOSSARY['MAE']}")
            st.markdown(f"{ad.accuracy_sentence(metrics)} (MAE של המודל שנבחר בקבוצת הבדיקה).")
            table = pd.DataFrame([
                {"מודל": name + (" (נבחר)" if name == metrics["selected"] else ""),
                 "CV MAE": ad.round_for_display(m["cv"]["mae"]),
                 "MAE (test)": ad.round_for_display(m["test"]["mae"]),
                 "RMSE (test)": ad.round_for_display(m["test"]["rmse"]),
                 "R² (test)": ad.round_for_display(m["test"]["r2"])}
                for name, m in metrics["models"].items()
            ])
            mae_help = ad.GLOSSARY["MAE"]
            show_table_rtl(table, column_config={
                col: st.column_config.NumberColumn(col, format="%.2f",
                                                   help=mae_help if "MAE" in col else None)
                for col in ("CV MAE", "MAE (test)", "RMSE (test)", "R² (test)")})
            st.caption("MAE הוא השגיאה הממוצעת בנקודות ניקוד. המודל נבחר לפי MAE ממוצע ב-CV, "
                       "וקבוצת הבדיקה משמשת רק לדיווח.")

    # Run card and model comparison side by side (each alone when the other file is missing).
    if metrics is None:
        _missing("artifacts/scientist/metrics.json")
    if summary is not None and metrics is not None:
        with st.container(key="dash_row_run"):
            left, right = st.columns(2)
            with left:
                run_card()
            with right:
                models_card()
    elif summary is not None:
        run_card()
    elif metrics is not None:
        models_card()
    if data["model_card"] or data["evaluation_report"]:
        with st.container(border=True, key="card_model_card"):
            if data["model_card"]:
                st.markdown(data["model_card"])
            if data["evaluation_report"]:
                with st.expander("דוח ההערכה המלא (evaluation_report.md)"):
                    st.markdown(data["evaluation_report"])

st.html(FOOTER)
