"""Streamlit app for wine-crew, in Hebrew (RTL).

It only reads artifacts/ and model.pkl: it never runs the Flow and needs no API
key. Run with:
    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

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
reload_if_changed(SRC / "wine_crew", clear_caches=_clear_caches)

from wine_crew import app_data as ad  # noqa: E402
from wine_crew.config import load_config, resolve_path  # noqa: E402

st.set_page_config(page_title="wine-crew", page_icon="🍷", layout="wide")

RTL_CSS = """
<style>
.stApp, .stMarkdown, .stTabs, [data-testid="stSidebar"] { direction: rtl; text-align: right; }
.stMarkdown ul, .stMarkdown ol, .stMarkdown li { direction: rtl; text-align: right; }
/* Headings and captions: set the direction explicitly, so a line that starts with Latin text
   (e.g. "wine-crew · ...") is still laid out right-to-left. */
[data-testid="stHeading"], [data-testid="stHeading"] h1, [data-testid="stHeading"] h2,
[data-testid="stHeading"] h3, [data-testid="stCaptionContainer"] { direction: rtl; text-align: right; }
[data-testid="stDataFrame"], [data-testid="stTable"], code, pre { direction: ltr; text-align: left; }
/* Dropdown lists open in a popover outside the widget, so target the dropdown itself. */
[data-testid="stSelectboxVirtualDropdown"], [data-testid="stMultiSelectDropdown"] { direction: rtl; }
[data-testid="stSelectboxVirtualDropdown"] [role="option"],
[data-testid="stMultiSelectDropdown"] [role="option"] { direction: rtl; text-align: right; }
.stMarkdown table { direction: rtl; }
</style>
"""
st.html(RTL_CSS)

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
    x = alt.X("price:Q", title="מחיר", scale=alt.Scale(type="log"), axis=alt.Axis(format="$,.0f"))
    y = alt.Y("predicted:Q", title="ניקוד צפוי", scale=alt.Scale(zero=False))
    tooltip = [alt.Tooltip("price:Q", title="מחיר", format="$,.0f"),
               alt.Tooltip("predicted:Q", title="ניקוד צפוי", format=".1f")]
    line = alt.Chart(curve).mark_line().encode(x=x, y=y, tooltip=tooltip)
    dot = alt.Chart(pd.DataFrame({"price": [chosen], "predicted": [chosen_points]})).mark_point(
        size=120, filled=True).encode(x=x, y=y, tooltip=tooltip)
    return (line + dot).properties(title="הניקוד הצפוי ליין כזה לפי מחיר", height=260)


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


def _missing(name: str) -> None:
    st.warning(f"הקובץ {name} לא נמצא. הריצו `python -m wine_crew.main` כדי לייצר את התוצרים.")


config = load_config()
data = _data(ad.file_fingerprint(ad.data_files(config).values()))

ranking_fp = ad.file_fingerprint([resolve_path(config, "value_ranking")])
ranking = _ranking(ranking_fp)
band = _band(ranking_fp)
metrics = data["metrics"]

st.title("🍷 ניקוד יינות ויינות ששווים את מחירם")
st.caption("wine-crew · כל המספרים מגיעים מתוצרי הריצה האחרונה של ה-Flow. האפליקציה לא מריצה מודלי שפה.")
st.markdown(INTRO)

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
        shown = result.assign(vintage=result["vintage"].map(lambda v: "" if pd.isna(v) else str(int(v))),
                              country=result["country"].map(ad.country_label),
                              label=result["points"].map(ad.score_label))
        event = show_table_rtl(
            shown[["rank", "title", "country", "province", "variety", "vintage", "price", "points",
                   "label", "predicted", "value_score"]],
            key="value_table", on_select="rerun", selection_mode="single-row",
            column_config={
                "rank": "דירוג כללי", "title": "יין", "country": "מדינה", "province": "מחוז",
                "variety": "זן", "vintage": st.column_config.TextColumn("בציר"),
                "price": st.column_config.NumberColumn("מחיר", format="$%.0f"),
                "points": "ניקוד", "label": "דרגה",
                "predicted": st.column_config.NumberColumn("ניקוד צפוי", format="%.1f",
                                                           help=ad.GLOSSARY["ניקוד צפוי"]),
                "value_score": st.column_config.NumberColumn("ציון שווי", format="%+.1f",
                                                             help=ad.GLOSSARY["ציון שווי"]),
            })
        if len(result):
            first = result.iloc[0]
            st.caption(f"לדוגמה, ליין הראשון ציון שווי {first['value_score']:+.1f}: "
                       f"{ad.value_sentence(first['value_score'])}.")
            st.download_button("הורדת הרשימה (CSV)", ad.ranking_csv(result), file_name="value_wines.csv",
                               mime="text/csv", key="value_csv")
            rows = event.selection.rows if event is not None else []
            if not rows:
                st.caption("בחרו שורה בטבלה כדי לראות את פרטי היין.")
            else:
                wine = result.iloc[rows[0]]
                pct, count = ad.variety_percentile(ranking, wine["variety"], wine["value_score"])
                region = wine["region_1"] if isinstance(wine["region_1"], str) else ad.UNKNOWN
                vintage_text = "" if pd.isna(wine["vintage"]) else str(int(wine["vintage"]))
                with st.container(border=True, key="wine_card"):
                    st.markdown(f"**{wine['title']}**")
                    st.markdown(
                        f"{ad.country_label(wine['country'])} · {wine['province']} · {region} · "
                        f"{wine['variety']} · בציר {vintage_text or ad.UNKNOWN} · {ad.money(wine['price'])}")
                    st.markdown(f"ניקוד בפועל: **{ad.score_text(wine['points'])}** · "
                                f"ניקוד צפוי: {wine['predicted']:.1f}")
                    st.markdown(f"ציון שווי {wine['value_score']:+.1f}: {ad.value_sentence(wine['value_score'])}.")
                    st.markdown(ad.variety_rank_sentence(pct, wine["variety"], count))

# --------------------------------------------------------------------------- check a wine
with predict_tab:
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
                st.altair_chart(price_chart(curve, price, points), width="stretch", key="price_chart")
                sentence = ad.curve_sentence(curve, price, high)
                if sentence:
                    st.markdown(sentence)
                if ad.is_rare_price(price, (low, high)):
                    st.caption(ad.RARE_PRICE_NOTE)

                st.subheader("יינות שכדאי להכיר")
                similar, levels = ad.similar_wines(ranking, country, variety, price)
                st.caption(f"עד 5 יינות במחיר {ad.money(price / 1.5)}–{ad.money(price * 1.5)}, "
                           "עם ציון השווי הגבוה ביותר.")
                if len(similar):
                    shown = similar.assign(
                        vintage=similar["vintage"].map(lambda v: "" if pd.isna(v) else str(int(v))),
                        country=similar["country"].map(ad.country_label),
                        label=similar["points"].map(ad.score_label))
                    show_table_rtl(
                        shown[["title", "country", "variety", "vintage", "price", "points", "label",
                               "value_score"]],
                        key="similar_table",
                        column_config={
                            "title": "יין", "country": "מדינה", "variety": "זן",
                            "vintage": st.column_config.TextColumn("בציר"),
                            "price": st.column_config.NumberColumn("מחיר", format="$%.0f"),
                            "points": "ניקוד", "label": "דרגה",
                            "value_score": st.column_config.NumberColumn("ציון שווי", format="%+.1f",
                                                                         help=ad.GLOSSARY["ציון שווי"]),
                        })
                note = ad.similar_note(levels, country, variety)
                if note:
                    st.caption(note)

# --------------------------------------------------------------------------- what we learned
with learned_tab:
    st.markdown(TAB_INTRO["learned"])
    if data["insights"]:
        st.markdown(ad.localize_countries(data["insights"]))
    else:
        _missing("artifacts/analyst/insights.md")
    st.divider()
    st.subheader("גרפים", help=f"חציון: {ad.GLOSSARY['חציון']}")
    charts = ad.chart_paths(config)
    if not charts:
        _missing("artifacts/analyst/charts/")
    names = list(charts)
    for i in range(0, len(names), 2):
        cols = st.columns(2)
        for col, name in zip(cols, names[i:i + 2]):
            col.image(str(charts[name]), caption=ad.CHART_TITLES_HE[name], width="stretch")
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
        st.subheader("איך ה-Flow רץ")
        st.markdown(ad.flow_line(ad.flow_steps(summary)))
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

        st.subheader("הריצה האחרונה")
        passed = summary["status"] == "passed"
        (st.success if passed else st.error)(
            f"סטטוס הריצה האחרונה: **{summary['status']}**"
            + ("" if passed else f" (נעצרה בבדיקה {summary.get('failed_check')}: {summary.get('error')})"))
        st.markdown(f"**מזהה ריצה:** `{summary['run_id']}`")
        c1, c2, c3 = st.columns(3)
        c1.metric("תאריך (UTC)", summary["started_at"][:16].replace("T", " "))
        c2.metric("משך", f"{summary['seconds']:.0f} שניות")
        c3.metric("עלות LLM", f"${summary['total_cost_usd']:.4f}")
        tokens = summary.get("tokens")
        if tokens:   # counted from OpenAI's per-call usage (runs before this field: not shown)
            st.caption(ad.tokens_text(tokens))

        st.subheader("בדיקות")
        checks = pd.DataFrame([
            {"בדיקה": CHECK_NAMES.get(name, name), "עבר": "✅" if c["passed"] else "❌",
             "פרטים": ad.check_sentence(name, c, summary, data["contract"])}
            for name, c in summary["checks"].items()
        ])
        show_table_rtl(checks)
        with st.expander("פרטים טכניים"):
            for name, c in summary["checks"].items():
                st.markdown(f"**{CHECK_NAMES.get(name, name)}**")
                details = ad.check_details(c["message"])
                if details is not None:
                    st.json(details, expanded=True)
                else:
                    st.code(c["message"], language=None)

        st.subheader("שורות")
        rows = summary.get("rows", {})
        labels = {"raw": "בקובץ הגולמי", "clean": "אחרי ניקוי", "features": "ב-features",
                  "train": "אימון", "test": "בדיקה"}
        cols = st.columns(len(labels))
        for col, (key, label) in zip(cols, labels.items()):
            col.metric(label, f"{rows[key]:,}" if key in rows else "—")
        st.caption(f"sha256 של הקובץ הגולמי: `{summary.get('raw_sha256')}`")

    if metrics is None:
        _missing("artifacts/scientist/metrics.json")
    else:
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
    if data["model_card"]:
        st.divider()
        st.markdown(data["model_card"])
    if data["evaluation_report"]:
        with st.expander("דוח ההערכה המלא (evaluation_report.md)"):
            st.markdown(data["evaluation_report"])
