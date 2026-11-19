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
SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

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

TABS = ["סקירה", "ניתוח", "מודל", "מציאת שווים למחיר", "חיזוי"]
CHECK_NAMES = {"preflight": "בדיקה מקדימה (קובץ גולמי)", "contract": "שער 1 (חוזה נתונים)",
               "features": "שער 2 (features)", "model": "שער 3 (מודל)"}


@st.cache_data
def _data():
    return ad.load_all(load_config())


@st.cache_resource
def _model():
    return ad.load_model(load_config())


@st.cache_data
def _ranking():
    return ad.load_ranking(load_config())


def show_table_rtl(df: pd.DataFrame, column_config: dict | None = None, **kwargs) -> None:
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
    st.dataframe(ad.rtl_columns(df), hide_index=True, width="stretch", column_config=config, **kwargs)


def _missing(name: str) -> None:
    st.warning(f"הקובץ {name} לא נמצא. הריצו `python -m wine_crew.main` כדי לייצר את התוצרים.")


config = load_config()
data = _data()

st.title("🍷 ניקוד יינות ויינות ששווים את מחירם")
st.caption("wine-crew · כל המספרים מגיעים מתוצרי הריצה האחרונה של ה-Flow. האפליקציה לא מריצה מודלי שפה.")

overview, analysis, model_tab, value_tab, predict_tab = st.tabs(TABS)

# --------------------------------------------------------------------------- overview
with overview:
    summary = data["summary"]
    if summary is None:
        _missing("artifacts/run_summary.json")
    else:
        passed = summary["status"] == "passed"
        (st.success if passed else st.error)(
            f"סטטוס הריצה האחרונה: **{summary['status']}**"
            + ("" if passed else f" (נעצרה בבדיקה {summary.get('failed_check')}: {summary.get('error')})"))
        st.markdown(f"**מזהה ריצה:** `{summary['run_id']}`")
        c1, c2, c3 = st.columns(3)
        c1.metric("תאריך (UTC)", summary["started_at"][:16].replace("T", " "))
        c2.metric("משך", f"{summary['seconds']:.0f} שניות")
        c3.metric("עלות LLM", f"${summary['total_cost_usd']:.3f}")

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

# --------------------------------------------------------------------------- analysis
with analysis:
    if data["insights"]:
        st.markdown(ad.localize_countries(data["insights"]))
    else:
        _missing("artifacts/analyst/insights.md")
    st.divider()
    st.subheader("גרפים")
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

# --------------------------------------------------------------------------- model
with model_tab:
    metrics = data["metrics"]
    if metrics is None:
        _missing("artifacts/scientist/metrics.json")
    else:
        st.subheader("השוואת מודלים")
        table = pd.DataFrame([
            {"מודל": name + (" (נבחר)" if name == metrics["selected"] else ""),
             "CV MAE": ad.round_for_display(m["cv"]["mae"]),
             "MAE (test)": ad.round_for_display(m["test"]["mae"]),
             "RMSE (test)": ad.round_for_display(m["test"]["rmse"]),
             "R² (test)": ad.round_for_display(m["test"]["r2"])}
            for name, m in metrics["models"].items()
        ])
        show_table_rtl(table, column_config={
            col: st.column_config.NumberColumn(col, format="%.2f")
            for col in ("CV MAE", "MAE (test)", "RMSE (test)", "R² (test)")})
        st.caption("MAE הוא השגיאה הממוצעת בנקודות ניקוד. המודל נבחר לפי MAE ממוצע ב-CV, "
                   "וקבוצת הבדיקה משמשת רק לדיווח.")
    if data["model_card"]:
        st.divider()
        st.markdown(data["model_card"])
    if data["evaluation_report"]:
        with st.expander("דוח ההערכה המלא (evaluation_report.md)"):
            st.markdown(data["evaluation_report"])

# --------------------------------------------------------------------------- value wines
with value_tab:
    ranking = _ranking()
    if ranking is None:
        _missing("artifacts/scientist/value_ranking.csv")
    else:
        st.markdown("ציון שווי = ניקוד בפועל פחות הניקוד שהמודל חזה לפי המחיר, המקור, הזן והבציר. "
                    "התחזית לכל יין מגיעה ממודל שלא ראה אותו באימון (out-of-fold).")
        countries = ad.sort_countries(ranking["country"].dropna().unique().tolist())
        varieties = sorted(ranking["variety"].dropna().unique().tolist())
        f1, f2 = st.columns(2)
        chosen_countries = f1.multiselect("מדינה", countries, key="value_countries",
                                          format_func=ad.country_label,
                                          placeholder="כל המדינות (אפשר לבחור כמה)")
        chosen_varieties = f2.multiselect("זן", varieties, key="value_varieties",
                                          placeholder="כל הזנים (אפשר לבחור כמה)")
        lo, hi = ad.price_range(data["contract"])
        p1, p2, p3 = st.columns(3)
        price_min = p1.number_input("מחיר מינימלי ($)", lo, hi, lo, step=1.0, key="value_price_min")
        file_max = min(hi, float(ranking["price"].max()))
        price_max = p2.number_input("מחיר מקסימלי ($)", lo, hi, file_max, step=1.0, key="value_price_max")
        top = p3.slider("מספר יינות", 10, 200, 50, step=10, key="value_top")
        result = ad.filter_ranking(ranking, chosen_countries, chosen_varieties, price_min, price_max, top)
        st.caption(f"{len(result)} יינות מוצגים")
        shown = result.assign(vintage=result["vintage"].map(lambda v: "" if pd.isna(v) else str(int(v))),
                              country=result["country"].map(ad.country_label))
        show_table_rtl(
            shown[["rank", "title", "country", "province", "variety", "vintage", "price", "points",
                   "predicted", "value_score"]],
            key="value_table",
            column_config={
                "rank": "דירוג כללי", "title": "יין", "country": "מדינה", "province": "מחוז",
                "variety": "זן", "vintage": st.column_config.TextColumn("בציר"),
                "price": st.column_config.NumberColumn("מחיר", format="$%.0f"),
                "points": "ניקוד", "predicted": st.column_config.NumberColumn("ניקוד צפוי", format="%.1f"),
                "value_score": st.column_config.NumberColumn("ציון שווי", format="%.2f"),
            })

# --------------------------------------------------------------------------- prediction
with predict_tab:
    model = _model()
    metrics = data["metrics"]
    if model is None or metrics is None:
        _missing("artifacts/scientist/model.pkl")
    else:
        st.markdown("הניקוד הצפוי של יין לפי מחיר, מקור, זן ובציר. האפשרויות הן הקטגוריות שהמודל "
                    "מכיר; קטגוריות נדירות מטופלות כ-\"Other\".")
        opts = ad.model_options(model)
        ranking = _ranking()
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
            st.metric("ניקוד צפוי", f"{points:.1f}")
            mae = metrics["models"][metrics["selected"]]["test"]["mae"]
            st.caption(f"השגיאה הממוצעת של המודל ({metrics['selected']}) היא {mae:.2f} נקודות.")
