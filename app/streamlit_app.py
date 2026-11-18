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
[data-testid="stDataFrame"], [data-testid="stTable"], code, pre { direction: ltr; text-align: left; }
.stMarkdown table { direction: rtl; }
</style>
"""
st.markdown(RTL_CSS, unsafe_allow_html=True)

TABS = ["סקירה", "ניתוח", "מודל", "מציאת שווים למחיר", "חיזוי"]
CHECK_NAMES = {"preflight": "בדיקה מקדימה (קובץ גולמי)", "contract": "שער 1: חוזה נתונים",
               "features": "שער 2: features", "model": "שער 3: מודל"}


@st.cache_data
def _data():
    return ad.load_all(load_config())


@st.cache_resource
def _model():
    return ad.load_model(load_config())


@st.cache_data
def _ranking():
    return ad.load_ranking(load_config())


def _missing(name: str) -> None:
    st.warning(f"הקובץ {name} לא נמצא. הריצו `python -m wine_crew.main` כדי לייצר את התוצרים.")


config = load_config()
data = _data()

st.title("🍷 wine-crew: ניקוד יינות ויינות ששווים את מחירם")
st.caption("כל המספרים מגיעים מתוצרי הריצה האחרונה של ה-Flow. האפליקציה לא מריצה מודלי שפה.")

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
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("מזהה ריצה", summary["run_id"])
        c2.metric("תאריך (UTC)", summary["started_at"][:16].replace("T", " "))
        c3.metric("משך", f"{summary['seconds']:.0f} שניות")
        c4.metric("עלות LLM", f"${summary['total_cost_usd']:.3f}")

        st.subheader("בדיקות")
        checks = pd.DataFrame([
            {"בדיקה": CHECK_NAMES.get(name, name), "עבר": "✅" if c["passed"] else "❌", "פרטים": c["message"]}
            for name, c in summary["checks"].items()
        ])
        st.dataframe(checks, hide_index=True, width="stretch")

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
        st.markdown(data["insights"])
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
             "CV MAE": m["cv"]["mae"], "MAE (test)": m["test"]["mae"],
             "RMSE (test)": m["test"]["rmse"], "R² (test)": m["test"]["r2"]}
            for name, m in metrics["models"].items()
        ])
        st.dataframe(table.round(3), hide_index=True, width="stretch")
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
        countries = ranking["country"].value_counts().index.tolist()
        varieties = ranking["variety"].value_counts().index.tolist()
        f1, f2 = st.columns(2)
        chosen_countries = f1.multiselect("מדינה", countries, key="value_countries")
        chosen_varieties = f2.multiselect("זן", varieties, key="value_varieties")
        lo, hi = ad.price_range(data["contract"])
        p1, p2, p3 = st.columns(3)
        price_min = p1.number_input("מחיר מינימלי ($)", lo, hi, lo, step=1.0, key="value_price_min")
        price_max = p2.number_input("מחיר מקסימלי ($)", lo, hi, 50.0, step=1.0, key="value_price_max")
        top = p3.slider("מספר יינות", 10, 200, 50, step=10, key="value_top")
        result = ad.filter_ranking(ranking, chosen_countries, chosen_varieties, price_min, price_max, top)
        st.caption(f"{len(result)} יינות מוצגים")
        st.dataframe(
            result[["rank", "title", "country", "province", "variety", "vintage", "price", "points",
                    "predicted", "value_score"]],
            hide_index=True, width="stretch", key="value_table",
            column_config={
                "rank": "דירוג כללי", "title": "יין", "country": "מדינה", "province": "מחוז",
                "variety": "זן", "vintage": st.column_config.NumberColumn("בציר", format="%d"),
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
            c1, c2 = st.columns(2)
            country = c1.selectbox("מדינה", sorted(opts["country"]),
                                   index=sorted(opts["country"]).index("US") if "US" in opts["country"] else 0,
                                   key="form_country")
            province_opts = provinces.get(country) or sorted(opts["province"])
            province = c2.selectbox("מחוז", province_opts, key="form_province")
            region_opts = [ad.UNKNOWN] + regions.get(province, [])
            region = c1.selectbox("אזור (region_1)", region_opts, key="form_region")
            variety = c2.selectbox("זן", sorted(opts["variety"]), key="form_variety")
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
