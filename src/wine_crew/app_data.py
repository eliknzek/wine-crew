"""Data helpers for the Streamlit app. Pure functions, testable without Streamlit.

The app only reads artifacts/ and model.pkl: it never runs the Flow and needs
no API key. Form options come from the categories the model learned
(RareCategoryGrouper.keep_), not from the large clean_data.csv.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import yaml

from wine_crew.config import load_config, resolve_path

UNKNOWN = "לא ידוע"
# Same names and order as eda_tools.CHART_NAMES (not imported: it pulls in matplotlib).
CHART_TITLES_HE = {
    "points_distribution": "התפלגות הניקוד",
    "price_distribution_log": "התפלגות המחיר (סקאלה לוגריתמית)",
    "points_vs_log_price": "ניקוד מול מחיר",
    "by_country": "ניקוד חציוני ומחיר חציוני לפי מדינה",
    "by_variety": "ניקוד חציוני ומחיר חציוני לפי זן",
    "points_by_vintage": "ניקוד לפי בציר",
    "missing_values": "ערכים חסרים לפי עמודה",
}


def artifact_path(config: dict[str, Any], key: str) -> Path:
    return resolve_path(config, key)


def read_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def read_text(path: Path) -> str | None:
    try:
        return Path(path).read_text(encoding="utf-8")
    except OSError:
        return None


def data_files(config: dict[str, Any]) -> dict[str, Path]:
    """The small artifacts load_all reads."""
    sci = resolve_path(config, "artifacts_scientist")
    return {
        "summary": artifact_path(config, "run_summary"),
        "metrics": sci / "metrics.json",
        "contract": artifact_path(config, "contract"),
        "insights": artifact_path(config, "insights"),
        "model_card": sci / "model_card.md",
        "evaluation_report": sci / "evaluation_report.md",
        "eda_summary": artifact_path(config, "eda_summary"),
    }


def model_file(config: dict[str, Any]) -> Path:
    return resolve_path(config, "artifacts_scientist") / "model.pkl"


def file_fingerprint(paths) -> tuple:
    """(path, mtime_ns, size) per file, None for a missing one. A cache key that changes
    whenever an artifact is replaced (git rewrites changed files on pull)."""
    items = []
    for path in sorted(str(p) for p in paths):
        try:
            stat = Path(path).stat()
            items.append((path, stat.st_mtime_ns, stat.st_size))
        except OSError:
            items.append((path, None, None))
    return tuple(items)


def load_all(config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Small artifacts the app shows; a missing file is None."""
    config = config if config is not None else load_config()
    files = data_files(config)
    return {
        "summary": read_json(files["summary"]),
        "metrics": read_json(files["metrics"]),
        "contract": read_json(files["contract"]),
        "insights": read_text(files["insights"]),
        "model_card": read_text(files["model_card"]),
        "evaluation_report": read_text(files["evaluation_report"]),
        "eda_summary": read_json(files["eda_summary"]),
    }


def chart_paths(config: dict[str, Any]) -> dict[str, Path]:
    charts = resolve_path(config, "charts_dir")
    return {n: charts / f"{n}.png" for n in CHART_TITLES_HE if (charts / f"{n}.png").exists()}


def load_model(config: dict[str, Any]):
    path = model_file(config)
    return joblib.load(path) if path.exists() else None


def load_ranking(config: dict[str, Any]) -> pd.DataFrame | None:
    path = resolve_path(config, "value_ranking")
    return pd.read_csv(path) if path.exists() else None


# --------------------------------------------------------------------------- form options


def model_options(model) -> dict[str, list[str]]:
    """The categories the model knows for each categorical input (rarer ones map to "Other")."""
    return {col: list(values) for col, values in model.named_steps["group_rare"].keep_.items()}


def price_range(contract: dict[str, Any] | None) -> tuple[float, float]:
    spec = next((c for c in (contract or {}).get("columns", []) if c["name"] == "price"), {})
    return float(spec.get("min", 4)), float(spec.get("max", 3300))


def vintage_range(config: dict[str, Any]) -> tuple[int, int]:
    return int(config["cleaning"]["vintage_min"]), int(config["cleaning"]["vintage_max"])


def options_by_parent(ranking: pd.DataFrame | None, options: dict[str, list[str]],
                      parent: str, child: str) -> dict[str, list[str]]:
    """parent value -> known child values seen with it (e.g. country -> provinces).

    Only categories the model knows are kept. Without the ranking file every
    parent gets the full known child list.
    """
    known_parent, known_child = set(options.get(parent, [])), options.get(child, [])
    if ranking is None:
        return {p: list(known_child) for p in sorted(known_parent)}
    pairs = ranking[[parent, child]].dropna().drop_duplicates()
    pairs = pairs[pairs[parent].isin(known_parent) & pairs[child].isin(set(known_child))]
    return {p: sorted(g[child].tolist()) for p, g in pairs.groupby(parent)}


# --------------------------------------------------------------------------- prediction and ranking


def _frame(metrics: dict[str, Any], rows: list[dict[str, Any]]) -> pd.DataFrame:
    """Model input for several wines. vintage or region_1 of None / UNKNOWN become NaN."""
    columns = metrics["features"]["input_columns"][metrics["selected"]]
    values = {col: [] for col in columns}
    for row in rows:
        for col in columns:
            value = row.get(col)
            values[col].append(np.nan if value is None or value == UNKNOWN else value)
    frame = pd.DataFrame(values, columns=columns)
    frame["price"] = frame["price"].astype("float64")
    frame["vintage"] = pd.to_numeric(frame["vintage"]).astype("float64")
    return frame


def predict_points(model, metrics: dict[str, Any], row: dict[str, Any]) -> float:
    """Predicted points for one wine. vintage or region_1 of None / UNKNOWN become NaN."""
    return float(model.predict(_frame(metrics, [row]))[0])


def price_curve(model, metrics: dict[str, Any], row: dict[str, Any], prices) -> pd.DataFrame:
    """Predicted points for the same wine at each price (one predict call for the whole grid)."""
    prices = [float(p) for p in prices]
    predicted = model.predict(_frame(metrics, [{**row, "price": p} for p in prices]))
    return pd.DataFrame({"price": prices, "predicted": predicted.astype(float)})


def filter_ranking(ranking: pd.DataFrame, countries: list[str] | None = None,
                   varieties: list[str] | None = None, price_min: float | None = None,
                   price_max: float | None = None, top: int = 50) -> pd.DataFrame:
    """Wines matching the filters, best value_score first."""
    df = ranking
    if countries:
        df = df[df["country"].isin(countries)]
    if varieties:
        df = df[df["variety"].isin(varieties)]
    if price_min is not None:
        df = df[df["price"] >= price_min]
    if price_max is not None:
        df = df[df["price"] <= price_max]
    return df.sort_values(["value_score", "title"], ascending=[False, True], kind="mergesort").head(top)


# --------------------------------------------------------------------------- display helpers

DETAIL_LABELS = {
    "raw_csv": "קובץ", "sha256": "sha256", "rows": "שורות", "columns": "עמודות",
    "selected": "מודל נבחר", "mae": "MAE", "baseline_mae": "MAE של baseline",
    "sample_predictions": "תחזיות לדוגמה", "ranked_wines": "יינות בדירוג",
}


def _detail_value(key: str, value: Any) -> str:
    if key == "sha256" and isinstance(value, str):
        return value[:12] + "…"
    if isinstance(value, bool) or value is None:
        return str(value)
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, float):
        return f"{value:.2f}"
    if isinstance(value, list):
        return ", ".join(_detail_value(key, v) for v in value)
    return str(value)


def check_details_text(message: str) -> str:
    """A check message from run_summary.json ("ok {json}" or a failure text) as readable text."""
    if not message.startswith("ok "):
        return message
    try:
        details = json.loads(message[3:])
    except ValueError:
        return message
    return " · ".join(f"{DETAIL_LABELS.get(k, k)}: {_detail_value(k, v)}" for k, v in details.items())


def most_common(ranking: pd.DataFrame | None, column: str, options: list[str],
                where: dict[str, Any] | None = None) -> str | None:
    """The option that appears most often in the ranking (optionally within a filter)."""
    if ranking is None or not options:
        return None
    df = ranking
    for col, value in (where or {}).items():
        df = df[df[col] == value]
    counts = df[column].value_counts()
    counts = counts[counts.index.isin(set(options))]
    return str(counts.index[0]) if len(counts) else None


FAILED_PREFIXES = {
    "preflight": ("Preflight check failed: ", "הבדיקה המקדימה נכשלה"),
    "contract": ("Contract check failed: ", "הנתונים הנקיים לא עומדים בחוזה"),
    "features": ("Feature check failed: ", "ה-features לא תקינים"),
    "model": ("Model check failed: ", "המודל לא עבר את הבדיקה"),
}


def check_details(message: str) -> dict[str, Any] | None:
    """The details dict of a passed check ("ok {json}"), or None."""
    if not message.startswith("ok "):
        return None
    try:
        details = json.loads(message[3:])
    except ValueError:
        return None
    return details if isinstance(details, dict) else None


def check_sentence(name: str, check: dict[str, Any], summary: dict[str, Any],
                   contract: dict[str, Any] | None = None) -> str:
    """One plain Hebrew sentence per check. Every number comes from the summary or the artifacts."""
    message = check.get("message", "")
    if not check.get("passed"):
        english, hebrew = FAILED_PREFIXES.get(name, ("", "הבדיקה נכשלה"))
        if name == "preflight" and "raw file not found" in message:
            return "קובץ הנתונים הגולמי לא נמצא, ולכן הריצה לא התחילה."
        return f"{hebrew}: {message.removeprefix(english)}"
    details = check_details(message) or {}
    rows = summary.get("rows", {})
    if name == "preflight":
        sha = details.get("sha256") or summary.get("raw_sha256")
        contract_sha = (contract or {}).get("source", {}).get("sha256")
        if sha and contract_sha:
            relation = "זהה לזו" if sha == contract_sha else "שונה מזו"
            return f"קובץ הנתונים הגולמי נמצא, וטביעת האצבע שלו (sha256) {relation} שנרשמה בחוזה הנתונים."
        return "קובץ הנתונים הגולמי נמצא, וטביעת האצבע שלו (sha256) נרשמה."
    if name == "contract":
        n = details.get("rows", rows.get("clean"))
        return f"הנתונים הנקיים עומדים בכל כללי החוזה: {n:,} שורות." if n is not None else \
            "הנתונים הנקיים עומדים בכל כללי החוזה."
    if name == "features":
        n_cols, n_rows = len(details.get("columns", [])), details.get("rows", rows.get("features"))
        return f"נבנו {n_cols} עמודות features לכל {n_rows:,} השורות."
    if name == "model":
        selected = details.get("selected", summary.get("selected"))
        mae, base = details.get("mae"), details.get("baseline_mae")
        if mae is None or base is None:
            return f"המודל שנבחר ({selected}) עבר את הבדיקה."
        return (f"המודל שנבחר ({selected}) טועה בממוצע ב-{mae:.2f} נקודות, "
                f"לעומת {base:.2f} בניחוש הבסיס.")
    return check_details_text(message)


def rtl_columns(df: pd.DataFrame) -> pd.DataFrame:
    """The same table with its columns in reverse order, so the first column ends up on the
    right in a right-to-left layout. Rows, values and column names are unchanged."""
    return df[df.columns[::-1]]


def is_text_column(series: pd.Series) -> bool:
    return not (pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series))


# --------------------------------------------------------------------------- Hebrew country names (display only)

# Every country in the cleaned data. The data, the filters and the model keep the English
# name; this is only how a country is shown.
COUNTRY_HE = {
    "Argentina": "ארגנטינה", "Armenia": "ארמניה", "Australia": "אוסטרליה", "Austria": "אוסטריה",
    "Bosnia and Herzegovina": "בוסניה והרצגובינה", "Brazil": "ברזיל", "Bulgaria": "בולגריה",
    "Canada": "קנדה", "Chile": "צ'ילה", "China": "סין", "Croatia": "קרואטיה", "Cyprus": "קפריסין",
    "Czech Republic": "צ'כיה", "England": "אנגליה", "France": "צרפת", "Georgia": "גאורגיה",
    "Germany": "גרמניה", "Greece": "יוון", "Hungary": "הונגריה", "India": "הודו", "Israel": "ישראל",
    "Italy": "איטליה", "Lebanon": "לבנון", "Luxembourg": "לוקסמבורג", "Macedonia": "מקדוניה",
    "Mexico": "מקסיקו", "Moldova": "מולדובה", "Morocco": "מרוקו", "New Zealand": "ניו זילנד",
    "Peru": "פרו", "Portugal": "פורטוגל", "Romania": "רומניה", "Serbia": "סרביה",
    "Slovakia": "סלובקיה", "Slovenia": "סלובניה", "South Africa": "דרום אפריקה", "Spain": "ספרד",
    "Switzerland": "שווייץ", "Turkey": "טורקיה", "US": "ארצות הברית", "Ukraine": "אוקראינה",
    "Uruguay": "אורוגוואי",
}


def country_label(country: Any) -> str:
    """'ישראל (Israel)'; a country without a translation stays in English."""
    name = str(country)
    hebrew = COUNTRY_HE.get(name)
    return f"{hebrew} ({name})" if hebrew else name


def sort_countries(countries: list[str]) -> list[str]:
    """English names ordered by their Hebrew name (alphabetical); untranslated ones last, A-Z."""
    return sorted(countries, key=lambda c: (c not in COUNTRY_HE, COUNTRY_HE.get(c, c)))


def round_for_display(value: float, decimals: int = 2) -> float:
    """Round for a table cell, turning -0.0 into 0.0 (R² of -0.0001 shows as 0.00, not -0.00)."""
    return round(float(value), decimals) + 0.0


def localize_countries(text: str) -> str:
    """Show every country name from COUNTRY_HE in text as 'אוסטריה (Austria)'. Display only.

    Longest names first, so "South Africa" is replaced as one name. A name is replaced only
    when no Latin letter touches it ("US" is left alone inside "USD"), and a name that is
    already labeled ("ישראל (Israel)") is not labeled again.
    """
    for name in sorted(COUNTRY_HE, key=len, reverse=True):
        pattern = re.compile(rf"(?<![A-Za-z(]){re.escape(name)}(?![A-Za-z)])")
        text = pattern.sub(lambda m: country_label(m.group(0)), text)
    return text


def tokens_text(tokens: dict[str, int]) -> str:
    """One Hebrew line for the run's counted LLM usage."""
    return (f"{tokens.get('requests', 0):,} בקשות · טוקני קלט: {tokens.get('input_tokens', 0):,} "
            f"(מהם {tokens.get('cached_input_tokens', 0):,} מהמטמון) · "
            f"טוקני פלט: {tokens.get('output_tokens', 0):,}")


# --------------------------------------------------------------------------- numbers in words

# WineEnthusiast's official rating scale (lowest points of each band).
SCORE_SCALE = [(98, "קלאסי"), (94, "יוצא דופן"), (90, "מצוין"), (87, "טוב מאוד"), (83, "טוב"), (80, "סביר")]
POINTS_MIN, POINTS_MAX = 80, 100

# One plain sentence per term, shown as a ⓘ (help=) next to it.
GLOSSARY = {
    "ציון שווי": "כמה נקודות היין קיבל מעל הניקוד שצפוי ליין כזה במחיר הזה (או מתחת, כשהציון שלילי). "
                 "\"יין כזה\": אותו מקור, זן ובציר.",
    "ניקוד צפוי": "הניקוד שהמודל נותן ליין לפי המחיר, המקור, הזן והבציר, בלי לטעום אותו.",
    "חציון": "הערך האמצעי: חצי מהיינות מעליו וחצי מתחתיו.",
    "out-of-fold": "התחזית לכל יין באה ממודל שלא ראה את היין הזה באימון, כך שהיא לא \"מכירה את התשובה\".",
    "MAE": "השגיאה הממוצעת, בנקודות ניקוד, בין התחזית לניקוד האמיתי. ככל שהיא נמוכה יותר, המודל מדויק יותר.",
}


def _clip_points(value: float) -> int:
    """Round half up (82.5 -> 83, not Python's banker's 82) and clip to 80-100."""
    return int(min(max(math.floor(value + 0.5), POINTS_MIN), POINTS_MAX))


def score_label(points: float) -> str:
    """The WineEnthusiast word for a score, after rounding to a whole number (clipped to 80-100)."""
    rounded = _clip_points(points)
    return next(label for low, label in SCORE_SCALE if rounded >= low)


def score_text(points: float) -> str:
    """'89: טוב מאוד' (whole number and its word)."""
    return f"{_clip_points(points)}: {score_label(points)}"


def value_sentence(score: float) -> str:
    """value_score in words: how far above (or below) the points expected for such a wine at this price."""
    shown = round(float(score), 1)
    if shown == 0:
        return "בדיוק הניקוד שצפוי ליין כזה במחיר הזה"
    side = "מעל ה" if shown > 0 else "מתחת ל"
    return f"{abs(shown):.1f} נקודות {side}ניקוד שצפוי ליין כזה במחיר הזה"


def signed(value: float) -> str:
    """'+9.1' kept as one left-to-right unit inside Hebrew text (Unicode isolate), so the sign stays put."""
    return f"\u2066{float(value):+.1f}\u2069"


def signed_html(value: float) -> str:
    """'+9.1' inside an HTML <bdi dir="ltr"> element as well: an element-level isolate keeps the
    sign before the number in Hebrew text even where the Unicode isolate characters alone are not
    honored (seen on mobile next to "כלומר"). For text rendered as HTML."""
    return f'<bdi dir="ltr">{signed(value)}</bdi>'


def model_mae(metrics: dict[str, Any]) -> float:
    """Test MAE of the selected model."""
    return float(metrics["models"][metrics["selected"]]["test"]["mae"])


def accuracy_sentence(metrics: dict[str, Any]) -> str:
    return f"המודל טועה בממוצע בכ-{model_mae(metrics):.1f} נקודות"


def residual_band(ranking: pd.DataFrame, low_q: float = 0.1, high_q: float = 0.9) -> tuple[float, float]:
    """10th and 90th percentile of points - predicted (out-of-fold predictions) in the ranking file.
    8 in 10 wines scored within [predicted + low, predicted + high]."""
    gaps = (ranking["points"] - ranking["predicted"]).dropna()
    return float(gaps.quantile(low_q)), float(gaps.quantile(high_q))


def prediction_range(points: float, band: tuple[float, float]) -> tuple[int, int]:
    """Whole-number range for the real score, clipped to 80-100.

    Rounded inward (ceil / floor): real scores are whole numbers, so this range holds exactly
    the scores inside the band. Rounding to nearest would widen it (about 87% instead of 80%).
    """
    low = math.ceil(points + band[0] - 1e-9)
    high = math.floor(points + band[1] + 1e-9)
    return min(max(low, POINTS_MIN), POINTS_MAX), min(max(high, POINTS_MIN), POINTS_MAX)


def prediction_sentence(points: float, band: tuple[float, float]) -> str:
    low, high = prediction_range(points, band)
    return f"ב-8 מתוך 10 יינות דומים, הניקוד האמיתי היה בין {low} ל-{high}."


# --------------------------------------------------------------------------- the Flow, step by step

CREW_DIRS = {"crew1": "analyst", "crew2": "scientist"}
# Order in which the Flow runs them (flow.py). Training is a task inside Crew 2.
FLOW_STEPS = [
    ("preflight", "בדיקה מקדימה"),
    ("crew1", "Crew 1"),
    ("contract", "שער 1 (חוזה נתונים)"),
    ("crew2", "Crew 2 (כולל אימון)"),
    ("features", "שער 2 (features)"),
    ("model", "שער 3 (מודל)"),
]
# Status words (shown next to an SVG icon drawn in app/assets/status-*.svg; no emoji anywhere).
STATUS_HE = {"passed": "עבר", "failed": "נכשל", "not_run": "לא רץ"}
# Colors of the status icons (also in the SVG files); each has >= 3:1 on cream and white.
STATUS_COLORS = {"passed": "#3E6B48", "failed": "#6E0F1A", "not_run": "#8A7A6E"}
AGENT_HE = {
    "data_cleaner": "טוען ומנקה את הנתונים הגולמיים",
    "eda_analyst": "מחשב סיכומים ומייצר גרפים ודוח",
    "insights_writer": "כותב את הממצאים מתוך הסיכומים",
    "contract_author": "כותב ומאמת את חוזה הנתונים",
    "contract_validator": "בודק שהנתונים עומדים בחוזה",
    "feature_engineer": "בונה את ה-features למודל",
    "model_trainer": "מאמן ומשווה את המודלים",
    "model_card_writer": "כותב את כרטיס המודל",
}


def flow_steps(summary: dict[str, Any]) -> list[dict[str, str]]:
    """Each Flow step with its status from run_summary.json: passed, failed or not_run."""
    checks, crews = summary.get("checks", {}), summary.get("crews", {})
    failed = summary.get("failed_check")
    steps = []
    for key, label in FLOW_STEPS:
        if key in CREW_DIRS:
            status = "passed" if key in crews else "failed" if failed == key else "not_run"
        elif key in checks:
            status = "passed" if checks[key].get("passed") else "failed"
        else:
            status = "not_run"
        steps.append({"key": key, "label": label, "status": status})
    return steps


def flow_html(steps: list[dict[str, str]]) -> str:
    """The Flow steps as HTML: each step a span with its status class (the icon comes from CSS)
    and the status word for screen readers; arrows between steps."""
    import html as _html

    items = [f'<span class="flow-step step-{s["status"]}" title="{STATUS_HE[s["status"]]}">'
             f'<span class="sr-only">{STATUS_HE[s["status"]]}: </span>{_html.escape(s["label"])}</span>'
             for s in steps]
    return '<div class="flow-line">' + '<span class="flow-arrow">←</span>'.join(items) + "</div>"


def agent_roles(crew: str) -> list[tuple[str, str]]:
    """(agent key, role) for each agent of crew1/crew2, read from its agents.yaml (no crewai)."""
    path = Path(__file__).resolve().parent / "crews" / CREW_DIRS[crew] / "agents.yaml"
    agents = yaml.safe_load(path.read_text(encoding="utf-8"))
    return [(key, spec["role"]) for key, spec in agents.items()]


# --------------------------------------------------------------------------- is it worth paying more?

PRICE_BAND_MIN_WINES = 30          # fewer wines than this: widen the price band to the next level
PRICE_BAND_QUANTILES = (0.05, 0.95)
FLAT_GAIN = 0.5                    # points; less than this up to the top of the band = "almost flat"
FLAT_SPAN = 2.0                    # the flat stretch must cover at least a doubling of the price
RARE_PRICE_NOTE = "המחיר שבחרת נדיר ליין כזה בנתונים, ולכן התחזית פחות בטוחה."


def money(price: float) -> str:
    return f"${price:,.0f}"


def price_band(ranking: pd.DataFrame, country: str, variety: str) -> tuple[float, float, str]:
    """(low, high, level): 5th-95th percentile of prices of the same country and variety in the
    ranking; with fewer than 30 such wines, the variety only; then all wines."""
    levels = [
        ("country_variety", (ranking["country"] == country) & (ranking["variety"] == variety)),
        ("variety", ranking["variety"] == variety),
        ("all", pd.Series(True, index=ranking.index)),
    ]
    for level, mask in levels:
        prices = ranking.loc[mask, "price"].dropna()
        if len(prices) >= PRICE_BAND_MIN_WINES or level == "all":
            low, high = prices.quantile(list(PRICE_BAND_QUANTILES))
            return float(low), float(high), level
    raise AssertionError("unreachable")


def is_rare_price(price: float, band: tuple[float, float, str] | tuple[float, float]) -> bool:
    return not (band[0] <= price <= band[1])


def price_grid(chosen: float, low: float, high: float, n: int = 41) -> list[float]:
    """Log-spaced prices from low to high, plus the chosen price and twice it (when inside the band),
    so the sentence under the chart uses predicted values, not interpolated ones."""
    grid = set(np.round(np.geomspace(low, high, n), 2).tolist())
    grid.update({float(chosen), float(high)})
    if 2 * chosen <= high:
        grid.add(float(2 * chosen))
    return sorted(grid)


def _at(curve: pd.DataFrame, price: float) -> float:
    return float(curve.loc[np.isclose(curve["price"], price), "predicted"].iloc[0])


def flat_from(curve: pd.DataFrame, top: float) -> float | None:
    """Lowest price from which the prediction rises less than FLAT_GAIN up to top,
    over at least a doubling of the price; None when there is no such stretch."""
    inside = curve[curve["price"] <= top + 1e-9].sort_values("price")
    for price, value in zip(inside["price"], inside["predicted"]):
        if price * FLAT_SPAN > top + 1e-9:
            return None
        rest = inside.loc[inside["price"] >= price, "predicted"]
        if rest.max() - value < FLAT_GAIN:
            return float(price)
    return None


def curve_sentence(curve: pd.DataFrame, chosen: float, top: float) -> str | None:
    """One or two sentences computed from the curve; worded as what the model expects, not as cause."""
    flat = flat_from(curve, top)
    flat_text = (f"לפי המודל, מעל {money(flat)} הניקוד הצפוי כמעט לא עולה "
                 f"(פחות מ-{FLAT_GAIN:g} נקודה עד {money(top)}).") if flat is not None else None
    if flat is not None and chosen >= flat:
        return flat_text
    parts = []
    # Compare with twice the price, or with the top of the band when twice is beyond it.
    other = 2 * chosen if 2 * chosen <= top + 1e-9 else top
    if other >= chosen * 1.2:
        gain = _at(curve, other) - _at(curve, chosen)
        where = "" if other == 2 * chosen else " (המחיר הגבוה בטווח הנתונים)"
        if gain >= 0.05:
            parts.append(f"לפי המודל, יין כזה במחיר {money(other)}{where} צפוי לקבל בערך {gain:.1f} נקודות יותר.")
        else:
            parts.append(f"לפי המודל, יין כזה במחיר {money(other)}{where} צפוי לקבל ניקוד כמעט זהה.")
    if flat_text:
        parts.append(flat_text)
    return " ".join(parts) or None


# --------------------------------------------------------------------------- similar wines

SIMILAR_LEVELS = ("country_variety", "variety", "country")


def similar_wines(ranking: pd.DataFrame, country: str, variety: str, price: float,
                  n: int = 5, ratio: float = 1.5) -> tuple[pd.DataFrame, list[str]]:
    """Up to n wines priced within price/ratio..price*ratio, best value_score first: same country
    and variety, then (if fewer than n) same variety anywhere, then same country any variety."""
    in_price = ranking[(ranking["price"] >= price / ratio) & (ranking["price"] <= price * ratio)]
    masks = {
        "country_variety": (in_price["country"] == country) & (in_price["variety"] == variety),
        "variety": in_price["variety"] == variety,
        "country": in_price["country"] == country,
    }
    chosen, levels = [], []
    taken: set = set()
    for level in SIMILAR_LEVELS:
        if len(taken) >= n:
            break
        part = in_price[masks[level] & ~in_price.index.isin(taken)]
        part = part.sort_values(["value_score", "title"], ascending=[False, True], kind="mergesort")
        part = part.head(n - len(taken))
        if len(part):
            chosen.append(part)
            taken.update(part.index)
            levels.append(level)
    result = pd.concat(chosen) if chosen else ranking.head(0)
    return result, levels


def similar_note(levels: list[str], country: str, variety: str, n: int = 5, found: int | None = None) -> str | None:
    """Which widening was used, in words; None when all wines are of the same country and variety."""
    if not levels:
        return "לא נמצאו יינות דומים בטווח המחיר הזה."
    extra = [lvl for lvl in levels if lvl != "country_variety"]
    if not extra:
        return None
    added = {"variety": f"יינות {variety} ממדינות אחרות",
             "country": f"יינות מזנים אחרים מ{COUNTRY_HE.get(country, country)}"}
    return (f"לא נמצאו {n} יינות מאותה מדינה ואותו זן בטווח המחיר, ולכן נוספו "
            + " ואחר כך ".join(added[lvl] for lvl in extra) + ".")


# --------------------------------------------------------------------------- wine card and download

SMALL_VARIETY = 20


def variety_percentile(ranking: pd.DataFrame, variety: str, score: float) -> tuple[int, int]:
    """(percent of wines of this variety with a lower value_score, number of wines of the variety)."""
    scores = ranking.loc[ranking["variety"] == variety, "value_score"]
    if not len(scores):
        return 0, 0
    return int(math.floor((scores < score).mean() * 100)), int(len(scores))   # floor: never "100%"


def variety_rank_sentence(pct: int, variety: str, count: int) -> str:
    text = f"ציון השווי שלו גבוה יותר מ-{pct}% מיינות ה-{variety} ({count:,} יינות)"
    return text + (" (זן עם מעט יינות בנתונים)." if count < SMALL_VARIETY else ".")


CSV_COLUMNS = {
    "rank": "דירוג כללי", "title": "יין", "country": "מדינה", "province": "מחוז", "region_1": "אזור",
    "variety": "זן", "vintage": "בציר", "price": "מחיר ($)", "points": "ניקוד", "label": "דרגה",
    "predicted": "ניקוד צפוי", "value_score": "ציון שווי",
}


def ranking_csv(df: pd.DataFrame) -> bytes:
    """The shown list as a CSV with Hebrew column names (utf-8 with BOM, so Excel shows Hebrew)."""
    out = df.assign(
        country=df["country"].map(country_label),
        vintage=df["vintage"].map(lambda v: "" if pd.isna(v) else str(int(v))),
        label=df["points"].map(score_label),
        predicted=df["predicted"].round(2),
        value_score=df["value_score"].round(2),
    )
    return out[list(CSV_COLUMNS)].rename(columns=CSV_COLUMNS).to_csv(index=False).encode("utf-8-sig")


# --------------------------------------------------------------------------- table display

def short_title(title: Any, max_chars: int = 40) -> str:
    """A long wine name cut at a word boundary with "…", for table cells (the grid cannot wrap
    text). The full name stays in the wine card, the top-3 cards and the CSV."""
    text = str(title)
    if len(text) <= max_chars:
        return text
    cut = text[:max_chars].rsplit(" ", 1)[0].rstrip(" ,-–(")
    return (cut or text[:max_chars]) + "…"


WINE_PROPS = [("country", "מדינה"), ("province", "מחוז"), ("region_1", "אזור"), ("variety", "זן"),
              ("vintage", "בציר")]


def wine_props(wine) -> list[tuple[str, str]]:
    """(Hebrew label, display value) for the wine card's property list; missing values read 'לא ידוע'."""
    out = []
    for key, label in WINE_PROPS:
        value = wine[key]
        if value is None or (not isinstance(value, str) and pd.isna(value)) or value == "":
            text = UNKNOWN
        elif key == "country":
            text = country_label(value)
        elif key == "vintage":
            text = str(int(value))
        else:
            text = str(value)
        out.append((label, text))
    return out


# --------------------------------------------------------------------------- dashboard (KPI tiles and chart data)

# Same headings as eda_tools.INSIGHT_HEADINGS (not imported: it pulls in matplotlib); a test
# checks they stay equal.
INSIGHT_HEADINGS = [
    "## באיזה טווח מחיר העלייה בניקוד מאטה?",
    "## אילו מדינות וזנים נותנים ניקוד גבוה במחיר נמוך?",
    "## היכן הנתונים מוטים?",
    "## המלצות לקנייה",
]

# Chart series colors. Every one has at least 3:1 contrast against the cream page and the white
# cards (tests/test_app_text.py), as graphical objects need. Gold is only for reference lines.
CHART_COLORS = {
    "main": "#6E0F1A",          # deep wine red: lines, bars, highlighted ("value") points
    "other": "#8A7A6E",         # muted grey-brown for the rest (3.8:1 on cream)
    "reference": "#C9A227",     # gold: reference lines only, never a data series
}
# One color per WineEnthusiast band, light to dark as the score rises (all >= 3:1).
SCORE_COLORS = {"סביר": "#9A8478", "טוב": "#B5651D", "טוב מאוד": "#A0522D",
                "מצוין": "#8C4A3C", "יוצא דופן": "#7B2D43", "קלאסי": "#6E0F1A"}


def whole_percent(value: float) -> int:
    """Round half up to a whole percent (73.54 -> 74), the way insights.md shows shares."""
    return int(math.floor(float(value) + 0.5))


def top_countries_share(eda: dict[str, Any], n: int = 3) -> int:
    """Whole percent of wines from the n countries with the most wines (by_country)."""
    rows = sorted(eda["by_country"], key=lambda r: r["count"], reverse=True)[:n]
    return whole_percent(sum(r["share_pct"] for r in rows))


def learned_kpis(eda: dict[str, Any]) -> list[dict[str, str]]:
    """The four KPI tiles of 'מה למדנו', every number from eda_summary.json. Labels are short
    (one or two lines at 375px) and have no parentheses that would mix directions; the unit or
    scale is in the help text."""
    return [
        {"label": "יינות אחרי ניקוי", "value": f"{int(eda['row_count']):,}",
         "help": "מספר היינות שנשארו בנתונים אחרי הסרת כפילויות ושורות בלי מחיר, מדינה או זן."},
        {"label": "מחיר חציוני לבקבוק", "value": money(eda["overall_median_price"]),
         "help": f"בדולרים. {GLOSSARY['חציון']}"},
        {"label": "ניקוד חציוני", "value": f"{float(eda['overall_median_points']):g}",
         "help": f"בסולם של WineEnthusiast, מ-80 עד 100. {GLOSSARY['חציון']}"},
        {"label": "משלוש המדינות המובילות", "value": f"{top_countries_share(eda)}%",
         "help": "אחוז מכלל היינות בנתונים שמגיעים משלוש המדינות עם הכי הרבה יינות."},
    ]


def how_kpis(summary: dict[str, Any], metrics: dict[str, Any] | None) -> list[dict[str, Any]]:
    """KPI tiles of 'איך זה עובד': the last run (run_summary.json) and the model (metrics.json)."""
    passed = summary.get("status") == "passed"
    tiles = [
        {"label": "סטטוס הריצה", "value": "עברה" if passed else "נכשלה",
         "key": "status_passed" if passed else "status_failed"},   # icon added in CSS by key
        {"label": "משך", "value": f"{summary['seconds']:.0f} שניות"},
        {"label": "עלות LLM", "value": f"${summary['total_cost_usd']:.4f}"},
        {"label": "בקשות ל-LLM", "value": f"{int((summary.get('tokens') or {}).get('requests', 0)):,}"},
    ]
    if metrics is not None:
        selected = metrics["models"][metrics["selected"]]["test"]
        baseline = metrics["models"]["baseline"]["test"]
        diff = round(selected["mae"] - baseline["mae"], 2)
        tiles += [
            # delta: compared with the baseline guess; the unit (score points) is in the help
            {"label": "שגיאת המודל", "value": f"{selected['mae']:.2f}",
             "delta": f"{diff:+.2f} מהבסיס", "delta_color": "inverse",
             "help": f"MAE בנקודות ניקוד, בקבוצת הבדיקה. {GLOSSARY['MAE']}"},
            {"label": "מקדם R²", "value": f"{round_for_display(selected['r2']):.2f}",
             "help": "בקבוצת הבדיקה: כמה מהשונות בניקוד המודל מסביר. 0 כמו ניחוש הבסיס, 1 מושלם."},
        ]
    return tiles


def insight_sections(text: str | None) -> list[str] | None:
    """insights.md split at its four fixed headings (each part keeps its heading), or None when
    the headings do not match, so the app can show the whole text as before."""
    if not text:
        return None
    positions = [text.find(h) for h in INSIGHT_HEADINGS]
    if any(p < 0 for p in positions) or positions != sorted(positions):
        return None
    ends = positions[1:] + [len(text)]
    return [text[a:b].strip() for a, b in zip(positions, ends)]


def section_body(section: str) -> str:
    """A section without its '## ' heading line."""
    return section.split("\n", 1)[1].strip() if "\n" in section else ""


def subsection(section: str, heading: str) -> str:
    """The '### heading' part of a section (until the next ### or the end), without the heading."""
    start = section.find(heading)
    if start < 0:
        return ""
    rest = section[start + len(heading):]
    nxt = rest.find("\n### ")
    return (rest if nxt < 0 else rest[:nxt]).strip()


def bias_share_percent(section: str) -> int | None:
    """The top-3 countries share written in the bias line of insights.md ('…ו-74% מגיעים…')."""
    match = re.search(r"(\d+)% מגיעים משלוש המדינות המובילות", section)
    return int(match.group(1)) if match else None


def price_band_data(eda: dict[str, Any]) -> pd.DataFrame:
    bands = eda["price_bands"]["bands"]
    return pd.DataFrame([{"band": b["band"], "geo_mid": b["geo_mid"], "count": int(b["count"]),
                          "median_points": b["median_points"]} for b in bands])


def group_chart_data(rows: list[dict[str, Any]], value_rows: list[dict[str, Any]],
                     hebrew: bool = False) -> pd.DataFrame:
    """One row per country or variety: medians and count, and whether it is on the value list."""
    value_names = {r["name"] for r in value_rows}
    return pd.DataFrame([{
        "name": r["name"],
        "label": COUNTRY_HE.get(r["name"], r["name"]) if hebrew else r["name"],
        "median_price": r["median_price"], "median_points": r["median_points"],
        "count": int(r["count"]), "share_pct": r["share_pct"],
        "group": "שווים" if r["name"] in value_names else "אחרים",
    } for r in rows])


def points_histogram(ranking: pd.DataFrame) -> pd.DataFrame:
    """Number of wines per whole score, with the WineEnthusiast word (a count, nothing more)."""
    counts = ranking["points"].round().astype(int).value_counts().sort_index()
    return pd.DataFrame({"points": counts.index.astype(int), "count": counts.values.astype(int),
                         "label": [score_label(p) for p in counts.index]})


def vintage_data(eda: dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame([{"vintage": int(r["vintage"]), "mean_points": r["mean_points"],
                          "count": int(r["count"])} for r in eda["by_vintage"]])
