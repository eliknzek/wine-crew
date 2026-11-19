"""Data helpers for the Streamlit app. Pure functions, testable without Streamlit.

The app only reads artifacts/ and model.pkl: it never runs the Flow and needs
no API key. Form options come from the categories the model learned
(RareCategoryGrouper.keep_), not from the large clean_data.csv.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

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


def predict_points(model, metrics: dict[str, Any], row: dict[str, Any]) -> float:
    """Predicted points for one wine. vintage or region_1 of None / UNKNOWN become NaN."""
    columns = metrics["features"]["input_columns"][metrics["selected"]]
    values = {}
    for col in columns:
        value = row.get(col)
        if value is None or value == UNKNOWN:
            value = np.nan
        values[col] = [value]
    frame = pd.DataFrame(values, columns=columns)
    frame["price"] = frame["price"].astype("float64")
    frame["vintage"] = pd.to_numeric(frame["vintage"]).astype("float64")
    return float(model.predict(frame)[0])


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
