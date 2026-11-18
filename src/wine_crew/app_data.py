"""Data helpers for the Streamlit app. Pure functions, testable without Streamlit.

The app only reads artifacts/ and model.pkl: it never runs the Flow and needs
no API key. Form options come from the categories the model learned
(RareCategoryGrouper.keep_), not from the large clean_data.csv.
"""

from __future__ import annotations

import json
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


def load_all(config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Small artifacts the app shows; a missing file is None."""
    config = config if config is not None else load_config()
    sci = resolve_path(config, "artifacts_scientist")
    return {
        "summary": read_json(artifact_path(config, "run_summary")),
        "metrics": read_json(sci / "metrics.json"),
        "contract": read_json(artifact_path(config, "contract")),
        "insights": read_text(artifact_path(config, "insights")),
        "model_card": read_text(sci / "model_card.md"),
        "evaluation_report": read_text(sci / "evaluation_report.md"),
    }


def chart_paths(config: dict[str, Any]) -> dict[str, Path]:
    charts = resolve_path(config, "charts_dir")
    return {n: charts / f"{n}.png" for n in CHART_TITLES_HE if (charts / f"{n}.png").exists()}


def load_model(config: dict[str, Any]):
    path = resolve_path(config, "artifacts_scientist") / "model.pkl"
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
