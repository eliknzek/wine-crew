"""Crew 2 reports built in code: the metrics JSON, evaluation_report.md, and the
deterministic style check for model_card.md.

The metrics JSON is the only source the Model Card Writer may quote, so every
number the card needs (including the limitation facts: data year, top-country
share, minimum points, the region_1 category cap) is put there by code.
"""

from __future__ import annotations

import re
from typing import Any

import pandas as pd

from wine_crew.tools.eda_tools import _LONG_PERCENT, _TRAILING_ZERO, _section

METRIC_KEYS = ("mae", "rmse", "r2")
# Hebrew names for countries the card may mention; others stay in English.
COUNTRY_HE = {"US": 'ארה"ב'}
MODEL_CARD_HEADINGS = [
    "## מטרה",
    "## נתונים",
    "## מדדים",
    "## מגבלות",
    "## היבטים אתיים",
]
# The Model Card Writer writes this line; run_scientist replaces it with metrics_table_md.
METRICS_TABLE_PLACEHOLDER = "{{METRICS_TABLE}}"
ETHICS_PHRASES = ["תמיכה בהחלטה", "לא להחליף טעימה"]
GUARDRAIL_LIMIT_PHRASE = "אבל לא שהוא משויך לטענה הנכונה"


def _r(x: float, nd: int = 4) -> float:
    return round(float(x), nd)


def _round_metrics(m: dict[str, float]) -> dict[str, float]:
    return {k: _r(v) for k, v in m.items()}


def metrics_document(
    result: dict[str, Any], clean_df: pd.DataFrame, contract: dict[str, Any], config: dict[str, Any]
) -> dict[str, Any]:
    """The metrics JSON for one train_models result. Deterministic; values rounded to 4 decimals."""
    metrics = result["metrics"]
    selected = result["selected"]
    roles = {c["name"]: c["role"] for c in contract["columns"]}
    shares = clean_df["country"].astype(str).value_counts(normalize=True)
    top_country = str(shares.index[0])
    test_size = config["model"]["test_size"]
    base_mae, sel_mae = metrics["baseline"]["test"]["mae"], metrics[selected]["test"]["mae"]
    return {
        "target": contract["target"],
        "selected": selected,
        "selection_metric": config["model"]["selection_metric"],
        "models": {
            name: {"cv": _round_metrics(m["cv"]), "test": _round_metrics(m["test"])}
            for name, m in metrics.items()
        },
        "selected_test_mae_reduction_vs_baseline_pct": _r(100 * (base_mae - sel_mae) / base_mae),
        "data": {
            "source": config["data_source"]["kaggle_dataset"],
            "collected_year": int(config["data_source"]["collected_year"]),
            "rows": int(len(clean_df)),
            "n_train": result["n_train"],
            "n_test": result["n_test"],
            "train_pct": int(round(100 * (1 - test_size))),
            "test_pct": int(round(100 * test_size)),
            "cv_folds": int(config["model"]["cv_folds"]),
            "seed": int(config["seed"]),
            "points_min": int(clean_df[contract["target"]].min()),
            "points_max": int(clean_df[contract["target"]].max()),
            "top_country": top_country,
            "top_country_he": COUNTRY_HE.get(top_country, top_country),
            "top_country_share_pct": _r(100 * shares.iloc[0]),
        },
        "features": {
            "input_columns": result["input_columns"],
            "leakage_columns": [c for c, r in roles.items() if r == "leakage"],
            "excluded_columns": [c for c, r in roles.items() if r == "excluded"],
            "region_1_max_categories": int(config["model"]["hgb"]["max_categories"]),
            "min_category_count": int(config["features"]["min_category_count"]),
        },
        "value_ranking": {"top_n": int(config["model"]["value_top_n"]), "method": "out-of-fold"},
    }


# --------------------------------------------------------------------------- evaluation report


def _cell(value: Any) -> str:
    if value is None or (not isinstance(value, str) and pd.isna(value)):
        return ""
    return str(value).replace("|", "\\|").replace("\n", " ")


def evaluation_report_md(metrics: dict[str, Any], ranking: pd.DataFrame) -> str:
    """Markdown with the metrics table and the value-wine ranking; written by code, not the LLM."""
    selected = metrics["selected"]
    data = metrics["data"]
    lines = [
        "# דוח הערכת מודלים",
        "",
        f"משתנה המטרה: `{metrics['target']}`. חלוקה {data['train_pct']}/{data['test_pct']} "
        f"({data['n_train']:,} אימון, {data['n_test']:,} בדיקה), {data['cv_folds']}-fold CV על קבוצת "
        f"האימון, seed {data['seed']}. המודל נבחר לפי MAE ממוצע ב-CV.",
        "",
        "## מדדים",
        "",
        "| מודל | CV MAE | CV MAE std | MAE (test) | RMSE (test) | R² (test) |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for name, m in metrics["models"].items():
        label = f"**{name}** (נבחר)" if name == selected else name
        lines.append(f"| {label} | {m['cv']['mae']:.3f} | {m['cv']['mae_std']:.3f} | "
                     f"{m['test']['mae']:.3f} | {m['test']['rmse']:.3f} | {m['test']['r2']:.3f} |")
    lines += [
        "",
        f"## {len(ranking)} היינות המובילים בדירוג השווים למחירם",
        "",
        f"value_score = ניקוד בפועל פחות הניקוד שחזה {selected}, עם תחזית out-of-fold "
        "(cross_val_predict), כך שאף יין לא נחזה ע\"י מודל שראה אותו.",
        "",
        "| # | title | country | variety | vintage | price | points | predicted | value_score |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for i, row in enumerate(ranking.itertuples(index=False), start=1):
        lines.append(
            f"| {i} | {_cell(row.title)} | {_cell(row.country)} | {_cell(row.variety)} | "
            f"{_cell(row.vintage)} | ${row.price:g} | {row.points} | {row.predicted:.2f} | "
            f"{row.value_score:.2f} |"
        )
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- model card table


def _fmt2(x: float) -> str:
    """Two decimals, never "-0.00" (a tiny negative such as R² -0.0001 prints as 0.00)."""
    text = f"{x:.2f}"
    return "0.00" if text == "-0.00" else text


def metrics_table_md(metrics: dict[str, Any]) -> str:
    """The model card's metrics table, built from metrics.json in code."""
    lines = [
        "| מודל | CV MAE | MAE (test) | RMSE (test) | R² (test) |",
        "| --- | --- | --- | --- | --- |",
    ]
    for name, m in metrics["models"].items():
        cells = [m["cv"]["mae"], m["test"]["mae"], m["test"]["rmse"], m["test"]["r2"]]
        lines.append(f"| {name} | " + " | ".join(_fmt2(v) for v in cells) + " |")
    return "\n".join(lines)


def fill_metrics_table(text: str, metrics: dict[str, Any]) -> str:
    """Replace the single placeholder line with the table. Raises if it is not there exactly once."""
    count = text.count(METRICS_TABLE_PLACEHOLDER)
    if count != 1:
        raise ValueError(f"{METRICS_TABLE_PLACEHOLDER} must appear exactly once, found {count}")
    # Exactly one blank line before and after: without the one after, GitHub
    # renders the next line as part of the table.
    pattern = re.compile(r"\n*[ \t]*" + re.escape(METRICS_TABLE_PLACEHOLDER) + r"[ \t]*\n*")
    return pattern.sub(lambda _: "\n\n" + metrics_table_md(metrics) + "\n\n", text)


# --------------------------------------------------------------------------- model card style


def _limitations_required(metrics: dict[str, Any]) -> list[tuple[str, list[str]]]:
    d, f = metrics["data"], metrics["features"]
    return [
        ("data year and single source", [str(d["collected_year"]), "ממקור אחד"]),
        ("top-country share", [f"{round(d['top_country_share_pct'])}%", d["top_country_he"], "אחרי ניקוי"]),
        ("minimum points", [f"{d['points_min']} נקודות ומעלה"]),
        ("subjective scores", ["סובייקטיבי"]),
        ("leakage columns", list(f["leakage_columns"])),
        ("vintage known before tasting", ["vintage", "ידוע לפני הטעימה"]),
        ("region_1 category cap", ["region_1", str(f["region_1_max_categories"])]),
        ("number guardrail limit", [GUARDRAIL_LIMIT_PHRASE]),
    ]


def _normalize(text: str) -> str:
    # Hebrew gershayim and markdown emphasis should not decide a match.
    return text.replace("״", '"').replace("**", "").replace("`", "")


def model_card_style_violations(text: str, metrics: dict[str, Any]) -> list[str]:
    """Deterministic rules for model_card.md: headings, rounding, required limitations and ethics."""
    text = _normalize(text)
    out: list[str] = []
    positions = [text.find(h) for h in MODEL_CARD_HEADINGS]
    for heading, pos in zip(MODEL_CARD_HEADINGS, positions):
        if pos < 0:
            out.append(f"missing heading: {heading}")
    found = [p for p in positions if p >= 0]
    if found != sorted(found):
        out.append("headings are not in the required order")
    for token in dict.fromkeys(_TRAILING_ZERO.findall(text)):
        out.append(f"write {token} as a whole number ({token[:-2]})")
    for token in dict.fromkeys(_LONG_PERCENT.findall(text)):
        out.append(f"round the percentage {token.strip()} to a whole number")

    metrics_section = _section(text, "## מדדים")
    count = text.count(METRICS_TABLE_PLACEHOLDER)
    if count != 1:
        out.append(f"write the line {METRICS_TABLE_PLACEHOLDER} exactly once, in '## מדדים' "
                   f"(found {count}); do not write the metrics table yourself")
    elif METRICS_TABLE_PLACEHOLDER not in metrics_section:
        out.append(f"the line {METRICS_TABLE_PLACEHOLDER} must be inside '## מדדים'")
    if any(line.lstrip().startswith("|") for line in metrics_section.splitlines()):
        out.append(f"do not write a metrics table; the code inserts it at {METRICS_TABLE_PLACEHOLDER}")
    selected_line = f"המודל שנבחר: {metrics['selected']}"
    if selected_line not in metrics_section:
        out.append(f"section '## מדדים' must include the line '{selected_line}, ...'")

    limits = _section(text, "## מגבלות")
    for label, phrases in _limitations_required(metrics):
        missing = [p for p in phrases if p not in limits]
        if missing:
            out.append(f"section '## מגבלות' must include the {label} line from the template "
                       f"(missing: {', '.join(missing)})")

    ethics = _section(text, "## היבטים אתיים")
    missing = [p for p in ETHICS_PHRASES if p not in ethics]
    if missing:
        out.append("section '## היבטים אתיים' must include the line "
                   f"'המודל נועד לתמיכה בהחלטה ולא להחליף טעימה' (missing: {', '.join(missing)})")
    return out
