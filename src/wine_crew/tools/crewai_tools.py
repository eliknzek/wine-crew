"""Thin CrewAI tool wrappers around the deterministic tools.

Each tool takes and returns file paths (never DataFrames) and gives the agent a
short JSON summary. The logic lives in plain run_* functions so it can be
tested without CrewAI or an LLM; the BaseTool classes only serialize.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
from crewai.tools import BaseTool
from pydantic import BaseModel, Field

from wine_crew.config import REPO_ROOT, load_config
from wine_crew.tools.contract_tools import build_contract, save_contract, validate_contract
from wine_crew.tools.data_tools import (
    clean_data,
    load_contract_template,
    load_raw,
    read_clean_csv,
    sha256_file,
)
from wine_crew.tools.eda_tools import build_html_report, eda_summary, make_charts
from wine_crew.tools.model_report import evaluation_report_md, metrics_document
from wine_crew.tools.model_tools import build_features, rank_value_wines, save_model, train_models

# Columns of value_ranking.csv, the full out-of-fold ranking the app filters.
VALUE_RANKING_COLUMNS = ["rank", "title", "country", "province", "region_1", "variety", "vintage",
                         "price", "points", "predicted", "value_score"]

# Tools may read inside ALLOWED_ROOTS and write only inside WRITE_ROOTS, so an
# agent can never overwrite data/raw or source files. Tests add tmp dirs to both.
ALLOWED_ROOTS: list[Path] = [REPO_ROOT]
WRITE_ROOTS: list[Path] = [REPO_ROOT / "artifacts"]


def _resolve(path: str | Path) -> Path:
    p = Path(path)
    if not p.is_absolute():
        p = REPO_ROOT / p
    return p.resolve()


def _safe_path(path: str | Path) -> Path:
    """Resolved path for reading; must be inside ALLOWED_ROOTS."""
    p = _resolve(path)
    if not any(p.is_relative_to(root.resolve()) for root in ALLOWED_ROOTS):
        raise ValueError(f"path outside the allowed directories: {path}")
    return p


def _safe_output_path(path: str | Path) -> Path:
    """Resolved path for writing; must be inside WRITE_ROOTS (artifacts/)."""
    p = _resolve(path)
    if not any(p.is_relative_to(root.resolve()) for root in WRITE_ROOTS):
        raise ValueError(f"output path must be inside artifacts/: {path}")
    return p


def _rel(p: Path) -> str:
    try:
        return str(p.relative_to(REPO_ROOT))
    except ValueError:
        return str(p)


def _config_with_raw(config: dict[str, Any], raw: Path) -> dict[str, Any]:
    cfg = dict(config)
    cfg["paths"] = {**config["paths"], "raw_csv": str(raw)}
    return cfg


# --------------------------------------------------------------------------- plain functions


def run_load_raw(config: dict[str, Any], raw_csv_path: str) -> dict[str, Any]:
    raw = _safe_path(raw_csv_path)
    df, sha = load_raw(_config_with_raw(config, raw))
    required = config["cleaning"]["required_columns"]
    return {
        "path": _rel(raw),
        "rows": int(len(df)),
        "columns": list(df.columns),
        "sha256": sha,
        "nulls_in_required_columns": {c: int(df[c].isna().sum()) for c in required},
        "fully_duplicated_rows": int(df.duplicated().sum()),
    }


def run_clean_data(config: dict[str, Any], raw_csv_path: str, output_csv_path: str) -> dict[str, Any]:
    raw, out = _safe_path(raw_csv_path), _safe_output_path(output_csv_path)
    df, _ = load_raw(_config_with_raw(config, raw))
    clean, report = clean_data(df, config)
    out.parent.mkdir(parents=True, exist_ok=True)
    clean.to_csv(out, index=False)
    return {"output_csv": _rel(out), **report}


def run_eda_summary(config: dict[str, Any], clean_csv_path: str, output_json_path: str) -> dict[str, Any]:
    clean_path, out = _safe_path(clean_csv_path), _safe_output_path(output_json_path)
    df = read_clean_csv(clean_path, load_contract_template(config))
    summary = eda_summary(df, config)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return {
        "output_json": _rel(out),
        "row_count": summary["row_count"],
        "mean_points": summary["points"]["mean"],
        "median_price": summary["price"]["median"],
        "corr_points_log_price": summary["corr_points_log_price"],
        "plateau_starts_at_band": summary["price_bands"]["plateau_starts_at_band"],
        "top_country_share_pct": summary["country_bias"]["top_country_share_pct"],
    }


def run_make_charts(config: dict[str, Any], clean_csv_path: str, charts_dir: str) -> dict[str, Any]:
    clean_path, out_dir = _safe_path(clean_csv_path), _safe_output_path(charts_dir)
    df = read_clean_csv(clean_path, load_contract_template(config))
    paths = make_charts(df, out_dir, config)
    return {"charts_dir": _rel(out_dir), "charts": [p.name for p in paths.values()]}


def run_build_html_report(summary_json_path: str, charts_dir: str, output_html_path: str) -> dict[str, Any]:
    summary_path, chart_dir, out = (
        _safe_path(summary_json_path), _safe_path(charts_dir), _safe_output_path(output_html_path),
    )
    from wine_crew.tools.eda_tools import CHART_NAMES

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    charts = {n: chart_dir / f"{n}.png" for n in CHART_NAMES if (chart_dir / f"{n}.png").exists()}
    path = build_html_report(summary, charts, out)
    return {"output_html": _rel(path), "charts_embedded": len(charts), "size_kb": path.stat().st_size // 1024}


def run_read_eda_summary(summary_json_path: str) -> dict[str, Any]:
    return json.loads(_safe_path(summary_json_path).read_text(encoding="utf-8"))


def run_build_contract(
    config: dict[str, Any], clean_csv_path: str, raw_csv_path: str, output_json_path: str
) -> dict[str, Any]:
    clean_path, raw, out = (
        _safe_path(clean_csv_path), _safe_path(raw_csv_path), _safe_output_path(output_json_path)
    )
    df = read_clean_csv(clean_path, load_contract_template(config))
    contract = build_contract(df, sha256_file(raw), config)
    save_contract(contract, out)
    return {"output_json": _rel(out), "sha256": contract["source"]["sha256"],
            "row_count": contract["observed"]["row_count"]}


def run_validate_contract(
    config: dict[str, Any], clean_csv_path: str, contract_json_path: str,
    output_json_path: str | None = None,
) -> dict[str, Any]:
    """Validate; with output_json_path also write the result (Crew 2's contract_check.json)."""
    clean_path, contract_path = _safe_path(clean_csv_path), _safe_path(contract_json_path)
    out = _safe_output_path(output_json_path) if output_json_path else None
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    violations = validate_contract(read_clean_csv(clean_path, contract), contract)
    result = {"passed": not violations, "violations": violations}
    if out is not None:
        _write_json(out, result)
        result = {**result, "output_json": _rel(out)}
    return result


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _load_clean_and_contract(clean_csv_path: str, contract_json_path: str):
    contract = json.loads(_safe_path(contract_json_path).read_text(encoding="utf-8"))
    return read_clean_csv(_safe_path(clean_csv_path), contract), contract


def run_build_features(
    config: dict[str, Any], clean_csv_path: str, contract_json_path: str, output_csv_path: str
) -> dict[str, Any]:
    """v2 features (the superset of v1). A forbidden column returns passed=false and writes nothing."""
    out = _safe_output_path(output_csv_path)
    df, contract = _load_clean_and_contract(clean_csv_path, contract_json_path)
    try:
        features = build_features(df, contract, config, version="v2")
    except ValueError as exc:
        return {"passed": False, "violations": str(exc).split("; ")}
    out.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(out, index=False)
    return {"passed": True, "output_csv": _rel(out), "rows": int(len(features)),
            "columns": list(features.columns)}


def run_train_models(
    config: dict[str, Any], clean_csv_path: str, contract_json_path: str,
    model_path: str, metrics_json_path: str, report_md_path: str, ranking_csv_path: str,
) -> dict[str, Any]:
    """Train baseline/v1/v2, rank every wine out-of-fold with the selected version, and write
    model.pkl, metrics.json, evaluation_report.md (top value_top_n) and value_ranking.csv (all wines)."""
    model_out, metrics_out, report_out, ranking_out = (
        _safe_output_path(model_path), _safe_output_path(metrics_json_path),
        _safe_output_path(report_md_path), _safe_output_path(ranking_csv_path),
    )
    df, contract = _load_clean_and_contract(clean_csv_path, contract_json_path)
    result = train_models(df, contract, config)
    metrics = metrics_document(result, df, contract, config)
    ranking = rank_value_wines(df, config, result["selected"])
    save_model(result["models"][result["selected"]], model_out)
    _write_json(metrics_out, metrics)
    report_out.parent.mkdir(parents=True, exist_ok=True)
    report_out.write_text(evaluation_report_md(metrics, ranking.head(config["model"]["value_top_n"])),
                          encoding="utf-8")
    full = ranking.assign(rank=range(1, len(ranking) + 1),
                          vintage=pd.to_numeric(ranking["vintage"]).astype("Int64"),
                          predicted=ranking["predicted"].round(4), value_score=ranking["value_score"].round(4))
    full[VALUE_RANKING_COLUMNS].to_csv(ranking_out, index=False)
    return {
        "selected": metrics["selected"],
        "test_mae": {n: m["test"]["mae"] for n, m in metrics["models"].items()},
        "model_path": _rel(model_out), "metrics_json": _rel(metrics_out), "report_md": _rel(report_out),
        "ranking_csv": _rel(ranking_out), "ranked_wines": int(len(full)),
    }


def run_read_metrics(metrics_json_path: str) -> dict[str, Any]:
    return json.loads(_safe_path(metrics_json_path).read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- CrewAI tools


class _RawInput(BaseModel):
    raw_csv_path: str = Field(..., description="Path to the raw CSV, relative to the repo root")


class _CleanInput(BaseModel):
    raw_csv_path: str = Field(..., description="Path to the raw CSV")
    output_csv_path: str = Field(..., description="Where to write clean_data.csv")


class _SummaryInput(BaseModel):
    clean_csv_path: str = Field(..., description="Path to clean_data.csv")
    output_json_path: str = Field(..., description="Where to write eda_summary.json")


class _ChartsInput(BaseModel):
    clean_csv_path: str = Field(..., description="Path to clean_data.csv")
    charts_dir: str = Field(..., description="Directory for the PNG charts")


class _ReportInput(BaseModel):
    summary_json_path: str = Field(..., description="Path to eda_summary.json")
    charts_dir: str = Field(..., description="Directory with the PNG charts")
    output_html_path: str = Field(..., description="Where to write eda_report.html")


class _ReadSummaryInput(BaseModel):
    summary_json_path: str = Field(..., description="Path to eda_summary.json")


class _BuildContractInput(BaseModel):
    clean_csv_path: str = Field(..., description="Path to clean_data.csv")
    raw_csv_path: str = Field(..., description="Path to the raw CSV (for its sha256)")
    output_json_path: str = Field(..., description="Where to write dataset_contract.json")


class _ValidateInput(BaseModel):
    clean_csv_path: str = Field(..., description="Path to clean_data.csv")
    contract_json_path: str = Field(..., description="Path to dataset_contract.json")
    output_json_path: str | None = Field(None, description="Optional: where to write the result JSON")


class _FeaturesInput(BaseModel):
    clean_csv_path: str = Field(..., description="Path to clean_data.csv")
    contract_json_path: str = Field(..., description="Path to dataset_contract.json")
    output_csv_path: str = Field(..., description="Where to write features.csv")


class _TrainInput(BaseModel):
    clean_csv_path: str = Field(..., description="Path to clean_data.csv")
    contract_json_path: str = Field(..., description="Path to dataset_contract.json")
    model_path: str = Field(..., description="Where to write model.pkl")
    metrics_json_path: str = Field(..., description="Where to write metrics.json")
    report_md_path: str = Field(..., description="Where to write evaluation_report.md")
    ranking_csv_path: str = Field(..., description="Where to write value_ranking.csv")


class _ReadMetricsInput(BaseModel):
    metrics_json_path: str = Field(..., description="Path to metrics.json")


class _ConfiguredTool(BaseTool):
    config: dict[str, Any] = Field(default_factory=load_config, exclude=True)


def _dump(result: dict[str, Any]) -> str:
    return json.dumps(result, ensure_ascii=False)


class LoadRawTool(_ConfiguredTool):
    name: str = "load_raw"
    description: str = "Load the raw wine CSV and report rows, columns, sha256, duplicates and missing required values."
    args_schema: type[BaseModel] = _RawInput

    def _run(self, raw_csv_path: str) -> str:
        return _dump(run_load_raw(self.config, raw_csv_path))


class CleanDataTool(_ConfiguredTool):
    name: str = "clean_data"
    description: str = "Apply the fixed cleaning rules to the raw CSV, write clean_data.csv, and report rows removed per rule."
    args_schema: type[BaseModel] = _CleanInput

    def _run(self, raw_csv_path: str, output_csv_path: str) -> str:
        return _dump(run_clean_data(self.config, raw_csv_path, output_csv_path))


class EdaSummaryTool(_ConfiguredTool):
    name: str = "eda_summary"
    description: str = "Compute the EDA statistics from clean_data.csv, write them to a JSON file, and return headline numbers."
    args_schema: type[BaseModel] = _SummaryInput

    def _run(self, clean_csv_path: str, output_json_path: str) -> str:
        return _dump(run_eda_summary(self.config, clean_csv_path, output_json_path))


class MakeChartsTool(_ConfiguredTool):
    name: str = "make_charts"
    description: str = "Draw the EDA charts from clean_data.csv as PNG files."
    args_schema: type[BaseModel] = _ChartsInput

    def _run(self, clean_csv_path: str, charts_dir: str) -> str:
        return _dump(run_make_charts(self.config, clean_csv_path, charts_dir))


class BuildHtmlReportTool(BaseTool):
    name: str = "build_html_report"
    description: str = "Build a self-contained eda_report.html from eda_summary.json and the chart PNGs."
    args_schema: type[BaseModel] = _ReportInput

    def _run(self, summary_json_path: str, charts_dir: str, output_html_path: str) -> str:
        return _dump(run_build_html_report(summary_json_path, charts_dir, output_html_path))


class ReadEdaSummaryTool(BaseTool):
    name: str = "read_eda_summary"
    description: str = "Read the full EDA summary JSON. These are the only numbers you may quote."
    args_schema: type[BaseModel] = _ReadSummaryInput

    def _run(self, summary_json_path: str) -> str:
        return _dump(run_read_eda_summary(summary_json_path))


class BuildContractTool(_ConfiguredTool):
    name: str = "build_contract"
    description: str = "Build dataset_contract.json from the fixed template, the clean data and the raw file's sha256."
    args_schema: type[BaseModel] = _BuildContractInput

    def _run(self, clean_csv_path: str, raw_csv_path: str, output_json_path: str) -> str:
        return _dump(run_build_contract(self.config, clean_csv_path, raw_csv_path, output_json_path))


class ValidateContractTool(_ConfiguredTool):
    name: str = "validate_contract"
    description: str = "Check clean_data.csv against dataset_contract.json; returns passed and the list of violations."
    args_schema: type[BaseModel] = _ValidateInput

    def _run(self, clean_csv_path: str, contract_json_path: str, output_json_path: str | None = None) -> str:
        return _dump(run_validate_contract(self.config, clean_csv_path, contract_json_path, output_json_path))


class BuildFeaturesTool(_ConfiguredTool):
    name: str = "build_features"
    description: str = ("Build the model features from clean_data.csv using only columns the contract "
                        "allows, and write features.csv; returns passed=false with violations otherwise.")
    args_schema: type[BaseModel] = _FeaturesInput

    def _run(self, clean_csv_path: str, contract_json_path: str, output_csv_path: str) -> str:
        return _dump(run_build_features(self.config, clean_csv_path, contract_json_path, output_csv_path))


class TrainModelsTool(_ConfiguredTool):
    name: str = "train_models"
    description: str = ("Train and evaluate baseline, v1 (Ridge) and v2 (HistGradientBoosting) with the fixed "
                        "split and CV, select the best by CV MAE, rank value wines out-of-fold, and write "
                        "model.pkl, metrics.json, evaluation_report.md and value_ranking.csv.")
    args_schema: type[BaseModel] = _TrainInput

    def _run(self, clean_csv_path: str, contract_json_path: str, model_path: str,
             metrics_json_path: str, report_md_path: str, ranking_csv_path: str) -> str:
        return _dump(run_train_models(self.config, clean_csv_path, contract_json_path,
                                      model_path, metrics_json_path, report_md_path, ranking_csv_path))


class ReadMetricsTool(BaseTool):
    name: str = "read_metrics"
    description: str = "Read the full metrics JSON. These are the only numbers you may quote."
    args_schema: type[BaseModel] = _ReadMetricsInput

    def _run(self, metrics_json_path: str) -> str:
        return _dump(run_read_metrics(metrics_json_path))
