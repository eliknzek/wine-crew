"""Deterministic Flow gate checks. A failure raises FlowCheckError.

The gates read the files themselves and never trust an agent's report. Each
returns a small dict of details for the run summary.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from wine_crew.config import REPO_ROOT, resolve_path
from wine_crew.tools.contract_tools import validate_contract
from wine_crew.tools.data_tools import read_clean_csv, sha256_file
from wine_crew.tools.model_tools import (
    DERIVED_NUMERIC,
    feature_violations,
    input_columns,
    load_model,
)


class FlowCheckError(Exception):
    """A gate failed. `check` names the gate; str() is the user-facing message."""

    def __init__(self, check: str, message: str, details: dict[str, Any] | None = None):
        super().__init__(message)
        self.check = check
        self.message = message
        self.details = details or {}


def _rel(path: Path) -> str:
    try:
        return str(Path(path).resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _read_json(path: Path, check: str, label: str) -> dict[str, Any]:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise FlowCheckError(check, f"{label}: {_rel(path)} not found") from None
    except (OSError, ValueError) as exc:
        raise FlowCheckError(check, f"{label}: {_rel(path)} could not be read ({exc})") from None


def _read_csv(path: Path, check: str, label: str, contract: dict[str, Any] | None = None) -> pd.DataFrame:
    try:
        return read_clean_csv(path, contract) if contract is not None else pd.read_csv(path)
    except FileNotFoundError:
        raise FlowCheckError(check, f"{label}: {_rel(path)} not found") from None
    except Exception as exc:  # noqa: BLE001 - any unreadable file is a gate failure, not a crash
        raise FlowCheckError(check, f"{label}: {_rel(path)} could not be read ({exc})") from None


# --------------------------------------------------------------------------- preflight


def check_preflight(config: dict[str, Any]) -> dict[str, Any]:
    """The raw file exists; returns its sha256."""
    raw = resolve_path(config, "raw_csv")
    if not raw.is_file():
        raise FlowCheckError(
            "preflight",
            f"Preflight check failed: raw file not found: {_rel(raw)} "
            "(run `python scripts/download_data.py`)",
        )
    return {"raw_csv": _rel(raw), "sha256": sha256_file(raw)}


# --------------------------------------------------------------------------- gate 1


def check_contract(clean_csv: Path, contract_json: Path) -> dict[str, Any]:
    """Gate 1: clean_data.csv conforms to dataset_contract.json."""
    label = "Contract check failed"
    contract = _read_json(contract_json, "contract", label)
    df = _read_csv(clean_csv, "contract", label, contract)
    violations = validate_contract(df, contract)
    if violations:
        raise FlowCheckError("contract", f"{label}: " + "; ".join(violations),
                             {"violations": violations})
    return {"rows": int(len(df)), "sha256": contract.get("source", {}).get("sha256")}


# --------------------------------------------------------------------------- gate 2


def expected_feature_columns(config: dict[str, Any]) -> list[str]:
    """Columns build_features writes for v2 (the superset of v1)."""
    raw = input_columns(config, "v2")
    return DERIVED_NUMERIC + [c for c in raw if c not in ("price", "vintage")]


def check_features(features_csv: Path, clean_csv: Path, contract: dict[str, Any],
                   config: dict[str, Any]) -> dict[str, Any]:
    """Gate 2: features.csv is complete, leak-free, has no disallowed NaN, and matches clean_data rows.

    NaN is allowed only where the contract allows a missing value: vintage when
    vintage_missing is 1, and feature_optional columns up to their max_null_pct.
    """
    label = "Feature check failed"
    df = _read_csv(features_csv, "features", label)
    problems: list[str] = []

    missing = [c for c in expected_feature_columns(config) if c not in df.columns]
    problems += [f"missing feature column: {c}" for c in missing]
    problems += [v.removeprefix(f"{label}: ") for v in feature_violations(list(df.columns), contract)]

    specs = {c["name"]: c for c in contract["columns"]}
    for col in df.columns:
        if col in missing or col not in specs and col not in DERIVED_NUMERIC:
            continue
        values = df[col]
        nulls = values.isna()
        if col == "vintage":
            flag = df["vintage_missing"] if "vintage_missing" in df.columns else 0
            bad = int((nulls & (flag != 1)).sum())
            if bad:
                problems.append(f"vintage has {bad} NaN where vintage_missing is not 1")
            continue
        if pd.api.types.is_numeric_dtype(values):
            nulls = nulls | ~np.isfinite(values.astype("float64").fillna(0))
        spec = specs.get(col, {})
        if spec.get("role") == "feature_optional":
            pct = 100 * nulls.mean() if len(df) else 0.0
            limit = spec.get("max_null_pct", 0)
            if pct > limit:
                problems.append(f"{col} has {pct:.1f}% NaN (allowed {limit}%)")
        elif nulls.any():
            problems.append(f"{col} has {int(nulls.sum())} NaN (allowed 0)")

    clean_rows = len(_read_csv(clean_csv, "features", label))
    if len(df) != clean_rows:
        problems.append(f"features.csv has {len(df)} rows, clean_data has {clean_rows}")

    if problems:
        raise FlowCheckError("features", f"{label}: " + "; ".join(problems), {"problems": problems})
    return {"rows": int(len(df)), "columns": list(df.columns)}


# --------------------------------------------------------------------------- gate 3


def check_model(metrics_json: Path, model_pkl: Path, clean_csv: Path,
                contract: dict[str, Any]) -> dict[str, Any]:
    """Gate 3: the selected model beats the baseline on test MAE, and model.pkl loads and predicts."""
    label = "Model check failed"
    metrics = _read_json(metrics_json, "model", label)
    try:
        selected = metrics["selected"]
        mae = float(metrics["models"][selected]["test"]["mae"])
        baseline = float(metrics["models"]["baseline"]["test"]["mae"])
        columns = metrics["features"]["input_columns"][selected]
    except (KeyError, TypeError, ValueError) as exc:
        raise FlowCheckError("model", f"{label}: metrics.json is missing {exc}") from None
    if not mae < baseline:
        raise FlowCheckError("model", f"{label}: MAE {mae:.2f} >= baseline {baseline:.2f}",
                             {"mae": mae, "baseline_mae": baseline})

    sample = _read_csv(clean_csv, "model", label, contract).head(5)
    try:
        model = load_model(model_pkl)
    except Exception as exc:  # noqa: BLE001
        raise FlowCheckError("model", f"{label}: model.pkl could not be loaded ({type(exc).__name__}: {exc})") from None
    try:
        pred = np.asarray(model.predict(sample[columns]), dtype="float64")
    except Exception as exc:  # noqa: BLE001
        raise FlowCheckError("model", f"{label}: model.pkl could not predict ({type(exc).__name__}: {exc})") from None
    if pred.shape != (len(sample),) or not np.isfinite(pred).all():
        raise FlowCheckError("model", f"{label}: model.pkl returned invalid predictions {pred.tolist()}")
    return {"selected": selected, "mae": mae, "baseline_mae": baseline,
            "sample_predictions": [round(float(p), 2) for p in pred]}
