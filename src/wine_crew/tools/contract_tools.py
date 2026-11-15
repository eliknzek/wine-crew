"""Dataset contract: build_contract, save_contract, validate_contract.

The contract's ranges and roles come from the fixed template
(config paths.contract_template), never from the LLM or from the data.
build_contract only fills in the source fingerprint and observed statistics.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from wine_crew.tools.data_tools import load_contract_template


def _observed(df: pd.DataFrame) -> dict[str, Any]:
    columns: dict[str, Any] = {}
    for name in df.columns:
        s = df[name]
        stats: dict[str, Any] = {
            "null_pct": round(float(s.isna().mean() * 100), 4) if len(s) else 0.0,
            "n_unique": int(s.nunique(dropna=True)),
        }
        if pd.api.types.is_numeric_dtype(s) and s.notna().any():
            stats["min"] = float(s.min())
            stats["max"] = float(s.max())
        columns[name] = stats
    return {"row_count": int(len(df)), "columns": columns}


def build_contract(clean_df: pd.DataFrame, sha256: str, config: dict[str, Any]) -> dict[str, Any]:
    """Contract = fixed template + source sha256 + observed statistics. Deterministic."""
    contract = copy.deepcopy(load_contract_template(config))
    contract["source"]["sha256"] = sha256
    contract["observed"] = _observed(clean_df)
    return contract


def save_contract(contract: dict[str, Any], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(contract, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


_DTYPE_CHECKS: dict[str, Callable[[pd.Series], bool]] = {
    "int64": lambda s: s.dtype == "int64",
    "float64": lambda s: s.dtype == "float64",
    "Int64": lambda s: isinstance(s.dtype, pd.Int64Dtype),
    "category": lambda s: isinstance(s.dtype, pd.CategoricalDtype),
    "string": lambda s: pd.api.types.is_string_dtype(s) and not isinstance(s.dtype, pd.CategoricalDtype),
}


def _check_columns(df: pd.DataFrame, contract: dict[str, Any]) -> list[str]:
    expected = [c["name"] for c in contract["columns"]]
    out = [f"missing column: {c}" for c in expected if c not in df.columns]
    out += [f"unexpected column: {c}" for c in df.columns if c not in expected]
    return out


def _check_column(s: pd.Series, spec: dict[str, Any]) -> list[str]:
    name = spec["name"]
    out: list[str] = []

    dtype = spec.get("dtype")
    check = _DTYPE_CHECKS.get(dtype)
    if check is not None and not check(s):
        out.append(f"{name} has dtype {s.dtype} (expected {dtype})")

    nulls = int(s.isna().sum())
    if not spec.get("nullable", True) and nulls:
        out.append(f"{name} has {nulls} nulls (allowed 0)")
    max_null_pct = spec.get("max_null_pct")
    if max_null_pct is not None and len(s):
        pct = nulls / len(s) * 100
        if pct > max_null_pct:
            out.append(f"{name} has {pct:.2f}% nulls (allowed {max_null_pct}%)")

    if "min" in spec or "max" in spec:
        values = pd.to_numeric(s, errors="coerce")
        non_numeric = int((values.isna() & s.notna()).sum())
        if non_numeric:
            out.append(f"{name} has {non_numeric} non-numeric values")
        if "min" in spec:
            below = int((values < spec["min"]).sum())
            if below:
                out.append(f"{name}: {below} values below min {spec['min']}")
        if "max" in spec:
            above = int((values > spec["max"]).sum())
            if above:
                out.append(f"{name}: {above} values above max {spec['max']}")

    max_unique = spec.get("max_unique")
    if max_unique is not None:
        n_unique = int(s.nunique(dropna=True))
        if n_unique > max_unique:
            out.append(f"{name} has {n_unique} unique values (allowed {max_unique})")
    return out


def _check_rows(df: pd.DataFrame, contract: dict[str, Any]) -> list[str]:
    out: list[str] = []
    bounds = contract.get("row_count", {})
    n = len(df)
    if "min" in bounds and n < bounds["min"]:
        out.append(f"row count {n} is below min {bounds['min']}")
    if "max" in bounds and n > bounds["max"]:
        out.append(f"row count {n} is above max {bounds['max']}")
    if contract.get("unique_rows"):
        dups = int(df.duplicated().sum())
        if dups:
            out.append(f"{dups} fully duplicated rows (allowed 0)")
    return out


def validate_contract(df: Any, contract: dict[str, Any]) -> list[str]:
    """Return a list of human-readable contract violations; empty means the data conforms.

    Never raises: an error inside a check is reported as a violation.
    """
    if not isinstance(df, pd.DataFrame):
        return [f"expected a pandas DataFrame, got {type(df).__name__}"]

    violations: list[str] = []

    def run(label: str, fn: Callable[[], list[str]]) -> None:
        try:
            violations.extend(fn())
        except Exception as exc:  # noqa: BLE001 - report, never crash
            violations.append(f"{label}: check could not run ({type(exc).__name__}: {exc})")

    run("columns", lambda: _check_columns(df, contract))
    for spec in contract.get("columns", []):
        if spec.get("name") in df.columns:
            run(spec["name"], lambda spec=spec: _check_column(df[spec["name"]], spec))
    run("rows", lambda: _check_rows(df, contract))
    return violations
