"""Loading and cleaning: load_raw, clean_data, read_clean_csv.

Plain deterministic functions; every parameter comes from config.yaml and the
dataset contract template. The cleaning rules follow docs/spec.md, in order.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from wine_crew.config import resolve_path

YEAR_PATTERN = re.compile(r"(?<!\d)(\d{4})(?!\d)")


def sha256_file(path: str | Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_raw(config: dict[str, Any]) -> tuple[pd.DataFrame, str]:
    """Read the raw CSV (first column is a row index) and return it with its sha256."""
    path = resolve_path(config, "raw_csv")
    if not path.exists():
        raise FileNotFoundError(
            f"Raw data file not found: {path}. Run `python scripts/download_data.py` first."
        )
    df = pd.read_csv(path, index_col=0)
    return df, sha256_file(path)


def load_contract_template(config: dict[str, Any]) -> dict[str, Any]:
    with resolve_path(config, "contract_template").open(encoding="utf-8") as fh:
        return json.load(fh)


def extract_vintage(title: str | float, vintage_min: int, vintage_max: int) -> int | None:
    """First 4-digit year in [vintage_min, vintage_max] found in the title, else None."""
    if not isinstance(title, str):
        return None
    for match in YEAR_PATTERN.finditer(title):
        year = int(match.group(1))
        if vintage_min <= year <= vintage_max:
            return year
    return None


def apply_contract_dtypes(df: pd.DataFrame, contract: dict[str, Any]) -> pd.DataFrame:
    """Cast contract columns to their contract dtype and order them as in the contract.

    Columns not in the contract are kept at the end so validation can report them.
    """
    df = df.copy()
    contract_cols = [c["name"] for c in contract["columns"]]
    for col in contract["columns"]:
        name = col["name"]
        if name in df.columns:
            df[name] = df[name].astype(col["dtype"])
    ordered = [c for c in contract_cols if c in df.columns]
    extras = [c for c in df.columns if c not in contract_cols]
    return df[ordered + extras]


def clean_data(df: pd.DataFrame, config: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Apply the cleaning rules from docs/spec.md in order. The input is not modified.

    Returns the cleaned frame and a report with row counts per rule.
    """
    cleaning = config["cleaning"]
    contract = load_contract_template(config)
    out = df.copy()
    report: dict[str, Any] = {"rows_in": int(len(out))}

    # 1. Fully duplicated rows.
    before = len(out)
    out = out.drop_duplicates()
    report["removed_duplicates"] = int(before - len(out))

    # 2. Rows missing a required column.
    before = len(out)
    out = out.dropna(subset=cleaning["required_columns"])
    report["removed_missing_required"] = int(before - len(out))

    # 3. Drop unused columns.
    drop = [c for c in cleaning["drop_columns"] if c in out.columns]
    out = out.drop(columns=drop)
    report["dropped_columns"] = drop

    # 4. Vintage from title.
    vmin, vmax = cleaning["vintage_min"], cleaning["vintage_max"]
    out["vintage"] = pd.array(
        [extract_vintage(t, vmin, vmax) for t in out["title"]], dtype="Int64"
    )
    report["vintage_missing"] = int(out["vintage"].isna().sum())

    # 5. log of price.
    out["log_price"] = np.log(out["price"].astype("float64"))

    # 6. Strip whitespace from text columns.
    for col in out.columns:
        if pd.api.types.is_string_dtype(out[col]) or out[col].dtype == object:
            out[col] = out[col].str.strip()

    out = apply_contract_dtypes(out, contract).reset_index(drop=True)
    report["rows_out"] = int(len(out))
    return out, report


def read_clean_csv(path: str | Path, contract: dict[str, Any]) -> pd.DataFrame:
    """Read clean_data.csv back with the contract dtypes (CSV loses them)."""
    return apply_contract_dtypes(pd.read_csv(path), contract)
