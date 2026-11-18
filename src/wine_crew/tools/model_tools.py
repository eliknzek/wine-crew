"""Modeling: build_features, train_models, evaluate, rank_value_wines.

Plain deterministic functions. Every parameter comes from config.yaml, and
every random step uses config["seed"]. Each model is an sklearn Pipeline that
takes raw clean-data columns (price, vintage, country, ...), so the app can
predict straight from a form row.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import KFold, cross_val_predict, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

OTHER = "Other"
ALLOWED_FEATURE_ROLES = ("feature", "feature_optional")
# Features derived inside the pipeline that the contract does not list.
EXTRA_DERIVED = {"vintage_missing": "vintage"}
DERIVED_NUMERIC = ["log_price", "vintage", "vintage_missing"]
SCORING = {
    "mae": "neg_mean_absolute_error",
    "rmse": "neg_root_mean_squared_error",
    "r2": "r2",
}
RANKING_COLUMNS = [
    "title", "winery", "country", "province", "region_1", "variety", "vintage", "price", "points",
]


# --------------------------------------------------------------------------- transformers


class DeriveFeatures(BaseEstimator, TransformerMixin):
    """Raw columns -> model features: adds log_price and vintage_missing, drops price."""

    def fit(self, X: pd.DataFrame, y: Any = None) -> "DeriveFeatures":
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = pd.DataFrame(X).copy()
        out = pd.DataFrame(index=X.index)
        out["log_price"] = np.log(X["price"].astype("float64"))
        vintage = pd.to_numeric(X["vintage"], errors="coerce").astype("float64")
        out["vintage"] = vintage
        out["vintage_missing"] = vintage.isna().astype("int64")
        for col in X.columns:
            if col not in ("price", "vintage"):
                out[col] = X[col]
        return out


class RareCategoryGrouper(BaseEstimator, TransformerMixin):
    """Map categories seen fewer than min_count times during fit (or unseen) to "Other".

    Counts come only from the data passed to fit, i.e. the training set or CV fold.
    Missing values stay missing.
    """

    def __init__(self, columns: list[str], min_count: int):
        self.columns = columns
        self.min_count = min_count

    def fit(self, X: pd.DataFrame, y: Any = None) -> "RareCategoryGrouper":
        self.keep_ = {}
        for col in self.columns:
            counts = X[col].astype(object).value_counts(dropna=True)
            self.keep_[col] = sorted(counts[counts >= self.min_count].index.tolist())
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for col in self.columns:
            values = X[col].astype(object)
            missing = values.isna()
            rare = ~missing & ~values.isin(self.keep_[col])
            values = values.where(~rare, OTHER).where(~missing, np.nan)
            X[col] = values
        return X


# --------------------------------------------------------------------------- features


def input_columns(config: dict[str, Any], version: str) -> list[str]:
    """Raw clean-data columns a model version reads. baseline uses the v1 inputs."""
    feats = config["features"]
    cols = list(feats["numeric"]) + list(feats["categorical"])
    if version == "v2":
        cols += list(feats["v2_extra"])
    return cols


def _categorical_columns(config: dict[str, Any], version: str) -> list[str]:
    feats = config["features"]
    cols = list(feats["categorical"])
    if version == "v2":
        cols += list(feats["v2_extra"])
    return cols


def feature_violations(columns: list[str], contract: dict[str, Any]) -> list[str]:
    """Violations for feature columns whose source column is not allowed as a feature."""
    specs = {c["name"]: c for c in contract["columns"]}
    forbidden = set(contract.get("forbidden_feature_roles", []))
    violations = []
    for col in columns:
        source = EXTRA_DERIVED.get(col, col)
        if source not in specs:
            violations.append(f"Feature check failed: {col} is not in the contract")
            continue
        # The contract column's own role decides (e.g. vintage is a feature even
        # though it is parsed from the identifier column title).
        role = specs[source]["role"]
        if role in forbidden:
            article = "an" if role[0] in "aeiou" else "a"
            violations.append(f"Feature check failed: {col} is {article} {role} column")
        elif role not in ALLOWED_FEATURE_ROLES:
            violations.append(
                f"Feature check failed: {col} has role {role} "
                f"(allowed: {', '.join(ALLOWED_FEATURE_ROLES)})"
            )
    return violations


def _require_allowed(columns: list[str], contract: dict[str, Any]) -> None:
    violations = feature_violations(columns, contract)
    if violations:
        raise ValueError("; ".join(violations))


def build_features(
    clean_df: pd.DataFrame, contract: dict[str, Any], config: dict[str, Any], version: str = "v2"
) -> pd.DataFrame:
    """Model features (before rare-category grouping) for a version; no target column.

    Raises ValueError if any input or derived feature comes from a forbidden column.
    """
    cols = input_columns(config, version)
    _require_allowed(cols, contract)
    features = DeriveFeatures().transform(clean_df[cols])
    _require_allowed(list(features.columns), contract)
    return features


# --------------------------------------------------------------------------- models


def make_pipelines(config: dict[str, Any]) -> dict[str, Pipeline]:
    """Fresh, unfitted pipelines for baseline, v1 (Ridge) and v2 (HistGradientBoosting)."""
    seed = config["seed"]
    min_count = config["features"]["min_category_count"]
    model_cfg = config["model"]
    hgb_cfg = model_cfg["hgb"]

    v1_cats = _categorical_columns(config, "v1")
    v1 = Pipeline([
        ("derive", DeriveFeatures()),
        ("group_rare", RareCategoryGrouper(v1_cats, min_count)),
        ("encode", ColumnTransformer([
            ("num", Pipeline([
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
            ]), ["log_price", "vintage"]),
            ("flag", "passthrough", ["vintage_missing"]),
            ("cat", OneHotEncoder(handle_unknown="ignore"), v1_cats),
        ])),
        ("model", Ridge(alpha=model_cfg["ridge_alpha"])),
    ])

    v2_cats = _categorical_columns(config, "v2")
    n_num = len(DERIVED_NUMERIC)
    v2 = Pipeline([
        ("derive", DeriveFeatures()),
        ("group_rare", RareCategoryGrouper(v2_cats, min_count)),
        ("encode", ColumnTransformer([
            ("num", "passthrough", DERIVED_NUMERIC),
            ("cat", OrdinalEncoder(
                handle_unknown="use_encoded_value",
                unknown_value=np.nan,
                encoded_missing_value=np.nan,
                max_categories=hgb_cfg["max_categories"],
            ), v2_cats),
        ])),
        ("model", HistGradientBoostingRegressor(
            max_iter=hgb_cfg["max_iter"],
            learning_rate=hgb_cfg["learning_rate"],
            early_stopping=hgb_cfg["early_stopping"],
            categorical_features=list(range(n_num, n_num + len(v2_cats))),
            random_state=seed,
        )),
    ])

    baseline = Pipeline([("model", DummyRegressor(strategy="mean"))])
    return {"baseline": baseline, "v1": v1, "v2": v2}


def evaluate(model: Any, X: pd.DataFrame, y: pd.Series) -> dict[str, float]:
    """MAE (primary), RMSE and R² of a fitted model on (X, y)."""
    pred = model.predict(X)
    return {
        "mae": float(mean_absolute_error(y, pred)),
        "rmse": float(root_mean_squared_error(y, pred)),
        "r2": float(r2_score(y, pred)),
    }


def _kfold(config: dict[str, Any]) -> KFold:
    return KFold(n_splits=config["model"]["cv_folds"], shuffle=True, random_state=config["seed"])


def train_models(
    clean_df: pd.DataFrame, contract: dict[str, Any], config: dict[str, Any]
) -> dict[str, Any]:
    """Fixed train/test split, CV on train, fit on train, evaluate on test.

    The selected model is the v1/v2 version with the lowest mean CV MAE; the test
    set is only used for reporting.
    """
    target = contract["target"]
    for version in ("v1", "v2"):
        _require_allowed(input_columns(config, version), contract)

    train_idx, test_idx = train_test_split(
        np.arange(len(clean_df)),
        test_size=config["model"]["test_size"],
        random_state=config["seed"],
    )
    train, test = clean_df.iloc[train_idx], clean_df.iloc[test_idx]
    y_train, y_test = train[target], test[target]

    metrics: dict[str, Any] = {}
    models: dict[str, Pipeline] = {}
    for name, pipeline in make_pipelines(config).items():
        cols = input_columns(config, "v1" if name == "baseline" else name)
        cv = cross_validate(pipeline, train[cols], y_train, cv=_kfold(config), scoring=SCORING)
        cv_summary = {}
        for metric in SCORING:
            scores = cv[f"test_{metric}"]
            scores = -scores if metric != "r2" else scores
            cv_summary[metric] = float(np.mean(scores))
            cv_summary[f"{metric}_std"] = float(np.std(scores))
        pipeline.fit(train[cols], y_train)
        models[name] = pipeline
        metrics[name] = {"cv": cv_summary, "test": evaluate(pipeline, test[cols], y_test)}

    if config["model"]["selection_metric"] != "cv_mae":
        raise ValueError(f"unsupported selection_metric: {config['model']['selection_metric']}")
    selected = min(("v1", "v2"), key=lambda n: metrics[n]["cv"]["mae"])
    return {
        "metrics": metrics,
        "selected": selected,
        "models": models,
        "input_columns": {n: input_columns(config, "v1" if n == "baseline" else n) for n in models},
        "n_train": int(len(train)),
        "n_test": int(len(test)),
    }


def save_model(model: Any, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    return path


def load_model(path: str | Path) -> Any:
    return joblib.load(path)


def rank_value_wines(
    clean_df: pd.DataFrame, config: dict[str, Any], version: str, top_n: int | None = None
) -> pd.DataFrame:
    """Wines ranked by actual points minus out-of-fold predicted points.

    Every prediction comes from a model that did not see that wine
    (cross_val_predict), so memorization cannot shrink the score to zero.
    """
    cols = input_columns(config, version)
    pipeline = make_pipelines(config)[version]
    predicted = cross_val_predict(pipeline, clean_df[cols], clean_df["points"], cv=_kfold(config))
    ranking = clean_df[RANKING_COLUMNS].copy()
    ranking["predicted"] = predicted
    ranking["value_score"] = ranking["points"] - ranking["predicted"]
    ranking = ranking.sort_values(
        ["value_score", "title"], ascending=[False, True], kind="mergesort"
    ).reset_index(drop=True)
    return ranking.head(top_n) if top_n is not None else ranking
