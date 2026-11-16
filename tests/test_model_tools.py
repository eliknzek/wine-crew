import copy

import numpy as np
import pandas as pd
import pytest

from wine_crew.tools.data_tools import load_contract_template
from wine_crew.tools.model_tools import (
    OTHER,
    RareCategoryGrouper,
    build_features,
    feature_violations,
    input_columns,
    load_model,
    rank_value_wines,
    save_model,
    train_models,
)

LEAKAGE_OR_IDS = {"description", "taster_name", "title", "winery", "designation", "points"}


@pytest.fixture(scope="module")
def contract():
    from wine_crew.config import load_config

    return load_contract_template(load_config())


@pytest.fixture
def trained(model_df, contract, model_config):
    return train_models(model_df, contract, model_config)


# --------------------------------------------------------------------------- leakage


def test_features_have_no_leakage_columns(model_df, contract, model_config):
    for version in ("v1", "v2"):
        features = build_features(model_df, contract, model_config, version)
        assert not LEAKAGE_OR_IDS & set(features.columns)
        assert feature_violations(list(features.columns), contract) == []
    v2 = build_features(model_df, contract, model_config, "v2")
    assert list(v2.columns) == [
        "log_price", "vintage", "vintage_missing", "country", "province", "variety", "region_1",
    ]
    assert "region_1" not in build_features(model_df, contract, model_config, "v1").columns


def test_feature_violations_messages(contract):
    assert feature_violations(["taster_name"], contract) == [
        "Feature check failed: taster_name is a leakage column"
    ]
    assert feature_violations(["title"], contract) == [
        "Feature check failed: title is an identifier column"
    ]
    assert feature_violations(["points"], contract) == [
        "Feature check failed: points is a target column"
    ]
    assert feature_violations(["mystery"], contract) == [
        "Feature check failed: mystery is not in the contract"
    ]
    assert feature_violations(["log_price", "vintage", "vintage_missing", "region_1"], contract) == []


def test_build_features_rejects_forbidden_config(model_df, contract, model_config):
    cfg = copy.deepcopy(model_config)
    cfg["features"]["categorical"].append("taster_name")
    with pytest.raises(ValueError, match="taster_name is a leakage column"):
        build_features(model_df, contract, cfg)
    with pytest.raises(ValueError, match="taster_name is a leakage column"):
        train_models(model_df, contract, cfg)


# --------------------------------------------------------------------------- rare categories


def test_rare_grouping_counts_fit_data_only():
    train = pd.DataFrame({"country": ["US"] * 6 + ["France"] * 2 + [np.nan]})
    test = pd.DataFrame({"country": ["France"] * 10 + ["Japan", "US", np.nan]})
    grouper = RareCategoryGrouper(["country"], min_count=5).fit(train)
    out = grouper.transform(test)["country"].tolist()
    assert out[:10] == [OTHER] * 10          # rare in train, common in test -> still Other
    assert out[10] == OTHER                  # unseen -> Other
    assert out[11] == "US"
    assert pd.isna(out[12])                  # missing stays missing


# --------------------------------------------------------------------------- models


def test_models_beat_baseline(trained):
    metrics = trained["metrics"]
    for version in ("v1", "v2"):
        assert metrics[version]["test"]["mae"] < metrics["baseline"]["test"]["mae"]
        assert metrics[version]["cv"]["mae"] < metrics["baseline"]["cv"]["mae"]
    assert trained["selected"] in ("v1", "v2")
    assert set(metrics["v2"]["test"]) == {"mae", "rmse", "r2"}
    assert trained["n_train"] + trained["n_test"] == 600


def test_train_models_is_deterministic(model_df, contract, model_config, trained):
    again = train_models(model_df, contract, model_config)
    assert again["metrics"] == trained["metrics"]
    assert again["selected"] == trained["selected"]
    cols = input_columns(model_config, "v2")
    np.testing.assert_array_equal(
        again["models"]["v2"].predict(model_df[cols]),
        trained["models"]["v2"].predict(model_df[cols]),
    )


def test_pipeline_predicts_from_raw_form_row(trained, tmp_path):
    form = pd.DataFrame([{
        "price": 25.0, "vintage": None, "country": "France", "province": "Nowhere",
        "variety": "Merlot", "region_1": None,
    }])
    for name in ("v1", "v2"):
        model = trained["models"][name]
        pred = model.predict(form[trained["input_columns"][name]])
        assert pred.shape == (1,) and np.isfinite(pred).all()
        path = save_model(model, tmp_path / f"{name}.pkl")
        np.testing.assert_array_equal(load_model(path).predict(form[trained["input_columns"][name]]), pred)


# --------------------------------------------------------------------------- ranking


def test_rank_value_wines(model_df, model_config):
    ranking = rank_value_wines(model_df, model_config, "v2")
    assert len(ranking) == len(model_df)
    np.testing.assert_allclose(ranking["value_score"], ranking["points"] - ranking["predicted"])
    assert ranking["value_score"].is_monotonic_decreasing
    assert not LEAKAGE_OR_IDS - {"title", "winery", "points"} & set(ranking.columns)
    pd.testing.assert_frame_equal(ranking, rank_value_wines(model_df, model_config, "v2"))
    assert len(rank_value_wines(model_df, model_config, "v1", top_n=10)) == 10
