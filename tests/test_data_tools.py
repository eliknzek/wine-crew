import copy
import hashlib

import numpy as np
import pandas as pd
import pytest

from wine_crew.tools.data_tools import clean_data, extract_vintage, load_raw, read_clean_csv


def test_clean_data_is_deterministic(raw_df, config):
    original = raw_df.copy()
    first, report1 = clean_data(raw_df, config)
    second, report2 = clean_data(raw_df, config)
    pd.testing.assert_frame_equal(first, second)
    assert report1 == report2
    digest = lambda df: hashlib.sha256(df.to_csv(index=False).encode()).hexdigest()
    assert digest(first) == digest(second)
    pd.testing.assert_frame_equal(raw_df, original)  # input untouched


def test_clean_data_report_counts(raw_df, config):
    clean, report = clean_data(raw_df, config)
    assert report["rows_in"] == 35
    assert report["removed_duplicates"] == 1
    assert report["removed_missing_required"] == 2
    assert report["dropped_columns"] == ["region_2", "taster_twitter_handle"]
    assert report["rows_out"] == len(clean) == 32


def test_clean_data_rules(clean_df):
    assert not clean_df.duplicated().any()
    assert clean_df[["price", "country", "variety"]].notna().all().all()
    assert "region_2" not in clean_df.columns
    assert "taster_twitter_handle" not in clean_df.columns
    np.testing.assert_allclose(clean_df["log_price"], np.log(clean_df["price"]))
    assert "Winery 43" in clean_df["winery"].tolist()
    assert "Merlot" in clean_df["variety"].astype(str).tolist()
    assert list(clean_df.index) == list(range(len(clean_df)))


def test_clean_data_dtypes_follow_contract(clean_df, small_contract):
    assert [c["name"] for c in small_contract["columns"]] == list(clean_df.columns)
    assert clean_df["vintage"].dtype == "Int64"
    assert isinstance(clean_df["country"].dtype, pd.CategoricalDtype)


@pytest.mark.parametrize(
    "title, expected",
    [
        ("Nicosia 2013 Vulkà Bianco  (Etna)", 2013),
        ("Winery NV Brut (Champagne)", None),
        ("Winery 1975 2010 Merlot", 2010),
        ("Winery 2018 Pinot Noir", None),
        ("Château 12345 2001 Red", 2001),
        (np.nan, None),
    ],
)
def test_extract_vintage(title, expected):
    assert extract_vintage(title, 1980, 2017) == expected


def test_vintage_column(clean_df):
    by_winery = dict(zip(clean_df["winery"], clean_df["vintage"]))
    assert by_winery["Winery 1"] == 2013
    assert pd.isna(by_winery["Winery 42"])
    assert by_winery["Winery 43"] == 2010


def test_load_raw_returns_sha256(tmp_path, raw_df, config):
    path = tmp_path / "raw.csv"
    raw_df.to_csv(path)
    cfg = copy.deepcopy(config)
    cfg["paths"]["raw_csv"] = str(path)
    df, sha = load_raw(cfg)
    assert sha == hashlib.sha256(path.read_bytes()).hexdigest()
    assert len(df) == len(raw_df)
    assert "Unnamed: 0" not in df.columns


def test_load_raw_missing_file(tmp_path, config):
    cfg = copy.deepcopy(config)
    cfg["paths"]["raw_csv"] = str(tmp_path / "missing.csv")
    with pytest.raises(FileNotFoundError, match="download_data.py"):
        load_raw(cfg)


def test_read_clean_csv_round_trip(tmp_path, clean_df, small_contract):
    path = tmp_path / "clean.csv"
    clean_df.to_csv(path, index=False)
    pd.testing.assert_frame_equal(read_clean_csv(path, small_contract), clean_df)
