import pandas as pd
import pytest

from wine_crew.tools.contract_tools import build_contract, save_contract, validate_contract


def test_clean_data_passes(clean_df, small_contract):
    assert validate_contract(clean_df, small_contract) == []


def test_missing_column(clean_df, small_contract):
    violations = validate_contract(clean_df.drop(columns=["price"]), small_contract)
    assert "missing column: price" in violations


def test_null_price(clean_df, small_contract):
    broken = clean_df.copy()
    broken.loc[[0, 3], "price"] = float("nan")
    assert "price has 2 nulls (allowed 0)" in validate_contract(broken, small_contract)


def test_extra_column(clean_df, small_contract):
    broken = clean_df.assign(taster_twitter_handle="@x")
    assert "unexpected column: taster_twitter_handle" in validate_contract(broken, small_contract)


def test_points_out_of_range(clean_df, small_contract):
    broken = clean_df.copy()
    broken.loc[0, "points"] = 101
    broken.loc[1, "points"] = 79
    violations = validate_contract(broken, small_contract)
    assert "points: 1 values above max 100" in violations
    assert "points: 1 values below min 80" in violations


def test_duplicate_rows(clean_df, small_contract):
    broken = pd.concat([clean_df, clean_df.iloc[[0]]], ignore_index=True)
    assert "1 fully duplicated rows (allowed 0)" in validate_contract(broken, small_contract)


def test_wrong_dtype(clean_df, small_contract):
    broken = clean_df.assign(points=clean_df["points"].astype("float64"))
    assert "points has dtype float64 (expected int64)" in validate_contract(broken, small_contract)


def test_row_count_out_of_range(clean_df, config):
    from wine_crew.tools.data_tools import load_contract_template

    violations = validate_contract(clean_df, load_contract_template(config))
    assert f"row count {len(clean_df)} is below min 100000" in violations


def test_vintage_null_pct(clean_df, small_contract):
    broken = clean_df.copy()
    broken["vintage"] = pd.array([pd.NA] * len(broken), dtype="Int64")
    assert any(v.startswith("vintage has 100.00% nulls") for v in validate_contract(broken, small_contract))


@pytest.mark.parametrize("bad", [pd.DataFrame(), pd.DataFrame({"x": [1, 2]}), None, "not a frame"])
def test_never_raises_on_garbage(bad, small_contract):
    violations = validate_contract(bad, small_contract)
    assert isinstance(violations, list) and violations


def test_check_error_is_reported_not_raised(clean_df, small_contract):
    small_contract["columns"][0]["min"] = "not-a-number"
    violations = validate_contract(clean_df, small_contract)
    assert any("check could not run" in v for v in violations)


def test_build_contract(clean_df, config, tmp_path):
    first = build_contract(clean_df, "abc123", config)
    second = build_contract(clean_df, "abc123", config)
    assert first == second
    assert first["source"]["sha256"] == "abc123"
    assert first["observed"]["row_count"] == len(clean_df)
    assert first["observed"]["columns"]["points"]["max"] <= 100
    path = save_contract(first, tmp_path / "out" / "contract.json")
    assert path.read_text(encoding="utf-8").startswith("{")
