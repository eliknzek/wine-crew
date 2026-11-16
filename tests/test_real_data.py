"""End-to-end checks of the tools on the real raw file.

Skipped when data/raw/ is empty (run scripts/download_data.py). Deselect with
`pytest -m "not real_data"`.
"""

import pytest

from wine_crew.config import load_config, resolve_path
from wine_crew.tools.contract_tools import build_contract, validate_contract
from wine_crew.tools.data_tools import clean_data, load_raw
from wine_crew.tools.model_tools import train_models

pytestmark = pytest.mark.real_data


@pytest.fixture(scope="module")
def real_clean():
    config = load_config()
    if not resolve_path(config, "raw_csv").exists():
        pytest.skip("raw data file not downloaded")
    raw, sha = load_raw(config)
    clean, report = clean_data(raw, config)
    return clean, report, build_contract(clean, sha, config)


def test_real_file_cleans_and_meets_contract(real_clean):
    clean, report, contract = real_clean
    assert contract["row_count"]["min"] <= report["rows_out"] <= contract["row_count"]["max"]
    assert validate_contract(clean, contract) == []


def test_real_file_selected_model_beats_baseline(real_clean, config):
    clean, _, contract = real_clean
    result = train_models(clean, contract, config)
    metrics = result["metrics"]
    assert metrics[result["selected"]]["test"]["mae"] < metrics["baseline"]["test"]["mae"]
