"""End-to-end check of the cleaning and contract tools on the real raw file.

Skipped when data/raw/ is empty (run scripts/download_data.py). Deselect with
`pytest -m "not real_data"`.
"""

import pytest

from wine_crew.config import resolve_path
from wine_crew.tools.contract_tools import build_contract, validate_contract
from wine_crew.tools.data_tools import clean_data, load_raw

pytestmark = pytest.mark.real_data


def test_real_file_cleans_and_meets_contract(config):
    if not resolve_path(config, "raw_csv").exists():
        pytest.skip("raw data file not downloaded")
    raw, sha = load_raw(config)
    clean, report = clean_data(raw, config)
    contract = build_contract(clean, sha, config)
    assert contract["row_count"]["min"] <= report["rows_out"] <= contract["row_count"]["max"]
    assert validate_contract(clean, contract) == []
