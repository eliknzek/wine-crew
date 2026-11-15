import copy

import numpy as np
import pandas as pd
import pytest

from wine_crew.config import load_config
from wine_crew.tools.data_tools import clean_data, load_contract_template

RAW_COLUMNS = [
    "country", "description", "designation", "points", "price", "province", "region_1",
    "region_2", "taster_name", "taster_twitter_handle", "title", "variety", "winery",
]


def _row(i, **overrides):
    row = {
        "country": "US",
        "description": f"Tasting note {i}.",
        "designation": "Reserve",
        "points": 85 + i % 10,
        "price": 10.0 + i,
        "province": "California",
        "region_1": "Napa Valley",
        "region_2": "Napa",
        "taster_name": "Taster A",
        "taster_twitter_handle": "@tastera ",
        "title": f"Winery {i} 2013 Cabernet (Napa Valley)",
        "variety": "Cabernet Sauvignon",
        "winery": f"Winery {i}",
    }
    row.update(overrides)
    return row


@pytest.fixture
def raw_df():
    """Small raw frame shaped like winemag-data-130k-v2.csv, with known defects."""
    rows = [_row(i) for i in range(10)]
    rows.append(_row(0))                                            # full duplicate of row 0
    rows.append(_row(20, price=np.nan))                             # missing price
    rows.append(_row(21, country=np.nan))                           # missing country
    rows.append(_row(22, title="Winery 22 NV Brut (Champagne)"))    # no vintage
    rows.append(_row(23, title="Winery 1975 2010 Merlot", winery="  Winery 23  ",
                     variety=" Merlot "))                           # year outside range first, whitespace
    df = pd.DataFrame(rows, columns=RAW_COLUMNS)
    df.index = range(100, 100 + len(df))
    return df


@pytest.fixture
def config():
    return load_config()


@pytest.fixture
def small_contract(config):
    """The real contract template with row_count bounds relaxed for tiny test frames."""
    contract = copy.deepcopy(load_contract_template(config))
    contract["row_count"] = {"min": 1, "max": 1000}
    return contract


@pytest.fixture
def clean_df(raw_df, config):
    return clean_data(raw_df, config)[0]
