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
    rows = [_row(i) for i in range(30)]  # 1 missing vintage in 32 clean rows stays under 5%
    rows.append(_row(0))                                            # full duplicate of row 0
    rows.append(_row(40, price=np.nan))                             # missing price
    rows.append(_row(41, country=np.nan))                           # missing country
    rows.append(_row(42, title="Winery 42 NV Brut (Champagne)"))    # no vintage
    rows.append(_row(43, title="Winery 1975 2010 Merlot", winery="  Winery 43  ",
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


@pytest.fixture
def model_config(config):
    """Real config with thresholds scaled down for a few hundred rows."""
    cfg = copy.deepcopy(config)
    cfg["features"]["min_category_count"] = 5
    cfg["model"]["hgb"]["max_iter"] = 50
    return cfg


@pytest.fixture(scope="session")
def model_df():
    """~600 clean rows where points depend on price, country and variety, plus noise."""
    rng = np.random.default_rng(42)
    n = 600
    countries = {"US": 0.0, "France": 1.5, "Italy": 1.0, "Chile": -1.5, "Portugal": 0.5}
    varieties = {"Pinot Noir": 1.0, "Merlot": -0.5, "Riesling": 0.5, "Malbec": -1.0}
    country = rng.choice(list(countries), size=n)
    variety = rng.choice(list(varieties), size=n)
    country[:3] = "Rareland"          # rare category -> Other
    price = np.round(np.exp(rng.normal(3.2, 0.7, size=n)), 0).clip(4, 3300)
    noise = rng.normal(0, 1.0, size=n)
    points = 76 + 3.5 * np.log(price) + np.vectorize(countries.get)(country, 0.0) \
        + np.vectorize(varieties.get)(variety) + noise
    vintage = rng.integers(1995, 2017, size=n).astype(object)
    vintage[rng.random(n) < 0.03] = None
    rows = []
    for i in range(n):
        year = f" {vintage[i]}" if vintage[i] is not None else " NV"
        rows.append(_row(
            i,
            country=country[i],
            province=f"{country[i]} province {i % 3}",
            region_1=f"{country[i]} region {i % 4}" if i % 7 else np.nan,
            variety=variety[i],
            price=float(price[i]),
            points=int(np.clip(np.round(points[i]), 80, 100)),
            title=f"Winery {i}{year} {variety[i]}",
        ))
    raw = pd.DataFrame(rows, columns=RAW_COLUMNS)
    return clean_data(raw, load_config())[0]
