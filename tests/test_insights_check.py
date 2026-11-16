from wine_crew.tools.eda_tools import insight_number_violations

SUMMARY = {
    "row_count": 111537,
    "top_n": 15,
    "points": {"mean": 88.5612, "median": 88.0},
    "country_bias": {"top_country": "US", "top_country_share_pct": 45.04},
    "price_bands": {"bands": [{"band": "$4-$10", "mean_points": 84.97}],
                    "plateau_starts_at_band": "$500-$3300"},
    "by_variety": [{"name": "Riesling", "median_points": 90.0, "median_price": 22.0}],
    "flag": True,
}


def test_text_with_only_summary_numbers_passes():
    text = (
        "# תובנות\n"
        "בנתונים יש 111,537 יינות. 45% מהם מארה\"ב.\n"
        "ממוצע הניקוד 88.56, ובעיגול 88.6 או 89.\n"
        "בין $4 ל-$10 הממוצע 84.97. הניקוד מאט מעל $500 ועד $3,300.\n"
        "בין 15 הזנים הגדולים, Riesling: חציון 90 ב-$22.\n"
    )
    assert insight_number_violations(text, SUMMARY) == []


def test_invented_numbers_are_caught():
    text = "45% מארה\"ב, ו-12% מצרפת. Riesling עולה $19 ומקבל 90.5."
    assert insight_number_violations(text, SUMMARY) == ["12", "19", "90.5"]


def test_rounding_tolerance_follows_written_precision():
    assert insight_number_violations("45.0%", SUMMARY) == []      # 45.04 -> 45.0
    assert insight_number_violations("45.1%", SUMMARY) == ["45.1"]
    assert insight_number_violations("88.561", SUMMARY) == []     # 88.5612 -> 88.561
    assert insight_number_violations("88.57", SUMMARY) == ["88.57"]


def test_numbered_list_markers_are_ignored():
    text = "1. מחיר: הניקוד מאט מעל $500\n2) הטיה: 45% מארה\"ב\n### 3. זנים\n"
    assert insight_number_violations(text, SUMMARY) == []


def test_duplicates_reported_once_and_booleans_ignored():
    # The summary's True must not count as the number 1, so "1" is reported too.
    assert insight_number_violations("1 1 7 7", SUMMARY) == ["1", "7"]
