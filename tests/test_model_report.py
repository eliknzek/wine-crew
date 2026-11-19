import copy
import json

import pytest

from wine_crew.config import load_config
from wine_crew.tools.data_tools import load_contract_template
from wine_crew.tools.eda_tools import insight_number_violations
from wine_crew.tools.model_report import (
    METRICS_TABLE_PLACEHOLDER,
    MODEL_CARD_HEADINGS,
    evaluation_report_md,
    fill_metrics_table,
    metrics_table_md,
    metrics_document,
    model_card_style_violations,
)
from wine_crew.tools.model_tools import rank_value_wines, train_models


@pytest.fixture(scope="module")
def trained(model_df):
    # Same scaled-down thresholds as conftest.model_config, at module scope.
    cfg = copy.deepcopy(load_config())
    cfg["features"]["min_category_count"] = 5
    cfg["model"]["hgb"]["max_iter"] = 50
    contract = load_contract_template(cfg)
    result = train_models(model_df, contract, cfg)
    return cfg, contract, result


@pytest.fixture(scope="module")
def metrics(trained, model_df):
    cfg, contract, result = trained
    return metrics_document(result, model_df, contract, cfg)


def test_metrics_document_is_deterministic(trained, model_df, metrics):
    cfg, contract, _ = trained
    again = metrics_document(train_models(model_df, contract, cfg), model_df, contract, cfg)
    assert json.dumps(again, sort_keys=True) == json.dumps(metrics, sort_keys=True)
    assert json.loads(json.dumps(metrics)) == metrics


def test_metrics_document_fields(metrics, model_df):
    assert set(metrics["models"]) == {"baseline", "v1", "v2"}
    assert metrics["selected"] in ("v1", "v2")
    d, f = metrics["data"], metrics["features"]
    assert (d["train_pct"], d["test_pct"], d["cv_folds"], d["seed"]) == (80, 20, 5, 42)
    assert d["collected_year"] == 2017
    assert d["rows"] == d["n_train"] + d["n_test"] == len(model_df)
    assert d["points_min"] >= 80
    assert d["top_country"] == "US" or d["top_country_he"] == d["top_country"]
    assert f["region_1_max_categories"] == 255
    assert f["leakage_columns"] == ["description", "taster_name"]
    assert "region_1" in f["input_columns"]["v2"] and "region_1" not in f["input_columns"]["v1"]
    assert metrics["selected_test_mae_reduction_vs_baseline_pct"] > 0
    assert metrics["value_ranking"]["top_n"] == 50


def test_country_he_mapping(trained, model_df):
    cfg, contract, result = trained
    us = model_df.copy()
    us["country"] = "US"
    assert metrics_document(result, us, contract, cfg)["data"]["top_country_he"] == 'ארה"ב'


def test_evaluation_report(trained, model_df, metrics):
    cfg, _, _ = trained
    ranking = rank_value_wines(model_df, cfg, metrics["selected"], top_n=50)
    ranking.loc[0, "title"] = "Pipe | Wine"
    report = evaluation_report_md(metrics, ranking)
    for name in ("baseline", "v1", "v2"):
        assert f"| {name}" in report or f"**{name}**" in report
    assert f"**{metrics['selected']}** (נבחר)" in report
    rows = [line for line in report.splitlines() if line.startswith("| ") and line.split("|")[1].strip().isdigit()]
    assert len(rows) == 50
    assert "Pipe \\| Wine" in report


# --------------------------------------------------------------------------- model card style


def card(metrics, **replace):
    d = metrics["data"]
    sections = {
        "goal": "המודל חוזה את הניקוד של יין לפי מחיר, מקור, זן ובציר.",
        "data": f"- מקור: {d['source']}, נאסף ב-{d['collected_year']}.\n- {d['rows']} יינות אחרי ניקוי.",
        "scores": f"{METRICS_TABLE_PLACEHOLDER}\n"
                  f"המודל שנבחר: {metrics['selected']}, לפי MAE ממוצע ב-CV.",
        "limits": "\n".join([
            f"- הנתונים נאספו ב-{d['collected_year']} ממקור אחד.",
            f"- {round(d['top_country_share_pct'])}% מהיינות מ{d['top_country_he']} אחרי ניקוי.",
            f"- נכללו רק יינות שקיבלו {d['points_min']} נקודות ומעלה.",
            "- הניקוד ניתן ע\"י טועמים ולכן הוא סובייקטיבי.",
            "- הוחרגו עמודות דליפה: description ו-taster_name.",
            "- vintage (בציר) מותר כי הוא ידוע לפני הטעימה.",
            f"- ב-region_1 יש לכל היותר {metrics['features']['region_1_max_categories']} קטגוריות.",
            "- בדיקת המספרים מוודאת שכל מספר מגיע מהנתונים, אבל לא שהוא משויך לטענה הנכונה.",
        ]),
        "ethics": "- המודל נועד לתמיכה בהחלטה ולא להחליף טעימה.",
    }
    sections.update(replace)
    h = MODEL_CARD_HEADINGS
    return (f"# Model Card\n\n{h[0]}\n{sections['goal']}\n\n{h[1]}\n{sections['data']}\n\n"
            f"{h[2]}\n{sections['scores']}\n\n{h[3]}\n{sections['limits']}\n\n{h[4]}\n{sections['ethics']}\n")


@pytest.fixture
def us_metrics(metrics):
    m = json.loads(json.dumps(metrics))
    m["data"].update(top_country="US", top_country_he='ארה"ב', top_country_share_pct=45.04)
    return m


def test_valid_card_passes(us_metrics):
    assert model_card_style_violations(card(us_metrics), us_metrics) == []


def test_gershayim_and_bold_are_accepted(us_metrics):
    text = card(us_metrics).replace('ארה"ב', "**ארה״ב**")
    assert model_card_style_violations(text, us_metrics) == []


@pytest.mark.parametrize("line_index, label", [
    (0, "data year and single source"),
    (1, "top-country share"),
    (2, "minimum points"),
    (3, "subjective scores"),
    (4, "leakage columns"),
    (5, "vintage known before tasting"),
    (6, "region_1 category cap"),
    (7, "number guardrail limit"),
])
def test_each_limitation_is_required(us_metrics, line_index, label):
    lines = card(us_metrics).split("## מגבלות\n")[1].split("\n\n")[0].splitlines()
    del lines[line_index]
    violations = model_card_style_violations(card(us_metrics, limits="\n".join(lines)), us_metrics)
    assert any(f"the {label} line" in v for v in violations)


def test_top_country_must_be_written_in_hebrew(us_metrics):
    text = card(us_metrics).replace('מארה"ב', "מ-US")
    violations = model_card_style_violations(text, us_metrics)
    assert any("top-country share" in v and 'ארה"ב' in v for v in violations)


def test_missing_and_out_of_order_headings(us_metrics):
    text = card(us_metrics)
    assert "missing heading: ## מגבלות" in model_card_style_violations(
        text.replace("## מגבלות\n", ""), us_metrics)
    swapped = text.replace("## מטרה", "@@").replace("## נתונים", "## מטרה").replace("@@", "## נתונים")
    assert "headings are not in the required order" in model_card_style_violations(swapped, us_metrics)


def test_trailing_zero_and_selected_line_and_ethics(us_metrics):
    text = card(us_metrics, data="- 1.0 משהו.")
    assert "write 1.0 as a whole number (1)" in model_card_style_violations(text, us_metrics)
    no_selected = card(us_metrics, scores=METRICS_TABLE_PLACEHOLDER)
    assert any(v.startswith("section '## מדדים'") for v in model_card_style_violations(no_selected, us_metrics))
    no_ethics = card(us_metrics, ethics="- המודל טוב.")
    assert any(v.startswith("section '## היבטים אתיים'") for v in model_card_style_violations(no_ethics, us_metrics))


def test_number_check_against_metrics(us_metrics):
    us_metrics["models"]["v2"]["test"]["mae"] = 1.6903
    assert insight_number_violations("MAE של 1.69 ו-R² של v2", us_metrics) == []
    assert insight_number_violations("שיפור של 37.5%", us_metrics) == ["37.5"]
    assert insight_number_violations(card(us_metrics), us_metrics) == []


# --------------------------------------------------------------------------- metrics table placeholder


def _table_metrics():
    return {"models": {
        "baseline": {"cv": {"mae": 2.538}, "test": {"mae": 2.538, "rmse": 3.1006, "r2": -0.0001}},
        "v2": {"cv": {"mae": 1.6991}, "test": {"mae": 1.6903, "rmse": 2.1524, "r2": 0.5183}},
    }}


def test_metrics_table_has_two_decimals_and_no_negative_zero():
    table = metrics_table_md(_table_metrics())
    assert "| baseline | 2.54 | 2.54 | 3.10 | 0.00 |" in table
    assert "| v2 | 1.70 | 1.69 | 2.15 | 0.52 |" in table
    assert "-0" not in table


def test_fill_metrics_table_replaces_exactly_one_placeholder():
    filled = fill_metrics_table(f"## מדדים\n{METRICS_TABLE_PLACEHOLDER}\nסוף", _table_metrics())
    assert METRICS_TABLE_PLACEHOLDER not in filled and "| v2 | 1.70 |" in filled
    for text in ("## מדדים\n", METRICS_TABLE_PLACEHOLDER * 2):
        with pytest.raises(ValueError, match="exactly once"):
            fill_metrics_table(text, _table_metrics())


@pytest.mark.parametrize("scores, expected", [
    ("המודל שנבחר: {sel}.", "exactly once"),
    ("{ph}\n{ph}\nהמודל שנבחר: {sel}.", "exactly once"),
    ("{ph}\n| v2 | 1.69 |\nהמודל שנבחר: {sel}.", "do not write a metrics table"),
])
def test_placeholder_rules(us_metrics, scores, expected):
    scores = scores.format(ph=METRICS_TABLE_PLACEHOLDER, sel=us_metrics["selected"])
    violations = model_card_style_violations(card(us_metrics, scores=scores), us_metrics)
    assert any(expected in v for v in violations)


def test_placeholder_outside_metrics_section(us_metrics):
    text = card(us_metrics, scores=f"המודל שנבחר: {us_metrics['selected']}.",
                goal=f"מטרה.\n{METRICS_TABLE_PLACEHOLDER}")
    assert any("must be inside '## מדדים'" in v for v in model_card_style_violations(text, us_metrics))


@pytest.mark.parametrize("before, after", [("\n", "\n"), ("\n\n", "\n\n"), ("\n  ", "  \n\n\n")])
def test_fill_metrics_table_puts_one_blank_line_around_table(before, after):
    table = metrics_table_md(_table_metrics())
    filled = fill_metrics_table(f"## מדדים{before}{METRICS_TABLE_PLACEHOLDER}{after}המודל שנבחר: v2.",
                                _table_metrics())
    assert filled == f"## מדדים\n\n{table}\n\nהמודל שנבחר: v2."


def test_filled_card_passes_number_check(us_metrics):
    filled = fill_metrics_table(card(us_metrics), us_metrics)
    assert insight_number_violations(filled, us_metrics) == []


# --------------------------------------------------------------------------- thousands separators


from wine_crew.tools.model_report import _fmt2, format_counts  # noqa: E402

COUNTS = {"data": {"rows": 111537, "n_train": 89229, "n_test": 22308, "collected_year": 2017, "seed": 42}}


@pytest.mark.parametrize("text, expected", [
    ("- 111537 יינות אחרי ניקוי: 89229 לאימון ו-22308 לבדיקה.",
     "- 111,537 יינות אחרי ניקוי: 89,229 לאימון ו-22,308 לבדיקה."),
    ("סה\"כ 111537.", "סה\"כ 111,537."),                                   # end of sentence
    ("נאספו ב-2017, seed 42.", "נאספו ב-2017, seed 42."),               # years and the seed stay
    ("כבר עם פסיקים: 111,537", "כבר עם פסיקים: 111,537"),              # no double commas
    ("1115370 ו-0.111537", "1115370 ו-0.111537"),                       # not inside other numbers
])
def test_format_counts(text, expected):
    assert format_counts(text, COUNTS) == expected


def test_format_counts_keeps_the_number_check_passing():
    text = "- 111537 יינות: 89229 לאימון ו-22308 לבדיקה, נאספו ב-2017."
    assert insight_number_violations(text, COUNTS) == []
    assert insight_number_violations(format_counts(text, COUNTS), COUNTS) == []


def test_fmt2_adds_thousands_separators():
    assert _fmt2(1234.5) == "1,234.50"
    assert _fmt2(-0.0001) == "0.00"
    assert _fmt2(1.6895) == "1.69"


def test_card_after_fill_and_format(us_metrics):
    us_metrics["data"].update(rows=111537, n_train=89229, n_test=22308)
    text = card(us_metrics)
    final = format_counts(fill_metrics_table(text, us_metrics), us_metrics)
    assert "111,537" in final and "111537" not in final
    assert insight_number_violations(final, us_metrics) == []
