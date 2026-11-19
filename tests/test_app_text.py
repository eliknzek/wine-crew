"""Numbers in words, glossary and Flow steps for the app (pure functions, no artifacts, no LLM)."""

import pandas as pd
import pytest

from wine_crew import app_data as ad


@pytest.mark.parametrize("points,label", [
    (80, "סביר"), (82, "סביר"), (83, "טוב"), (86, "טוב"), (87, "טוב מאוד"), (89, "טוב מאוד"),
    (90, "מצוין"), (93, "מצוין"), (94, "יוצא דופן"), (97, "יוצא דופן"), (98, "קלאסי"), (100, "קלאסי"),
    (89.4, "טוב מאוד"), (89.5, "מצוין"), (82.49, "סביר"), (82.5, "טוב"),
    (77, "סביר"), (103, "קלאסי"),
])
def test_score_label_boundaries(points, label):
    assert ad.score_label(points) == label


def test_score_text():
    assert ad.score_text(88.6) == "89: טוב מאוד"
    assert ad.score_text(89.5) == "90: מצוין"


def test_value_sentence():
    assert ad.value_sentence(9.081) == "9.1 נקודות מעל הניקוד שצפוי ליין כזה במחיר הזה"
    assert ad.value_sentence(-0.42) == "0.4 נקודות מתחת לניקוד שצפוי ליין כזה במחיר הזה"
    assert ad.value_sentence(0.04) == ad.value_sentence(-0.04) == "בדיוק הניקוד שצפוי ליין כזה במחיר הזה"


def _metrics(mae):
    return {"selected": "v2", "models": {"v2": {"test": {"mae": mae}}, "v1": {"test": {"mae": 9.9}}}}


def test_accuracy_sentence_comes_from_metrics():
    assert ad.model_mae(_metrics(2.34)) == 2.34
    assert ad.accuracy_sentence(_metrics(2.34)) == "המודל טועה בממוצע בכ-2.3 נקודות"
    assert ad.accuracy_sentence(_metrics(1.6895)) == "המודל טועה בממוצע בכ-1.7 נקודות"


def test_residual_band_is_10th_and_90th_percentile():
    ranking = pd.DataFrame({"points": list(range(80, 101)) * 5, "predicted": [90.0] * 105})
    low, high = ad.residual_band(ranking)
    gaps = ranking["points"] - ranking["predicted"]
    assert (low, high) == (gaps.quantile(0.1), gaps.quantile(0.9))


def test_prediction_range_rounds_inward_and_clips():
    assert ad.prediction_range(89.0, (-2.78, 2.63)) == (87, 91)    # 86.22..91.63 holds 87..91
    assert ad.prediction_range(88.6, (-2.78, 2.63)) == (86, 91)
    assert ad.prediction_range(89.0, (-2.0, 2.0)) == (87, 91)      # exact whole numbers stay in
    assert ad.prediction_range(81.0, (-3.0, 2.0)) == (80, 83)
    assert ad.prediction_range(99.0, (-2.0, 3.0)) == (97, 100)


def test_prediction_sentence():
    assert ad.prediction_sentence(89.0, (-2.78, 2.63)) == \
        "89: טוב מאוד. ב-8 מתוך 10 יינות דומים, הניקוד האמיתי היה בין 87 ל-91."


def test_glossary_has_every_term():
    assert set(ad.GLOSSARY) == {"ציון שווי", "ניקוד צפוי", "חציון", "out-of-fold", "MAE"}
    assert all(text.strip() for text in ad.GLOSSARY.values())
    assert "ליין כזה במחיר הזה" in ad.GLOSSARY["ציון שווי"]


def _ok(passed=True):
    return {"passed": passed, "message": "ok {}" if passed else "failed"}


def test_flow_steps_passed_run():
    summary = {"checks": {k: _ok() for k in ("preflight", "contract", "features", "model")},
               "crews": {"crew1": {}, "crew2": {}}, "failed_check": None}
    steps = ad.flow_steps(summary)
    assert [s["key"] for s in steps] == ["preflight", "crew1", "contract", "crew2", "features", "model"]
    assert {s["status"] for s in steps} == {"passed"}
    assert ad.flow_line(steps).count("✅") == 6


def test_flow_steps_gate2_failed():
    summary = {"checks": {"preflight": _ok(), "contract": _ok(), "features": _ok(False)},
               "crews": {"crew1": {}, "crew2": {}}, "failed_check": "features"}
    status = {s["key"]: s["status"] for s in ad.flow_steps(summary)}
    assert status == {"preflight": "passed", "crew1": "passed", "contract": "passed",
                      "crew2": "passed", "features": "failed", "model": "not_run"}


def test_flow_steps_crew1_crashed():
    summary = {"checks": {"preflight": _ok()}, "crews": {}, "failed_check": "crew1"}
    status = {s["key"]: s["status"] for s in ad.flow_steps(summary)}
    assert status == {"preflight": "passed", "crew1": "failed", "contract": "not_run",
                      "crew2": "not_run", "features": "not_run", "model": "not_run"}


@pytest.mark.parametrize("crew", ["crew1", "crew2"])
def test_agent_roles_come_from_agents_yaml(crew):
    roles = ad.agent_roles(crew)
    assert len(roles) == 4
    assert all(key in ad.AGENT_HE for key, _ in roles)
