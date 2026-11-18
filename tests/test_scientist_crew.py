"""Crew 2 structure and guardrails, without calling an LLM."""

import json

import pytest
from crewai import Process

from wine_crew.crews.scientist.crew import (
    ScientistCrew,
    make_contract_guardrail,
    make_model_card_guardrail,
)
from wine_crew.tools.model_report import MODEL_CARD_HEADINGS

EXPECTED_TOOLS = {
    "Contract Validator": {"validate_contract"},
    "Feature Engineer": {"build_features"},
    "Model Trainer": {"train_models"},
    "Model Card Writer": {"read_metrics"},
}


@pytest.fixture
def scientist(monkeypatch, tmp_path, config):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-a-real-key")
    return ScientistCrew(config=config, out_dir=tmp_path)


class _Out:
    def __init__(self, raw):
        self.raw = raw


def test_crew_has_four_agents_and_tasks_in_sequence(scientist):
    crew = scientist.crew()
    assert len(crew.agents) == 4 and len(crew.tasks) == 4
    assert crew.process == Process.sequential
    assert [t.agent.role.strip() for t in crew.tasks] == list(EXPECTED_TOOLS)


def test_each_agent_has_exactly_its_tools(scientist):
    for a in scientist.crew().agents:
        assert {t.name for t in a.tools} == EXPECTED_TOOLS[a.role.strip()]
        assert a.allow_delegation is False


def test_llm_comes_from_config(scientist, config):
    for a in scientist.crew().agents:
        assert config["llm"]["model"] in a.llm.model
        assert a.llm.temperature == config["llm"]["temperature"] == 0


def test_every_task_has_a_guardrail_and_contract_failure_is_not_retried(scientist):
    tasks = scientist.crew().tasks
    assert all(t.guardrail is not None for t in tasks)
    assert tasks[0].guardrail_max_retries == 0
    assert tasks[3].guardrail_max_retries == 3


def test_paths_default_to_crew1_outputs_and_are_inputs(scientist, config, tmp_path):
    inputs = scientist.inputs()
    assert inputs["clean_csv"].endswith(config["paths"]["clean_data"])
    assert inputs["contract_json"].endswith(config["paths"]["contract"])
    assert inputs["metrics_json"] == str(tmp_path / "metrics.json")
    assert "{metrics_json}" in scientist.crew().tasks[3].description


def test_contract_guardrail(tmp_path):
    check = tmp_path / "contract_check.json"
    guardrail = make_contract_guardrail(check)
    ok, message = guardrail(_Out("done"))
    assert not ok and "contract_check.json" in message
    violations = ["row count 32 is below min 100000", "missing column: price"]
    check.write_text(json.dumps({"passed": False, "violations": violations}))
    ok, message = guardrail(_Out("done"))
    assert not ok and message.startswith("Contract check failed")
    assert all(v in message for v in violations)
    check.write_text(json.dumps({"passed": True, "violations": []}))
    assert guardrail(_Out("done")) == (True, "done")


def _metrics():
    return {
        "selected": "v2",
        "models": {"v2": {"test": {"mae": 1.6903}}},
        "data": {"collected_year": 2017, "top_country": "US", "top_country_he": 'ארה"ב',
                 "top_country_share_pct": 45.04, "points_min": 80},
        "features": {"leakage_columns": ["description", "taster_name"], "region_1_max_categories": 255},
    }


def _card(extra=""):
    h = MODEL_CARD_HEADINGS
    limits = "\n".join([
        "- הנתונים נאספו ב-2017 ממקור אחד.",
        '- 45% מהיינות מארה"ב אחרי ניקוי.',
        "- נכללו רק יינות שקיבלו 80 נקודות ומעלה.",
        "- הניקוד סובייקטיבי.",
        "- הוחרגו description ו-taster_name.",
        "- vintage מותר כי הוא ידוע לפני הטעימה.",
        "- ב-region_1 לכל היותר 255 קטגוריות.",
        "- מספרים מגיעים מהנתונים, אבל לא שהוא משויך לטענה הנכונה.",
    ])
    return (f"# Model Card\n\n{h[0]}\nמטרה.\n\n{h[1]}\nנתונים.\n\n{h[2]}\nMAE 1.69. {extra}\n"
            f"המודל שנבחר: v2, לפי MAE.\n\n{h[3]}\n{limits}\n\n{h[4]}\n- המודל נועד לתמיכה בהחלטה ולא להחליף טעימה.\n")


def test_model_card_guardrail(tmp_path):
    metrics = tmp_path / "metrics.json"
    guardrail = make_model_card_guardrail(metrics)
    ok, message = guardrail(_Out(_card()))
    assert not ok and "not found" in message
    metrics.write_text(json.dumps(_metrics()))
    ok, text = guardrail(_Out(f"```markdown\n{_card()}\n```"))
    assert ok and text == _card().strip()
    ok, message = guardrail(_Out(_card("השיפור 33%.")))
    assert not ok and "33" in message
    ok, message = guardrail(_Out(_card().replace('מארה"ב', "מ-US")))
    assert not ok and "top-country share" in message
