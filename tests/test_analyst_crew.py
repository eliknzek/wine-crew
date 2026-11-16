"""Crew 1 structure, without calling an LLM."""

import json

import pytest
from crewai import Process

from wine_crew.crews.analyst.crew import (
    AnalystCrew,
    make_files_guardrail,
    make_insights_guardrail,
    strip_fences,
)
from wine_crew.tools.eda_tools import INSIGHT_HEADINGS

EXPECTED_TOOLS = {
    "Data Cleaner": {"load_raw", "clean_data"},
    "EDA Analyst": {"eda_summary", "make_charts", "build_html_report"},
    "Insights Writer": {"read_eda_summary"},
    "Contract Author": {"build_contract", "validate_contract"},
}


@pytest.fixture
def analyst(monkeypatch, tmp_path, config):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-a-real-key")
    return AnalystCrew(config=config, out_dir=tmp_path)


def test_crew_has_four_agents_and_tasks_in_sequence(analyst):
    crew = analyst.crew()
    assert len(crew.agents) == 4
    assert len(crew.tasks) == 4
    assert crew.process == Process.sequential
    assert [t.agent.role.strip() for t in crew.tasks] == list(EXPECTED_TOOLS)


def test_each_agent_has_exactly_its_tools(analyst):
    for a in analyst.crew().agents:
        assert {t.name for t in a.tools} == EXPECTED_TOOLS[a.role.strip()]
        assert a.allow_delegation is False


def test_llm_comes_from_config(analyst, config):
    for a in analyst.crew().agents:
        assert config["llm"]["model"] in a.llm.model
        assert a.llm.temperature == config["llm"]["temperature"] == 0


def test_insights_task_has_guardrail_and_paths_are_inputs(analyst, tmp_path):
    insights = analyst.crew().tasks[2]
    assert insights.agent.role.strip() == "Insights Writer"
    assert insights.guardrail is not None
    inputs = analyst.inputs()
    assert inputs["summary_json"] == str(tmp_path / "eda_summary.json")
    assert "{summary_json}" in insights.description


class _Out:
    def __init__(self, raw):
        self.raw = raw


def _insights(bias_line: str) -> str:
    h1, h2, h3, h4 = INSIGHT_HEADINGS
    return f"{h1}\n- מאטה.\n\n{h2}\n- Austria.\n\n{h3}\n{bias_line}\n\n{h4}\n- המלצה."


def test_guardrail_accepts_summary_numbers_and_rejects_invented_ones(tmp_path):
    summary = tmp_path / "eda_summary.json"
    summary.write_text(json.dumps({"country_bias": {"top_country_share_pct": 45.04}, "top_n": 15,
                                   "value_countries": [{"name": "Austria"}]}))
    guardrail = make_insights_guardrail(summary)
    good = _insights("- 45% מארה\"ב")
    ok, text = guardrail(_Out(f"```markdown\n{good}\n```"))
    assert ok and text == good
    ok, message = guardrail(_Out(_insights("- 45% מארה\"ב ו-17% מצרפת")))
    assert not ok and "17" in message


def test_guardrail_rejects_style_problems(tmp_path):
    summary = tmp_path / "eda_summary.json"
    summary.write_text(json.dumps({"country_bias": {"top_country_share_pct": 45.04}}))
    ok, message = make_insights_guardrail(summary)(_Out("- 45.04% מארה\"ב"))
    assert not ok
    assert "missing heading" in message and "45.04%" in message


def test_every_task_has_a_guardrail(analyst):
    assert all(t.guardrail is not None for t in analyst.crew().tasks)


def test_files_guardrail_requires_tool_outputs(tmp_path):
    report = tmp_path / "eda_report.html"
    guardrail = make_files_guardrail([report])
    ok, message = guardrail(_Out("done"))
    assert not ok and "eda_report.html" in message
    report.write_text("<html></html>")
    assert guardrail(_Out("done")) == (True, "done")


def test_guardrail_fails_without_summary(tmp_path):
    ok, message = make_insights_guardrail(tmp_path / "missing.json")(_Out("text"))
    assert not ok and "not found" in message


def test_strip_fences():
    assert strip_fences("```md\n# a\n```") == "# a"
    assert strip_fences("# a") == "# a"
