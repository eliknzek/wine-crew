"""Fewer and counted LLM calls: result_as_answer on tool-only tasks, max_iter, RequestCounter."""

from types import SimpleNamespace

import pytest
from crewai.events.event_bus import crewai_event_bus
from crewai.events.types.llm_events import LLMCallCompletedEvent, LLMCallType

from wine_crew.crews.analyst.crew import AnalystCrew
from wine_crew.crews.common import AGENT_MAX_ITER, RequestCounter
from wine_crew.crews.scientist.crew import ScientistCrew

# The last tool of each tool-only task; its output becomes the task answer.
EXPECTED_ANSWER_TOOLS = {
    "clean_task": {"clean_data"}, "eda_task": {"build_html_report"},
    "insights_task": set(), "contract_task": {"validate_contract"},
    "validate_task": {"validate_contract"}, "features_task": {"build_features"},
    "train_task": {"train_models"}, "model_card_task": set(),
}
# Tasks whose guardrail checks the text the agent writes: they must never get the flag.
TEXT_CHECKED_TASKS = {"insights_task", "model_card_task"}


@pytest.fixture
def crews(monkeypatch, tmp_path, config):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-a-real-key")
    return [AnalystCrew(config=config, out_dir=tmp_path / "a").crew(),
            ScientistCrew(config=config, out_dir=tmp_path / "s").crew()]


def test_result_as_answer_only_on_the_last_tool_of_tool_only_tasks(crews):
    for crew in crews:
        for task in crew.tasks:
            flagged = {t.name for t in task.agent.tools if t.result_as_answer}
            assert flagged == EXPECTED_ANSWER_TOOLS[task.name], task.name
            if task.name in TEXT_CHECKED_TASKS:
                assert flagged == set()


def test_every_agent_has_a_max_iter_cap(crews):
    assert AGENT_MAX_ITER == 6
    assert all(agent.max_iter == AGENT_MAX_ITER for crew in crews for agent in crew.agents)


def _event(task_name, i):
    return LLMCallCompletedEvent(response="ok", call_type=LLMCallType.LLM_CALL, call_id=f"c{i}",
                                 task_name=task_name)


def test_request_counter_counts_per_task():
    tasks = [SimpleNamespace(name="clean_task", description="d1", retry_count=0),
             SimpleNamespace(name="insights_task", description="d2", retry_count=2)]
    with RequestCounter() as counter:
        for i, name in enumerate(["clean_task", "clean_task", "insights_task", "insights_task",
                                  "insights_task", "something_else"]):
            crewai_event_bus.emit(None, _event(name, i))
    assert counter.by_task(tasks) == {
        "clean_task": {"requests": 2, "guardrail_retries": 0},
        "insights_task": {"requests": 3, "guardrail_retries": 2},
        "other": {"requests": 1, "guardrail_retries": 0},
    }
    # The handler is removed on exit: later events are not counted.
    crewai_event_bus.emit(None, _event("clean_task", 99))
    crewai_event_bus.flush()
    assert counter.counts["clean_task"] == 2
