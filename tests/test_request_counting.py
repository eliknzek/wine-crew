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


def _event(task_name, i, prompt=100, cached=40, completion=10):
    return LLMCallCompletedEvent(
        response="ok", call_type=LLMCallType.LLM_CALL, call_id=f"c{i}", task_name=task_name,
        usage={"prompt_tokens": prompt, "cached_prompt_tokens": cached, "completion_tokens": completion})


def test_request_counter_counts_requests_and_tokens_per_task():
    tasks = [SimpleNamespace(name="clean_task", description="d1", retry_count=0),
             SimpleNamespace(name="insights_task", description="d2", retry_count=2)]
    with RequestCounter() as counter:
        for i, name in enumerate(["clean_task", "clean_task", "insights_task", "insights_task",
                                  "insights_task", "something_else"]):
            crewai_event_bus.emit(None, _event(name, i))
    row = lambda n, retries: {"requests": n, "input_tokens": 100 * n, "cached_input_tokens": 40 * n,  # noqa: E731
                              "output_tokens": 10 * n, "guardrail_retries": retries}
    assert counter.by_task(tasks) == {"clean_task": row(2, 0), "insights_task": row(3, 2), "other": row(1, 0)}
    assert counter.totals() == {"requests": 6, "input_tokens": 600, "cached_input_tokens": 240, "output_tokens": 60}
    # The handler is removed on exit: later events are not counted.
    crewai_event_bus.emit(None, _event("clean_task", 99))
    crewai_event_bus.flush()
    assert counter.tasks["clean_task"]["requests"] == 2


def test_missing_usage_counts_the_request_with_zero_tokens():
    with RequestCounter() as counter:
        crewai_event_bus.emit(None, LLMCallCompletedEvent(
            response="ok", call_type=LLMCallType.LLM_CALL, call_id="x", task_name="t"))
    assert counter.totals() == {"requests": 1, "input_tokens": 0, "cached_input_tokens": 0, "output_tokens": 0}


def test_no_two_agents_share_an_llm_object(crews):
    for crew in crews:
        ids = [id(agent.llm) for agent in crew.agents]
        assert len(set(ids)) == len(ids) == 4


def test_crewai_usage_is_not_multiplied_by_the_number_of_agents(crews):
    """Before the fix, all 4 agents shared one LLM and CrewAI added its usage once per agent."""
    for crew in crews:
        for i, agent in enumerate(crew.agents):
            agent.llm._track_token_usage_internal(
                {"prompt_tokens": 100 * (i + 1), "completion_tokens": 10, "total_tokens": 100 * (i + 1) + 10})
        usage = crew.calculate_usage_metrics()
        assert usage.successful_requests == 4                      # one call per agent, not 16
        assert usage.prompt_tokens == 100 + 200 + 300 + 400
