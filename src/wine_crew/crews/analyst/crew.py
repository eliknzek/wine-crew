"""Crew 1 – Data Analyst: four agents in a sequential process.

Data Cleaner -> EDA Analyst -> Insights Writer -> Contract Author. Agents only
call the deterministic tools; the Insights Writer's output must pass a
guardrail that rejects any number not found in eda_summary.json.

Manual run (writes to artifacts/analyst/, needs OPENAI_API_KEY in .env):
    python -m wine_crew.crews.analyst.crew
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from wine_crew.config import load_config, resolve_path
from wine_crew.crews.common import (  # loads .env first
    AGENT_MAX_ITER,
    RequestCounter,
    make_files_guardrail,
    make_llm,
    strip_fences,
)

from crewai import Agent, Crew, Process, Task  # noqa: E402
from crewai.project import CrewBase, agent, crew, task  # noqa: E402

from wine_crew.tools import crewai_tools as ct  # noqa: E402
from wine_crew.tools.eda_tools import (  # noqa: E402
    insight_number_violations,
    insight_style_violations,
)

__all__ = ["AnalystCrew", "analyst_paths", "make_files_guardrail", "make_insights_guardrail",
           "make_llm", "run_analyst", "strip_fences"]


def analyst_paths(config: dict[str, Any], out_dir: str | Path | None = None) -> dict[str, Path]:
    """Input and output paths for one run. out_dir defaults to paths.artifacts_analyst."""
    out = Path(out_dir) if out_dir is not None else resolve_path(config, "artifacts_analyst")
    return {
        "raw_csv": resolve_path(config, "raw_csv"),
        "clean_csv": out / "clean_data.csv",
        "summary_json": out / "eda_summary.json",
        "charts_dir": out / "charts",
        "report_html": out / "eda_report.html",
        "insights_md": out / "insights.md",
        "contract_json": out / "dataset_contract.json",
    }


def make_insights_guardrail(summary_json: Path):
    """Guardrail: every number must appear in eda_summary.json and the style rules must hold."""

    # No annotations here: CrewAI validates the return annotation, and with
    # `from __future__ import annotations` it would be a string and be rejected.
    def guardrail(output):  # (TaskOutput) -> tuple[bool, Any]
        text = strip_fences(output.raw)
        if not summary_json.exists():
            return False, f"{summary_json.name} not found; call read_eda_summary first."
        summary = json.loads(summary_json.read_text(encoding="utf-8"))
        problems = []
        bad_numbers = insight_number_violations(text, summary)
        if bad_numbers:
            problems.append("these numbers do not appear in the eda_summary JSON: " + ", ".join(bad_numbers))
        problems += insight_style_violations(text, summary)
        if problems:
            return False, "Fix these problems and rewrite the full Markdown:\n- " + "\n- ".join(problems)
        return True, text

    return guardrail


@CrewBase
class AnalystCrew:
    """Crew 1. Construct with a config and output directory, then kickoff(inputs=self.inputs())."""

    agents_config = "agents.yaml"
    tasks_config = "tasks.yaml"

    def __init__(self, config: dict[str, Any] | None = None, out_dir: str | Path | None = None,
                 verbose: bool = False) -> None:
        self.config = config if config is not None else load_config()
        self.paths = analyst_paths(self.config, out_dir)
        self.verbose = verbose

    def inputs(self) -> dict[str, str]:
        return {k: str(v) for k, v in self.paths.items()}

    def _agent(self, name: str, tools: list) -> Agent:
        # A separate LLM object per agent: CrewAI's crew.token_usage adds every agent's
        # llm usage, so a shared object would be counted once per agent (4x).
        return Agent(config=self.agents_config[name], tools=tools, llm=make_llm(self.config),
                     allow_delegation=False, max_iter=AGENT_MAX_ITER, verbose=self.verbose)

    @agent
    def data_cleaner(self) -> Agent:
        # result_as_answer on the last tool of a tool-only task: its JSON output is the task
        # answer, which saves the final LLM call. Safe because these tasks' guardrails check
        # files, not the answer text, and CrewAI never turns a failed tool call into the answer.
        return self._agent("data_cleaner", [ct.LoadRawTool(config=self.config),
                                            ct.CleanDataTool(config=self.config, result_as_answer=True)])

    @agent
    def eda_analyst(self) -> Agent:
        return self._agent("eda_analyst", [ct.EdaSummaryTool(config=self.config),
                                           ct.MakeChartsTool(config=self.config),
                                           ct.BuildHtmlReportTool(result_as_answer=True)])

    @agent
    def insights_writer(self) -> Agent:
        return self._agent("insights_writer", [ct.ReadEdaSummaryTool()])

    @agent
    def contract_author(self) -> Agent:
        return self._agent("contract_author", [ct.BuildContractTool(config=self.config),
                                               ct.ValidateContractTool(config=self.config,
                                                                       result_as_answer=True)])

    @task
    def clean_task(self) -> Task:
        return Task(config=self.tasks_config["clean_task"],
                    guardrail=make_files_guardrail([self.paths["clean_csv"]]))

    @task
    def eda_task(self) -> Task:
        p = self.paths
        return Task(config=self.tasks_config["eda_task"],
                    guardrail=make_files_guardrail([p["summary_json"], p["charts_dir"], p["report_html"]]))

    @task
    def insights_task(self) -> Task:
        return Task(config=self.tasks_config["insights_task"],
                    guardrail=make_insights_guardrail(self.paths["summary_json"]),
                    guardrail_max_retries=3)

    @task
    def contract_task(self) -> Task:
        return Task(config=self.tasks_config["contract_task"],
                    guardrail=make_files_guardrail([self.paths["contract_json"]]))

    @crew
    def crew(self) -> Crew:
        return Crew(agents=self.agents, tasks=self.tasks, process=Process.sequential,
                    verbose=self.verbose)


def run_analyst(config: dict[str, Any] | None = None, out_dir: str | Path | None = None,
                verbose: bool = False) -> dict[str, Any]:
    """Run Crew 1 end to end. Writes insights.md only after the guardrail passed."""
    analyst = AnalystCrew(config=config, out_dir=out_dir, verbose=verbose)
    insights_path = ct._safe_output_path(analyst.paths["insights_md"])
    crew = analyst.crew()
    start = time.monotonic()
    with RequestCounter() as counter:
        result = crew.kickoff(inputs=analyst.inputs())
    seconds = time.monotonic() - start

    outputs = {t.name or str(i): t.raw for i, t in enumerate(result.tasks_output)}
    insights = strip_fences(result.tasks_output[2].raw)
    insights = "\n".join(line.rstrip() for line in insights.splitlines())   # no trailing spaces
    insights_path.parent.mkdir(parents=True, exist_ok=True)
    insights_path.write_text(insights + "\n", encoding="utf-8")

    usage = result.token_usage
    return {
        "seconds": round(seconds, 1),
        "paths": analyst.inputs(),
        "task_outputs": outputs,
        "token_usage": usage.model_dump() if usage is not None else None,
        "usage": counter.totals(),                      # counted from OpenAI's per-call usage
        "requests_by_task": counter.by_task(crew.tasks),
    }


if __name__ == "__main__":
    summary = run_analyst(verbose=True)
    print(json.dumps({k: summary[k] for k in ("seconds", "token_usage", "paths")}, indent=2))
