"""Crew 2 – Data Scientist: four agents in a sequential process.

Contract Validator -> Feature Engineer -> Model Trainer -> Model Card Writer.
Agents only call the deterministic tools. The crew stops at the first task if
the contract check fails, so no model is ever trained on broken data. The
model card must pass a guardrail that rejects any number not found in
metrics.json and checks the fixed templates; its metrics table is built in code
from metrics.json and replaces the {{METRICS_TABLE}} line before saving.

Manual run (reads artifacts/analyst/, writes artifacts/scientist/, needs
OPENAI_API_KEY in .env):
    python -m wine_crew.crews.scientist.crew
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from wine_crew.config import load_config, resolve_path
from wine_crew.crews.common import make_files_guardrail, make_llm, strip_fences  # loads .env first

from crewai import Agent, Crew, Process, Task  # noqa: E402
from crewai.project import CrewBase, agent, crew, task  # noqa: E402

from wine_crew.checks import FlowCheckError, check_features  # noqa: E402
from wine_crew.tools import crewai_tools as ct  # noqa: E402
from wine_crew.tools.eda_tools import insight_number_violations  # noqa: E402
from wine_crew.tools.model_report import (  # noqa: E402
    METRICS_TABLE_PLACEHOLDER,
    fill_metrics_table,
    model_card_style_violations,
)


def scientist_paths(config: dict[str, Any], out_dir: str | Path | None = None,
                    clean_csv: str | Path | None = None,
                    contract_json: str | Path | None = None) -> dict[str, Path]:
    """Input and output paths for one run. out_dir defaults to paths.artifacts_scientist,
    and the inputs default to Crew 1's outputs (paths.clean_data, paths.contract)."""
    out = Path(out_dir) if out_dir is not None else resolve_path(config, "artifacts_scientist")
    return {
        "clean_csv": Path(clean_csv) if clean_csv is not None else resolve_path(config, "clean_data"),
        "contract_json": (Path(contract_json) if contract_json is not None
                          else resolve_path(config, "contract")),
        "contract_check": out / "contract_check.json",
        "features_csv": out / "features.csv",
        "model_pkl": out / "model.pkl",
        "metrics_json": out / "metrics.json",
        "evaluation_report": out / "evaluation_report.md",
        "value_ranking": out / "value_ranking.csv",
        "model_card": out / "model_card.md",
    }


def make_contract_guardrail(check_json: Path):
    """Guardrail: the contract check file must exist and say passed=true.

    Used with guardrail_max_retries=0, so a failed check raises at once and the
    crew stops before features or training.
    """

    # Unannotated: CrewAI validates the return annotation (see crews/common.py).
    def guardrail(output):  # (TaskOutput) -> tuple[bool, Any]
        if not check_json.exists():
            return False, f"{check_json.name} was not created; call validate_contract with output_json_path."
        check = json.loads(check_json.read_text(encoding="utf-8"))
        if not check.get("passed"):
            return False, "Contract check failed:\n- " + "\n- ".join(check.get("violations") or ["unknown"])
        return True, output.raw

    return guardrail


def make_features_guardrail(features_csv: Path, clean_csv: Path, contract_json: Path,
                            config: dict[str, Any]):
    """Guardrail: Flow gate 2 (checks.check_features) on features.csv, before training starts.

    Used with guardrail_max_retries=0, so bad features stop the crew at once.
    """

    def guardrail(output):  # (TaskOutput) -> tuple[bool, Any]; unannotated, see above
        try:
            contract = json.loads(contract_json.read_text(encoding="utf-8"))
            check_features(features_csv, clean_csv, contract, config)
        except FlowCheckError as exc:
            return False, exc.message
        except OSError as exc:
            return False, f"Feature check failed: {exc}"
        return True, output.raw

    return guardrail


def make_model_card_guardrail(metrics_json: Path):
    """Guardrail: every number must appear in metrics.json and the model card templates must hold."""

    def guardrail(output):  # (TaskOutput) -> tuple[bool, Any]; unannotated, see above
        text = strip_fences(output.raw)
        if not metrics_json.exists():
            return False, f"{metrics_json.name} not found; call read_metrics first."
        metrics = json.loads(metrics_json.read_text(encoding="utf-8"))
        problems = []
        bad_numbers = insight_number_violations(text, metrics)
        if bad_numbers:
            problems.append("these numbers do not appear in the metrics JSON: " + ", ".join(bad_numbers))
        problems += model_card_style_violations(text, metrics)
        if problems:
            return False, "Fix these problems and rewrite the full Markdown:\n- " + "\n- ".join(problems)
        return True, text

    return guardrail


@CrewBase
class ScientistCrew:
    """Crew 2. Construct with a config and paths, then kickoff(inputs=self.inputs())."""

    agents_config = "agents.yaml"
    tasks_config = "tasks.yaml"

    def __init__(self, config: dict[str, Any] | None = None, out_dir: str | Path | None = None,
                 clean_csv: str | Path | None = None, contract_json: str | Path | None = None,
                 verbose: bool = False) -> None:
        self.config = config if config is not None else load_config()
        self.paths = scientist_paths(self.config, out_dir, clean_csv, contract_json)
        self.verbose = verbose
        self.llm = make_llm(self.config)

    def inputs(self) -> dict[str, str]:
        # The placeholder goes in as an input: written literally in tasks.yaml,
        # CrewAI would read {METRICS_TABLE} inside it as a missing variable.
        return {**{k: str(v) for k, v in self.paths.items()},
                "metrics_table_placeholder": METRICS_TABLE_PLACEHOLDER}

    def _agent(self, name: str, tools: list) -> Agent:
        return Agent(config=self.agents_config[name], tools=tools, llm=self.llm,
                     allow_delegation=False, verbose=self.verbose)

    @agent
    def contract_validator(self) -> Agent:
        return self._agent("contract_validator", [ct.ValidateContractTool(config=self.config)])

    @agent
    def feature_engineer(self) -> Agent:
        return self._agent("feature_engineer", [ct.BuildFeaturesTool(config=self.config)])

    @agent
    def model_trainer(self) -> Agent:
        return self._agent("model_trainer", [ct.TrainModelsTool(config=self.config)])

    @agent
    def model_card_writer(self) -> Agent:
        return self._agent("model_card_writer", [ct.ReadMetricsTool()])

    @task
    def validate_task(self) -> Task:
        return Task(config=self.tasks_config["validate_task"],
                    guardrail=make_contract_guardrail(self.paths["contract_check"]),
                    guardrail_max_retries=0)

    @task
    def features_task(self) -> Task:
        p = self.paths
        return Task(config=self.tasks_config["features_task"],
                    guardrail=make_features_guardrail(p["features_csv"], p["clean_csv"],
                                                      p["contract_json"], self.config),
                    guardrail_max_retries=0)

    @task
    def train_task(self) -> Task:
        p = self.paths
        return Task(config=self.tasks_config["train_task"],
                    guardrail=make_files_guardrail([p["model_pkl"], p["metrics_json"],
                                                    p["evaluation_report"], p["value_ranking"]]))

    @task
    def model_card_task(self) -> Task:
        return Task(config=self.tasks_config["model_card_task"],
                    guardrail=make_model_card_guardrail(self.paths["metrics_json"]),
                    guardrail_max_retries=3)

    @crew
    def crew(self) -> Crew:
        return Crew(agents=self.agents, tasks=self.tasks, process=Process.sequential,
                    verbose=self.verbose)


def run_scientist(config: dict[str, Any] | None = None, out_dir: str | Path | None = None,
                  clean_csv: str | Path | None = None, contract_json: str | Path | None = None,
                  verbose: bool = False) -> dict[str, Any]:
    """Run Crew 2 end to end. Writes model_card.md only after the guardrail passed,
    with the metrics table placeholder replaced by the table built from metrics.json."""
    scientist = ScientistCrew(config=config, out_dir=out_dir, clean_csv=clean_csv,
                              contract_json=contract_json, verbose=verbose)
    card_path = ct._safe_output_path(scientist.paths["model_card"])
    start = time.monotonic()
    result = scientist.crew().kickoff(inputs=scientist.inputs())
    seconds = time.monotonic() - start

    outputs = {t.name or str(i): t.raw for i, t in enumerate(result.tasks_output)}
    metrics = json.loads(scientist.paths["metrics_json"].read_text(encoding="utf-8"))
    card = fill_metrics_table(strip_fences(result.tasks_output[3].raw), metrics)
    card_path.parent.mkdir(parents=True, exist_ok=True)
    card_path.write_text(card + "\n", encoding="utf-8")

    usage = result.token_usage
    return {
        "seconds": round(seconds, 1),
        "paths": scientist.inputs(),
        "task_outputs": outputs,
        "token_usage": usage.model_dump() if usage is not None else None,
    }


if __name__ == "__main__":
    summary = run_scientist(verbose=True)
    print(json.dumps({k: summary[k] for k in ("seconds", "token_usage", "paths")}, indent=2))
