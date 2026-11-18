"""WineFlow: preflight -> Crew 1 -> gate 1 -> Crew 2 -> gate 2 -> gate 3 -> run_summary.json.

Every gate is plain Python in checks.py and reads the files itself. A failed
gate raises FlowCheckError; run_flow catches it, logs the failed check, marks
run_summary.json as failed and returns exit code 1, with a one-line message
for the user and the traceback (if any) only in the log.
"""

from __future__ import annotations

import json
import logging
import sys
import time
import traceback
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import pandas as pd
from pydantic import PrivateAttr

from wine_crew.checks import FlowCheckError, check_contract, check_features, check_model, check_preflight
from wine_crew.config import REPO_ROOT, load_config, resolve_path
from wine_crew.crews import common  # noqa: F401 - loads .env and disables telemetry before crewai

from crewai.flow.flow import Flow, FlowState, listen, start  # noqa: E402

CrewFn = Callable[[dict[str, Any]], dict[str, Any]]


def _rel(path: Path | str) -> str:
    try:
        return str(Path(path).resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


# --------------------------------------------------------------------------- cost and logging


def crew_cost(token_usage: dict[str, Any] | None, config: dict[str, Any]) -> float:
    """USD cost of one crew from its token usage and llm.price_per_million in config."""
    if not token_usage:
        return 0.0
    price = config["llm"]["price_per_million"]
    prompt = token_usage.get("prompt_tokens", 0)
    cached = token_usage.get("cached_prompt_tokens", 0)
    completion = token_usage.get("completion_tokens", 0)
    cost = ((prompt - cached) * price["input"] + cached * price["cached_input"]
            + completion * price["output"]) / 1_000_000
    return round(cost, 6)


class RunLog:
    """One log file per run (logs/run_<timestamp>.log); every line carries the run id."""

    def __init__(self, run_id: str, path: Path):
        self.run_id, self.path = run_id, path
        path.parent.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(f"wine_crew.run.{run_id}")
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False
        self.handler = logging.FileHandler(path, encoding="utf-8")
        self.handler.setFormatter(logging.Formatter(
            f"%(asctime)s %(levelname)s [{run_id}] %(message)s", "%Y-%m-%dT%H:%M:%S"))
        self.logger.addHandler(self.handler)
        self.current_step: str | None = None

    def info(self, message: str, echo: bool = True) -> None:
        self.logger.info(message)
        if echo:
            print(f"[{self.run_id}] {message}", flush=True)

    def error(self, message: str) -> None:
        self.logger.error(message)

    @contextmanager
    def step(self, name: str):
        self.current_step = name
        start = time.monotonic()
        self.info(f"step {name}: start")
        yield
        self.info(f"step {name}: end ({time.monotonic() - start:.1f}s)")

    def check(self, name: str, passed: bool, message: str) -> None:
        line = f"check {name}: {'PASSED' if passed else 'FAILED'} - {message}"
        (self.info if passed else self.error)(line)

    def close(self) -> None:
        self.logger.removeHandler(self.handler)
        self.handler.close()


# --------------------------------------------------------------------------- flow


class WineState(FlowState):
    run_id: str = ""
    raw_sha256: str | None = None
    rows: dict[str, int] = {}
    checks: dict[str, dict[str, Any]] = {}
    crews: dict[str, dict[str, Any]] = {}
    selected: str | None = None
    test_mae: dict[str, float] = {}


def _default_analyst(config: dict[str, Any]) -> dict[str, Any]:
    from wine_crew.crews.analyst.crew import run_analyst

    return run_analyst(config)


def _default_scientist(config: dict[str, Any]) -> dict[str, Any]:
    from wine_crew.crews.scientist.crew import run_scientist

    return run_scientist(config)


def flow_paths(config: dict[str, Any]) -> dict[str, Path]:
    sci = resolve_path(config, "artifacts_scientist")
    return {
        "clean_csv": resolve_path(config, "clean_data"),
        "contract_json": resolve_path(config, "contract"),
        "features_csv": sci / "features.csv",
        "metrics_json": sci / "metrics.json",
        "model_pkl": sci / "model.pkl",
        "evaluation_report": sci / "evaluation_report.md",
        "model_card": sci / "model_card.md",
        "insights": resolve_path(config, "insights"),
        "eda_report": resolve_path(config, "eda_report"),
    }


class WineFlow(Flow[WineState]):
    _config: dict[str, Any] = PrivateAttr()
    _log: RunLog = PrivateAttr()
    _analyst: CrewFn = PrivateAttr()
    _scientist: CrewFn = PrivateAttr()
    _paths: dict[str, Path] = PrivateAttr()

    def __init__(self, config: dict[str, Any], run_log: RunLog,
                 analyst_fn: CrewFn | None = None, scientist_fn: CrewFn | None = None):
        super().__init__(suppress_flow_events=True)
        self._config = config
        self._log = run_log
        self._analyst = analyst_fn or _default_analyst
        self._scientist = scientist_fn or _default_scientist
        self._paths = flow_paths(config)

    def _gate(self, name: str, fn: Callable[[], dict[str, Any]]) -> dict[str, Any]:
        try:
            details = fn()
        except FlowCheckError as exc:
            self.state.checks[name] = {"passed": False, "message": exc.message}
            self._log.check(name, False, exc.message)
            raise
        message = "ok " + json.dumps(details, ensure_ascii=False, default=str)
        self.state.checks[name] = {"passed": True, "message": message}
        self._log.check(name, True, message)
        return details

    def _crew(self, name: str, fn: CrewFn) -> None:
        result = fn(self._config) or {}
        usage = result.get("token_usage") or {}
        cost = crew_cost(usage, self._config)
        self.state.crews[name] = {"seconds": result.get("seconds"), "tokens": usage, "cost_usd": cost}
        self._log.info(
            f"{name} tokens: total={usage.get('total_tokens', 0)} prompt={usage.get('prompt_tokens', 0)} "
            f"cached={usage.get('cached_prompt_tokens', 0)} completion={usage.get('completion_tokens', 0)} "
            f"requests={usage.get('successful_requests', 0)} cost=${cost:.4f}")

    def _rows(self, path: Path) -> int:
        return int(len(pd.read_csv(path, usecols=[0])))

    @start()
    def preflight(self):
        with self._log.step("preflight"):
            details = self._gate("preflight", lambda: check_preflight(self._config))
            self.state.raw_sha256 = details["sha256"]
            raw = resolve_path(self._config, "raw_csv")
            self.state.rows["raw"] = int(len(pd.read_csv(raw, usecols=[0])))
            self._log.info(f"raw file {details['raw_csv']} sha256={details['sha256']} "
                           f"rows={self.state.rows['raw']}")

    @listen(preflight)
    def crew1(self):
        with self._log.step("crew1"):
            self._crew("crew1", self._analyst)
            self.state.rows["clean"] = self._rows(self._paths["clean_csv"])
            self._log.info(f"crew1 rows: in={self.state.rows['raw']} out={self.state.rows['clean']}")

    @listen(crew1)
    def gate_contract(self):
        with self._log.step("gate_contract"):
            self._gate("contract", lambda: check_contract(self._paths["clean_csv"],
                                                          self._paths["contract_json"]))

    @listen(gate_contract)
    def crew2(self):
        with self._log.step("crew2"):
            self._crew("crew2", self._scientist)
            features = self._paths["features_csv"]
            self.state.rows["features"] = self._rows(features) if features.exists() else 0
            self._log.info(f"crew2 rows: in={self.state.rows['clean']} out={self.state.rows['features']}")

    def _contract(self) -> dict[str, Any]:
        return json.loads(self._paths["contract_json"].read_text(encoding="utf-8"))

    @listen(crew2)
    def gate_features(self):
        with self._log.step("gate_features"):
            self._gate("features", lambda: check_features(
                self._paths["features_csv"], self._paths["clean_csv"], self._contract(), self._config))

    @listen(gate_features)
    def gate_model(self):
        with self._log.step("gate_model"):
            details = self._gate("model", lambda: check_model(
                self._paths["metrics_json"], self._paths["model_pkl"], self._paths["clean_csv"],
                self._contract()))
            metrics = json.loads(self._paths["metrics_json"].read_text(encoding="utf-8"))
            self.state.selected = details["selected"]
            self.state.test_mae = {n: m["test"]["mae"] for n, m in metrics["models"].items()}
            self.state.rows["train"] = metrics["data"]["n_train"]
            self.state.rows["test"] = metrics["data"]["n_test"]
            self._log.info(f"train rows: train={self.state.rows['train']} test={self.state.rows['test']}; "
                           f"selected {details['selected']} MAE {details['mae']:.4f} "
                           f"vs baseline {details['baseline_mae']:.4f}")


# --------------------------------------------------------------------------- entry point


def _summary(state: WineState, config: dict[str, Any], status: str, started: datetime,
             seconds: float, log_path: Path, paths: dict[str, Path],
             failed_check: str | None = None, error: str | None = None) -> dict[str, Any]:
    summary = {
        "run_id": state.run_id,
        "status": status,
        "started_at": started.isoformat(timespec="seconds"),
        "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "seconds": round(seconds, 1),
        "raw_sha256": state.raw_sha256,
        "rows": state.rows,
        "checks": state.checks,
        "selected": state.selected,
        "test_mae": state.test_mae,
        "crews": state.crews,
        "total_cost_usd": round(sum(c.get("cost_usd", 0) for c in state.crews.values()), 6),
        "log": _rel(log_path),
        "artifacts": {k: _rel(v) for k, v in paths.items()},
    }
    if status == "failed":
        summary["failed_check"] = failed_check
        summary["error"] = error
    return summary


def run_flow(config: dict[str, Any] | None = None, analyst_fn: CrewFn | None = None,
             scientist_fn: CrewFn | None = None) -> int:
    """Run the whole Flow. Returns the process exit code: 0 passed, 1 stopped or failed."""
    config = config if config is not None else load_config()
    started = datetime.now(timezone.utc)
    stamp = started.strftime("%Y%m%d-%H%M%S")
    run_id = f"{stamp}-{uuid.uuid4().hex[:6]}"
    log_path = resolve_path(config, "logs") / f"run_{stamp}.log"
    log = RunLog(run_id, log_path)
    flow = WineFlow(config, log, analyst_fn, scientist_fn)
    flow.state.run_id = run_id
    log.info(f"run {run_id} started; log {_rel(log_path)}")
    t0 = time.monotonic()
    status, failed_check, error, code = "passed", None, None, 0
    try:
        flow.kickoff()
    except FlowCheckError as exc:
        status, failed_check, error, code = "failed", exc.check, exc.message, 1
        log.error(f"run stopped at check {exc.check}: {exc.message}")
        print(f"Run {run_id} stopped: {exc.message}\nLog: {_rel(log_path)}", file=sys.stderr)
    except Exception as exc:  # noqa: BLE001 - any crash becomes a controlled stop
        step = log.current_step or "unknown"
        first = str(exc).strip().splitlines()[0] if str(exc).strip() else type(exc).__name__
        status, failed_check, error, code = "failed", step, f"{type(exc).__name__}: {first}", 1
        log.error(f"run failed in step {step}:\n{traceback.format_exc()}")
        print(f"Run {run_id} failed in step {step}: {error}\nLog: {_rel(log_path)}", file=sys.stderr)

    seconds = time.monotonic() - t0
    summary = _summary(flow.state, config, status, started, seconds, log_path, flow_paths(config),
                       failed_check, error)
    summary_path = resolve_path(config, "run_summary")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    log.info(f"run {status} in {seconds:.1f}s; total cost ${summary['total_cost_usd']:.4f}; "
             f"summary {_rel(summary_path)}")
    log.close()
    return code
