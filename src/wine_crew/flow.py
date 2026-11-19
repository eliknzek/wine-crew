"""WineFlow: preflight -> Crew 1 -> gate 1 -> Crew 2 -> gate 2 -> gate 3 -> promote.

The crews write into a staging folder (paths.staging_root/<run_id>/), never into
artifacts/. Every gate is plain Python in checks.py and reads the staged files
itself. Only when all gates pass are the staged analyst/ and scientist/ folders
moved into artifacts/ (with a rollback if the move fails) and
artifacts/run_summary.json written. A failed gate raises FlowCheckError;
run_flow catches it, leaves artifacts/ exactly as it was, writes the failed
summary next to the log (logs/run_<ts>_summary.json), keeps the staging folder
for inspection and returns exit code 1, with a one-line message for the user
and the traceback (if any) only in the log.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
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

from wine_crew.tools import crewai_tools as ct  # noqa: E402

CrewFn = Callable[[dict[str, Any]], dict[str, Any]]


def _rel(path: Path | str) -> str:
    try:
        return str(Path(path).resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


# --------------------------------------------------------------------------- cost and logging


def utc_stamp(moment: datetime | None = None) -> str:
    """UTC time as 2026-11-19T10:34:51Z, the format of every timestamp the run writes."""
    moment = moment or datetime.now(timezone.utc)
    return moment.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def token_cost(usage: dict[str, Any] | None, config: dict[str, Any]) -> float:
    """USD cost from counted tokens (input includes cached input) and llm.price_per_million."""
    if not usage:
        return 0.0
    price = config["llm"]["price_per_million"]
    inp = usage.get("input_tokens", 0)
    cached = usage.get("cached_input_tokens", 0)
    out = usage.get("output_tokens", 0)
    cost = ((inp - cached) * price["input"] + cached * price["cached_input"] + out * price["output"]) / 1_000_000
    return round(cost, 6)


TOKEN_KEYS = ("requests", "input_tokens", "cached_input_tokens", "output_tokens")


class RunLog:
    """One log file per run (logs/run_<timestamp>.log); every line carries the run id."""

    def __init__(self, run_id: str, path: Path):
        self.run_id, self.path = run_id, path
        path.parent.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(f"wine_crew.run.{run_id}")
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False
        self.handler = logging.FileHandler(path, encoding="utf-8")
        formatter = logging.Formatter(
            f"%(asctime)sZ %(levelname)s [{run_id}] %(message)s", "%Y-%m-%dT%H:%M:%S")
        formatter.converter = time.gmtime  # UTC, like the run id and run_summary.json
        self.handler.setFormatter(formatter)
        self.logger.addHandler(self.handler)
        self.current_step: str | None = None

    def info(self, message: str, echo: bool = True) -> None:
        self.logger.info(message)
        if echo:
            print(f"[{self.run_id}] {message}", flush=True)

    def error(self, message: str) -> None:
        self.logger.error(message)

    def warning(self, message: str) -> None:
        self.logger.warning(message)
        print(f"[{self.run_id}] WARNING {message}", flush=True)

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
        "value_ranking": sci / "value_ranking.csv",
        "model_card": sci / "model_card.md",
        "insights": resolve_path(config, "insights"),
        "eda_report": resolve_path(config, "eda_report"),
    }


# --------------------------------------------------------------------------- staging and promotion

STAGED_DIRS = {"analyst": "artifacts_analyst", "scientist": "artifacts_scientist"}


def staging_config(config: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    """The same config with every path under artifacts/analyst or artifacts/scientist moved to
    run_dir/analyst or run_dir/scientist. Other paths (raw_csv, logs, run_summary) stay."""
    roots = {name: resolve_path(config, key).resolve() for name, key in STAGED_DIRS.items()}
    paths = dict(config["paths"])
    for key in config["paths"]:
        target = resolve_path(config, key).resolve()
        for name, root in roots.items():
            if target == root or target.is_relative_to(root):
                paths[key] = str(Path(run_dir) / name / target.relative_to(root))
                break
    return {**config, "paths": paths}


class PromotionError(Exception):
    """Moving the staged folders into artifacts/ failed; artifacts/ was restored."""


def promote(run_dir: Path, config: dict[str, Any]) -> None:
    """Replace artifacts/analyst and artifacts/scientist with the staged folders.

    Each existing folder first moves to run_dir/previous/, then the staged one takes
    its place. If any move fails, every folder already replaced goes back to its
    staged place and every backup is restored, so artifacts/ ends as it started.
    """
    previous = Path(run_dir) / "previous"
    previous.mkdir(parents=True, exist_ok=True)
    done: list[tuple[Path, Path, Path | None]] = []   # (staged, target, backup) fully swapped
    for name, key in STAGED_DIRS.items():
        staged, target = Path(run_dir) / name, resolve_path(config, key)
        backup = previous / name if target.exists() else None
        target.parent.mkdir(parents=True, exist_ok=True)   # a fresh checkout has no artifacts/ yet
        moved_out = False
        try:
            if backup is not None:
                os.replace(target, backup)
                moved_out = True
            os.replace(staged, target)
        except OSError as exc:
            if moved_out:                                  # this folder: put the old one back
                os.replace(backup, target)
            for staged_done, target_done, backup_done in reversed(done):   # earlier folders
                os.replace(target_done, staged_done)
                if backup_done is not None:
                    os.replace(backup_done, target_done)
            raise PromotionError(f"could not move {name}/ into {_rel(target)}: {exc}") from exc
        done.append((staged, target, backup))
    for _staged, target, _backup in done:
        (target / ".gitkeep").touch()                      # the tracked placeholder file


# --------------------------------------------------------------------------- demo failures

DEMO_FAILURES = ("contract", "features", "model")


def inject_demo_failure(gate: str, paths: dict[str, Path]) -> str:
    """Break one staged file so the given gate fails, for demos. Touches staging only."""
    if gate == "contract":
        df = pd.read_csv(paths["clean_csv"])
        df.drop(columns=["price"]).to_csv(paths["clean_csv"], index=False)
        return f"removed the price column from {paths['clean_csv'].name}"
    if gate == "features":
        df = pd.read_csv(paths["features_csv"])
        df.assign(taster_name="demo").to_csv(paths["features_csv"], index=False)
        return f"added the leakage column taster_name to {paths['features_csv'].name}"
    if gate == "model":
        metrics = json.loads(paths["metrics_json"].read_text(encoding="utf-8"))
        base = metrics["models"]["baseline"]["test"]["mae"]
        metrics["models"][metrics["selected"]]["test"]["mae"] = round(base + 0.07, 4)
        paths["metrics_json"].write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        return f"set the selected model's test MAE above the baseline in {paths['metrics_json'].name}"
    raise ValueError(f"unknown demo failure: {gate} (choose from {', '.join(DEMO_FAILURES)})")


class WineFlow(Flow[WineState]):
    _config: dict[str, Any] = PrivateAttr()
    _log: RunLog = PrivateAttr()
    _analyst: CrewFn = PrivateAttr()
    _scientist: CrewFn = PrivateAttr()
    _paths: dict[str, Path] = PrivateAttr()
    _demo_fail: str | None = PrivateAttr(default=None)

    def __init__(self, config: dict[str, Any], run_log: RunLog,
                 analyst_fn: CrewFn | None = None, scientist_fn: CrewFn | None = None,
                 demo_fail: str | None = None):
        super().__init__(suppress_flow_events=True)
        self._config = config
        self._log = run_log
        self._analyst = analyst_fn or _default_analyst
        self._scientist = scientist_fn or _default_scientist
        self._paths = flow_paths(config)
        self._demo_fail = demo_fail

    def _demo(self, gate: str) -> None:
        if self._demo_fail == gate:
            what = inject_demo_failure(gate, self._paths)
            self._log.info(f"DEMO: {what} (staging folder only), so gate {gate} fails")

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
        usage = {k: int((result.get("usage") or {}).get(k, 0)) for k in TOKEN_KEYS}
        cost = token_cost(usage, self._config)
        crewai_usage = result.get("token_usage") or {}
        self.state.crews[name] = {"seconds": result.get("seconds"), **usage, "cost_usd": cost,
                                  "crewai_reported": crewai_usage}
        self._log.info(
            f"{name} usage: requests={usage['requests']} input={usage['input_tokens']} "
            f"cached_input={usage['cached_input_tokens']} output={usage['output_tokens']} cost=${cost:.4f}")
        # Cross-check against CrewAI's own count; a mismatch is logged, never fatal.
        reported = crewai_usage.get("successful_requests")
        if reported is not None and reported != usage["requests"]:
            self._log.warning(f"{name}: CrewAI reports {reported} requests but {usage['requests']} were "
                              "counted from OpenAI's per-call usage; the counted numbers are used")
        by_task = result.get("requests_by_task")
        if by_task:
            self.state.crews[name]["requests_by_task"] = by_task
            parts = [f"{task}={v['requests']}" + (f" (guardrail retries {v['guardrail_retries']})"
                                                  if v.get("guardrail_retries") else "")
                     for task, v in by_task.items()]
            self._log.info(f"{name} requests by task: " + " ".join(parts))

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
            self._demo("contract")

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
            self._demo("features")
            self._demo("model")

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
                self._contract(), self._paths["value_ranking"]))
            metrics = json.loads(self._paths["metrics_json"].read_text(encoding="utf-8"))
            self.state.selected = details["selected"]
            self.state.test_mae = {n: m["test"]["mae"] for n, m in metrics["models"].items()}
            self.state.rows["train"] = metrics["data"]["n_train"]
            self.state.rows["test"] = metrics["data"]["n_test"]
            self._log.info(f"train rows: train={self.state.rows['train']} test={self.state.rows['test']}; "
                           f"selected {details['selected']} MAE {details['mae']:.4f} "
                           f"vs baseline {details['baseline_mae']:.4f}")


# --------------------------------------------------------------------------- entry point


def _summary(state: WineState, status: str, started: datetime, seconds: float, log_path: Path,
             paths: dict[str, Path], failed_check: str | None = None, error: str | None = None,
             extra: dict[str, Any] | None = None) -> dict[str, Any]:
    summary = {
        "run_id": state.run_id,
        "status": status,
        "started_at": utc_stamp(started),
        "finished_at": utc_stamp(),
        "seconds": round(seconds, 1),
        "raw_sha256": state.raw_sha256,
        "rows": state.rows,
        "checks": state.checks,
        "selected": state.selected,
        "test_mae": state.test_mae,
        "crews": state.crews,
        "tokens": {k: sum(int(c.get(k, 0)) for c in state.crews.values()) for k in TOKEN_KEYS},
        "total_cost_usd": round(sum(c.get("cost_usd", 0) for c in state.crews.values()), 6),
        "log": _rel(log_path),
        "artifacts": {k: _rel(v) for k, v in paths.items()},
        **(extra or {}),
    }
    if status == "failed":
        summary["failed_check"] = failed_check
        summary["error"] = error
    return summary


def _write_json(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def run_flow(config: dict[str, Any] | None = None, analyst_fn: CrewFn | None = None,
             scientist_fn: CrewFn | None = None, demo_fail: str | None = None) -> int:
    """Run the whole Flow. Returns the process exit code: 0 passed, 1 stopped or failed.

    artifacts/ changes only when every gate passed and the promotion succeeded.
    """
    config = config if config is not None else load_config()
    if demo_fail is not None and demo_fail not in DEMO_FAILURES:
        raise ValueError(f"unknown demo failure: {demo_fail} (choose from {', '.join(DEMO_FAILURES)})")
    started = datetime.now(timezone.utc)
    stamp = started.strftime("%Y%m%d-%H%M%S")
    run_id = f"{stamp}-{uuid.uuid4().hex[:6]}"
    logs_dir = resolve_path(config, "logs")
    log_path = logs_dir / f"run_{stamp}.log"
    run_dir = resolve_path(config, "staging_root") / run_id
    run_dir.mkdir(parents=True, exist_ok=True)   # exists even if the run stops before any crew writes
    staged = staging_config(config, run_dir)
    log = RunLog(run_id, log_path)
    flow = WineFlow(staged, log, analyst_fn, scientist_fn, demo_fail)
    flow.state.run_id = run_id
    log.info(f"run {run_id} started; log {_rel(log_path)}; staging {_rel(run_dir)}")
    if demo_fail:
        log.info(f"DEMO: this run will fail at gate {demo_fail} on purpose; artifacts/ will not change")

    # The tools may write only inside WRITE_ROOTS; open the staging folder for this run only.
    added_roots = [run_dir]
    ct.WRITE_ROOTS.extend(added_roots)
    ct.ALLOWED_ROOTS.extend(added_roots)
    t0 = time.monotonic()
    status, failed_check, error, code = "passed", None, None, 0
    try:
        flow.kickoff()
        with log.step("promote"):
            promote(run_dir, config)
            log.info(f"promoted {_rel(run_dir)} into artifacts/")
    except FlowCheckError as exc:
        status, failed_check, error, code = "failed", exc.check, exc.message, 1
        log.error(f"run stopped at check {exc.check}: {exc.message}")
    except PromotionError as exc:
        status, failed_check, error, code = "failed", "promote", str(exc), 1
        log.error(f"promotion failed, artifacts/ restored: {exc}")
    except Exception as exc:  # noqa: BLE001 - any crash becomes a controlled stop
        step = log.current_step or "unknown"
        first = str(exc).strip().splitlines()[0] if str(exc).strip() else type(exc).__name__
        status, failed_check, error, code = "failed", step, f"{type(exc).__name__}: {first}", 1
        log.error(f"run failed in step {step}:\n{traceback.format_exc()}")
    finally:
        for root in added_roots:
            ct.WRITE_ROOTS.remove(root)
            ct.ALLOWED_ROOTS.remove(root)

    seconds = time.monotonic() - t0
    extra = {"demo_fail": demo_fail} if demo_fail else {}
    if status == "passed":
        summary_path = resolve_path(config, "run_summary")
        summary = _summary(flow.state, status, started, seconds, log_path, flow_paths(config), extra=extra)
        _write_json(summary_path, summary)
        shutil.rmtree(run_dir, ignore_errors=True)
    else:
        # artifacts/ stays as it was: the failed summary goes next to the log, staging is kept.
        summary_path = logs_dir / f"run_{stamp}_summary.json"
        summary = _summary(flow.state, status, started, seconds, log_path, flow_paths(staged),
                           failed_check, error, {**extra, "staging_dir": _rel(run_dir)})
        _write_json(summary_path, summary)
        verb = "stopped" if failed_check in ("preflight", "contract", "features", "model") else "failed"
        where = "" if verb == "stopped" else f" in step {failed_check}"
        print(f"Run {run_id} {verb}{where}: {error}\n"
              f"artifacts/ was not changed. Log: {_rel(log_path)}; summary: {_rel(summary_path)}; "
              f"staged files: {_rel(run_dir)}", file=sys.stderr)
    log.info(f"run {status} in {seconds:.1f}s; total cost ${summary['total_cost_usd']:.4f}; "
             f"summary {_rel(summary_path)}")
    log.close()
    return code
