"""Helpers shared by both crews: environment, LLM, fence stripping, file guardrail.

Importing this module loads .env and turns off CrewAI telemetry and tracing,
so import it before crewai.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from wine_crew.config import REPO_ROOT

load_dotenv(REPO_ROOT / ".env")
# No anonymous telemetry or tracing from CrewAI.
os.environ.setdefault("CREWAI_DISABLE_TELEMETRY", "true")
os.environ.setdefault("OTEL_SDK_DISABLED", "true")
os.environ.setdefault("CREWAI_TRACING_ENABLED", "false")

from crewai import LLM  # noqa: E402

_FENCE = re.compile(r"^```(?:markdown|md)?\s*\n(.*?)\n```\s*$", re.DOTALL)


def make_llm(config: dict[str, Any]) -> LLM:
    return LLM(model=config["llm"]["model"], temperature=config["llm"]["temperature"])


def strip_fences(text: str) -> str:
    text = text.strip()
    match = _FENCE.match(text)
    return match.group(1).strip() if match else text


def make_files_guardrail(required: list[Path]):
    """Guardrail: pass only if the task's tools actually wrote every required file."""

    # No annotations here: CrewAI validates the return annotation, and with
    # `from __future__ import annotations` it would be a string and be rejected.
    def guardrail(output):  # (TaskOutput) -> tuple[bool, Any]
        missing = [p for p in required if not p.exists()]
        if missing:
            return False, (
                "These files were not created: " + ", ".join(str(p) for p in missing)
                + ". Call the remaining tools from the task description with the exact paths given."
            )
        return True, output.raw

    return guardrail


# --------------------------------------------------------------------------- LLM request counting

import threading  # noqa: E402
from collections import Counter  # noqa: E402

from crewai.events.event_bus import crewai_event_bus  # noqa: E402
from crewai.events.types.llm_events import LLMCallCompletedEvent  # noqa: E402

# Every agent stops after this many reasoning steps; a normal task needs at most 4
# (3 tool calls + an answer), so this only bounds a runaway loop.
AGENT_MAX_ITER = 6


TOKEN_FIELDS = ("input_tokens", "cached_input_tokens", "output_tokens")


class RequestCounter:
    """Count LLM requests and tokens per task while a crew runs (context manager).

    Reads OpenAI's own usage from every LLMCallCompletedEvent (prompt, cached prompt and
    completion tokens), so these are the numbers to report, not CrewAI's crew.token_usage
    (which adds a shared LLM's totals once per agent). Handlers run in a thread pool, so
    the bus is flushed on exit before the handler is removed.
    """

    def __init__(self) -> None:
        self.tasks: dict[str, dict[str, int]] = {}
        self._lock = threading.Lock()

    def _handler(self, source, event) -> None:  # (source, LLMCallCompletedEvent)
        usage = event.usage or {}
        with self._lock:
            row = self.tasks.setdefault(event.task_name or "unknown",
                                        {"requests": 0, **{f: 0 for f in TOKEN_FIELDS}})
            row["requests"] += 1
            row["input_tokens"] += int(usage.get("prompt_tokens") or 0)
            row["cached_input_tokens"] += int(usage.get("cached_prompt_tokens") or 0)
            row["output_tokens"] += int(usage.get("completion_tokens") or 0)

    def __enter__(self) -> "RequestCounter":
        crewai_event_bus.register_handler(LLMCallCompletedEvent, self._handler)
        return self

    def __exit__(self, *exc) -> None:
        crewai_event_bus.flush()
        crewai_event_bus.off(LLMCallCompletedEvent, self._handler)

    def totals(self) -> dict[str, int]:
        """Requests and tokens summed over every task."""
        keys = ("requests", *TOKEN_FIELDS)
        return {k: sum(row[k] for row in self.tasks.values()) for k in keys}

    def by_task(self, tasks) -> dict[str, dict[str, int]]:
        """{task name: requests, tokens and guardrail_retries} for the crew's tasks, in order."""
        empty = {"requests": 0, **{f: 0 for f in TOKEN_FIELDS}}
        out = {}
        known = set()
        for i, task in enumerate(tasks):
            key = task.name or task.description
            known.add(key)
            out[task.name or f"task_{i}"] = {**self.tasks.get(key, empty),
                                             "guardrail_retries": int(getattr(task, "retry_count", 0) or 0)}
        other = [row for key, row in self.tasks.items() if key not in known]
        if other:
            out["other"] = {k: sum(r[k] for r in other) for k in ("requests", *TOKEN_FIELDS)}
            out["other"]["guardrail_retries"] = 0
        return out
