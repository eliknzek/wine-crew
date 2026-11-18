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
