"""One real end-to-end run of Crew 1 against the LLM. Costs money.

Not selected by default (pyproject addopts); run with `pytest -m llm -s`.
Skipped without OPENAI_API_KEY (from the environment or .env) or the raw file.
"""

import json
import os

import pytest

from wine_crew.config import resolve_path
from wine_crew.tools import crewai_tools as ct
from wine_crew.tools.eda_tools import insight_number_violations

pytestmark = pytest.mark.llm


def test_analyst_crew_end_to_end(tmp_path, monkeypatch, config):
    from wine_crew.crews.analyst.crew import run_analyst  # loads .env

    if not os.environ.get("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not set")
    if not resolve_path(config, "raw_csv").exists():
        pytest.skip("raw data file not downloaded")
    monkeypatch.setattr(ct, "ALLOWED_ROOTS", [*ct.ALLOWED_ROOTS, tmp_path])
    monkeypatch.setattr(ct, "WRITE_ROOTS", [*ct.WRITE_ROOTS, tmp_path])

    result = run_analyst(config, out_dir=tmp_path)

    # Print before asserting, so time and tokens are never lost on a failure.
    print("\n=== RUN ===")
    print(json.dumps({"seconds": result["seconds"], "token_usage": result["token_usage"]}, indent=2))
    print("=== TASK OUTPUTS ===")
    for name, text in result["task_outputs"].items():
        print(f"--- {name} ---\n{text}\n")

    summary = json.loads((tmp_path / "eda_summary.json").read_text(encoding="utf-8"))
    insights = (tmp_path / "insights.md").read_text(encoding="utf-8")
    assert insight_number_violations(insights, summary) == []
    assert (tmp_path / "eda_report.html").exists()
    assert (tmp_path / "clean_data.csv").exists()
    contract = json.loads((tmp_path / "dataset_contract.json").read_text(encoding="utf-8"))
    assert ct.run_validate_contract(config, str(tmp_path / "clean_data.csv"),
                                    str(tmp_path / "dataset_contract.json"))["passed"]
    assert contract["source"]["sha256"]
