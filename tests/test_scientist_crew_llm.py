"""One real end-to-end run of Crew 2 against the LLM. Costs money.

Not selected by default (pyproject addopts); run with `pytest -m llm -s`.
Skipped without OPENAI_API_KEY (from the environment or .env) or the raw file.
Crew 1's outputs are produced with its deterministic tools (no LLM) first.
"""

import json
import os

import pytest

from wine_crew.config import resolve_path
from wine_crew.tools import crewai_tools as ct
from wine_crew.tools.eda_tools import insight_number_violations
from wine_crew.tools.model_report import (
    METRICS_TABLE_PLACEHOLDER,
    metrics_table_md,
    model_card_style_violations,
)
from wine_crew.tools.model_tools import load_model

pytestmark = pytest.mark.llm


def test_scientist_crew_end_to_end(tmp_path, monkeypatch, config):
    from wine_crew.crews.scientist.crew import run_scientist  # loads .env

    if not os.environ.get("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not set")
    raw = resolve_path(config, "raw_csv")
    if not raw.exists():
        pytest.skip("raw data file not downloaded")
    monkeypatch.setattr(ct, "ALLOWED_ROOTS", [*ct.ALLOWED_ROOTS, tmp_path])
    monkeypatch.setattr(ct, "WRITE_ROOTS", [*ct.WRITE_ROOTS, tmp_path])
    analyst, out = tmp_path / "analyst", tmp_path / "scientist"
    ct.run_clean_data(config, str(raw), str(analyst / "clean_data.csv"))
    ct.run_build_contract(config, str(analyst / "clean_data.csv"), str(raw),
                          str(analyst / "dataset_contract.json"))

    result = run_scientist(config, out_dir=out, clean_csv=analyst / "clean_data.csv",
                           contract_json=analyst / "dataset_contract.json")

    # Print before asserting, so time and tokens are never lost on a failure.
    print("\n=== RUN ===")
    print(json.dumps({"seconds": result["seconds"], "token_usage": result["token_usage"]}, indent=2))
    print("=== TASK OUTPUTS ===")
    for name, text in result["task_outputs"].items():
        print(f"--- {name} ---\n{text}\n")

    for name in ("contract_check.json", "features.csv", "model.pkl", "metrics.json",
                 "evaluation_report.md", "model_card.md"):
        assert (out / name).exists(), name
    metrics = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["selected"] == "v2"
    assert json.loads((out / "contract_check.json").read_text())["passed"] is True
    card = (out / "model_card.md").read_text(encoding="utf-8")
    assert insight_number_violations(card, metrics) == []
    table = metrics_table_md(metrics)
    assert METRICS_TABLE_PLACEHOLDER not in card and card.count(table) == 1
    # The style rules apply to what the agent wrote, i.e. with the placeholder.
    assert model_card_style_violations(card.replace(table, METRICS_TABLE_PLACEHOLDER), metrics) == []
    assert load_model(out / "model.pkl") is not None
