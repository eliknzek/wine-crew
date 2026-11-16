import json

import pandas as pd
import pytest

from wine_crew.tools import crewai_tools as ct


@pytest.fixture
def workdir(tmp_path, monkeypatch, raw_df):
    monkeypatch.setattr(ct, "ALLOWED_ROOTS", [*ct.ALLOWED_ROOTS, tmp_path])
    monkeypatch.setattr(ct, "WRITE_ROOTS", [*ct.WRITE_ROOTS, tmp_path])
    raw_df.to_csv(tmp_path / "raw.csv")
    return tmp_path


@pytest.fixture
def pipeline_outputs(workdir, config):
    raw, clean = workdir / "raw.csv", workdir / "out" / "clean_data.csv"
    ct.run_clean_data(config, str(raw), str(clean))
    summary = workdir / "out" / "eda_summary.json"
    ct.run_eda_summary(config, str(clean), str(summary))
    return {"raw": raw, "clean": clean, "summary": summary, "dir": workdir / "out"}


def test_load_raw(workdir, config):
    result = ct.run_load_raw(config, str(workdir / "raw.csv"))
    assert result["rows"] == 35
    assert result["fully_duplicated_rows"] == 1
    assert result["nulls_in_required_columns"] == {"price": 1, "country": 1, "variety": 0}
    assert len(result["sha256"]) == 64


def test_clean_data_writes_csv(pipeline_outputs):
    df = pd.read_csv(pipeline_outputs["clean"])
    assert len(df) == 32
    assert "taster_twitter_handle" not in df.columns


def test_eda_tools_write_files(pipeline_outputs, config):
    summary = json.loads(pipeline_outputs["summary"].read_text())
    assert summary["row_count"] == 32
    charts = ct.run_make_charts(config, str(pipeline_outputs["clean"]), str(pipeline_outputs["dir"] / "charts"))
    assert len(charts["charts"]) == 7
    report = ct.run_build_html_report(
        str(pipeline_outputs["summary"]), str(pipeline_outputs["dir"] / "charts"),
        str(pipeline_outputs["dir"] / "eda_report.html"),
    )
    assert report["charts_embedded"] == 7
    assert ct.run_read_eda_summary(str(pipeline_outputs["summary"])) == summary


def test_contract_tools(pipeline_outputs, config):
    contract_path = pipeline_outputs["dir"] / "dataset_contract.json"
    built = ct.run_build_contract(
        config, str(pipeline_outputs["clean"]), str(pipeline_outputs["raw"]), str(contract_path))
    assert built["row_count"] == 32 and len(built["sha256"]) == 64
    result = ct.run_validate_contract(config, str(pipeline_outputs["clean"]), str(contract_path))
    # The tiny sample is below the contract's 100k-row minimum; nothing else is wrong.
    assert result == {"passed": False, "violations": ["row count 32 is below min 100000"]}


def test_validate_contract_reports_broken_csv(pipeline_outputs, config):
    contract_path = pipeline_outputs["dir"] / "dataset_contract.json"
    ct.run_build_contract(config, str(pipeline_outputs["clean"]), str(pipeline_outputs["raw"]), str(contract_path))
    broken = pipeline_outputs["dir"] / "broken.csv"
    pd.read_csv(pipeline_outputs["clean"]).drop(columns=["price"]).to_csv(broken, index=False)
    result = ct.run_validate_contract(config, str(broken), str(contract_path))
    assert result["passed"] is False
    assert "missing column: price" in result["violations"]


def test_paths_outside_allowed_roots_are_rejected(config):
    with pytest.raises(ValueError, match="outside the allowed"):
        ct.run_load_raw(config, "/etc/passwd")
    with pytest.raises(ValueError, match="outside the allowed"):
        ct.run_read_eda_summary("../../outside.json")


def test_outputs_are_limited_to_artifacts(workdir, config):
    raw = str(workdir / "raw.csv")
    for target in ("data/raw/winemag-data-130k-v2.csv", "data/raw/new.csv", "src/wine_crew/x.csv",
                   "artifacts/../data/raw/x.csv", "config.yaml"):
        with pytest.raises(ValueError, match="inside artifacts/"):
            ct.run_clean_data(config, raw, target)
    with pytest.raises(ValueError, match="inside artifacts/"):
        ct.run_eda_summary(config, raw, "docs/eda_summary.json")
    with pytest.raises(ValueError, match="inside artifacts/"):
        ct.run_make_charts(config, raw, "data/raw/charts")


def test_basetools_return_json_strings(pipeline_outputs, config):
    tool = ct.ReadEdaSummaryTool()
    out = tool.run(summary_json_path=str(pipeline_outputs["summary"]))
    assert isinstance(out, str) and json.loads(out)["row_count"] == 32
    out = ct.LoadRawTool(config=config).run(raw_csv_path=str(pipeline_outputs["raw"]))
    assert json.loads(out)["rows"] == 35
    names = {t().name for t in (ct.LoadRawTool, ct.CleanDataTool, ct.EdaSummaryTool, ct.MakeChartsTool,
                                ct.BuildHtmlReportTool, ct.ReadEdaSummaryTool, ct.BuildContractTool,
                                ct.ValidateContractTool)}
    assert names == {"load_raw", "clean_data", "eda_summary", "make_charts", "build_html_report",
                     "read_eda_summary", "build_contract", "validate_contract"}
