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


# --------------------------------------------------------------------------- Crew 2 tools


@pytest.fixture
def model_inputs(workdir, model_df, small_contract):
    clean, contract = workdir / "model_clean.csv", workdir / "contract.json"
    model_df.to_csv(clean, index=False)
    contract.write_text(json.dumps(small_contract))
    return {"clean": str(clean), "contract": str(contract), "dir": workdir / "sci"}


def test_validate_contract_writes_check_file(model_inputs, config):
    out = model_inputs["dir"] / "contract_check.json"
    result = ct.run_validate_contract(config, model_inputs["clean"], model_inputs["contract"], str(out))
    assert result["passed"] is True
    assert json.loads(out.read_text()) == {"passed": True, "violations": []}
    with pytest.raises(ValueError, match="inside artifacts/"):
        ct.run_validate_contract(config, model_inputs["clean"], model_inputs["contract"], "data/raw/x.json")


def test_build_features_writes_only_allowed_columns(model_inputs, model_config):
    out = model_inputs["dir"] / "features.csv"
    result = ct.run_build_features(model_config, model_inputs["clean"], model_inputs["contract"], str(out))
    assert result["passed"] is True
    df = pd.read_csv(out)
    assert list(df.columns) == result["columns"]
    assert {"log_price", "vintage", "vintage_missing", "region_1"} <= set(df.columns)
    assert not {"description", "taster_name", "title", "winery", "designation", "points"} & set(df.columns)


def test_build_features_with_forbidden_column_fails_without_writing(model_inputs, model_config):
    cfg = {**model_config, "features": {**model_config["features"],
                                        "categorical": [*model_config["features"]["categorical"], "description"]}}
    out = model_inputs["dir"] / "features.csv"
    result = ct.run_build_features(cfg, model_inputs["clean"], model_inputs["contract"], str(out))
    assert result == {"passed": False, "violations": ["Feature check failed: description is a leakage column"]}
    assert not out.exists()


def test_train_models_writes_model_metrics_and_report(model_inputs, model_config, model_df):
    from wine_crew.tools.model_tools import load_model

    d = model_inputs["dir"]
    result = ct.run_train_models(model_config, model_inputs["clean"], model_inputs["contract"],
                                 str(d / "model.pkl"), str(d / "metrics.json"), str(d / "evaluation_report.md"),
                                 str(d / "value_ranking.csv"))
    metrics = json.loads((d / "metrics.json").read_text())
    assert result["selected"] == metrics["selected"]
    assert set(result["test_mae"]) == {"baseline", "v1", "v2"}
    model = load_model(d / "model.pkl")
    cols = metrics["features"]["input_columns"][metrics["selected"]]
    assert len(model.predict(model_df[cols].head(5))) == 5
    report = (d / "evaluation_report.md").read_text()
    assert "## מדדים" in report and report.count("\n| ") >= 50
    assert ct.run_read_metrics(str(d / "metrics.json")) == metrics
    ranking = pd.read_csv(d / "value_ranking.csv")
    assert list(ranking.columns) == ct.VALUE_RANKING_COLUMNS
    assert len(ranking) == result["ranked_wines"] == len(model_df)
    assert ranking["rank"].tolist() == list(range(1, len(model_df) + 1))
    assert ranking["value_score"].is_monotonic_decreasing
    with pytest.raises(ValueError, match="inside artifacts/"):
        ct.run_train_models(model_config, model_inputs["clean"], model_inputs["contract"],
                            "src/model.pkl", str(d / "m.json"), str(d / "r.md"), str(d / "v.csv"))


def test_crew2_basetool_names():
    names = {t().name for t in (ct.BuildFeaturesTool, ct.TrainModelsTool, ct.ReadMetricsTool)}
    assert names == {"build_features", "train_models", "read_metrics"}
