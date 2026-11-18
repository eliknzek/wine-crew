"""WineFlow with stub crews (no LLM): a passing run, and a controlled stop at every gate."""

import copy
import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from wine_crew.checks import FlowCheckError, check_features, expected_feature_columns
from wine_crew.flow import crew_cost, run_flow
from wine_crew.tools.contract_tools import build_contract, save_contract
from wine_crew.tools.data_tools import clean_data, load_contract_template, load_raw, read_clean_csv
from wine_crew.tools.model_report import evaluation_report_md, metrics_document
from wine_crew.tools.model_tools import build_features, rank_value_wines, save_model, train_models

FAKE_USAGE = {"total_tokens": 1500, "prompt_tokens": 1000, "cached_prompt_tokens": 400,
              "completion_tokens": 500, "successful_requests": 3}


@pytest.fixture
def flow_config(tmp_path, model_config, model_df):
    cfg = copy.deepcopy(model_config)
    analyst, sci = tmp_path / "artifacts" / "analyst", tmp_path / "artifacts" / "scientist"
    template = copy.deepcopy(load_contract_template(model_config))
    template["row_count"] = {"min": 1, "max": 1000}
    (tmp_path / "template.json").write_text(json.dumps(template))
    cfg["paths"].update({
        "raw_csv": str(tmp_path / "raw.csv"),
        "contract_template": str(tmp_path / "template.json"),
        "artifacts_analyst": str(analyst), "artifacts_scientist": str(sci),
        "clean_data": str(analyst / "clean_data.csv"), "contract": str(analyst / "dataset_contract.json"),
        "eda_report": str(analyst / "eda_report.html"), "insights": str(analyst / "insights.md"),
        "run_summary": str(tmp_path / "artifacts" / "run_summary.json"),
        "logs": str(tmp_path / "logs"),
    })
    raw = model_df.drop(columns=["vintage"]).assign(region_2=np.nan, taster_twitter_handle="@t")
    raw.index = range(len(raw))
    raw.to_csv(tmp_path / "raw.csv")
    return cfg


def stub_analyst(corrupt=None):
    def run(config):
        df, sha = load_raw(config)
        clean, _ = clean_data(df, config)
        out = config["paths"]
        _mk(out["clean_data"])
        clean.to_csv(out["clean_data"], index=False)
        save_contract(build_contract(clean, sha, config), out["contract"])
        if corrupt:
            corrupt(config)
        return {"seconds": 1.0, "token_usage": FAKE_USAGE}
    return run


def stub_scientist(corrupt=None, calls=None):
    def run(config):
        if calls is not None:
            calls.append("crew2")
        p = config["paths"]
        contract = json.loads(open(p["contract"]).read())
        clean = read_clean_csv(p["clean_data"], contract)
        sci = p["artifacts_scientist"]
        _mk(f"{sci}/x")
        build_features(clean, contract, config, "v2").to_csv(f"{sci}/features.csv", index=False)
        result = train_models(clean, contract, config)
        metrics = metrics_document(result, clean, contract, config)
        save_model(result["models"][result["selected"]], f"{sci}/model.pkl")
        with open(f"{sci}/metrics.json", "w") as f:
            json.dump(metrics, f)
        ranking = rank_value_wines(clean, config, result["selected"], top_n=10)
        with open(f"{sci}/evaluation_report.md", "w") as f:
            f.write(evaluation_report_md(metrics, ranking))
        if corrupt:
            corrupt(config)
        return {"seconds": 2.0, "token_usage": FAKE_USAGE}
    return run


def _mk(path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    return True


def _summary(cfg):
    return json.loads(open(cfg["paths"]["run_summary"]).read())


def _log_text(cfg):
    logs = sorted(Path(cfg["paths"]["logs"]).glob("run_*.log"))
    assert len(logs) == 1
    return logs[0].read_text()


def _assert_stopped(cfg, capsys, code, check, message_start):
    err = capsys.readouterr().err
    assert code == 1
    summary = _summary(cfg)
    assert summary["status"] == "failed"
    assert summary["failed_check"] == check
    assert summary["error"].startswith(message_start)
    assert "Traceback" not in err
    assert message_start in err and "Log: " in err
    log = _log_text(cfg)
    assert f"check {check}: FAILED" in log or f"failed in step {check}" in log
    return summary, log


# --------------------------------------------------------------------------- passing run


def test_passing_run(flow_config, capsys):
    code = run_flow(flow_config, stub_analyst(), stub_scientist())
    assert code == 0, capsys.readouterr().err
    s = _summary(flow_config)
    assert s["status"] == "passed" and "failed_check" not in s
    assert {k: v["passed"] for k, v in s["checks"].items()} == {
        "preflight": True, "contract": True, "features": True, "model": True}
    assert s["rows"]["raw"] == 600 and s["rows"]["clean"] == s["rows"]["features"] > 0
    assert s["rows"]["train"] + s["rows"]["test"] == s["rows"]["clean"]
    assert len(s["raw_sha256"]) == 64 and s["selected"] in ("v1", "v2")
    assert s["crews"]["crew1"]["cost_usd"] == crew_cost(FAKE_USAGE, flow_config)
    assert s["total_cost_usd"] == pytest.approx(2 * crew_cost(FAKE_USAGE, flow_config))
    log = _log_text(flow_config)
    assert s["run_id"] in log
    for step in ("preflight", "crew1", "gate_contract", "crew2", "gate_features", "gate_model"):
        assert f"step {step}: start" in log and f"step {step}: end (" in log
    for check in ("preflight", "contract", "features", "model"):
        assert f"check {check}: PASSED" in log
    assert "crew1 rows: in=600 out=" in log and "crew2 rows: in=" in log and "train rows: train=" in log
    assert "crew1 tokens: total=1500" in log and "crew2 tokens:" in log and "cost=$" in log
    assert s["raw_sha256"] in log
    assert re.match(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ INFO \[" + re.escape(s["run_id"]) + "]", log)


def test_crew_cost():
    cfg = {"llm": {"price_per_million": {"input": 0.15, "cached_input": 0.075, "output": 0.60}}}
    assert crew_cost(FAKE_USAGE, cfg) == pytest.approx((600 * 0.15 + 400 * 0.075 + 500 * 0.60) / 1e6)
    assert crew_cost(None, cfg) == 0.0


# --------------------------------------------------------------------------- preflight


def test_preflight_stops_without_raw_file(flow_config, capsys):
    os.remove(flow_config["paths"]["raw_csv"])
    calls = []
    code = run_flow(flow_config, stub_analyst(), stub_scientist(calls=calls))
    s, _ = _assert_stopped(flow_config, capsys, code, "preflight", "Preflight check failed: raw file not found")
    assert "crew1" not in s["crews"] and calls == []


# --------------------------------------------------------------------------- gate 1


def _edit_clean(fn):
    def corrupt(config):
        path = config["paths"]["clean_data"]
        fn(pd.read_csv(path)).to_csv(path, index=False)
    return corrupt


@pytest.mark.parametrize("corrupt, expected", [
    (_edit_clean(lambda d: d.drop(columns=["price"])), "missing column: price"),
    (_edit_clean(lambda d: d.assign(price=d["price"].where(d.index > 2))), "price"),
    (_edit_clean(lambda d: pd.concat([d, d.head(2)])), "duplicate"),
])
def test_gate1_stops_on_broken_clean_data(flow_config, capsys, corrupt, expected):
    calls = []
    code = run_flow(flow_config, stub_analyst(corrupt), stub_scientist(calls=calls))
    s, _ = _assert_stopped(flow_config, capsys, code, "contract", "Contract check failed")
    assert expected in s["error"]
    assert calls == []                         # Crew 2 never ran
    assert "features" not in s["checks"]


# --------------------------------------------------------------------------- gate 2


def _edit_features(fn):
    def corrupt(config):
        path = f"{config['paths']['artifacts_scientist']}/features.csv"
        fn(pd.read_csv(path)).to_csv(path, index=False)
    return corrupt


def _vintage_nan_without_flag(d):
    d.loc[d.index[d["vintage_missing"] == 0][:2], "vintage"] = np.nan
    return d


@pytest.mark.parametrize("corrupt, expected", [
    (_edit_features(lambda d: d.assign(taster_name="Taster A")), "taster_name is a leakage column"),
    (_edit_features(lambda d: d.drop(columns=["province"])), "missing feature column: province"),
    (_edit_features(lambda d: d.assign(country=d["country"].where(d.index > 0))), "country has 1 NaN"),
    (_edit_features(_vintage_nan_without_flag), "vintage has 2 NaN where vintage_missing is not 1"),
    (_edit_features(lambda d: d.iloc[1:]), "rows, clean_data has"),
])
def test_gate2_stops_on_broken_features(flow_config, capsys, corrupt, expected):
    code = run_flow(flow_config, stub_analyst(), stub_scientist(corrupt))
    s, _ = _assert_stopped(flow_config, capsys, code, "features", "Feature check failed")
    assert expected in s["error"]
    assert "model" not in s["checks"]


def test_features_allow_contract_nans(flow_config, model_df):
    contract = load_contract_template(flow_config)
    feats = build_features(model_df, contract, flow_config, "v2")
    assert feats["vintage"].isna().any() and feats["region_1"].isna().any()
    path = f"{flow_config['paths']['artifacts_scientist']}/f.csv"
    _mk(path)
    feats.to_csv(path, index=False)
    clean = f"{flow_config['paths']['artifacts_scientist']}/c.csv"
    model_df.to_csv(clean, index=False)
    assert check_features(path, clean, contract, flow_config)["columns"] == expected_feature_columns(flow_config)
    feats.assign(region_1=np.nan).to_csv(path, index=False)
    with pytest.raises(FlowCheckError, match=r"region_1 has 100.0% NaN \(allowed 20%\)"):
        check_features(path, clean, contract, flow_config)


# --------------------------------------------------------------------------- gate 3


def _worse_than_baseline(config):
    path = f"{config['paths']['artifacts_scientist']}/metrics.json"
    m = json.loads(open(path).read())
    m["models"][m["selected"]]["test"]["mae"] = m["models"]["baseline"]["test"]["mae"] + 0.07
    open(path, "w").write(json.dumps(m))


def _broken_pickle(config):
    open(f"{config['paths']['artifacts_scientist']}/model.pkl", "wb").write(b"\x00not a pickle")


@pytest.mark.parametrize("corrupt, expected", [
    (_worse_than_baseline, ">= baseline"),
    (_broken_pickle, "model.pkl could not be loaded"),
])
def test_gate3_stops_on_bad_model(flow_config, capsys, corrupt, expected):
    code = run_flow(flow_config, stub_analyst(), stub_scientist(corrupt))
    s, _ = _assert_stopped(flow_config, capsys, code, "model", "Model check failed")
    assert expected in s["error"]
    assert s["selected"] is None


def test_gate3_message_format(flow_config, capsys):
    run_flow(flow_config, stub_analyst(), stub_scientist(_worse_than_baseline))
    err = _summary(flow_config)["error"]
    assert re.fullmatch(r"Model check failed: MAE \d+\.\d\d >= baseline \d+\.\d\d", err)


# --------------------------------------------------------------------------- crash inside a crew


def test_crew_exception_is_a_controlled_stop(flow_config, capsys):
    def crashing(config):
        raise RuntimeError("Task failed guardrail validation after 0 retries.\nLast error: boom")

    code = run_flow(flow_config, crashing, stub_scientist())
    s, log = _assert_stopped(flow_config, capsys, code, "crew1",
                             "RuntimeError: Task failed guardrail validation after 0 retries.")
    assert "Traceback" in log                  # the traceback goes to the log only
