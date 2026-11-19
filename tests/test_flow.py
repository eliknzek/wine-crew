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
from wine_crew.tools.crewai_tools import VALUE_RANKING_COLUMNS
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
    # Every artifacts/ path points into tmp, so no test can touch the real artifacts/.
    cfg["paths"].update({k: str(tmp_path / v) for k, v in cfg["paths"].items()
                         if isinstance(v, str) and v.startswith("artifacts/")})
    cfg["paths"].update({
        "staging_root": str(tmp_path / "build" / "runs"),
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
        ranking = rank_value_wines(clean, config, result["selected"])
        with open(f"{sci}/evaluation_report.md", "w") as f:
            f.write(evaluation_report_md(metrics, ranking.head(10)))
        ranking.assign(rank=range(1, len(ranking) + 1))[VALUE_RANKING_COLUMNS].to_csv(
            f"{sci}/value_ranking.csv", index=False)
        if corrupt:
            corrupt(config)
        return {"seconds": 2.0, "token_usage": FAKE_USAGE}
    return run


def _mk(path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    return True


def _summary(cfg):
    """artifacts/run_summary.json: written only by a run that passed and was promoted."""
    return json.loads(open(cfg["paths"]["run_summary"]).read())


def _failed_summary(cfg):
    """logs/run_<ts>_summary.json: where a failed run writes its summary."""
    found = sorted(Path(cfg["paths"]["logs"]).glob("run_*_summary.json"))
    assert len(found) == 1, found
    return json.loads(found[0].read_text())


def _log_text(cfg):
    logs = sorted(Path(cfg["paths"]["logs"]).glob("run_*.log"))
    assert len(logs) == 1
    return logs[0].read_text()


def _assert_stopped(cfg, capsys, code, check, message_start):
    err = capsys.readouterr().err
    assert code == 1
    summary = _failed_summary(cfg)
    assert summary["status"] == "failed"
    assert "artifacts/ was not changed" in err
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
    assert s["artifacts"]["value_ranking"].endswith("value_ranking.csv")
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


def _edit_ranking(fn):
    def corrupt(config):
        path = f"{config['paths']['artifacts_scientist']}/value_ranking.csv"
        fn(path)
    return corrupt


@pytest.mark.parametrize("corrupt, expected", [
    (_worse_than_baseline, ">= baseline"),
    (_broken_pickle, "model.pkl could not be loaded"),
    (_edit_ranking(os.remove), "value_ranking.csv not found"),
    (_edit_ranking(lambda p: pd.read_csv(p).iloc[1:].to_csv(p, index=False)),
     "value_ranking.csv has 599 rows, clean_data has 600"),
    (_edit_ranking(lambda p: pd.read_csv(p).drop(columns=["value_score"]).to_csv(p, index=False)),
     "value_ranking.csv is missing columns: value_score"),
])
def test_gate3_stops_on_bad_model(flow_config, capsys, corrupt, expected):
    code = run_flow(flow_config, stub_analyst(), stub_scientist(corrupt))
    s, _ = _assert_stopped(flow_config, capsys, code, "model", "Model check failed")
    assert expected in s["error"]
    assert s["selected"] is None


def test_gate3_message_format(flow_config, capsys):
    run_flow(flow_config, stub_analyst(), stub_scientist(_worse_than_baseline))
    err = _failed_summary(flow_config)["error"]
    assert re.fullmatch(r"Model check failed: MAE \d+\.\d\d >= baseline \d+\.\d\d", err)


# --------------------------------------------------------------------------- crash inside a crew


def test_crew_exception_is_a_controlled_stop(flow_config, capsys):
    def crashing(config):
        raise RuntimeError("Task failed guardrail validation after 0 retries.\nLast error: boom")

    code = run_flow(flow_config, crashing, stub_scientist())
    s, log = _assert_stopped(flow_config, capsys, code, "crew1",
                             "RuntimeError: Task failed guardrail validation after 0 retries.")
    assert "Traceback" in log                  # the traceback goes to the log only


def test_timestamps_are_utc_with_z(flow_config, capsys):
    from wine_crew.flow import utc_stamp
    from datetime import datetime, timezone, timedelta

    assert utc_stamp(datetime(2026, 11, 19, 12, 34, 51, tzinfo=timezone(timedelta(hours=2)))) == "2026-11-19T10:34:51Z"
    assert run_flow(flow_config, stub_analyst(), stub_scientist()) == 0
    s = _summary(flow_config)
    stamp = r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ"
    assert re.fullmatch(stamp, s["started_at"]) and re.fullmatch(stamp, s["finished_at"])
    assert all(re.match(stamp + " ", line) for line in _log_text(flow_config).splitlines() if line.strip())


# --------------------------------------------------------------------------- staging and promotion


import hashlib  # noqa: E402

from wine_crew import flow as flow_module  # noqa: E402
from wine_crew.tools import crewai_tools as ct  # noqa: E402


def _snapshot(root):
    """sha256 of every file under root (relative path -> digest)."""
    root = Path(root)
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.rglob("*")) if p.is_file()}


@pytest.fixture
def promoted(flow_config, capsys):
    """artifacts/ after one passing run, and its snapshot."""
    assert run_flow(flow_config, stub_analyst(), stub_scientist()) == 0
    capsys.readouterr()
    for log in Path(flow_config["paths"]["logs"]).glob("run_*"):
        log.unlink()                                   # later checks look at the next run only
    art = Path(flow_config["paths"]["run_summary"]).parent
    return art, _snapshot(art)


def test_passing_run_promotes_and_cleans_up(flow_config, promoted):
    art, snap = promoted
    assert {"analyst/.gitkeep", "scientist/.gitkeep", "run_summary.json",
            "analyst/clean_data.csv", "scientist/model.pkl", "scientist/value_ranking.csv"} <= set(snap)
    staging = Path(flow_config["paths"]["staging_root"])
    assert not staging.exists() or list(staging.iterdir()) == []      # the run folder was removed
    assert all("build" not in str(root) for root in ct.WRITE_ROOTS)    # write access closed again
    assert _summary(flow_config)["artifacts"]["clean_csv"].endswith("artifacts/analyst/clean_data.csv")


def _assert_artifacts_unchanged(flow_config, art, snap, code, check):
    assert code == 1
    assert _snapshot(art) == snap                                      # byte-identical
    failed = _failed_summary(flow_config)
    assert failed["failed_check"] == check
    staging_dir = Path(flow_config["paths"]["staging_root"]) / failed["run_id"]
    assert staging_dir.is_dir() and failed["staging_dir"].endswith(failed["run_id"])
    return failed


@pytest.mark.parametrize("make_run, check", [
    (lambda: (stub_analyst(_edit_clean(lambda d: d.drop(columns=["price"]))), stub_scientist()), "contract"),
    (lambda: (stub_analyst(), stub_scientist(_edit_features(lambda d: d.assign(taster_name="x")))), "features"),
    (lambda: (stub_analyst(), stub_scientist(_worse_than_baseline)), "model"),
    (lambda: (lambda config: (_ for _ in ()).throw(RuntimeError("boom")), stub_scientist()), "crew1"),
])
def test_failed_run_leaves_artifacts_byte_identical(flow_config, promoted, capsys, make_run, check):
    art, snap = promoted
    analyst, scientist = make_run()
    code = run_flow(flow_config, analyst, scientist)
    _assert_artifacts_unchanged(flow_config, art, snap, code, check)
    assert "artifacts/ was not changed" in capsys.readouterr().err


def test_preflight_failure_leaves_artifacts_byte_identical(flow_config, promoted):
    art, snap = promoted
    os.remove(flow_config["paths"]["raw_csv"])
    _assert_artifacts_unchanged(flow_config, art, snap, run_flow(flow_config, stub_analyst(), stub_scientist()),
                                "preflight")


@pytest.mark.parametrize("fail_on_call", [1, 2, 3, 4])
def test_promotion_failure_rolls_back_both_folders(flow_config, promoted, monkeypatch, fail_on_call):
    """os.replace calls: 1 analyst->backup, 2 staged analyst->artifacts, 3 scientist->backup,
    4 staged scientist->artifacts. Failing 3 or 4 happens after analyst was already replaced."""
    art, snap = promoted
    real_replace, calls = os.replace, {"n": 0}

    def flaky_replace(src, dst):
        calls["n"] += 1
        if calls["n"] == fail_on_call:
            raise OSError("simulated failure")
        return real_replace(src, dst)

    monkeypatch.setattr(flow_module.os, "replace", flaky_replace)
    code = run_flow(flow_config, stub_analyst(), stub_scientist())
    failed = _assert_artifacts_unchanged(flow_config, art, snap, code, "promote")
    assert "simulated failure" in failed["error"]
    staging_dir = Path(flow_config["paths"]["staging_root"]) / failed["run_id"]
    assert (staging_dir / "analyst" / "clean_data.csv").exists()          # staged files are back
    assert (staging_dir / "scientist" / "model.pkl").exists()


# --------------------------------------------------------------------------- demo failures


@pytest.mark.parametrize("gate, message", [
    ("contract", "Contract check failed: missing column: price"),
    ("features", "Feature check failed: taster_name is a leakage column"),
    ("model", "Model check failed: MAE"),
])
def test_demo_failure_stops_at_its_gate_without_touching_artifacts(flow_config, promoted, capsys, gate, message):
    art, snap = promoted
    code = run_flow(flow_config, stub_analyst(), stub_scientist(), demo_fail=gate)
    failed = _assert_artifacts_unchanged(flow_config, art, snap, code, gate)
    assert failed["error"].startswith(message) and failed["demo_fail"] == gate
    log = _log_text(flow_config)
    assert f"DEMO: this run will fail at gate {gate} on purpose" in log and "(staging folder only)" in log
    assert message in capsys.readouterr().err


def test_unknown_demo_failure_is_rejected(flow_config):
    with pytest.raises(ValueError, match="unknown demo failure"):
        run_flow(flow_config, stub_analyst(), stub_scientist(), demo_fail="nope")


def test_cli_flags(monkeypatch, tmp_path):
    from wine_crew import main as main_module

    seen = {}
    monkeypatch.setattr(main_module, "run_flow", lambda config=None, demo_fail=None: seen.update(
        raw=config["paths"]["raw_csv"], demo=demo_fail) or 1)
    assert main_module.main(["--demo-fail", "features", "--raw", str(tmp_path / "x.csv")]) == 1
    assert seen == {"raw": str(tmp_path / "x.csv"), "demo": "features"}
    assert main_module.main([]) == 1 and seen["demo"] is None


def test_requests_by_task_are_logged_and_summarized(flow_config):
    by_task = {"clean_task": {"requests": 2, "guardrail_retries": 0},
               "insights_task": {"requests": 4, "guardrail_retries": 1}}
    analyst = stub_analyst()

    def counted(config):
        return {**analyst(config), "requests_by_task": by_task}

    assert run_flow(flow_config, counted, stub_scientist()) == 0
    assert _summary(flow_config)["crews"]["crew1"]["requests_by_task"] == by_task
    assert "crew1 requests by task: clean_task=2 insights_task=4 (guardrail retries 1)" in _log_text(flow_config)
