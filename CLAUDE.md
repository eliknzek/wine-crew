# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

A CrewAI Flow (`WineFlow`) runs two crews in sequence, Data Analyst and then Data Scientist. It predicts the WineEnthusiast score `points` (80–100) from price, origin, variety and vintage, and ranks "value" wines that scored above what their price predicts. `docs/spec.md` (Hebrew) is the full spec and the source of truth for architecture, agents, gates and the work order ("סדר עבודה"). `docs/dataset_contract.json` is the interface between the two crews.

Current state: steps 1–5 and 7 are done (tools, both crews, `WineFlow` with its gates, logs and `run_summary.json`, and the Streamlit app in `app/streamlit_app.py` with its helpers in `src/wine_crew/app_data.py`). Do only the step you are asked for.

## Commands

```bash
uv venv --python 3.13 .venv && source .venv/bin/activate   # Python 3.13 (not 3.14)
uv pip install -r requirements.txt && uv pip install -e .
python scripts/download_data.py [--force]   # raw CSV -> data/raw/, prints sha256
pytest                                      # all tests
pytest tests/test_file.py::test_name        # single test
python -m wine_crew.main                    # run the whole Flow; exit 1 if a gate stops it
python -m wine_crew.main --demo-fail contract   # demo a controlled stop (contract|features|model); --raw PATH for another raw file
streamlit run app/streamlit_app.py          # the app (reads artifacts/ only, no API key)
```

To change dependencies, edit `requirements.in` and re-lock with `uv pip compile requirements.in --python-version 3.13 -o requirements.txt`. Never hand-edit the lock.

## Working principles

1. **All computation happens in deterministic Python tools** (`src/wine_crew/tools/`, pandas and scikit-learn) that use `seed` from `config.yaml` (42) for every split, CV and model. Agents decide, call tools and document. The LLM never computes numbers or invents ranges or thresholds, and writes text only from what the tools returned. LLM temperature is 0 (`config.yaml` → `llm`). Two runs on the same raw file must give identical rows, metrics and rankings.
2. **Every change goes on its own branch** (names from the spec: `crew1/cleaning`, `crew2/models`, `flow/checks`, …) and reaches `main` only through a PR opened with `gh`. The PR description follows `.github/pull_request_template.md`. Never commit or push directly to `main`. Artifacts in `artifacts/` (and that run's `logs/run_<timestamp>.log`, added with `git add -f`) are committed only from deliberate full runs, not after every run.
3. **Commit authorship:** commits are made as `eliknzek` (the configured git user), and every commit message ends with:
   `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`
   No other co-author lines.

## Architecture notes

- Thresholds, paths and hyperparameters live in `config.yaml`. Read them from there rather than hard-coding them.
- Gates in `checks.py` are pure Python and independent of the LLM. A failure raises `FlowCheckError`, and the Flow stops cleanly: it logs the check, writes the failed summary to `logs/run_<ts>_summary.json`, leaves `artifacts/` exactly as it was, and exits with code 1. Crews write into `build/runs/<run_id>/` (`paths.staging_root`); only when every gate passed does `promote` move `analyst/` and `scientist/` into `artifacts/` (rolling back both on any failure) and write `artifacts/run_summary.json`.
- The Streamlit app only reads `artifacts/` and `model.pkl`. It never runs the Flow and needs no API key. It must not import crewai or matplotlib (keep `app_data.py` light), and it adds `src/` to `sys.path` itself so it runs without installing the package. Form options come from the model's learned categories (`group_rare.keep_`), and the value screen reads `artifacts/scientist/value_ranking.csv`. Streamlit reloads only modules under `app/`, so `app/code_reload.py` evicts `wine_crew.*` when `src/wine_crew` changes; cached loaders take a file fingerprint (mtime, size) as their key, never an argument starting with `_` (Streamlit does not hash those).

## Dataset contract (docs/dataset_contract.json)

- **Feature roles:** columns tagged `leakage` (`description`, `taster_name`, which exist only after tasting), `identifier` (`winery`, `title`), `excluded` (`designation`) and `target` (`points`) must never be model features. Crew 2 fails deliberately if one slips in.
- **Derived columns:** `log_price` = ln(`price`). `vintage` is the first year between 1980 and 2017 found in `title` (nullable `Int64`, ≤5% null).
- **Dropped columns:** `region_2` and `taster_twitter_handle`. The cleaned data has exactly the listed columns and dtypes.
- **Constraints:** no full duplicates; no nulls in `points`, `price`, `country` or `variety`; 100k–130k rows; per-column min/max, `max_unique` and `max_null_pct` as listed.
- `source.sha256` is filled at run time by `load_raw`.

## Raw data quirks

- The first CSV column is an unnamed row index, so read with `index_col=0`.
- The raw file has 129,971 rows × 13 columns, with 9,983 full duplicates and ~9k missing `price`. After basic cleaning about 111.5k rows remain.
- `data/raw/` is git-ignored. Get the file with `scripts/download_data.py`.
