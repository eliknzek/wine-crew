"""What Streamlit Community Cloud needs: a slim app/requirements.txt pinned like the main lock,
and an app that imports no crewai and reads no .env or clean_data.csv."""

import os
import re
import subprocess
import sys

import pytest

from wine_crew.config import REPO_ROOT, load_config, resolve_path

APP_REQUIREMENTS = REPO_ROOT / "app" / "requirements.txt"


def _pins(path):
    pins = {}
    for line in path.read_text().splitlines():
        match = re.match(r"^([A-Za-z0-9_.-]+)==([^\s#]+)", line.strip())
        if match:
            pins[match.group(1).lower()] = match.group(2)
    return pins


def test_app_requirements_match_the_main_lock_and_skip_crewai():
    app, main = _pins(APP_REQUIREMENTS), _pins(REPO_ROOT / "requirements.txt")
    assert {"streamlit", "pandas", "numpy", "scikit-learn", "joblib", "pyyaml"} <= set(app)
    assert {name: app[name] for name in app} == {name: main[name] for name in app}
    assert not any(name.startswith("crewai") or name in ("python-dotenv", "openai") for name in app)
    assert all("==" in line or not line.strip() or line.lstrip().startswith("#")
               for line in APP_REQUIREMENTS.read_text().splitlines())   # no "-e ." or unpinned lines


PROBE = r"""
import os, sys
opened = []
sys.addaudithook(lambda e, a: opened.append(str(a[0])) if e == "open" and a else None)
from streamlit.testing.v1 import AppTest
at = AppTest.from_file(sys.argv[1], default_timeout=60)
at.run()
at.button(key="predict").click().run()
assert not at.exception, at.exception
assert any(m.label == "ניקוד צפוי" for m in at.metric)
bad = [m for m in sys.modules if m.split(".")[0] in ("crewai", "dotenv", "matplotlib", "openai")]
assert not bad, bad
assert not any(p.endswith("clean_data.csv") for p in opened)
assert not any(os.path.basename(p) == ".env" for p in opened)
print("probe ok")
"""


@pytest.mark.skipif(not resolve_path(load_config(), "value_ranking").exists(), reason="artifacts not present")
def test_app_runs_without_crewai_env_or_clean_data():
    env = {k: v for k, v in os.environ.items() if k != "OPENAI_API_KEY"}
    env.pop("PYTHONPATH", None)          # the app must find wine_crew through its own sys.path line
    result = subprocess.run([sys.executable, "-c", PROBE, str(REPO_ROOT / "app" / "streamlit_app.py")],
                            capture_output=True, text=True, env=env, cwd=REPO_ROOT, timeout=180)
    assert result.returncode == 0, result.stderr[-2000:]
    assert "probe ok" in result.stdout


def test_hero_image_is_a_small_webp_served_as_a_static_file():
    import tomllib

    hero = REPO_ROOT / "app" / "static" / "hero-vineyard.webp"
    head = hero.read_bytes()[:12]
    assert head[:4] == b"RIFF" and head[8:12] == b"WEBP"
    assert hero.stat().st_size <= 500_000
    config = tomllib.loads((REPO_ROOT / ".streamlit" / "config.toml").read_text())
    assert config["server"]["enableStaticServing"] is True
    css = (REPO_ROOT / "app" / "style.css").read_text()
    assert 'url("app/static/hero-vineyard.webp")' in css     # relative: works under Cloud's path prefix
    assert "base64" not in css
