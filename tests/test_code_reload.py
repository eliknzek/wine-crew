"""app/code_reload.py: reload src/wine_crew when its files change (no Streamlit needed)."""

import os
import sys
import time

import pytest

from wine_crew.config import REPO_ROOT

sys.path.insert(0, str(REPO_ROOT / "app"))
import code_reload  # noqa: E402


@pytest.fixture
def fake_package(tmp_path, monkeypatch):
    """A tiny package named like the real one, so eviction can be checked safely."""
    pkg = tmp_path / "fakepkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    (pkg / "mod.py").write_text("VALUE = 1\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setattr(code_reload, "PACKAGE", "fakepkg")
    monkeypatch.delattr(sys, code_reload._STATE_ATTR, raising=False)
    yield pkg
    code_reload.evict_package("fakepkg")
    monkeypatch.delattr(sys, code_reload._STATE_ATTR, raising=False)


def _touch(path, text):
    path.write_text(text)
    later = time.time_ns() + 5_000_000_000
    os.utime(path, ns=(later, later))


def test_fingerprint_changes_only_when_a_file_changes(fake_package):
    first = code_reload.code_fingerprint(fake_package)
    assert first == code_reload.code_fingerprint(fake_package)
    assert [item[0] for item in first] == ["__init__.py", "mod.py"]
    _touch(fake_package / "mod.py", "VALUE = 2\n")
    assert code_reload.code_fingerprint(fake_package) != first


def test_reload_evicts_only_after_a_change(fake_package):
    import importlib

    clears = []
    mod = importlib.import_module("fakepkg.mod")
    # First call with the package already loaded: evicts once, to be safe.
    assert code_reload.reload_if_changed(fake_package, clear_caches=lambda: clears.append(1)) is True
    mod = importlib.import_module("fakepkg.mod")
    assert code_reload.reload_if_changed(fake_package, clear_caches=lambda: clears.append(1)) is False
    assert sys.modules["fakepkg.mod"] is mod and mod.VALUE == 1

    _touch(fake_package / "mod.py", "VALUE = 2\n")
    assert code_reload.reload_if_changed(fake_package, clear_caches=lambda: clears.append(1)) is True
    assert "fakepkg" not in sys.modules and "fakepkg.mod" not in sys.modules
    fresh = importlib.import_module("fakepkg.mod")
    assert fresh is not mod and fresh.VALUE == 2
    assert clears == [1, 1]


def test_first_call_without_loaded_package_does_nothing(fake_package):
    assert "fakepkg" not in sys.modules
    assert code_reload.reload_if_changed(fake_package) is False



def test_a_changed_app_module_evicts_it_with_the_package(fake_package, tmp_path):
    """app/charts.py-style modules: a change there evicts the package and the module."""
    helper = tmp_path / "helper_mod.py"
    helper.write_text("X = 1\n")
    modules = {"helper_mod": helper}
    import helper_mod  # noqa: F401
    import fakepkg.mod  # noqa: F401
    assert code_reload.reload_if_changed(fake_package, app_modules=modules) is True   # first: loaded, evict
    import helper_mod  # noqa: F401,F811
    import fakepkg.mod  # noqa: F401,F811
    assert code_reload.reload_if_changed(fake_package, app_modules=modules) is False  # unchanged
    later = time.time_ns() + 5_000_000_000
    helper.write_text("X = 22\n")
    os.utime(helper, ns=(later, later))
    assert code_reload.reload_if_changed(fake_package, app_modules=modules) is True
    assert "helper_mod" not in sys.modules and "fakepkg" not in sys.modules
    sys.modules.pop("helper_mod", None)
