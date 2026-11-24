"""Reload src/wine_crew after an update, without relying on Streamlit's file watcher.

Streamlit reloads only modules under the entrypoint's folder (app/) or PYTHONPATH,
so after a git pull the app script is new but wine_crew.* stays the old version in
sys.modules (that caused the AttributeError on localize_countries). This module
lives in app/, so Streamlit itself reloads it when it changes.
"""

from __future__ import annotations

import sys
from pathlib import Path

PACKAGE = "wine_crew"
# Helper modules in app/ that import wine_crew: a running server does not reload them either,
# and they keep a reference to the old wine_crew, so they are reloaded together with it.
APP_MODULES = {"charts": Path(__file__).resolve().parent / "charts.py",
               "karmi": Path(__file__).resolve().parent / "karmi.py"}
# Kept on the sys module so it survives a reload of this module too.
_STATE_ATTR = "_wine_crew_code_fingerprint"


def code_fingerprint(src_dir: Path) -> tuple:
    """(relative path, mtime_ns, size) of every .py file in the package, sorted."""
    src_dir = Path(src_dir)
    items = []
    for path in sorted(src_dir.rglob("*.py")):
        stat = path.stat()
        items.append((str(path.relative_to(src_dir)), stat.st_mtime_ns, stat.st_size))
    return tuple(items)


def evict_package(package: str | None = None) -> list[str]:
    """Remove the package and its submodules from sys.modules, so the next import is fresh."""
    package = package or PACKAGE
    names = [n for n in sys.modules if n == package or n.startswith(package + ".")]
    for name in names:
        sys.modules.pop(name, None)
    return names


def current_fingerprint(src_dir: Path, app_modules: dict[str, Path] | None = None) -> tuple:
    """The package's fingerprint plus (name, mtime_ns, size) of each app module."""
    app_modules = APP_MODULES if app_modules is None else app_modules
    return code_fingerprint(src_dir) + tuple(
        (name, path.stat().st_mtime_ns, path.stat().st_size) for name, path in sorted(app_modules.items()))


def reload_if_changed(src_dir: Path, clear_caches=None, app_modules: dict[str, Path] | None = None) -> bool:
    """Evict wine_crew.* and the app modules (and clear Streamlit caches) when any of their
    source files changed.

    Returns True when it evicted. With no fingerprint recorded yet but wine_crew already
    imported (this module was itself reloaded), it evicts too, to be safe.
    """
    app_modules = APP_MODULES if app_modules is None else app_modules
    current = current_fingerprint(src_dir, app_modules)
    previous = getattr(sys, _STATE_ATTR, None)
    setattr(sys, _STATE_ATTR, current)
    loaded = PACKAGE in sys.modules
    if previous == current or (previous is None and not loaded):
        return False
    evict_package()
    for name in app_modules:
        sys.modules.pop(name, None)
    if clear_caches is not None:
        clear_caches()
    return True
