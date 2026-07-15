"""
Scheduler plugin loader.

Scans the top-level ``schedulers/`` directory for user-written algorithms and
returns them as ``{display_name: Scheduler subclass}``, ready to populate the
dashboard dropdown. Each plugin is a plain ``.py`` file that subclasses
``Scheduler`` (see Scheduling_control/scheduler.py for the contract) and sets a
``NAME`` attribute used as the display label.

Loading is by file path (importlib), so dropping a new script into the folder
and re-running :func:`load_schedulers` -- e.g. via the dashboard's "Reload
schedulers" button -- picks it up without restarting the app.
"""

import importlib.util
import inspect
import traceback
from pathlib import Path

from Scheduling_control.scheduler import Scheduler

# schedulers/ lives at the project root, one level up from this package.
SCHEDULERS_DIR = Path(__file__).resolve().parent.parent / "schedulers"


def _import_from_path(path):
    """Import a single plugin file as a throwaway module and return it.

    A fresh module object is built on every call, so re-scanning re-executes
    the file and reflects edits made since the last load."""
    mod_name = f"_scheduler_plugin_{path.stem}"
    spec = importlib.util.spec_from_file_location(mod_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_schedulers(directory=SCHEDULERS_DIR):
    """Return ``{display_name: Scheduler subclass}`` for every plugin found.

    Files whose name starts with ``_`` are skipped. Only classes *defined in*
    the plugin file are considered (the imported ``Scheduler`` base and any
    other imports are ignored). A file that fails to import is skipped with a
    warning rather than taking the whole scan down."""
    directory = Path(directory)
    result = {}
    if not directory.is_dir():
        return result

    for path in sorted(directory.glob("*.py")):
        if path.name.startswith("_"):
            continue
        try:
            module = _import_from_path(path)
        except Exception:  # a broken plugin shouldn't break the others
            print(f"[schedulers] failed to load {path.name}:")
            traceback.print_exc()
            continue

        for _, obj in inspect.getmembers(module, inspect.isclass):
            if (
                issubclass(obj, Scheduler)
                and obj is not Scheduler
                and obj.__module__ == module.__name__  # defined here, not imported
            ):
                display = getattr(obj, "NAME", None) or obj.__name__
                result[display] = obj

    return result
