"""Stateful Python executor: maintains interpreter session across steps.

Like DataMind's CodeRunner, this keeps an InteractiveInterpreter with a
persistent namespace, so variables (DataFrames, results) survive across
code executions within the same task. This saves 2-3 steps per task that
would otherwise be spent re-loading data.

Key differences from the stateless `python_exec.py`:
1. Variables persist across calls (df loaded once, reused)
2. Checkpoint/rollback for recovery from bad code
3. Pre-imported data science libraries (pandas, numpy, scipy, etc.)
4. Timeout per execution (thread-based)
"""

from __future__ import annotations

import code
import contextlib
import copy
import io
import os
import sys
import threading
from pathlib import Path
from types import ModuleType
from typing import Any

logger = __import__("logging").getLogger("dabench.stateful_python")

# ── Pre-imported libraries (like DataMind's IMPORT_HELPER) ──

PRE_IMPORTS = [
    # Core
    "import math",
    "import re",
    "import json",
    "import sys",
    "import os",
    "import copy",
    "import datetime",
    "import statistics",
    "import functools",
    "import itertools",
    "import collections",
    # Data
    "import numpy",
    "import numpy as np",
    "import pandas",
    "import pandas as pd",
    "import duckdb",
    "import csv",
    "from pathlib import Path",
    # Serialization
    "import pickle",
    "import io as _io",
]

OPTIONAL_PRE_IMPORTS = [
    # Stats
    "from scipy import stats",
    "from scipy.stats import pearsonr, ttest_ind, norm, chi2_contingency",
    # ML
    "from sklearn.linear_model import LinearRegression, LogisticRegression",
    "from sklearn.model_selection import train_test_split",
    "from sklearn.preprocessing import StandardScaler",
    "from sklearn.metrics import mean_squared_error, accuracy_score",
]

# Modules that shouldn't be pickled (they get auto-reimported)
SKIP_STATE_TYPES = (ModuleType, type(io), type(sys))


def _detect_runtime_error(stderr: str) -> bool:
    """Check if stderr contains a Python runtime error traceback."""
    if not stderr or not stderr.strip():
        return False
    # Skip warnings (Commonly triggered by pandas, scipy, sklearn)
    stripped_lines = [line for line in stderr.split("\n") if line.strip()]
    if not stripped_lines:
        return False
    # Check for traceback marker
    if "Traceback (most recent call last)" in stderr:
        return True
    # Check for common error patterns in last few lines
    for line in stripped_lines[-5:]:
        if line.strip().startswith((
            "NameError", "TypeError", "ValueError", "KeyError",
            "AttributeError", "IndexError", "ZeroDivisionError",
            "FileNotFoundError", "ModuleNotFoundError", "ImportError",
            "SyntaxError", "IndentationError",
        )):
            return True
    return False


def _extract_error_line(stderr: str) -> str:
    """Extract the last meaningful error line from traceback."""
    stripped_lines = [line.strip() for line in stderr.split("\n") if line.strip()]
    for line in reversed(stripped_lines):
        # Skip "Traceback" header and file location lines
        if line.startswith("Traceback") or line.startswith("File "):
            continue
        if "Error" in line or len(line) > 20:
            return line
    return stripped_lines[-1] if stripped_lines else "unknown error"


class StatefulPythonInterpreter:
    """Interactive Python interpreter that keeps state across executions."""

    def __init__(self, context_root: Path, *, timeout: int = 30):
        self.context_root = Path(context_root).resolve()
        self.timeout = timeout
        self._interpreter = code.InteractiveInterpreter()
        self._lock = threading.Lock()
        self._exec_count = 0
        self._error_count = 0

        # Initialize with pre-imports
        self._init_namespace()

    def _init_namespace(self) -> None:
        """Set up the initial namespace with common data science imports."""
        for import_stmt in PRE_IMPORTS:
            self._interpreter.runsource(import_stmt, "<init>", "exec")
        for import_stmt in OPTIONAL_PRE_IMPORTS:
            stderr_capture = io.StringIO()
            with contextlib.redirect_stderr(stderr_capture):
                self._interpreter.runsource(import_stmt, "<init>", "exec")
            if stderr_capture.getvalue():
                logger.debug("optional pre-import failed: %s", import_stmt)

    @property
    def context_root_str(self) -> str:
        return str(self.context_root)

    def execute(self, code: str) -> dict[str, Any]:
        """Execute Python code in the persistent interpreter.

        Returns dict with: success, output, stderr, error, exec_count
        """
        with self._lock:
            return self._execute_locked(code)

    def _execute_locked(self, code: str) -> dict[str, Any]:
        self._exec_count += 1

        # Save state before execution (for rollback on error)
        saved_state = self._save_state()

        # Change to context root
        original_cwd = os.getcwd()
        try:
            os.chdir(self.context_root)
        except OSError:
            pass

        # Setup timeout
        timed_out = threading.Event()
        timer = threading.Timer(self.timeout, lambda: timed_out.set())
        timer.start()

        try:
            # Capture stdout and stderr
            stdout_capture = io.StringIO()
            stderr_capture = io.StringIO()

            with contextlib.redirect_stdout(stdout_capture), contextlib.redirect_stderr(stderr_capture):
                # Run the code
                more = self._interpreter.runsource(code, "<agent>", "exec")

            timer.cancel()

            output = stdout_capture.getvalue()
            stderr_out = stderr_capture.getvalue()

            # Check for runtime errors captured in stderr
            # (runsource catches exceptions internally, so we check stderr)
            has_runtime_error = _detect_runtime_error(stderr_out)

            if timed_out.is_set():
                self._restore_state(saved_state)
                self._error_count += 1
                return {
                    "success": False,
                    "output": output,
                    "stderr": stderr_out,
                    "error": f"Python execution timed out after {self.timeout} seconds.",
                    "exec_count": self._exec_count,
                    "state_rolled_back": True,
                }

            if has_runtime_error:
                self._restore_state(saved_state)
                self._error_count += 1
                error_line = _extract_error_line(stderr_out)
                return {
                    "success": False,
                    "output": output,
                    "stderr": stderr_out,
                    "error": error_line or stderr_out.strip().split("\n")[-2] if "\n" in stderr_out else stderr_out.strip(),
                    "exec_count": self._exec_count,
                    "state_rolled_back": True,
                }

            if more:
                output += "\n[interpreter: code is incomplete; expecting more input]"

            self._error_count = 0  # Reset error counter on success

            return {
                "success": True,
                "output": output,
                "stderr": stderr_out,
                "exec_count": self._exec_count,
                "more": more,
            }

        except Exception as exc:
            timer.cancel()
            self._restore_state(saved_state)
            self._error_count += 1
            return {
                "success": False,
                "output": "",
                "stderr": str(exc),
                "error": str(exc),
                "exec_count": self._exec_count,
                "state_rolled_back": True,
            }

        finally:
            try:
                os.chdir(original_cwd)
            except OSError:
                pass

    def execute_with_rollback(self, code: str) -> dict[str, Any]:
        """Execute code. If it fails, silently rollback and return error.

        Unlike execute(), this is the default behavior for agent tool calls.
        It maintains a clean state even when the LLM generates buggy code.
        """
        result = self.execute(code)
        return result

    def _save_state(self) -> dict[str, Any]:
        """Snapshot current interpreter namespace for rollback."""
        state: dict[str, Any] = {}
        for key, value in self._interpreter.locals.items():
            if key.startswith("__") and key.endswith("__"):
                continue
            try:
                if isinstance(value, SKIP_STATE_TYPES):
                    continue
                # Try shallow copy first, then deep copy
                if isinstance(value, (int, float, str, bool, type(None))):
                    state[key] = value
                elif isinstance(value, (list, dict)):
                    state[key] = copy.deepcopy(value)
                else:
                    # For pandas DataFrames etc., try copy
                    if hasattr(value, "copy"):
                        try:
                            state[key] = value.copy()
                        except Exception:
                            state[key] = f"<{type(value).__name__}: copy failed>"
                    else:
                        state[key] = f"<{type(value).__name__}: not serializable>"
            except Exception:
                state[key] = f"<{type(value).__name__}: save failed>"
        return state

    def _restore_state(self, state: dict[str, Any]) -> None:
        """Restore namespace from a saved state."""
        self._interpreter.locals.clear()
        self._init_namespace()
        for key, value in state.items():
            self._interpreter.locals[key] = value

    def get_state(self) -> dict[str, Any]:
        """Get current interpreter state (for serialization)."""
        with self._lock:
            return self._save_state()

    def set_state(self, state: dict[str, Any]) -> None:
        """Restore interpreter from saved state."""
        with self._lock:
            self._restore_state(state)

    def reset(self) -> None:
        """Clear all state and reinitialize."""
        with self._lock:
            self._interpreter.locals.clear()
            self._init_namespace()
            self._exec_count = 0
            self._error_count = 0

    @property
    def error_count(self) -> int:
        return self._error_count

    @property
    def exec_count(self) -> int:
        return self._exec_count


# ── Thread-safe global registry (one interpreter per task) ──

_TASK_INTERPRETERS: dict[str, StatefulPythonInterpreter] = {}
_REGISTRY_LOCK = threading.Lock()


def get_or_create_interpreter(task_id: str, context_root: Path, *, timeout: int = 30) -> StatefulPythonInterpreter:
    """Get or create a stateful interpreter for a task. Thread-safe."""
    with _REGISTRY_LOCK:
        if task_id not in _TASK_INTERPRETERS:
            _TASK_INTERPRETERS[task_id] = StatefulPythonInterpreter(
                context_root, timeout=timeout,
            )
        return _TASK_INTERPRETERS[task_id]


def has_interpreter(task_id: str) -> bool:
    """Check if an interpreter exists for the given task."""
    with _REGISTRY_LOCK:
        return task_id in _TASK_INTERPRETERS


def remove_interpreter(task_id: str) -> None:
    """Remove and cleanup interpreter for a task."""
    with _REGISTRY_LOCK:
        _TASK_INTERPRETERS.pop(task_id, None)


def reset_all_interpreters() -> None:
    """Reset all interpreters (e.g., between benchmark runs)."""
    with _REGISTRY_LOCK:
        _TASK_INTERPRETERS.clear()


def execute_stateful_python(task_id: str, context_root: Path, code: str, *, timeout: int = 30) -> dict[str, Any]:
    """Main entry point: execute Python code with stateful interpreter.

    Args:
        task_id: Unique task identifier (for interpreter lookup)
        context_root: Task context directory (working dir for execution)
        code: Python code to execute
        timeout: Per-execution timeout in seconds

    Returns:
        dict with success, output, stderr, error, exec_count
    """
    interpreter = get_or_create_interpreter(task_id, context_root, timeout=timeout)
    return interpreter.execute_with_rollback(code)
