from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from data_agent_baseline.benchmark.schema import PublicTask, TaskAssets, TaskRecord
from data_agent_baseline.tools.duckdb_sql import execute_data_sql
from data_agent_baseline.tools.profiler import profile_context
from data_agent_baseline.tools.registry import _answer


def _make_task(tmp_path: Path) -> PublicTask:
    task_dir = tmp_path / "task_demo_1"
    context_dir = task_dir / "context"
    (context_dir / "csv").mkdir(parents=True)
    (context_dir / "json").mkdir()
    (context_dir / "db").mkdir()

    (context_dir / "csv" / "sales.csv").write_text(
        "region,amount\nEast,10\nWest,20\nEast,5\n",
        encoding="utf-8",
    )
    (context_dir / "json" / "users.json").write_text(
        json.dumps([{"id": 1, "name": "Ada"}, {"id": 2, "name": "Lin"}]),
        encoding="utf-8",
    )
    (context_dir / "knowledge.md").write_text(
        "# Demo Knowledge\n\nEast and West are valid sales regions.\n",
        encoding="utf-8",
    )

    with sqlite3.connect(context_dir / "db" / "demo.db") as conn:
        conn.execute("CREATE TABLE customers (id INTEGER, region TEXT)")
        conn.executemany("INSERT INTO customers VALUES (?, ?)", [(1, "East"), (2, "West")])

    return PublicTask(
        record=TaskRecord(
            task_id="task_demo_1",
            difficulty="easy",
            question="What is total sales by region?",
        ),
        assets=TaskAssets(task_dir=task_dir, context_dir=context_dir),
    )


def test_profile_context_detects_supported_sources(tmp_path: Path) -> None:
    task = _make_task(tmp_path)

    profile = profile_context(task.context_dir)

    assert profile["file_count"] == 4
    assert profile["files"]["csv/sales.csv"]["type"] == "csv"
    assert profile["files"]["csv/sales.csv"]["duckdb_table"] == "csv__sales"
    assert profile["files"]["json/users.json"]["type"] == "json"
    assert profile["files"]["db/demo.db"]["type"] == "sqlite"
    assert profile["files"]["knowledge.md"]["type"] == "doc"


def test_execute_data_sql_queries_csv_and_rejects_write_sql(tmp_path: Path) -> None:
    task = _make_task(tmp_path)

    result = execute_data_sql(
        task.context_dir,
        "SELECT region, SUM(amount) AS total FROM csv__sales GROUP BY region ORDER BY region",
    )

    assert result["columns"] == ["region", "total"]
    assert result["rows"] == [["East", 15], ["West", 20]]

    with pytest.raises(ValueError, match="Only SELECT/WITH"):
        execute_data_sql(task.context_dir, "DELETE FROM csv__sales")


def test_answer_tool_validates_shape(tmp_path: Path) -> None:
    task = _make_task(tmp_path)

    result = _answer(task, {"columns": ["answer"], "rows": [["42"]]})

    assert result.ok is True
    assert result.is_terminal is True
    assert result.answer is not None
    assert result.answer.columns == ["answer"]

    with pytest.raises(ValueError, match="match the number of columns"):
        _answer(task, {"columns": ["a", "b"], "rows": [["only-one-cell"]]})
