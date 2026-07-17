from __future__ import annotations

import json
from pathlib import Path

import pytest

from data_agent_baseline.config import AgentConfig, AppConfig, DatasetConfig, RunConfig
from data_agent_baseline.run import runner
from data_agent_baseline.run.runner import TaskRunArtifacts, normalize_difficulty_filter, run_benchmark


def _write_task(root: Path, task_id: str, difficulty: str) -> None:
    task_dir = root / task_id
    task_dir.mkdir(parents=True)
    (task_dir / "context").mkdir()
    (task_dir / "task.json").write_text(
        json.dumps(
            {
                "task_id": task_id,
                "difficulty": difficulty,
                "question": "Return the answer.",
            }
        ),
        encoding="utf-8",
    )


def test_run_benchmark_filters_by_difficulty(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_dir = tmp_path / "input"
    _write_task(input_dir, "task_1", "easy")
    _write_task(input_dir, "task_2", "medium")
    _write_task(input_dir, "task_3", "easy")

    seen: list[str] = []

    def fake_run_single_task(*, task_id, config, run_output_dir, model=None, tools=None):
        seen.append(task_id)
        task_dir = run_output_dir / task_id
        task_dir.mkdir(parents=True)
        trace_path = task_dir / "trace.json"
        trace_path.write_text("{}", encoding="utf-8")
        prediction_path = task_dir / "prediction.csv"
        prediction_path.write_text("answer\n42\n", encoding="utf-8")
        return TaskRunArtifacts(
            task_id=task_id,
            task_output_dir=task_dir,
            prediction_csv_path=prediction_path,
            trace_path=trace_path,
            succeeded=True,
            failure_reason=None,
        )

    monkeypatch.setattr(runner, "run_single_task", fake_run_single_task)
    config = AppConfig(
        dataset=DatasetConfig(root_path=input_dir),
        agent=AgentConfig(api_key="test-key"),
        run=RunConfig(output_dir=tmp_path / "runs", max_workers=4),
    )

    run_dir, artifacts = run_benchmark(
        config=config,
        difficulty="Easy",
        model=object(),
        tools=object(),
    )

    assert seen == ["task_1", "task_3"]
    assert [artifact.task_id for artifact in artifacts] == ["task_1", "task_3"]
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["difficulty_filter"] == "easy"
    assert summary["task_count"] == 2


def test_normalize_difficulty_filter_rejects_unknown() -> None:
    assert normalize_difficulty_filter(None) is None
    assert normalize_difficulty_filter(" HARD ") == "hard"
    with pytest.raises(ValueError, match="difficulty must be one of"):
        normalize_difficulty_filter("advanced")
