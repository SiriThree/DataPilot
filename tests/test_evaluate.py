from __future__ import annotations

import json
from pathlib import Path

from evaluate import evaluate_all


def _write_task(root: Path, task_id: str, difficulty: str = "easy") -> None:
    task_dir = root / "input" / task_id
    task_dir.mkdir(parents=True)
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
    (task_dir / "context").mkdir()
    gold_dir = root / "output" / task_id
    gold_dir.mkdir(parents=True)
    (gold_dir / "gold.csv").write_text("answer\n42\n", encoding="utf-8")


def test_evaluate_all_only_existing_skips_missing_predictions(tmp_path: Path) -> None:
    _write_task(tmp_path, "task_1")
    _write_task(tmp_path, "task_2")

    predictions_dir = tmp_path / "predictions"
    pred_task_dir = predictions_dir / "task_1"
    pred_task_dir.mkdir(parents=True)
    (pred_task_dir / "prediction.csv").write_text("answer\n42\n", encoding="utf-8")

    result = evaluate_all(
        tmp_path / "input",
        tmp_path / "output",
        predictions_dir,
        only_existing=True,
    )

    assert result["summary"]["total_tasks"] == 1
    assert result["summary"]["skipped_missing_predictions"] == 1
    assert result["summary"]["overall_score"] == 1.0
    assert result["skipped_missing_prediction_task_ids"] == ["task_2"]


def test_evaluate_all_counts_missing_predictions_without_only_existing(tmp_path: Path) -> None:
    _write_task(tmp_path, "task_1")
    _write_task(tmp_path, "task_2")

    result = evaluate_all(
        tmp_path / "input",
        tmp_path / "output",
        tmp_path / "predictions",
    )

    assert result["summary"]["total_tasks"] == 2
    assert result["summary"]["skipped_missing_predictions"] == 0
    assert result["summary"]["overall_score"] == 0.0
    assert len(result["summary"]["failures"]) == 2
