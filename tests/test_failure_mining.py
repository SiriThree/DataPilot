from __future__ import annotations

import json
from pathlib import Path

from data_agent_baseline.run.failure_mining import (
    analyze_run_failures,
    render_failure_mining_markdown,
    write_failure_mining_outputs,
)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def test_failure_mining_groups_low_score_tasks(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _write_json(
        run_dir / "evaluation.json",
        {
            "total_tasks": 2,
            "overall_score": 0.5,
            "failures": [
                {
                    "task_id": "task_10",
                    "score": 0.0,
                    "difficulty": "hard",
                    "gold_rows": 1,
                    "pred_rows": 1,
                }
            ],
        },
    )
    _write_json(
        run_dir / "task_10" / "trace.json",
        {
            "task_id": "task_10",
            "_route_decision": {
                "route": "hybrid_doc_table",
                "task_profile": {
                    "task_type": "threshold_count",
                    "operation": "threshold_count",
                    "output_shape": "scalar",
                    "domains": ["medical_patient"],
                    "flags": ["doc_table_grounding"],
                },
            },
            "_selected_attempt": "retry",
            "_post_process": {
                "repair_execution": {
                    "actions": [{"action_type": "fix_patient_threshold_count"}]
                }
            },
        },
    )
    _write_json(
        run_dir / "task_10" / "failure_analysis.json",
        {
            "task_id": "task_10",
            "signals": [
                {
                    "code": "threshold_grounding_risk",
                    "severity": "warning",
                    "detail": "threshold detected",
                }
            ],
            "commitments": {
                "tool_counts": {"extract_doc_records": 2, "execute_python": 1},
            },
        },
    )
    _write_json(
        run_dir / "task_11" / "trace.json",
        {
            "task_id": "task_11",
            "_route_decision": {
                "route": "python_first",
                "task_profile": {"task_type": "aggregation", "operation": "aggregate_average"},
            },
        },
    )

    result = analyze_run_failures(run_dir=run_dir)

    assert result.summary["low_score_task_count"] == 1
    assert result.tasks[0]["task_id"] == "task_10"
    assert result.tasks[0]["selected_attempt"] == "retry"
    assert result.groups["by_task_type"] == {"threshold_count": 1}
    assert result.groups["by_signal"] == {"threshold_grounding_risk": 1}
    assert result.groups["by_domain"] == {"medical_patient": 1}
    assert result.recommendations
    assert result.recommendations[0]["task_ids"] == ["task_10"]
    assert any(item["key"] == "threshold_count" for item in result.recommendations)
    assert any(item["key"] == "threshold_grounding_risk" for item in result.recommendations)

    markdown = render_failure_mining_markdown(result)
    assert "task_10" in markdown
    assert "threshold_count" in markdown
    assert "fix_patient_threshold_count" in markdown
    assert "Development Recommendations" in markdown
    assert "threshold-count solver" in markdown

    json_path, md_path = write_failure_mining_outputs(result, run_dir)
    assert json_path.exists()
    assert md_path.exists()
