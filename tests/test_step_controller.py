from __future__ import annotations

from pathlib import Path

from data_agent_baseline.agents.runtime import StepRecord
from data_agent_baseline.agents.step_controller import review_next_step
from data_agent_baseline.benchmark.schema import PublicTask, TaskAssets, TaskRecord


def _task(tmp_path: Path, *, question: str = "List their ID, sex, and diagnosis.") -> PublicTask:
    task_dir = tmp_path / "task_1"
    context_dir = task_dir / "context"
    context_dir.mkdir(parents=True)
    (context_dir / "patients.csv").write_text("ID,sex,Diagnosis\n1,F,SLE\n", encoding="utf-8")
    return PublicTask(
        record=TaskRecord(task_id="task_1", difficulty="easy", question=question),
        assets=TaskAssets(task_dir=task_dir, context_dir=context_dir),
    )


def test_step_controller_blocks_placeholder_answer(tmp_path: Path) -> None:
    decision = review_next_step(
        task=_task(tmp_path),
        steps=[],
        step_index=3,
        max_steps=36,
        thought="Formatter: final answer.",
        action="answer",
        action_input={"columns": ["prediction"], "rows": [["prediction.csv"]]},
    )

    assert decision.should_block
    assert decision.code == "placeholder_answer_blocked"


def test_step_controller_blocks_tabular_answer_without_evidence(tmp_path: Path) -> None:
    decision = review_next_step(
        task=_task(tmp_path),
        steps=[],
        step_index=3,
        max_steps=36,
        thought="Formatter: final answer.",
        action="answer",
        action_input={"columns": ["ID", "sex", "Diagnosis"], "rows": [["1", "F", "SLE"]]},
    )

    assert decision.should_block
    assert decision.code == "tabular_answer_without_evidence"


def test_step_controller_blocks_position_as_rank(tmp_path: Path) -> None:
    task = _task(tmp_path, question="What's the finish time for the driver who ranked second?")
    decision = review_next_step(
        task=task,
        steps=[
            StepRecord(
                step_index=3,
                thought="Data Engineer: queried result where positionOrder = 2.",
                action="execute_python",
                action_input={"code": "second = results[results['positionOrder'] == 2]"},
                raw_response="{}",
                observation={"ok": True, "content": {"output": "+14.925"}},
                ok=True,
            )
        ],
        step_index=4,
        max_steps=36,
        thought="Verifier: used positionOrder = 2 to answer ranked second.",
        action="answer",
        action_input={"columns": ["time"], "rows": [["+14.925"]]},
    )

    assert decision.should_block
    assert decision.code == "rank_field_evidence_missing"
