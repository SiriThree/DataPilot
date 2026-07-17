from __future__ import annotations

from data_agent_baseline.agents.multi_agent import _ensure_current_plan_step
from data_agent_baseline.agents.planner import PlanStep


def test_ensure_current_plan_step_appends_answer_when_index_runs_past_plan() -> None:
    plan_steps = [
        PlanStep(1, "execute_data_sql", "Compute answer", "rows"),
        PlanStep(2, "answer", "Submit answer", "prediction.csv"),
    ]

    current_idx = _ensure_current_plan_step(plan_steps, 5)

    assert current_idx == 2
    assert len(plan_steps) == 3
    assert plan_steps[current_idx].tool == "answer"


def test_ensure_current_plan_step_handles_empty_plan() -> None:
    plan_steps: list[PlanStep] = []

    current_idx = _ensure_current_plan_step(plan_steps, 0)

    assert current_idx == 0
    assert len(plan_steps) == 1
    assert plan_steps[0].tool == "answer"
