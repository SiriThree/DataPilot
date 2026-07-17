from __future__ import annotations

from pathlib import Path

from data_agent_baseline.run.difficulty_policy import (
    build_strategy_policy,
    build_strategy_prompt_hint,
    strategy_policy_to_dict,
)
from data_agent_baseline.run.route_decision import decide_route


def test_easy_sql_task_uses_fast_react_policy(tmp_path: Path) -> None:
    context_dir = tmp_path / "context"
    context_dir.mkdir()
    (context_dir / "sales.csv").write_text("id,amount\n1,10\n2,20\n", encoding="utf-8")
    route = decide_route(
        question="What is the total amount?",
        context_dir=context_dir,
        difficulty="easy",
    )

    policy = build_strategy_policy(
        difficulty="easy",
        route_decision=route,
        configured_max_steps=36,
        configured_use_multi_agent=True,
        configured_enable_guided_retry=True,
    )

    assert policy.mode == "easy_fast_react"
    assert policy.max_steps == 12
    assert policy.use_multi_agent is False
    assert policy.enable_guided_retry is False
    assert policy.use_decomposer is False


def test_medium_policy_keeps_planner_without_decomposer(tmp_path: Path) -> None:
    context_dir = tmp_path / "context"
    context_dir.mkdir()
    (context_dir / "a.csv").write_text("id,value\n1,10\n", encoding="utf-8")
    route = decide_route(
        question="Calculate the average value by id.",
        context_dir=context_dir,
        difficulty="medium",
    )

    policy = build_strategy_policy(
        difficulty="medium",
        route_decision=route,
        configured_max_steps=36,
        configured_use_multi_agent=True,
        configured_enable_guided_retry=True,
    )

    assert policy.mode == "medium_planner_executor"
    assert policy.max_steps == 24
    assert policy.use_multi_agent is True
    assert policy.use_decomposer is False
    assert policy.enable_guided_retry is True


def test_hard_hybrid_policy_enables_decomposer(tmp_path: Path) -> None:
    context_dir = tmp_path / "context"
    context_dir.mkdir()
    (context_dir / "rules.md").write_text("Use threshold 10 for valid records.", encoding="utf-8")
    (context_dir / "records.csv").write_text("id,value\n1,12\n2,8\n", encoding="utf-8")
    route = decide_route(
        question="According to the document rule, how many records are above the threshold?",
        context_dir=context_dir,
        difficulty="hard",
    )

    policy = build_strategy_policy(
        difficulty="hard",
        route_decision=route,
        configured_max_steps=36,
        configured_use_multi_agent=True,
        configured_enable_guided_retry=True,
    )

    assert policy.mode == "hard_multi_agent"
    assert policy.max_steps >= 36
    assert policy.use_decomposer is True
    assert policy.guided_retry_mode == "aggressive"
    assert "Strategy policy:" in build_strategy_prompt_hint(policy)
    assert strategy_policy_to_dict(policy)["mode"] == "hard_multi_agent"


def test_extreme_policy_reserves_larger_budget(tmp_path: Path) -> None:
    context_dir = tmp_path / "context"
    context_dir.mkdir()
    (context_dir / "records.csv").write_text("id,value\n1,12\n", encoding="utf-8")
    route = decide_route(
        question="Compare all ratios and explain the final answer.",
        context_dir=context_dir,
        difficulty="extreme",
    )

    policy = build_strategy_policy(
        difficulty="extreme",
        route_decision=route,
        configured_max_steps=36,
        configured_use_multi_agent=True,
        configured_enable_guided_retry=True,
    )

    assert policy.mode == "extreme_task_graph_ready"
    assert policy.max_steps == 48
    assert policy.use_decomposer is True
