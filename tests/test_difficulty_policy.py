from __future__ import annotations

import sqlite3
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


def test_medium_sql_aggregation_uses_lightweight_react(tmp_path: Path) -> None:
    context_dir = tmp_path / "context"
    context_dir.mkdir()
    db_path = context_dir / "sales.db"
    with sqlite3.connect(db_path) as conn:
        conn.execute("create table sales (id integer, amount integer)")
        conn.executemany("insert into sales values (?, ?)", [(1, 10), (2, 20)])
    route = decide_route(
        question="What is the total amount?",
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

    assert route.route == "sql_first"
    assert route.task_profile.task_type == "aggregation"
    assert policy.mode == "medium_sql_react"
    assert policy.max_steps == 20
    assert policy.use_multi_agent is False
    assert policy.enable_guided_retry is True
    assert policy.required_first_tools == ["profile_context", "execute_data_sql"]


def test_easy_large_csv_uses_lightweight_sql_react(tmp_path: Path) -> None:
    context_dir = tmp_path / "context"
    context_dir.mkdir()
    large_csv = context_dir / "transactions.csv"
    with large_csv.open("w", encoding="utf-8", newline="") as handle:
        handle.write("id,amount\n")
        row = "1,10\n"
        for _ in range((5 * 1024 * 1024 // len(row)) + 1):
            handle.write(row)
    route = decide_route(
        question="List all transaction ids.",
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

    assert "large_csv_sql_first" in route.risk_flags
    assert policy.mode == "easy_large_sql_react"
    assert policy.max_steps == 16
    assert policy.use_multi_agent is False
    assert policy.enable_guided_retry is False


def test_hard_threshold_policy_uses_solver_ready_mode(tmp_path: Path) -> None:
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

    assert policy.mode == "hard_threshold_solver_ready"
    assert policy.max_steps == 40
    assert policy.use_decomposer is True
    assert policy.verifier_frequency == "every_step"
    assert policy.guided_retry_mode == "aggressive"
    assert "Strategy policy:" in build_strategy_prompt_hint(policy)
    assert "ground_thresholds" in policy.required_first_tools
    assert strategy_policy_to_dict(policy)["mode"] == "hard_threshold_solver_ready"


def test_hard_sql_policy_skips_decomposer_but_keeps_multi_agent(tmp_path: Path) -> None:
    context_dir = tmp_path / "context"
    context_dir.mkdir()
    db_path = context_dir / "sales.db"
    with sqlite3.connect(db_path) as conn:
        conn.execute("create table sales (id integer, amount integer)")
        conn.executemany("insert into sales values (?, ?)", [(1, 10), (2, 20)])
    route = decide_route(
        question="What is the average amount?",
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

    assert route.route == "sql_first"
    assert policy.mode == "hard_sql_controlled"
    assert policy.max_steps == 32
    assert policy.use_multi_agent is True
    assert policy.use_decomposer is False
    assert policy.use_verifier is True
    assert policy.required_first_tools == ["profile_context", "execute_data_sql"]


def test_hard_doc_table_policy_uses_decomposition(tmp_path: Path) -> None:
    context_dir = tmp_path / "context"
    context_dir.mkdir()
    (context_dir / "policy.md").write_text("Only records with approved status are valid.", encoding="utf-8")
    (context_dir / "records.csv").write_text("id,status,value\n1,approved,12\n2,draft,8\n", encoding="utf-8")
    route = decide_route(
        question="According to the policy document, list the approved record ids.",
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

    assert route.route in {"document_first", "hybrid_doc_table"}
    assert policy.mode == "hard_doc_table_decompose"
    assert policy.max_steps == 40
    assert policy.use_decomposer is True
    assert "extract_doc_records" in policy.required_first_tools


def test_hard_ratio_policy_avoids_decomposer_overhead(tmp_path: Path) -> None:
    context_dir = tmp_path / "context"
    context_dir.mkdir()
    with sqlite3.connect(context_dir / "results.db") as conn:
        conn.execute("create table results (driver_id integer, rank integer, seconds real)")
        conn.executemany("insert into results values (?, ?, ?)", [(1, 1, 100.0), (2, 20, 120.0)])
    (context_dir / "race.md").write_text("Australian Grand Prix race metadata.", encoding="utf-8")
    route = decide_route(
        question="How much faster in percentage is the champion than the driver who finished last?",
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

    assert route.task_profile.task_type == "ratio_or_percentage"
    assert policy.mode == "hard_ratio_crosscheck"
    assert policy.max_steps == 32
    assert policy.use_decomposer is False
    assert policy.use_verifier is True
    assert policy.required_first_tools == ["profile_context", "execute_data_sql"]


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


def test_extreme_pure_doc_threshold_uses_structured_react(tmp_path: Path) -> None:
    context_dir = tmp_path / "context"
    context_dir.mkdir()
    doc_dir = context_dir / "doc"
    doc_dir.mkdir()
    (doc_dir / "Laboratory.md").write_text(
        "Patient 1001 has creatinine level 2.1 and is marked abnormal.\n"
        "Patient 1002 has creatinine level 0.8 and is marked normal.\n",
        encoding="utf-8",
    )
    (doc_dir / "Patient.md").write_text(
        "Patient 1001 birth year 1960.\nPatient 1002 birth year 1990.\n",
        encoding="utf-8",
    )
    route = decide_route(
        question="Among the patients whose creatinine level is abnormal, how many of them aren't 70 yet?",
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

    assert "pure_doc_no_structured_source" in route.risk_flags
    assert route.task_profile.task_type == "threshold_count"
    assert policy.mode == "extreme_pure_doc_threshold_react"
    assert policy.max_steps == 32
    assert policy.use_multi_agent is False
    assert policy.use_decomposer is False
    assert policy.required_first_tools == [
        "profile_context",
        "extract_doc_records",
        "ground_thresholds",
        "execute_python",
    ]
