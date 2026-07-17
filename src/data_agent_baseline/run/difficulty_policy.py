"""Difficulty-aware execution policy for DataAgent runs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from data_agent_baseline.run.route_decision import RouteDecision


SIMPLE_TASK_TYPES = {"aggregation", "count", "lookup", "rank_lookup"}
SIMPLE_ROUTES = {"sql_first", "python_first", "hybrid_sql_python"}


@dataclass(frozen=True, slots=True)
class StrategyPolicy:
    difficulty: str
    mode: str
    max_steps: int
    use_multi_agent: bool
    use_verifier: bool
    use_router: bool
    use_debugger: bool
    use_decomposer: bool
    verifier_frequency: str
    guided_retry_mode: str
    enable_guided_retry: bool
    repair_scope: str
    required_first_tools: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def _norm_difficulty(difficulty: str | None) -> str:
    value = (difficulty or "medium").strip().lower()
    return value if value in {"easy", "medium", "hard", "extreme"} else "medium"


def _bounded_steps(configured: int, *, lower: int, upper: int | None = None) -> int:
    configured = max(int(configured or 1), 1)
    value = max(configured, lower)
    if upper is not None:
        value = min(value, upper)
    return value


def _is_easy_fast_path(route_decision: RouteDecision) -> bool:
    profile = route_decision.task_profile
    if route_decision.route not in SIMPLE_ROUTES:
        return False
    if profile.task_type not in SIMPLE_TASK_TYPES:
        return False
    if "doc_table_grounding" in profile.flags:
        return False
    if "large_csv_sql_first" in route_decision.risk_flags:
        return False
    return True


def build_strategy_policy(
    *,
    difficulty: str | None,
    route_decision: RouteDecision,
    configured_max_steps: int,
    configured_use_multi_agent: bool,
    configured_enable_guided_retry: bool,
) -> StrategyPolicy:
    """Map difficulty + route + task profile to an execution strategy."""
    normalized = _norm_difficulty(difficulty)
    profile = route_decision.task_profile
    notes: list[str] = [
        f"route={route_decision.route}",
        f"task_type={profile.task_type}",
    ]

    if normalized == "easy":
        fast_path = _is_easy_fast_path(route_decision)
        return StrategyPolicy(
            difficulty=normalized,
            mode="easy_fast_react" if fast_path else "easy_guarded",
            max_steps=12 if fast_path else 16,
            use_multi_agent=False if fast_path else configured_use_multi_agent,
            use_verifier=not fast_path,
            use_router=not fast_path,
            use_debugger=not fast_path,
            use_decomposer=False,
            verifier_frequency="near_end",
            guided_retry_mode="off" if fast_path else "conservative",
            enable_guided_retry=False if fast_path else configured_enable_guided_retry,
            repair_scope="format_and_safe_generic",
            required_first_tools=["profile_context"],
            notes=notes + (["easy fast path: avoid planner/router/decomposer"] if fast_path else []),
        )

    if normalized == "medium":
        max_steps = _bounded_steps(configured_max_steps, lower=20, upper=24)
        if route_decision.route == "hybrid_doc_table" or profile.task_type == "threshold_count":
            max_steps = max(max_steps, 24)
        return StrategyPolicy(
            difficulty=normalized,
            mode="medium_planner_executor",
            max_steps=max_steps,
            use_multi_agent=configured_use_multi_agent,
            use_verifier=True,
            use_router=True,
            use_debugger=True,
            use_decomposer=False,
            verifier_frequency="every_compute",
            guided_retry_mode="normal",
            enable_guided_retry=configured_enable_guided_retry,
            repair_scope="generic_pattern",
            required_first_tools=["profile_context"],
            notes=notes + ["medium: plan and verify, no recursive decomposition"],
        )

    if normalized == "hard":
        max_steps = _bounded_steps(configured_max_steps, lower=32, upper=40)
        if route_decision.route == "hybrid_doc_table" or profile.task_type in {"threshold_count", "ratio_or_percentage"}:
            max_steps = max(max_steps, 36)
        return StrategyPolicy(
            difficulty=normalized,
            mode="hard_multi_agent",
            max_steps=max_steps,
            use_multi_agent=configured_use_multi_agent,
            use_verifier=True,
            use_router=True,
            use_debugger=True,
            use_decomposer=True,
            verifier_frequency="every_compute",
            guided_retry_mode="aggressive",
            enable_guided_retry=configured_enable_guided_retry,
            repair_scope="generic_pattern_plus_guarded_solver",
            required_first_tools=["profile_context"],
            notes=notes + ["hard: enable decomposition metadata and stronger verification"],
        )

    max_steps = _bounded_steps(configured_max_steps, lower=48, upper=60)
    return StrategyPolicy(
        difficulty=normalized,
        mode="extreme_task_graph_ready",
        max_steps=max_steps,
        use_multi_agent=configured_use_multi_agent,
        use_verifier=True,
        use_router=True,
        use_debugger=True,
        use_decomposer=True,
        verifier_frequency="every_compute",
        guided_retry_mode="aggressive",
        enable_guided_retry=configured_enable_guided_retry,
        repair_scope="generic_pattern_plus_guarded_solver",
        required_first_tools=["profile_context"],
        notes=notes + ["extreme: reserve budget for future DAG/subtask execution"],
    )


def strategy_policy_to_dict(policy: StrategyPolicy) -> dict[str, Any]:
    return {
        "difficulty": policy.difficulty,
        "mode": policy.mode,
        "max_steps": policy.max_steps,
        "use_multi_agent": policy.use_multi_agent,
        "use_verifier": policy.use_verifier,
        "use_router": policy.use_router,
        "use_debugger": policy.use_debugger,
        "use_decomposer": policy.use_decomposer,
        "verifier_frequency": policy.verifier_frequency,
        "guided_retry_mode": policy.guided_retry_mode,
        "enable_guided_retry": policy.enable_guided_retry,
        "repair_scope": policy.repair_scope,
        "required_first_tools": policy.required_first_tools,
        "notes": policy.notes,
    }


def build_strategy_prompt_hint(policy: StrategyPolicy) -> str:
    lines = [
        "Strategy policy:",
        f"- Mode: {policy.mode}",
        f"- Difficulty: {policy.difficulty}",
        f"- Step budget: {policy.max_steps}",
        f"- Guided retry mode: {policy.guided_retry_mode}",
        f"- Repair scope: {policy.repair_scope}",
    ]
    if policy.required_first_tools:
        lines.append(f"- Required first tools: {', '.join(policy.required_first_tools)}")
    if policy.notes:
        lines.append(f"- Notes: {'; '.join(policy.notes[:3])}")
    return "\n".join(lines)
