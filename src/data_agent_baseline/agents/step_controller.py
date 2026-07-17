from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from data_agent_baseline.agents.runtime import StepRecord
from data_agent_baseline.benchmark.schema import PublicTask


SCOUT_ACTIONS = {
    "list_context",
    "profile_context",
    "read_csv",
    "read_json",
    "read_doc",
    "search_doc",
}
COMPUTE_ACTIONS = {"execute_context_sql", "execute_data_sql", "execute_python"}
STRUCTURED_EVIDENCE_ACTIONS = COMPUTE_ACTIONS | {"extract_doc_records", "ground_thresholds"}
TABULAR_SUFFIXES = {".csv", ".json", ".sqlite", ".db", ".sqlite3", ".db3"}


NUMERIC_RE = re.compile(
    r"\b(count|how many|number of|sum|total|average|avg|mean|percent|percentage|ratio|"
    r"rank|top|bottom|median|variance|correlation|calculate|compute)\b",
    re.IGNORECASE,
)
RATIO_RE = re.compile(r"\b(percent|percentage|ratio|how many times|faster|slower)\b", re.IGNORECASE)
COUNT_RE = re.compile(r"\b(count|how many|number of)\b", re.IGNORECASE)
AVERAGE_MONTHLY_RE = re.compile(r"\b(avg|average|mean)\b.*\bmonthly\b|\bmonthly\b.*\b(avg|average|mean)\b", re.IGNORECASE)
RANK_RE = re.compile(r"\brank(?:ed)?\b", re.IGNORECASE)
THRESHOLD_RE = re.compile(
    r"\b(normal|abnormal|threshold|range|level|above|below|higher than|lower than|"
    r"at least|less than|greater than)\b",
    re.IGNORECASE,
)
PLACEHOLDER_ANSWER_VALUES = {
    "",
    "answer",
    "prediction",
    "prediction.csv",
    "result",
    "results",
    "output",
    "n/a",
    "na",
    "none",
    "unknown",
    "placeholder",
}


@dataclass(frozen=True, slots=True)
class StepControllerDecision:
    mode: str  # allow | override | block
    code: str
    message: str
    override_action: str | None = None
    override_action_input: dict[str, Any] | None = None
    required_next_action: str | None = None

    @property
    def should_allow(self) -> bool:
        return self.mode == "allow"

    @property
    def should_override(self) -> bool:
        return self.mode == "override"

    @property
    def should_block(self) -> bool:
        return self.mode == "block"


def allow() -> StepControllerDecision:
    return StepControllerDecision(
        mode="allow",
        code="allow",
        message="step allowed",
    )


def _action_key(action: str, action_input: dict[str, Any]) -> str:
    if action == "search_doc":
        return f"{action}:{action_input.get('path', '')}:{action_input.get('query', '')}"
    if action in {"read_csv", "read_json", "read_doc"}:
        return f"{action}:{action_input.get('path', '')}"
    if action == "execute_context_sql":
        return f"{action}:{action_input.get('path', '')}:{str(action_input.get('sql', ''))[:100]}"
    if action == "execute_data_sql":
        return f"{action}:{str(action_input.get('sql', ''))[:100]}"
    if action == "execute_python":
        return f"{action}:{str(action_input.get('code', ''))[:100]}"
    return f"{action}:{str(action_input)[:100]}"


def _recent_repeat_count(
    steps: list[StepRecord],
    action: str,
    action_input: dict[str, Any],
    *,
    window: int = 5,
) -> int:
    key = _action_key(action, action_input)
    return sum(
        1
        for step in steps[-window:]
        if _action_key(step.action, step.action_input) == key
    )


def _used_action(steps: list[StepRecord], action: str) -> bool:
    return any(step.action == action for step in steps)


def _used_any_action(steps: list[StepRecord], actions: set[str]) -> bool:
    return any(step.action in actions for step in steps)


def _context_has_tabular_data(context_dir: Path) -> bool:
    if not context_dir.exists():
        return False
    return any(path.is_file() and path.suffix.lower() in TABULAR_SUFFIXES for path in context_dir.rglob("*"))


def _context_has_real_document(context_dir: Path) -> bool:
    if not context_dir.exists():
        return False
    for path in context_dir.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in {".md", ".txt"}:
            continue
        if path.name.lower() == "knowledge.md":
            continue
        return True
    return False


def _is_light_table_task(task: PublicTask) -> bool:
    return (
        (task.difficulty or "").lower() in {"easy", "medium"}
        and _context_has_tabular_data(task.context_dir)
        and not _context_has_real_document(task.context_dir)
    )


def _trace_text(
    steps: list[StepRecord],
    *,
    current_thought: str = "",
    current_action_input: dict[str, Any] | None = None,
) -> str:
    parts: list[str] = [current_thought]
    if current_action_input:
        for value in current_action_input.values():
            if isinstance(value, str):
                parts.append(value)
            elif isinstance(value, list):
                parts.extend(str(item) for item in value[:20])
    for step in steps:
        parts.append(step.thought)
        if step.action_input:
            for value in step.action_input.values():
                if isinstance(value, str):
                    parts.append(value)
        content = step.observation.get("content") if isinstance(step.observation, dict) else None
        if isinstance(content, dict):
            for key in (
                "recommended_probe_summary",
                "summary",
                "query",
                "sql",
                "output",
                "preview",
            ):
                value = content.get(key)
                if value:
                    parts.append(str(value))
    return "\n".join(parts).lower()


def _answer_cells(action_input: dict[str, Any]) -> list[str]:
    values: list[str] = []
    columns = action_input.get("columns")
    if isinstance(columns, list):
        values.extend(str(column).strip() for column in columns)
    rows = action_input.get("rows")
    if isinstance(rows, list):
        for row in rows:
            if isinstance(row, list):
                values.extend(str(cell).strip() for cell in row)
            else:
                values.append(str(row).strip())
    return values


def _is_placeholder_answer(action_input: dict[str, Any]) -> bool:
    cells = [cell.lower() for cell in _answer_cells(action_input)]
    if not cells:
        return True
    non_empty = [cell for cell in cells if cell]
    if not non_empty:
        return True
    placeholder_count = sum(1 for cell in non_empty if cell in PLACEHOLDER_ANSWER_VALUES)
    return placeholder_count == len(non_empty)


def _controller_message(
    *,
    code: str,
    message: str,
    attempted_action: str,
    attempted_action_input: dict[str, Any],
    remaining_steps: int,
    required_next_action: str | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "controller": "step_controller",
        "code": code,
        "message": message,
        "attempted_action": attempted_action,
        "attempted_action_input": attempted_action_input,
        "remaining_steps_after_this": remaining_steps,
    }
    if required_next_action:
        payload["required_next_action"] = required_next_action
    return payload


def controller_observation(
    *,
    decision: StepControllerDecision,
    attempted_action: str,
    attempted_action_input: dict[str, Any],
    remaining_steps: int,
) -> dict[str, Any]:
    return {
        "ok": False,
        "tool": "__step_controller__",
        "content": _controller_message(
            code=decision.code,
            message=decision.message,
            attempted_action=attempted_action,
            attempted_action_input=attempted_action_input,
            remaining_steps=remaining_steps,
            required_next_action=decision.required_next_action,
        ),
    }


def override_metadata(
    *,
    decision: StepControllerDecision,
    attempted_action: str,
    attempted_action_input: dict[str, Any],
) -> dict[str, Any]:
    return {
        "_controller_override": {
            "code": decision.code,
            "message": decision.message,
            "attempted_action": attempted_action,
            "attempted_action_input": attempted_action_input,
        }
    }


def review_next_step(
    *,
    task: PublicTask,
    steps: list[StepRecord],
    step_index: int,
    max_steps: int,
    thought: str,
    action: str,
    action_input: dict[str, Any],
) -> StepControllerDecision:
    """Review a proposed ReAct action before executing the tool.

    The controller intentionally uses lightweight deterministic checks. It is a
    guardrail, not a second solver: when the next action is clearly unsafe, it
    either overrides to a safe scouting tool or blocks the action and asks the
    model to take a more constrained next step.
    """

    remaining_steps = max(max_steps - step_index, 0)
    question = task.question
    evidence_text = _trace_text(steps, current_thought=thought, current_action_input=action_input)
    light_table_task = _is_light_table_task(task)

    if step_index <= 2 and not _used_action(steps, "profile_context") and action != "profile_context":
        return StepControllerDecision(
            mode="override",
            code="profile_context_required",
            message=(
                "The first phase must profile files, schemas, and DuckDB table names before "
                "specialized reads or computation."
            ),
            override_action="profile_context",
            override_action_input={},
            required_next_action="profile_context",
        )

    repeat_count = _recent_repeat_count(steps, action, action_input)
    if action != "answer" and repeat_count >= 2 and remaining_steps > 0:
        return StepControllerDecision(
            mode="block",
            code="repeated_tool_call_blocked",
            message=(
                f"The same `{action}` call has already appeared {repeat_count} time(s) "
                "recently. Change tool, query, path, or submit the best grounded answer."
            ),
            required_next_action="different_tool_or_answer",
        )

    if (
        not light_table_task
        and
        remaining_steps <= 3
        and remaining_steps > 0
        and action in SCOUT_ACTIONS
        and (action != "profile_context" or _used_action(steps, "profile_context"))
    ):
        return StepControllerDecision(
            mode="block",
            code="low_budget_exploration_blocked",
            message=(
                "Step budget is almost exhausted. Stop broad scouting/searching; use existing "
                "evidence, run one compact computation if needed, or call answer."
            ),
            required_next_action="execute_data_sql_or_execute_python_or_answer",
        )

    # 4a. Excessive doc search without compute
    if action in ("search_doc", "read_doc"):
        recent_actions = [step.action for step in steps[-6:]]
        scout_count = sum(1 for a in recent_actions if a in SCOUT_ACTIONS)
        compute_count = sum(1 for a in recent_actions if a in COMPUTE_ACTIONS)
        if scout_count >= 4 and compute_count == 0:
            return StepControllerDecision(
                mode="block",
                code="excessive_scout_without_compute",
                message=(
                    "The last 6 steps contain multiple document/scout actions with zero "
                    "compute steps. Switch to profile_context to re-assess, or run a "
                    "compact compute action (SQL/Python) to test your current hypothesis."
                ),
                required_next_action="profile_context_or_execute",
            )

    # 4b. SQL returning very few rows repeatedly
    if action in ("execute_data_sql", "execute_context_sql"):
        sql_empty_count = 0
        for step in reversed(steps):
            if step.action in ("execute_data_sql", "execute_context_sql"):
                content = step.observation.get("content") if isinstance(step.observation, dict) else {}
                rows = content.get("rows") if isinstance(content, dict) else None
                if isinstance(rows, list) and len(rows) <= 2:
                    sql_empty_count += 1
                else:
                    break
            else:
                break
        if sql_empty_count >= 3:
            return StepControllerDecision(
                mode="block",
                code="repeated_empty_sql_results",
                message=(
                    f"The last {sql_empty_count} SQL calls each returned <= 2 rows or were empty. "
                    "Check your filters, broaden the query, or inspect schema columns before "
                    "running another SQL query."
                ),
                required_next_action="profile_context_or_different_query",
            )

    if action != "answer":
        return allow()

    numeric_question = bool(NUMERIC_RE.search(question))
    has_tabular_context = _context_has_tabular_data(task.context_dir)
    used_compute = _used_any_action(steps, COMPUTE_ACTIONS)
    used_structured_evidence = _used_any_action(steps, STRUCTURED_EVIDENCE_ACTIONS)
    if _is_placeholder_answer(action_input):
        return StepControllerDecision(
            mode="block",
            code="placeholder_answer_blocked",
            message=(
                "The proposed answer appears to be a placeholder, file name, or empty fallback. "
                "Use context evidence and submit concrete values from the data."
            ),
            required_next_action="execute_data_sql_or_execute_python",
        )

    if remaining_steps == 0:
        return allow()

    if has_tabular_context and not used_structured_evidence:
        return StepControllerDecision(
            mode="block",
            code="tabular_answer_without_evidence",
            message=(
                "The context contains tabular data, but no SQL/Python/document extraction result "
                "has been observed. Query or compute the requested rows before answering."
            ),
            required_next_action="execute_data_sql_or_execute_python",
        )

    if numeric_question and has_tabular_context and not used_compute:
        return StepControllerDecision(
            mode="block",
            code="numeric_answer_without_computation",
            message=(
                "The question has numeric/aggregation intent over tabular data, but no SQL/Python "
                "computation has been executed. Compute a compact aggregate before answering."
            ),
            required_next_action="execute_data_sql_or_execute_python",
        )

    if not light_table_task and RATIO_RE.search(question):
        has_denominator_commitment = "numerator" in evidence_text and "denominator" in evidence_text
        if not has_denominator_commitment:
            return StepControllerDecision(
                mode="block",
                code="denominator_evidence_missing",
                message=(
                    "Percentage/ratio answers must state or compute numerator and denominator "
                    "before the final answer."
                ),
                required_next_action="execute_data_sql_or_execute_python",
            )

    if not light_table_task and AVERAGE_MONTHLY_RE.search(question):
        compared_candidates = (
            "avg(" in evidence_text
            and "sum(" in evidence_text
            and ("count(distinct" in evidence_text or "candidate" in evidence_text)
        )
        if not compared_candidates:
            return StepControllerDecision(
                mode="block",
                code="average_monthly_candidates_missing",
                message=(
                    "Average-monthly wording is ambiguous. Before answering, compute and compare "
                    "candidate formulas such as AVG(raw metric)/12, SUM(metric)/12, "
                    "SUM(metric)/COUNT(DISTINCT month), and entity-level AVG annual/12."
                ),
                required_next_action="execute_data_sql_or_execute_python",
            )

    if RANK_RE.search(question):
        used_position_as_rank = any(token in evidence_text for token in ("positionorder = 2", "position = 2"))
        used_literal_rank = any(token in evidence_text for token in ("rank = 2", "rank=2", " rank "))
        if used_position_as_rank and not used_literal_rank:
            return StepControllerDecision(
                mode="block",
                code="rank_field_evidence_missing",
                message=(
                    "The question uses rank/ranked wording. If a literal rank column exists, "
                    "filter/order by rank instead of position or positionOrder."
                ),
                required_next_action="execute_data_sql_or_execute_python",
            )

    if not light_table_task and THRESHOLD_RE.search(question):
        has_threshold_tool = _used_action(steps, "ground_thresholds")
        has_threshold_evidence = any(
            token in evidence_text
            for token in ("threshold", "range", " between ", ">=", "<=", "recommended_probe_summary")
        )
        if not has_threshold_tool and not has_threshold_evidence:
            return StepControllerDecision(
                mode="block",
                code="threshold_evidence_missing",
                message=(
                    "Threshold/range/normal-abnormal wording needs context-grounded threshold "
                    "evidence before answering."
                ),
                required_next_action="ground_thresholds",
            )

    multiplicity_hint = any(
        token in evidence_text
        for token in (" join ", " group by ", "measurement", "measurements", "relationship", "relationships")
    )
    if (
        not light_table_task
        and
        COUNT_RE.search(question)
        and "count(*)" in evidence_text
        and "distinct" not in evidence_text
        and multiplicity_hint
    ):
        return StepControllerDecision(
            mode="block",
            code="count_target_evidence_missing",
            message=(
                "The trace uses COUNT(*) for a count question without distinct/entity-count "
                "evidence. Compare row count with COUNT(DISTINCT target key) before answering."
            ),
            required_next_action="execute_data_sql_or_execute_python",
        )

    return allow()
