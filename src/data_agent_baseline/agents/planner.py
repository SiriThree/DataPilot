"""Explicit pre-execution planner: generates structured plan BEFORE tool execution.

Replaces the current "think then act" pattern with "plan first, then execute each step."
The planner produces a numbered list of concrete steps, each with:
- step number
- description of what to do
- expected tool to use
- expected output

This plan guides the execution loop and serves as input to the verifier and router.

For complex Hard tasks, the planner can recursively decompose into sub-tasks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

PLANNER_SYSTEM_PROMPT = """You are a data analysis planner. Your job is to create a concrete, executable step-by-step plan to answer a data question.

Before planning, you receive:
- The question to answer
- A summary of available data (from profile_context)
- A route hint suggesting the best approach (SQL-first, Python-first, document-first, or hybrid)

Create a plan with 3-8 concrete steps. Each step should:
1. Say EXACTLY which tool to use
2. Say what data/columns/tables to target
3. Say what the expected output should be

Planning rules:
- Step 1 is ALWAYS "profile_context" to understand the data landscape
- Middle steps do the actual computation (SQL queries, Python scripts, document extraction)
- Include at least one verification step before the final answer
- For document tasks: plan doc extraction BEFORE computation
- For SQL tasks: plan schema inspection before complex queries
- For hybrid tasks: plan data integration after individual source analysis
- Final step is ALWAYS "answer"

Output format: a JSON array of step objects, each with "step", "tool", "description", and "expected_output" keys."""

PLANNER_USER_TEMPLATE = """Question: {question}

Available data summary:
{data_summary}

Route hint: {route_hint}
Task difficulty: {difficulty}

Create a step-by-step plan to answer this question. Output as JSON array:

```json
[
  {{"step": 1, "tool": "profile_context", "description": "...", "expected_output": "..."}},
  ...
]
```"""


@dataclass(slots=True)
class PlanStep:
    step: int
    tool: str
    description: str
    expected_output: str = ""


@dataclass(slots=True)
class Plan:
    steps: list[PlanStep] = field(default_factory=list)
    raw_response: str = ""

    @property
    def step_count(self) -> int:
        return len(self.steps)

    def format_for_verifier(self) -> str:
        """Format the plan for the verifier prompt."""
        lines = []
        for s in self.steps:
            lines.append(
                f"Step {s.step} [{s.tool}]: {s.description}"
                f"{' → ' + s.expected_output if s.expected_output else ''}"
            )
        return "\n".join(lines)

    def format_for_router(self) -> str:
        """Format the plan for the router prompt, showing step indices."""
        return self.format_for_verifier()

    def get_step(self, step_num: int) -> PlanStep | None:
        for s in self.steps:
            if s.step == step_num:
                return s
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "steps": [
                {"step": s.step, "tool": s.tool, "description": s.description,
                 "expected_output": s.expected_output}
                for s in self.steps
            ],
            "raw_response": self.raw_response,
        }


def parse_plan(response: str) -> Plan:
    """Parse the LLM's plan response into a Plan object."""
    import json
    import re

    text = response.strip()

    # Try JSON block first
    fence_match = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if fence_match:
        json_text = fence_match.group(1).strip()
    else:
        # Try raw JSON array
        json_text = text

    try:
        steps_data = json.loads(json_text)
    except json.JSONDecodeError:
        # Fallback: parse numbered list
        return _parse_text_plan(text)

    if not isinstance(steps_data, list):
        return _parse_text_plan(text)

    steps = []
    for i, s in enumerate(steps_data, 1):
        if isinstance(s, dict):
            steps.append(PlanStep(
                step=int(s.get("step", i)),
                tool=str(s.get("tool", "unknown")),
                description=str(s.get("description", "")),
                expected_output=str(s.get("expected_output", "")),
            ))

    if not steps:
        return _parse_text_plan(text)

    return Plan(steps=steps, raw_response=response)


def _parse_text_plan(text: str) -> Plan:
    """Fallback: parse a numbered text plan (1. tool: description)."""
    import re

    steps = []
    pattern = r"^\s*(\d+)[\.\)]\s*(?:\[?(\w+)\]?[\s:]+)?(.+)$"
    for line in text.split("\n"):
        match = re.match(pattern, line.strip())
        if match:
            step_num = int(match.group(1))
            tool = match.group(2) or "unknown"
            desc = match.group(3).strip()
            steps.append(PlanStep(step=step_num, tool=tool, description=desc))

    return Plan(steps=steps, raw_response=text)


def run_planner(
    *,
    model,
    question: str,
    data_summary: str,
    route_hint: str,
    difficulty: str = "medium",
) -> Plan:
    """Generate an explicit execution plan.

    Returns a Plan object with ordered steps.
    """
    from data_agent_baseline.agents.model import ModelMessage

    messages = [
        ModelMessage(role="system", content=PLANNER_SYSTEM_PROMPT),
        ModelMessage(
            role="user",
            content=PLANNER_USER_TEMPLATE.format(
                question=question,
                data_summary=data_summary[:3000],
                route_hint=route_hint[:500],
                difficulty=difficulty,
            ),
        ),
    ]

    try:
        response = model.complete(messages)
        return parse_plan(response)
    except Exception:
        # Fallback: default plan based on route
        return _default_plan(route_hint)


def _default_plan(route_hint: str) -> Plan:
    """Generate a default plan when LLM planning fails."""
    route_lower = route_hint.lower()

    if "sql" in route_lower:
        steps = [
            PlanStep(1, "profile_context", "Scan context files to get table schemas", "Table names, column lists, DuckDB table IDs"),
            PlanStep(2, "execute_data_sql", "Run SQL query to compute the answer", "Query result with requested metrics"),
            PlanStep(3, "execute_data_sql", "Verify results with a different aggregation", "Confirmation result"),
            PlanStep(4, "answer", "Submit final verified answer", "prediction.csv"),
        ]
    elif "python" in route_lower:
        steps = [
            PlanStep(1, "profile_context", "Scan context files to get schemas", "Table names, column lists"),
            PlanStep(2, "execute_python", "Run Python analysis to compute the answer", "Analysis output"),
            PlanStep(3, "answer", "Submit final verified answer", "prediction.csv"),
        ]
    elif "doc" in route_lower:
        steps = [
            PlanStep(1, "profile_context", "Scan context files to get document structure", "Document types, headings, key terms"),
            PlanStep(2, "extract_doc_records", "Extract structured records from documents", "Entity records with metrics"),
            PlanStep(3, "execute_python", "Compute answer from extracted records", "Computed result"),
            PlanStep(4, "execute_data_sql", "Verify result against source data", "Verification result"),
            PlanStep(5, "answer", "Submit final verified answer", "prediction.csv"),
        ]
    else:
        steps = [
            PlanStep(1, "profile_context", "Scan context files", "Schemas, table names, document structure"),
            PlanStep(2, "execute_data_sql", "Retrieve and compute the answer", "Query result"),
            PlanStep(3, "answer", "Submit final answer", "prediction.csv"),
        ]

    return Plan(steps=steps, raw_response="(default plan: LLM planner failed)")


# ── ROMA-inspired recursive decomposition for complex tasks ──

DECOMPOSER_SYSTEM_PROMPT = """You are a task decomposition expert. You receive a complex data analysis question and break it into independent sub-questions that can be solved in parallel or sequence.

Each sub-question should:
1. Be self-contained (can be answered independently)
2. Target a specific data source or computation
3. Have a clear expected output format

Output as JSON array:
```json
[
  {"sub_task": "description of sub-question 1", "depends_on": [], "data_source": "file or table name"},
  {"sub_task": "description of sub-question 2", "depends_on": [1], "data_source": "..."},
]
```

Only decompose if the task is TRULY complex (4+ data sources, or requires both document reading AND computation). For simple tasks, return an empty array."""

DECOMPOSER_USER_TEMPLATE = """Question: {question}

Available data files:
{file_list}

Route hint: {route_hint}
Difficulty: {difficulty}

Should this task be decomposed into sub-tasks? If yes, output JSON array. If no, output empty array []."""


@dataclass(slots=True)
class SubTask:
    description: str
    depends_on: list[int] = field(default_factory=list)
    data_source: str = ""


@dataclass(slots=True)
class TaskDecomposition:
    sub_tasks: list[SubTask] = field(default_factory=list)
    is_complex: bool = False

    @property
    def should_decompose(self) -> bool:
        return self.is_complex and len(self.sub_tasks) > 1


def should_attempt_decomposition(
    *,
    difficulty: str,
    route: str,
    file_count: int,
) -> bool:
    """Determine if recursive decomposition is worth attempting.

    Decomposition adds overhead (extra LLM calls). Only use for tasks that are:
    - Hard or extreme difficulty
    - Hybrid (doc+table) routes
    - Many files (4+)
    """
    if difficulty.lower() not in ("hard", "extreme"):
        return False
    if route in ("hybrid_doc_table", "document_first"):
        return True
    if file_count >= 4:
        return True
    return False


def run_decomposer(
    *,
    model,
    question: str,
    file_list: str,
    route_hint: str,
    difficulty: str,
) -> TaskDecomposition:
    """Attempt to recursively decompose a complex task into sub-tasks."""
    import json
    import re

    from data_agent_baseline.agents.model import ModelMessage

    messages = [
        ModelMessage(role="system", content=DECOMPOSER_SYSTEM_PROMPT),
        ModelMessage(
            role="user",
            content=DECOMPOSER_USER_TEMPLATE.format(
                question=question,
                file_list=file_list[:2000],
                route_hint=route_hint[:300],
                difficulty=difficulty,
            ),
        ),
    ]

    try:
        response = model.complete(messages)
        # Extract JSON
        fence = re.search(r"```(?:json)?\s*(.*?)```", response, re.DOTALL | re.IGNORECASE)
        json_text = fence.group(1) if fence else response
        data = json.loads(json_text.strip())
        if not isinstance(data, list) or not data:
            return TaskDecomposition(is_complex=False)

        sub_tasks = []
        for item in data:
            if isinstance(item, dict):
                sub_tasks.append(SubTask(
                    description=item.get("sub_task", item.get("description", "")),
                    depends_on=list(item.get("depends_on", [])),
                    data_source=item.get("data_source", ""),
                ))

        return TaskDecomposition(
            sub_tasks=sub_tasks,
            is_complex=len(sub_tasks) > 1,
        )
    except Exception:
        return TaskDecomposition(is_complex=False)
