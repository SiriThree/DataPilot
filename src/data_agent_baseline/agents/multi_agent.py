"""Multi-Agent Loop: Plan → Execute → Verify → Router → (Fix or Continue).

Replaces the flat ReAct loop with a structured multi-phase pipeline:

Phase 0: Profile (get data landscape)
Phase 1: Plan (LLM generates explicit step-by-step plan)
Phase 2: Execute + Verify loop:
  - Execute current plan step
  - Critic check (rule-based, fast)
  - Verifier: LLM judges if results SUFFICIENTLY answer the question
  - If INSUFFICIENT: Router decides ADD_STEP or FIX_STEP
  - If SUFFICIENT: proceed to answer
Phase 3: Answer with best evidence

For complex Hard tasks, ROMA-inspired recursive decomposition:
  - Break task into sub-tasks
  - Execute sub-tasks in dependency order
  - Aggregate results

Configuration:
  - use_verifier: enable LLM verifier (default True)
  - use_router: enable LLM router for ambiguous cases (default True)
  - use_debugger: enable schema-aware code repair (default True)
  - verifier_frequency: how often to run verifier ("every_compute" | "every_step" | "near_end")
  - max_plan_steps: max plan steps to generate (default 8)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from data_agent_baseline.agents.model import ModelAdapter, ModelMessage
from data_agent_baseline.agents.prompt import (
    REACT_SYSTEM_PROMPT,
    build_system_prompt,
    build_task_prompt,
)
from data_agent_baseline.agents.runtime import AgentRunResult, AgentRuntimeState, StepRecord
from data_agent_baseline.agents.step_controller import (
    controller_observation,
    override_metadata,
    review_next_step,
)
from data_agent_baseline.agents.critic import critic_check
from data_agent_baseline.agents.verifier import (
    build_evidence_summary,
    run_llm_verifier,
    should_force_verifier,
    should_skip_verifier,
)
from data_agent_baseline.agents.router import (
    RouterDecision,
    decide_router_action,
    format_steps_for_router,
    run_llm_router,
)
from data_agent_baseline.agents.debugger import (
    build_schema_context_from_profile,
    build_schema_context_from_sqlite_inspect,
    classify_error,
    run_debugger,
)
from data_agent_baseline.agents.planner import (
    Plan,
    PlanStep,
    TaskDecomposition,
    run_decomposer,
    run_planner,
    should_attempt_decomposition,
)
from data_agent_baseline.benchmark.schema import PublicTask
from data_agent_baseline.tools.registry import ToolRegistry

# Re-use these from react.py (they're stable utilities)
from data_agent_baseline.agents.react import (
    _compact_observation_prompt,
    parse_model_step,
)


@dataclass(frozen=True, slots=True)
class MultiAgentConfig:
    max_steps: int = 36
    use_verifier: bool = True
    use_router: bool = True
    use_debugger: bool = True
    use_decomposer: bool = True  # ROMA-inspired recursive decomposition
    verifier_frequency: str = "every_compute"  # "every_compute" | "every_step" | "near_end"
    max_plan_steps: int = 8
    max_debug_retries: int = 2  # Max debug attempts per step
    max_verifier_rounds: int = 12  # Max plan-execute-verify rounds


def _fallback_answer_step(step_num: int) -> PlanStep:
    return PlanStep(
        step=max(step_num, 1),
        tool="answer",
        description="Submit final answer with best available evidence",
        expected_output="prediction.csv",
    )


def _ensure_current_plan_step(plan_steps: list[PlanStep], current_idx: int) -> int:
    if current_idx < 0:
        current_idx = 0
    if current_idx >= len(plan_steps):
        plan_steps.append(_fallback_answer_step(len(plan_steps) + 1))
        current_idx = len(plan_steps) - 1
    return current_idx


def _summarize_profile_for_plan(profile_content: dict[str, Any]) -> str:
    """Extract compact data summary from profile_context result for the planner."""
    parts: list[str] = []
    files = profile_content.get("files", {})
    if not isinstance(files, dict):
        return str(profile_content)[:2000]

    for fname, finfo in files.items():
        if not isinstance(finfo, dict):
            continue
        ftype = finfo.get("type", "unknown")
        if ftype in ("csv", "json"):
            cols = finfo.get("columns", [])
            duckdb = finfo.get("duckdb_table", "")
            row_count = finfo.get("row_count", "?")
            parts.append(f"  {fname} ({ftype}, {row_count} rows, table={duckdb}, cols={cols[:20]})")
        elif ftype in ("sqlite", "db"):
            tables = finfo.get("tables", [])
            for t in tables[:5]:
                tname = t.get("name", "?") if isinstance(t, dict) else str(t)
                tcols = t.get("columns", []) if isinstance(t, dict) else []
                parts.append(f"  {fname} ({ftype}, table={tname}, cols={tcols[:20]})")
        elif ftype == "doc":
            headings = finfo.get("headings", [])
            terms = finfo.get("terms", [])
            size = finfo.get("size_bytes", "?")
            parts.append(f"  {fname} ({ftype}, {size}B, headings={headings[:10]}, terms={terms[:10]})")
        else:
            parts.append(f"  {fname} ({ftype})")

    summary = profile_content.get("summary", "")
    if summary:
        parts.insert(0, f"Summary: {summary}")

    return "\n".join(parts)[:3000]


def _summarize_latest_result(step_record: StepRecord) -> str:
    """Create a compact summary of the latest tool execution result."""
    parts: list[str] = []
    action = step_record.action
    obs = step_record.observation
    content = obs.get("content", {}) if isinstance(obs, dict) else {}

    if not isinstance(content, dict):
        return f"[{action}] {str(content)[:500]}"

    # Key fields for each tool type
    if action in ("execute_data_sql", "execute_context_sql"):
        rows = content.get("rows", [])
        cols = content.get("columns", [])
        row_count = len(rows) if isinstance(rows, list) else 0
        parts.append(f"[{action}] returned {row_count} rows, columns: {cols[:10]}")
        if rows:
            parts.append(f"  first row: {rows[0]}")
        if row_count > 1:
            parts.append(f"  last row: {rows[-1]}")

    elif action == "execute_python":
        output = content.get("output", "")
        stderr = content.get("stderr", "")
        success = content.get("success", False)
        parts.append(f"[{action}] success={success}")
        if output:
            parts.append(f"  output: {str(output)[:300]}")
        if stderr:
            parts.append(f"  stderr: {str(stderr)[:200]}")

    elif action in ("extract_doc_records", "extract_doc_evidence_table"):
        record_count = content.get("record_count", content.get("entity_count", "?"))
        fields = content.get("fields_used", content.get("fields", []))
        parts.append(f"[{action}] extracted {record_count} records, fields: {fields}")

    elif action == "ground_thresholds":
        probes = content.get("recommended_probe_summary", "")
        parts.append(f"[{action}] {str(probes)[:300]}")

    elif action == "profile_context":
        files = content.get("files", {})
        file_count = len(files) if isinstance(files, dict) else 0
        parts.append(f"[{action}] profiled {file_count} files")

    elif action == "search_doc":
        match_count = content.get("match_count", 0)
        parts.append(f"[{action}] found {match_count} matches")

    else:
        # Generic summary
        for key in ("summary", "result", "preview"):
            if key in content:
                parts.append(f"[{action}] {key}: {str(content[key])[:300]}")
                break
        else:
            parts.append(f"[{action}] keys: {list(content.keys())[:10]}")

    # Add error info if present
    error = content.get("error", "")
    if error:
        parts.append(f"  ERROR: {str(error)[:200]}")

    # Add critic warnings
    warnings = obs.get("_critic_warnings", [])
    if warnings:
        parts.append(f"  warnings: {warnings}")

    return "\n".join(parts)[:1000]


class MultiAgentLoop:
    """Orchestrates the Plan→Execute→Verify→Router→(Fix|Continue) loop."""

    def __init__(
        self,
        *,
        model: ModelAdapter,
        tools: ToolRegistry,
        config: MultiAgentConfig | None = None,
        system_prompt: str | None = None,
        route_hint: str = "",
        difficulty: str = "medium",
    ) -> None:
        self.model = model
        self.tools = tools
        self.config = config or MultiAgentConfig()
        self.system_prompt = system_prompt or REACT_SYSTEM_PROMPT
        self.route_hint = route_hint
        self.difficulty = difficulty

        # Enable stateful Python interpreter (persistent DataFrame/variables)
        tools.use_stateful_python = True

        # State accumulated during the run
        self._profile_data: dict[str, Any] = {}
        self._debug_attempts: dict[int, int] = {}

    def _build_messages(
        self,
        task: PublicTask,
        state: AgentRuntimeState,
        *,
        control_hint: str = "",
    ) -> list[ModelMessage]:
        system_content = build_system_prompt(
            self.tools.describe_for_prompt(),
            system_prompt=self.system_prompt,
        )
        messages = [ModelMessage(role="system", content=system_content)]
        messages.append(
            ModelMessage(role="user", content=build_task_prompt(task, route_hint=self.route_hint))
        )
        for step in state.steps:
            messages.append(ModelMessage(role="assistant", content=step.raw_response))
            messages.append(
                ModelMessage(role="user", content=_compact_observation_prompt(step.observation))
            )
        if control_hint:
            messages.append(ModelMessage(role="user", content=control_hint))
        return messages

    def run(self, task: PublicTask) -> AgentRunResult:
        state = AgentRuntimeState()

        # ── Phase 0: Profile ──
        profile_result = self.tools.execute(task, "profile_context", {})
        profile_content = profile_result.content if profile_result.ok else {}
        self._profile_data = profile_content

        state.steps.append(StepRecord(
            step_index=0,
            thought="Phase 0: Profile context to understand data landscape",
            action="profile_context",
            action_input={},
            raw_response="",
            observation={"ok": True, "tool": "profile_context", "content": profile_content},
            ok=True,
        ))

        # ── Phase 0.5: ROMA-inspired decomposition (for complex Hard tasks) ──
        decomposition = TaskDecomposition(is_complex=False)
        if self.config.use_decomposer and should_attempt_decomposition(
            difficulty=self.difficulty,
            route=self.route_hint.split("\n")[0] if self.route_hint else "unknown",
            file_count=len(profile_content.get("files", {})),
        ):
            file_list = "\n".join(
                f"  {fname} ({finfo.get('type', '?')})"
                for fname, finfo in (profile_content.get("files", {}) or {}).items()
            )
            decomposition = run_decomposer(
                model=self.model,
                question=task.question,
                file_list=file_list,
                route_hint=self.route_hint,
                difficulty=self.difficulty,
            )

        # ── Phase 1: Plan ──
        data_summary = _summarize_profile_for_plan(profile_content)
        plan = run_planner(
            model=self.model,
            question=task.question,
            data_summary=data_summary,
            route_hint=self.route_hint,
            difficulty=self.difficulty,
        )

        state.steps.append(StepRecord(
            step_index=0,  # step 0 for plan
            thought=f"Phase 1: Generated plan ({plan.step_count} steps)\n{plan.format_for_verifier()}",
            action="__plan__",
            action_input={"plan": plan.to_dict()},
            raw_response=plan.raw_response,
            observation={
                "ok": True,
                "tool": "__plan__",
                "content": {
                    "plan_steps": plan.step_count,
                    "plan_text": plan.format_for_verifier(),
                    "decomposition": {
                        "is_complex": decomposition.is_complex,
                        "sub_tasks": [s.description for s in decomposition.sub_tasks],
                    } if decomposition.is_complex else None,
                },
            },
            ok=True,
        ))

        # ── Phase 2: Execute→Verify→Router loop ──
        current_plan_step_idx = 0
        plan_steps = plan.steps
        error_count = 0
        verifier_rounds = 0
        step_index = len(state.steps) + 1  # Continue from where profile+plan ended

        while step_index <= self.config.max_steps and verifier_rounds < self.config.max_verifier_rounds:
            # Determine which plan step to execute
            current_plan_step_idx = _ensure_current_plan_step(plan_steps, current_plan_step_idx)
            target_step = plan_steps[current_plan_step_idx]
            steps_remaining = self.config.max_steps - step_index + 1

            # Skip profile_context in plan steps (already done)
            if target_step.tool == "profile_context" and current_plan_step_idx == 0:
                current_plan_step_idx += 1
                continue

            # ── Build prompt with plan guidance ──
            plan_context = self._build_plan_context(
                plan_steps, current_plan_step_idx, steps_remaining
            )
            budget_hint = self._build_budget_hint(steps_remaining)
            recovery_hint = self._build_recovery_hint()

            raw_response = self.model.complete(
                self._build_messages(
                    task, state,
                    control_hint=plan_context + budget_hint + recovery_hint,
                )
            )

            try:
                model_step = parse_model_step(raw_response)
                controller_decision = review_next_step(
                    task=task,
                    steps=state.steps,
                    step_index=step_index,
                    max_steps=self.config.max_steps,
                    thought=model_step.thought,
                    action=model_step.action,
                    action_input=model_step.action_input,
                )

                if controller_decision.should_block:
                    observation = controller_observation(
                        decision=controller_decision,
                        attempted_action=model_step.action,
                        attempted_action_input=model_step.action_input,
                        remaining_steps=steps_remaining,
                    )
                    state.steps.append(StepRecord(
                        step_index=step_index,
                        thought=model_step.thought,
                        action="__step_controller__",
                        action_input={
                            "attempted_action": model_step.action,
                            "attempted_action_input": model_step.action_input,
                            "code": controller_decision.code,
                        },
                        raw_response=raw_response,
                        observation=observation,
                        ok=False,
                    ))
                    step_index += 1
                    continue

                action = model_step.action
                action_input = model_step.action_input
                controller_override = None
                if controller_decision.should_override:
                    action = controller_decision.override_action or action
                    action_input = controller_decision.override_action_input or {}
                    controller_override = override_metadata(
                        decision=controller_decision,
                        attempted_action=model_step.action,
                        attempted_action_input=model_step.action_input,
                    )

                # ── Execute tool ──
                tool_result = self.tools.execute(task, action, action_input)
                observation: dict[str, Any] = {
                    "ok": tool_result.ok,
                    "tool": action,
                    "content": tool_result.content,
                }
                if controller_override:
                    observation.update(controller_override)

                # ── Debugger: auto-repair failed executions ──
                debug_attempts = self._debug_attempts.get(current_plan_step_idx, 0)
                if self.config.use_debugger and not tool_result.ok and debug_attempts < self.config.max_debug_retries:
                    repaired = self._attempt_debug(
                        action=action,
                        action_input=action_input,
                        error_content=tool_result.content,
                    )
                    if repaired:
                        debug_attempts += 1
                        self._debug_attempts[current_plan_step_idx] = debug_attempts
                        observation["_debugged"] = True
                        observation["_debug_attempt"] = debug_attempts
                        # Re-execute the repaired tool
                        try:
                            fixed_input = dict(action_input)
                            if action in ("execute_data_sql", "execute_context_sql"):
                                fixed_input["sql"] = repaired
                            elif action == "execute_python":
                                fixed_input["code"] = repaired
                            retry_result = self.tools.execute(task, action, fixed_input)
                            if retry_result.ok:
                                observation = {
                                    "ok": True,
                                    "tool": action,
                                    "content": retry_result.content,
                                }
                                observation["_debugged"] = True
                                observation["_original_error"] = str(tool_result.content.get("error", ""))[:200]
                        except Exception:
                            pass  # Keep original error

                # ── Critic check (rule-based, fast) ──
                critic_result = critic_check(
                    current_observation=observation,
                    question=task.question,
                )
                if critic_result.get("warnings"):
                    observation["_critic_warnings"] = critic_result["warnings"]
                observation["_critic_pass"] = critic_result["pass"]

                # ── Record step ──
                step_record = StepRecord(
                    step_index=step_index,
                    thought=model_step.thought,
                    action=action,
                    action_input=action_input,
                    raw_response=raw_response,
                    observation=observation,
                    ok=tool_result.ok,
                )
                state.steps.append(step_record)

                if not tool_result.ok:
                    error_count += 1

                # ── Terminal action ──
                if tool_result.is_terminal:
                    state.answer = tool_result.answer
                    return AgentRunResult(
                        task_id=task.task_id,
                        answer=state.answer,
                        steps=list(state.steps),
                        failure_reason=state.failure_reason,
                    )

                # ── Verifier: LLM judges sufficiency ──
                should_verify = (
                    self.config.use_verifier
                    and not should_skip_verifier(
                        step_index=step_index, action=action, max_steps=self.config.max_steps,
                    )
                    and (
                        should_force_verifier(
                            step_index=step_index,
                            max_steps=self.config.max_steps,
                            steps_remaining=steps_remaining,
                            action=action,
                        )
                        or self.config.verifier_frequency == "every_step"
                        or (
                            self.config.verifier_frequency == "every_compute"
                            and action in ("execute_data_sql", "execute_context_sql", "execute_python")
                        )
                    )
                )

                if should_verify:
                    verifier_rounds += 1
                    evidence = build_evidence_summary(
                        [s.to_dict() for s in state.steps],
                    )
                    latest = _summarize_latest_result(step_record)
                    verifier_result = run_llm_verifier(
                        model=self.model,
                        question=task.question,
                        plan=plan.format_for_verifier(),
                        latest_result=latest,
                        evidence_summary=evidence,
                        step_count=step_index,
                        steps_remaining=steps_remaining,
                    )
                    observation["_verifier"] = verifier_result

                    if verifier_result["sufficient"]:
                        # Generate final answer
                        answer_result = self._force_answer(task, state, step_index, plan)
                        if answer_result:
                            return answer_result

                    else:
                        # ── Router: decide next action ──
                        router_decision = self._route(
                            verifier_reason=verifier_result["reason"],
                            steps=[s.to_dict() for s in state.steps],
                            step_count=step_index,
                            plan_steps=plan_steps,
                            plan=plan,
                            error_count=error_count,
                            question=task.question,
                        )
                        observation["_router"] = {
                            "action": router_decision.action,
                            "target_step": router_decision.target_step,
                            "reasoning": router_decision.reasoning,
                        }

                        if router_decision.should_answer_now:
                            answer_result = self._force_answer(task, state, step_index, plan)
                            if answer_result:
                                return answer_result

                        elif router_decision.should_fix_step and router_decision.target_step is not None:
                            # Truncate plan back to before the erroneous step
                            fix_target = min(router_decision.target_step - 1, len(plan_steps))
                            plan_steps = plan_steps[:fix_target]
                            current_plan_step_idx = fix_target
                            # Re-plan from here
                            new_plan = run_planner(
                                model=self.model,
                                question=task.question,
                                data_summary=_summarize_profile_for_plan(self._profile_data),
                                route_hint=self.route_hint,
                                difficulty=self.difficulty,
                            )
                            plan_steps.extend(new_plan.steps)
                            observation["_replanned"] = True

                # ── Advance to next plan step ──
                current_plan_step_idx += 1

            except Exception as exc:
                observation = {"ok": False, "error": str(exc)}
                state.steps.append(StepRecord(
                    step_index=step_index,
                    thought="",
                    action="__error__",
                    action_input={},
                    raw_response=raw_response,
                    observation=observation,
                    ok=False,
                ))
                error_count += 1

            step_index += 1

        # ── Budget exhausted: force answer ──
        if state.answer is None:
            return self._force_answer(task, state, step_index, plan) or AgentRunResult(
                task_id=task.task_id,
                answer=state.answer,
                steps=list(state.steps),
                failure_reason="Agent did not submit an answer within max_steps.",
            )

        return AgentRunResult(
            task_id=task.task_id,
            answer=state.answer,
            steps=list(state.steps),
            failure_reason=state.failure_reason,
        )

    def _attempt_debug(
        self,
        action: str,
        action_input: dict[str, Any],
        error_content: dict[str, Any],
    ) -> str | None:
        """Attempt to repair failed code using schema-aware debugger."""
        if action not in ("execute_data_sql", "execute_context_sql", "execute_python"):
            return None

        error_text = error_content.get("error", str(error_content))
        error_class = classify_error(error_text)

        if action in ("execute_data_sql", "execute_context_sql"):
            code = str(action_input.get("sql", ""))
            tool_type = "sql"
        else:
            code = str(action_input.get("code", ""))
            tool_type = "python"

        schema_context = build_schema_context_from_profile(self._profile_data)
        if action == "execute_context_sql":
            db_context = build_schema_context_from_sqlite_inspect(
                error_content.get("schema", {})
            )
            if db_context:
                schema_context = db_context

        return run_debugger(
            model=self.model,
            tool_type=tool_type,
            code=code,
            error_text=error_text,
            schema_context=schema_context,
            instructions=error_class["hint"],
        )

    def _route(
        self,
        verifier_reason: str,
        steps: list[dict[str, Any]],
        step_count: int,
        plan_steps: list[PlanStep],
        plan: Plan,
        error_count: int,
        question: str,
    ) -> RouterDecision:
        """Route after verification failure: add_step, fix_step, or answer_now."""
        # Try rule-based first
        decision = decide_router_action(
            verifier_reason=verifier_reason,
            steps=steps,
            step_count=step_count,
            max_steps=self.config.max_steps,
            plan_steps=[s.description for s in plan_steps],
            error_count=error_count,
        )

        # Use LLM router for low-confidence cases
        if self.config.use_router and decision.confidence < 0.6:
            recent = format_steps_for_router(steps, limit=3)
            error_summary = f"{error_count} errors in {step_count} steps"
            llm_decision = run_llm_router(
                model=self.model,
                question=question,
                plan=plan.format_for_router(),
                verifier_reason=verifier_reason,
                recent_steps=recent,
                error_summary=error_summary,
                step_count=step_count,
                max_steps=self.config.max_steps,
            )
            if llm_decision.confidence > decision.confidence:
                return llm_decision

        return decision

    def _force_answer(
        self,
        task: PublicTask,
        state: AgentRuntimeState,
        step_index: int,
        plan: Plan,
    ) -> AgentRunResult | None:
        """Force the agent to submit an answer with best available evidence."""
        steps_remaining = self.config.max_steps - step_index

        if steps_remaining < 0:
            # No steps left at all — answer is impossible
            state.failure_reason = "Budget fully exhausted before answer could be submitted."
            return AgentRunResult(
                task_id=task.task_id,
                answer=None,
                steps=list(state.steps),
                failure_reason=state.failure_reason,
            )

        answer_hint = (
            "\n[SYSTEM: Verification indicates you have sufficient evidence. "
            "Submit your best answer NOW using the `answer` action. "
            "Include only the columns the question asks for. "
            "Do not include explanation or intermediate columns.]"
        )

        try:
            raw_response = self.model.complete(
                self._build_messages(task, state, control_hint=answer_hint)
            )
            model_step = parse_model_step(raw_response)

            if model_step.action == "answer":
                tool_result = self.tools.execute(task, "answer", model_step.action_input)
                if tool_result.is_terminal:
                    state.steps.append(StepRecord(
                        step_index=step_index + 1,
                        thought=model_step.thought,
                        action="answer",
                        action_input=model_step.action_input,
                        raw_response=raw_response,
                        observation={"ok": True, "tool": "answer", "content": tool_result.content},
                        ok=True,
                    ))
                    state.answer = tool_result.answer
                    return AgentRunResult(
                        task_id=task.task_id,
                        answer=state.answer,
                        steps=list(state.steps),
                        failure_reason=None,
                    )

            # Model didn't answer — record the step and continue
            state.steps.append(StepRecord(
                step_index=step_index + 1,
                thought=model_step.thought,
                action=model_step.action,
                action_input=model_step.action_input,
                raw_response=raw_response,
                observation={"ok": True, "tool": model_step.action,
                             "content": {"note": "forced answer attempt; model chose different action"}},
                ok=True,
            ))
            return None
        except Exception:
            state.failure_reason = "Failed to force answer: model call or parsing error."
            return AgentRunResult(
                task_id=task.task_id,
                answer=None,
                steps=list(state.steps),
                failure_reason=state.failure_reason,
            )

    def _build_plan_context(
        self,
        plan_steps: list[PlanStep],
        current_idx: int,
        steps_remaining: int,
    ) -> str:
        """Build a plan-aware context hint for the current step."""
        hint_parts = ["\n[PLAN PROGRESS]"]

        for i, s in enumerate(plan_steps):
            if i < current_idx:
                hint_parts.append(f"  [DONE] Step {s.step}: {s.description}")
            elif i == current_idx:
                hint_parts.append(f"  [CURRENT] Step {s.step}: {s.description}")
                if s.expected_output:
                    hint_parts.append(f"    Expected: {s.expected_output}")
            else:
                hint_parts.append(f"  [PENDING] Step {s.step}: {s.description}")

        hint_parts.append(f"\nYou are on step {plan_steps[current_idx].step if current_idx < len(plan_steps) else '?'}. "
                          f"Focus on this step. You have {steps_remaining} step(s) remaining.")

        return "\n".join(hint_parts)

    def _build_budget_hint(self, steps_remaining: int) -> str:
        if steps_remaining <= 3:
            return (
                f"\n\n[SYSTEM: Only {steps_remaining} step(s) remaining. "
                "If you have enough data, use `answer` NOW. "
                "On the final step, submit the best supported answer.]"
            )
        return ""

    def _build_recovery_hint(self) -> str:
        if self.tools.should_force_recovery:
            return (
                "\n[SYSTEM: You have had 3+ consecutive errors. "
                "Simplify: use basic tool calls, check column/table names.]"
            )
        return ""
