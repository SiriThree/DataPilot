from __future__ import annotations

import json
import re
from dataclasses import dataclass

from data_agent_baseline.agents.model import ModelAdapter, ModelMessage, ModelStep
from data_agent_baseline.agents.prompt import (
    REACT_SYSTEM_PROMPT,
    build_observation_prompt,
    build_system_prompt,
    build_task_prompt,
)
from data_agent_baseline.agents.runtime import AgentRunResult, AgentRuntimeState, StepRecord
from data_agent_baseline.agents.step_controller import (
    controller_observation,
    override_metadata,
    review_next_step,
)
from data_agent_baseline.benchmark.schema import PublicTask
from data_agent_baseline.agents.critic import critic_check
from data_agent_baseline.tools.registry import ToolRegistry

MAX_OBSERVATION_PROMPT_CHARS = 18_000
MAX_OBSERVATION_STRING_CHARS = 8_000
MAX_OBSERVATION_LIST_ITEMS = 100
MAX_OBSERVATION_DICT_ITEMS = 80


@dataclass(frozen=True, slots=True)
class ReActAgentConfig:
    max_steps: int = 36
    use_stateful_python: bool = True


def _strip_json_fence(raw_response: str) -> str:
    text = raw_response.strip()
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.IGNORECASE | re.DOTALL).strip()
    fence_match = re.search(r"```json\s*(.*?)\s*```", text, flags=re.IGNORECASE | re.DOTALL)
    if fence_match is not None:
        return fence_match.group(1).strip()
    generic_fence_match = re.search(r"```\s*(.*?)\s*```", text, flags=re.DOTALL)
    if generic_fence_match is not None:
        return generic_fence_match.group(1).strip()
    unterminated_fence = re.match(r"```(?:json)?\s*(.*)$", text, flags=re.IGNORECASE | re.DOTALL)
    if unterminated_fence is not None:
        return unterminated_fence.group(1).strip()
    return text


def _extract_json_object_candidate(text: str) -> str:
    decoder = json.JSONDecoder()
    for match in re.finditer(r"{", text):
        candidate = text[match.start():]
        try:
            payload, end = decoder.raw_decode(candidate)
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict):
            continue
        if {"thought", "action", "action_input"}.issubset(payload):
            return candidate[:end].strip()
    return text


def _load_single_json_object(text: str) -> dict[str, object]:
    payload, end = json.JSONDecoder().raw_decode(text)
    remainder = text[end:].strip()
    if remainder:
        cleaned_remainder = re.sub(r"(?:\\[nrt])+", "", remainder).strip()
        if cleaned_remainder:
            raise ValueError("Model response must contain only one JSON object.")
    if not isinstance(payload, dict):
        raise ValueError("Model response must be a JSON object.")
    return payload


def _escape_raw_control_chars_in_strings(text: str) -> str:
    """Escape raw control characters that models sometimes place inside JSON strings."""
    output: list[str] = []
    in_string = False
    escaped = False

    for char in text:
        if not in_string:
            output.append(char)
            if char == '"':
                in_string = True
            continue

        if escaped:
            output.append(char)
            escaped = False
            continue
        if char == "\\":
            output.append(char)
            escaped = True
            continue
        if char == '"':
            output.append(char)
            in_string = False
            continue
        if char == "\n":
            output.append("\\n")
            continue
        if char == "\r":
            output.append("\\r")
            continue
        if char == "\t":
            output.append("\\t")
            continue
        if ord(char) < 0x20:
            output.append(f"\\u{ord(char):04x}")
            continue
        output.append(char)

    return "".join(output)


def _repair_json(text: str) -> str:
    """Attempt common JSON repairs before giving up."""
    # Strip trailing commas before } or ]
    text = re.sub(r",\s*([}\]])", r"\1", text)
    # Fix raw newlines/tabs that appear inside JSON string values.
    text = _escape_raw_control_chars_in_strings(text)
    # Remove trailing text after the JSON object closes
    brace_end = text.rfind("}")
    bracket_end = text.rfind("]")
    end = max(brace_end, bracket_end)
    if end > 0 and end < len(text) - 1:
        text = text[:end + 1]
    return text


def _compact_for_prompt(value: object, *, depth: int = 0) -> object:
    """Bound prior tool observations so one large file cannot exhaust model context."""
    if depth >= 10:
        return "<nested content omitted>"
    if isinstance(value, str):
        if len(value) > MAX_OBSERVATION_STRING_CHARS:
            omitted = len(value) - MAX_OBSERVATION_STRING_CHARS
            return value[:MAX_OBSERVATION_STRING_CHARS] + f"... <truncated {omitted} chars>"
        return value
    if isinstance(value, list):
        items = [_compact_for_prompt(item, depth=depth + 1) for item in value[:MAX_OBSERVATION_LIST_ITEMS]]
        if len(value) > MAX_OBSERVATION_LIST_ITEMS:
            items.append(f"... <{len(value) - MAX_OBSERVATION_LIST_ITEMS} more items omitted>")
        return items
    if isinstance(value, dict):
        compact: dict[str, object] = {}
        for index, (key, child) in enumerate(value.items()):
            if index >= MAX_OBSERVATION_DICT_ITEMS:
                compact["..."] = f"<{len(value) - MAX_OBSERVATION_DICT_ITEMS} more keys omitted>"
                break
            compact[str(key)] = _compact_for_prompt(child, depth=depth + 1)
        return compact
    return value


def _compact_observation_prompt(observation: dict[str, object]) -> str:
    compact = _compact_for_prompt(observation)
    rendered = build_observation_prompt(compact if isinstance(compact, dict) else {"content": compact})
    if len(rendered) <= MAX_OBSERVATION_PROMPT_CHARS:
        return rendered
    return (
        rendered[:MAX_OBSERVATION_PROMPT_CHARS]
        + f"\n... <observation truncated {len(rendered) - MAX_OBSERVATION_PROMPT_CHARS} chars>"
    )


def parse_model_step(raw_response: str) -> ModelStep:
    normalized = _strip_json_fence(raw_response)
    candidate = _extract_json_object_candidate(normalized)
    errors = []
    # Try direct parse first, then repaired version
    for attempt, text in enumerate([normalized, candidate, _repair_json(candidate)]):
        try:
            payload = _load_single_json_object(text)
            break
        except (ValueError, json.JSONDecodeError) as exc:
            errors.append(str(exc))
    else:
        raise ValueError(f"JSON parse failed after repair: {'; '.join(errors)}")

    thought = payload.get("thought", "")
    action = payload.get("action")
    action_input = payload.get("action_input", {})
    if not isinstance(thought, str):
        raise ValueError("thought must be a string.")
    if not isinstance(action, str) or not action:
        raise ValueError("action must be a non-empty string.")
    if not isinstance(action_input, dict):
        raise ValueError("action_input must be a JSON object.")

    return ModelStep(
        thought=thought,
        action=action,
        action_input=action_input,
        raw_response=raw_response,
    )


class ReActAgent:
    def __init__(
        self,
        *,
        model: ModelAdapter,
        tools: ToolRegistry,
        config: ReActAgentConfig | None = None,
        system_prompt: str | None = None,
        route_hint: str = "",
    ) -> None:
        self.model = model
        self.tools = tools
        self.config = config or ReActAgentConfig()
        self.system_prompt = system_prompt or REACT_SYSTEM_PROMPT
        self.route_hint = route_hint

        if self.config.use_stateful_python:
            tools.use_stateful_python = True

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
        messages.append(ModelMessage(role="user", content=build_task_prompt(task, route_hint=self.route_hint)))
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

        for step_index in range(1, self.config.max_steps + 1):
            # ── Inner Shell: step budget warning ──
            steps_remaining = self.config.max_steps - step_index
            budget_hint = ""
            if steps_remaining <= 3:
                budget_hint = (
                    f"\n[SYSTEM: Only {steps_remaining} step(s) remaining. "
                    "If you have enough data, use the `answer` action NOW. "
                    "Do not explore further. On the final step, submit the best supported answer.]"
                )

            # ── Inner Shell: recovery hint ──
            recovery_hint = ""
            if self.tools.should_force_recovery:
                recovery_hint = (
                    "\n[SYSTEM: You have had 3+ consecutive errors. "
                    "Your JSON output format may be malformed. "
                    "Simplify your response — use basic key names, no special characters.]"
                )

            raw_response = self.model.complete(
                self._build_messages(task, state, control_hint=budget_hint + recovery_hint)
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
                        remaining_steps=self.config.max_steps - step_index,
                    )
                    state.steps.append(
                        StepRecord(
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
                        )
                    )
                    continue

                action = model_step.action
                action_input = model_step.action_input
                controller_override: dict[str, object] | None = None
                if controller_decision.should_override:
                    action = controller_decision.override_action or action
                    action_input = controller_decision.override_action_input or {}
                    controller_override = override_metadata(
                        decision=controller_decision,
                        attempted_action=model_step.action,
                        attempted_action_input=model_step.action_input,
                    )

                tool_result = self.tools.execute(task, action, action_input)
                observation = {
                    "ok": tool_result.ok,
                    "tool": action,
                    "content": tool_result.content,
                }
                if controller_override:
                    observation.update(controller_override)

                # ── Inner Shell: Critic check ──
                critic_result = critic_check(
                    current_observation=observation,
                    question=task.question,
                )
                if critic_result.get("warnings"):
                    observation["_critic_warnings"] = critic_result["warnings"]
                observation["_critic_pass"] = critic_result["pass"]

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

                if tool_result.is_terminal:
                    state.answer = tool_result.answer
                    break
            except Exception as exc:
                observation = {
                    "ok": False,
                    "error": str(exc),
                }
                state.steps.append(
                    StepRecord(
                        step_index=step_index,
                        thought="",
                        action="__error__",
                        action_input={},
                        raw_response=raw_response,
                        observation=observation,
                        ok=False,
                    )
                )

        if state.answer is None and state.failure_reason is None:
            state.failure_reason = "Agent did not submit an answer within max_steps."

        return AgentRunResult(
            task_id=task.task_id,
            answer=state.answer,
            steps=list(state.steps),
            failure_reason=state.failure_reason,
        )
