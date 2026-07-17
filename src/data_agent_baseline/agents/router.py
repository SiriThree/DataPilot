"""Router: decides next action when verification fails.

Modeled after DS-STAR's A_router. When the verifier says INSUFFICIENT,
the router decides:
  - ADD_STEP: Keep current plan, generate the next step
  - FIX_STEP_K: Truncate plan back to before step K, re-plan from there
  - ANSWER_NOW: Budget critical, submit best answer immediately

The router can operate in two modes:
  - Rule-based (fast, deterministic): for clear-cut cases
  - LLM-based (more nuanced): for ambiguous situations
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class RouterDecision:
    action: str  # "add_step" | "fix_step" | "answer_now"
    target_step: int | None = None  # for fix_step: truncate to step K-1, re-plan K
    reasoning: str = ""
    confidence: float = 1.0

    @property
    def should_add_step(self) -> bool:
        return self.action == "add_step"

    @property
    def should_fix_step(self) -> bool:
        return self.action == "fix_step"

    @property
    def should_answer_now(self) -> bool:
        return self.action == "answer_now"


ROUTER_SYSTEM_PROMPT = """You are a data analysis router. The verifier has determined that the current results are INSUFFICIENT to answer the question. Your job is to decide the best recovery strategy.

Options:
1. "ADD_STEP: <reason>" — the plan is on the right track but needs another step. Keep all current work and generate the next logical step.
2. "FIX_STEP <K>: <reason>" — step K (1-indexed) was wrong. Truncate the plan back to before step K and re-plan from there. Use this when a specific step produced incorrect results.

Decision guidelines:
- If the latest step failed with an error → FIX_STEP targeting that step
- If results are partially correct but incomplete → ADD_STEP
- If a wrong table/column was used → FIX_STEP targeting the step that chose it
- If the question needs more computation than planned → ADD_STEP
- If results contradict earlier evidence → FIX_STEP targeting the contradiction source

Return EXACTLY one line in the format above."""

ROUTER_USER_TEMPLATE = """Question: {question}

Current plan (1-indexed):
{plan}

Verifier feedback: {verifier_reason}

Last 3 step results:
{recent_steps}

Error history: {error_summary}

Steps taken: {step_count}/{max_steps}
Steps remaining: {steps_remaining}

What recovery action? ADD_STEP or FIX_STEP <K>?"""


def decide_router_action(
    *,
    verifier_reason: str,
    steps: list[dict[str, Any]],
    step_count: int,
    max_steps: int,
    plan_steps: list[str],
    error_count: int = 0,
) -> RouterDecision:
    """Rule-based router: fast, deterministic decisions for clear cases.

    LLM router is only invoked when the situation is genuinely ambiguous.
    """
    steps_remaining = max_steps - step_count

    # Rule 1: Budget critical → answer now
    if steps_remaining <= 1:
        return RouterDecision(
            action="answer_now",
            reasoning="Only 1 step remaining; must submit best answer now",
            confidence=0.95,
        )

    # Rule 2: Last step was an error → fix that step
    if steps:
        last_step = steps[-1]
        last_ok = last_step.get("ok", False)
        last_action = last_step.get("action", "")
        if not last_ok and last_action not in ("__step_controller__", "__error__"):
            failed_step_idx = len(plan_steps)  # the current planned step
            return RouterDecision(
                action="fix_step",
                target_step=failed_step_idx,
                reasoning=f"Step {failed_step_idx} failed; fix and retry",
                confidence=0.9,
            )

    # Rule 3: Budget constraint → answer now
    if steps_remaining <= 2:
        return RouterDecision(
            action="add_step",
            reasoning=f"Only {steps_remaining} steps left; try one more then answer",
            confidence=0.7,
        )

    # Rule 4: Too many errors → answer now with best evidence
    if error_count >= 5:
        return RouterDecision(
            action="answer_now",
            reasoning=f"{error_count} errors accumulated; best effort answer now",
            confidence=0.85,
        )

    # Rule 5: More than 3 consecutive compute steps → add step
    compute_actions = {"execute_context_sql", "execute_data_sql", "execute_python"}
    recent_compute = sum(
        1 for s in steps[-4:]
        if s.get("action") in compute_actions
    )
    if recent_compute >= 3 and len(plan_steps) == len(steps):
        return RouterDecision(
            action="add_step",
            reasoning="Multiple compute steps executed; need analysis/verification step",
            confidence=0.7,
        )

    # Default: ADD_STEP (optimistic — assume we're on the right track)
    return RouterDecision(
        action="add_step",
        reasoning="Continue with next planned step",
        confidence=0.5,
    )


def run_llm_router(
    *,
    model,
    question: str,
    plan: str,
    verifier_reason: str,
    recent_steps: str,
    error_summary: str,
    step_count: int,
    max_steps: int,
) -> RouterDecision:
    """LLM-based router for ambiguous cases."""
    from data_agent_baseline.agents.model import ModelMessage

    messages = [
        ModelMessage(role="system", content=ROUTER_SYSTEM_PROMPT),
        ModelMessage(
            role="user",
            content=ROUTER_USER_TEMPLATE.format(
                question=question,
                plan=plan,
                verifier_reason=verifier_reason,
                recent_steps=recent_steps,
                error_summary=error_summary,
                step_count=step_count,
                max_steps=max_steps,
                steps_remaining=max_steps - step_count,
            ),
        ),
    ]

    try:
        response = model.complete(messages)
        return _parse_router_response(response, max_steps - step_count)
    except Exception:
        return RouterDecision(
            action="add_step",
            reasoning="LLM router failed; defaulting to add_step",
            confidence=0.3,
        )


def _parse_router_response(response: str, steps_remaining: int) -> RouterDecision:
    """Parse the LLM router response into a decision."""
    response = response.strip()
    upper = response.upper()

    if upper.startswith("ADD_STEP"):
        reason = response.split(":", 1)[1].strip() if ":" in response else ""
        return RouterDecision(action="add_step", reasoning=reason, confidence=0.7)

    if upper.startswith("ANSWER"):
        reason = response.split(":", 1)[1].strip() if ":" in response else "budget critical"
        return RouterDecision(action="answer_now", reasoning=reason, confidence=0.8)

    # Parse FIX_STEP <K>
    import re
    fix_match = re.match(r"FIX_STEP\s+(\d+)", upper)
    if fix_match:
        k = int(fix_match.group(1))
        reason = response.split(":", 1)[1].strip() if ":" in response else ""
        return RouterDecision(action="fix_step", target_step=k, reasoning=reason, confidence=0.7)

    # Fallback
    return RouterDecision(
        action="add_step",
        reasoning=f"unparseable router response: {response[:100]}",
        confidence=0.3,
    )


def format_steps_for_router(steps: list[dict[str, Any]], limit: int = 3) -> str:
    """Format recent steps for the router prompt."""
    lines = []
    for s in steps[-limit:]:
        action = s.get("action", "?")
        ok = "OK" if s.get("ok") else "FAIL"
        obs = s.get("observation", {})
        content = obs.get("content", {}) if isinstance(obs, dict) else {}
        snippet = _summarize_content(content)
        lines.append(f"  [{ok}] {action}: {snippet}")
    return "\n".join(lines) if lines else "(no steps)"


def _summarize_content(content: dict[str, Any], max_len: int = 150) -> str:
    """Create a compact summary of tool output content."""
    if not isinstance(content, dict):
        return str(content)[:max_len]

    for key in ("summary", "error", "output", "result"):
        if key in content and content[key]:
            return str(content[key])[:max_len]

    for key in ("rows", "columns", "matches"):
        if key in content:
            val = content[key]
            if isinstance(val, list):
                return f"{len(val)} {key}"
            return str(val)[:max_len]

    return str(list(content.keys()))[:max_len]
