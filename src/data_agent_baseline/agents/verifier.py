"""LLM-based verifier: judges whether current execution sufficiently answers the question.

Modeled after DS-STAR's A_verifier. Runs AFTER each tool execution to determine
if the accumulated evidence is sufficient or if more work is needed.

Unlike the old critic (rule-based only), this verifier:
1. Evaluates semantic alignment: do results answer what was asked?
2. Checks completeness: are all required columns/metrics present?
3. Detects contradictions: do current results conflict with previous steps?
4. Returns structured verdict: sufficient/insufficient + reasoning
"""

from __future__ import annotations

from typing import Any

VERIFIER_SYSTEM_PROMPT = """You are a rigorous data analysis verifier. Your job is to judge whether the current execution results SUFFICIENTLY answer the user's question.

Evaluate against these criteria:
1. COMPLETENESS: Are all requested columns/metrics present in the result?
2. CORRECTNESS: Do the numbers/values make sense given the question and data context?
3. GROUNDING: Is every value traceable to data sources (not invented)?
4. FORMAT: Does the result shape (columns, rows) match what the question asks for?

Reply with EXACTLY one verdict and a one-sentence reason:
- "SUFFICIENT: <reason>" — the answer is complete and correct
- "INSUFFICIENT: <reason>" — more work is needed

Be strict: if ANY criterion fails, return INSUFFICIENT."""

VERIFIER_USER_TEMPLATE = """Question: {question}

Plan for this task:
{plan}

Latest execution result:
{latest_result}

Key evidence accumulated so far:
{evidence_summary}

Steps taken: {step_count}
Steps remaining: {steps_remaining}

Judge: SUFFICIENT or INSUFFICIENT?"""


def build_verdict(verdict: str, reason: str) -> dict[str, Any]:
    """Build a structured verdict from the LLM response."""
    return {
        "sufficient": verdict.upper().startswith("SUFFICIENT"),
        "reason": reason,
        "raw": f"{verdict}: {reason}",
    }


def parse_verifier_response(response: str) -> dict[str, Any]:
    """Parse the LLM's verifier response into structured verdict."""
    response = response.strip()
    for prefix in ("SUFFICIENT:", "SUFFICIENT", "INSUFFICIENT:", "INSUFFICIENT"):
        if response.upper().startswith(prefix):
            verdict = "SUFFICIENT" if prefix.upper().startswith("SUFFICIENT") else "INSUFFICIENT"
            reason = response[len(prefix):].strip().lstrip(":").strip()
            return build_verdict(verdict, reason if reason else "no reason provided")

    # Fallback: check keywords
    upper = response.upper()
    if "SUFFICIENT" in upper and "INSUFFICIENT" not in upper:
        return build_verdict("SUFFICIENT", response)
    if "INSUFFICIENT" in upper:
        return build_verdict("INSUFFICIENT", response)
    return build_verdict("INSUFFICIENT", f"unparseable: {response[:200]}")


def run_llm_verifier(
    *,
    model,
    question: str,
    plan: str,
    latest_result: str,
    evidence_summary: str,
    step_count: int,
    steps_remaining: int,
) -> dict[str, Any]:
    """Run the LLM-based verifier to judge execution sufficiency.

    Returns:
        dict with keys: sufficient (bool), reason (str), raw (str)
    """
    from data_agent_baseline.agents.model import ModelMessage

    messages = [
        ModelMessage(role="system", content=VERIFIER_SYSTEM_PROMPT),
        ModelMessage(
            role="user",
            content=VERIFIER_USER_TEMPLATE.format(
                question=question,
                plan=plan,
                latest_result=latest_result,
                evidence_summary=evidence_summary,
                step_count=step_count,
                steps_remaining=steps_remaining,
            ),
        ),
    ]

    try:
        response = model.complete(messages)
        return parse_verifier_response(response)
    except Exception:
        return build_verdict("INSUFFICIENT", "verifier call failed; assume insufficient")


def build_evidence_summary(
    steps: list[dict[str, Any]],
    max_chars: int = 1500,
) -> str:
    """Build a compact summary of evidence accumulated across steps."""
    parts: list[str] = []

    for step in steps:
        action = step.get("action", "unknown")
        obs = step.get("observation", {})

        if action == "__step_controller__" or action == "__error__":
            continue

        content = obs.get("content", {}) if isinstance(obs, dict) else {}
        if not isinstance(content, dict):
            continue

        # Extract key evidence
        for key in ("columns", "column_names", "schema", "files"):
            if key in content:
                val = content[key]
                if isinstance(val, dict):
                    parts.append(f"[{action}] {key}: {list(val.keys())[:20]}")
                elif isinstance(val, list):
                    parts.append(f"[{action}] {key}: {len(val)} items")
                else:
                    parts.append(f"[{action}] {key}: {str(val)[:200]}")

        for key in ("rows", "result", "output", "preview"):
            if key in content:
                val = content[key]
                if isinstance(val, list) and val:
                    parts.append(f"[{action}] {key}: {len(val)} rows, first: {str(val[0])[:150]}")
                elif isinstance(val, str) and val:
                    parts.append(f"[{action}] {key}: {val[:200]}")

        for key in ("summary", "error"):
            if key in content and content[key]:
                parts.append(f"[{action}] {key}: {str(content[key])[:200]}")

    joined = "\n".join(parts)
    if len(joined) > max_chars:
        return joined[:max_chars] + f"\n... <{len(joined) - max_chars} more chars truncated>"
    return joined or "(no evidence yet)"


def should_skip_verifier(
    *,
    step_index: int,
    action: str,
    max_steps: int,
) -> bool:
    """Determine if we should skip the verifier for this step.

    Skips verifier on:
    - Very early scouting steps (step 1-2, before meaningful data is available)
    - Step controller blocks (not real tool executions)
    - The answer action itself (already terminal)
    """
    if action == "__step_controller__":
        return True
    if action == "answer":
        return True
    return False


def should_force_verifier(
    *,
    step_index: int,
    max_steps: int,
    steps_remaining: int,
    action: str,
) -> bool:
    """Determine if we MUST run the verifier (even if normally skipped).

    Forces verification when:
    - Close to budget exhaustion (must decide whether to answer)
    - After a compute action (SQL/Python) that may have produced the answer
    """
    compute_actions = {"execute_context_sql", "execute_data_sql", "execute_python"}
    if steps_remaining <= 3:
        return True
    if action in compute_actions and step_index >= 3:
        return True
    return False
