"""Critic: lightweight step-level sanity checker.

Runs AFTER each tool execution to detect obvious errors before they
propagate into the agent's next step. Operates in two tiers:

Tier 1 (rule-based, zero cost): numeric range, empty outputs, error text
Tier 2 (LLM-based, triggered only on suspicion): semantic contradiction check

Not a separate agent — just a function that looks at the observation and
returns pass/fail with a one-sentence reason.
"""

from __future__ import annotations

import re
from typing import Any

# ── Tier 1: Rule-based checks ─────────────────────────────────────

# Known reasonable ranges for common metrics (domain knowledge)
REASONABLE_RANGES: dict[str, tuple[float, float]] = {
    "gpa": (0.0, 4.0),
    "grade": (0.0, 100.0),
    "age": (0.0, 120.0),
    "salary": (0.0, 10_000_000.0),
    "price": (0.0, 1_000_000.0),
    "score": (0.0, 100.0),
    "rate": (0.0, 1.0),
    "percentage": (0.0, 100.0),
    "count": (0.0, 1_000_000_000.0),
}


def _extract_numeric_values(content: dict[str, Any]) -> list[tuple[str, float]]:
    """Extract all numeric values with context from tool output."""
    results: list[tuple[str, float]] = []
    for key, value in content.items():
        if isinstance(value, (int, float)):
            results.append((key, float(value)))
        elif isinstance(value, str):
            try:
                results.append((key, float(value)))
            except (ValueError, TypeError):
                pass
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    for k, v in item.items():
                        if isinstance(v, (int, float)):
                            results.append((f"{key}.{k}", float(v)))
    return results


def _guess_domain_from_question(question: str) -> str:
    """Guess what kind of metric is being computed."""
    q = question.lower()
    for domain in REASONABLE_RANGES:
        if domain in q:
            return domain
    return ""


def _check_range(
    values: list[tuple[str, float]], question: str,
) -> list[str]:
    """Check if any values fall outside reasonable ranges."""
    warnings: list[str] = []
    for name, val in values:
        name_lower = name.lower()
        # Skip file metadata fields — their values are not domain metrics
        if "size_bytes" in name_lower or "bytes" in name_lower:
            continue
        for d, (lo, hi) in REASONABLE_RANGES.items():
            if d in name_lower or d in question.lower():
                if val < lo or val > hi:
                    warnings.append(
                        f"Value {name}={val} is outside "
                        f"reasonable range for '{d}' [{lo}, {hi}]"
                    )
                break
    return warnings


def _check_empty(content: dict[str, Any]) -> str | None:
    """Check if tool output is suspiciously empty."""
    # CSV/SQL results
    rows = content.get("rows", content.get("matches", content.get("result", None)))
    if isinstance(rows, list) and len(rows) == 0:
        return "Tool returned empty results. Verify the query/filter is correct."
    # Python execution
    output = content.get("output", "")
    if isinstance(output, str) and output.strip() == "":
        stderr = content.get("stderr", "")
        if stderr:
            return f"Python stdout is empty but stderr says: {str(stderr)[:200]}"
    return None


def _check_error_text(content: dict[str, Any]) -> str | None:
    """Check if tool output contains error-like text."""
    error_patterns = [
        r"traceback", r"error:", r"exception", r"permission denied",
        r"cannot", r"failed", r"invalid", r"not found",
    ]
    for key in ("output", "stderr", "error", "result"):
        text = str(content.get(key, "")).lower()
        for pat in error_patterns:
            if re.search(pat, text):
                return f"Tool output contains '{pat}' — may indicate an error: {text[:150]}"
    return None


# ── Tier 2: LLM-based contradiction check ─────────────────────────

CRITIC_PROMPT = """You are a fact-checker reviewing a single step of a data analysis. Your job is to catch OBVIOUS errors.

Previous step result: {previous_summary}
Current step result: {current_summary}
Question being answered: {question}

Check ONLY for clear contradictions or impossibilities:
- Does the current result directly contradict the previous step?
- Is a numeric value impossibly large/small for its context?
- Does the agent appear to be looking at the wrong column/table?

Reply with EXACTLY one of:
- "PASS" if nothing is clearly wrong
- "FAIL: <one sentence explaining the specific error>"

Do not overthink. Only flag OBVIOUS problems. If uncertain, say PASS."""


def critic_check(
    *,
    current_observation: dict[str, Any],
    question: str,
    previous_outputs: list[str] | None = None,
) -> dict[str, Any]:
    """Run tier-1 (rule) checks. Returns pass/fail verdict + warnings.

    Tier-2 (LLM) is invoked only if tier-1 finds something suspicious
    AND a model adapter is provided.
    """
    content = current_observation.get("content", {})
    if not isinstance(content, dict):
        return {"pass": True, "warnings": [], "tier": 0}

    warnings: list[str] = []

    # Rule checks
    empty_warning = _check_empty(content)
    if empty_warning:
        warnings.append(empty_warning)

    error_warning = _check_error_text(content)
    if error_warning:
        warnings.append(error_warning)

    values = _extract_numeric_values(content)
    range_warnings = _check_range(values, question)
    warnings.extend(range_warnings)

    return {
        "pass": len(warnings) == 0,
        "warnings": warnings,
        "tier": 1,
        "suspicious": len(warnings) >= 2,  # flag for potential tier-2 escalation
    }


def critic_llm_check(
    *,
    model,
    current_observation: dict[str, Any],
    question: str,
    previous_summary: str = "",
) -> str:
    """Tier-2: LLM-based contradiction check. Only call when tier-1 flags suspicion."""
    content = current_observation.get("content", {})
    current_summary = str(content)[:500]

    messages = [
        {"role": "system", "content": CRITIC_PROMPT.format(
            previous_summary=previous_summary or "(first step)",
            current_summary=current_summary,
            question=question,
        )},
    ]
    try:
        response = model.complete(messages)  # type: ignore[arg-type]
        return response.strip()
    except Exception:
        return "PASS"  # If critic fails, don't block execution
