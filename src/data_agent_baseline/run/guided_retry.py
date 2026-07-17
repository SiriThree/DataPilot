"""Guided retry planning and deterministic attempt selection."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any


SEMANTIC_RISK_CODES = {
    "denominator_risk",
    "rank_semantics_risk",
    "field_attachment_risk",
    "temporal_aggregation_risk",
    "entity_identity_risk",
    "long_doc_metric_risk",
    "numeric_grounding_error",
    "threshold_grounding_risk",
    "output_shape_semantic_risk",
}

RETRY_TRIGGER_CODES = SEMANTIC_RISK_CODES | {
    "budget_exhaustion",
    "no_grounded_answer",
    "output_contract_error",
    "output_sanity_error",
    "output_shape_error",
    "task_contract_error",
    "output_readability_error",
}

HARD_ERROR_CODES = {
    "budget_exhaustion",
    "no_grounded_answer",
    "output_contract_error",
    "output_sanity_error",
    "output_shape_error",
    "task_contract_error",
    "output_readability_error",
    "runtime_timeout",
    "runner_failure",
}


@dataclass(frozen=True, slots=True)
class GuidedRetryDecision:
    should_retry: bool
    reason: str
    trigger_codes: list[str]
    retry_hint: str


@dataclass(frozen=True, slots=True)
class AttemptEvaluation:
    name: str
    answer_present: bool
    hard_error_count: int
    semantic_risk_count: int
    warning_count: int
    data_row_count: int
    suspicious_output_penalty: int
    row_count_penalty: int
    penalty_tuple: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class AttemptSelection:
    selected_attempt: str
    rationale: str
    original: AttemptEvaluation
    retry: AttemptEvaluation


def _failure_analysis(trace: dict[str, Any]) -> dict[str, Any]:
    value = trace.get("_failure_analysis")
    return value if isinstance(value, dict) else {}


def _signals(trace: dict[str, Any]) -> list[dict[str, str]]:
    raw = _failure_analysis(trace).get("signals", [])
    if not isinstance(raw, list):
        return []
    return [item for item in raw if isinstance(item, dict)]


def _signal_codes(trace: dict[str, Any]) -> list[str]:
    codes: list[str] = []
    seen: set[str] = set()
    for signal in _signals(trace):
        code = str(signal.get("code", "")).strip()
        if code and code not in seen:
            seen.add(code)
            codes.append(code)
    return codes


def _retry_trigger_codes(trace: dict[str, Any]) -> list[str]:
    codes: list[str] = []
    seen: set[str] = set()
    for signal in _signals(trace):
        code = str(signal.get("code", "")).strip()
        severity = str(signal.get("severity", "")).strip()
        if code not in RETRY_TRIGGER_CODES or code in seen:
            continue
        if code in SEMANTIC_RISK_CODES and severity == "info":
            continue
        seen.add(code)
        codes.append(code)
    return codes


def _signal_details(trace: dict[str, Any], trigger_codes: set[str]) -> list[str]:
    details: list[str] = []
    for signal in _signals(trace):
        code = str(signal.get("code", "")).strip()
        if code in trigger_codes:
            detail = str(signal.get("detail", "")).strip()
            details.append(f"{code}: {detail}" if detail else code)
    return details


def _severity_count(trace: dict[str, Any], severity: str) -> int:
    return sum(1 for signal in _signals(trace) if signal.get("severity") == severity)


def _commitments(trace: dict[str, Any]) -> dict[str, Any]:
    raw = _failure_analysis(trace).get("commitments", {})
    return raw if isinstance(raw, dict) else {}


def _predicted_columns(trace: dict[str, Any]) -> list[str]:
    raw = _commitments(trace).get("predicted_columns", [])
    if not isinstance(raw, list):
        return []
    return [str(column).strip() for column in raw if str(column).strip()]


def _prediction_preview(path: Path | None, *, max_rows: int = 4) -> str:
    if path is None or not path.exists():
        return "<missing prediction.csv>"
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.reader(handle))
    except (OSError, UnicodeDecodeError, csv.Error) as exc:
        return f"<prediction.csv unreadable: {exc}>"
    rendered = []
    for row in rows[:max_rows]:
        rendered.append(",".join(str(cell) for cell in row))
    if len(rows) > max_rows:
        rendered.append(f"... <{len(rows) - max_rows} more row(s)>")
    return "\n".join(rendered) if rendered else "<empty prediction.csv>"


def _last_actions(trace: dict[str, Any], *, max_items: int = 6) -> list[str]:
    steps = trace.get("steps", [])
    if not isinstance(steps, list):
        return []
    actions: list[str] = []
    for step in steps[-max_items:]:
        if isinstance(step, dict):
            actions.append(str(step.get("action") or "unknown"))
    return actions


def _risk_protocol_lines(trigger_set: set[str]) -> list[str]:
    lines: list[str] = []
    if "denominator_risk" in trigger_set:
        lines.extend([
            "Denominator retry protocol:",
            "- Write the numerator and denominator in the thought field before any final calculation.",
            "- Run or inspect at least one query that separately computes numerator and denominator.",
            "- Use the denominator implied by the target population, not by a convenient joined subset.",
        ])
    if "temporal_aggregation_risk" in trigger_set:
        lines.extend([
            "Aggregation retry protocol:",
            "- Preserve the operator requested by the question first: AVG stays AVG, SUM stays SUM, COUNT stays COUNT.",
            "- Apply monthly/yearly normalization only after the base aggregate is chosen.",
            "- Compare AVG-based and SUM-based formulas if both seem plausible, then choose by wording.",
        ])
    if "threshold_grounding_risk" in trigger_set:
        lines.extend([
            "Threshold grounding protocol:",
            "- Call `ground_thresholds` with the target measurement terms and population terms before answering.",
            "- If `ground_thresholds` returns `recommended_probe_summary`, verify that candidate first.",
            "- Preserve the recommended population source and entity-level/same-row mode when reproducing the count.",
            "- Search context documents and schema notes for the exact normal/abnormal/range definition first.",
            "- If no explicit rule exists, compare observed candidate probes instead of guessing a default threshold.",
            "- Do not override missing context with memorized clinical, financial, or domain thresholds.",
            "- Check whether the wording asks about an entity satisfying conditions across rows; if so, prefer entity-level distinct counts over same-row counts.",
            "- Compare population source coverage when both helper CSVs and narrative docs define the filtered population.",
            "- Do not use outside default thresholds unless the task context explicitly supports them.",
        ])
    if "output_shape_semantic_risk" in trigger_set:
        lines.extend([
            "Output-shape retry protocol:",
            "- Infer the requested fields from the question and keep them as separate columns.",
            "- Do not pack multiple requested values into a single joined string cell.",
            "- Before answering, verify column count and row count against the requested granularity.",
        ])
    if "field_attachment_risk" in trigger_set:
        lines.extend([
            "Modifier-attachment retry protocol:",
            "- Decide which noun each modifier applies to before joining or grouping.",
            "- Keep descriptive entity fields separate from filtered fact rows.",
            "- Add grouping only when the question explicitly asks by/per/for each/list all.",
        ])
    if "entity_identity_risk" in trigger_set:
        lines.extend([
            "Identity retry protocol:",
            "- Inspect source schema for canonical identity/name columns and join keys.",
            "- Do not invent display names by concatenating or splitting unless the source schema supports that shape.",
            "- Verify the answer's identity value comes from the same entity as the requested metric.",
        ])
    if "rank_semantics_risk" in trigger_set:
        lines.extend([
            "Rank/order retry protocol:",
            "- Inspect relevant column names and value previews before choosing rank, position, order, or time fields.",
            "- Distinguish recorded rank/order fields from derived sorting by another metric.",
        ])
    if "long_doc_metric_risk" in trigger_set:
        lines.extend([
            "Document-metric retry protocol:",
            "- Extract only the requested metric and entity id; ignore nearby unrelated metrics.",
            "- Prefer corrected/final/confirmed values when text states multiple values.",
            "- If extraction coverage is suspiciously small, broaden target terms once, then answer.",
        ])
    if "numeric_grounding_error" in trigger_set:
        lines.extend([
            "Numeric grounding retry protocol:",
            "- Produce a concrete formula/query and inspect compact numeric output before answering.",
            "- If the final answer is numeric, ensure at least one numeric source value or aggregate is observed.",
        ])
    return lines


def build_guided_retry_decision(
    *,
    original_trace: dict[str, Any],
    prediction_path: Path | None,
) -> GuidedRetryDecision:
    trigger_codes = _retry_trigger_codes(original_trace)
    if not trigger_codes:
        return GuidedRetryDecision(
            should_retry=False,
            reason="no retry trigger codes detected",
            trigger_codes=[],
            retry_hint="",
        )

    trigger_set = set(trigger_codes)

    # Idempotency check: if the model is stuck in a loop of repeated
    # scout/doc actions, a retry will produce the same result.
    last_actions = _last_actions(original_trace)
    if last_actions and (
        all(a == "search_doc" for a in last_actions)
        or all(a == "read_csv" for a in last_actions)
    ):
        return GuidedRetryDecision(
            should_retry=False,
            reason=(
                f"last {len(last_actions)} actions are all identical scout/doc calls "
                f"({last_actions[0]}); model appears stuck — retry would repeat the same pattern"
            ),
            trigger_codes=trigger_codes,
            retry_hint="",
        )

    parts: list[str] = [
        "Guided retry mode: the previous attempt was not trusted.",
        f"Trigger codes: {', '.join(trigger_codes)}.",
        "Previous prediction preview:",
        _prediction_preview(prediction_path),
    ]
    if last_actions:
        parts.append(f"Last attempted actions: {', '.join(last_actions)}.")
    details = _signal_details(original_trace, trigger_set)
    if details:
        parts.append("Failure/risk details:")
        parts.extend(f"- {detail}" for detail in details[:8])

    if "budget_exhaustion" in trigger_set or "no_grounded_answer" in trigger_set:
        parts.extend([
            "Budget recovery protocol:",
            "- Use at most 2 scouting/profile steps.",
            "- Do not repeat failed searches or broad document scans from the previous attempt.",
            "- Produce a candidate formula/query by the middle of the retry.",
            "- If evidence is incomplete near the end, submit the best grounded answer instead of leaving prediction empty.",
        ])

    semantic_triggers = trigger_set & SEMANTIC_RISK_CODES
    if semantic_triggers:
        parts.extend([
            "Semantic risk protocol:",
            "- Before answering, explicitly verify target noun/entity, filters, join keys, aggregation operator, output granularity, and source columns in the thought field.",
        ])
        parts.extend(_risk_protocol_lines(trigger_set))

    parts.append("Final requirement: call `answer` with a compact CSV-shaped table.")
    return GuidedRetryDecision(
        should_retry=True,
        reason="retry trigger codes detected",
        trigger_codes=trigger_codes,
        retry_hint="\n".join(parts),
    )


def evaluate_attempt(name: str, trace: dict[str, Any]) -> AttemptEvaluation:
    codes = _signal_codes(trace)
    commitments = _commitments(trace)
    answer_present = bool(commitments.get("answer_present"))
    hard_error_count = sum(1 for code in codes if code in HARD_ERROR_CODES)
    semantic_risk_count = sum(1 for code in codes if code in SEMANTIC_RISK_CODES)
    warning_count = _severity_count(trace, "warning")
    data_row_count = int(commitments.get("data_row_count") or 0)
    no_answer_penalty = 1 if not answer_present else 0
    predicted_columns = [column.lower() for column in _predicted_columns(trace)]
    suspicious_output_penalty = 0
    if predicted_columns == ["answer"] and data_row_count <= 1:
        suspicious_output_penalty += 1
    if any(code in {"no_grounded_answer", "output_shape_error"} for code in codes):
        suspicious_output_penalty += 1
    row_count_penalty = 1 if answer_present and data_row_count == 0 else 0
    penalty_tuple = (
        no_answer_penalty,
        hard_error_count,
        warning_count,
        semantic_risk_count,
        suspicious_output_penalty,
        row_count_penalty,
    )
    return AttemptEvaluation(
        name=name,
        answer_present=answer_present,
        hard_error_count=hard_error_count,
        semantic_risk_count=semantic_risk_count,
        warning_count=warning_count,
        data_row_count=data_row_count,
        suspicious_output_penalty=suspicious_output_penalty,
        row_count_penalty=row_count_penalty,
        penalty_tuple=penalty_tuple,
    )


def _row_count_regressed(original: AttemptEvaluation, retry: AttemptEvaluation) -> bool:
    if original.data_row_count <= 0 or retry.data_row_count <= 0:
        return False
    if retry.data_row_count <= original.data_row_count:
        return False
    if retry.hard_error_count != original.hard_error_count:
        return False
    if retry.warning_count != original.warning_count:
        return False
    if retry.semantic_risk_count != original.semantic_risk_count:
        return False
    inflated_by_ratio = retry.data_row_count > int(original.data_row_count * 1.25)
    inflated_by_absolute = retry.data_row_count - original.data_row_count > 10
    return inflated_by_ratio and inflated_by_absolute


def select_best_attempt(
    *,
    original_trace: dict[str, Any],
    retry_trace: dict[str, Any],
    trigger_codes: list[str],
) -> AttemptSelection:
    original_eval = evaluate_attempt("original", original_trace)
    retry_eval = evaluate_attempt("retry", retry_trace)
    trigger_set = set(trigger_codes)

    selected = "original"
    rationale = (
        "kept original because retry did not improve deterministic attempt penalties "
        f"{retry_eval.penalty_tuple} >= {original_eval.penalty_tuple}"
    )
    if retry_eval.answer_present and _row_count_regressed(original_eval, retry_eval):
        rationale = (
            "kept original because retry only increased row count without reducing hard errors, "
            "warnings, or semantic risks "
            f"({original_eval.data_row_count} -> {retry_eval.data_row_count})"
        )
    elif retry_eval.answer_present and retry_eval.penalty_tuple < original_eval.penalty_tuple:
        selected = "retry"
        rationale = (
            "selected retry because deterministic attempt penalties improved "
            f"{original_eval.penalty_tuple} -> {retry_eval.penalty_tuple}"
        )
    elif (
        retry_eval.answer_present
        and trigger_set & SEMANTIC_RISK_CODES
        and retry_eval.hard_error_count <= original_eval.hard_error_count
        and (
            retry_eval.warning_count < original_eval.warning_count
            or retry_eval.semantic_risk_count < original_eval.semantic_risk_count
        )
    ):
        selected = "retry"
        rationale = (
            "selected retry because semantic-risk retry reduced warning or semantic-risk counts "
            "without increasing hard verification errors"
        )

    return AttemptSelection(
        selected_attempt=selected,
        rationale=rationale,
        original=original_eval,
        retry=retry_eval,
    )


def guided_retry_decision_to_dict(decision: GuidedRetryDecision) -> dict[str, Any]:
    return {
        "should_retry": decision.should_retry,
        "reason": decision.reason,
        "trigger_codes": decision.trigger_codes,
        "retry_hint": decision.retry_hint,
    }


def attempt_evaluation_to_dict(evaluation: AttemptEvaluation) -> dict[str, Any]:
    return {
        "name": evaluation.name,
        "answer_present": evaluation.answer_present,
        "hard_error_count": evaluation.hard_error_count,
        "semantic_risk_count": evaluation.semantic_risk_count,
        "warning_count": evaluation.warning_count,
        "data_row_count": evaluation.data_row_count,
        "suspicious_output_penalty": evaluation.suspicious_output_penalty,
        "row_count_penalty": evaluation.row_count_penalty,
        "penalty_tuple": list(evaluation.penalty_tuple),
    }


def attempt_selection_to_dict(selection: AttemptSelection) -> dict[str, Any]:
    return {
        "selected_attempt": selection.selected_attempt,
        "rationale": selection.rationale,
        "original": attempt_evaluation_to_dict(selection.original),
        "retry": attempt_evaluation_to_dict(selection.retry),
    }
