"""General failure taxonomy and semantic-commitment logging.

This module is intentionally observational: it does not change predictions.
It turns each task trace into reusable evolution signals without encoding
public-set task ids, fixed table names, or fixed answer SQL.
"""

from __future__ import annotations

import csv
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from data_agent_baseline.run.semantic_verifier import (
    semantic_report_to_dict,
    verify_semantic_contract,
)
from data_agent_baseline.run.verification_chain import VerificationReport


@dataclass(slots=True)
class FailureSignal:
    code: str
    severity: str
    detail: str


@dataclass(slots=True)
class SemanticCommitments:
    question_intents: list[str]
    predicted_columns: list[str]
    data_row_count: int
    tool_counts: dict[str, int]
    route: str | None
    answer_present: bool
    budget_exhausted: bool


@dataclass(slots=True)
class FailureAnalysis:
    task_id: str
    signals: list[FailureSignal]
    commitments: SemanticCommitments
    evolution_notes: list[str]
    semantic_verification: dict[str, Any]


NUMERIC_RE = re.compile(r"[-+]?\d+(?:,\d{3})*(?:\.\d+)?")


def _read_prediction_rows(path: Path | None) -> list[list[str]]:
    if path is None or not path.exists():
        return []
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            return list(csv.reader(handle))
    except (OSError, UnicodeDecodeError, csv.Error):
        return []


def _question_intents(question: str) -> list[str]:
    q = question.lower()
    intents: list[str] = []
    intent_patterns = {
        "count": r"\b(count|how many|number of)\b",
        "sum_total": r"\b(sum|total|overall|combined)\b",
        "average": r"\b(average|avg|mean)\b",
        "ratio_percentage": r"\b(percent|percentage|ratio|how many times|faster|slower)\b",
        "rank_order": r"\b(rank|ranked|top|bottom|position|finish|place|order)\b",
        "grouping": r"\b(by|per|for each|grouped|each)\b",
        "entity_identity": r"\b(id|identifier|name|full name|member|person|patient|customer)\b",
        "field_attachment": r"\b(type|category|status|approved|valid|confirmed)\b",
        "temporal": r"\b(year|month|monthly|annual|quarter|date|time)\b",
        "document_metric": r"\b(according to|document|text|note|record|reported|corrected|final)\b",
    }
    for intent, pattern in intent_patterns.items():
        if re.search(pattern, q):
            intents.append(intent)
    return intents


def _prediction_stats(rows: list[list[str]]) -> tuple[list[str], int, int, int]:
    if not rows:
        return [], 0, 0, 0
    header = [cell.strip() for cell in rows[0]]
    data_rows = rows[1:]
    cells = [cell.strip() for row in data_rows for cell in row if cell.strip()]
    numeric_cells = sum(1 for cell in cells if NUMERIC_RE.fullmatch(cell.replace("%", "")))
    return header, len(data_rows), len(cells), numeric_cells


def _tool_counts(run_result: dict[str, Any]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for step in run_result.get("steps", []) or []:
        if isinstance(step, dict):
            action = str(step.get("action") or "unknown")
            counts[action] += 1
    return dict(sorted(counts.items()))


def _budget_exhausted(run_result: dict[str, Any]) -> bool:
    failure = str(run_result.get("failure_reason") or "").lower()
    if "max_steps" in failure or "maximum" in failure:
        return True
    for step in run_result.get("steps", []) or []:
        if isinstance(step, dict):
            thought = str(step.get("thought") or "").lower()
            observation = str(step.get("observation") or "").lower()
            if "answer now" in thought or "steps left" in observation:
                return True
    return False


def _route_from_result(run_result: dict[str, Any]) -> str | None:
    route = run_result.get("_route_decision")
    if isinstance(route, dict):
        value = route.get("route")
        return str(value) if value else None
    return None


def _verification_signals(report: VerificationReport | None) -> list[FailureSignal]:
    if report is None:
        return [FailureSignal("verification_missing", "warning", "no post-run verification report was available")]
    signals: list[FailureSignal] = []
    for check in report.checks:
        if check.passed:
            continue
        code = {
            "readability_check": "output_readability_error",
            "contract_check": "output_contract_error",
            "task_contract_check": "task_contract_error",
            "sanity_check": "output_sanity_error",
            "shape_check": "output_shape_error",
        }.get(check.name, "verification_error")
        signals.append(FailureSignal(code, "error", check.detail))
    return signals


def analyze_task_failure(
    *,
    task_id: str,
    question: str,
    run_result: dict[str, Any],
    prediction_path: Path | None,
    verification_report: VerificationReport | None,
) -> FailureAnalysis:
    rows = _read_prediction_rows(prediction_path)
    header, data_row_count, non_empty_cells, numeric_cells = _prediction_stats(rows)
    intents = _question_intents(question)
    answer_present = bool(run_result.get("answer")) and data_row_count > 0
    budget_exhausted = _budget_exhausted(run_result)

    signals = _verification_signals(verification_report)
    failure_reason = str(run_result.get("failure_reason") or "").strip()
    lower_reason = failure_reason.lower()
    semantic_report = verify_semantic_contract(
        question=question,
        prediction_path=prediction_path,
        run_result=run_result,
    )
    semantic_report_dict = semantic_report_to_dict(semantic_report)
    if lower_reason:
        if "timeout" in lower_reason:
            signals.append(FailureSignal("runtime_timeout", "error", failure_reason))
        elif "max_steps" in lower_reason:
            signals.append(FailureSignal("budget_exhaustion", "error", failure_reason))
        else:
            signals.append(FailureSignal("runner_failure", "error", failure_reason))

    if not answer_present:
        signals.append(FailureSignal("no_grounded_answer", "error", "prediction has no non-empty data rows"))

    if budget_exhausted:
        signals.append(FailureSignal("budget_exhaustion", "warning", "trace indicates the agent exhausted or nearly exhausted its step budget"))

    if any(intent in intents for intent in ("count", "sum_total", "average", "ratio_percentage")):
        if non_empty_cells and numeric_cells == 0:
            signals.append(FailureSignal("numeric_grounding_error", "warning", "numeric intent detected but prediction contains no numeric cells"))

    # These semantic codes are always info-level with no actionable follow-up;
    # suppressing them keeps the signal taxonomy focused on actionable risks.
    _INFO_ONLY_SEMANTIC_CODES = {"rank_semantics_risk", "entity_identity_risk"}

    existing_codes = {signal.code for signal in signals}
    for check in semantic_report.checks:
        if check.code == "semantic_contract" or check.code in existing_codes:
            continue
        if check.code in _INFO_ONLY_SEMANTIC_CODES and check.severity == "info":
            continue
        signals.append(FailureSignal(check.code, check.severity, check.detail))
        existing_codes.add(check.code)

    if not signals:
        signals.append(FailureSignal("passed_no_obvious_risk", "info", "verification passed and no high-risk semantic signal was detected"))

    notes = _build_evolution_notes(signals)
    commitments = SemanticCommitments(
        question_intents=intents,
        predicted_columns=header,
        data_row_count=data_row_count,
        tool_counts=_tool_counts(run_result),
        route=_route_from_result(run_result),
        answer_present=answer_present,
        budget_exhausted=budget_exhausted,
    )
    return FailureAnalysis(
        task_id=task_id,
        signals=signals,
        commitments=commitments,
        evolution_notes=notes,
        semantic_verification=semantic_report_dict,
    )


def _build_evolution_notes(signals: list[FailureSignal]) -> list[str]:
    codes = {signal.code for signal in signals}
    notes: list[str] = []
    if "budget_exhaustion" in codes:
        notes.append("Improve route-specific early-exit and retry prompts instead of raising all task budgets uniformly.")
    if "numeric_grounding_error" in codes:
        notes.append("Add a generic numeric evidence verifier that requires formula, filters, and source columns before final answer.")
    if "denominator_risk" in codes:
        notes.append("Log numerator/denominator commitments for percentage tasks and compare alternative denominator choices during retry.")
    if "rank_semantics_risk" in codes:
        notes.append("Use schema-driven rank/position disambiguation based on column definitions and value previews.")
    if "field_attachment_risk" in codes:
        notes.append("Add a generic modifier-attachment check for descriptive entity fields versus filtered fact rows.")
    if "entity_identity_risk" in codes:
        notes.append("Prefer schema-derived identity columns and avoid hard-coded name concatenation/splitting unless source schema supports it.")
    if "long_doc_metric_risk" in codes:
        notes.append("Keep document extraction metric-local; evaluate extracted entity count and broaden terms only when coverage is suspiciously low.")
    if "target_noun_count_risk" in codes:
        notes.append("Add a target-noun count verifier that compares COUNT(*) with distinct entity counts when joins or repeated rows can duplicate the target.")
    if "threshold_grounding_risk" in codes:
        notes.append("Add threshold grounding evidence: locate context rules first, then compare explicit data-derived candidate cutoffs instead of using outside defaults.")
    if "output_shape_semantic_risk" in codes:
        notes.append("Add output-shape contract inference from requested fields and prevent packing multiple requested fields into one cell.")
    if "target_field_semantic_risk" in codes:
        notes.append("Add target-field verification so answers return the requested field (for example Text/URL/name) instead of nearby ids or evidence fields.")
    if "scalar_format_semantic_risk" in codes:
        notes.append("Normalize scalar answer formatting, especially percent signs and delimited values that evaluators expect as plain cells.")
    if "population_filter_semantic_risk" in codes:
        notes.append("Verify population scope before aggregation; avoid extra null/positive-value filtering when the question asks for all records.")
    if not notes:
        notes.append("No new evolution action suggested by deterministic taxonomy.")
    return notes


def failure_analysis_to_dict(analysis: FailureAnalysis) -> dict[str, Any]:
    return {
        "task_id": analysis.task_id,
        "signals": [
            {"code": signal.code, "severity": signal.severity, "detail": signal.detail}
            for signal in analysis.signals
        ],
        "commitments": {
            "question_intents": analysis.commitments.question_intents,
            "predicted_columns": analysis.commitments.predicted_columns,
            "data_row_count": analysis.commitments.data_row_count,
            "tool_counts": analysis.commitments.tool_counts,
            "route": analysis.commitments.route,
            "answer_present": analysis.commitments.answer_present,
            "budget_exhausted": analysis.commitments.budget_exhausted,
        },
        "evolution_notes": analysis.evolution_notes,
        "semantic_verification": analysis.semantic_verification,
    }
