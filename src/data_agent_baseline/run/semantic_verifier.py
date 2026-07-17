"""Question-level semantic contract inference and verification.

This module is deliberately domain-neutral. It does not know public task ids,
gold answers, or fixed schema names. It inspects the question, prediction
shape, and trace evidence to surface reusable semantic risks.
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


COUNT_RE = re.compile(r"\b(count|how many|number of)\b", re.IGNORECASE)
SUM_RE = re.compile(r"\b(sum|total|overall|combined)\b", re.IGNORECASE)
AVERAGE_RE = re.compile(r"\b(average|avg|mean)\b", re.IGNORECASE)
RATIO_RE = re.compile(r"\b(percent|percentage|ratio|how many times|faster|slower)\b", re.IGNORECASE)
RANK_RE = re.compile(r"\b(rank|ranked|top|bottom|position|finish|place|order)\b", re.IGNORECASE)
TEMPORAL_RE = re.compile(r"\b(year|month|monthly|annual|quarter|date|time)\b", re.IGNORECASE)
THRESHOLD_RE = re.compile(
    r"\b(normal|abnormal|threshold|range|level|above|below|higher than|lower than|at least|less than|greater than)\b",
    re.IGNORECASE,
)
GROUPING_RE = re.compile(r"\b(by|per|for each|grouped by|each|list all|for every)\b", re.IGNORECASE)
IDENTITY_RE = re.compile(r"\b(id|identifier|name|full name|member|person|patient|customer|user)\b", re.IGNORECASE)
DOCUMENT_RE = re.compile(
    r"\b(according to|document|text|note|record|reported|corrected|final|confirmed|described)\b",
    re.IGNORECASE,
)
FIELD_ATTACHMENT_RE = re.compile(r"\b(type|category|status|approved|valid|confirmed)\b", re.IGNORECASE)
VALUE_METRIC_RE = re.compile(r"\b(total|value|cost|expense|amount|score|views|consumption)\b", re.IGNORECASE)
DELIMITED_CELL_RE = re.compile(r"^\s*[^,]+(?:-|/|:)[^,]+\s*$")
URL_RE = re.compile(r"https?://|www\.", re.IGNORECASE)


@dataclass(frozen=True, slots=True)
class SemanticContract:
    intents: list[str]
    aggregation: str | None = None
    target_noun: str | None = None
    requested_fields: list[str] = field(default_factory=list)
    output_shape: str = "unknown"
    risk_codes: list[str] = field(default_factory=list)
    requirements: list[str] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class SemanticCheck:
    code: str
    severity: str
    detail: str


@dataclass(frozen=True, slots=True)
class SemanticVerificationReport:
    contract: SemanticContract
    checks: list[SemanticCheck]


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        key = item.strip().lower()
        if key and key not in seen:
            seen.add(key)
            out.append(item.strip())
    return out


def _clean_target_noun(raw: str) -> str:
    text = raw.lower().strip(" ?.,:;")
    text = re.sub(r"^(?:the|all|distinct|unique|number of|count of|of)\s+", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _extract_antecedent_target(question: str) -> str | None:
    match = re.search(
        r"\bamong\s+(?:the\s+)?(.+?)(?:\s+(?:who|that|which|with|where|have|has|are|is|were|was)\b|,)",
        question,
        flags=re.IGNORECASE,
    )
    if not match:
        return None
    target = _clean_target_noun(match.group(1))
    return target or None


def _extract_count_target(question: str) -> str | None:
    patterns = [
        r"\bhow many\s+(.+?)(?:\s+(?:who|that|which|with|where|have|has|are|is|were|was|from|in|for)\b|[?.,]|$)",
        r"\bnumber of\s+(.+?)(?:\s+(?:who|that|which|with|where|have|has|are|is|were|was|from|in|for)\b|[?.,]|$)",
        r"\bcount(?: of)?\s+(.+?)(?:\s+(?:who|that|which|with|where|have|has|are|is|were|was|from|in|for)\b|[?.,]|$)",
    ]
    for pattern in patterns:
        match = re.search(pattern, question, flags=re.IGNORECASE)
        if match:
            target = _clean_target_noun(match.group(1))
            if target in {"them", "those", "of them", "of those"}:
                target = _extract_antecedent_target(question) or target
            if target:
                return target
    return None


def _extract_requested_fields(question: str) -> list[str]:
    q = question.strip()
    patterns = [
        r"\b(?:report|return|provide|give|list|show|find)\s+(?:the\s+)?(.+?)(?:\s+(?:for|of|where|when|whose|that|with|among|in)\b|[?])",
        r"\bwhat\s+(?:are|is)\s+(?:the\s+)?(.+?)(?:\s+(?:for|of|where|when|whose|that|with|among|in)\b|[?])",
    ]
    for pattern in patterns:
        match = re.search(pattern, q, flags=re.IGNORECASE)
        if not match:
            continue
        span = match.group(1)
        if len(span) > 120:
            continue
        parts = re.split(r"\s*,\s*|\s+\band\b\s+", span, flags=re.IGNORECASE)
        fields = [
            re.sub(r"^(?:the|a|an)\s+", "", part.strip(" .,:;")).strip()
            for part in parts
        ]
        fields = [field for field in fields if field and not COUNT_RE.search(field)]
        if len(fields) >= 2:
            return _dedupe(fields)
    return []


def _aggregation(question: str) -> str | None:
    if COUNT_RE.search(question):
        return "count"
    if AVERAGE_RE.search(question):
        return "average"
    if RATIO_RE.search(question):
        return "ratio_percentage"
    if SUM_RE.search(question):
        return "sum_total"
    return None


def infer_semantic_contract(question: str) -> SemanticContract:
    intents: list[str] = []
    risk_codes: list[str] = []
    requirements: list[str] = []
    aggregation = _aggregation(question)
    target_noun = _extract_count_target(question) if aggregation == "count" else None
    requested_fields = _extract_requested_fields(question)

    checks = [
        ("count", COUNT_RE),
        ("sum_total", SUM_RE),
        ("average", AVERAGE_RE),
        ("ratio_percentage", RATIO_RE),
        ("rank_order", RANK_RE),
        ("temporal", TEMPORAL_RE),
        ("threshold", THRESHOLD_RE),
        ("grouping", GROUPING_RE),
        ("entity_identity", IDENTITY_RE),
        ("document_metric", DOCUMENT_RE),
    ]
    for name, pattern in checks:
        if pattern.search(question):
            intents.append(name)

    if aggregation in {"count", "sum_total", "average", "ratio_percentage"} and not requested_fields:
        output_shape = "scalar"
    elif requested_fields:
        output_shape = f"{len(requested_fields)} requested field(s)"
    else:
        output_shape = "unknown"

    if aggregation == "count" and target_noun and target_noun not in {"rows", "records", "entries"}:
        risk_codes.append("target_noun_count_risk")
        requirements.append("count the target noun/entity after filters; compare row count with distinct entity count when joins can duplicate rows")
    if RATIO_RE.search(question):
        risk_codes.append("denominator_risk")
        requirements.append("state numerator and denominator before computing percentage or ratio")
    if AVERAGE_RE.search(question) and TEMPORAL_RE.search(question):
        risk_codes.append("temporal_aggregation_risk")
        requirements.append("preserve AVG/mean semantics before month/year normalization")
    if RANK_RE.search(question):
        risk_codes.append("rank_semantics_risk")
        requirements.append("inspect field definitions/value previews before choosing rank, order, place, or time fields")
    if THRESHOLD_RE.search(question):
        risk_codes.append("threshold_grounding_risk")
        requirements.append("ground normal/abnormal/threshold/range filters in context evidence or explicit data-derived cutoffs")
    if requested_fields:
        risk_codes.append("output_shape_semantic_risk")
        requirements.append("answer fields separately when the question asks for multiple fields")
    if FIELD_ATTACHMENT_RE.search(question) and VALUE_METRIC_RE.search(question):
        risk_codes.append("field_attachment_risk")
        requirements.append("separate descriptive entity fields from filtered fact-table rows and aggregations")
    if IDENTITY_RE.search(question):
        risk_codes.append("entity_identity_risk")
        requirements.append("derive identity/name output from source schema rather than presentation assumptions")
    if DOCUMENT_RE.search(question):
        risk_codes.append("long_doc_metric_risk")
        requirements.append("extract only the target metric and prefer corrected/final/confirmed values when present")

    return SemanticContract(
        intents=_dedupe(intents),
        aggregation=aggregation,
        target_noun=target_noun,
        requested_fields=requested_fields,
        output_shape=output_shape,
        risk_codes=_dedupe(risk_codes),
        requirements=_dedupe(requirements),
    )


def _read_prediction_rows(path: Path | None) -> list[list[str]]:
    if path is None or not path.exists():
        return []
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            return list(csv.reader(handle))
    except (OSError, UnicodeDecodeError, csv.Error):
        return []


def _trace_text(run_result: dict[str, Any], *, max_steps: int = 12) -> str:
    parts: list[str] = []
    steps = run_result.get("steps", []) or []
    if not isinstance(steps, list):
        return ""
    for step in steps[-max_steps:]:
        if not isinstance(step, dict):
            continue
        for key in ("thought", "action"):
            value = step.get(key)
            if value:
                parts.append(str(value))
        action_input = step.get("action_input")
        if isinstance(action_input, dict):
            for key in ("sql", "code"):
                if action_input.get(key):
                    parts.append(str(action_input[key]))
    return "\n".join(parts).lower()


def _prediction_shape(rows: list[list[str]]) -> tuple[int, int, list[str], list[list[str]]]:
    if not rows:
        return 0, 0, [], []
    header = [cell.strip() for cell in rows[0]]
    data_rows = rows[1:]
    return len(header), len(data_rows), header, data_rows


def _header_has(header: list[str], terms: tuple[str, ...]) -> bool:
    lower = [column.lower() for column in header]
    return any(any(term in column for term in terms) for column in lower)


def _flat_prediction_values(data_rows: list[list[str]]) -> list[str]:
    return [cell.strip() for row in data_rows for cell in row if cell.strip()]


def _question_requests_comment_text(question: str) -> bool:
    q = question.lower()
    return "comment" in q and not re.search(r"\bcomment\s+id\b|\bid\s+of\s+the\s+comment\b", q)


def _question_requests_url(question: str) -> bool:
    q = question.lower()
    return any(term in q for term in ("website", "web site", "url", "link"))


def _question_requests_final_score_split(question: str) -> bool:
    q = question.lower()
    return "final score" in q and ("home" in q or "away" in q or "between" in q)


def _question_requests_plain_percentage(question: str) -> bool:
    q = question.lower()
    return any(term in q for term in ("percentage", "percent", "how much faster", "how much slower"))


def _question_requests_all_population(question: str) -> bool:
    return bool(re.search(r"\ball\b", question, flags=re.IGNORECASE))


def _has_external_threshold_language(evidence: str) -> bool:
    return any(term in evidence for term in ("standard", "common", "typical", "usually", "normal range is"))


def verify_semantic_contract(
    *,
    question: str,
    prediction_path: Path | None,
    run_result: dict[str, Any],
) -> SemanticVerificationReport:
    contract = infer_semantic_contract(question)
    rows = _read_prediction_rows(prediction_path)
    width, data_row_count, header, data_rows = _prediction_shape(rows)
    flat_values = _flat_prediction_values(data_rows)
    evidence = _trace_text(run_result)
    checks: list[SemanticCheck] = [
        SemanticCheck(
            "semantic_contract",
            "info",
            (
                f"aggregation={contract.aggregation or 'unknown'}; "
                f"target={contract.target_noun or 'unknown'}; "
                f"shape={contract.output_shape}; risks={contract.risk_codes}"
            ),
        )
    ]

    if "target_noun_count_risk" in contract.risk_codes:
        if "count(*)" in evidence and "distinct" not in evidence:
            checks.append(SemanticCheck(
                "target_noun_count_risk",
                "warning",
                "count question appears to use COUNT(*) without distinct/entity-count evidence",
            ))
        else:
            checks.append(SemanticCheck(
                "target_noun_count_risk",
                "info",
                "count target noun requires row-multiplicity check when joins or repeated measurements exist",
            ))

    if "denominator_risk" in contract.risk_codes:
        has_commitment = "numerator" in evidence and "denominator" in evidence
        checks.append(SemanticCheck(
            "denominator_risk",
            "info" if has_commitment else "warning",
            "percentage/ratio task should explicitly commit to numerator and denominator",
        ))

    if "temporal_aggregation_risk" in contract.risk_codes:
        has_sum_without_avg = "sum(" in evidence and "avg(" not in evidence and "average" in question.lower()
        checks.append(SemanticCheck(
            "temporal_aggregation_risk",
            "warning" if has_sum_without_avg else "info",
            "average plus temporal wording requires preserving AVG/mean before normalization",
        ))

    if "rank_semantics_risk" in contract.risk_codes:
        checks.append(SemanticCheck(
            "rank_semantics_risk",
            "info",
            "rank/order wording requires schema/value preview disambiguation before filtering or sorting",
        ))

    if "threshold_grounding_risk" in contract.risk_codes:
        has_threshold_evidence = any(token in evidence for token in (" between ", ">=", "<=", "threshold", "range"))
        severity = "info" if has_threshold_evidence and not _has_external_threshold_language(evidence) else "warning"
        checks.append(SemanticCheck(
            "threshold_grounding_risk",
            severity,
            "normal/abnormal/range wording must be grounded in context evidence or explicit data-derived thresholds",
        ))

    if "output_shape_semantic_risk" in contract.risk_codes:
        expected = len(contract.requested_fields)
        joined_single_cell = bool(data_rows and width == 1 and any(DELIMITED_CELL_RE.match(row[0]) for row in data_rows if row))
        if expected and width < expected:
            checks.append(SemanticCheck(
                "output_shape_semantic_risk",
                "warning",
                f"question requests {expected} fields but prediction has {width} column(s)",
            ))
        elif joined_single_cell:
            checks.append(SemanticCheck(
                "output_shape_semantic_risk",
                "warning",
                "single output cell appears to combine multiple requested fields",
            ))
        else:
            checks.append(SemanticCheck(
                "output_shape_semantic_risk",
                "info",
                "multi-field wording requires separate answer columns unless source schema says otherwise",
            ))
    elif (
        width == 1
        and data_rows
        and re.search(r"\b(score|result|tally|record)\b", question, flags=re.IGNORECASE)
        and any(DELIMITED_CELL_RE.match(row[0]) for row in data_rows if row)
    ):
        checks.append(SemanticCheck(
            "output_shape_semantic_risk",
            "warning",
            "single delimited score/result cell may need separate output columns",
        ))

    if _question_requests_comment_text(question):
        has_text_output = _header_has(header, ("text", "comment"))
        has_id_only_output = width == 1 and _header_has(header, ("id",)) and not has_text_output
        selected_id_in_trace = re.search(r"\bselect\s+(?:c\.)?id\b", evidence) is not None and "text" not in evidence
        if has_id_only_output or selected_id_in_trace:
            checks.append(SemanticCheck(
                "target_field_semantic_risk",
                "warning",
                "question asks for comment content but prediction appears to return an id field",
            ))

    if _question_requests_url(question):
        has_url_column = _header_has(header, ("url", "website", "web site", "link"))
        has_url_value = any(URL_RE.search(value) for value in flat_values)
        if not has_url_column and not has_url_value:
            checks.append(SemanticCheck(
                "target_field_semantic_risk",
                "warning",
                "question asks for website/url/link but prediction has no url-like column or value",
            ))

    if _question_requests_final_score_split(question) and width == 1:
        if data_rows and any(DELIMITED_CELL_RE.match(row[0]) for row in data_rows if row):
            checks.append(SemanticCheck(
                "target_field_semantic_risk",
                "warning",
                "final score for home/away teams is packed into one delimited column",
            ))

    if _question_requests_plain_percentage(question):
        if any(value.endswith("%") for value in flat_values):
            checks.append(SemanticCheck(
                "scalar_format_semantic_risk",
                "warning",
                "percentage answer includes a literal percent sign; evaluator expects numeric cell values",
            ))

    if contract.aggregation == "average" and _question_requests_all_population(question):
        if any(term in evidence for term in ("> 0", ">0", "dropna", "notnull", "is not null")):
            checks.append(SemanticCheck(
                "population_filter_semantic_risk",
                "warning",
                "question asks for all records but trace applies a non-null or positive-value filter before averaging",
            ))

    if "field_attachment_risk" in contract.risk_codes:
        singular_without_group = not GROUPING_RE.search(question)
        if singular_without_group and data_row_count > 1:
            checks.append(SemanticCheck(
                "field_attachment_risk",
                "warning",
                "singular entity wording produced multiple rows; verify grouping/modifier attachment",
            ))
        else:
            checks.append(SemanticCheck(
                "field_attachment_risk",
                "info",
                "descriptive fields and filtered fact rows may attach to different entities",
            ))

    if "entity_identity_risk" in contract.risk_codes:
        checks.append(SemanticCheck(
            "entity_identity_risk",
            "info",
            "identity/name output should follow source schema and join keys",
        ))

    if "long_doc_metric_risk" in contract.risk_codes:
        used_doc_tool = any(tool in evidence for tool in ("extract_doc_records", "read_doc", "search_doc"))
        checks.append(SemanticCheck(
            "long_doc_metric_risk",
            "info" if used_doc_tool else "warning",
            "document metric tasks should extract metric-local evidence before answering",
        ))

    return SemanticVerificationReport(contract=contract, checks=checks)


def semantic_report_to_dict(report: SemanticVerificationReport) -> dict[str, Any]:
    return {
        "contract": {
            "intents": report.contract.intents,
            "aggregation": report.contract.aggregation,
            "target_noun": report.contract.target_noun,
            "requested_fields": report.contract.requested_fields,
            "output_shape": report.contract.output_shape,
            "risk_codes": report.contract.risk_codes,
            "requirements": report.contract.requirements,
        },
        "checks": [
            {"code": check.code, "severity": check.severity, "detail": check.detail}
            for check in report.checks
        ],
    }
