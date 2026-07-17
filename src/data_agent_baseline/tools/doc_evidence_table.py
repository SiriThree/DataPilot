from __future__ import annotations

import csv
import hashlib
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from data_agent_baseline.tools.doc_extract import _CORRECTION_WORDS, _find_entity_id


@dataclass(slots=True)
class FieldValue:
    value: Any
    paragraph_index: int
    snippet: str
    confidence: int


FIELD_ALIASES = {
    "height": ["height", "standing height", "recorded height", "height_cm"],
    "weight": ["weight", "body weight", "recorded weight", "weight_kg"],
    "publisher": ["publisher", "publisher affiliation", "publisher_id", "publisher code"],
    "publisher affiliation": ["publisher", "publisher affiliation", "publisher_id", "publisher code"],
    "alignment": ["alignment", "moral alignment", "alignment_id"],
    "gender": ["gender", "sex", "gender_id"],
    "birth": ["birth", "born", "birth year", "year of birth"],
    "birth_year": ["birth year", "born", "year of birth", "birth"],
}

ID_HINTS = ["entity", "record", "operative", "subject", "patient", "hero", "unit", "asset"]
NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")


def _norm(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def _field_terms(field: str) -> list[str]:
    normalized = field.strip().lower()
    terms = [normalized]
    terms.extend(FIELD_ALIASES.get(normalized, []))
    return sorted({term for term in terms if term}, key=len, reverse=True)


def _term_re(term: str) -> re.Pattern[str]:
    words = re.escape(term.strip()).replace(r"\ ", r"\s+")
    return re.compile(rf"\b{words}\b", flags=re.IGNORECASE)


def _find_entity_any(paragraph: str, entity_hint: str) -> str | None:
    hints = [entity_hint, *ID_HINTS]
    for hint in hints:
        if not hint:
            continue
        entity_id = _find_entity_id(paragraph, hint)
        if entity_id is not None:
            return entity_id
    # Handle compact parenthetical forms such as "Curse (ID 198)".
    match = re.search(r"\b(?:ID|identifier|registry number|reference code)\s*[:#]?\s*(\d+)\b", paragraph, re.IGNORECASE)
    if match:
        return match.group(1)
    return None


def _sentences(paragraph: str) -> list[str]:
    return [item.strip() for item in re.split(r"(?<=[.!?])\s+", paragraph) if item.strip()]


def _compact(text: str, *, max_chars: int = 260) -> str:
    return re.sub(r"\s+", " ", text).strip()[:max_chars]


def _numbers(text: str) -> list[float]:
    return [float(match.group(0)) for match in NUMBER_RE.finditer(text)]


def _field_pattern_value(sentence: str, field: str, terms: list[str]) -> tuple[float | None, int]:
    lower = sentence.lower()
    correction_bonus = 8 if any(word in lower for word in _CORRECTION_WORDS) else 0
    placeholder_bonus = 5 if "placeholder" in lower and "field" in lower else 0

    # "placeholder data of 0.0 for both height and weight" should bind 0.0
    # to the field, even if a later narrative sentence mentions another value.
    for term in terms:
        if term in {"height", "weight"}:
            match = re.search(
                rf"(?:placeholder|field|fields|data)[^.?!]{{0,80}}?({_number_pattern()})"
                rf"[^.?!]{{0,80}}?\b{re.escape(term)}\b",
                sentence,
                flags=re.IGNORECASE,
            )
            if match:
                return float(match.group(1)), 80 + placeholder_bonus

    for term in terms:
        term_pattern = _term_re(term).pattern
        corrected_patterns = [
            rf"{term_pattern}[^.?!]{{0,220}}?(?:{'|'.join(_CORRECTION_WORDS)})[^.?!]{{0,120}}?"
            rf"(?:to|at|as|be|being|value\s+of)\s+({_number_pattern()})",
            rf"(?:{'|'.join(_CORRECTION_WORDS)})[^.?!]{{0,160}}?{term_pattern}[^.?!]{{0,120}}?"
            rf"(?:to|at|as|be|being|value\s+of)?\s*({_number_pattern()})",
        ]
        for pattern in corrected_patterns:
            matches = re.findall(pattern, sentence, flags=re.IGNORECASE)
            if matches:
                return float(matches[-1]), 90 + correction_bonus

    for term in terms:
        term_pattern = _term_re(term).pattern
        direct_patterns = [
            rf"{term_pattern}[^.?!;]{{0,80}}?(?:is|was|recorded|documented|listed|logged|classified|affiliation|code|as|at|=|:)\s+(?:as\s+)?({_number_pattern()})",
            rf"{term_pattern}\s+({_number_pattern()})",
            rf"({_number_pattern()})\s*(?:centimeters|kilograms|kg|cm)\b[^.?!;]{{0,80}}?{term_pattern}",
        ]
        if term in {"publisher", "publisher affiliation", "publisher_id", "publisher code"}:
            direct_patterns.extend([
                rf"(?:registered|classified|documented|listed|affiliated|on\s+record|under)[^.?!;]{{0,120}}?{term_pattern}\s+({_number_pattern()})",
                rf"(?:jurisdiction|oversight)[^.?!;]{{0,80}}?of\s+{term_pattern}\s+({_number_pattern()})",
                rf"{term_pattern}[^.?!;]{{0,40}}?(?:confirmed|finalized)[^.?!;]{{0,80}}?(?:as|to)\s+({_number_pattern()})",
            ])
        for pattern in direct_patterns:
            match = re.search(pattern, sentence, flags=re.IGNORECASE)
            if match:
                return float(match.group(1)), 60 + correction_bonus

    values = _numbers(sentence)
    if values and any(_term_re(term).search(sentence) for term in terms):
        return values[-1], 20 + correction_bonus
    return None, 0


def _number_pattern() -> str:
    return r"-?\d+(?:\.\d+)?"


def _extract_field_value(paragraph: str, field: str, paragraph_index: int) -> FieldValue | None:
    terms = _field_terms(field)
    candidate_sentences = [
        sentence
        for sentence in _sentences(paragraph)
        if any(_term_re(term).search(sentence) for term in terms)
    ]
    if not candidate_sentences:
        return None

    best: FieldValue | None = None
    for sentence in candidate_sentences:
        value, confidence = _field_pattern_value(sentence, field.lower(), terms)
        if value is None:
            continue
        if value.is_integer():
            value = int(value)
        item = FieldValue(
            value=value,
            paragraph_index=paragraph_index,
            snippet=_compact(sentence),
            confidence=confidence,
        )
        if best is None or item.confidence > best.confidence:
            best = item
    return best


def _should_replace(existing: FieldValue | None, candidate: FieldValue) -> bool:
    if existing is None:
        return True
    if candidate.confidence != existing.confidence:
        return candidate.confidence > existing.confidence
    return candidate.paragraph_index > existing.paragraph_index


def _safe_output_path(context_root: Path, relative_path: str, fields: list[str]) -> Path:
    digest_input = f"{context_root.resolve()}::{relative_path}::{','.join(fields)}"
    digest = hashlib.sha1(digest_input.encode("utf-8")).hexdigest()[:16]
    out_dir = Path(tempfile.gettempdir()) / "data_agent_evidence_tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f"doc_evidence_{digest}.csv"


def extract_doc_evidence_table(
    context_root: Path,
    relative_path: str,
    fields: list[str],
    *,
    entity_hint: str = "entity",
    max_records: int = 1_000,
) -> dict[str, Any]:
    """Build an entity-keyed evidence table from repeated narrative records.

    This tool is intentionally generic: it extracts numeric field values near
    requested field terms, merges them by entity id across document sections,
    writes a temporary CSV table, and returns compact coverage/evidence details.
    """

    path = (context_root / relative_path).resolve()
    if context_root.resolve() not in path.parents and path != context_root.resolve():
        raise ValueError(f"Path escapes context dir: {relative_path}")
    if not path.exists():
        raise FileNotFoundError(f"Missing context asset: {relative_path}")
    normalized_fields = [field.strip() for field in fields if isinstance(field, str) and field.strip()]
    if not normalized_fields:
        raise ValueError("fields must be a non-empty list of field names")

    text = path.read_text(encoding="utf-8", errors="replace")
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    by_entity: dict[str, dict[str, FieldValue]] = {}

    for paragraph_index, paragraph in enumerate(paragraphs):
        entity_id = _find_entity_any(paragraph, entity_hint)
        if entity_id is None:
            continue
        row = by_entity.setdefault(entity_id, {})
        for field in normalized_fields:
            candidate = _extract_field_value(paragraph, field, paragraph_index)
            if candidate is not None and _should_replace(row.get(field), candidate):
                row[field] = candidate

    records: list[dict[str, Any]] = []
    for entity_id in sorted(by_entity, key=lambda item: (len(item), item)):
        fields_for_entity = by_entity[entity_id]
        row: dict[str, Any] = {"entity_id": entity_id}
        evidence: dict[str, Any] = {}
        for field in normalized_fields:
            value = fields_for_entity.get(field)
            row[field] = value.value if value is not None else None
            evidence[field] = (
                {
                    "paragraph_index": value.paragraph_index,
                    "snippet": value.snippet,
                    "confidence": value.confidence,
                }
                if value is not None
                else None
            )
        row["evidence"] = evidence
        records.append(row)
        if len(records) >= max_records:
            break

    output_path = _safe_output_path(context_root, relative_path, normalized_fields)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["entity_id", *normalized_fields])
        writer.writeheader()
        for row in records:
            writer.writerow({key: row.get(key) for key in ["entity_id", *normalized_fields]})

    coverage = {
        field: sum(1 for row in records if row.get(field) not in (None, ""))
        for field in normalized_fields
    }
    complete_records = sum(
        1 for row in records
        if all(row.get(field) not in (None, "") for field in normalized_fields)
    )

    return {
        "path": relative_path,
        "fields": normalized_fields,
        "entity_count": len(records),
        "complete_record_count": complete_records,
        "field_coverage": coverage,
        "evidence_table_csv": output_path.as_posix(),
        "records_preview": records[:30],
        "truncated": len(by_entity) > len(records),
        "usage_hint": (
            "Use execute_python to read evidence_table_csv and compute filters, numerator, "
            "denominator, percentages, or grouped aggregates deterministically."
        ),
    }
