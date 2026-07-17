from __future__ import annotations

import re
from pathlib import Path
from typing import Any


_CORRECTION_WORDS = (
    "corrected",
    "confirmed",
    "verified",
    "adjusted",
    "revised",
    "rectified",
    "amended",
    "finalized",
    "final",
)

_UNIT_RE = r"(?:mg/dL|K/uL|million/uL|U/L|IU/mL|g/dL|seconds?|%)"
_VALUE_RE = rf"(-?\d+(?:\.\d+)?(?:\s*{_UNIT_RE})?)"


def _to_float(raw: str) -> float:
    match = re.search(r"-?\d+(?:\.\d+)?", raw)
    if match is None:
        raise ValueError(f"No numeric value found in {raw!r}")
    return float(match.group(0))


def _term_pattern(term: str) -> str:
    if "publisher" in term.strip().lower():
        return r"\bpublisher(?:\s+affiliation)?\b"
    escaped = re.escape(term.strip())
    if term.strip().isalnum():
        return rf"\b{escaped}\b"
    return escaped


def _find_entity_id(text: str, entity_hint: str) -> str | None:
    hint = re.escape(entity_hint.strip() or "patient")
    patterns = [
        rf"\b{hint}\s+(?:with\s+Medical Record Number\s+|assigned\s+Medical Record Number\s+|registered\s+under\s+file\s+|associated\s+with\s+file(?:\s+number)?\s+|file(?:\s+number)?\s+)?(\d+)\b",
        r"\bMedical Record Number\s+(\d+)\b",
        r"\bfile(?:\s+number)?\s+(\d+)\b",
        r"\b(?:Case ID|Patient ID|record ID|reference ID|reference code|registry number)\s+(\d+)\b",
        r"\b(?:reference number|registry code|registration number)\s+(\d+)\b",
        r"\b(?:registered|cataloged|catalogued|filed|tracked|identified|maintained)\s+(?:under|with|by|at)\s+(?:the\s+)?(?:unique\s+)?(?:identifier|ID|reference code|reference ID|registry number|registry code|registration number|reference number|code|number)\s+(\d+)\b",
        r"\bentry\s+is\s+referenced\s+by\s+ID\s+(\d+)\b",
        r"\b(?:identifier|ID)\s+(\d+)\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return match.group(1)
    return None


def _sentence_containing(text: str, term: str) -> str:
    pattern = re.compile(_term_pattern(term), flags=re.IGNORECASE)
    sentences = re.split(r"(?<=[.!?])\s+", text)
    matches: list[str] = []
    for sentence in sentences:
        if pattern.search(sentence):
            matches.append(sentence.strip())
    if not matches:
        return text.strip()

    invalid_words = ("placeholder", "inaccurate", "anomaly", "unknown", "redacted")
    grounded = [
        sentence for sentence in matches
        if not any(word in sentence.lower() for word in invalid_words)
    ]
    candidates = grounded or matches
    metric_re = re.compile(_term_pattern(term) + r"[^.?!]{0,180}\d", flags=re.IGNORECASE)
    numeric_candidates = [sentence for sentence in candidates if metric_re.search(sentence)]
    return (numeric_candidates or candidates)[-1]


def _extract_number_after_metric(sentence: str, term: str) -> float | None:
    term_re = _term_pattern(term)
    correction_re = "|".join(_CORRECTION_WORDS)

    corrected_patterns = [
        rf"{term_re}.{{0,220}}?(?:{correction_re}).{{0,120}}?(?:to|at|as|be|being|value\s+of)\s+{_VALUE_RE}",
        rf"{term_re}.{{0,220}}?(?:{correction_re}).{{0,120}}?{_VALUE_RE}",
    ]
    for pattern in corrected_patterns:
        matches = re.findall(pattern, sentence, flags=re.IGNORECASE)
        if matches:
            return _to_float(matches[-1])

    simple_patterns = [
        rf"{term_re}[^.?!;]{{0,160}}?(?:code|as|to|at|with)\s+{_VALUE_RE}",
        rf"{term_re}[^.?!;]{{0,80}}?(?:was|is|of|at|=|:)\s+{_VALUE_RE}",
        rf"{term_re}[^.?!;]{{0,80}}?{_VALUE_RE}",
    ]
    for pattern in simple_patterns:
        match = re.search(pattern, sentence, flags=re.IGNORECASE)
        if match:
            return _to_float(match.group(1))

    return None


def _extract_year_after_metric(sentence: str, term: str) -> int | None:
    match = re.search(_term_pattern(term), sentence, flags=re.IGNORECASE)
    if not match:
        return None
    window = sentence[match.start():match.start() + 180]
    year_match = re.search(r"\b(18|19|20)\d{2}\b", window)
    if year_match:
        return int(year_match.group(0))
    return None


def _compact_snippet(text: str, *, max_chars: int = 360) -> str:
    normalized = re.sub(r"\s+", " ", text).strip()
    return normalized[:max_chars]


def extract_doc_records(
    context_root: Path,
    relative_path: str,
    target_terms: list[str],
    *,
    entity_hint: str = "patient",
    max_records: int = 200,
) -> dict[str, Any]:
    path = (context_root / relative_path).resolve()
    if context_root.resolve() not in path.parents and path != context_root.resolve():
        raise ValueError(f"Path escapes context dir: {relative_path}")
    if not path.exists():
        raise FileNotFoundError(f"Missing context asset: {relative_path}")
    if not target_terms:
        raise ValueError("target_terms must be a non-empty list")

    text = path.read_text(encoding="utf-8", errors="replace")
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    normalized_terms = [term.strip() for term in target_terms if term.strip()]
    term_res = [(term, re.compile(_term_pattern(term), flags=re.IGNORECASE)) for term in normalized_terms]

    records: list[dict[str, Any]] = []
    for paragraph_index, paragraph in enumerate(paragraphs):
        matched_terms = [term for term, term_re in term_res if term_re.search(paragraph)]
        if not matched_terms:
            continue

        entity_id = _find_entity_id(paragraph, entity_hint)
        for term in matched_terms:
            sentence = _sentence_containing(paragraph, term)
            records.append(
                {
                    "paragraph_index": paragraph_index,
                    "entity_id": entity_id,
                    "matched_term": term,
                    "number": _extract_number_after_metric(sentence, term),
                    "year": _extract_year_after_metric(sentence, term),
                    "sentence": _compact_snippet(sentence),
                }
            )
            if len(records) >= max_records:
                return {
                    "path": relative_path,
                    "target_terms": normalized_terms,
                    "record_count": len(records),
                    "truncated": True,
                    "records": records,
                }

    return {
        "path": relative_path,
        "target_terms": normalized_terms,
        "record_count": len(records),
        "truncated": False,
        "records": records,
    }
