"""5-layer verification chain for prediction.csv.

Layers (run in order, each produces a VerificationCheck):
  1. readability_check  — file exists, is UTF-8, CSV parses without error
  2. contract_check     — has header, consistent column width, no duplicate headers
  3. task_contract_check — matches inferred output contract from task.json
  4. sanity_check       — data looks reasonable (not all empty, not all dupes, no error text)
  5. shape_check        — column count sane, not a fallback single-empty-cell output

Also includes infer_output_contract() to extract expected output shape from task.json.

Adapted from soonhp/KddCupDataAgents verification.py.
"""

from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

MAX_DATA_ROWS = 1_000_000
MAX_CELL_CHARS = 20_000
MAX_HEADER_CHARS = 512
SUSPICIOUS_TOKENS = (
    "traceback", "exception", "error:", "api key",
    "authentication", "permission denied", "stack trace",
)
CONTRACT_COLUMN_KEYS = (
    "expected_columns", "output_columns", "columns", "headers", "prediction_columns",
)
CONTRACT_SCHEMA_KEYS = (
    "output_schema", "answer_schema", "prediction_schema", "schema",
)


@dataclass
class VerificationCheck:
    name: str
    passed: bool
    detail: str


@dataclass
class OutputContract:
    expected_columns: list[str] = field(default_factory=list)
    expected_column_count: int | None = None
    min_rows: int | None = None
    max_rows: int | None = None
    exact_columns: bool = True
    source: str = ""


@dataclass
class VerificationReport:
    task_id: str
    all_passed: bool
    checks: list[VerificationCheck]
    output_contract: OutputContract | None = None


# ── Output contract inference ─────────────────────────────────────

def _normalize_column_name(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip())


def _dedupe_columns(columns: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for col in columns:
        n = _normalize_column_name(str(col))
        if not n or n.lower() in seen:
            continue
        seen.add(n.lower())
        out.append(n)
    return out


def _extract_columns_from_value(value: Any) -> list[str]:
    """Extract column names from various formats found in task.json."""
    if value is None:
        return []
    if isinstance(value, str):
        cleaned = value.strip()
        if not cleaned:
            return []
        cleaned = re.sub(r"^(columns?|headers?|output)\s*[:=]\s*", "", cleaned, flags=re.IGNORECASE)
        if "," in cleaned:
            return _dedupe_columns(cleaned.split(","))
        return _dedupe_columns([cleaned])
    if isinstance(value, list):
        cols = []
        for item in value:
            if isinstance(item, str):
                cols.append(item)
            elif isinstance(item, dict):
                name = item.get("name") or item.get("column") or item.get("field")
                if name:
                    cols.append(str(name))
        return _dedupe_columns(cols)
    if isinstance(value, dict):
        for key in CONTRACT_COLUMN_KEYS:
            if key in value:
                cols = _extract_columns_from_value(value[key])
                if cols:
                    return cols
        props = value.get("properties")
        if isinstance(props, dict):
            return _dedupe_columns(list(props.keys()))
        fields = value.get("fields")
        if isinstance(fields, list):
            return _extract_columns_from_value(fields)
    return []


def _extract_question_columns(question: str) -> list[str]:
    """Regex-based column extraction from question text."""
    if not question:
        return []
    patterns = [
        r"columns?\s*(?:named|are|:)\s*([A-Za-z0-9_ ,\-/]+)",
        r"headers?\s*(?:named|are|:)\s*([A-Za-z0-9_ ,\-/]+)",
        r"output\s+fields?\s*(?:named|are|:)\s*([A-Za-z0-9_ ,\-/]+)",
    ]
    for pat in patterns:
        m = re.search(pat, question, re.IGNORECASE)
        if m:
            raw = m.group(1)
            raw = re.split(r"\s+(?:and|with|where|for|from)\s+", raw, maxsplit=1, flags=re.IGNORECASE)[0]
            cols = _dedupe_columns(raw.split(","))
            if cols:
                return cols
    return []


def _infer_question_column_count(question: str) -> int | None:
    q = question.strip().lower()
    if not q:
        return None

    if (
        "type" in q
        and "expense" in q
        and any(term in q for term in ("total value", "total cost", "sum", "total"))
    ):
        return 2

    # Explicit "A, B and C" list-style field requests.
    field_patterns = [
        r"\blist\s+(?:their\s+|the\s+)?(.+?)(?:\s+for\b|\s+of\b|\s+where\b|\s+that\b|[?.]?$)",
        r"\bgive\s+(?:their\s+|the\s+)?(.+?)(?:\s+for\b|\s+of\b|\s+where\b|\s+that\b|[?.]?$)",
        r"\breturn\s+(?:their\s+|the\s+)?(.+?)(?:\s+for\b|\s+of\b|\s+where\b|\s+that\b|[?.]?$)",
        r"\bidentify\s+(?:the\s+)?(.+?)(?:\s+approved\b|\s+for\b|\s+of\b|\s+where\b|[?.]?$)",
    ]
    stop_terms = {
        "all",
        "the",
        "their",
        "a",
        "an",
        "in",
        "with",
        "who",
        "that",
        "which",
        "what",
    }
    for pattern in field_patterns:
        match = re.search(pattern, q)
        if not match:
            continue
        phrase = match.group(1)
        phrase = re.sub(r"\b(and\s+)?their\b", "", phrase).strip(" ,")
        parts = [
            part.strip(" ,")
            for part in re.split(r"\s*,\s*|\s+and\s+|\s*&\s*", phrase)
            if part.strip(" ,")
        ]
        parts = [part for part in parts if part not in stop_terms and len(part.split()) <= 6]
        if len(parts) >= 2:
            return len(parts)

    # Single scalar or entity-field questions. These are intentionally broad but
    # only produce a column-count contract, not a header-name contract.
    single_field_patterns = (
        r"\blist\s+all\s+(?:the\s+)?[^,?]+",
        r"\bwhat(?:'s| is| was| were)?\s+the\s+[^?]+",
        r"\bwhich\s+[^?]+",
        r"\bhow many\b",
        r"\bcount\b",
        r"\baverage\b",
        r"\bmean\b",
        r"\bsum\b",
        r"\btotal\b",
    )
    if any(re.search(pattern, q) for pattern in single_field_patterns):
        return 1
    return None


def _infer_question_max_rows(question: str) -> int | None:
    q = question.lower()
    patterns = (
        r"\btop\s+(\d+)\b",
        r"\bbottom\s+(\d+)\b",
        r"\bfirst\s+(\d+)\b",
        r"\blast\s+(\d+)\b",
        r"\b(\d+)\s+(?:largest|smallest|highest|lowest|oldest|youngest|most recent)\b",
    )
    for pattern in patterns:
        match = re.search(pattern, q)
        if match:
            value = int(match.group(1))
            if value > 0:
                return value
    return None


def infer_output_contract(task_dir: Path) -> OutputContract | None:
    """Infer expected output shape from task.json."""
    task_json = task_dir / "task.json"
    if not task_json.exists():
        return None
    try:
        payload = json.loads(task_json.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None

    expected_columns: list[str] = []
    source = ""
    for key in CONTRACT_COLUMN_KEYS:
        if key in payload:
            expected_columns = _extract_columns_from_value(payload[key])
            source = f"task_json.{key}"
            if expected_columns:
                break
    if not expected_columns:
        for key in CONTRACT_SCHEMA_KEYS:
            if key in payload:
                expected_columns = _extract_columns_from_value(payload[key])
                source = f"task_json.{key}"
                if expected_columns:
                    break
    if not expected_columns:
        question = str(payload.get("question", ""))
        expected_columns = _extract_question_columns(question)
        if expected_columns:
            source = "task_json.question"

    question = str(payload.get("question", ""))
    expected_column_count = None
    if expected_columns:
        expected_column_count = len(expected_columns)
    else:
        expected_column_count = _infer_question_column_count(question)
        if expected_column_count is not None and not source:
            source = "task_json.question_shape"

    min_rows = payload.get("min_rows") or payload.get("expected_min_rows")
    max_rows = payload.get("max_rows") or payload.get("expected_max_rows")
    exact_columns = bool(payload.get("exact_columns", True))

    n_min = int(min_rows) if isinstance(min_rows, int) and min_rows >= 0 else None
    n_max = int(max_rows) if isinstance(max_rows, int) and max_rows >= 0 else None
    inferred_max = _infer_question_max_rows(question)
    if n_max is None and inferred_max is not None:
        n_max = inferred_max
        source = f"{source}+question_row_cap" if source else "task_json.question_row_cap"

    if not expected_columns and expected_column_count is None and n_min is None and n_max is None:
        return None
    return OutputContract(
        expected_columns=expected_columns,
        expected_column_count=expected_column_count,
        min_rows=n_min,
        max_rows=n_max,
        exact_columns=exact_columns,
        source=source or "task_json",
    )


# ── 5-layer checks ────────────────────────────────────────────────

def _check(name: str, passed: bool, detail: str) -> VerificationCheck:
    return VerificationCheck(name=name, passed=passed, detail=detail)


def _read_csv_safe(path: Path) -> tuple[list[list[str]], VerificationCheck | None]:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            return list(csv.reader(f)), None
    except UnicodeDecodeError as e:
        return [], _check("readability_check", False, f"not valid UTF-8: {e}")
    except csv.Error as e:
        return [], _check("readability_check", False, f"CSV parse error: {e}")
    except OSError as e:
        return [], _check("readability_check", False, f"read error: {e}")


def _run_readability_check(path: Path) -> tuple[list[list[str]], VerificationCheck]:
    if not path.exists():
        return [], _check("readability_check", False, "prediction.csv missing")
    if path.stat().st_size == 0:
        return [], _check("readability_check", False, "prediction.csv is empty")
    rows, err = _read_csv_safe(path)
    if err is not None:
        return [], err
    return rows, _check("readability_check", True, "parsed as UTF-8 CSV")


def _run_contract_check(rows: list[list[str]]) -> VerificationCheck:
    if not rows or not rows[0]:
        return _check("contract_check", False, "header missing")
    header = [c.strip() for c in rows[0]]
    if not any(header):
        return _check("contract_check", False, "header has no non-empty columns")

    dupes = sorted({c for c in header if c and header.count(c) > 1})
    if dupes:
        return _check("contract_check", False, f"duplicate headers: {dupes[:5]}")

    long_headers = [c for c in header if len(c) > MAX_HEADER_CHARS]
    if long_headers:
        return _check("contract_check", False, f"header too long at {len(long_headers)} columns")

    width = len(rows[0])
    inconsistent = [i for i, row in enumerate(rows[1:], start=2) if len(row) != width]
    if inconsistent:
        return _check("contract_check", False, f"inconsistent column width at rows {inconsistent[:5]}")

    return _check("contract_check", True, "header/width consistent")


def _run_task_contract_check(rows: list[list[str]], contract: OutputContract | None) -> VerificationCheck:
    if contract is None:
        return _check("task_contract_check", True, "no task-specific output contract")
    if not rows or not rows[0]:
        return _check("task_contract_check", False, "cannot validate without header")

    header = [_normalize_column_name(c).lower() for c in rows[0]]
    data_rows = max(len(rows) - 1, 0)

    if contract.expected_column_count is not None and len(header) != contract.expected_column_count:
        return _check(
            "task_contract_check",
            False,
            (
                f"expected {contract.expected_column_count} column(s) from {contract.source}, "
                f"got {len(header)}"
            ),
        )

    if contract.expected_columns:
        expected = [_normalize_column_name(c).lower() for c in contract.expected_columns]
        missing = [c for c in expected if c not in header]
        if missing:
            return _check("task_contract_check", False,
                          f"missing columns from {contract.source}: {missing}")
        if contract.exact_columns and header != expected:
            return _check("task_contract_check", False,
                          f"header mismatch: expected {expected}, got {header}")

    if contract.min_rows is not None and data_rows < contract.min_rows:
        return _check("task_contract_check", False,
                      f"expected ≥{contract.min_rows} rows, got {data_rows}")
    if contract.max_rows is not None and data_rows > contract.max_rows:
        return _check("task_contract_check", False,
                      f"expected ≤{contract.max_rows} rows, got {data_rows}")

    return _check("task_contract_check", True, f"contract satisfied ({contract.source})")


def _run_sanity_check(rows: list[list[str]]) -> VerificationCheck:
    data_rows = rows[1:] if rows else []
    if not data_rows:
        return _check("sanity_check", False, "no data rows")

    if len(data_rows) > MAX_DATA_ROWS:
        return _check("sanity_check", False, f"too many rows ({len(data_rows)})")

    empty_count = 0
    long_cells = 0
    suspicious = 0
    for row in data_rows:
        if all(not c.strip() for c in row):
            empty_count += 1
        for cell in row:
            s = cell.strip()
            if len(s) > MAX_CELL_CHARS:
                long_cells += 1
            if any(tok in s.lower() for tok in SUSPICIOUS_TOKENS):
                suspicious += 1

    empty_ratio = empty_count / len(data_rows)
    unique_ratio = len({tuple(row) for row in data_rows}) / len(data_rows)

    if empty_ratio > 0.95:
        return _check("sanity_check", False, f"too many empty rows ({empty_ratio:.1%})")
    if unique_ratio < 0.05 and len(data_rows) >= 20:
        return _check("sanity_check", False, f"too many duplicates (unique {unique_ratio:.1%})")
    if long_cells:
        return _check("sanity_check", False, f"cell length exceeded in {long_cells} cells")
    if suspicious:
        return _check("sanity_check", False, f"error-like text in {suspicious} cells")

    return _check("sanity_check", True,
                  f"empty={empty_ratio:.1%}, unique={unique_ratio:.1%}")


def _run_shape_check(rows: list[list[str]]) -> VerificationCheck:
    if not rows or not rows[0]:
        return _check("shape_check", False, "cannot inspect without header")
    width = len(rows[0])
    data_rows = rows[1:]
    if width > 1000:
        return _check("shape_check", False, f"too many columns ({width})")
    if len(data_rows) == 1 and width == 1:
        val = data_rows[0][0].strip() if data_rows[0] else ""
        if val == "":
            return _check("shape_check", False, "single empty answer cell fallback")
    return _check("shape_check", True, f"{len(data_rows)} rows × {width} cols")


# ── Main verification entry point ─────────────────────────────────

def run_full_verification(
    task_id: str,
    prediction_path: Path,
    output_contract: OutputContract | None = None,
) -> VerificationReport:
    """Run all 5 layers of verification against a prediction.csv."""
    checks: list[VerificationCheck] = []

    # Layer 1: readability
    rows, readability = _run_readability_check(prediction_path)
    checks.append(readability)
    if not readability.passed:
        return VerificationReport(task_id=task_id, all_passed=False, checks=checks, output_contract=output_contract)

    # Layer 2-5
    checks.append(_run_contract_check(rows))
    checks.append(_run_task_contract_check(rows, output_contract))
    checks.append(_run_sanity_check(rows))
    checks.append(_run_shape_check(rows))

    return VerificationReport(
        task_id=task_id,
        all_passed=all(c.passed for c in checks),
        checks=checks,
        output_contract=output_contract,
    )


def verification_report_to_dict(report: VerificationReport) -> dict[str, Any]:
    return {
        "task_id": report.task_id,
        "all_passed": report.all_passed,
        "checks": [{"name": c.name, "passed": c.passed, "detail": c.detail} for c in report.checks],
        "output_contract": {
            "expected_columns": report.output_contract.expected_columns,
            "expected_column_count": report.output_contract.expected_column_count,
            "min_rows": report.output_contract.min_rows,
            "max_rows": report.output_contract.max_rows,
            "source": report.output_contract.source,
        } if report.output_contract else None,
    }
