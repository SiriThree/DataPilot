"""Answer Verifier: validate prediction.csv before submission.

Performs 5 checks:
1. prediction.csv exists (file-system level, checked in runner)
2. Header row present
3. No redundant/extra columns beyond what question asks
4. Numeric values rounded to 2 decimal places
5. Dates in ISO format (YYYY-MM-DD)
"""

from __future__ import annotations

import csv
import re
from io import StringIO
from pathlib import Path
from typing import Any


ISO_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}(T\d{2}:\d{2}:\d{2})?$")


def _is_numeric(s: str) -> bool:
    try:
        float(s)
        return True
    except (ValueError, TypeError):
        return False


def _is_date_like(s: str) -> bool:
    """Detect strings that look like dates but are not ISO format."""
    if ISO_DATE_RE.match(s):
        return False  # Already correct
    # Common non-ISO patterns
    return bool(re.match(r"^\d{1,2}[/-]\d{1,2}[/-]\d{2,4}$", s))


def verify_answer(
    columns: list[str],
    rows: list[list[Any]],
    question: str = "",
) -> dict[str, Any]:
    """Validate the answer table before writing.

    Returns a dict with:
        - ok: bool (True if passes all checks)
        - warnings: list[str] (non-fatal issues)
        - errors: list[str] (fatal issues)
        - suggested_fixes: list[str]
    """
    warnings: list[str] = []
    errors: list[str] = []
    fixes: list[str] = []

    # Check 1: non-empty columns
    if not columns:
        errors.append("Answer has no columns")
        return {"ok": False, "warnings": warnings, "errors": errors, "suggested_fixes": fixes}

    if not rows:
        warnings.append("Answer has no data rows — may be intentional if query returns no results")

    # Check 2: redundant / duplicate columns (skip if no data rows)
    if rows:
        col_value_sets: list[set[str]] = []
        for ci in range(len(columns)):
            vals = {str(rows[ri][ci]) if ci < len(rows[ri]) else "" for ri in range(len(rows))}
            col_value_sets.append(vals)

        redundant_pairs: list[tuple[int, int]] = []
        for i in range(len(col_value_sets)):
            for j in range(i + 1, len(col_value_sets)):
                if col_value_sets[i] == col_value_sets[j]:
                    redundant_pairs.append((i, j))
                    fixes.append(
                        f"Columns [{i}]'{columns[i]}' and [{j}]'{columns[j]}' have identical values — "
                        "consider removing one to avoid redundancy penalty"
                    )

        if redundant_pairs:
            warnings.append(f"Found {len(redundant_pairs)} redundant column pair(s)")

    # Check 3: numeric precision
    bad_numeric: list[tuple[int, int, str]] = []
    for ri, row in enumerate(rows):
        for ci, val in enumerate(row):
            s = str(val)
            if _is_numeric(s):
                # Check if it has more than 2 decimal places
                if "." in s and len(s.split(".")[1]) > 2:
                    try:
                        fixed = f"{float(s):.2f}"
                        bad_numeric.append((ri, ci, fixed))
                    except ValueError:
                        pass

    if bad_numeric:
        fixes.extend(
            f"Row {ri} col '{columns[ci]}' value '{rows[ri][ci]}' should be rounded to 2 decimals (e.g. {fixed})"
            for ri, ci, fixed in bad_numeric[:10]
        )
        warnings.append(f"Found {len(bad_numeric)} value(s) with >2 decimal places")

    # Check 4: date format
    bad_dates: list[tuple[int, int]] = []
    for ri, row in enumerate(rows):
        for ci, val in enumerate(row):
            s = str(val).strip()
            if _is_date_like(s):
                bad_dates.append((ri, ci))

    if bad_dates:
        fixes.extend(
            f"Row {ri} col '{columns[ci]}' value '{rows[ri][ci]}' is not ISO date format (use YYYY-MM-DD)"
            for ri, ci in bad_dates[:10]
        )
        warnings.append(f"Found {len(bad_dates)} date-like value(s) not in ISO format")

    # Check 5: column count heuristics — if > 10 columns, likely too many
    if len(columns) > 10:
        warnings.append(
            f"Answer has {len(columns)} columns — most questions expect fewer. "
            "Extra columns will be penalized."
        )

    return {
        "ok": len(errors) == 0,
        "warnings": warnings,
        "errors": errors,
        "suggested_fixes": fixes,
    }


def verify_prediction_csv(path: Path, question: str = "") -> dict[str, Any]:
    """Verify an already-written prediction.csv file."""
    if not path.exists():
        return {"ok": False, "errors": [f"File not found: {path}"], "warnings": [], "suggested_fixes": []}

    try:
        text = path.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError:
        return {"ok": False, "errors": ["prediction.csv is not valid UTF-8"], "warnings": [], "suggested_fixes": []}

    if not text.strip():
        return {"ok": False, "errors": ["prediction.csv is empty"], "warnings": [], "suggested_fixes": []}

    reader = csv.reader(StringIO(text))
    rows: list[list[str]] = list(reader)

    if not rows:
        return {"ok": False, "errors": ["No header row"], "warnings": [], "suggested_fixes": []}

    columns = rows[0]
    data_rows = rows[1:]

    return verify_answer(columns, data_rows, question)


def sanitize_answer_table(
    columns: list[str],
    rows: list[list[Any]],
) -> tuple[list[str], list[list[str]]]:
    """Auto-fix common formatting issues: round numbers, normalize empty values, strip whitespace."""
    clean_columns = [c.strip() for c in columns]
    clean_rows: list[list[str]] = []

    for row in rows:
        clean_row: list[str] = []
        for val in row:
            s = str(val).strip()
            s_lower = s.lower()
            if s == "" or s_lower in ("null", "none", "nan", "inf", "-inf", "infinity", "-infinity"):
                clean_row.append("")
            elif _is_numeric(s):
                try:
                    clean_row.append(f"{float(s):.2f}")
                except ValueError:
                    clean_row.append(s)
            else:
                clean_row.append(s)
        clean_rows.append(clean_row)

    return clean_columns, clean_rows
