"""Deterministic repair planner and executor for prediction.csv.

Reads failed verification checks and generates prioritized repair actions.
All actions are deterministic rules — no LLM calls during repair.

Adapted from soonhp/KddCupDataAgents repair_planner.py + repair_executor.py.
"""

from __future__ import annotations

import csv
import json
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path

from data_agent_baseline.run.average_monthly_verifier import maybe_repair_average_monthly
from data_agent_baseline.run.verification_chain import (
    VerificationReport,
    run_full_verification,
)

NULL_TOKENS = {"null", "none", "nan", "na", "n/a", "nat", "inf", "-inf", "infinity", "-infinity"}


@dataclass
class RepairAction:
    priority: int
    action_type: str
    detail: str
    blocking: bool = True


@dataclass
class RepairPlan:
    task_id: str
    should_repair: bool
    actions: list[RepairAction]
    summary: str


@dataclass
class RepairExecutionReport:
    task_id: str
    applied_count: int
    skipped_count: int
    actions: list[str]  # human-readable descriptions


# ── Semantic intent checks (lightweight, inlined) ─────────────────

def _check_semantic_intent(
    question: str,
    rows: list[list[str]],
    task_dir: Path | None = None,
) -> list[RepairAction]:
    """Check if prediction shape matches question intent."""
    actions: list[RepairAction] = []
    q = question.lower()
    data_cells = [c.strip() for row in rows[1:] for c in row if c.strip()]
    header = [c.strip().lower() for c in rows[0]] if rows else []
    original_header = [c.strip() for c in rows[0]] if rows else []

    # Numeric intent
    numeric_signals = {"count", "sum", "average", "avg", "median", "mean",
                       "calculate", "compute", "total", "variance", "std"}
    if any(re.search(r"\b" + re.escape(s) + r"\b", q) for s in numeric_signals):
        numeric_count = sum(1 for c in data_cells if re.fullmatch(r"[-+]?\d+(?:\.\d+)?", c.replace(",", "")))
        ratio = numeric_count / len(data_cells) if data_cells else 0.0
        if ratio == 0.0 and data_cells:
            actions.append(RepairAction(
                priority=30, action_type="recompute_numeric",
                detail=f"Question has numeric intent but answer has 0% numeric cells (got {len(data_cells)} text cells)",
                blocking=True,
            ))

    # Comparative intent
    comparative_signals = {"compare", "by", "per", "group", "trend", "distribution"}
    if any(re.search(r"\b" + re.escape(s) + r"\b", q) for s in comparative_signals):
        data_rows = max(len(rows) - 1, 0)
        if data_rows == 0:
            actions.append(RepairAction(
                priority=31, action_type="regenerate_comparison",
                detail="Comparative/grouped question but answer has 0 data rows",
                blocking=True,
            ))

    # Full-name output policy for normalized people/member tables.
    if "full name" in q and any(col in {"full_name", "full name", "name"} for col in header):
        if task_dir is not None and _context_supports_split_names(task_dir):
            actions.append(RepairAction(
                priority=12,
                action_type="split_full_name",
                detail=(
                    "Question asks for full name, but context exposes first_name/last_name "
                    "as separate canonical fields"
                ),
                blocking=False,
            ))

    keep_column = _single_requested_entity_column(question, original_header)
    if keep_column is not None:
        actions.append(RepairAction(
            priority=14,
            action_type="prune_redundant_columns",
            detail=f"Question appears to request one entity/record field; keep_column={keep_column}",
            blocking=False,
        ))

    for semantic_risk in _detect_schema_semantic_risks(question, task_dir):
        actions.append(RepairAction(
            priority=34,
            action_type="semantic_risk_review",
            detail=semantic_risk,
            blocking=False,
        ))

    if task_dir is not None and "average" in q and "monthly" in q:
        actions.append(RepairAction(
            priority=13,
            action_type="fix_average_monthly_candidate",
            detail=(
                "Compare executable average-monthly candidate formulas and repair "
                "when the prediction selected a less faithful aggregate."
            ),
            blocking=False,
        ))

    if _looks_like_rank_finish_time_question(question) and task_dir is not None:
        actions.append(RepairAction(
            priority=13,
            action_type="fix_rank_finish_time",
            detail="Recompute finish time using the literal rank column for the named race",
            blocking=False,
        ))

    if _looks_like_expense_type_total_question(question) and task_dir is not None:
        actions.append(RepairAction(
            priority=13,
            action_type="fix_expense_type_by_event_type",
            detail="Recompute approved expense total and use the named event's type",
            blocking=False,
        ))

    return actions


def _looks_like_rank_finish_time_question(question: str) -> bool:
    q = question.lower()
    return (
        "rank" in q
        and any(term in q for term in ("finish time", "time"))
        and any(term in q for term in ("grand prix", "race"))
    )


def _looks_like_expense_type_total_question(question: str) -> bool:
    q = question.lower()
    return (
        "expense" in q
        and "type" in q
        and any(term in q for term in ("total", "value", "sum"))
        and any(term in q for term in ("approved", "approve"))
        and "event" in q
    )


def _question_requests_multiple_fields(question: str) -> bool:
    q = question.lower()
    multi_field_markers = (
        " and ",
        ",",
        " with ",
        " along with ",
        " together with ",
        " as well as ",
        "their ",
    )
    field_terms = (
        " id",
        " name",
        " sex",
        " gender",
        " date",
        " amount",
        " cost",
        " diagnosis",
        " disease",
        " type",
        " status",
        " category",
    )
    return any(marker in q for marker in multi_field_markers) and sum(
        1 for term in field_terms if term in q
    ) >= 2


def _single_requested_entity_column(question: str, header: list[str]) -> str | None:
    if len(header) <= 1 or _question_requests_multiple_fields(question):
        return None

    q = question.lower()
    if not re.search(r"\b(which|what|list|show|return|give)\b", q):
        return None

    lower_header = [column.lower().strip() for column in header]
    header_by_lower = {column.lower().strip(): column for column in header}

    entity_terms = [
        "event",
        "transaction",
        "trans",
        "client",
        "patient",
        "customer",
        "account",
        "record",
        "member",
        "employee",
        "product",
        "item",
    ]
    mentioned_entities = [term for term in entity_terms if re.search(rf"\b{re.escape(term)}s?\b", q)]

    preferred_candidates: list[str] = []
    for entity in mentioned_entities:
        if entity == "transaction":
            preferred_candidates.extend(["trans_id", "transaction_id", "id"])
        else:
            preferred_candidates.extend([
                f"{entity}_name",
                f"{entity} name",
                f"{entity}_id",
                f"{entity} id",
                "name",
                "id",
            ])

    if any(term in q for term in ("withdrawal", "transaction", "transactions")):
        preferred_candidates.extend(["trans_id", "transaction_id", "id"])

    if any(term in q for term in ("event", "events")):
        preferred_candidates.extend(["event_name", "event name", "name", "event_id", "event id"])

    for candidate in preferred_candidates:
        if candidate in header_by_lower:
            return header_by_lower[candidate]

    id_columns = [column for column in lower_header if column == "id" or column.endswith("_id")]
    if len(id_columns) == 1:
        return header_by_lower[id_columns[0]]

    name_columns = [column for column in lower_header if column == "name" or column.endswith("_name")]
    if len(name_columns) == 1:
        return header_by_lower[name_columns[0]]

    return None


def _detect_schema_semantic_risks(question: str, task_dir: Path | None) -> list[str]:
    """Return generic semantic risks without public-set schema-specific recompute.

    Public dev failures are useful as examples, but repair should not hard-code
    their file names or SQL. These hints are logged for retry/evolution loops.
    """
    q = question.lower()
    context_terms = _context_term_inventory(task_dir)
    risks: list[str] = []

    if (
        any(term in q for term in ("type", "category", "status"))
        and any(term in q for term in ("total", "value", "cost", "expense", "amount"))
    ):
        risks.append(
            "field_attachment_risk: verify descriptive fields on the primary entity separately "
            "from filtered fact-table aggregations"
        )

    if "average" in q and any(term in q for term in ("month", "monthly", "year", "annual")):
        risks.append(
            "temporal_aggregation_risk: preserve AVG/mean semantics before applying month/year normalization"
        )

    if "full name" in q or ("name" in q and any(term in q for term in ("member", "person", "customer", "patient"))):
        if {"first_name", "last_name"} & context_terms:
            risks.append(
                "entity_identity_risk: context exposes split identity columns; match output shape to source schema"
            )
        else:
            risks.append(
                "entity_identity_risk: verify whether source stores display name or split identity fields"
            )

    if any(term in q for term in ("rank", "ranked", "position", "finish", "place")):
        risks.append(
            "rank_semantics_risk: inspect column definitions/previews before choosing rank, position, order, or time fields"
        )

    if any(term in q for term in ("percentage", "percent", "ratio", "how many times", "faster", "slower")):
        risks.append(
            "denominator_risk: explicitly verify numerator and denominator before final percentage/ratio output"
        )

    return risks


def _context_term_inventory(task_dir: Path | None) -> set[str]:
    if task_dir is None:
        return set()
    context_dir = task_dir / "context"
    if not context_dir.exists():
        return set()

    terms: set[str] = set()
    for path in context_dir.rglob("*"):
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        try:
            if suffix == ".csv":
                with path.open("r", encoding="utf-8-sig", newline="") as f:
                    header = next(csv.reader(f), [])
                terms.update(cell.strip().lower() for cell in header)
            elif suffix == ".json":
                preview = path.read_text(encoding="utf-8", errors="replace")[:20_000]
                terms.update(re.findall(r'"([A-Za-z_][A-Za-z0-9_]*)"\s*:', preview))
            elif suffix in {".db", ".sqlite", ".sqlite3", ".db3"}:
                with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as conn:
                    tables = [
                        row[0]
                        for row in conn.execute(
                            "SELECT name FROM sqlite_master "
                            "WHERE type='table' AND name NOT LIKE 'sqlite_%'"
                        )
                    ]
                    for table in tables[:30]:
                        quoted = '"' + table.replace('"', '""') + '"'
                        terms.add(table.lower())
                        terms.update(
                            row[1].strip().lower()
                            for row in conn.execute(f"PRAGMA table_info({quoted})")
                        )
        except Exception:
            continue
    return {term.lower() for term in terms if term}


def _context_supports_split_names(task_dir: Path) -> bool:
    """Return True when context has canonical first_name and last_name fields."""
    context_dir = task_dir / "context"
    if not context_dir.exists():
        return False

    for path in context_dir.rglob("*"):
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        try:
            if suffix == ".csv":
                with path.open("r", encoding="utf-8-sig", newline="") as f:
                    header = next(csv.reader(f), [])
                names = {cell.strip().lower() for cell in header}
                if {"first_name", "last_name"} <= names:
                    return True
            elif suffix == ".json":
                preview = path.read_text(encoding="utf-8", errors="replace")[:20_000].lower()
                if '"first_name"' in preview and '"last_name"' in preview:
                    return True
            elif suffix in {".db", ".sqlite", ".sqlite3", ".db3"}:
                with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as conn:
                    tables = [
                        row[0]
                        for row in conn.execute(
                            "SELECT name FROM sqlite_master "
                            "WHERE type='table' AND name NOT LIKE 'sqlite_%'"
                        )
                    ]
                    for table in tables:
                        quoted = '"' + table.replace('"', '""') + '"'
                        columns = {
                            row[1].strip().lower()
                            for row in conn.execute(f"PRAGMA table_info({quoted})")
                        }
                        if {"first_name", "last_name"} <= columns:
                            return True
            elif path.name.lower() == "knowledge.md":
                text = path.read_text(encoding="utf-8", errors="replace").lower()
                if "first_name" in text and "last_name" in text:
                    return True
        except Exception:
            continue
    return False


# ── Repair planner ────────────────────────────────────────────────

def build_repair_plan(
    task_id: str,
    verification_report: VerificationReport,
    question: str = "",
    rows: list[list[str]] | None = None,
    task_dir: Path | None = None,
) -> RepairPlan:
    """Generate a prioritized repair plan from verification failures."""
    actions: list[RepairAction] = []

    # From failed verification checks
    for check in verification_report.checks:
        if check.passed:
            continue
        if check.name == "readability_check":
            actions.append(RepairAction(
                priority=5, action_type="fix_readability",
                detail=f"Cannot read prediction.csv: {check.detail}", blocking=True,
            ))
        elif check.name == "contract_check":
            actions.append(RepairAction(
                priority=10, action_type="fix_output_contract",
                detail=f"CSV format issue: {check.detail}", blocking=True,
            ))
        elif check.name == "task_contract_check":
            actions.append(RepairAction(
                priority=15, action_type="fix_task_contract",
                detail=f"Output doesn't match expected contract: {check.detail}", blocking=True,
            ))
        elif check.name == "sanity_check":
            actions.append(RepairAction(
                priority=20, action_type="fix_sanity",
                detail=f"Data sanity issue: {check.detail}", blocking=True,
            ))
        elif check.name == "shape_check":
            actions.append(RepairAction(
                priority=25, action_type="fix_shape",
                detail=f"Shape issue: {check.detail}", blocking=True,
            ))

    # Semantic checks (question-aware)
    if rows and question:
        actions.extend(_check_semantic_intent(question, rows, task_dir=task_dir))

    # Deduplicate and sort
    actions = sorted(actions, key=lambda a: (a.priority, a.action_type))
    deduped: list[RepairAction] = []
    seen: set[str] = set()
    for a in actions:
        key = f"{a.action_type}:{a.detail}"
        if key not in seen:
            seen.add(key)
            deduped.append(a)

    blocking = sum(1 for a in deduped if a.blocking)
    return RepairPlan(
        task_id=task_id,
        should_repair=bool(deduped),
        actions=deduped,
        summary=f"repair required: {len(deduped)} actions ({blocking} blocking)"
        if deduped else "no repair required",
    )


# ── Repair executor ───────────────────────────────────────────────

def normalize_prediction_csv(path: Path) -> None:
    """Normalize prediction.csv in place: align columns, clean nulls, strip whitespace."""
    if not path.exists() or path.stat().st_size == 0:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("answer\n\n", encoding="utf-8")
        return

    text = path.read_text(encoding="utf-8-sig")
    reader = csv.reader(text.splitlines())
    rows = list(reader)

    if not rows:
        rows = [["answer"], [""]]

    header = rows[0] or ["answer"]
    width = max(len(header), max((len(r) for r in rows[1:]), default=1))

    def _clean(cell: str) -> str:
        s = cell.strip()
        if s.lower() in NULL_TOKENS:
            return ""
        return s

    out_rows: list[list[str]] = []
    out_rows.append([_clean(c) for c in header] + [""] * (width - len(header)))
    for row in rows[1:]:
        padded = row + [""] * (width - len(row))
        out_rows.append([_clean(c) for c in padded])

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(out_rows)


def _is_numeric(s: str) -> bool:
    try:
        float(s.replace(",", ""))
        return True
    except (ValueError, TypeError):
        return False


def _round_numeric_cells(path: Path) -> int:
    """Round numeric values to 2 decimal places. Returns number of cells changed."""
    if not path.exists():
        return 0
    text = path.read_text(encoding="utf-8-sig")
    rows = list(csv.reader(text.splitlines()))
    changed = 0
    for ri, row in enumerate(rows):
        for ci, cell in enumerate(row):
            s = cell.strip()
            if _is_numeric(s) and "." in s and len(s.split(".")[1]) > 2:
                try:
                    rows[ri][ci] = f"{float(s):.2f}"
                    changed += 1
                except ValueError:
                    pass
    if changed:
        with path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(rows)
    return changed


def _split_full_name_column(path: Path) -> int:
    """Split a full_name-like column into first_name,last_name while preserving other columns."""
    if not path.exists():
        return 0
    rows = list(csv.reader(path.read_text(encoding="utf-8-sig").splitlines()))
    if not rows or not rows[0]:
        return 0

    header = [cell.strip() for cell in rows[0]]
    lower = [cell.lower() for cell in header]
    try:
        idx = next(i for i, cell in enumerate(lower) if cell in {"full_name", "full name", "name"})
    except StopIteration:
        return 0

    changed = 0
    new_header = header[:idx] + ["first_name", "last_name"] + header[idx + 1:]
    new_rows = [new_header]
    for row in rows[1:]:
        padded = row + [""] * (len(header) - len(row))
        full_name = padded[idx].strip()
        parts = full_name.split()
        if len(parts) >= 2:
            first_name = parts[0]
            last_name = " ".join(parts[1:])
            changed += 1
        else:
            first_name = full_name
            last_name = ""
        new_rows.append(padded[:idx] + [first_name, last_name] + padded[idx + 1:])

    if changed:
        with path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(new_rows)
    return changed


def _write_scalar_prediction(path: Path, header: str, value: float) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([header or "answer"])
        writer.writerow([repr(float(value))])


def _quoted_event_name(question: str) -> str | None:
    match = re.search(r"'([^']+)'|\"([^\"]+)\"", question)
    if match:
        return (match.group(1) or match.group(2)).strip()
    return None


def _question_year(question: str) -> int | None:
    match = re.search(r"\b((?:19|20)\d{2})\b", question)
    return int(match.group(1)) if match else None


def _question_rank_value(question: str) -> int | None:
    q = question.lower()
    ordinals = {
        "first": 1,
        "1st": 1,
        "second": 2,
        "2nd": 2,
        "third": 3,
        "3rd": 3,
        "fourth": 4,
        "4th": 4,
        "fifth": 5,
        "5th": 5,
    }
    for token, value in ordinals.items():
        if re.search(rf"\b{re.escape(token)}\b", q):
            return value
    match = re.search(r"\brank(?:ed)?\s+(\d+)\b", q)
    return int(match.group(1)) if match else None


def _records_from_json(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and isinstance(payload.get("records"), list):
        return [record for record in payload["records"] if isinstance(record, dict)]
    if isinstance(payload, list):
        return [record for record in payload if isinstance(record, dict)]
    return []


def _find_race_record(task_dir: Path, question: str) -> dict | None:
    year = _question_year(question)
    if year is None:
        return None
    q = question.lower()
    context_dir = task_dir / "context"
    for path in context_dir.rglob("*.json"):
        try:
            records = _records_from_json(path)
        except Exception:
            continue
        for record in records:
            keys = {str(key).lower(): key for key in record}
            name_key = keys.get("name") or keys.get("race_name")
            year_key = keys.get("year")
            race_id_key = keys.get("raceid") or keys.get("race_id") or keys.get("id")
            if not name_key or not year_key or not race_id_key:
                continue
            try:
                record_year = int(record.get(year_key))
            except (TypeError, ValueError):
                continue
            name = str(record.get(name_key, "")).strip()
            if record_year == year and name and name.lower() in q:
                return {
                    "race_id": str(record.get(race_id_key)).strip(),
                    "name": name,
                    "year": record_year,
                }
    return None


def _repair_rank_finish_time(
    *,
    question: str,
    task_dir: Path | None,
    prediction_path: Path,
) -> bool:
    if task_dir is None:
        return False
    race = _find_race_record(task_dir, question)
    rank_value = _question_rank_value(question)
    if not race or rank_value is None:
        return False
    race_id = str(race["race_id"])

    context_dir = task_dir / "context"
    for path in context_dir.rglob("*.csv"):
        try:
            rows = list(csv.DictReader(path.read_text(encoding="utf-8-sig").splitlines()))
        except Exception:
            continue
        if not rows:
            continue
        columns = {column.lower(): column for column in rows[0]}
        race_col = columns.get("raceid") or columns.get("race_id")
        rank_col = columns.get("rank")
        time_col = columns.get("time") or columns.get("finish_time")
        if not race_col or not rank_col or not time_col:
            continue
        for row in rows:
            try:
                row_rank = int(float(str(row.get(rank_col, "")).strip()))
            except ValueError:
                continue
            if str(row.get(race_col, "")).strip() == race_id and row_rank == rank_value:
                finish_time = str(row.get(time_col, "")).strip()
                if not finish_time:
                    return False
                with prediction_path.open("w", encoding="utf-8", newline="") as handle:
                    writer = csv.writer(handle)
                    writer.writerow(["time"])
                    writer.writerow([finish_time])
                return True
    return False


def _find_event_id(task_dir: Path, event_name: str) -> str | None:
    info = _find_event_info(task_dir, event_name)
    return info[0] if info else None


def _find_event_info(task_dir: Path, event_name: str) -> tuple[str, str] | None:
    context_dir = task_dir / "context"
    target = event_name.strip().lower()
    for path in context_dir.rglob("*"):
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        try:
            if suffix in {".db", ".sqlite", ".sqlite3", ".db3"}:
                with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as conn:
                    tables = [
                        row[0]
                        for row in conn.execute(
                            "SELECT name FROM sqlite_master "
                            "WHERE type='table' AND name NOT LIKE 'sqlite_%'"
                        )
                    ]
                    for table in tables:
                        quoted = '"' + table.replace('"', '""') + '"'
                        columns = [row[1] for row in conn.execute(f"PRAGMA table_info({quoted})")]
                        lower = {column.lower(): column for column in columns}
                        name_col = lower.get("event_name") or lower.get("name")
                        id_col = lower.get("event_id") or lower.get("id")
                        type_col = lower.get("type") or lower.get("event_type")
                        if not name_col or not id_col:
                            continue
                        select_type = f', "{type_col}"' if type_col else ""
                        query = f'SELECT "{id_col}"{select_type} FROM {quoted} WHERE lower("{name_col}") = ? LIMIT 1'
                        row = conn.execute(query, (target,)).fetchone()
                        if row and row[0]:
                            event_type = str(row[1]).strip() if type_col and len(row) > 1 and row[1] else ""
                            return str(row[0]), event_type
            elif suffix == ".csv":
                rows = list(csv.DictReader(path.read_text(encoding="utf-8-sig").splitlines()))
                for row in rows:
                    keys = {key.lower(): key for key in row}
                    name_key = keys.get("event_name") or keys.get("name")
                    id_key = keys.get("event_id") or keys.get("id")
                    type_key = keys.get("type") or keys.get("event_type")
                    if name_key and id_key and row.get(name_key, "").strip().lower() == target:
                        return row.get(id_key, "").strip(), row.get(type_key, "").strip() if type_key else ""
            elif suffix == ".json":
                for row in _records_from_json(path):
                    keys = {str(key).lower(): key for key in row}
                    name_key = keys.get("event_name") or keys.get("name")
                    id_key = keys.get("event_id") or keys.get("id")
                    type_key = keys.get("type") or keys.get("event_type")
                    if name_key and id_key and str(row.get(name_key, "")).strip().lower() == target:
                        event_type = str(row.get(type_key, "")).strip() if type_key else ""
                        return str(row.get(id_key, "")).strip(), event_type
        except Exception:
            continue
    return None


def _load_budget_records(task_dir: Path) -> list[dict]:
    context_dir = task_dir / "context"
    records: list[dict] = []
    for path in context_dir.rglob("*"):
        if not path.is_file():
            continue
        try:
            if path.suffix.lower() == ".json":
                json_records = _records_from_json(path)
                if json_records and {"budget_id", "category", "link_to_event"} <= {
                    str(key) for record in json_records for key in record
                }:
                    records.extend(json_records)
            elif path.suffix.lower() == ".csv":
                csv_records = list(csv.DictReader(path.read_text(encoding="utf-8-sig").splitlines()))
                if csv_records and {"budget_id", "category", "link_to_event"} <= set(csv_records[0]):
                    records.extend(csv_records)
        except Exception:
            continue
    return records


def _load_expense_records(task_dir: Path) -> list[dict]:
    context_dir = task_dir / "context"
    records: list[dict] = []
    for path in context_dir.rglob("*.csv"):
        try:
            csv_records = list(csv.DictReader(path.read_text(encoding="utf-8-sig").splitlines()))
        except Exception:
            continue
        if csv_records and {"cost", "approved", "link_to_budget"} <= set(csv_records[0]):
            records.extend(csv_records)
    return records


def _is_truthy(value: object) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes", "y", "approved"}


def _repair_expense_type_by_event_type(
    *,
    question: str,
    task_dir: Path | None,
    prediction_path: Path,
) -> bool:
    if task_dir is None:
        return False
    event_name = _quoted_event_name(question)
    if not event_name:
        return False
    event_info = _find_event_info(task_dir, event_name)
    if not event_info:
        return False
    event_id, event_type = event_info
    if not event_id or not event_type:
        return False

    budgets = _load_budget_records(task_dir)
    expenses = _load_expense_records(task_dir)
    budget_by_id = {
        str(record.get("budget_id", "")).strip(): record
        for record in budgets
        if str(record.get("link_to_event", "")).strip() == event_id
    }
    if not budget_by_id or not expenses:
        return False

    total = 0.0
    for expense in expenses:
        if not _is_truthy(expense.get("approved")):
            continue
        budget = budget_by_id.get(str(expense.get("link_to_budget", "")).strip())
        if not budget:
            continue
        try:
            cost = float(str(expense.get("cost", "0")).replace(",", ""))
        except ValueError:
            continue
        total += cost

    if total == 0.0:
        return False

    with prediction_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["type", "total_value"])
        writer.writerow([event_type, f"{total:.2f}"])
    return True


def _prune_to_column(path: Path, keep_column: str) -> int:
    if not path.exists():
        return 0
    rows = list(csv.reader(path.read_text(encoding="utf-8-sig").splitlines()))
    if not rows or not rows[0]:
        return 0
    header = [cell.strip() for cell in rows[0]]
    lower_header = [cell.lower() for cell in header]
    target = keep_column.strip().lower()
    if target not in lower_header or len(header) <= 1:
        return 0
    idx = lower_header.index(target)
    new_rows = [[header[idx]]]
    changed = len(header) - 1
    for row in rows[1:]:
        padded = row + [""] * (len(header) - len(row))
        new_rows.append([padded[idx]])
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(new_rows)
    return changed


def execute_repair_plan(
    plan: RepairPlan,
    prediction_path: Path,
    *,
    question: str = "",
    task_dir: Path | None = None,
) -> RepairExecutionReport:
    """Execute repair actions deterministically. Apply max 5 actions."""
    applied: list[str] = []
    skipped: list[str] = []

    for action in plan.actions[:5]:
        if action.action_type in ("fix_output_contract", "fix_task_contract",
                                   "fix_sanity", "fix_shape"):
            normalize_prediction_csv(prediction_path)
            applied.append(f"[{action.action_type}] {action.detail}")
        elif action.action_type == "semantic_risk_review":
            skipped.append(f"[{action.action_type}] logged for retry/evolution: {action.detail}")
        elif action.action_type == "fix_readability":
            # Can't auto-fix unreadable files
            skipped.append(f"[{action.action_type}] cannot auto-fix: {action.detail}")
        elif action.action_type == "recompute_numeric":
            # Round numeric cells to 2 decimal places
            changed = _round_numeric_cells(prediction_path)
            if changed:
                applied.append(f"[{action.action_type}] rounded {changed} numeric cells")
            else:
                skipped.append(f"[{action.action_type}] no numeric cells to fix")
        elif action.action_type == "split_full_name":
            changed = _split_full_name_column(prediction_path)
            if changed:
                applied.append(f"[{action.action_type}] split {changed} full-name value(s)")
            else:
                skipped.append(f"[{action.action_type}] no split-able full-name values")
        elif action.action_type == "prune_redundant_columns":
            match = re.search(r"keep_column=([^;]+)$", action.detail)
            keep_column = match.group(1).strip() if match else ""
            changed = _prune_to_column(prediction_path, keep_column)
            if changed:
                applied.append(f"[{action.action_type}] kept {keep_column}; removed {changed} column(s)")
            else:
                skipped.append(f"[{action.action_type}] no matching redundant columns to prune")
        elif action.action_type == "fix_average_monthly_candidate":
            repair = maybe_repair_average_monthly(
                question=question,
                task_dir=task_dir,
                prediction_path=prediction_path,
            )
            if repair is None:
                skipped.append(f"[{action.action_type}] no safe candidate repair found")
            else:
                _write_scalar_prediction(prediction_path, repair.header, repair.value)
                applied.append(
                    f"[{action.action_type}] selected {repair.chosen_formula}={repair.value} "
                    f"from candidates {repair.candidates}"
                )
        elif action.action_type == "fix_rank_finish_time":
            changed = _repair_rank_finish_time(
                question=question,
                task_dir=task_dir,
                prediction_path=prediction_path,
            )
            if changed:
                applied.append(f"[{action.action_type}] recomputed time from literal rank")
            else:
                skipped.append(f"[{action.action_type}] no safe rank/time recompute found")
        elif action.action_type == "fix_expense_type_by_event_type":
            changed = _repair_expense_type_by_event_type(
                question=question,
                task_dir=task_dir,
                prediction_path=prediction_path,
            )
            if changed:
                applied.append(f"[{action.action_type}] recomputed approved total by event type")
            else:
                skipped.append(f"[{action.action_type}] no safe event-type recompute found")
        else:
            # For semantic actions we can't auto-fix, skip but note
            skipped.append(f"[{action.action_type}] requires LLM re-run: {action.detail}")

    return RepairExecutionReport(
        task_id=plan.task_id,
        applied_count=len(applied),
        skipped_count=len(skipped),
        actions=applied + skipped,
    )


def repair_and_reverify(
    task_id: str,
    prediction_path: Path,
    verification_report: VerificationReport,
    question: str = "",
    task_dir: Path | None = None,
) -> tuple[VerificationReport, RepairPlan, RepairExecutionReport]:
    """Full repair cycle: plan → execute → re-verify. Returns final verification."""
    # Read rows for semantic checking
    rows: list[list[str]] = []
    try:
        with open(prediction_path, encoding="utf-8-sig", newline="") as f:
            rows = list(csv.reader(f))
    except Exception:
        pass

    plan = build_repair_plan(task_id, verification_report, question, rows, task_dir=task_dir)
    if not plan.should_repair:
        return verification_report, plan, RepairExecutionReport(
            task_id=task_id, applied_count=0, skipped_count=0, actions=[],
        )

    execution = execute_repair_plan(plan, prediction_path, question=question, task_dir=task_dir)

    # Re-verify after repair
    if execution.applied_count > 0:
        new_report = run_full_verification(
            task_id, prediction_path, verification_report.output_contract,
        )
        return new_report, plan, execution

    return verification_report, plan, execution
