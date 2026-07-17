from __future__ import annotations

import csv
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import duckdb

from data_agent_baseline.tools.duckdb_sql import _quote_identifier, _register_context_tables


AVERAGE_MONTHLY_RE = re.compile(
    r"\b(avg|average|mean)\b.*\bmonthly\b|\bmonthly\b.*\b(avg|average|mean)\b",
    re.IGNORECASE,
)
YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")
NUMERIC_TYPES = ("INT", "DOUBLE", "FLOAT", "DECIMAL", "REAL", "NUMERIC", "BIGINT", "HUGEINT")
DATE_COLUMN_HINTS = ("date", "month", "yearmonth", "year_month", "period")


@dataclass(frozen=True, slots=True)
class AverageMonthlyRepair:
    header: str
    value: float
    chosen_formula: str
    candidates: dict[str, float]
    reason: str


@dataclass(frozen=True, slots=True)
class TableColumn:
    table: str
    name: str
    type_name: str


@dataclass(frozen=True, slots=True)
class FilterCandidate:
    table: str
    column: str
    value: str


def _normalize(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()


def _word_in_question(value: str, question: str) -> bool:
    normalized_value = _normalize(value)
    if not normalized_value:
        return False
    normalized_question = f" {_normalize(question)} "
    return f" {normalized_value} " in normalized_question


def _safe_float(value: Any) -> float | None:
    try:
        result = float(str(value).replace(",", ""))
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _read_prediction_scalar(prediction_path: Path) -> tuple[str, float | None]:
    try:
        rows = list(csv.reader(prediction_path.read_text(encoding="utf-8-sig").splitlines()))
    except Exception:
        return "answer", None
    header = rows[0][0].strip() if rows and rows[0] else "answer"
    value = None
    if len(rows) >= 2 and rows[1]:
        value = _safe_float(rows[1][0])
    return header or "answer", value


def _table_names(registered: list[dict[str, Any]]) -> list[str]:
    names: list[str] = []
    for item in registered:
        table = item.get("table")
        if isinstance(table, str):
            names.append(table)
        records_table = item.get("records_table")
        if isinstance(records_table, str):
            names.append(records_table)
        for sqlite_table in item.get("sqlite_tables", []) or []:
            if not isinstance(sqlite_table, dict) or not sqlite_table.get("registered"):
                continue
            duckdb_table = sqlite_table.get("duckdb_table")
            if isinstance(duckdb_table, str):
                names.append(duckdb_table)
            for alias in sqlite_table.get("aliases", []) or []:
                if isinstance(alias, str):
                    names.append(alias)
    seen: set[str] = set()
    out: list[str] = []
    for name in names:
        if name not in seen:
            seen.add(name)
            out.append(name)
    return out


def _columns(conn: duckdb.DuckDBPyConnection, table: str) -> list[TableColumn]:
    try:
        rows = conn.execute(f"PRAGMA table_info({_quote_identifier(table)})").fetchall()
    except Exception:
        return []
    return [TableColumn(table=table, name=str(row[1]), type_name=str(row[2]).upper()) for row in rows]


def _is_numeric_column(column: TableColumn) -> bool:
    return any(token in column.type_name for token in NUMERIC_TYPES)


def _choose_fact_columns(
    conn: duckdb.DuckDBPyConnection,
    tables: list[str],
    question: str,
) -> tuple[str, str, str] | None:
    best: tuple[int, str, str, str] | None = None
    for table in tables:
        cols = _columns(conn, table)
        date_cols = [
            col.name for col in cols
            if any(hint in col.name.lower().replace("_", "") for hint in DATE_COLUMN_HINTS)
        ]
        if not date_cols:
            continue
        for col in cols:
            if not _is_numeric_column(col):
                continue
            if col.name.lower() in {"id", "rowid"} or col.name.lower().endswith("id"):
                continue
            score = 0
            if _word_in_question(col.name, question):
                score += 10
            if "consumption" in question.lower() and "consumption" in col.name.lower():
                score += 10
            if score <= 0:
                continue
            table_score = score + (3 if table.startswith("csv__") else 0)
            candidate = (table_score, table, col.name, date_cols[0])
            if best is None or candidate[0] > best[0]:
                best = candidate
    if best is None:
        return None
    _, table, metric, date_col = best
    return table, metric, date_col


def _find_filter_candidate(
    conn: duckdb.DuckDBPyConnection,
    tables: list[str],
    question: str,
    *,
    fact_table: str,
    metric_col: str,
    date_col: str,
) -> FilterCandidate | None:
    ignored = {metric_col.lower(), date_col.lower()}
    best: tuple[int, FilterCandidate] | None = None
    for table in tables:
        for col in _columns(conn, table):
            if col.name.lower() in ignored:
                continue
            if _is_numeric_column(col) and not col.name.lower().endswith("id"):
                continue
            try:
                rows = conn.execute(
                    f"""
                    SELECT DISTINCT CAST({_quote_identifier(col.name)} AS VARCHAR) AS value
                    FROM {_quote_identifier(table)}
                    WHERE {_quote_identifier(col.name)} IS NOT NULL
                    LIMIT 500
                    """
                ).fetchall()
            except Exception:
                continue
            for (value,) in rows:
                raw = str(value).strip()
                if not raw or len(raw) > 80:
                    continue
                if not _word_in_question(raw, question):
                    continue
                score = 10
                if table != fact_table:
                    score += 2
                if col.name.lower() in {"segment", "category", "type", "status"}:
                    score += 3
                candidate = FilterCandidate(table=table, column=col.name, value=raw)
                if best is None or score > best[0]:
                    best = (score, candidate)
    return best[1] if best else None


def _common_join_key(
    conn: duckdb.DuckDBPyConnection,
    left_table: str,
    right_table: str,
) -> tuple[str, str] | None:
    left = _columns(conn, left_table)
    right = _columns(conn, right_table)
    right_by_lower = {col.name.lower(): col.name for col in right}
    preferred: list[tuple[str, str]] = []
    fallback: list[tuple[str, str]] = []
    for col in left:
        match = right_by_lower.get(col.name.lower())
        if not match:
            continue
        pair = (col.name, match)
        if col.name.lower().endswith("id"):
            preferred.append(pair)
        else:
            fallback.append(pair)
    return (preferred or fallback or [None])[0]


def _where_sql(
    *,
    fact_alias: str,
    date_col: str,
    years: list[str],
    filter_alias: str | None,
    filter_candidate: FilterCandidate | None,
) -> str:
    predicates: list[str] = []
    if years:
        year = years[0]
        predicates.append(
            f"CAST({fact_alias}.{_quote_identifier(date_col)} AS VARCHAR) LIKE '{year}%'"
        )
    if filter_alias and filter_candidate is not None:
        escaped_value = filter_candidate.value.replace("'", "''")
        predicates.append(
            "LOWER(CAST("
            f"{filter_alias}.{_quote_identifier(filter_candidate.column)} AS VARCHAR)) = "
            f"LOWER('{escaped_value}')"
        )
    return "WHERE " + " AND ".join(predicates) if predicates else ""


def _from_sql(
    conn: duckdb.DuckDBPyConnection,
    *,
    fact_table: str,
    filter_candidate: FilterCandidate | None,
) -> tuple[str, str | None, str | None]:
    if filter_candidate is None or filter_candidate.table == fact_table:
        return f"FROM {_quote_identifier(fact_table)} m", None, None
    join_key = _common_join_key(conn, fact_table, filter_candidate.table)
    if join_key is None:
        return f"FROM {_quote_identifier(fact_table)} m", None, None
    left_key, right_key = join_key
    return (
        f"FROM {_quote_identifier(fact_table)} m "
        f"JOIN {_quote_identifier(filter_candidate.table)} f "
        f"ON m.{_quote_identifier(left_key)} = f.{_quote_identifier(right_key)}",
        "f",
        left_key,
    )


def _fetch_scalar(conn: duckdb.DuckDBPyConnection, sql: str) -> float | None:
    try:
        row = conn.execute(sql).fetchone()
    except Exception:
        return None
    if not row:
        return None
    return _safe_float(row[0])


def _close(a: float | None, b: float | None) -> bool:
    if a is None or b is None:
        return False
    return math.isclose(a, b, rel_tol=1e-6, abs_tol=1e-6)


def maybe_repair_average_monthly(
    *,
    question: str,
    task_dir: Path | None,
    prediction_path: Path,
) -> AverageMonthlyRepair | None:
    if task_dir is None or not AVERAGE_MONTHLY_RE.search(question):
        return None
    context_dir = task_dir / "context"
    if not context_dir.exists():
        return None

    header, predicted_value = _read_prediction_scalar(prediction_path)
    years = YEAR_RE.findall(question)

    with duckdb.connect(database=":memory:") as conn:
        conn.execute("PRAGMA disable_progress_bar")
        registered = _register_context_tables(conn, context_dir.resolve())
        tables = _table_names(registered)
        fact = _choose_fact_columns(conn, tables, question)
        if fact is None:
            return None
        fact_table, metric_col, date_col = fact
        filter_candidate = _find_filter_candidate(
            conn, tables, question, fact_table=fact_table, metric_col=metric_col, date_col=date_col,
        )
        from_clause, filter_alias, entity_key = _from_sql(
            conn, fact_table=fact_table, filter_candidate=filter_candidate,
        )
        where = _where_sql(
            fact_alias="m",
            date_col=date_col,
            years=years,
            filter_alias=filter_alias,
            filter_candidate=filter_candidate,
        )
        metric_expr = f"TRY_CAST(m.{_quote_identifier(metric_col)} AS DOUBLE)"
        candidates: dict[str, float] = {}

        formulas = {
            "avg_metric_div_12": f"SELECT AVG({metric_expr}) / 12.0 {from_clause} {where}",
            "sum_metric_div_12": f"SELECT SUM({metric_expr}) / 12.0 {from_clause} {where}",
            "sum_metric_div_distinct_month": (
                f"SELECT SUM({metric_expr}) / NULLIF(COUNT(DISTINCT CAST(m.{_quote_identifier(date_col)} AS VARCHAR)), 0) "
                f"{from_clause} {where}"
            ),
        }
        if entity_key:
            formulas["avg_entity_annual_div_12"] = (
                f"WITH per_entity AS ("
                f"SELECT m.{_quote_identifier(entity_key)} AS entity_id, SUM({metric_expr}) AS annual_metric "
                f"{from_clause} {where} GROUP BY m.{_quote_identifier(entity_key)}"
                f") SELECT AVG(annual_metric) / 12.0 FROM per_entity"
            )

        for name, sql in formulas.items():
            value = _fetch_scalar(conn, sql)
            if value is not None:
                candidates[name] = value

    chosen = candidates.get("avg_metric_div_12")
    if chosen is None:
        return None
    if _close(predicted_value, chosen):
        return None

    predicted_matches_other = any(
        name != "avg_metric_div_12" and _close(predicted_value, value)
        for name, value in candidates.items()
    )
    if predicted_value is not None and not predicted_matches_other:
        return None

    return AverageMonthlyRepair(
        header=header,
        value=chosen,
        chosen_formula="avg_metric_div_12",
        candidates=candidates,
        reason=(
            "average monthly question over monthly fact records: selected AVG(metric)/12 "
            "after comparing executable candidates"
        ),
    )
