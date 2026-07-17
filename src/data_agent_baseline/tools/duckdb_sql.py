from __future__ import annotations

import re
import sqlite3
from datetime import date, datetime, time
from decimal import Decimal
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd


SUPPORTED_SUFFIXES = {".csv", ".json", ".db", ".sqlite", ".sqlite3", ".db3"}
MAX_AUTO_JSON_BYTES = 50_000_000
MAX_NESTED_ITEMS = 20
MAX_NESTED_DEPTH = 3
MAX_STRING_CHARS = 4_000
MAX_SQLITE_ROWS_PER_TABLE = 1_000_000


def _quote_identifier(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def _sql_literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _safe_table_name(path: Path) -> str:
    raw = "__".join(path.with_suffix("").parts)
    safe = re.sub(r"\W+", "_", raw).strip("_").lower()
    if not safe:
        safe = "table"
    if safe[0].isdigit():
        safe = f"t_{safe}"
    return safe[:96]


def _is_read_only(sql: str) -> bool:
    cleaned = re.sub(r"/\*.*?\*/", "", sql, flags=re.DOTALL)
    cleaned = re.sub(r"--.*?$", "", cleaned, flags=re.MULTILINE).lstrip().lower()
    return cleaned.startswith(("select", "with"))


def _relative_path(context_root: Path, path: Path) -> str:
    return path.relative_to(context_root).as_posix()


def _json_safe(value: Any) -> Any:
    if isinstance(value, (date, datetime, time)):
        return value.isoformat()
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, dict):
        return {
            str(key): _json_safe_nested(child, depth=1)
            for key, child in list(value.items())[:MAX_NESTED_ITEMS]
        }
    if isinstance(value, (list, tuple)):
        return _json_safe_sequence(value, depth=1)
    if isinstance(value, str) and len(value) > MAX_STRING_CHARS:
        return value[:MAX_STRING_CHARS] + f"... <truncated {len(value) - MAX_STRING_CHARS} chars>"
    return value


def _json_safe_nested(value: Any, *, depth: int) -> Any:
    if isinstance(value, (date, datetime, time)):
        return value.isoformat()
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, str):
        if len(value) > MAX_STRING_CHARS:
            return value[:MAX_STRING_CHARS] + f"... <truncated {len(value) - MAX_STRING_CHARS} chars>"
        return value
    if isinstance(value, dict):
        if depth >= MAX_NESTED_DEPTH:
            return f"<nested dict with {len(value)} keys omitted>"
        return {
            str(key): _json_safe_nested(child, depth=depth + 1)
            for key, child in list(value.items())[:MAX_NESTED_ITEMS]
        }
    if isinstance(value, (list, tuple)):
        return _json_safe_sequence(value, depth=depth)
    return value


def _json_safe_sequence(value: list[Any] | tuple[Any, ...], *, depth: int) -> list[Any]:
    if depth >= MAX_NESTED_DEPTH:
        return [f"<nested sequence with {len(value)} items omitted>"]
    items = [_json_safe_nested(child, depth=depth + 1) for child in list(value)[:MAX_NESTED_ITEMS]]
    if len(value) > MAX_NESTED_ITEMS:
        items.append(f"... <{len(value) - MAX_NESTED_ITEMS} more items>")
    return items


def _register_context_tables(conn: duckdb.DuckDBPyConnection, context_root: Path) -> list[dict[str, Any]]:
    tables: list[dict[str, Any]] = []
    used_names: set[str] = set()

    def _unique_name(preferred: str) -> str:
        table_name = preferred
        if table_name in used_names:
            suffix = 2
            base = table_name[:90]
            while f"{base}_{suffix}" in used_names:
                suffix += 1
            table_name = f"{base}_{suffix}"
        used_names.add(table_name)
        return table_name

    for path in sorted(context_root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue

        rel_path = _relative_path(context_root, path)
        table_name = _unique_name(_safe_table_name(Path(rel_path)))

        info: dict[str, Any] = {
            "table": table_name,
            "path": rel_path,
            "type": path.suffix.lower().lstrip("."),
            "size_bytes": path.stat().st_size,
        }

        try:
            literal_path = _sql_literal(path.as_posix())
            identifier = _quote_identifier(table_name)
            if path.suffix.lower() == ".csv":
                conn.execute(
                    f"""
                    CREATE VIEW {identifier} AS
                    SELECT * FROM read_csv_auto(
                        {literal_path},
                        header=true,
                        ignore_errors=true,
                        sample_size=20480
                    )
                    """
                )
                info["registered"] = True
            elif path.suffix.lower() == ".json" and path.stat().st_size <= MAX_AUTO_JSON_BYTES:
                conn.execute(
                    f"""
                    CREATE VIEW {identifier} AS
                    SELECT * FROM read_json_auto({literal_path})
                    """
                )
                info["registered"] = True
                try:
                    records_table_name = f"{table_name}_records"
                    records_identifier = _quote_identifier(records_table_name)
                    conn.execute(
                        f"""
                        CREATE VIEW {records_identifier} AS
                        SELECT unnest(records, recursive := true)
                        FROM read_json_auto({literal_path})
                        """
                    )
                    info["records_table"] = records_table_name
                except Exception:
                    pass
            elif path.suffix.lower() in {".db", ".sqlite", ".sqlite3", ".db3"}:
                sqlite_tables = _register_sqlite_tables(
                    conn=conn,
                    path=path,
                    base_table_name=table_name,
                    used_names=used_names,
                )
                info["registered"] = bool(sqlite_tables)
                info["sqlite_tables"] = sqlite_tables
            else:
                info["registered"] = False
                info["reason"] = "large JSON skipped for automatic DuckDB registration"
        except Exception as exc:  # noqa: BLE001
            info["registered"] = False
            info["error"] = str(exc)
        tables.append(info)

    return tables


def _register_sqlite_tables(
    *,
    conn: duckdb.DuckDBPyConnection,
    path: Path,
    base_table_name: str,
    used_names: set[str],
) -> list[dict[str, Any]]:
    """Load SQLite tables into DuckDB so cross-source joins stay in one SQL tool."""

    def _unique_name(preferred: str) -> str:
        table_name = preferred
        if table_name in used_names:
            suffix = 2
            base = table_name[:90]
            while f"{base}_{suffix}" in used_names:
                suffix += 1
            table_name = f"{base}_{suffix}"
        used_names.add(table_name)
        return table_name

    registered: list[dict[str, Any]] = []
    with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as sqlite_conn:
        sqlite_conn.row_factory = sqlite3.Row
        table_names = [
            row[0]
            for row in sqlite_conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
            )
        ]
        for sqlite_table in table_names:
            quoted_sqlite_table = '"' + sqlite_table.replace('"', '""') + '"'
            row_count = sqlite_conn.execute(
                f"SELECT COUNT(*) FROM {quoted_sqlite_table}"
            ).fetchone()[0]
            if row_count > MAX_SQLITE_ROWS_PER_TABLE:
                registered.append({
                    "sqlite_table": sqlite_table,
                    "registered": False,
                    "reason": f"row count {row_count} exceeds safe load limit",
                })
                continue

            df = pd.read_sql_query(f"SELECT * FROM {quoted_sqlite_table}", sqlite_conn)
            preferred = _safe_table_name(Path(f"{base_table_name}_{sqlite_table}"))
            duckdb_table = _unique_name(preferred)
            conn.register(duckdb_table, df)

            aliases: list[str] = []
            simple_alias = _safe_table_name(Path(sqlite_table))
            if simple_alias not in used_names:
                used_names.add(simple_alias)
                conn.execute(
                    f"CREATE VIEW {_quote_identifier(simple_alias)} AS "
                    f"SELECT * FROM {_quote_identifier(duckdb_table)}"
                )
                aliases.append(simple_alias)

            registered.append({
                "sqlite_table": sqlite_table,
                "duckdb_table": duckdb_table,
                "aliases": aliases,
                "row_count": row_count,
                "registered": True,
            })
    return registered


def execute_data_sql(context_root: Path, sql: str, *, limit: int = 200) -> dict[str, Any]:
    """Run a read-only DuckDB query over CSV/JSON files inside the task context."""
    if not _is_read_only(sql):
        raise ValueError("Only SELECT/WITH SQL statements are allowed.")

    resolved_context_root = context_root.resolve()
    with duckdb.connect(database=":memory:") as conn:
        conn.execute("PRAGMA disable_progress_bar")
        registered_tables = _register_context_tables(conn, resolved_context_root)
        cursor = conn.execute(sql)
        columns = [item[0] for item in cursor.description or []]
        rows = cursor.fetchmany(limit + 1)

    truncated = len(rows) > limit
    limited_rows = rows[:limit]
    return {
        "columns": columns,
        "rows": [[_json_safe(value) for value in row] for row in limited_rows],
        "row_count": len(limited_rows),
        "truncated": truncated,
        "registered_tables": registered_tables,
    }
