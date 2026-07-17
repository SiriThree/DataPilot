"""Context Profiler: auto-scan task context and produce structured metadata.

Generates a context_profile.json-like dict covering all data sources
(CSV, JSON, SQLite, docs) with schema, statistics, and previews.
"""

from __future__ import annotations

import csv
import json
import re
import sqlite3
from pathlib import Path
from typing import Any

from data_agent_baseline.tools.pdf_tools import profile_pdf

LARGE_FILE_BYTES = 50_000_000
MAX_PROFILE_ROWS = 5_000
MAX_JSON_PARSE_BYTES = 5_000_000
MAX_DUCKDB_JSON_BYTES = 50_000_000


# ── CSV helpers ────────────────────────────────────────────────────

def _duckdb_table_name(path: Path) -> str:
    raw = "__".join(path.with_suffix("").parts)
    safe = re.sub(r"\W+", "_", raw).strip("_").lower()
    if not safe:
        safe = "table"
    if safe[0].isdigit():
        safe = f"t_{safe}"
    return safe[:96]


def _count_lines_fast(path: Path) -> int:
    count = 0
    last = b""
    with path.open("rb") as f:
        while chunk := f.read(1024 * 1024):
            count += chunk.count(b"\n")
            last = chunk
    if last and not last.endswith(b"\n"):
        count += 1
    return count


def _profile_csv(path: Path, max_preview_rows: int = 5) -> dict[str, Any]:
    file_size = path.stat().st_size
    preview_rows: list[list[str]] = []
    sampled_rows: list[list[str]] = []

    with open(path, newline="", encoding="utf-8-sig", errors="replace") as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration:
            return {
                "type": "csv",
                "path": str(path),
                "columns": [],
                "row_count": 0,
                "file_size_bytes": file_size,
                "error": "empty or unreadable",
            }
        for row_index, row in enumerate(reader):
            if row_index < max_preview_rows:
                preview_rows.append(row)
            if row_index < MAX_PROFILE_ROWS:
                sampled_rows.append(row)
                if file_size > LARGE_FILE_BYTES and row_index + 1 >= MAX_PROFILE_ROWS:
                    break
            elif file_size <= LARGE_FILE_BYTES:
                sampled_rows.append(row)

    row_count_estimate: int | None = None
    if file_size > LARGE_FILE_BYTES:
        row_count_estimate = max(_count_lines_fast(path) - 1, 0)
        num_rows = None
        stats_truncated = True
    else:
        num_rows = len(sampled_rows)
        stats_truncated = False
    num_cols = len(header)
    sampled_count = len(sampled_rows)

    # Column-level statistics
    columns: list[dict[str, Any]] = []
    for ci in range(num_cols):
        vals = [r[ci] if ci < len(r) else "" for r in sampled_rows]
        non_empty = [v for v in vals if v.strip() != ""]
        missing = sampled_count - len(non_empty)

        numeric_vals: list[float] = []
        for v in non_empty:
            try:
                numeric_vals.append(float(v))
            except ValueError:
                pass

        col_info: dict[str, Any] = {
            "index": ci,
            "name": header[ci] if ci < len(header) else f"col_{ci}",
            "missing_count": missing,
            "missing_rate": round(missing / sampled_count, 3) if sampled_count > 0 else 0.0,
            "sample_values": [r[ci] if ci < len(r) else "" for r in preview_rows],
        }

        if numeric_vals:
            col_info["numeric"] = {
                "count": len(numeric_vals),
                "min": round(min(numeric_vals), 4),
                "max": round(max(numeric_vals), 4),
                "avg": round(sum(numeric_vals) / len(numeric_vals), 4),
            }

        columns.append(col_info)

    return {
        "type": "csv",
        "path": str(path),
        "file_size_bytes": file_size,
        "duckdb_table": _duckdb_table_name(path),
        "columns": columns,
        "row_count": num_rows,
        "row_count_estimate": row_count_estimate,
        "column_count": num_cols,
        "profiled_rows": sampled_count,
        "stats_truncated": stats_truncated,
        "preview_rows": [header] + preview_rows,
    }


# ── JSON helpers ───────────────────────────────────────────────────

def _extract_json_paths(obj: Any, prefix: str = "$") -> list[str]:
    """Extract JSONPath-like key paths."""
    paths: list[str] = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            safe_key = k
            paths.append(f"{prefix}.{safe_key}")
            if isinstance(v, (dict, list)):
                paths.extend(_extract_json_paths(v, f"{prefix}.{safe_key}"))
            else:
                paths.append(f"{prefix}.{safe_key} (value)")
    elif isinstance(obj, list):
        paths.append(f"{prefix}[0..{len(obj) - 1}]")
        if obj and isinstance(obj[0], (dict, list)):
            paths.extend(_extract_json_paths(obj[0], f"{prefix}[0]"))
    return paths


def _profile_json(path: Path) -> dict[str, Any]:
    file_size = path.stat().st_size
    if file_size > MAX_JSON_PARSE_BYTES:
        text = path.read_text(encoding="utf-8", errors="replace")
        table_match = re.search(r'"table"\s*:\s*"([^"]+)"', text[:20_000])
        has_records = bool(re.search(r'"records"\s*:', text[:20_000]))
        with path.open("r", encoding="utf-8", errors="replace") as f:
            preview = f.read(2000)
        return {
            "type": "json",
            "path": str(path),
            "file_size_bytes": file_size,
            "top_type": "records_wrapper" if table_match and has_records else "large_json",
            "records_table": table_match.group(1) if table_match and has_records else None,
            "preview": preview,
            "truncated": True,
            "note": "large JSON was not fully parsed during profiling",
        }

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        return {"type": "json", "path": str(path), "error": str(exc)}

    key_paths = _extract_json_paths(data)

    def _nesting_depth(obj: Any) -> int:
        if isinstance(obj, dict):
            children = [_nesting_depth(v) for v in obj.values()]
            return 1 + max(children) if children else 1
        if isinstance(obj, list):
            children = [_nesting_depth(item) for item in obj]
            return 1 + max(children) if children else 1
        return 1

    depth = _nesting_depth(data)

    # For list-of-dicts, detect common fields
    records_table = None
    if isinstance(data, dict) and isinstance(data.get("records"), list):
        records_table = str(data.get("table") or path.stem)
    elif (
        isinstance(data, list)
        and data
        and isinstance(data[0], dict)
        and isinstance(data[0].get("records"), list)
    ):
        records_table = str(data[0].get("table") or path.stem)

    if isinstance(data, list) and data and all(isinstance(item, dict) for item in data):
        common_keys: set[str] = set(data[0].keys())
        for item in data[1:]:
            common_keys &= set(item.keys())
        array_field_summary = f"array of {len(data)} dicts, common keys: {sorted(common_keys)}"
    else:
        array_field_summary = None

    # Serialize a size-bounded preview
    preview_raw = json.dumps(data, ensure_ascii=False, default=str)
    preview = preview_raw[:2000]

    return {
        "type": "json",
        "path": str(path),
        "top_type": type(data).__name__,
        "nesting_depth": depth,
        "records_table": records_table,
        "key_paths": key_paths[:50],
        "array_field_summary": array_field_summary,
        "preview": preview,
    }


# ── SQLite helpers ─────────────────────────────────────────────────

def _profile_sqlite(path: Path) -> dict[str, Any]:
    tables: list[dict[str, Any]] = []
    try:
        with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")
            table_names = [row[0] for row in cursor.fetchall()]

            for tname in table_names:
                # Use double-quoted identifiers (SQL standard) for table names
                quoted = f'"{tname}"'
                cursor.execute(f"PRAGMA table_info({quoted})")
                col_info = [
                    {
                        "name": row["name"],
                        "type": row["type"],
                        "notnull": bool(row["notnull"]),
                        "pk": bool(row["pk"]),
                    }
                    for row in cursor.fetchall()
                ]

                cursor.execute(f"PRAGMA foreign_key_list({quoted})")
                fks = [
                    {"from": row["from"], "table": row["table"], "to": row["to"]}
                    for row in cursor.fetchall()
                ]

                try:
                    cursor.execute(f"SELECT * FROM {quoted} LIMIT 3")
                    sample_rows = [list(row) for row in cursor.fetchall()]
                    column_names = [desc[0] for desc in cursor.description] if cursor.description else []
                except sqlite3.Error:
                    sample_rows = []
                    column_names = []

                try:
                    cursor.execute(f"SELECT COUNT(*) FROM {quoted}")
                    row_count = cursor.fetchone()[0]
                except sqlite3.Error:
                    row_count = -1

                tables.append({
                    "name": tname,
                    "duckdb_table": _duckdb_table_name(
                        Path(f"{path.with_suffix('').name}_{tname}")
                    ),
                    "columns": col_info,
                    "foreign_keys": fks,
                    "row_count": row_count,
                    "column_names": column_names,
                    "sample_rows": sample_rows,
                })
    except sqlite3.Error as exc:
        return {"type": "sqlite", "path": str(path), "error": str(exc)}

    return {
        "type": "sqlite",
        "path": str(path),
        "table_count": len(tables),
        "tables": tables,
    }


# ── Doc helpers ────────────────────────────────────────────────────

def _extract_ranges(text: str) -> list[dict[str, str]]:
    """Extract numeric ranges from document text.

    Detects patterns like:
    - "X is between A and B" / "X ranges from A to B"
    - "normal X: A-B" / "normal X is A to B"
    - "X > A" / "X < B" / "X >= A"
    - "A-B" / "A to B" (with preceding label)
    """
    ranges: list[dict[str, str]] = []
    # Pattern 1: "LABEL between N and M" or "LABEL from N to M"
    p1 = re.finditer(
        r'(\w+(?:\s+\w+){0,2})\s+(?:is\s+)?(?:between\s+)?'
        r'(\d+\.?\d*)\s*(?:-|–|to|and)\s*(\d+\.?\d*)',
        text, re.IGNORECASE,
    )
    for m in p1:
        label = m.group(1).strip().lower()
        lo, hi = float(m.group(2)), float(m.group(3))
        if lo > hi:
            lo, hi = hi, lo
        ranges.append({"label": label, "type": "range", "lo": str(lo), "hi": str(hi)})

    # Pattern 2: "LABEL (>= | > | < | <=) N"
    p2 = re.finditer(
        r'(\w+(?:\s+\w+){0,2})\s*(>=?|<=?)\s*(\d+\.?\d*)',
        text, re.IGNORECASE,
    )
    for m in p2:
        label = m.group(1).strip().lower()
        op = m.group(2)
        val = float(m.group(3))
        if op in (">", ">="):
            ranges.append({"label": label, "type": "gt", "val": str(val), "op": op})
        else:
            ranges.append({"label": label, "type": "lt", "val": str(val), "op": op})

    # Pattern 3: Headings or bullet points with "normal LABEL" followed by numbers
    p3 = re.finditer(
        r'normal\s+(\w+(?:\s+\w+){0,2})\s*(?:level|range|value)?\s*[:=-]?\s*'
        r'(\d+\.?\d*)\s*(?:-|–|to)\s*(\d+\.?\d*)',
        text, re.IGNORECASE,
    )
    for m in p3:
        label = f"normal {m.group(1).strip().lower()}"
        lo, hi = float(m.group(2)), float(m.group(3))
        if lo > hi:
            lo, hi = hi, lo
        ranges.append({"label": label, "type": "normal_range", "lo": str(lo), "hi": str(hi)})

    # Deduplicate
    seen: set[str] = set()
    unique: list[dict[str, str]] = []
    for r in ranges:
        key = f"{r['label']}:{r.get('lo','')}-{r.get('hi','')}{r.get('op','')}{r.get('val','')}"
        if key not in seen:
            seen.add(key)
            unique.append(r)
    return unique[:20]


def _profile_doc(path: Path) -> dict[str, Any]:
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return {"type": "doc", "path": str(path), "error": "unicode decode failed"}

    # Simple chunking by paragraphs
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    # Detect headings (Markdown-style)
    headings: list[str] = []
    for line in text.split("\n"):
        stripped = line.strip()
        if stripped.startswith("#"):
            headings.append(stripped)

    # Keyword extraction: collect capitalized words / acronyms
    words = text.split()
    potential_terms: set[str] = set()
    for w in words:
        clean = w.strip(".,;:()[]{}'\"!?").replace("-", "_")
        if clean and clean[0].isupper() and len(clean) > 1:
            potential_terms.add(clean)

    # Numeric range extraction (for threshold-dependent questions)
    extracted_ranges = _extract_ranges(text)

    return {
        "type": "doc",
        "path": str(path),
        "size_chars": len(text),
        "paragraph_count": len(paragraphs),
        "headings": headings[:20],
        "potential_terms": sorted(potential_terms)[:30],
        "numeric_ranges": extracted_ranges,
        "preview": text[:3000],
    }


# ── Unified profile entry point ────────────────────────────────────

def profile_context(context_dir: Path) -> dict[str, Any]:
    """Scan a task context directory and return a structured profile."""
    files: dict[str, Any] = {}
    extensions = {
        ".csv": _profile_csv,
        ".json": _profile_json,
        ".md": _profile_doc,
        ".txt": _profile_doc,
        ".pdf": profile_pdf,
    }

    for fpath in sorted(context_dir.rglob("*")):
        if not fpath.is_file():
            continue
        rel_path = fpath.relative_to(context_dir).as_posix()
        suffix = fpath.suffix.lower()

        if suffix in extensions:
            files[rel_path] = extensions[suffix](fpath)
            if suffix == ".csv" or (
                suffix == ".json" and fpath.stat().st_size <= MAX_DUCKDB_JSON_BYTES
            ):
                table_name = _duckdb_table_name(Path(rel_path))
                records_table = files[rel_path].get("records_table")
                files[rel_path]["duckdb_table"] = (
                    f"{table_name}_records" if records_table else table_name
                )
                if records_table:
                    files[rel_path]["raw_duckdb_table"] = table_name
        elif suffix in (".sqlite", ".db", ".sqlite3", ".db3"):
            files[rel_path] = _profile_sqlite(fpath)
            base_table = _duckdb_table_name(Path(rel_path))
            for table in files[rel_path].get("tables", []):
                table["duckdb_table"] = _duckdb_table_name(Path(f"{base_table}_{table['name']}"))
        else:
            files[rel_path] = {"type": "unknown", "path": rel_path, "size": fpath.stat().st_size}

    # Build a human-readable summary
    summary_parts: list[str] = []
    for rel_path, info in files.items():
        ftype = info.get("type", "?")
        if ftype == "csv":
            row_count = info.get("row_count")
            if row_count is None:
                row_count = f"~{info.get('row_count_estimate')} estimated"
            summary_parts.append(
                f"CSV '{rel_path}': {info.get('column_count')} cols x {row_count} rows "
                f"(DuckDB table: {info.get('duckdb_table')})"
            )
        elif ftype == "json":
            table = info.get("duckdb_table")
            table_hint = f" (DuckDB table: {table})" if table else ""
            records_hint = (
                f", records table: {info.get('records_table')}"
                if info.get("records_table") else ""
            )
            summary_parts.append(
                f"JSON '{rel_path}': {info.get('top_type')}, depth {info.get('nesting_depth')}"
                f"{records_hint}{table_hint}"
            )
        elif ftype == "sqlite":
            table_parts = [
                f"{t['name']} as {t.get('duckdb_table')}"
                for t in info.get("tables", [])
            ]
            summary_parts.append(
                f"SQLite '{rel_path}': {info.get('table_count')} tables "
                f"({', '.join(table_parts)})"
            )
        elif ftype == "doc":
            summary_parts.append(
                f"Doc '{rel_path}': {info.get('size_chars')} chars, {info.get('paragraph_count')} paragraphs"
            )
        elif ftype == "pdf":
            summary_parts.append(
                f"PDF '{rel_path}': {info.get('page_count')} pages, "
                f"{info.get('table_count_estimate')} tables estimated"
            )
        else:
            summary_parts.append(f"File '{rel_path}': {ftype}")

    return {
        "context_dir": str(context_dir),
        "file_count": len(files),
        "summary": "\n".join(summary_parts),
        "files": files,
    }
