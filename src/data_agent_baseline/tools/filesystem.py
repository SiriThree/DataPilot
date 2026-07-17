from __future__ import annotations

import csv
import json
from pathlib import Path

from data_agent_baseline.benchmark.schema import PublicTask


def resolve_context_path(task: PublicTask, relative_path: str) -> Path:
    candidate = (task.context_dir / relative_path).resolve()
    context_root = task.context_dir.resolve()
    if context_root not in candidate.parents and candidate != context_root:
        raise ValueError(f"Path escapes context dir: {relative_path}")
    if not candidate.exists():
        raise FileNotFoundError(f"Missing context asset: {relative_path}")
    return candidate


def list_context_tree(task: PublicTask, *, max_depth: int = 4) -> dict[str, object]:
    entries: list[dict[str, object]] = []

    def walk(path: Path, depth: int) -> None:
        if depth > max_depth:
            return
        for child in sorted(path.iterdir(), key=lambda item: (item.is_file(), item.name)):
            rel_path = child.relative_to(task.context_dir).as_posix()
            entries.append(
                {
                    "path": rel_path,
                    "kind": "dir" if child.is_dir() else "file",
                    "size": child.stat().st_size if child.is_file() else None,
                }
            )
            if child.is_dir():
                walk(child, depth + 1)

    walk(task.context_dir, 1)
    return {
        "root": str(task.context_dir),
        "entries": entries,
    }


def read_csv_preview(task: PublicTask, relative_path: str, *, max_rows: int = 20) -> dict[str, object]:
    path = resolve_context_path(task, relative_path)
    large_file = path.stat().st_size > 50_000_000
    rows: list[list[str]] = []
    row_count: int | None = 0
    with path.open(newline="", encoding="utf-8-sig", errors="replace") as handle:
        reader = csv.reader(handle)
        try:
            header = next(reader)
        except StopIteration:
            return {
                "path": relative_path,
                "columns": [],
                "rows": [],
                "row_count": 0,
                "truncated": False,
            }
        seen_rows = 0
        for row in reader:
            if seen_rows < max_rows:
                rows.append(row)
            seen_rows += 1
            if large_file and seen_rows >= max_rows:
                row_count = None
                break
        if row_count is not None:
            row_count = seen_rows

    if not header:
        return {
            "path": relative_path,
            "columns": [],
            "rows": [],
            "row_count": row_count,
            "truncated": row_count is None or row_count > max_rows,
        }

    return {
        "path": relative_path,
        "columns": header,
        "rows": rows,
        "row_count": row_count,
        "truncated": row_count is None or row_count > max_rows,
        "note": "large CSV preview only; use execute_data_sql for full-table analysis"
        if row_count is None else "",
    }


def read_json_preview(task: PublicTask, relative_path: str, *, max_chars: int = 4000) -> dict[str, object]:
    path = resolve_context_path(task, relative_path)
    size = path.stat().st_size
    if size > 5_000_000:
        with path.open("r", encoding="utf-8", errors="replace") as f:
            preview = f.read(max_chars)
        return {
            "path": relative_path,
            "preview": preview,
            "truncated": True,
            "size_bytes": size,
            "note": "large JSON preview only; use execute_python or execute_data_sql for analysis",
        }

    payload = json.loads(path.read_text(encoding="utf-8"))
    preview = json.dumps(payload, ensure_ascii=False, indent=2)
    return {
        "path": relative_path,
        "preview": preview[:max_chars],
        "truncated": len(preview) > max_chars,
    }


def read_doc_preview(task: PublicTask, relative_path: str, *, max_chars: int = 4000) -> dict[str, object]:
    path = resolve_context_path(task, relative_path)
    text = path.read_text(errors="replace")
    return {
        "path": relative_path,
        "preview": text[:max_chars],
        "truncated": len(text) > max_chars,
    }
