from __future__ import annotations

from pathlib import Path
from typing import Any

import pdfplumber
from pypdf import PdfReader


MAX_PAGE_TEXT_CHARS = 2_000
MAX_TOTAL_TEXT_CHARS = 8_000
MAX_MATCHES = 10
MAX_TABLE_ROWS = 50
MAX_TABLES = 10


def read_pdf(path: Path, *, max_pages: int = 5, max_chars: int = MAX_TOTAL_TEXT_CHARS) -> dict[str, Any]:
    """Extract bounded text from the first pages of a PDF."""
    reader = PdfReader(str(path))
    page_count = len(reader.pages)
    pages: list[dict[str, Any]] = []
    total_chars = 0

    for index, page in enumerate(reader.pages[:max_pages], start=1):
        text = page.extract_text() or ""
        remaining = max(max_chars - total_chars, 0)
        if remaining <= 0:
            break
        excerpt = text[: min(MAX_PAGE_TEXT_CHARS, remaining)]
        total_chars += len(excerpt)
        pages.append({
            "page": index,
            "text": excerpt,
            "truncated": len(text) > len(excerpt),
        })

    return {
        "path": str(path),
        "page_count": page_count,
        "pages_returned": len(pages),
        "pages": pages,
        "truncated": page_count > max_pages or total_chars >= max_chars,
    }


def search_pdf(path: Path, query: str, *, max_matches: int = MAX_MATCHES) -> dict[str, Any]:
    """Search PDF page text for a keyword or phrase."""
    normalized_query = query.lower().strip()
    if not normalized_query:
        raise ValueError("query must not be empty")

    reader = PdfReader(str(path))
    matches: list[dict[str, Any]] = []
    for page_index, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        lower_text = text.lower()
        start = lower_text.find(normalized_query)
        if start < 0:
            continue
        excerpt_start = max(start - 180, 0)
        excerpt_end = min(start + len(normalized_query) + 320, len(text))
        matches.append({
            "page": page_index,
            "excerpt": text[excerpt_start:excerpt_end].strip(),
        })
        if len(matches) >= max_matches:
            break

    return {
        "path": str(path),
        "query": query,
        "match_count": len(matches),
        "matches": matches,
    }


def extract_pdf_tables(path: Path, *, max_tables: int = MAX_TABLES, max_rows: int = MAX_TABLE_ROWS) -> dict[str, Any]:
    """Extract bounded tables from a PDF using pdfplumber."""
    tables: list[dict[str, Any]] = []
    with pdfplumber.open(path) as pdf:
        for page_index, page in enumerate(pdf.pages, start=1):
            page_tables = page.extract_tables() or []
            for table_index, table in enumerate(page_tables, start=1):
                rows = table[:max_rows]
                tables.append({
                    "page": page_index,
                    "table_index": table_index,
                    "row_count_returned": len(rows),
                    "row_count_estimate": len(table),
                    "truncated": len(table) > len(rows),
                    "rows": rows,
                })
                if len(tables) >= max_tables:
                    return {
                        "path": str(path),
                        "table_count_returned": len(tables),
                        "truncated": True,
                        "tables": tables,
                    }

    return {
        "path": str(path),
        "table_count_returned": len(tables),
        "truncated": False,
        "tables": tables,
    }


def profile_pdf(path: Path, *, max_preview_pages: int = 3) -> dict[str, Any]:
    """Build a compact PDF profile for context profiling."""
    reader = PdfReader(str(path))
    page_count = len(reader.pages)
    page_previews: list[dict[str, Any]] = []
    for index, page in enumerate(reader.pages[:max_preview_pages], start=1):
        text = page.extract_text() or ""
        page_previews.append({
            "page": index,
            "text_preview": text[:1_000],
            "text_chars": len(text),
        })

    table_count_estimate = 0
    try:
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages[:max_preview_pages]:
                table_count_estimate += len(page.extract_tables() or [])
    except Exception:
        table_count_estimate = -1

    return {
        "type": "pdf",
        "path": str(path),
        "file_size_bytes": path.stat().st_size,
        "page_count": page_count,
        "preview_pages": page_previews,
        "table_count_estimate": table_count_estimate,
        "truncated": page_count > max_preview_pages,
    }
