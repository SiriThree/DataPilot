from __future__ import annotations

from pathlib import Path

from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle

from data_agent_baseline.benchmark.schema import PublicTask, TaskAssets, TaskRecord
from data_agent_baseline.tools.pdf_tools import extract_pdf_tables, read_pdf, search_pdf
from data_agent_baseline.tools.profiler import profile_context
from data_agent_baseline.tools.registry import create_default_tool_registry


def _write_pdf(path: Path) -> None:
    doc = SimpleDocTemplate(str(path), pagesize=letter)
    table = Table(
        [
            ["Region", "Sales"],
            ["East", "15"],
            ["West", "20"],
        ]
    )
    table.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), 1, "black"),
    ]))
    story = [table]
    doc.build(story)


def _make_pdf_task(tmp_path: Path) -> PublicTask:
    task_dir = tmp_path / "task_pdf_1"
    context_dir = task_dir / "context"
    context_dir.mkdir(parents=True)
    _write_pdf(context_dir / "report.pdf")
    return PublicTask(
        record=TaskRecord(
            task_id="task_pdf_1",
            difficulty="easy",
            question="Find East sales from the PDF report.",
        ),
        assets=TaskAssets(task_dir=task_dir, context_dir=context_dir),
    )


def test_profile_context_detects_pdf(tmp_path: Path) -> None:
    task = _make_pdf_task(tmp_path)

    profile = profile_context(task.context_dir)

    pdf_info = profile["files"]["report.pdf"]
    assert pdf_info["type"] == "pdf"
    assert pdf_info["page_count"] == 1
    assert "PDF 'report.pdf'" in profile["summary"]


def test_pdf_text_search_and_table_extraction(tmp_path: Path) -> None:
    task = _make_pdf_task(tmp_path)
    pdf_path = task.context_dir / "report.pdf"

    read_result = read_pdf(pdf_path)
    search_result = search_pdf(pdf_path, "East")
    table_result = extract_pdf_tables(pdf_path)

    assert read_result["page_count"] == 1
    assert search_result["match_count"] >= 1
    assert "East" in search_result["matches"][0]["excerpt"]
    assert table_result["table_count_returned"] >= 1
    assert table_result["tables"][0]["rows"][0] == ["Region", "Sales"]


def test_default_registry_exposes_pdf_tools(tmp_path: Path) -> None:
    task = _make_pdf_task(tmp_path)
    registry = create_default_tool_registry()

    read_result = registry.execute(task, "read_pdf", {"path": "report.pdf"})
    search_result = registry.execute(task, "search_pdf", {"path": "report.pdf", "query": "West"})
    tables_result = registry.execute(task, "extract_pdf_tables", {"path": "report.pdf"})

    assert read_result.ok is True
    assert search_result.ok is True
    assert search_result.content["match_count"] >= 1
    assert tables_result.ok is True
    assert tables_result.content["table_count_returned"] >= 1
