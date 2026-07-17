from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from data_agent_baseline.benchmark.schema import AnswerTable, PublicTask
from data_agent_baseline.tools.doc_evidence_table import extract_doc_evidence_table
from data_agent_baseline.tools.doc_extract import extract_doc_records
from data_agent_baseline.tools.filesystem import (
    list_context_tree,
    read_csv_preview,
    read_doc_preview,
    read_json_preview,
    resolve_context_path,
)
from data_agent_baseline.tools.pdf_tools import (
    extract_pdf_tables,
    read_pdf,
    search_pdf,
)
from data_agent_baseline.tools.duckdb_sql import execute_data_sql
from data_agent_baseline.tools.python_exec import execute_python_code
from data_agent_baseline.tools.profiler import profile_context
from data_agent_baseline.tools.sqlite import execute_read_only_sql, inspect_sqlite_schema
from data_agent_baseline.tools.stateful_python import execute_stateful_python, remove_interpreter
from data_agent_baseline.tools.threshold_grounding import ground_thresholds

EXECUTE_PYTHON_TIMEOUT_SECONDS = 30


@dataclass(frozen=True, slots=True)
class ToolSpec:
    name: str
    description: str
    input_schema: dict[str, Any]


@dataclass(frozen=True, slots=True)
class ToolExecutionResult:
    ok: bool
    content: dict[str, Any]
    is_terminal: bool = False
    answer: AnswerTable | None = None


ToolHandler = Callable[[PublicTask, dict[str, Any]], ToolExecutionResult]


def _list_context(task: PublicTask, action_input: dict[str, Any]) -> ToolExecutionResult:
    max_depth = int(action_input.get("max_depth", 4))
    return ToolExecutionResult(ok=True, content=list_context_tree(task, max_depth=max_depth))


def _read_csv(task: PublicTask, action_input: dict[str, Any]) -> ToolExecutionResult:
    path = str(action_input["path"])
    max_rows = int(action_input.get("max_rows", 20))
    return ToolExecutionResult(ok=True, content=read_csv_preview(task, path, max_rows=max_rows))


def _read_json(task: PublicTask, action_input: dict[str, Any]) -> ToolExecutionResult:
    path = str(action_input["path"])
    max_chars = int(action_input.get("max_chars", 4000))
    return ToolExecutionResult(ok=True, content=read_json_preview(task, path, max_chars=max_chars))


def _read_doc(task: PublicTask, action_input: dict[str, Any]) -> ToolExecutionResult:
    path = str(action_input["path"])
    max_chars = int(action_input.get("max_chars", 4000))
    return ToolExecutionResult(ok=True, content=read_doc_preview(task, path, max_chars=max_chars))


def _read_pdf(task: PublicTask, action_input: dict[str, Any]) -> ToolExecutionResult:
    path = resolve_context_path(task, str(action_input["path"]))
    max_pages = int(action_input.get("max_pages", 5))
    max_chars = int(action_input.get("max_chars", 8000))
    return ToolExecutionResult(ok=True, content=read_pdf(path, max_pages=max_pages, max_chars=max_chars))


def _search_pdf(task: PublicTask, action_input: dict[str, Any]) -> ToolExecutionResult:
    path = resolve_context_path(task, str(action_input["path"]))
    query = str(action_input["query"])
    max_matches = int(action_input.get("max_matches", 10))
    return ToolExecutionResult(ok=True, content=search_pdf(path, query, max_matches=max_matches))


def _extract_pdf_tables(task: PublicTask, action_input: dict[str, Any]) -> ToolExecutionResult:
    path = resolve_context_path(task, str(action_input["path"]))
    max_tables = int(action_input.get("max_tables", 10))
    max_rows = int(action_input.get("max_rows", 50))
    return ToolExecutionResult(
        ok=True,
        content=extract_pdf_tables(path, max_tables=max_tables, max_rows=max_rows),
    )


def _inspect_sqlite_schema(task: PublicTask, action_input: dict[str, Any]) -> ToolExecutionResult:
    path = resolve_context_path(task, str(action_input["path"]))
    return ToolExecutionResult(ok=True, content=inspect_sqlite_schema(path))


def _execute_context_sql(task: PublicTask, action_input: dict[str, Any]) -> ToolExecutionResult:
    path = resolve_context_path(task, str(action_input["path"]))
    sql = str(action_input["sql"])
    limit = int(action_input.get("limit", 200))
    return ToolExecutionResult(ok=True, content=execute_read_only_sql(path, sql, limit=limit))


def _execute_data_sql(task: PublicTask, action_input: dict[str, Any]) -> ToolExecutionResult:
    sql = str(action_input["sql"])
    limit = int(action_input.get("limit", 200))
    return ToolExecutionResult(ok=True, content=execute_data_sql(task.context_dir, sql, limit=limit))


def _execute_python(task: PublicTask, action_input: dict[str, Any]) -> ToolExecutionResult:
    code = str(action_input["code"])
    content = execute_python_code(
        context_root=task.context_dir,
        code=code,
        timeout_seconds=EXECUTE_PYTHON_TIMEOUT_SECONDS,
    )
    return ToolExecutionResult(ok=bool(content.get("success")), content=content)


def _execute_python_stateful(task: PublicTask, action_input: dict[str, Any]) -> ToolExecutionResult:
    code = str(action_input["code"])
    content = execute_stateful_python(
        task_id=task.task_id,
        context_root=task.context_dir,
        code=code,
        timeout=EXECUTE_PYTHON_TIMEOUT_SECONDS,
    )
    return ToolExecutionResult(ok=bool(content.get("success")), content=content)


def _profile_context(task: PublicTask, _action_input: dict[str, Any]) -> ToolExecutionResult:
    return ToolExecutionResult(ok=True, content=profile_context(task.context_dir))


def _extract_doc_records(task: PublicTask, action_input: dict[str, Any]) -> ToolExecutionResult:
    path = str(action_input["path"])
    raw_terms = action_input.get("target_terms", [])
    if isinstance(raw_terms, str):
        target_terms = [raw_terms]
    elif isinstance(raw_terms, list) and all(isinstance(term, str) for term in raw_terms):
        target_terms = raw_terms
    else:
        raise ValueError("target_terms must be a string or list of strings")
    entity_hint = str(action_input.get("entity_hint", "patient"))
    max_records = int(action_input.get("max_records", 200))
    return ToolExecutionResult(
        ok=True,
        content=extract_doc_records(
            task.context_dir,
            path,
            target_terms,
            entity_hint=entity_hint,
            max_records=max_records,
        ),
    )


def _extract_doc_evidence_table(task: PublicTask, action_input: dict[str, Any]) -> ToolExecutionResult:
    path = str(action_input["path"])
    raw_fields = action_input.get("fields", [])
    if isinstance(raw_fields, str):
        fields = [raw_fields]
    elif isinstance(raw_fields, list) and all(isinstance(field, str) for field in raw_fields):
        fields = raw_fields
    else:
        raise ValueError("fields must be a string or list of strings")
    entity_hint = str(action_input.get("entity_hint", "entity"))
    max_records = int(action_input.get("max_records", 1000))
    return ToolExecutionResult(
        ok=True,
        content=extract_doc_evidence_table(
            task.context_dir,
            path,
            fields,
            entity_hint=entity_hint,
            max_records=max_records,
        ),
    )


def _ground_thresholds(task: PublicTask, action_input: dict[str, Any]) -> ToolExecutionResult:
    raw_terms = action_input.get("target_terms", [])
    if isinstance(raw_terms, str):
        target_terms = [raw_terms]
    elif isinstance(raw_terms, list) and all(isinstance(term, str) for term in raw_terms):
        target_terms = raw_terms
    else:
        raise ValueError("target_terms must be a string or list of strings")

    raw_population_terms = action_input.get("population_terms", [])
    if isinstance(raw_population_terms, str):
        population_terms = [raw_population_terms]
    elif isinstance(raw_population_terms, list) and all(isinstance(term, str) for term in raw_population_terms):
        population_terms = raw_population_terms
    else:
        raise ValueError("population_terms must be a string or list of strings")

    max_candidates = int(action_input.get("max_candidates", 20))
    return ToolExecutionResult(
        ok=True,
        content=ground_thresholds(
            task.context_dir,
            target_terms,
            population_terms=population_terms,
            max_candidates=max_candidates,
        ),
    )


def _query_json(task: PublicTask, action_input: dict[str, Any]) -> ToolExecutionResult:
    import json as _json
    path = resolve_context_path(task, str(action_input["path"]))
    query = str(action_input.get("query", ""))
    try:
        data = _json.loads(path.read_text(encoding="utf-8"))
    except (_json.JSONDecodeError, UnicodeDecodeError) as exc:
        return ToolExecutionResult(ok=False, content={"error": str(exc)})

    # query can be a dot-separated key path like "data.users.0.name"
    keys = [k.strip() for k in query.split(".") if k.strip()]
    result = data
    for key in keys:
        if isinstance(result, list):
            try:
                idx = int(key)
                result = result[idx]
            except (ValueError, IndexError):
                return ToolExecutionResult(ok=False, content={"error": f"index '{key}' out of range"})
        elif isinstance(result, dict):
            if key not in result:
                return ToolExecutionResult(ok=False, content={"error": f"key '{key}' not found"})
            result = result[key]
        else:
            return ToolExecutionResult(ok=False, content={"error": f"cannot navigate into {type(result).__name__} with key '{key}'"})

    # Serialize bounded output
    result_str = _json.dumps(result, ensure_ascii=False, default=str)
    return ToolExecutionResult(ok=True, content={"path": str(action_input["path"]), "query": query, "result": result_str[:4000]})


def _search_doc(task: PublicTask, action_input: dict[str, Any]) -> ToolExecutionResult:
    path = resolve_context_path(task, str(action_input["path"]))
    query = str(action_input.get("query", "")).lower()
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return ToolExecutionResult(ok=False, content={"error": "cannot read file as text"})

    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    matches: list[dict[str, Any]] = []
    for pi, para in enumerate(paragraphs):
        if query in para.lower():
            matches.append({"paragraph_index": pi, "excerpt": para[:500]})
        if len(matches) >= 5:
            break

    return ToolExecutionResult(ok=True, content={
        "path": str(action_input["path"]),
        "query": str(action_input["query"]),
        "match_count": len(matches),
        "matches": matches,
    })


def _answer(_: PublicTask, action_input: dict[str, Any]) -> ToolExecutionResult:
    columns = action_input.get("columns")
    rows = action_input.get("rows")
    if not isinstance(columns, list) or not columns or not all(isinstance(item, str) for item in columns):
        raise ValueError("answer.columns must be a non-empty list of strings.")
    if not isinstance(rows, list):
        raise ValueError("answer.rows must be a list.")

    normalized_rows: list[list[Any]] = []
    for row in rows:
        if not isinstance(row, list):
            raise ValueError("Each answer row must be a list.")
        if len(row) != len(columns):
            raise ValueError("Each answer row must match the number of columns.")
        normalized_rows.append(list(row))

    answer = AnswerTable(columns=list(columns), rows=normalized_rows)
    return ToolExecutionResult(
        ok=True,
        content={
            "status": "submitted",
            "column_count": len(columns),
            "row_count": len(normalized_rows),
        },
        is_terminal=True,
        answer=answer,
    )


@dataclass(slots=True)
class ToolRegistry:
    specs: dict[str, ToolSpec]
    handlers: dict[str, ToolHandler]
    _action_history: list[str] = None  # type: ignore[assignment]
    _consecutive_errors: int = 0
    _use_stateful_python: bool = False

    def __post_init__(self) -> None:
        if self._action_history is None:
            object.__setattr__(self, "_action_history", [])

    @property
    def use_stateful_python(self) -> bool:
        return self._use_stateful_python

    @use_stateful_python.setter
    def use_stateful_python(self, value: bool) -> None:
        self._use_stateful_python = value
        if value:
            # Replace execute_python handler with stateful version
            self.handlers["execute_python"] = _execute_python_stateful
            self.specs["execute_python"] = ToolSpec(
                name="execute_python",
                description=(
                    "Execute Python code with the task context directory as the working "
                    "directory. **Stateful mode**: variables (including DataFrames) persist "
                    "across calls — you can load data once and reuse it in later steps. "
                    f"The execution timeout is {EXECUTE_PYTHON_TIMEOUT_SECONDS}s. "
                    "Use print() to display results."
                ),
                input_schema={
                    "code": "import pandas as pd\nprint(sorted(os.listdir('.')))",
                },
            )
        else:
            self.handlers["execute_python"] = _execute_python
            self.specs["execute_python"] = ToolSpec(
                name="execute_python",
                description=(
                    "Execute arbitrary Python code with the task context directory as the "
                    "working directory. The tool returns the code's captured stdout as `output`. "
                    f"The execution timeout is fixed at {EXECUTE_PYTHON_TIMEOUT_SECONDS} seconds."
                ),
                input_schema={
                    "code": "import os\nprint(sorted(os.listdir('.')))",
                },
            )

    def cleanup_task(self, task_id: str) -> None:
        """Cleanup per-task resources (e.g., stateful interpreter)."""
        remove_interpreter(task_id)

    def describe_for_prompt(self) -> str:
        lines = []
        for name in sorted(self.specs):
            spec = self.specs[name]
            lines.append(f"- {spec.name}: {spec.description}")
            lines.append(f"  input_schema: {spec.input_schema}")
        return "\n".join(lines)

    def execute(self, task: PublicTask, action: str, action_input: dict[str, Any]) -> ToolExecutionResult:
        if action not in self.handlers:
            raise KeyError(f"Unknown tool: {action}")

        # ── Smart Tool: repetition detection ──
        action_key = self._make_action_key(action, action_input)
        repeat_count = self._count_recent(action_key, window=5)

        if repeat_count >= 2 and action not in ("answer",):
            # Inject a warning into the tool result instead of re-executing blindly
            warning = {
                "repetition_warning": (
                    f"You have called `{action}` with nearly identical input "
                    f"{repeat_count + 1} times in the last 5 steps. "
                    f"This suggests you are stuck in a loop. "
                    f"Consider: (1) using a different tool, (2) reading a different file, "
                    f"or (3) answering with your best estimate based on what you already know."
                ),
                "repetition_count": repeat_count + 1,
            }
            # Still execute (don't block), but prepend the warning
            result = self.handlers[action](task, action_input)
            if result.ok:
                warned_content = dict(warning, **result.content)
                result = ToolExecutionResult(
                    ok=result.ok, content=warned_content,
                    is_terminal=result.is_terminal, answer=result.answer,
                )
            self._action_history.append(action_key)
            if not result.ok:
                self._consecutive_errors += 1
            else:
                self._consecutive_errors = 0
            return result

        self._action_history.append(action_key)
        result = self.handlers[action](task, action_input)
        if not result.ok:
            self._consecutive_errors += 1
        else:
            self._consecutive_errors = 0
        return result

    def _make_action_key(self, action: str, action_input: dict[str, Any]) -> str:
        """Create a normalized key for detecting repeated actions."""
        # For search/read tools, use the query/path as the distinguishing field
        if action in ("search_doc",):
            return f"{action}:{action_input.get('path','')}:{action_input.get('query','')}"
        if action in ("read_csv", "read_json", "read_doc", "read_pdf"):
            return f"{action}:{action_input.get('path','')}"
        if action in ("search_pdf",):
            return f"{action}:{action_input.get('path','')}:{action_input.get('query','')}"
        if action in ("execute_context_sql",):
            return f"{action}:{action_input.get('path','')}:{str(action_input.get('sql',''))[:100]}"
        if action in ("execute_data_sql",):
            return f"{action}:{str(action_input.get('sql',''))[:100]}"
        if action in ("execute_python",):
            return f"{action}:{str(action_input.get('code',''))[:100]}"
        return f"{action}:{str(action_input)[:100]}"

    def _count_recent(self, key: str, window: int = 5) -> int:
        """Count how many times this action key appears in the recent history."""
        return sum(1 for h in self._action_history[-window:] if h == key)

    @property
    def should_force_recovery(self) -> bool:
        """True when the agent has had 3+ consecutive errors and needs a reset."""
        return self._consecutive_errors >= 3


def create_default_tool_registry() -> ToolRegistry:
    specs = {
        "answer": ToolSpec(
            name="answer",
            description="Submit the final answer table. This is the only valid terminating action.",
            input_schema={
                "columns": ["column_name"],
                "rows": [["value_1"]],
            },
        ),
        "execute_context_sql": ToolSpec(
            name="execute_context_sql",
            description="Run a read-only SQL query against a sqlite/db file inside context.",
            input_schema={"path": "relative/path/to/file.sqlite", "sql": "SELECT ...", "limit": 200},
        ),
        "execute_data_sql": ToolSpec(
            name="execute_data_sql",
            description=(
                "Run a read-only DuckDB SQL query over CSV/JSON/SQLite files in context. "
                "This is preferred for large CSV files and joins/aggregations across sources. "
                "profile_context lists DuckDB table names for CSV, JSON, and SQLite tables; "
                "use those table names in SQL."
            ),
            input_schema={"sql": "SELECT ... FROM csv_table_name", "limit": 200},
        ),
        "execute_python": ToolSpec(
            name="execute_python",
            description=(
                "Execute arbitrary Python code with the task context directory as the "
                "working directory. The tool returns the code's captured stdout as `output`. "
                f"The execution timeout is fixed at {EXECUTE_PYTHON_TIMEOUT_SECONDS} seconds."
            ),
            input_schema={
                "code": "import os\nprint(sorted(os.listdir('.')))",
            },
        ),
        "extract_doc_records": ToolSpec(
            name="extract_doc_records",
            description=(
                "Extract structured entity records from long narrative documents. "
                "Use this for Markdown/doc text that describes repeated entities such as patients, budgets, events, "
                "or assets. It returns entity_id, matched term, nearby corrected/final numeric value, nearby year, "
                "and a compact sentence snippet."
            ),
            input_schema={
                "path": "relative/path/to/file.md",
                "target_terms": ["creatinine", "CRE"],
                "entity_hint": "patient",
                "max_records": 200,
            },
        ),
        "extract_doc_evidence_table": ToolSpec(
            name="extract_doc_evidence_table",
            description=(
                "Build an entity-keyed evidence table from a long narrative document. "
                "Use this when repeated records describe the same entities across sections and the question "
                "requires combining multiple fields such as height+publisher, lab value+birth year, "
                "event type+expense, or metric+category. The tool writes a temporary CSV and returns "
                "field coverage plus source snippets; use execute_python on evidence_table_csv for deterministic "
                "filters, numerator/denominator, percentages, counts, or grouped calculations."
            ),
            input_schema={
                "path": "relative/path/to/file.md",
                "fields": ["height", "publisher"],
                "entity_hint": "hero",
                "max_records": 1000,
            },
        ),
        "extract_pdf_tables": ToolSpec(
            name="extract_pdf_tables",
            description=(
                "Extract bounded tables from a PDF file inside context. Use this when the answer "
                "depends on tabular data embedded in a PDF report. Returns page number, table index, "
                "and rows for each extracted table."
            ),
            input_schema={
                "path": "relative/path/to/report.pdf",
                "max_tables": 10,
                "max_rows": 50,
            },
        ),
        "ground_thresholds": ToolSpec(
            name="ground_thresholds",
            description=(
                "Compare threshold/range evidence from context docs and observed data. "
                "Use this for questions with normal, abnormal, above/below, range, threshold, or level wording, "
                "especially when explicit rules are missing or when same-row versus same-entity counting is ambiguous. "
                "It returns matched numeric columns, explicit threshold snippets, population source coverage, "
                "and candidate count probes."
            ),
            input_schema={
                "target_terms": ["white blood cells", "fibrinogen"],
                "population_terms": ["male"],
                "max_candidates": 20,
            },
        ),
        "inspect_sqlite_schema": ToolSpec(
            name="inspect_sqlite_schema",
            description="Inspect tables and columns in a sqlite/db file inside context.",
            input_schema={"path": "relative/path/to/file.sqlite"},
        ),
        "list_context": ToolSpec(
            name="list_context",
            description="List files and directories available under context.",
            input_schema={"max_depth": 4},
        ),
        "read_csv": ToolSpec(
            name="read_csv",
            description="Read a preview of a CSV file inside context.",
            input_schema={"path": "relative/path/to/file.csv", "max_rows": 20},
        ),
        "read_doc": ToolSpec(
            name="read_doc",
            description="Read a text-like document inside context.",
            input_schema={"path": "relative/path/to/file.md", "max_chars": 4000},
        ),
        "read_pdf": ToolSpec(
            name="read_pdf",
            description=(
                "Read bounded text from a PDF file inside context. Use this for PDF reports, "
                "rules, policies, and narrative evidence when text preview is needed."
            ),
            input_schema={"path": "relative/path/to/report.pdf", "max_pages": 5, "max_chars": 8000},
        ),
        "read_json": ToolSpec(
            name="read_json",
            description="Read a preview of a JSON file inside context.",
            input_schema={"path": "relative/path/to/file.json", "max_chars": 4000},
        ),
        "profile_context": ToolSpec(
            name="profile_context",
            description="Auto-scan all files in context and return structured metadata: column names/types/statistics for CSV, schema/key paths for JSON, tables/columns/foreign keys for SQLite, headings/terms for documents. Use this FIRST to understand the data landscape before making detailed reads.",
            input_schema={},
        ),
        "query_json": ToolSpec(
            name="query_json",
            description="Query a JSON file by dot-separated key path (e.g. 'data.users.0.name') and return the matched value.",
            input_schema={"path": "relative/path/to/file.json", "query": "dot.separated.key.path"},
        ),
        "search_doc": ToolSpec(
            name="search_doc",
            description="Search a document for paragraphs containing a keyword or phrase. Returns matching paragraph excerpts.",
            input_schema={"path": "relative/path/to/file.md", "query": "keyword or phrase to search"},
        ),
        "search_pdf": ToolSpec(
            name="search_pdf",
            description=(
                "Search PDF page text for a keyword or phrase. Returns page numbers and excerpts. "
                "Use this before reading a whole PDF when looking for a specific rule, entity, or metric."
            ),
            input_schema={
                "path": "relative/path/to/report.pdf",
                "query": "keyword or phrase to search",
                "max_matches": 10,
            },
        ),
    }
    handlers = {
        "answer": _answer,
        "execute_context_sql": _execute_context_sql,
        "execute_data_sql": _execute_data_sql,
        "extract_pdf_tables": _extract_pdf_tables,
        "extract_doc_evidence_table": _extract_doc_evidence_table,
        "extract_doc_records": _extract_doc_records,
        "ground_thresholds": _ground_thresholds,
        "execute_python": _execute_python,
        "inspect_sqlite_schema": _inspect_sqlite_schema,
        "list_context": _list_context,
        "profile_context": _profile_context,
        "query_json": _query_json,
        "read_csv": _read_csv,
        "read_doc": _read_doc,
        "read_json": _read_json,
        "read_pdf": _read_pdf,
        "search_doc": _search_doc,
        "search_pdf": _search_pdf,
    }
    return ToolRegistry(specs=specs, handlers=handlers)
