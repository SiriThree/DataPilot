"""Schema-aware code debugger: repairs failed SQL/Python with data context.

Modeled after DS-STAR's A_debugger. Unlike generic debugging that only sees
tracebacks, this debugger injects schema context (column names, types, sample
values) and file metadata to help the LLM produce correct repairs.

Key innovations over generic retry:
1. Schema injection: shows actual column names/types from profile_context
2. Error pattern matching: recognizes common SQL/Python error patterns
3. Multi-attempt: tries repair, validates, retries if still broken
"""

from __future__ import annotations

from typing import Any

DEBUGGER_SYSTEM_PROMPT = """You are a data-aware code debugger. Your job is to fix a broken SQL query or Python script.
You have access to the actual schema (column names, types, sample values) and the error traceback.

Debugging rules:
1. READ THE ERROR carefully — most fixes are simple (typo, wrong column name, missing import)
2. CHECK THE SCHEMA — use exact column names and types from the schema context
3. For SQL errors:
   - "no such column" → check schema for the correct column name
   - "no such table" → check the DuckDB table name from profile_context
   - "ambiguous column" → qualify with table alias
   - "syntax error" → check quotes, commas, parentheses
4. For Python errors:
   - "NameError" → check imports or variable names
   - "KeyError" → check dict/column access patterns
   - "FileNotFoundError" → use correct relative path
   - "TypeError" → check data types of operands

Return ONLY the fixed code, nothing else. Keep the same tool (SQL or Python)."""

DEBUGGER_USER_TEMPLATE = """The following {tool_type} code FAILED:

```{language}
{code}
```

Error traceback:
```
{error_text}
```

Available schema and data context:
{schema_context}

Instructions: {instructions}

Return ONLY the fixed {tool_type} code:"""


def run_debugger(
    *,
    model,
    tool_type: str,  # "sql" or "python"
    code: str,
    error_text: str,
    schema_context: str,
    instructions: str = "",
) -> str | None:
    """Attempt to repair failed SQL/Python code.

    Args:
        model: Model adapter for LLM calls
        tool_type: "sql" or "python"
        code: The failed code
        error_text: Error message or traceback
        schema_context: Schema info from profile_context or inspect_sqlite_schema
        instructions: Optional additional instructions (e.g., "must use DuckDB table names")

    Returns:
        Repaired code string, or None if repair failed
    """
    from data_agent_baseline.agents.model import ModelMessage

    if not instructions:
        if tool_type == "sql":
            instructions = "Use exact table/column names from the schema. Quote identifiers if they contain spaces."
        else:
            instructions = "Use the correct file paths relative to the context directory."

    messages = [
        ModelMessage(role="system", content=DEBUGGER_SYSTEM_PROMPT),
        ModelMessage(
            role="user",
            content=DEBUGGER_USER_TEMPLATE.format(
                tool_type="SQL" if tool_type == "sql" else "Python",
                language="sql" if tool_type == "sql" else "python",
                code=code,
                error_text=error_text[:2000],
                schema_context=schema_context[:2000],
                instructions=instructions,
            ),
        ),
    ]

    try:
        response = model.complete(messages)
        return _extract_code_block(response, tool_type)
    except Exception:
        return None


def _extract_code_block(response: str, tool_type: str) -> str:
    """Extract code from the LLM response, stripping markdown fences if present."""
    import re

    text = response.strip()

    # Try fenced code block
    lang = "sql" if tool_type == "sql" else "python"
    pattern = rf"```(?:{lang})?\s*\n?(.*?)```"
    match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()

    # If no fence, return as-is (stripping any leading/trailing prose)
    lines = text.split("\n")
    code_lines = []
    in_code = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("```"):
            in_code = not in_code
            continue
        if in_code or not stripped.startswith(("Here", "The ", "Fixed", "Corrected")):
            code_lines.append(line)

    if code_lines:
        return "\n".join(code_lines).strip()

    return text


def build_schema_context_from_profile(profile_result: dict[str, Any]) -> str:
    """Build compact schema context from a profile_context result."""
    parts: list[str] = []
    files = profile_result.get("files", {})
    if not isinstance(files, dict):
        return str(profile_result)[:2000]

    for fname, finfo in files.items():
        if not isinstance(finfo, dict):
            continue
        ftype = finfo.get("type", "unknown")

        if ftype in ("csv", "json"):
            columns = finfo.get("columns", [])
            types_hint = finfo.get("column_types", {})
            duckdb_table = finfo.get("duckdb_table", "")
            sample = finfo.get("sample", "")

            col_strs = []
            for c in columns[:30]:
                ct = types_hint.get(c, "?") if isinstance(types_hint, dict) else "?"
                col_strs.append(f"{c}({ct})")
            parts.append(f"[{fname}] type={ftype} duckdb_table={duckdb_table} columns=[{', '.join(col_strs)}]")
            if sample:
                parts.append(f"  sample: {str(sample)[:200]}")

        elif ftype in ("sqlite", "db"):
            tables = finfo.get("tables", [])
            for t in tables[:10]:
                if isinstance(t, dict):
                    tname = t.get("name", "?")
                    tcols = t.get("columns", [])
                    parts.append(f"[{fname}] table={tname} columns={tcols[:30]}")

        elif ftype == "doc":
            headings = finfo.get("headings", [])
            terms = finfo.get("terms", [])
            parts.append(f"[{fname}] type=doc headings={headings[:10]} terms={terms[:10]}")

    return "\n".join(parts)[:2000] if parts else "(no schema available)"


def build_schema_context_from_sqlite_inspect(inspect_result: dict[str, Any]) -> str:
    """Build compact schema context from inspect_sqlite_schema result."""
    tables = inspect_result.get("tables", [])
    parts: list[str] = []
    for t in tables:
        if isinstance(t, dict):
            parts.append(f"Table: {t.get('name', '?')}")
            cols = t.get("columns", [])
            for c in cols:
                if isinstance(c, dict):
                    parts.append(f"  {c.get('name', '?')} {c.get('type', '?')}")
                else:
                    parts.append(f"  {str(c)}")
    return "\n".join(parts)[:2000] if parts else "(no schema available)"


# ── Error pattern matching ──

def classify_error(error_text: str) -> dict[str, Any]:
    """Classify an error into a known pattern for better debugging hints.

    Returns dict with: pattern, severity, hint
    """
    error_lower = error_text.lower()

    patterns = [
        ("missing_column", "no such column", "Check column name spelling; use exact names from schema"),
        ("missing_table", "no such table", "Use DuckDB table name from profile_context (e.g., csv__filename)"),
        ("ambiguous_column", "ambiguous column", "Qualify column with table alias: table.column"),
        ("syntax_error", "syntax error", "Check SQL syntax: quotes, commas, parentheses"),
        ("type_error", "typeerror", "Check data types; may need CAST or type conversion"),
        ("name_error", "nameerror", "Variable not defined; check imports and variable names"),
        ("key_error", "keyerror", "Dict key not found; check column name or dict access"),
        ("file_not_found", "filenotfounderror", "Use correct relative path to context directory"),
        ("parse_error", "parse error", "Check JSON/CSV parsing syntax"),
        ("permission", "permission denied", "Use read-only operations only"),
        ("timeout", "timeout", "Query timed out; optimize or add LIMIT"),
        ("memory", "out of memory", "Reduce result size; add LIMIT or filter"),
    ]

    for name, keyword, hint in patterns:
        if keyword in error_lower:
            return {"pattern": name, "severity": "medium", "hint": hint}

    return {"pattern": "unknown", "severity": "high", "hint": "Inspect error traceback carefully"}


# ── Simple code repair (rule-based, no LLM) ──

def simple_sql_repair(code: str, schema_context: str) -> str | None:
    """Attempt simple regex-based SQL repairs. Returns repaired code or None."""
    import re

    fixed = code

    # Fix: SELECT * FROM table without DuckDB prefix
    # (too risky to automate; skip)

    # Fix: trailing comma before FROM
    fixed = re.sub(r",\s*FROM\b", "\nFROM", fixed, flags=re.IGNORECASE)

    # Fix: missing semicolon (rarely the issue)
    # (skip; most SQL engines don't require semicolons)

    # Fix: unquoted identifiers with spaces or special chars
    # (skip; requires schema knowledge)

    if fixed != code:
        return fixed
    return None
