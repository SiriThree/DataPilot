"""Signal-based route decision for Data Agent tasks.

Deterministically decides the best tool path (SQL-first, Python-first,
Document-first, or Hybrid) by matching question keywords against signal
sets and scoring available context file types.

Adapted from soonhp/KddCupDataAgents task_intelligence.py.
"""

from __future__ import annotations

import csv
import json
import re
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# ── Signal dictionaries ──────────────────────────────────────────

SQL_SIGNALS = {
    "join", "group", "count", "sum", "average", "avg", "top", "rank",
    "filter", "where", "by", "per", "compare", "aggregate", "total",
}

PYTHON_SIGNALS = {
    "correlation", "trend", "time series", "forecast", "regression",
    "distribution", "variance", "median", "normalize", "calculate",
    "compute", "standard deviation", "std", "rolling",
}

DOCUMENT_SIGNALS = {
    "according to", "policy", "definition", "define", "rule",
    "manual", "guideline", "explain", "extract from document",
    "document", "based on the text",
}


# ── Data structures ───────────────────────────────────────────────

@dataclass
class RouteDecision:
    route: str  # sql_first | python_first | document_first | hybrid_sql_python | hybrid_doc_table
    scores: dict[str, int]
    reasons: list[str]
    recommended_tools: list[str]
    risk_flags: list[str]


@dataclass
class SchemaHint:
    """Lightweight schema summary for one file."""
    path: str
    type: str  # csv, db, json, doc
    columns: list[str] = field(default_factory=list)
    tables: list[dict[str, Any]] = field(default_factory=list)
    top_keys: list[str] = field(default_factory=list)
    preview: str = ""
    error: str | None = None


# ── Helpers ───────────────────────────────────────────────────────

def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def _find_signals(question: str, signals: set[str]) -> list[str]:
    normalized = _normalize_text(question)
    matched: list[str] = []
    for signal in sorted(signals):
        if " " in signal:
            if signal.lower() in normalized:
                matched.append(signal)
        elif re.search(r"\b" + re.escape(signal.lower()) + r"\b", normalized):
            matched.append(signal)
    return matched


def _dedupe(lst: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in lst:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


# ── Schema hint extraction ────────────────────────────────────────

def _hint_csv(path: Path) -> SchemaHint:
    try:
        with open(path, encoding="utf-8-sig", newline="") as f:
            header = next(csv.reader(f), [])
        return SchemaHint(path=str(path), type="csv", columns=[c.strip() for c in header])
    except Exception as exc:
        return SchemaHint(path=str(path), type="csv", error=str(exc))


def _hint_db(path: Path) -> SchemaHint:
    try:
        with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as conn:
            rows = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
            ).fetchall()
            tables = []
            for (tname,) in rows[:30]:
                cols = conn.execute(f'PRAGMA table_info("{tname}")').fetchall()
                tables.append({
                    "name": tname,
                    "columns": [{"name": c[1], "type": c[2]} for c in cols[:50]],
                })
        return SchemaHint(path=str(path), type="db", tables=tables)
    except Exception as exc:
        return SchemaHint(path=str(path), type="db", error=str(exc))


def _hint_json(path: Path) -> SchemaHint:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        keys = list(data.keys())[:30] if isinstance(data, dict) else []
        return SchemaHint(path=str(path), type="json", top_keys=keys)
    except Exception as exc:
        return SchemaHint(path=str(path), type="json", error=str(exc))


def _hint_doc(path: Path) -> SchemaHint:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
        return SchemaHint(path=str(path), type="doc", preview=text[:500])
    except Exception as exc:
        return SchemaHint(path=str(path), type="doc", error=str(exc))


# ── Main decision function ────────────────────────────────────────

def decide_route(
    *,
    question: str,
    context_dir: Path,
    difficulty: str | None = None,
) -> RouteDecision:
    """Analyze question + context and return a deterministic route decision.

    Call BEFORE the ReAct loop so the route can be injected into the task prompt.
    """
    scores = {
        "sql_first": 0,
        "python_first": 0,
        "document_first": 0,
        "hybrid_sql_python": 0,
        "hybrid_doc_table": 0,
    }
    reasons: list[str] = []
    recommended_tools: list[str] = []
    risk_flags: list[str] = []

    # ── Scan context files ──
    csv_count = 0
    db_count = 0
    json_count = 0
    doc_count = 0
    knowledge_count = 0
    schema_hints: list[SchemaHint] = []

    if context_dir.exists():
        for path in context_dir.rglob("*"):
            if not path.is_file():
                continue
            suffix = path.suffix.lower()
            name = path.name.lower()

            if suffix == ".csv":
                csv_count += 1
                schema_hints.append(_hint_csv(path))
            elif suffix in (".sqlite", ".db", ".duckdb"):
                db_count += 1
                schema_hints.append(_hint_db(path))
            elif suffix == ".json":
                json_count += 1
                schema_hints.append(_hint_json(path))
            if name == "knowledge.md":
                knowledge_count += 1
                schema_hints.append(_hint_doc(path))
            elif suffix in (".md", ".txt", ".pdf"):
                doc_count += 1
                schema_hints.append(_hint_doc(path))

    # ── Signal matching ──
    sql_signals = _find_signals(question, SQL_SIGNALS)
    python_signals = _find_signals(question, PYTHON_SIGNALS)
    document_signals = _find_signals(question, DOCUMENT_SIGNALS)

    # ── Scoring ──
    if db_count:
        scores["sql_first"] += 4
        scores["hybrid_sql_python"] += 2
        recommended_tools.append("sqlite")
        reasons.append(f"db files detected ({db_count})")

    if csv_count:
        scores["python_first"] += 3
        scores["hybrid_sql_python"] += 1
        recommended_tools.extend(["pandas", "python"])
        reasons.append(f"csv files detected ({csv_count})")

    if json_count:
        scores["python_first"] += 2
        recommended_tools.append("python_json")
        reasons.append(f"json files detected ({json_count})")

    if doc_count:
        scores["document_first"] += 3
        scores["hybrid_doc_table"] += 2
        recommended_tools.append("document_reader")
        reasons.append(f"doc files detected ({doc_count})")

    if knowledge_count:
        # knowledge.md is background metadata — only boost document routing
        # when real doc files ALSO exist and document signals are present.
        # Otherwise knowledge.md alone should not influence routing.
        if doc_count and document_signals:
            scores["document_first"] += 1
        reasons.append("knowledge.md detected")

    if sql_signals:
        scores["sql_first"] += 2 + len(sql_signals)
        scores["hybrid_sql_python"] += len(sql_signals)
        reasons.append(f"sql signals: {', '.join(sql_signals)}")

    if python_signals:
        scores["python_first"] += 2 + len(python_signals)
        scores["hybrid_sql_python"] += len(python_signals)
        reasons.append(f"python signals: {', '.join(python_signals)}")

    if document_signals:
        scores["document_first"] += 2 + len(document_signals)
        scores["hybrid_doc_table"] += len(document_signals)
        reasons.append(f"document signals: {', '.join(document_signals)}")

    has_table = bool(db_count or csv_count or json_count)
    has_real_doc = bool(doc_count)
    has_doc = bool(doc_count or knowledge_count)

    if db_count and csv_count:
        scores["hybrid_sql_python"] += 3
        reasons.append("db and csv context both present")

    if has_real_doc and has_table:
        scores["hybrid_doc_table"] += 3
        reasons.append("data documents and table context both present")

    if db_count and python_signals:
        scores["hybrid_sql_python"] += 3
        reasons.append("db context requires python/statistical handling")

    if has_real_doc and sql_signals:
        scores["hybrid_doc_table"] += 2
        reasons.append("document context with table-style question signals")

    if not has_table and not has_doc:
        scores["python_first"] += 1
        risk_flags.append("no_supported_context_files")
        reasons.append("empty context fallback")

    if difficulty and difficulty.lower() in {"hard", "extreme"}:
        risk_flags.append("high_difficulty")
        reasons.append(f"difficulty={difficulty}")

    # Detect pure-doc tasks (extreme/hard with no structured data sources)
    has_structured_source = bool(db_count or csv_count or json_count)
    if doc_count and not has_structured_source and difficulty and difficulty.lower() in {"hard", "extreme"}:
        risk_flags.append("pure_doc_no_structured_source")

    if difficulty and difficulty.lower() in {"easy", "medium"} and has_table and not has_real_doc:
        if db_count:
            scores["sql_first"] += 4
            scores["hybrid_sql_python"] += 2
            reasons.append(f"difficulty={difficulty}: prefer SQL/table fast path without data docs")
        else:
            scores["python_first"] += 4
            scores["hybrid_sql_python"] += 1
            reasons.append(f"difficulty={difficulty}: prefer Python/table fast path without data docs")
        scores["document_first"] = min(scores["document_first"], 1)
        scores["hybrid_doc_table"] = 0

    route = max(scores, key=lambda k: scores[k])
    if all(v == 0 for v in scores.values()):
        route = "python_first"
        reasons.append("zero-score fallback")
        risk_flags.append("zero_score_route")

    return RouteDecision(
        route=route,
        scores=scores,
        reasons=_dedupe(reasons),
        recommended_tools=_dedupe(recommended_tools),
        risk_flags=_dedupe(risk_flags),
    )


def route_to_dict(rd: RouteDecision) -> dict[str, Any]:
    return {
        "route": rd.route,
        "scores": rd.scores,
        "reasons": rd.reasons,
        "recommended_tools": rd.recommended_tools,
        "risk_flags": rd.risk_flags,
    }


def build_route_prompt_hint(rd: RouteDecision) -> str:
    """Generate a concise prompt hint to inject before the task question."""
    route_name = rd.route.replace("_", " ").title()
    tools = ", ".join(rd.recommended_tools) if rd.recommended_tools else "auto-detect"
    role_focus = {
        "sql_first": (
            "Data Engineer focus: inspect DB schema, write one precise SQL query, "
            "then verify filters, joins, aggregation, and ordering."
        ),
        "python_first": (
            "Analyst focus: use Python for calculations, normalization, statistics, or reshaping; "
            "print compact intermediate results before answering."
        ),
        "document_first": (
            "Scout/Verifier focus: extract the relevant rule or definition from documents first; "
            "do not invent thresholds from external knowledge."
        ),
        "hybrid_sql_python": (
            "Planner focus: use SQL/DuckDB to reduce data, then Python only for final custom logic."
        ),
        "hybrid_doc_table": (
            "Planner focus: first extract document rules, then apply them to table data with SQL/Python."
        ),
    }.get(rd.route, "Follow the role protocol and choose tools from observed context.")
    lines = [
        f"Route: {route_name} (determined by signal matching)",
        f"Recommended tools: {tools}",
        f"Role focus: {role_focus}",
    ]
    if rd.risk_flags:
        lines.append(f"Risk flags: {', '.join(rd.risk_flags)}")
    if rd.reasons:
        lines.append(f"Key reasons: {'; '.join(rd.reasons[:3])}")

    # For pure-doc extreme/hard tasks: inject mandatory structured-extraction strategy.
    # Without this, agents often fall into a regex trap — burning all steps
    # parsing narrative prose with hand-rolled Python instead of using purpose-built tools.
    if "pure_doc_no_structured_source" in rd.risk_flags:
        lines.append(
            "\nMANDATORY PURE-DOC STRATEGY:\n"
            "This task has ONLY narrative markdown documents (no CSV/DB/JSON).\n"
            "Hand-rolled regex on raw markdown text will waste steps and produce wrong values.\n"
            "Follow this exact sequence:\n"
            "  1. profile_context\n"
            "  2. extract_doc_records on the primary data document to get structured entity/value rows\n"
            "  3. extract_doc_records on the secondary document (e.g. patient demographics) if needed\n"
            "  4. ground_thresholds to determine normal/abnormal cutoffs from context (do NOT guess from external knowledge)\n"
            "  5. execute_python on the extracted CSV tables for join + filter + count/aggregate (simple math only, no regex)\n"
            "  6. answer\n"
            "NEVER use execute_python to parse raw .md files with regex. Use extract_doc_records instead."
        )

    return "\n".join(lines)
