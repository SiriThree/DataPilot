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

LARGE_CSV_SQL_FIRST_BYTES = 5 * 1024 * 1024


# ── Data structures ───────────────────────────────────────────────

@dataclass
class RouteDecision:
    route: str  # sql_first | python_first | document_first | hybrid_sql_python | hybrid_doc_table
    task_profile: "TaskProfile"
    scores: dict[str, int]
    reasons: list[str]
    recommended_tools: list[str]
    risk_flags: list[str]


@dataclass
class TaskProfile:
    """Deterministic task classification used by prompts, trace mining, and retries."""
    task_type: str
    operation: str
    output_shape: str
    domains: list[str]
    validation_focus: list[str]
    flags: list[str] = field(default_factory=list)


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


def _contains_any(text: str, terms: set[str]) -> bool:
    return any(term in text for term in terms)


def _infer_domains(question: str, schema_hints: list[SchemaHint]) -> list[str]:
    q = _normalize_text(question)
    schema_text_parts = [q]
    for hint in schema_hints:
        schema_text_parts.append(Path(hint.path).name.lower())
        schema_text_parts.extend(column.lower() for column in hint.columns)
        schema_text_parts.extend(key.lower() for key in hint.top_keys)
        for table in hint.tables:
            schema_text_parts.append(str(table.get("name", "")).lower())
            for column in table.get("columns", []):
                if isinstance(column, dict):
                    schema_text_parts.append(str(column.get("name", "")).lower())
    text = " ".join(schema_text_parts)

    domains: list[str] = []
    if _contains_any(text, {"grand prix", "constructor", "raceid", "race_id", "driver", "circuit"}):
        domains.append("formula1")
    if _contains_any(text, {"patient", "hemoglobin", "glucose", "lab", "diagnosis", "medical"}):
        domains.append("medical_patient")
    if _contains_any(text, {"molecule", "atom", "bond", "toxicology", "compound"}):
        domains.append("toxicology")
    if _contains_any(text, {"superhero", "publisher", "alignment", "hero"}):
        domains.append("superhero")
    if _contains_any(text, {"school", "sat", "funding", "district"}):
        domains.append("school")
    if _contains_any(text, {"post", "comment", "stack", "score", "user_id", "stackoverflow"}):
        domains.append("stackexchange")
    if _contains_any(text, {"customer", "consumption", "tariff", "electric", "energy"}):
        domains.append("energy_consumption")
    return _dedupe(domains)


def _infer_task_profile(
    *,
    question: str,
    schema_hints: list[SchemaHint],
    has_table: bool,
    has_real_doc: bool,
) -> TaskProfile:
    q = _normalize_text(question)

    is_count = bool(re.search(r"\b(how many|count|number of|total number)\b", q))
    is_average = bool(re.search(r"\b(avg|average|mean)\b", q))
    is_sum = bool(re.search(r"\b(sum|total)\b", q)) and not is_count
    is_ratio = bool(re.search(r"\b(percent|percentage|ratio|proportion|rate)\b", q))
    is_rank = bool(re.search(r"\b(top|highest|lowest|most|least|rank|ranked|first|second|third|1st|2nd|3rd)\b", q))
    is_threshold = bool(re.search(r"\b(threshold|normal|abnormal|greater than|less than|above|below|over|under)\b", q))
    is_lookup = bool(re.search(r"\b(what|which|who|when|where|website|name|time|id)\b", q))
    is_join = bool(
        re.search(r"\b(join|merge|match|corresponding|associated|linked|for each|by .* id)\b", q)
        or (has_table and has_real_doc)
    )

    if re.search(r"\b(list|all|records|rows)\b", q):
        output_shape = "table"
    elif is_lookup and re.search(r"\b(name and|website and|id and|time and)\b", q):
        output_shape = "single_record"
    else:
        output_shape = "scalar"

    if is_threshold and is_count:
        operation = "threshold_count"
        task_type = "threshold_count"
    elif is_ratio:
        operation = "ratio"
        task_type = "ratio_or_percentage"
    elif is_average:
        operation = "aggregate_average"
        task_type = "aggregation"
    elif is_sum:
        operation = "aggregate_sum"
        task_type = "aggregation"
    elif is_rank:
        operation = "rank_lookup"
        task_type = "rank_lookup"
    elif is_count:
        operation = "count"
        task_type = "count"
    elif is_lookup:
        operation = "lookup"
        task_type = "document_table_lookup" if has_table and has_real_doc else "lookup"
    else:
        operation = "analysis"
        task_type = "general_analysis"

    flags: list[str] = []
    if is_join:
        flags.append("join_likely")
    if has_real_doc and has_table:
        flags.append("doc_table_grounding")
    if output_shape == "scalar" and task_type in {"count", "aggregation", "ratio_or_percentage", "threshold_count"}:
        flags.append("single_value_expected")

    validation_focus: list[str] = []
    if task_type in {"count", "threshold_count"}:
        validation_focus.append("verify population filters and exact counted entity IDs before answering")
    if task_type == "aggregation":
        validation_focus.append("verify grouping level, units, null handling, and aggregation operator")
    if task_type == "ratio_or_percentage":
        validation_focus.append("verify numerator, denominator, and whether output should be raw ratio or percent")
    if task_type == "rank_lookup":
        validation_focus.append("verify sort key, sort direction, tie handling, and requested rank")
    if task_type in {"lookup", "document_table_lookup"}:
        validation_focus.append("verify target entity identity and requested field names")
    if "doc_table_grounding" in flags:
        validation_focus.append("extract document rules first, then apply them to table rows")

    return TaskProfile(
        task_type=task_type,
        operation=operation,
        output_shape=output_shape,
        domains=_infer_domains(question, schema_hints),
        validation_focus=_dedupe(validation_focus),
        flags=_dedupe(flags),
    )


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
    large_csv_count = 0
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
                try:
                    if path.stat().st_size >= LARGE_CSV_SQL_FIRST_BYTES:
                        large_csv_count += 1
                except OSError:
                    pass
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

    if large_csv_count:
        scores["sql_first"] += 5
        scores["hybrid_sql_python"] += 3
        recommended_tools.extend(["execute_data_sql", "duckdb"])
        risk_flags.append("large_csv_sql_first")
        reasons.append(f"large csv files detected ({large_csv_count}); prefer DuckDB filtering/aggregation")

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
        elif large_csv_count:
            scores["sql_first"] += 4
            scores["hybrid_sql_python"] += 2
            reasons.append(f"difficulty={difficulty}: prefer SQL/table fast path for large CSV context")
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

    task_profile = _infer_task_profile(
        question=question,
        schema_hints=schema_hints,
        has_table=has_table,
        has_real_doc=has_real_doc,
    )

    return RouteDecision(
        route=route,
        task_profile=task_profile,
        scores=scores,
        reasons=_dedupe(reasons),
        recommended_tools=_dedupe(recommended_tools),
        risk_flags=_dedupe(risk_flags),
    )


def task_profile_to_dict(profile: TaskProfile) -> dict[str, Any]:
    return {
        "task_type": profile.task_type,
        "operation": profile.operation,
        "output_shape": profile.output_shape,
        "domains": profile.domains,
        "validation_focus": profile.validation_focus,
        "flags": profile.flags,
    }


def route_to_dict(rd: RouteDecision) -> dict[str, Any]:
    return {
        "route": rd.route,
        "task_profile": task_profile_to_dict(rd.task_profile),
        "scores": rd.scores,
        "reasons": rd.reasons,
        "recommended_tools": rd.recommended_tools,
        "risk_flags": rd.risk_flags,
    }


def build_route_prompt_hint(rd: RouteDecision) -> str:
    """Generate a concise prompt hint to inject before the task question."""
    route_name = rd.route.replace("_", " ").title()
    tools = ", ".join(rd.recommended_tools) if rd.recommended_tools else "auto-detect"
    domains = ", ".join(rd.task_profile.domains) if rd.task_profile.domains else "general"
    profile_flags = ", ".join(rd.task_profile.flags) if rd.task_profile.flags else "none"
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
        (
            "Task profile: "
            f"type={rd.task_profile.task_type}; operation={rd.task_profile.operation}; "
            f"output={rd.task_profile.output_shape}; domain={domains}; flags={profile_flags}"
        ),
        f"Recommended tools: {tools}",
        f"Role focus: {role_focus}",
    ]
    if rd.task_profile.validation_focus:
        lines.append("Validation focus: " + " | ".join(rd.task_profile.validation_focus[:3]))
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

    if "large_csv_sql_first" in rd.risk_flags:
        lines.append(
            "\nMANDATORY LARGE-CSV STRATEGY:\n"
            "At least one CSV is large enough that full pandas reads are risky.\n"
            "Follow this sequence:\n"
            "  1. profile_context to get DuckDB table names and column schemas\n"
            "  2. execute_data_sql for filtering, joins, aggregation, sorting, and top-k work\n"
            "  3. execute_python only after SQL has reduced the data, or with bounded pandas reads using nrows/chunksize/usecols\n"
            "Do NOT call pandas.read_csv on a large CSV without nrows, chunksize, or usecols."
        )

    return "\n".join(lines)
