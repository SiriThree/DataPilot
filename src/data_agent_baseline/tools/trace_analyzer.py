"""Trace Analyzer: parse trace.json files to generate debugging insights and case studies.

Produces:
- Per-task summary: step sequence, tool usage, outcome
- Failure analysis: categorize failure types
- Case study generator: render a human-readable case study for the report
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def analyze_trace(trace_path: Path) -> dict[str, Any]:
    """Parse a trace.json and return structured analysis."""
    if not trace_path.exists():
        return {"error": f"trace file not found: {trace_path}"}

    data = json.loads(trace_path.read_text())

    task_id = data.get("task_id", "unknown")
    succeeded = data.get("succeeded", False)
    failure_reason = data.get("failure_reason")
    steps = data.get("steps", [])
    verification = data.get("_verification", {})
    e2e_elapsed = data.get("e2e_elapsed_seconds", 0)

    # Tool usage summary
    tool_counts: dict[str, int] = {}
    tool_successes: dict[str, int] = {}
    step_details: list[dict[str, Any]] = []
    for step in steps:
        tool_name = step.get("action", "unknown")
        tool_counts[tool_name] = tool_counts.get(tool_name, 0) + 1
        if step.get("ok"):
            tool_successes[tool_name] = tool_successes.get(tool_name, 0) + 1

        step_details.append({
            "step": step.get("step_index"),
            "thought": step.get("thought", "")[:200],
            "action": tool_name,
            "ok": step.get("ok"),
            "error": step.get("observation", {}).get("error"),
        })

    # Failure categorization
    failure_category = "none"
    if not succeeded:
        if failure_reason and "timed out" in failure_reason.lower():
            failure_category = "timeout"
        elif failure_reason and "max_steps" in failure_reason:
            failure_category = "exceeded_max_steps"
        elif any(not s.get("ok") for s in steps):
            failure_category = "tool_error"
        else:
            failure_category = "no_answer"

    return {
        "task_id": task_id,
        "succeeded": succeeded,
        "e2e_elapsed_seconds": e2e_elapsed,
        "total_steps": len(steps),
        "failure_reason": failure_reason,
        "failure_category": failure_category,
        "tool_usage": {
            tool: {
                "calls": tool_counts.get(tool, 0),
                "success_rate": (
                    tool_successes.get(tool, 0) / tool_counts.get(tool, 1)
                    if tool_counts.get(tool, 0) > 0
                    else 0.0
                ),
            }
            for tool in sorted(tool_counts)
        },
        "verification": verification,
        "steps": step_details,
    }


def analyze_batch_run(run_dir: Path) -> dict[str, Any]:
    """Analyze all traces in a batch run directory."""
    tasks: list[dict[str, Any]] = []
    for task_dir in sorted(run_dir.iterdir()):
        if not task_dir.is_dir():
            continue
        trace_path = task_dir / "trace.json"
        if trace_path.exists():
            result = analyze_trace(trace_path)
            tasks.append(result)

    if not tasks:
        return {"error": "no traces found", "task_count": 0}

    succeeded = [t for t in tasks if t["succeeded"]]
    failed = [t for t in tasks if not t["succeeded"]]

    # Aggregate tool usage
    global_tool_counts: dict[str, int] = {}
    for t in tasks:
        for tool, info in t["tool_usage"].items():
            global_tool_counts[tool] = global_tool_counts.get(tool, 0) + info["calls"]

    # Failure categories
    failure_categories: dict[str, int] = {}
    for t in failed:
        cat = t.get("failure_category", "unknown")
        failure_categories[cat] = failure_categories.get(cat, 0) + 1

    # Average stats
    avg_steps = sum(t["total_steps"] for t in tasks) / len(tasks) if tasks else 0
    avg_elapsed = sum(t["e2e_elapsed_seconds"] for t in tasks) / len(tasks) if tasks else 0

    return {
        "run_dir": str(run_dir),
        "task_count": len(tasks),
        "succeeded_count": len(succeeded),
        "failed_count": len(failed),
        "success_rate": round(len(succeeded) / len(tasks), 3) if tasks else 0.0,
        "avg_steps_per_task": round(avg_steps, 1),
        "avg_elapsed_seconds": round(avg_elapsed, 1),
        "failure_categories": failure_categories,
        "top_tools": sorted(global_tool_counts.items(), key=lambda x: x[1], reverse=True)[:5],
        "tasks": tasks,
    }


def generate_case_study(trace: dict[str, Any], task_question: str = "", task_difficulty: str = "") -> str:
    """Generate a human-readable case study from a trace analysis.

    Returns a Markdown-formatted case study suitable for the course report.
    """
    lines: list[str] = []

    task_id = trace.get("task_id", "?")
    succeeded = trace.get("succeeded", False)
    status = "PASS" if succeeded else "FAIL"
    difficulty = task_difficulty or "unknown"

    lines.append(f"## Case Study: {task_id} [{difficulty}] — {status}")
    lines.append("")

    if task_question:
        lines.append(f"**Question:** {task_question}")
        lines.append("")

    lines.append(f"- **Steps:** {trace.get('total_steps', 0)}")
    lines.append(f"- **Elapsed:** {trace.get('e2e_elapsed_seconds', 0):.1f}s")
    lines.append("")

    # Tool execution flow
    lines.append("### Execution Flow")
    lines.append("")
    for step in trace.get("steps", []):
        step_num = step.get("step", "?")
        action = step.get("action", "?")
        ok = "OK" if step.get("ok") else "FAIL"
        thought = step.get("thought", "")[:120]
        lines.append(f"{step_num}. **{action}** [{ok}] — {thought}")
    lines.append("")

    # Failure analysis
    if not succeeded:
        lines.append("### Failure Analysis")
        lines.append(f"**Category:** {trace.get('failure_category', 'unknown')}")
        lines.append(f"**Reason:** {trace.get('failure_reason', 'unknown')}")
        lines.append("")

    # Verification
    verif = trace.get("verification", {})
    if verif:
        lines.append("### Post-Answer Verification")
        verif_ok = "PASS" if verif.get("ok") else "WARNINGS"
        lines.append(f"**Status:** {verif_ok}")
        for w in verif.get("warnings", [])[:3]:
            lines.append(f"- Warning: {w}")
        for f in verif.get("suggested_fixes", [])[:3]:
            lines.append(f"- Fix: {f}")
        lines.append("")

    return "\n".join(lines)


def generate_case_studies_report(run_dir: Path, input_dir: Path) -> str:
    """Generate a full case studies report for all tasks in a run."""
    analysis = analyze_batch_run(run_dir)
    if "error" in analysis:
        return f"# Case Studies Report\n\nError: {analysis['error']}"

    lines: list[str] = []
    lines.append("# DataAgent Case Studies Report")
    lines.append("")
    lines.append(f"**Run:** {run_dir.name}")
    lines.append(f"**Tasks:** {analysis['task_count']} ({analysis['succeeded_count']} passed, {analysis['failed_count']} failed)")
    lines.append(f"**Success Rate:** {analysis['success_rate']:.1%}")
    lines.append(f"**Avg Steps:** {analysis['avg_steps_per_task']}")
    lines.append(f"**Avg Time:** {analysis['avg_elapsed_seconds']:.1f}s")
    lines.append("")

    if analysis["failure_categories"]:
        lines.append("### Failure Categories")
        for cat, count in sorted(analysis["failure_categories"].items()):
            lines.append(f"- {cat}: {count}")
        lines.append("")

    # Pick representative cases: best, worst, and one of each difficulty
    tasks = analysis["tasks"]
    succeeded_tasks = [t for t in tasks if t["succeeded"]]
    failed_tasks = [t for t in tasks if not t["succeeded"]]

    # Show one successful case and one failed case at most
    cases_to_show: list[dict[str, Any]] = []
    if succeeded_tasks:
        # Pick one with the fewest steps (most efficient)
        best = min(succeeded_tasks, key=lambda t: t["total_steps"])
        cases_to_show.append(best)
    if failed_tasks:
        # Pick one with the most detailed failure info
        worst = max(failed_tasks, key=lambda t: len(t.get("failure_reason", "")))
        cases_to_show.append(worst)

    for task_analysis in cases_to_show:
        task_id = task_analysis["task_id"]
        # Try to get the question from the input directory
        question = ""
        difficulty = ""
        task_json = input_dir / task_id / "task.json"
        if task_json.exists():
            task_data = json.loads(task_json.read_text())
            question = task_data.get("question", "")
            difficulty = task_data.get("difficulty", "")

        lines.append(generate_case_study(task_analysis, question, difficulty))
        lines.append("---")
        lines.append("")

    return "\n".join(lines)
