"""Evaluation-driven trace mining for benchmark runs."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class FailureMiningResult:
    summary: dict[str, Any]
    tasks: list[dict[str, Any]]
    groups: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "summary": self.summary,
            "groups": self.groups,
            "tasks": self.tasks,
        }


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _evaluation_failures(evaluation: dict[str, Any]) -> dict[str, dict[str, Any]]:
    failures = evaluation.get("failures", [])
    if not isinstance(failures, list):
        return {}
    rows: dict[str, dict[str, Any]] = {}
    for item in failures:
        if not isinstance(item, dict):
            continue
        task_id = str(item.get("task_id") or "").strip()
        if task_id:
            rows[task_id] = item
    return rows


def _task_ids_from_run(run_dir: Path, failures: dict[str, dict[str, Any]]) -> list[str]:
    task_ids = set(failures)
    if run_dir.exists():
        for child in run_dir.iterdir():
            if child.is_dir() and child.name.startswith("task_"):
                task_ids.add(child.name)
    return sorted(task_ids, key=lambda value: (len(value), value))


def _task_profile(trace: dict[str, Any]) -> dict[str, Any]:
    route = trace.get("_route_decision")
    if not isinstance(route, dict):
        return {}
    profile = route.get("task_profile")
    return profile if isinstance(profile, dict) else {}


def _signal_codes(failure_analysis: dict[str, Any]) -> list[str]:
    codes: list[str] = []
    seen: set[str] = set()
    signals = failure_analysis.get("signals", [])
    if not isinstance(signals, list):
        return codes
    for signal in signals:
        if not isinstance(signal, dict):
            continue
        code = str(signal.get("code") or "").strip()
        if code and code not in seen:
            seen.add(code)
            codes.append(code)
    return codes


def _tool_counts(trace: dict[str, Any], failure_analysis: dict[str, Any]) -> dict[str, int]:
    commitments = failure_analysis.get("commitments")
    if isinstance(commitments, dict) and isinstance(commitments.get("tool_counts"), dict):
        return {str(k): int(v) for k, v in commitments["tool_counts"].items()}
    counter: Counter[str] = Counter()
    for step in trace.get("steps", []) or []:
        if isinstance(step, dict):
            counter[str(step.get("action") or "unknown")] += 1
    return dict(counter)


def _selected_attempt(trace: dict[str, Any]) -> str:
    selected = trace.get("_selected_attempt")
    if selected:
        return str(selected)
    guided = trace.get("_guided_retry")
    if isinstance(guided, dict) and guided.get("attempted"):
        selection = guided.get("selection")
        if isinstance(selection, dict) and selection.get("selected_attempt"):
            return str(selection["selected_attempt"])
    return "original"


def _task_row(
    *,
    run_dir: Path,
    task_id: str,
    eval_failure: dict[str, Any] | None,
) -> dict[str, Any]:
    task_dir = run_dir / task_id
    trace = _read_json(task_dir / "trace.json")
    failure_analysis = _read_json(task_dir / "failure_analysis.json")
    profile = _task_profile(trace)
    route = trace.get("_route_decision")
    route_name = str(route.get("route")) if isinstance(route, dict) and route.get("route") else "unknown"
    score = eval_failure.get("score") if eval_failure else 1.0
    difficulty = eval_failure.get("difficulty") if eval_failure else trace.get("difficulty")

    return {
        "task_id": task_id,
        "score": score,
        "difficulty": difficulty or "unknown",
        "route": route_name,
        "task_type": profile.get("task_type") or "unknown",
        "operation": profile.get("operation") or "unknown",
        "output_shape": profile.get("output_shape") or "unknown",
        "domains": profile.get("domains") or [],
        "profile_flags": profile.get("flags") or [],
        "signals": _signal_codes(failure_analysis),
        "tool_counts": _tool_counts(trace, failure_analysis),
        "selected_attempt": _selected_attempt(trace),
        "repair_actions": _repair_actions(trace),
        "eval": eval_failure or {},
        "paths": {
            "task_dir": str(task_dir),
            "trace": str(task_dir / "trace.json"),
            "prediction": str(task_dir / "prediction.csv"),
            "failure_analysis": str(task_dir / "failure_analysis.json"),
        },
    }


def _repair_actions(trace: dict[str, Any]) -> list[str]:
    post_process = trace.get("_post_process")
    if not isinstance(post_process, dict):
        return []
    execution = post_process.get("repair_execution")
    if not isinstance(execution, dict):
        return []
    actions = execution.get("actions", [])
    if not isinstance(actions, list):
        return []
    names: list[str] = []
    for action in actions:
        if isinstance(action, dict):
            value = action.get("action_type") or action.get("type") or action.get("action")
            if value:
                names.append(str(value))
    return names


def _counter_group(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    counter: Counter[str] = Counter(str(row.get(key) or "unknown") for row in rows)
    return dict(counter.most_common())


def _list_group(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for row in rows:
        values = row.get(key)
        if isinstance(values, list):
            for value in values:
                counter[str(value)] += 1
    return dict(counter.most_common())


def _tasks_by(rows: list[dict[str, Any]], key: str) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get(key) or "unknown")].append(str(row["task_id"]))
    return {key: value for key, value in sorted(grouped.items())}


def analyze_run_failures(
    *,
    run_dir: Path,
    score_threshold: float = 1.0,
    evaluation_path: Path | None = None,
) -> FailureMiningResult:
    evaluation_file = evaluation_path or (run_dir / "evaluation.json")
    evaluation = _read_json(evaluation_file)
    failures = _evaluation_failures(evaluation)
    task_ids = _task_ids_from_run(run_dir, failures)
    rows = [_task_row(run_dir=run_dir, task_id=task_id, eval_failure=failures.get(task_id)) for task_id in task_ids]
    low_score_rows = [
        row for row in rows
        if isinstance(row.get("score"), (int, float)) and float(row["score"]) < score_threshold
    ]
    low_score_rows.sort(key=lambda row: (float(row.get("score") or 0.0), str(row.get("difficulty")), str(row["task_id"])))

    groups = {
        "by_difficulty": _counter_group(low_score_rows, "difficulty"),
        "by_route": _counter_group(low_score_rows, "route"),
        "by_task_type": _counter_group(low_score_rows, "task_type"),
        "by_operation": _counter_group(low_score_rows, "operation"),
        "by_output_shape": _counter_group(low_score_rows, "output_shape"),
        "by_domain": _list_group(low_score_rows, "domains"),
        "by_signal": _list_group(low_score_rows, "signals"),
        "tasks_by_task_type": _tasks_by(low_score_rows, "task_type"),
        "tasks_by_route": _tasks_by(low_score_rows, "route"),
    }
    summary = {
        "run_dir": str(run_dir),
        "evaluation_path": str(evaluation_file),
        "total_tasks_in_evaluation": evaluation.get("total_tasks"),
        "overall_score": evaluation.get("overall_score"),
        "score_threshold": score_threshold,
        "task_count_seen": len(rows),
        "low_score_task_count": len(low_score_rows),
    }
    return FailureMiningResult(summary=summary, tasks=low_score_rows, groups=groups)


def write_failure_mining_outputs(result: FailureMiningResult, output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "failure_mining.json"
    md_path = output_dir / "failure_mining.md"
    json_path.write_text(
        json.dumps(result.to_dict(), ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    md_path.write_text(render_failure_mining_markdown(result), encoding="utf-8")
    return json_path, md_path


def render_failure_mining_markdown(result: FailureMiningResult) -> str:
    summary = result.summary
    lines = [
        "# Failure Mining Report",
        "",
        f"- Run: `{summary.get('run_dir')}`",
        f"- Evaluation: `{summary.get('evaluation_path')}`",
        f"- Overall score: `{summary.get('overall_score')}`",
        f"- Low-score threshold: `< {summary.get('score_threshold')}`",
        f"- Low-score tasks: `{summary.get('low_score_task_count')}` / `{summary.get('task_count_seen')}`",
        "",
        "## Groups",
        "",
    ]
    for label, key in [
        ("Difficulty", "by_difficulty"),
        ("Task Type", "by_task_type"),
        ("Operation", "by_operation"),
        ("Route", "by_route"),
        ("Signal", "by_signal"),
        ("Domain", "by_domain"),
    ]:
        values = result.groups.get(key, {})
        if not values:
            continue
        lines.append(f"### {label}")
        for name, count in values.items():
            lines.append(f"- `{name}`: {count}")
        lines.append("")

    lines.extend(["## Lowest-Score Tasks", ""])
    if not result.tasks:
        lines.append("No tasks below the score threshold.")
        lines.append("")
        return "\n".join(lines)

    lines.append("| Task | Score | Difficulty | Type | Route | Signals | Repair |")
    lines.append("| --- | ---: | --- | --- | --- | --- | --- |")
    for row in result.tasks[:50]:
        signals = ", ".join(row.get("signals") or []) or "-"
        repairs = ", ".join(row.get("repair_actions") or []) or "-"
        lines.append(
            f"| `{row['task_id']}` | {row.get('score')} | {row.get('difficulty')} | "
            f"{row.get('task_type')} | {row.get('route')} | {signals} | {repairs} |"
        )
    lines.append("")
    lines.append("## Recommended Reading Order")
    lines.append("")
    for row in result.tasks[:12]:
        lines.append(
            f"1. `{row['task_id']}`: score `{row.get('score')}`, "
            f"type `{row.get('task_type')}`, route `{row.get('route')}`, "
            f"signals `{', '.join(row.get('signals') or []) or '-'}`."
        )
    lines.append("")
    return "\n".join(lines)
