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
    recommendations: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "summary": self.summary,
            "groups": self.groups,
            "recommendations": self.recommendations,
            "tasks": self.tasks,
        }


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _evaluation_failures(evaluation: dict[str, Any]) -> dict[str, dict[str, Any]]:
    summary = evaluation.get("summary")
    source = summary if isinstance(summary, dict) else evaluation
    failures = source.get("failures", [])
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


def _evaluation_summary(evaluation: dict[str, Any]) -> dict[str, Any]:
    summary = evaluation.get("summary")
    return summary if isinstance(summary, dict) else evaluation


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


RECOMMENDATION_RULES: dict[str, dict[str, str]] = {
    "threshold_grounding_risk": {
        "area": "threshold verifier / solver",
        "action": (
            "Strengthen threshold grounding: extract explicit range rules, preserve population joins, "
            "and compare entity-level versus same-row counts."
        ),
    },
    "population_filter_semantic_risk": {
        "area": "population filter verifier",
        "action": (
            "Add population-scope checks before aggregation so the model does not add extra null, "
            "positive-value, or joined-subset filters."
        ),
    },
    "target_field_semantic_risk": {
        "area": "target-field verifier / repair",
        "action": "Verify the requested output field and repair nearby-id/evidence-column answers.",
    },
    "output_shape_semantic_risk": {
        "area": "output contract repair",
        "action": "Infer requested columns from wording and prevent packed multi-value scalar cells.",
    },
    "scalar_format_semantic_risk": {
        "area": "scalar format repair",
        "action": "Normalize percent signs, delimiters, and single-cell scalar formatting.",
    },
    "rank_semantics_risk": {
        "area": "rank/order solver",
        "action": "Disambiguate rank columns from derived sorting and verify requested rank direction.",
    },
    "denominator_risk": {
        "area": "ratio verifier",
        "action": "Require explicit numerator and denominator commitments before final ratio answers.",
    },
    "numeric_grounding_error": {
        "area": "numeric evidence verifier",
        "action": "Require a concrete formula/query and observed numeric evidence for numeric questions.",
    },
    "budget_exhaustion": {
        "area": "routing / budget control",
        "action": "Improve route-specific early exit and retry budgets for tasks that run out of steps.",
    },
    "no_grounded_answer": {
        "area": "answer fallback",
        "action": "Add a best-grounded fallback path so retries submit a compact answer instead of empty output.",
    },
}


TASK_TYPE_RECOMMENDATIONS: dict[str, dict[str, str]] = {
    "threshold_count": {
        "area": "threshold-count solver",
        "action": "Build a reusable solver for population extraction, threshold grounding, and distinct entity counts.",
    },
    "rank_lookup": {
        "area": "rank lookup solver",
        "action": "Build schema-driven rank lookup helpers for rank/order/position fields and attached values.",
    },
    "ratio_or_percentage": {
        "area": "ratio solver",
        "action": "Add a ratio planner that logs numerator, denominator, unit, and final output format.",
    },
    "aggregation": {
        "area": "aggregation verifier",
        "action": "Verify aggregation operator, grouping grain, time normalization, and units before answering.",
    },
}


def _score_loss(row: dict[str, Any]) -> float:
    score = row.get("score")
    if isinstance(score, (int, float)):
        return max(0.0, 1.0 - float(score))
    return 0.0


def _build_recommendations(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    buckets: dict[str, dict[str, Any]] = {}

    def add(kind: str, key: str, row: dict[str, Any], rule: dict[str, str]) -> None:
        bucket_key = f"{kind}:{key}"
        bucket = buckets.setdefault(
            bucket_key,
            {
                "kind": kind,
                "key": key,
                "area": rule["area"],
                "action": rule["action"],
                "task_ids": [],
                "total_score_loss": 0.0,
            },
        )
        bucket["task_ids"].append(str(row["task_id"]))
        bucket["total_score_loss"] += _score_loss(row)

    for row in rows:
        for signal in row.get("signals") or []:
            rule = RECOMMENDATION_RULES.get(str(signal))
            if rule:
                add("signal", str(signal), row, rule)

        task_type = str(row.get("task_type") or "")
        rule = TASK_TYPE_RECOMMENDATIONS.get(task_type)
        if rule:
            add("task_type", task_type, row, rule)

    recommendations = list(buckets.values())
    for item in recommendations:
        item["task_ids"] = sorted(set(item["task_ids"]), key=lambda value: (len(value), value))
        item["task_count"] = len(item["task_ids"])
        item["total_score_loss"] = round(float(item["total_score_loss"]), 4)
        item["priority"] = round(item["task_count"] * 10 + float(item["total_score_loss"]) * 5, 4)
    recommendations.sort(key=lambda item: (-float(item["priority"]), str(item["kind"]), str(item["key"])))
    return recommendations


def analyze_run_failures(
    *,
    run_dir: Path,
    score_threshold: float = 1.0,
    evaluation_path: Path | None = None,
) -> FailureMiningResult:
    evaluation_file = evaluation_path or (run_dir / "evaluation.json")
    evaluation = _read_json(evaluation_file)
    evaluation_summary = _evaluation_summary(evaluation)
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
        "total_tasks_in_evaluation": evaluation_summary.get("total_tasks"),
        "overall_score": evaluation_summary.get("overall_score"),
        "score_threshold": score_threshold,
        "task_count_seen": len(rows),
        "low_score_task_count": len(low_score_rows),
    }
    return FailureMiningResult(
        summary=summary,
        tasks=low_score_rows,
        groups=groups,
        recommendations=_build_recommendations(low_score_rows),
    )


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

    if result.recommendations:
        lines.extend(["## Development Recommendations", ""])
        lines.append("| Priority | Area | Trigger | Tasks | Suggested Action |")
        lines.append("| ---: | --- | --- | --- | --- |")
        for item in result.recommendations[:12]:
            task_ids = ", ".join(f"`{task_id}`" for task_id in item.get("task_ids", [])[:8])
            if item.get("task_count", 0) > 8:
                task_ids += ", ..."
            lines.append(
                f"| {item.get('priority')} | {item.get('area')} | "
                f"{item.get('kind')}:{item.get('key')} | {task_ids or '-'} | "
                f"{item.get('action')} |"
            )
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
