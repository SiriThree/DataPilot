from __future__ import annotations

import csv
import json
import shutil
from collections import Counter, defaultdict
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any

from data_agent_baseline.agents.model import OllamaModelAdapter, OpenAIModelAdapter
from data_agent_baseline.agents.multi_agent import MultiAgentLoop, MultiAgentConfig
from data_agent_baseline.agents.react import ReActAgent, ReActAgentConfig
from data_agent_baseline.benchmark.dataset import DABenchPublicDataset
from data_agent_baseline.config import AgentConfig, AppConfig
from data_agent_baseline.run.difficulty_policy import (
    StrategyPolicy,
    build_strategy_policy,
    build_strategy_prompt_hint,
    strategy_policy_to_dict,
)
from data_agent_baseline.run.failure_analysis import analyze_task_failure, failure_analysis_to_dict
from data_agent_baseline.run.guided_retry import (
    attempt_selection_to_dict,
    build_guided_retry_decision,
    guided_retry_decision_to_dict,
    select_best_attempt,
)
from data_agent_baseline.run.repair import repair_and_reverify, normalize_prediction_csv
from data_agent_baseline.run.route_decision import build_route_prompt_hint, decide_route, route_to_dict
from data_agent_baseline.run.verification_chain import (
    infer_output_contract,
    run_full_verification,
    verification_report_to_dict,
)
from data_agent_baseline.tools.registry import ToolRegistry, create_default_tool_registry
from data_agent_baseline.tools.stateful_python import remove_interpreter
from data_agent_baseline.tools.verifier import verify_answer


@dataclass(frozen=True, slots=True)
class TaskRunArtifacts:
    task_id: str
    task_output_dir: Path
    prediction_csv_path: Path | None
    trace_path: Path
    succeeded: bool
    failure_reason: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "task_output_dir": str(self.task_output_dir),
            "prediction_csv_path": str(self.prediction_csv_path) if self.prediction_csv_path else None,
            "trace_path": str(self.trace_path),
            "succeeded": self.succeeded,
            "failure_reason": self.failure_reason,
        }


def create_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def resolve_run_id(run_id: str | None = None) -> str:
    if run_id is None:
        return create_run_id()

    normalized = run_id.strip()
    if not normalized:
        raise ValueError("run_id must not be empty.")
    if normalized in {".", ".."} or "/" in normalized or "\\" in normalized:
        raise ValueError("run_id must be a single directory name, not a path.")
    return normalized


def create_run_output_dir(output_root: Path, *, run_id: str | None = None) -> tuple[str, Path]:
    effective_run_id = resolve_run_id(run_id)
    run_output_dir = output_root / effective_run_id
    run_output_dir.mkdir(parents=True, exist_ok=False)
    return effective_run_id, run_output_dir


def build_model_adapter(config: AppConfig):
    provider = (config.agent.provider or "openai_compatible").strip().lower()
    if provider == "ollama":
        return OllamaModelAdapter(
            model=config.agent.model,
            api_base=config.agent.api_base or "http://localhost:11434",
            temperature=config.agent.temperature,
            timeout=120.0,
            max_retries=1,
        )
    if provider not in {"openai", "openai_compatible"}:
        raise ValueError(f"Unsupported agent.provider: {config.agent.provider}")
    return OpenAIModelAdapter(
        model=config.agent.model,
        api_base=config.agent.api_base,
        api_key=config.agent.api_key,
        temperature=config.agent.temperature,
        timeout=120.0,
        max_retries=3,
    )


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, columns: list[str], rows: list[list[Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        for row in rows:
            writer.writerow(row)


def _write_failure_analysis_artifact(task_output_dir: Path, analysis: dict[str, Any]) -> None:
    _write_json(task_output_dir / "failure_analysis.json", analysis)


def _write_run_failure_taxonomy(run_output_dir: Path, artifacts: list[TaskRunArtifacts]) -> None:
    signal_counts: Counter[str] = Counter()
    severity_counts: Counter[str] = Counter()
    route_counts: Counter[str] = Counter()
    note_counts: Counter[str] = Counter()
    tasks_by_signal: dict[str, list[str]] = defaultdict(list)
    jsonl_rows: list[dict[str, Any]] = []

    for artifact in artifacts:
        analysis_path = artifact.task_output_dir / "failure_analysis.json"
        if not analysis_path.exists():
            continue
        try:
            analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue

        commitments = analysis.get("commitments", {}) if isinstance(analysis, dict) else {}
        route = str(commitments.get("route") or "unknown")
        route_counts[route] += 1
        task_id = str(analysis.get("task_id") or artifact.task_id)
        task_codes: list[str] = []
        for signal in analysis.get("signals", []) or []:
            if not isinstance(signal, dict):
                continue
            code = str(signal.get("code") or "unknown")
            severity = str(signal.get("severity") or "unknown")
            signal_counts[code] += 1
            severity_counts[severity] += 1
            tasks_by_signal[code].append(task_id)
            task_codes.append(code)
        for note in analysis.get("evolution_notes", []) or []:
            note_counts[str(note)] += 1
        jsonl_rows.append({
            "task_id": task_id,
            "succeeded": artifact.succeeded,
            "route": route,
            "signals": task_codes,
            "evolution_notes": analysis.get("evolution_notes", []),
        })

    taxonomy = {
        "task_count": len(artifacts),
        "signal_counts": dict(signal_counts.most_common()),
        "severity_counts": dict(severity_counts.most_common()),
        "route_counts": dict(route_counts.most_common()),
        "tasks_by_signal": {code: sorted(set(tasks)) for code, tasks in sorted(tasks_by_signal.items())},
        "top_evolution_notes": [
            {"note": note, "count": count}
            for note, count in note_counts.most_common(20)
        ],
    }
    _write_json(run_output_dir / "failure_taxonomy.json", taxonomy)
    with (run_output_dir / "failure_taxonomy.jsonl").open("w", encoding="utf-8") as handle:
        for row in jsonl_rows:
            handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def _failure_run_result_payload(task_id: str, failure_reason: str) -> dict[str, Any]:
    return {
        "task_id": task_id,
        "answer": None,
        "steps": [],
        "failure_reason": failure_reason,
        "succeeded": False,
    }


def _run_single_task_core(
    *,
    task_id: str,
    config: AppConfig,
    model=None,
    tools: ToolRegistry | None = None,
    route_hint: str = "",
    strategy_policy: StrategyPolicy | None = None,
) -> tuple[dict[str, Any], str]:
    public_dataset = DABenchPublicDataset(config.dataset.root_path)
    task = public_dataset.get_task(task_id)

    if config.agent.use_multi_agent:
        use_decomposer = (
            strategy_policy.use_decomposer
            if strategy_policy is not None
            else (task.difficulty or "medium") in ("hard", "extreme")
        )
        agent = MultiAgentLoop(
            model=model or build_model_adapter(config),
            tools=tools or create_default_tool_registry(),
            config=MultiAgentConfig(
                max_steps=config.agent.max_steps,
                use_verifier=strategy_policy.use_verifier if strategy_policy is not None else True,
                use_router=strategy_policy.use_router if strategy_policy is not None else True,
                use_debugger=strategy_policy.use_debugger if strategy_policy is not None else True,
                use_decomposer=use_decomposer,
                verifier_frequency=(
                    strategy_policy.verifier_frequency
                    if strategy_policy is not None
                    else "every_compute"
                ),
            ),
            route_hint=route_hint,
            difficulty=task.difficulty or "medium",
        )
    else:
        agent = ReActAgent(
            model=model or build_model_adapter(config),
            tools=tools or create_default_tool_registry(),
            config=ReActAgentConfig(max_steps=config.agent.max_steps),
            route_hint=route_hint,
        )
    run_result = agent.run(task)
    return run_result.to_dict(), task.question


def _run_single_task_with_timeout(
    *,
    task_id: str,
    config: AppConfig,
    route_hint: str = "",
    strategy_policy: StrategyPolicy | None = None,
) -> tuple[dict[str, Any], str]:
    timeout_seconds = config.run.task_timeout_seconds
    if timeout_seconds <= 0:
        return _run_single_task_core(
            task_id=task_id,
            config=config,
            route_hint=route_hint,
            strategy_policy=strategy_policy,
        )

    # Use Thread instead of multiprocessing.Process to avoid macOS spawn/fork issues
    import threading
    result_container: dict[str, Any] = {}

    def _target() -> None:
        try:
            run_result, question = _run_single_task_core(
                task_id=task_id,
                config=config,
                route_hint=route_hint,
                strategy_policy=strategy_policy,
            )
            result_container["ok"] = True
            result_container["run_result"] = run_result
            result_container["question"] = question
        except Exception as exc:
            result_container["ok"] = False
            result_container["error"] = str(exc)

    thread = threading.Thread(target=_target, daemon=True)
    thread.start()
    thread.join(timeout=timeout_seconds)

    if thread.is_alive():
        return _failure_run_result_payload(task_id, f"Task timed out after {timeout_seconds} seconds."), ""

    if result_container.get("ok"):
        return dict(result_container["run_result"]), result_container.get("question", "")
    return _failure_run_result_payload(task_id, f"Task failed: {result_container.get('error', 'unknown')}"), ""


def _write_task_outputs(
    task_id: str, run_output_dir: Path, run_result: dict[str, Any], question: str = "",
    task_dir: Path | None = None,
    task_output_dir_override: Path | None = None,
) -> TaskRunArtifacts:
    task_output_dir = task_output_dir_override or (run_output_dir / task_id)
    task_output_dir.mkdir(parents=True, exist_ok=True)

    prediction_csv_path: Path | None = None
    post_process_log: dict[str, Any] = {}
    post_verification = None

    answer = run_result.get("answer")
    if isinstance(answer, dict):
        columns = list(answer.get("columns", []))
        rows = [list(row) for row in answer.get("rows", [])]

        # Stage 1: Simple verifier (in-memory)
        verification = verify_answer(columns, rows, question)
        run_result["_verification"] = verification

        prediction_csv_path = task_output_dir / "prediction.csv"
        _write_csv(prediction_csv_path, columns, rows)

        # Stage 2: 5-layer verification chain
        output_contract = None
        if task_dir is not None:
            output_contract = infer_output_contract(task_dir)
        verification_report = run_full_verification(task_id, prediction_csv_path, output_contract)
        post_process_log["verification_report"] = verification_report_to_dict(verification_report)

        # Stage 3: Repair + re-verify
        post_verification, repair_plan, repair_execution = repair_and_reverify(
            task_id, prediction_csv_path, verification_report, question, task_dir=task_dir,
        )
        post_process_log["repair_plan"] = {
            "should_repair": repair_plan.should_repair,
            "summary": repair_plan.summary,
            "actions": [{"priority": a.priority, "action": a.action_type, "detail": a.detail}
                        for a in repair_plan.actions],
        }
        post_process_log["repair_execution"] = {
            "applied_count": repair_execution.applied_count,
            "skipped_count": repair_execution.skipped_count,
            "actions": repair_execution.actions,
        }
        post_process_log["post_repair_verification"] = verification_report_to_dict(post_verification)

    else:
        # Write fallback even when no answer
        prediction_csv_path = task_output_dir / "prediction.csv"
        normalize_prediction_csv(prediction_csv_path)
        output_contract = infer_output_contract(task_dir) if task_dir is not None else None
        post_verification = run_full_verification(task_id, prediction_csv_path, output_contract)
        post_process_log["verification_report"] = verification_report_to_dict(post_verification)

    run_result["_post_process"] = post_process_log

    succeeded = bool(run_result.get("succeeded"))
    # Also succeed if post-repair verification passes
    if not succeeded and post_process_log.get("post_repair_verification", {}).get("all_passed"):
        succeeded = True
        run_result["failure_reason"] = None
    run_result["succeeded"] = succeeded

    failure_analysis = failure_analysis_to_dict(
        analyze_task_failure(
            task_id=task_id,
            question=question,
            run_result=run_result,
            prediction_path=prediction_csv_path,
            verification_report=post_verification,
        )
    )
    run_result["_failure_analysis"] = failure_analysis
    _write_failure_analysis_artifact(task_output_dir, failure_analysis)

    trace_path = task_output_dir / "trace.json"
    _write_json(trace_path, run_result)

    return TaskRunArtifacts(
        task_id=task_id,
        task_output_dir=task_output_dir,
        prediction_csv_path=prediction_csv_path,
        trace_path=trace_path,
        succeeded=succeeded,
        failure_reason=run_result.get("failure_reason"),
    )


def _load_trace(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _config_with_retry_budget(config: AppConfig, *, max_steps: int, timeout_seconds: int) -> AppConfig:
    return AppConfig(
        dataset=config.dataset,
        agent=AgentConfig(
            provider=config.agent.provider,
            model=config.agent.model,
            api_base=config.agent.api_base,
            api_key=config.agent.api_key,
            max_steps=max_steps,
            temperature=config.agent.temperature,
            use_multi_agent=config.agent.use_multi_agent,
        ),
        run=type(config.run)(
            output_dir=config.run.output_dir,
            run_id=config.run.run_id,
            max_workers=config.run.max_workers,
            task_timeout_seconds=timeout_seconds,
        ),
    )


def _retry_step_budget(*, effective_max_steps: int, base_max_steps: int, budget_exhausted: bool) -> int:
    if not budget_exhausted:
        return effective_max_steps
    return min(effective_max_steps + 12, max(base_max_steps, 48))


def _run_attempt(
    *,
    task_id: str,
    config: AppConfig,
    route_hint: str,
    model,
    tools: ToolRegistry | None,
    strategy_policy: StrategyPolicy | None = None,
) -> tuple[dict[str, Any], str]:
    if model is None and tools is None:
        return _run_single_task_with_timeout(
            task_id=task_id,
            config=config,
            route_hint=route_hint,
            strategy_policy=strategy_policy,
        )
    return _run_single_task_core(
        task_id=task_id,
        config=config,
        model=model,
        tools=tools or create_default_tool_registry(),
        route_hint=route_hint,
        strategy_policy=strategy_policy,
    )


def _prediction_csvs_equal(path_a: Path | None, path_b: Path | None) -> bool:
    """Return True if both prediction CSV files exist and have identical content."""
    if path_a is None or path_b is None:
        return False
    if not path_a.exists() or not path_b.exists():
        return False
    try:
        return path_a.read_text(encoding="utf-8-sig") == path_b.read_text(encoding="utf-8-sig")
    except (OSError, UnicodeDecodeError):
        return False


def _maybe_run_guided_retry(
    *,
    artifact: TaskRunArtifacts,
    task_id: str,
    question: str,
    task_dir: Path,
    run_output_dir: Path,
    route_hint: str,
    route_decision_payload: dict[str, Any],
    strategy_policy_payload: dict[str, Any],
    strategy_policy: StrategyPolicy | None,
    config: AppConfig,
    effective_max_steps: int,
    started_at: float,
    model,
) -> TaskRunArtifacts:
    original_trace = _load_trace(artifact.trace_path)
    if not config.run.enable_guided_retry:
        original_trace["_guided_retry"] = {
            "decision": {
                "should_retry": False,
                "reason": "guided retry disabled by run.enable_guided_retry",
                "trigger_codes": [],
                "retry_hint": "",
            },
            "attempted": False,
        }
        _write_json(artifact.trace_path, original_trace)
        return artifact

    decision = build_guided_retry_decision(
        original_trace=original_trace,
        prediction_path=artifact.prediction_csv_path,
        route_decision_payload=route_decision_payload,
    )
    original_trace["_guided_retry"] = {
        "decision": guided_retry_decision_to_dict(decision),
        "attempted": False,
    }
    if not decision.should_retry:
        _write_json(artifact.trace_path, original_trace)
        return artifact

    elapsed = perf_counter() - started_at
    remaining_timeout = max(int(config.run.task_timeout_seconds - elapsed), 0)
    if config.run.task_timeout_seconds > 0 and remaining_timeout < 60:
        original_trace["_guided_retry"]["skip_reason"] = (
            f"not enough per-task timeout remains for retry ({remaining_timeout}s)"
        )
        _write_json(artifact.trace_path, original_trace)
        return artifact

    original_prediction_copy = artifact.task_output_dir / "prediction.original.csv"
    original_trace_copy = artifact.task_output_dir / "trace.original.json"
    if artifact.prediction_csv_path and artifact.prediction_csv_path.exists():
        shutil.copy2(artifact.prediction_csv_path, original_prediction_copy)
    shutil.copy2(artifact.trace_path, original_trace_copy)

    budget_exhausted = "budget_exhaustion" in decision.trigger_codes
    retry_steps = _retry_step_budget(
        effective_max_steps=effective_max_steps,
        base_max_steps=config.agent.max_steps,
        budget_exhausted=budget_exhausted,
    )
    retry_timeout = remaining_timeout if config.run.task_timeout_seconds > 0 else config.run.task_timeout_seconds
    retry_config = _config_with_retry_budget(config, max_steps=retry_steps, timeout_seconds=retry_timeout)
    retry_route_hint = (
        f"{route_hint}\n\n"
        "<guided_retry>\n"
        f"{decision.retry_hint}\n"
        "</guided_retry>"
    )

    retry_started = perf_counter()
    retry_tools = create_default_tool_registry() if model is not None else None
    retry_run_result, _ = _run_attempt(
        task_id=task_id,
        config=retry_config,
        route_hint=retry_route_hint,
        model=model,
        tools=retry_tools,
        strategy_policy=strategy_policy,
    )
    retry_run_result["e2e_elapsed_seconds"] = round(perf_counter() - retry_started, 3)
    retry_run_result["_route_decision"] = route_decision_payload
    retry_run_result["_strategy_policy"] = strategy_policy_payload

    retry_artifact = _write_task_outputs(
        task_id,
        run_output_dir,
        retry_run_result,
        question,
        task_dir=task_dir,
        task_output_dir_override=artifact.task_output_dir / "_retry",
    )
    retry_trace = _load_trace(retry_artifact.trace_path)

    # Same-answer detection: if retry produced identical prediction, skip
    # the full select_best_attempt + re-verification pipeline.
    if _prediction_csvs_equal(original_prediction_copy, retry_artifact.prediction_csv_path):
        original_trace["_guided_retry"] = {
            "decision": guided_retry_decision_to_dict(decision),
            "attempted": True,
            "retry_steps": retry_steps,
            "retry_timeout_seconds": retry_timeout,
            "original_prediction": str(original_prediction_copy),
            "original_trace": str(original_trace_copy),
            "retry_prediction": str(retry_artifact.prediction_csv_path) if retry_artifact.prediction_csv_path else None,
            "retry_trace": str(retry_artifact.trace_path),
            "skip_reason": "retry produced identical prediction; keeping original",
        }
        _write_json(artifact.trace_path, original_trace)
        return artifact

    selection = select_best_attempt(
        original_trace=original_trace,
        retry_trace=retry_trace,
        trigger_codes=decision.trigger_codes,
    )

    retry_payload = {
        "decision": guided_retry_decision_to_dict(decision),
        "attempted": True,
        "retry_steps": retry_steps,
        "retry_timeout_seconds": retry_timeout,
        "original_prediction": str(original_prediction_copy),
        "original_trace": str(original_trace_copy),
        "retry_prediction": str(retry_artifact.prediction_csv_path) if retry_artifact.prediction_csv_path else None,
        "retry_trace": str(retry_artifact.trace_path),
        "selection": attempt_selection_to_dict(selection),
    }

    if selection.selected_attempt == "retry" and retry_artifact.prediction_csv_path:
        shutil.copy2(retry_artifact.prediction_csv_path, artifact.prediction_csv_path)
        final_trace = dict(retry_trace)
        final_trace["_guided_retry"] = retry_payload
        final_trace["_selected_attempt"] = "retry"
        final_trace["_route_decision"] = route_decision_payload
        final_trace["_strategy_policy"] = strategy_policy_payload
        _write_json(artifact.trace_path, final_trace)
        return TaskRunArtifacts(
            task_id=artifact.task_id,
            task_output_dir=artifact.task_output_dir,
            prediction_csv_path=artifact.prediction_csv_path,
            trace_path=artifact.trace_path,
            succeeded=retry_artifact.succeeded,
            failure_reason=retry_artifact.failure_reason,
        )

    original_trace["_guided_retry"] = retry_payload
    original_trace["_selected_attempt"] = "original"
    _write_json(artifact.trace_path, original_trace)
    return artifact


def run_single_task(
    *,
    task_id: str,
    config: AppConfig,
    run_output_dir: Path,
    model=None,
    tools: ToolRegistry | None = None,
) -> TaskRunArtifacts:
    started_at = perf_counter()

    # ── Pre-run: route decision ──
    public_dataset = DABenchPublicDataset(config.dataset.root_path)
    task = public_dataset.get_task(task_id)
    difficulty = task.difficulty
    if not difficulty:
        import logging
        logging.getLogger("dabench").warning(
            "task %s has no difficulty set; defaulting to 'medium' for route scoring",
            task_id,
        )
        difficulty = "medium"
    route_decision = decide_route(
        question=task.question,
        context_dir=task.context_dir,
        difficulty=difficulty,
    )
    strategy_policy = build_strategy_policy(
        difficulty=difficulty,
        route_decision=route_decision,
        configured_max_steps=config.agent.max_steps,
        configured_use_multi_agent=config.agent.use_multi_agent,
        configured_enable_guided_retry=config.run.enable_guided_retry,
    )
    strategy_policy_payload = strategy_policy_to_dict(strategy_policy)
    route_hint = (
        f"{build_route_prompt_hint(route_decision)}\n\n"
        f"{build_strategy_prompt_hint(strategy_policy)}"
    )

    # ── Difficulty/route/task-profile strategy ──
    effective_max_steps = strategy_policy.max_steps
    adapted_config = AppConfig(
        dataset=config.dataset,
        agent=AgentConfig(
            provider=config.agent.provider,
            model=config.agent.model,
            api_base=config.agent.api_base,
            api_key=config.agent.api_key,
            max_steps=effective_max_steps,
            temperature=config.agent.temperature,
            use_multi_agent=strategy_policy.use_multi_agent,
        ),
        run=type(config.run)(
            output_dir=config.run.output_dir,
            run_id=config.run.run_id,
            max_workers=config.run.max_workers,
            task_timeout_seconds=config.run.task_timeout_seconds,
            enable_guided_retry=strategy_policy.enable_guided_retry,
        ),
    )

    question = task.question
    run_result, question = _run_attempt(
        task_id=task_id,
        config=adapted_config,
        route_hint=route_hint,
        model=model,
        tools=tools,
        strategy_policy=strategy_policy,
    )
    run_result["e2e_elapsed_seconds"] = round(perf_counter() - started_at, 3)
    route_decision_payload = route_to_dict(route_decision)
    run_result["_route_decision"] = route_decision_payload
    run_result["_strategy_policy"] = strategy_policy_payload
    artifact = _write_task_outputs(task_id, run_output_dir, run_result, question, task_dir=task.task_dir)
    # Cleanup stateful interpreter to free memory
    remove_interpreter(task_id)
    return _maybe_run_guided_retry(
        artifact=artifact,
        task_id=task_id,
        question=question,
        task_dir=task.task_dir,
        run_output_dir=run_output_dir,
        route_hint=route_hint,
        route_decision_payload=route_decision_payload,
        strategy_policy_payload=strategy_policy_payload,
        strategy_policy=strategy_policy,
        config=adapted_config,
        effective_max_steps=effective_max_steps,
        started_at=started_at,
        model=model,
    )


def run_benchmark(
    *,
    config: AppConfig,
    model=None,
    tools: ToolRegistry | None = None,
    limit: int | None = None,
    progress_callback: Callable[[TaskRunArtifacts], None] | None = None,
) -> tuple[Path, list[TaskRunArtifacts]]:
    effective_run_id, run_output_dir = create_run_output_dir(config.run.output_dir, run_id=config.run.run_id)

    dataset = DABenchPublicDataset(config.dataset.root_path)
    tasks = dataset.iter_tasks()
    if limit is not None:
        tasks = tasks[:limit]

    effective_workers = config.run.max_workers
    if effective_workers < 1:
        raise ValueError("max_workers must be at least 1.")
    if model is not None or tools is not None:
        effective_workers = 1

    task_ids = [task.task_id for task in tasks]

    task_artifacts: list[TaskRunArtifacts]
    if effective_workers == 1:
        shared_model = model or build_model_adapter(config)
        shared_tools = tools or create_default_tool_registry()
        task_artifacts = []
        for task_id in task_ids:
            artifact = run_single_task(
                task_id=task_id,
                config=config,
                run_output_dir=run_output_dir,
                model=shared_model,
                tools=shared_tools,
            )
            task_artifacts.append(artifact)
            if progress_callback is not None:
                progress_callback(artifact)
    else:
        with ThreadPoolExecutor(max_workers=effective_workers) as executor:
            future_to_index = {
                executor.submit(
                    run_single_task,
                    task_id=task_id,
                    config=config,
                    run_output_dir=run_output_dir,
                ): index
                for index, task_id in enumerate(task_ids)
            }
            indexed_artifacts: list[TaskRunArtifacts | None] = [None] * len(task_ids)
            for future in as_completed(future_to_index):
                artifact = future.result()
                indexed_artifacts[future_to_index[future]] = artifact
                if progress_callback is not None:
                    progress_callback(artifact)
            task_artifacts = [artifact for artifact in indexed_artifacts if artifact is not None]

    summary_path = run_output_dir / "summary.json"
    _write_json(
        summary_path,
        {
            "run_id": effective_run_id,
            "task_count": len(task_artifacts),
            "succeeded_task_count": sum(1 for artifact in task_artifacts if artifact.succeeded),
            "max_workers": effective_workers,
            "tasks": [artifact.to_dict() for artifact in task_artifacts],
        },
    )
    _write_run_failure_taxonomy(run_output_dir, task_artifacts)
    return run_output_dir, task_artifacts
