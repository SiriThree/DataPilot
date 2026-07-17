#!/usr/bin/env python3
"""Submission entry point for KDD Cup 2026 DataAgent-Bench.

Reads from `/input`, writes predictions to `/output`, and logs to `/logs/runtime.log`.

Environment variables (provided by the evaluation system):
  MODEL_API_URL  — OpenAI-compatible API base URL
  MODEL_API_KEY  — API key
  MODEL_NAME     — Model name to use
"""

from __future__ import annotations

import csv
import json
import logging
import os
import sys
import traceback
from collections.abc import Mapping
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any

from data_agent_baseline.benchmark.dataset import DABenchPublicDataset
from data_agent_baseline.config import AgentConfig, AppConfig, DatasetConfig, RunConfig
from data_agent_baseline.run.runner import run_single_task

# Submission paths
INPUT_DIR = Path("/input")
OUTPUT_DIR = Path("/output")
LOGS_DIR = Path("/logs")
RUNTIME_LOG = LOGS_DIR / "runtime.log"
REQUIRED_MODEL_ENV_VARS = ("MODEL_NAME", "MODEL_API_URL", "MODEL_API_KEY")
DEFAULT_MAX_WORKERS = 2
DEFAULT_MAX_STEPS = 36
DEFAULT_TASK_TIMEOUT_SECONDS = 600
DEFAULT_TOTAL_TIMEOUT_SECONDS = 42_600  # 11h50m, leaving room before the 12h hard limit.
DEFAULT_SHUTDOWN_BUFFER_SECONDS = 300
MIN_TASK_START_SECONDS = 60
DIFFICULTY_TIMEOUT_SECONDS = {
    "easy": 300,
    "medium": 600,
    "hard": 1000,
    "extreme": 1600,
}


def setup_logging() -> None:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(RUNTIME_LOG, encoding="utf-8"),
            logging.StreamHandler(sys.stderr),
        ],
    )


def read_env_config(environ: Mapping[str, str] | None = None) -> dict[str, str]:
    """Read official model configuration from environment variables."""
    source = os.environ if environ is None else environ
    missing = [name for name in REQUIRED_MODEL_ENV_VARS if not source.get(name)]
    if missing:
        raise RuntimeError(
            "Missing official model environment variable(s): "
            + ", ".join(missing)
            + ". Submission runtime must use MODEL_NAME/MODEL_API_URL/MODEL_API_KEY."
        )

    return {
        "model": source["MODEL_NAME"],
        "api_base": source["MODEL_API_URL"],
        "api_key": source["MODEL_API_KEY"],
    }


def read_positive_int_env(name: str, default: int, *, minimum: int = 1) -> int:
    """Read a positive integer env var with a safe default."""
    raw_value = os.environ.get(name, "").strip()
    if not raw_value:
        return default
    try:
        parsed = int(raw_value)
    except ValueError:
        logging.warning("Invalid %s=%r; using default %d", name, raw_value, default)
        return default
    if parsed < minimum:
        logging.warning("%s=%d is below minimum %d; using %d", name, parsed, minimum, minimum)
        return minimum
    return parsed


def build_app_config(env: dict[str, str]) -> AppConfig:
    """Build AppConfig from official submission environment."""
    return AppConfig(
        dataset=DatasetConfig(root_path=INPUT_DIR),
        agent=AgentConfig(
            provider="openai_compatible",
            model=env["model"],
            api_base=env["api_base"],
            api_key=env["api_key"],
            max_steps=read_positive_int_env("SUBMISSION_MAX_STEPS", DEFAULT_MAX_STEPS),
            temperature=0.0,
        ),
        run=RunConfig(
            output_dir=OUTPUT_DIR,
            max_workers=read_positive_int_env("SUBMISSION_MAX_WORKERS", DEFAULT_MAX_WORKERS),
            task_timeout_seconds=read_positive_int_env(
                "SUBMISSION_TASK_TIMEOUT_SECONDS", DEFAULT_TASK_TIMEOUT_SECONDS
            ),
        ),
    )


def config_with_task_timeout(config: AppConfig, task_timeout_seconds: int) -> AppConfig:
    """Create a per-task config with a deadline-aware timeout."""
    return AppConfig(
        dataset=config.dataset,
        agent=config.agent,
        run=RunConfig(
            output_dir=config.run.output_dir,
            run_id=config.run.run_id,
            max_workers=config.run.max_workers,
            task_timeout_seconds=task_timeout_seconds,
        ),
    )


def task_timeout_for_difficulty(config: AppConfig, difficulty: str | None) -> int:
    """Return per-task timeout, using difficulty defaults unless env overrides the global timeout."""
    if config.run.task_timeout_seconds != DEFAULT_TASK_TIMEOUT_SECONDS:
        return config.run.task_timeout_seconds
    normalized = (difficulty or "").strip().lower()
    return DIFFICULTY_TIMEOUT_SECONDS.get(normalized, config.run.task_timeout_seconds)


def write_fallback_task_output(task_id: str, reason: str) -> dict[str, Any]:
    """Write a valid empty prediction for tasks that fail before runner output exists."""
    task_output_dir = OUTPUT_DIR / task_id
    task_output_dir.mkdir(parents=True, exist_ok=True)

    prediction_path = task_output_dir / "prediction.csv"
    with prediction_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["answer"])

    payload = {
        "task_id": task_id,
        "answer": None,
        "steps": [],
        "failure_reason": reason,
        "succeeded": False,
        "prediction_csv_path": str(prediction_path),
    }
    (task_output_dir / "trace.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return payload


def run_task(task_id: str, config: AppConfig) -> dict[str, Any]:
    """Run a single task through the same optimized pipeline used by local runs."""
    artifact = run_single_task(
        task_id=task_id,
        config=config,
        run_output_dir=OUTPUT_DIR,
    )
    logging.info(
        "task_id=%s prediction=%s trace=%s succeeded=%s",
        task_id,
        artifact.prediction_csv_path,
        artifact.trace_path,
        artifact.succeeded,
    )

    if artifact.trace_path.exists():
        return json.loads(artifact.trace_path.read_text(encoding="utf-8"))
    return artifact.to_dict()


def submit_task(
    *,
    executor: ThreadPoolExecutor,
    task_id: str,
    difficulty: str | None,
    task_index: int,
    task_count: int,
    config: AppConfig,
    deadline_at: float,
    shutdown_buffer_seconds: int,
) -> Future[dict[str, Any]] | None:
    """Submit one task if enough global time remains."""
    seconds_left_for_work = int(deadline_at - perf_counter() - shutdown_buffer_seconds)
    if seconds_left_for_work < MIN_TASK_START_SECONDS:
        return None

    desired_timeout = task_timeout_for_difficulty(config, difficulty)
    task_timeout_seconds = min(desired_timeout, seconds_left_for_work)
    task_config = config_with_task_timeout(config, task_timeout_seconds)
    logging.info(
        "task_id=%s [%d/%d] difficulty=%s starting timeout=%ss",
        task_id,
        task_index + 1,
        task_count,
        difficulty or "unknown",
        task_timeout_seconds,
    )
    return executor.submit(run_task, task_id, task_config)


def write_submission_summary(
    *,
    task_count: int,
    succeeded_count: int,
    failed_count: int,
    skipped_count: int,
    elapsed_seconds: float,
    config: AppConfig,
) -> None:
    summary_path = OUTPUT_DIR / "submission_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "task_count": task_count,
                "succeeded_task_count": succeeded_count,
                "failed_task_count": failed_count,
                "skipped_task_count": skipped_count,
                "elapsed_seconds": elapsed_seconds,
                "model": config.agent.model,
                "api_base": config.agent.api_base,
                "max_workers": config.run.max_workers,
                "max_steps": config.agent.max_steps,
                "task_timeout_seconds": config.run.task_timeout_seconds,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> None:
    setup_logging()
    logging.info("DataAgent submission starting at %s", datetime.now(timezone.utc).isoformat())

    try:
        env = read_env_config()
    except RuntimeError as exc:
        logging.error("%s", exc)
        sys.exit(2)

    config = build_app_config(env)
    total_timeout_seconds = read_positive_int_env(
        "SUBMISSION_TOTAL_TIMEOUT_SECONDS", DEFAULT_TOTAL_TIMEOUT_SECONDS
    )
    shutdown_buffer_seconds = read_positive_int_env(
        "SUBMISSION_SHUTDOWN_BUFFER_SECONDS", DEFAULT_SHUTDOWN_BUFFER_SECONDS, minimum=30
    )

    logging.info(
        "model=%s api_base=%s max_workers=%d task_timeout=%ss total_budget=%ss",
        config.agent.model,
        config.agent.api_base,
        config.run.max_workers,
        config.run.task_timeout_seconds,
        total_timeout_seconds,
    )

    # Discover tasks
    dataset = DABenchPublicDataset(config.dataset.root_path)
    task_ids = sorted(dataset.list_task_ids())

    if not task_ids:
        logging.error("No tasks found in %s", config.dataset.root_path)
        sys.exit(1)

    logging.info("tasks=%d", len(task_ids))

    total_started = perf_counter()
    deadline_at = total_started + total_timeout_seconds
    succeeded_count = 0
    failed_count = 0
    skipped_count = 0
    next_task_index = 0
    running: dict[Future[dict[str, Any]], tuple[str, int]] = {}

    with ThreadPoolExecutor(max_workers=config.run.max_workers) as executor:
        while next_task_index < len(task_ids) and len(running) < config.run.max_workers:
            task_id = task_ids[next_task_index]
            future = submit_task(
                executor=executor,
                task_id=task_id,
                difficulty=dataset.get_task(task_id).difficulty,
                task_index=next_task_index,
                task_count=len(task_ids),
                config=config,
                deadline_at=deadline_at,
                shutdown_buffer_seconds=shutdown_buffer_seconds,
            )
            if future is None:
                break
            running[future] = (task_id, next_task_index)
            next_task_index += 1

        while running:
            done, _ = wait(running, timeout=10, return_when=FIRST_COMPLETED)
            if not done:
                if perf_counter() >= deadline_at - shutdown_buffer_seconds:
                    logging.warning(
                        "Global deadline buffer reached; waiting only for already-running tasks."
                    )
                continue

            for future in done:
                task_id, _task_index = running.pop(future)
                try:
                    result = future.result()
                except Exception:
                    failed_count += 1
                    logging.error("task_id=%s exception:\n%s", task_id, traceback.format_exc())
                    write_fallback_task_output(task_id, "Task exception before output was written.")
                    continue

                if result.get("succeeded"):
                    succeeded_count += 1
                else:
                    failed_count += 1
                    reason = result.get("failure_reason", "unknown")
                    logging.warning("task_id=%s failed: %s", task_id, reason)

                while next_task_index < len(task_ids) and len(running) < config.run.max_workers:
                    next_task_id = task_ids[next_task_index]
                    new_future = submit_task(
                        executor=executor,
                        task_id=next_task_id,
                        difficulty=dataset.get_task(next_task_id).difficulty,
                        task_index=next_task_index,
                        task_count=len(task_ids),
                        config=config,
                        deadline_at=deadline_at,
                        shutdown_buffer_seconds=shutdown_buffer_seconds,
                    )
                    if new_future is None:
                        break
                    running[new_future] = (next_task_id, next_task_index)
                    next_task_index += 1

    for skipped_task_id in task_ids[next_task_index:]:
        skipped_count += 1
        failed_count += 1
        logging.warning("task_id=%s skipped because global deadline is near.", skipped_task_id)
        write_fallback_task_output(skipped_task_id, "Skipped because global runtime deadline is near.")

    total_elapsed = round(perf_counter() - total_started, 1)
    logging.info(
        "Finished: tasks=%d succeeded=%d failed=%d skipped=%d elapsed=%ss",
        len(task_ids), succeeded_count, failed_count, skipped_count, total_elapsed,
    )

    write_submission_summary(
        task_count=len(task_ids),
        succeeded_count=succeeded_count,
        failed_count=failed_count,
        skipped_count=skipped_count,
        elapsed_seconds=total_elapsed,
        config=config,
    )


if __name__ == "__main__":
    main()
