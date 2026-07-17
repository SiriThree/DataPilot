from __future__ import annotations

import os
from pathlib import Path

import pytest

from data_agent_baseline.config import AgentConfig, AppConfig, RunConfig, load_app_config, load_dotenv
from data_agent_baseline.run.runner import run_single_task
from evaluate import evaluate


def _real_model_enabled() -> bool:
    return os.environ.get("RUN_REAL_MODEL_TESTS") == "1"


@pytest.mark.integration
def test_real_model_task_11_scores_high(tmp_path: Path) -> None:
    """Run one real benchmark task through the actual model and local evaluator.

    This test intentionally calls the configured model API. It is skipped by
    default and only runs when RUN_REAL_MODEL_TESTS=1 is set.
    """
    load_dotenv()
    if not _real_model_enabled():
        pytest.skip("set RUN_REAL_MODEL_TESTS=1 to run real model integration tests")
    if not os.environ.get("DEEPSEEK_API_KEY"):
        pytest.skip("DEEPSEEK_API_KEY is required for real model integration tests")

    project_root = Path(__file__).resolve().parents[1]
    config_path = project_root / "configs" / "react_baseline.local.yaml"
    data_root = project_root / "data" / "public"
    input_root = data_root / "input"
    gold_path = data_root / "output" / "task_11" / "gold.csv"
    if not (input_root / "task_11" / "task.json").exists() or not gold_path.exists():
        pytest.skip("Phase 1 demo dataset with task_11 is required")

    base_config = load_app_config(config_path)
    run_output_dir = tmp_path / "real_model_run"
    run_output_dir.mkdir()
    config = AppConfig(
        dataset=base_config.dataset,
        agent=AgentConfig(
            provider=base_config.agent.provider,
            model=base_config.agent.model,
            api_base=base_config.agent.api_base,
            api_key=base_config.agent.api_key,
            max_steps=base_config.agent.max_steps,
            temperature=base_config.agent.temperature,
            use_multi_agent=base_config.agent.use_multi_agent,
        ),
        run=RunConfig(
            output_dir=run_output_dir,
            max_workers=1,
            task_timeout_seconds=600,
        ),
    )

    artifact = run_single_task(
        task_id="task_11",
        config=config,
        run_output_dir=run_output_dir,
    )

    assert artifact.succeeded, artifact.failure_reason
    assert artifact.prediction_csv_path is not None
    assert artifact.prediction_csv_path.exists()
    assert artifact.trace_path.exists()

    score = evaluate(artifact.prediction_csv_path, gold_path)
    assert score["score"] >= 0.9
