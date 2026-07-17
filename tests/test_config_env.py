from __future__ import annotations

from pathlib import Path

from data_agent_baseline.config import load_app_config, load_dotenv


def test_load_dotenv_reads_values_without_overriding(monkeypatch, tmp_path: Path) -> None:
    env_path = tmp_path / ".env"
    env_path.write_text(
        "\ufeffDEEPSEEK_API_KEY=file-key\n"
        "DEEPSEEK_MODEL='file-model'\n"
        'DEEPSEEK_API_BASE="https://file.example/v1"\n',
        encoding="utf-8",
    )

    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.setenv("DEEPSEEK_MODEL", "real-env-model")
    monkeypatch.delenv("DEEPSEEK_API_BASE", raising=False)

    load_dotenv(env_path)

    assert load_app_config
    assert __import__("os").environ["DEEPSEEK_API_KEY"] == "file-key"
    assert __import__("os").environ["DEEPSEEK_MODEL"] == "real-env-model"
    assert __import__("os").environ["DEEPSEEK_API_BASE"] == "https://file.example/v1"


def test_load_app_config_resolves_dotenv_placeholders(monkeypatch, tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
dataset:
  root_path: data/public/input
agent:
  model: ${DEEPSEEK_MODEL:-fallback-model}
  api_base: ${DEEPSEEK_API_BASE:-https://fallback.example}
  api_key: ${DEEPSEEK_API_KEY}
  max_steps: 7
  temperature: 0.0
run:
  output_dir: artifacts/runs
  max_workers: 1
  task_timeout_seconds: 10
""".strip(),
        encoding="utf-8",
    )
    env_path = tmp_path / ".env"
    env_path.write_text(
        "DEEPSEEK_API_KEY=dotenv-key\nDEEPSEEK_MODEL=dotenv-model\n",
        encoding="utf-8",
    )

    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.delenv("DEEPSEEK_MODEL", raising=False)
    monkeypatch.setenv("DEEPSEEK_API_BASE", "")
    monkeypatch.chdir(tmp_path)

    # load_app_config reads the project .env; load this temp env explicitly for
    # an isolated test of the same parsing and precedence behavior.
    load_dotenv(env_path)
    config = load_app_config(config_path)

    assert config.agent.api_key == "dotenv-key"
    assert config.agent.model == "dotenv-model"
    assert config.agent.api_base == "https://fallback.example"
    assert config.agent.max_steps == 7
