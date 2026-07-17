from __future__ import annotations

import json
from pathlib import Path

import pytest

from data_agent_baseline.agents.model import ModelMessage, OllamaModelAdapter, OpenAIModelAdapter
from data_agent_baseline.config import AgentConfig, AppConfig, DatasetConfig, RunConfig, load_app_config
from data_agent_baseline.run.runner import build_model_adapter


def test_load_app_config_reads_agent_provider(tmp_path: Path) -> None:
    config_path = tmp_path / "ollama.yaml"
    config_path.write_text(
        """
dataset:
  root_path: data/public/input
agent:
  provider: ollama
  model: qwen2.5:14b
  api_base: http://localhost:11434
  api_key:
run:
  output_dir: artifacts/runs
  enable_guided_retry: false
""",
        encoding="utf-8",
    )

    config = load_app_config(config_path)

    assert config.agent.provider == "ollama"
    assert config.agent.model == "qwen2.5:14b"
    assert config.agent.api_base == "http://localhost:11434"
    assert config.agent.api_key == ""
    assert config.run.enable_guided_retry is False


def test_build_model_adapter_selects_ollama() -> None:
    config = AppConfig(
        dataset=DatasetConfig(),
        agent=AgentConfig(
            provider="ollama",
            model="qwen2.5:14b",
            api_base="http://localhost:11434",
        ),
        run=RunConfig(),
    )

    adapter = build_model_adapter(config)

    assert isinstance(adapter, OllamaModelAdapter)


def test_build_model_adapter_selects_openai_compatible() -> None:
    config = AppConfig(
        dataset=DatasetConfig(),
        agent=AgentConfig(
            provider="openai_compatible",
            model="deepseek-chat",
            api_base="https://api.deepseek.com",
            api_key="test-key",
        ),
        run=RunConfig(),
    )

    adapter = build_model_adapter(config)

    assert isinstance(adapter, OpenAIModelAdapter)


def test_three_provider_configs_load_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in (
        "QWEN_API_KEY",
        "QWEN_MODEL",
        "QWEN_API_BASE",
        "KIMI_API_KEY",
        "KIMI_MODEL",
        "KIMI_API_BASE",
    ):
        monkeypatch.delenv(key, raising=False)

    project_root = Path(__file__).resolve().parents[1]
    expected = {
        "react_baseline.qwen.example.yaml": (
            "qwen-max",
            "https://dashscope.aliyuncs.com/compatible-mode/v1",
        ),
        "react_baseline.kimi.example.yaml": ("kimi-k2.5", "https://api.moonshot.cn/v1"),
    }
    for filename, (model, api_base) in expected.items():
        config = load_app_config(project_root / "configs" / filename)
        assert config.agent.provider == "openai_compatible"
        assert config.agent.model == model
        assert config.agent.api_base == api_base
        assert config.run.max_workers == 2


def test_build_model_adapter_rejects_unknown_provider() -> None:
    config = AppConfig(
        dataset=DatasetConfig(),
        agent=AgentConfig(provider="mystery", model="x", api_key="test-key"),
        run=RunConfig(),
    )

    with pytest.raises(ValueError, match="Unsupported agent.provider"):
        build_model_adapter(config)


def test_ollama_model_adapter_posts_chat_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    class FakeResponse:
        def __enter__(self) -> "FakeResponse":
            return self

        def __exit__(self, *args: object) -> bool:
            return False

        def read(self) -> bytes:
            return b'{"message": {"content": "risk_topic,reason\\nA,B"}}'

    def fake_urlopen(request: object, timeout: float) -> FakeResponse:
        captured["url"] = getattr(request, "full_url")
        captured["timeout"] = timeout
        captured["headers"] = dict(getattr(request, "headers"))
        captured["body"] = json.loads(getattr(request, "data").decode("utf-8"))
        return FakeResponse()

    monkeypatch.setattr("data_agent_baseline.agents.model.urlopen", fake_urlopen)

    adapter = OllamaModelAdapter(
        model="qwen2.5:14b",
        api_base="http://localhost:11434/",
        temperature=0.2,
        timeout=12.0,
        max_retries=0,
    )
    response = adapter.complete(
        [
            ModelMessage(role="system", content="You write CSV."),
            ModelMessage(role="user", content="Answer."),
        ]
    )

    assert response == "risk_topic,reason\nA,B"
    assert captured["url"] == "http://localhost:11434/api/chat"
    assert captured["timeout"] == 12.0
    assert captured["headers"] == {"Content-type": "application/json"}
    assert captured["body"] == {
        "model": "qwen2.5:14b",
        "messages": [
            {"role": "system", "content": "You write CSV."},
            {"role": "user", "content": "Answer."},
        ],
        "stream": False,
        "think": False,
        "format": "json",
        "options": {"temperature": 0.2, "num_predict": 2048},
    }
