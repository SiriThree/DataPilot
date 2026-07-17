from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any, Protocol
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from openai import APIError, APITimeoutError, OpenAI


@dataclass(frozen=True, slots=True)
class ModelMessage:
    role: str
    content: str


@dataclass(frozen=True, slots=True)
class ModelStep:
    thought: str
    action: str
    action_input: dict[str, Any]
    raw_response: str


class ModelAdapter(Protocol):
    def complete(self, messages: list[ModelMessage]) -> str:
        raise NotImplementedError


class OpenAIModelAdapter:
    def __init__(
        self,
        *,
        model: str,
        api_base: str,
        api_key: str,
        temperature: float,
        timeout: float = 120.0,
        max_retries: int = 3,
    ) -> None:
        if not api_key:
            raise RuntimeError("Missing model API key in config.agent.api_key.")
        self.model = model
        self.api_base = api_base.rstrip("/")
        self.api_key = api_key
        self.temperature = temperature
        self.timeout = timeout
        self.max_retries = max_retries
        self._client = OpenAI(
            api_key=api_key,
            base_url=self.api_base,
            timeout=timeout,
            max_retries=0,  # We handle retries ourselves for better control
        )

    def complete(self, messages: list[ModelMessage]) -> str:
        client = self._client
        last_error: Exception | None = None

        for attempt in range(self.max_retries + 1):
            try:
                response = client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": m.role, "content": m.content} for m in messages],
                    temperature=self.temperature,
                )
                choices = response.choices or []
                if not choices:
                    raise RuntimeError("Model response missing choices.")
                content = choices[0].message.content
                if not isinstance(content, str):
                    raise RuntimeError("Model response missing text content.")
                return content
            except APITimeoutError as exc:
                last_error = exc
                if attempt < self.max_retries:
                    wait = 2 ** attempt
                    time.sleep(wait)
            except APIError as exc:
                # Don't retry on auth errors
                if hasattr(exc, "status_code") and exc.status_code in (401, 403):
                    raise RuntimeError(f"Model auth error: {exc}") from exc
                last_error = exc
                if attempt < self.max_retries:
                    wait = 2 ** attempt
                    time.sleep(wait)

        raise RuntimeError(
            f"Model request failed after {self.max_retries + 1} attempts: {last_error}"
        )


class OllamaModelAdapter:
    def __init__(
        self,
        *,
        model: str,
        api_base: str = "http://localhost:11434",
        temperature: float = 0.0,
        timeout: float = 120.0,
        max_retries: int = 1,
    ) -> None:
        if not model:
            raise RuntimeError("Missing Ollama model name in config.agent.model.")
        self.model = model
        self.api_base = api_base.rstrip("/")
        self.temperature = temperature
        self.timeout = timeout
        self.max_retries = max_retries

    def complete(self, messages: list[ModelMessage]) -> str:
        payload = {
            "model": self.model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "stream": False,
            "think": False,
            "format": "json",
            "options": {"temperature": self.temperature, "num_predict": 2048},
        }
        data = json.dumps(payload).encode("utf-8")
        last_error: Exception | None = None

        for attempt in range(self.max_retries + 1):
            request = Request(
                f"{self.api_base}/api/chat",
                data=data,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            try:
                with urlopen(request, timeout=self.timeout) as response:
                    body = response.read().decode("utf-8")
                parsed = json.loads(body)
                message = parsed.get("message")
                if not isinstance(message, dict):
                    raise RuntimeError("Ollama response missing message.")
                content = message.get("content")
                if isinstance(content, str) and content:
                    return content
                thinking = message.get("thinking")
                if isinstance(thinking, str) and thinking:
                    return thinking
                if isinstance(content, str):
                    return content
                raise RuntimeError("Ollama response missing message.content.")
            except HTTPError as exc:
                last_error = exc
                if exc.code in (400, 401, 403, 404):
                    raise RuntimeError(f"Ollama request error: HTTP {exc.code}") from exc
                if attempt < self.max_retries:
                    time.sleep(2 ** attempt)
            except (URLError, TimeoutError, json.JSONDecodeError, RuntimeError) as exc:
                last_error = exc
                if attempt < self.max_retries:
                    time.sleep(2 ** attempt)

        raise RuntimeError(
            f"Ollama request failed after {self.max_retries + 1} attempts: {last_error}"
        )


class ScriptedModelAdapter:
    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)

    def complete(self, messages: list[ModelMessage]) -> str:
        del messages
        if not self._responses:
            raise RuntimeError("No scripted model responses remaining.")
        return self._responses.pop(0)
