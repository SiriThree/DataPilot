#!/usr/bin/env python3
"""Run a local command with DeepSeek env loaded from Claude settings.

This helper is intentionally local-only. The official submission path is
`main.py`, which reads only MODEL_NAME/MODEL_API_URL/MODEL_API_KEY from the
evaluation runtime.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SETTINGS_PATHS = (
    PROJECT_ROOT / ".claude" / "settings.json",
    Path.home() / ".claude" / "settings.json",
)
MODEL_CONTEXT_SUFFIX_RE = re.compile(r"\[[^\]]+\]$")


def _strip_context_suffix(model: str) -> str:
    return MODEL_CONTEXT_SUFFIX_RE.sub("", model.strip())


def _derive_deepseek_openai_base(raw_base_url: str) -> str:
    if "deepseek.com" in raw_base_url:
        return "https://api.deepseek.com"
    return raw_base_url.rstrip("/") or "https://api.deepseek.com"


def _load_claude_env(settings_paths: list[Path]) -> tuple[Path, dict[str, str]]:
    for settings_path in settings_paths:
        if not settings_path.exists():
            continue
        payload = json.loads(settings_path.read_text(encoding="utf-8"))
        env = payload.get("env", {})
        if not isinstance(env, dict):
            continue
        token = env.get("DEEPSEEK_API_KEY") or env.get("ANTHROPIC_AUTH_TOKEN")
        if token:
            return settings_path, {str(key): str(value) for key, value in env.items()}
    searched = ", ".join(str(path) for path in settings_paths)
    raise RuntimeError(f"No DeepSeek token found in Claude settings. Searched: {searched}")


def build_deepseek_env(settings_paths: list[Path]) -> tuple[dict[str, str], Path]:
    source_path, claude_env = _load_claude_env(settings_paths)
    env = os.environ.copy()

    env.setdefault(
        "DEEPSEEK_API_KEY",
        claude_env.get("DEEPSEEK_API_KEY") or claude_env.get("ANTHROPIC_AUTH_TOKEN", ""),
    )
    env.setdefault(
        "DEEPSEEK_API_BASE",
        _derive_deepseek_openai_base(
            claude_env.get("DEEPSEEK_API_BASE") or claude_env.get("ANTHROPIC_BASE_URL", "")
        ),
    )

    configured_model = (
        claude_env.get("DEEPSEEK_MODEL")
        or claude_env.get("ANTHROPIC_MODEL")
        or claude_env.get("ANTHROPIC_DEFAULT_HAIKU_MODEL")
        or "deepseek-v4-flash"
    )
    env.setdefault("DEEPSEEK_MODEL", _strip_context_suffix(configured_model))
    return env, source_path


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load DeepSeek env from Claude settings and run a local command."
    )
    parser.add_argument(
        "--settings",
        action="append",
        type=Path,
        help="Claude settings.json path. May be passed multiple times.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Print a safe summary of the resolved DeepSeek settings without running a command.",
    )
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if args.command and args.command[0] == "--":
        args.command = args.command[1:]
    if not args.show and not args.command:
        parser.error("provide --show or a command after --")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    settings_paths = args.settings or list(DEFAULT_SETTINGS_PATHS)
    env, source_path = build_deepseek_env(settings_paths)

    if args.show:
        print(f"settings={source_path}")
        print(f"DEEPSEEK_API_BASE={env.get('DEEPSEEK_API_BASE', '')}")
        print(f"DEEPSEEK_MODEL={env.get('DEEPSEEK_MODEL', '')}")
        print("DEEPSEEK_API_KEY=loaded")
        return 0

    os.execvpe(args.command[0], args.command, env)
    return 127


if __name__ == "__main__":
    raise SystemExit(main())
