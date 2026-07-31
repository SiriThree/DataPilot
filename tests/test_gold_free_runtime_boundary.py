from __future__ import annotations

from pathlib import Path


RUNTIME_ROOTS = (
    Path("src/data_agent_baseline/agents"),
    Path("src/data_agent_baseline/benchmark"),
    Path("src/data_agent_baseline/run"),
    Path("src/data_agent_baseline/tools"),
)

OFFLINE_ONLY_MODULES = {
    Path("src/data_agent_baseline/run/failure_mining.py"),
}

FORBIDDEN_RUNTIME_PATTERNS = (
    "gold.csv",
    "data/public/output",
    "from evaluate import",
    "import evaluate",
)


def test_runtime_validation_and_repair_do_not_depend_on_gold_answers() -> None:
    """Runtime can inspect inputs, traces and predictions, but not gold answers."""

    offenders: list[str] = []
    for root in RUNTIME_ROOTS:
        for path in root.rglob("*.py"):
            if path in OFFLINE_ONLY_MODULES:
                continue
            text = path.read_text(encoding="utf-8").lower()
            for pattern in FORBIDDEN_RUNTIME_PATTERNS:
                if pattern in text:
                    offenders.append(f"{path}:{pattern}")

    assert offenders == []


def test_gold_answer_access_stays_in_offline_evaluation_layer() -> None:
    evaluator = Path("evaluate.py").read_text(encoding="utf-8").lower()
    assert "gold.csv" in evaluator
