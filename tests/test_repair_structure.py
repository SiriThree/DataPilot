from __future__ import annotations

from pathlib import Path


def test_repair_modules_are_pattern_named() -> None:
    repairs_dir = Path("src/data_agent_baseline/run/repairs")
    module_names = {path.stem for path in repairs_dir.glob("*.py")}

    domain_module_names = {
        "formula1",
        "medical",
        "school",
        "stackexchange",
        "superhero",
        "toxicology",
    }

    assert module_names.isdisjoint(domain_module_names)
