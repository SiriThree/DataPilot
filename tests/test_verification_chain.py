from __future__ import annotations

from pathlib import Path

from data_agent_baseline.run.verification_chain import run_full_verification


def _check(report, name: str):
    return next(check for check in report.checks if check.name == name)


def test_verification_chain_accepts_valid_prediction(tmp_path: Path) -> None:
    prediction = tmp_path / "prediction.csv"
    prediction.write_text("answer\n42\n", encoding="utf-8")

    report = run_full_verification("task_demo", prediction)

    assert report.all_passed is True
    assert _check(report, "readability_check").passed is True
    assert _check(report, "contract_check").passed is True


def test_verification_chain_rejects_duplicate_headers(tmp_path: Path) -> None:
    prediction = tmp_path / "prediction.csv"
    prediction.write_text("answer,answer\n42,43\n", encoding="utf-8")

    report = run_full_verification("task_demo", prediction)

    assert report.all_passed is False
    assert _check(report, "contract_check").passed is False
    assert "duplicate headers" in _check(report, "contract_check").detail


def test_verification_chain_rejects_empty_fallback(tmp_path: Path) -> None:
    prediction = tmp_path / "prediction.csv"
    prediction.write_text("answer\n\n", encoding="utf-8")

    report = run_full_verification("task_demo", prediction)

    assert report.all_passed is False
    assert _check(report, "sanity_check").passed is False
    assert _check(report, "shape_check").passed is False
