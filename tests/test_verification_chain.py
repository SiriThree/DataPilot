from __future__ import annotations

from pathlib import Path

from data_agent_baseline.run.verification_chain import run_full_verification
from data_agent_baseline.run.verification_chain import infer_output_contract


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


def test_infer_output_contract_column_count_from_question(tmp_path: Path) -> None:
    task_dir = tmp_path / "task_demo"
    task_dir.mkdir()
    (task_dir / "task.json").write_text(
        '{"question": "For patients with severe thrombosis, list their ID, sex and disease."}',
        encoding="utf-8",
    )

    contract = infer_output_contract(task_dir)

    assert contract is not None
    assert contract.expected_column_count == 3
    assert contract.expected_columns == []
    assert contract.source == "task_json.question_shape"


def test_verification_chain_rejects_question_shape_mismatch(tmp_path: Path) -> None:
    task_dir = tmp_path / "task_demo"
    task_dir.mkdir()
    (task_dir / "task.json").write_text(
        '{"question": "Which event has the lowest cost?"}',
        encoding="utf-8",
    )
    prediction = tmp_path / "prediction.csv"
    prediction.write_text("event_name,total_cost\nApril Meeting,12.3\n", encoding="utf-8")

    report = run_full_verification("task_demo", prediction, infer_output_contract(task_dir))

    assert report.all_passed is False
    assert _check(report, "task_contract_check").passed is False
    assert "expected 1 column" in _check(report, "task_contract_check").detail


def test_infer_output_contract_max_rows_from_top_n(tmp_path: Path) -> None:
    task_dir = tmp_path / "task_demo"
    task_dir.mkdir()
    (task_dir / "task.json").write_text(
        '{"question": "List the top 5 customers by total amount."}',
        encoding="utf-8",
    )

    contract = infer_output_contract(task_dir)

    assert contract is not None
    assert contract.max_rows == 5


def test_infer_output_contract_list_all_single_entity(tmp_path: Path) -> None:
    task_dir = tmp_path / "task_demo"
    task_dir.mkdir()
    (task_dir / "task.json").write_text(
        '{"question": "List all the withdrawals in cash transactions that the client makes."}',
        encoding="utf-8",
    )

    contract = infer_output_contract(task_dir)

    assert contract is not None
    assert contract.expected_column_count == 1
