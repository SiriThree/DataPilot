from __future__ import annotations

from pathlib import Path

from data_agent_baseline.run.semantic_verifier import verify_semantic_contract


def _codes(report) -> set[tuple[str, str]]:
    return {(check.code, check.severity) for check in report.checks}


def test_semantic_verifier_flags_comment_id_when_text_requested(tmp_path: Path) -> None:
    prediction = tmp_path / "prediction.csv"
    prediction.write_text("Id\n90813\n", encoding="utf-8")

    report = verify_semantic_contract(
        question="Among the posts with views ranging from 100 to 150, what is the comment with the highest score?",
        prediction_path=prediction,
        run_result={"steps": []},
    )

    assert ("target_field_semantic_risk", "warning") in _codes(report)


def test_semantic_verifier_flags_missing_url_when_website_requested(tmp_path: Path) -> None:
    prediction = tmp_path / "prediction.csv"
    prediction.write_text("constructor_reference_name\nmclaren\n", encoding="utf-8")

    report = verify_semantic_contract(
        question="What is the constructor reference name of the champion? Please give its website.",
        prediction_path=prediction,
        run_result={"steps": []},
    )

    assert ("target_field_semantic_risk", "warning") in _codes(report)


def test_semantic_verifier_flags_percent_sign_scalar(tmp_path: Path) -> None:
    prediction = tmp_path / "prediction.csv"
    prediction.write_text("percentage_faster\n0.32%\n", encoding="utf-8")

    report = verify_semantic_contract(
        question="How much faster in percentage is the champion than the last driver?",
        prediction_path=prediction,
        run_result={"steps": []},
    )

    assert ("scalar_format_semantic_risk", "warning") in _codes(report)


def test_semantic_verifier_flags_population_filter_for_all_average(tmp_path: Path) -> None:
    prediction = tmp_path / "prediction.csv"
    prediction.write_text("average_weight_kg\n78.51\n", encoding="utf-8")

    report = verify_semantic_contract(
        question="What is the average weight of all female superheroes?",
        prediction_path=prediction,
        run_result={
            "steps": [
                {
                    "action": "execute_python",
                    "action_input": {
                        "code": "df[(df.gender == 'Female') & (df.weight_kg > 0)].weight_kg.mean()"
                    },
                }
            ]
        },
    )

    assert ("population_filter_semantic_risk", "warning") in _codes(report)


def test_semantic_verifier_accepts_url_value_when_website_requested(tmp_path: Path) -> None:
    prediction = tmp_path / "prediction.csv"
    prediction.write_text(
        "constructor_reference_name,website\nmclaren,http://en.wikipedia.org/wiki/McLaren\n",
        encoding="utf-8",
    )

    report = verify_semantic_contract(
        question="What is the constructor reference name of the champion? Please give its website.",
        prediction_path=prediction,
        run_result={"steps": []},
    )

    assert ("target_field_semantic_risk", "warning") not in _codes(report)
