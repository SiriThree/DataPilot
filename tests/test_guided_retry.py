from __future__ import annotations

from data_agent_baseline.run.guided_retry import select_best_attempt


def _trace(
    *,
    answer_present: bool = True,
    data_row_count: int = 1,
    signals: list[dict[str, str]] | None = None,
    predicted_columns: list[str] | None = None,
) -> dict:
    return {
        "_failure_analysis": {
            "signals": signals or [],
            "commitments": {
                "answer_present": answer_present,
                "data_row_count": data_row_count,
                "predicted_columns": predicted_columns or ["answer"],
            },
        }
    }


def test_retry_selection_does_not_reward_row_count_inflation() -> None:
    signals = [
        {
            "code": "numeric_grounding_error",
            "severity": "warning",
            "detail": "numeric intent detected but prediction contains no numeric cells",
        },
        {
            "code": "output_shape_semantic_risk",
            "severity": "info",
            "detail": "multi-field wording requires separate answer columns",
        },
    ]

    selection = select_best_attempt(
        original_trace=_trace(
            data_row_count=57,
            signals=signals,
            predicted_columns=["School Name", "Funding Type"],
        ),
        retry_trace=_trace(
            data_row_count=88,
            signals=signals,
            predicted_columns=["School Name", "Funding Type"],
        ),
        trigger_codes=["numeric_grounding_error"],
    )

    assert selection.selected_attempt == "original"
    assert "increased row count" in selection.rationale


def test_retry_selection_prefers_retry_when_hard_error_is_fixed() -> None:
    selection = select_best_attempt(
        original_trace=_trace(
            answer_present=False,
            data_row_count=0,
            signals=[
                {
                    "code": "no_grounded_answer",
                    "severity": "error",
                    "detail": "prediction has no non-empty data rows",
                }
            ],
        ),
        retry_trace=_trace(answer_present=True, data_row_count=3, signals=[]),
        trigger_codes=["no_grounded_answer"],
    )

    assert selection.selected_attempt == "retry"


def test_retry_selection_prefers_retry_when_semantic_risk_is_reduced() -> None:
    selection = select_best_attempt(
        original_trace=_trace(
            data_row_count=1,
            signals=[
                {
                    "code": "output_shape_semantic_risk",
                    "severity": "warning",
                    "detail": "single delimited score/result cell may need separate output columns",
                }
            ],
        ),
        retry_trace=_trace(data_row_count=1, signals=[]),
        trigger_codes=["output_shape_semantic_risk"],
    )

    assert selection.selected_attempt == "retry"
