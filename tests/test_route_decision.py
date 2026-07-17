from __future__ import annotations

from pathlib import Path

from data_agent_baseline.run.route_decision import build_route_prompt_hint, decide_route


def test_large_csv_prefers_sql_first(tmp_path: Path) -> None:
    context_dir = tmp_path / "context"
    (context_dir / "csv").mkdir(parents=True)
    large_csv = context_dir / "csv" / "yearmonth.csv"
    with large_csv.open("w", encoding="utf-8", newline="") as f:
        f.write("CustomerID,Date,Consumption\n")
        row = "5,201207,528.30\n"
        for _ in range((5 * 1024 * 1024 // len(row)) + 1):
            f.write(row)

    decision = decide_route(
        question="What was the average monthly consumption of customers in SME for the year 2013?",
        context_dir=context_dir,
        difficulty="medium",
    )
    hint = build_route_prompt_hint(decision)

    assert decision.route in {"sql_first", "hybrid_sql_python"}
    assert "large_csv_sql_first" in decision.risk_flags
    assert decision.scores["sql_first"] > decision.scores["python_first"]
    assert decision.task_profile.task_type == "aggregation"
    assert decision.task_profile.operation == "aggregate_average"
    assert decision.task_profile.output_shape == "scalar"
    assert "MANDATORY LARGE-CSV STRATEGY" in hint
    assert "Task profile:" in hint
    assert "execute_data_sql" in hint


def test_patient_threshold_count_profile(tmp_path: Path) -> None:
    context_dir = tmp_path / "context"
    context_dir.mkdir(parents=True)
    (context_dir / "labs.md").write_text(
        "Patient P1 hemoglobin 8.5. Patient P2 hemoglobin 14.0.",
        encoding="utf-8",
    )
    (context_dir / "patients.csv").write_text(
        "patient_id,sex\nP1,F\nP2,M\n",
        encoding="utf-8",
    )

    decision = decide_route(
        question="How many female patients have abnormal hemoglobin below the normal threshold?",
        context_dir=context_dir,
        difficulty="hard",
    )
    hint = build_route_prompt_hint(decision)

    assert decision.task_profile.task_type == "threshold_count"
    assert decision.task_profile.operation == "threshold_count"
    assert decision.task_profile.output_shape == "scalar"
    assert "medical_patient" in decision.task_profile.domains
    assert "doc_table_grounding" in decision.task_profile.flags
    assert "verify population filters" in hint


def test_formula1_rank_lookup_profile(tmp_path: Path) -> None:
    context_dir = tmp_path / "context"
    context_dir.mkdir(parents=True)
    (context_dir / "races.json").write_text(
        '{"records": [{"raceId": 10, "name": "Italian Grand Prix", "year": 2004}]}',
        encoding="utf-8",
    )
    (context_dir / "results.csv").write_text(
        "raceId,rank,time\n10,2,+5.6\n",
        encoding="utf-8",
    )

    decision = decide_route(
        question="What was the finish time for the second ranked driver in the 2004 Italian Grand Prix?",
        context_dir=context_dir,
        difficulty="medium",
    )

    assert decision.task_profile.task_type == "rank_lookup"
    assert decision.task_profile.operation == "rank_lookup"
    assert decision.task_profile.output_shape == "scalar"
    assert "formula1" in decision.task_profile.domains
    assert any("requested rank" in item for item in decision.task_profile.validation_focus)
