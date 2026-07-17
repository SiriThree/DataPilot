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
    assert "MANDATORY LARGE-CSV STRATEGY" in hint
    assert "execute_data_sql" in hint
