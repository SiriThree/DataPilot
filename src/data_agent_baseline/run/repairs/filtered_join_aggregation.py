from __future__ import annotations

from pathlib import Path

from data_agent_baseline.tools.duckdb_sql import execute_data_sql


def repair_filtered_join_average(
    *,
    task_dir: Path | None,
    prediction_path: Path,
    find_csv_table_with_columns,
    records_from_json,
    safe_duckdb_table_name,
    prediction_header,
    write_scalar_text_prediction,
) -> bool:
    if task_dir is None:
        return False
    context_dir = task_dir / "context"
    superhero_table = find_csv_table_with_columns(task_dir, {"gender_id", "weight_kg"})
    gender_table = None
    for path in context_dir.rglob("*.json"):
        try:
            records = records_from_json(path)
        except Exception:
            continue
        if records and {"id", "gender"} <= {str(key).lower() for key in records[0]}:
            gender_table = safe_duckdb_table_name(path.relative_to(context_dir).with_suffix("")) + "_records"
            break
    if not superhero_table or not gender_table:
        return False
    sql = f"""
    SELECT AVG(s.weight_kg) AS average_weight
    FROM {superhero_table} s
    JOIN {gender_table} g
      ON s.gender_id = g.id
    WHERE lower(g.gender) = 'female'
    """
    try:
        result = execute_data_sql(context_dir, sql, limit=5)
    except Exception:
        return False
    rows = result.get("rows") or []
    if not rows or rows[0][0] is None:
        return False
    write_scalar_text_prediction(
        prediction_path,
        prediction_header(prediction_path, "average_weight_kg"),
        rows[0][0],
    )
    return True
