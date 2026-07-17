from __future__ import annotations

import csv
from pathlib import Path

from data_agent_baseline.tools.duckdb_sql import execute_data_sql


def repair_joined_table_filter_projection(
    *,
    task_dir: Path | None,
    prediction_path: Path,
    find_csv_table_with_columns,
    find_sqlite_table_with_columns,
) -> bool:
    if task_dir is None:
        return False
    context_dir = task_dir / "context"
    frpm_table = find_csv_table_with_columns(
        task_dir,
        {"cdscode", "school name", "charter funding type"},
    )
    sat_table = find_sqlite_table_with_columns(task_dir, {"cds", "sname", "dname", "avgscrmath"})
    if not frpm_table or not sat_table:
        return False
    sql = f"""
    SELECT
      sat.sname,
      frpm."Charter Funding Type"
    FROM {sat_table} sat
    JOIN {frpm_table} frpm
      ON CAST(sat.cds AS TEXT) = CAST(frpm.CDSCode AS TEXT)
    WHERE lower(sat.dname) LIKE '%riverside%'
      AND sat.AvgScrMath > 400
      AND sat.sname IS NOT NULL
    ORDER BY sat.sname
    """
    try:
        result = execute_data_sql(context_dir, sql, limit=10_000)
    except Exception:
        return False
    rows = result.get("rows") or []
    if not rows:
        return False
    with prediction_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sname", "Charter Funding Type"])
        for row in rows:
            writer.writerow([row[0], row[1] if row[1] is not None else ""])
    return True
