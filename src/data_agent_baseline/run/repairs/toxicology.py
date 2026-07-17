from __future__ import annotations

from pathlib import Path

from data_agent_baseline.tools.duckdb_sql import execute_data_sql


def repair_toxicology_atom_filter_count(
    *,
    question: str,
    task_dir: Path | None,
    prediction_path: Path,
    find_csv_table_with_columns,
    find_sqlite_table_with_columns,
    prediction_header,
    write_scalar_text_prediction,
) -> bool:
    if task_dir is None:
        return False
    context_dir = task_dir / "context"
    atom_table = find_csv_table_with_columns(task_dir, {"atom_id", "molecule_id", "element"})
    bond_table = find_sqlite_table_with_columns(task_dir, {"bond_id", "molecule_id", "bond_type"})
    if not atom_table or not bond_table:
        return False

    q = question.lower()
    elements: list[str] = []
    if "phosphorus" in q:
        elements.append("p")
    if "bromine" in q:
        elements.append("br")
    if not elements:
        return False
    element_sql = ", ".join("'" + element.replace("'", "''") + "'" for element in elements)

    sql = f"""
    SELECT COUNT(DISTINCT a.atom_id) AS answer
    FROM {atom_table} a
    WHERE LOWER(a.element) IN ({element_sql})
      AND a.molecule_id IN (
        SELECT DISTINCT molecule_id
        FROM {bond_table}
        WHERE bond_type = '#'
      )
    """
    try:
        result = execute_data_sql(context_dir, sql, limit=5)
    except Exception:
        return False
    rows = result.get("rows") or []
    if not rows:
        return False
    write_scalar_text_prediction(
        prediction_path,
        prediction_header(prediction_path, "total_atoms"),
        rows[0][0],
    )
    return True
