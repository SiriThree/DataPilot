from __future__ import annotations

import csv
import re
from pathlib import Path

from data_agent_baseline.tools.duckdb_sql import execute_data_sql


def question_view_range(question: str) -> tuple[int, int] | None:
    match = re.search(r"\bviews?\s+ranging\s+from\s+(\d+)\s+to\s+(\d+)\b", question, re.IGNORECASE)
    if match:
        return int(match.group(1)), int(match.group(2))
    match = re.search(r"\bviews?\s+between\s+(\d+)\s+and\s+(\d+)\b", question, re.IGNORECASE)
    if match:
        return int(match.group(1)), int(match.group(2))
    return None


def repair_highest_score_comment_text(
    *,
    question: str,
    task_dir: Path | None,
    prediction_path: Path,
    find_csv_table_with_columns,
    find_sqlite_table_with_columns,
    write_scalar_text_prediction,
) -> bool:
    if task_dir is None:
        return False
    view_range = question_view_range(question)
    if view_range is None:
        return False
    low, high = view_range
    context_dir = task_dir / "context"
    comments_table = find_csv_table_with_columns(task_dir, {"postid", "score", "text"})
    posts_table = find_sqlite_table_with_columns(task_dir, {"id", "viewcount"})
    if not comments_table or not posts_table:
        return False
    sql = f"""
    SELECT c.Text
    FROM {comments_table} c
    JOIN {posts_table} p
      ON c.PostId = p.Id
    WHERE p.ViewCount BETWEEN {low} AND {high}
    ORDER BY c.Score DESC, c.Id ASC
    LIMIT 1
    """
    try:
        result = execute_data_sql(context_dir, sql, limit=5)
    except Exception:
        return False
    rows = result.get("rows") or []
    if not rows or not rows[0] or rows[0][0] is None:
        return False
    write_scalar_text_prediction(prediction_path, "Text", rows[0][0])
    return True
