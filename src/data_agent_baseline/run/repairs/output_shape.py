from __future__ import annotations

import csv
import re
from pathlib import Path


def split_final_score_prediction(path: Path) -> bool:
    try:
        rows = list(csv.reader(path.read_text(encoding="utf-8-sig").splitlines()))
    except Exception:
        return False
    if len(rows) != 2 or not rows[1]:
        return False
    match = re.fullmatch(r"\s*(\d+)\s*[-:]\s*(\d+)\s*", rows[1][0])
    if not match:
        return False
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["home_team_goal", "away_team_goal"])
        writer.writerow([match.group(1), match.group(2)])
    return True
