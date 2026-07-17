from __future__ import annotations

import csv
import re
from pathlib import Path


def strip_percent_symbol_prediction(path: Path) -> bool:
    try:
        rows = list(csv.reader(path.read_text(encoding="utf-8-sig").splitlines()))
    except Exception:
        return False
    if len(rows) != 2 or len(rows[0]) != 1 or not rows[1]:
        return False
    value = rows[1][0].strip()
    match = re.fullmatch(r"([-+]?\d+(?:\.\d+)?)\s*%", value)
    if not match:
        return False
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([rows[0][0].strip() or "percentage"])
        writer.writerow([match.group(1)])
    return True
