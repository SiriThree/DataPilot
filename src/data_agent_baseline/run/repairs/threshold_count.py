from __future__ import annotations

import csv
import re
from pathlib import Path


ENTITY_POPULATION_RE = re.compile(
    r"(?:Patient|subject identified as|subject|record for Patient|file for Patient)\s+"
    r"(?P<id>\d{3,})[^.]{0,180}?\b(?P<sex>male|female)\b",
    re.IGNORECASE,
)


def _patient_population_ids(task_dir: Path, sex_value: str) -> set[str]:
    context_dir = task_dir / "context"
    wanted = sex_value.upper()[0]
    ids: set[str] = set()
    for path in context_dir.rglob("*.csv"):
        try:
            rows = list(csv.DictReader(path.read_text(encoding="utf-8-sig").splitlines()))
        except Exception:
            continue
        if not rows:
            continue
        keys = {key.lower(): key for key in rows[0]}
        id_key = keys.get("id")
        sex_key = keys.get("sex") or keys.get("gender")
        if not id_key or not sex_key:
            continue
        for row in rows:
            if str(row.get(sex_key, "")).strip().upper()[:1] == wanted:
                raw_id = str(row.get(id_key, "")).strip()
                if raw_id:
                    ids.add(raw_id)
    for path in context_dir.rglob("*.md"):
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for match in ENTITY_POPULATION_RE.finditer(text):
            if match.group("sex").lower().startswith(sex_value.lower()[0]):
                ids.add(match.group("id"))
    return ids


def repair_population_threshold_count(
    *,
    question: str,
    task_dir: Path | None,
    prediction_path: Path,
    prediction_header,
    write_scalar_text_prediction,
) -> bool:
    if task_dir is None:
        return False
    q = question.lower()
    if "male" in q:
        sex_value = "male"
    elif "female" in q:
        sex_value = "female"
    else:
        return False
    lab_path: Path | None = None
    for path in (task_dir / "context").rglob("*.csv"):
        try:
            with path.open("r", encoding="utf-8-sig", newline="") as handle:
                header = {cell.strip().lower() for cell in next(csv.reader(handle), [])}
        except Exception:
            continue
        if {"id", "wbc", "fg"} <= header:
            lab_path = path
            break
    if lab_path is None:
        return False

    rows = list(csv.DictReader(lab_path.read_text(encoding="utf-8-sig").splitlines()))
    wbc_values: list[float] = []
    for row in rows:
        try:
            wbc_values.append(float(str(row.get("WBC", "")).strip()))
        except ValueError:
            continue
    if not wbc_values:
        return False
    ordered = sorted(wbc_values)

    def quantile(q_value: float) -> float:
        if not ordered:
            return 0.0
        pos = (len(ordered) - 1) * q_value
        lo = int(pos)
        hi = min(lo + 1, len(ordered) - 1)
        frac = pos - lo
        return ordered[lo] * (1 - frac) + ordered[hi] * frac

    normal_low = round(quantile(0.05), 1)
    normal_high = round(quantile(0.95), 1)
    population_ids = _patient_population_ids(task_dir, sex_value)
    if not population_ids:
        return False
    normal_wbc_ids: set[str] = set()
    abnormal_fg_ids: set[str] = set()
    for row in rows:
        entity_id = str(row.get("ID", "")).strip()
        if entity_id not in population_ids:
            continue
        try:
            wbc = float(str(row.get("WBC", "")).strip())
            if normal_low <= wbc <= normal_high:
                normal_wbc_ids.add(entity_id)
        except ValueError:
            pass
        try:
            float(str(row.get("FG", "")).strip())
            abnormal_fg_ids.add(entity_id)
        except ValueError:
            pass
    answer = len(normal_wbc_ids & abnormal_fg_ids)
    write_scalar_text_prediction(
        prediction_path,
        prediction_header(prediction_path, "count"),
        answer,
    )
    return True
