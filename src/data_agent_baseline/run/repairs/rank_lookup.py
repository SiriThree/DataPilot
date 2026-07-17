from __future__ import annotations

import csv
import json
import re
import sqlite3
from pathlib import Path


def _question_year(question: str) -> int | None:
    match = re.search(r"\b((?:19|20)\d{2})\b", question)
    return int(match.group(1)) if match else None


def _question_rank_value(question: str) -> int | None:
    q = question.lower()
    ordinals = {
        "first": 1,
        "1st": 1,
        "second": 2,
        "2nd": 2,
        "third": 3,
        "3rd": 3,
        "fourth": 4,
        "4th": 4,
        "fifth": 5,
        "5th": 5,
    }
    for token, value in ordinals.items():
        if re.search(rf"\b{re.escape(token)}\b", q):
            return value
    match = re.search(r"\brank(?:ed)?\s+(\d+)\b", q)
    return int(match.group(1)) if match else None


def _records_from_json(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and isinstance(payload.get("records"), list):
        return [record for record in payload["records"] if isinstance(record, dict)]
    if isinstance(payload, list):
        return [record for record in payload if isinstance(record, dict)]
    return []


def _find_race_record(task_dir: Path, question: str) -> dict | None:
    year = _question_year(question)
    if year is None:
        return None
    q = question.lower()
    context_dir = task_dir / "context"
    for path in context_dir.rglob("*.json"):
        try:
            records = _records_from_json(path)
        except Exception:
            continue
        for record in records:
            keys = {str(key).lower(): key for key in record}
            name_key = keys.get("name") or keys.get("race_name")
            year_key = keys.get("year")
            race_id_key = keys.get("raceid") or keys.get("race_id") or keys.get("id")
            if not name_key or not year_key or not race_id_key:
                continue
            try:
                record_year = int(record.get(year_key))
            except (TypeError, ValueError):
                continue
            name = str(record.get(name_key, "")).strip()
            if record_year == year and name and name.lower() in q:
                return {
                    "race_id": str(record.get(race_id_key)).strip(),
                    "name": name,
                    "year": record_year,
                }
    return None


def repair_rank_attached_field(
    *,
    question: str,
    task_dir: Path | None,
    prediction_path: Path,
) -> bool:
    if task_dir is None:
        return False
    race = _find_race_record(task_dir, question)
    rank_value = _question_rank_value(question)
    if not race or rank_value is None:
        return False
    race_id = str(race["race_id"])

    context_dir = task_dir / "context"
    for path in context_dir.rglob("*.csv"):
        try:
            rows = list(csv.DictReader(path.read_text(encoding="utf-8-sig").splitlines()))
        except Exception:
            continue
        if not rows:
            continue
        columns = {column.lower(): column for column in rows[0]}
        race_col = columns.get("raceid") or columns.get("race_id")
        rank_col = columns.get("rank")
        time_col = columns.get("time") or columns.get("finish_time")
        if not race_col or not rank_col or not time_col:
            continue
        for row in rows:
            try:
                row_rank = int(float(str(row.get(rank_col, "")).strip()))
            except ValueError:
                continue
            if str(row.get(race_col, "")).strip() == race_id and row_rank == rank_value:
                finish_time = str(row.get(time_col, "")).strip()
                if not finish_time:
                    return False
                with prediction_path.open("w", encoding="utf-8", newline="") as handle:
                    writer = csv.writer(handle)
                    writer.writerow(["time"])
                    writer.writerow([finish_time])
                return True
    return False


def _question_grand_prix_name(question: str) -> str | None:
    match = re.search(r"\b([A-Za-z][A-Za-z ]+?\s+Grand Prix)\b", question, re.IGNORECASE)
    if not match:
        return None
    return re.sub(r"\s+", " ", match.group(1)).strip()


def _find_race_id_in_docs(task_dir: Path, race_name: str, year: int | None) -> str | None:
    context_dir = task_dir / "context"
    target = race_name.lower()
    for path in list(context_dir.rglob("*.md")) + list(context_dir.rglob("*.txt")):
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        paragraphs = re.split(r"\n\s*\n", text)
        for paragraph in paragraphs:
            compact = " ".join(paragraph.split())
            lower = compact.lower()
            if target not in lower:
                continue
            if year is not None and str(year) not in compact:
                continue
            match = re.search(
                r"\b(?:race\s*id|raceid|race_id)\s*[:#=]?\s*(\d+)\b",
                compact,
                re.IGNORECASE,
            )
            if match:
                return match.group(1)
    return None


def _winning_constructor_id(task_dir: Path, race_id: str) -> str | None:
    context_dir = task_dir / "context"
    for path in context_dir.rglob("*"):
        if path.suffix.lower() not in {".db", ".sqlite", ".sqlite3", ".db3"} or not path.is_file():
            continue
        try:
            with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as conn:
                table_names = [
                    row[0]
                    for row in conn.execute(
                        "SELECT name FROM sqlite_master "
                        "WHERE type='table' AND name NOT LIKE 'sqlite_%'"
                    )
                ]
                for table_name in table_names:
                    quoted = '"' + table_name.replace('"', '""') + '"'
                    columns = [row[1] for row in conn.execute(f"PRAGMA table_info({quoted})")]
                    lower = {column.lower(): column for column in columns}
                    race_col = lower.get("raceid") or lower.get("race_id")
                    constructor_col = lower.get("constructorid") or lower.get("constructor_id")
                    position_col = lower.get("positionorder") or lower.get("position_order")
                    if not race_col or not constructor_col or not position_col:
                        continue
                    query = (
                        f'SELECT "{constructor_col}" FROM {quoted} '
                        f'WHERE CAST("{race_col}" AS TEXT) = ? '
                        f'AND CAST("{position_col}" AS INTEGER) = 1 '
                        "LIMIT 1"
                    )
                    row = conn.execute(query, (race_id,)).fetchone()
                    if row and row[0] is not None:
                        return str(row[0]).strip()
        except Exception:
            continue
    return None


def _constructor_reference_record(task_dir: Path, constructor_id: str) -> tuple[str, str] | None:
    target = str(constructor_id).strip()
    for path in (task_dir / "context").rglob("*.json"):
        try:
            records = _records_from_json(path)
        except Exception:
            continue
        for record in records:
            keys = {str(key).lower(): key for key in record}
            id_key = keys.get("constructorid") or keys.get("constructor_id") or keys.get("id")
            ref_key = keys.get("constructorref") or keys.get("constructor_ref") or keys.get("ref")
            url_key = keys.get("url") or keys.get("website") or keys.get("web_site")
            if not id_key or not ref_key or not url_key:
                continue
            if str(record.get(id_key, "")).strip() != target:
                continue
            constructor_ref = str(record.get(ref_key, "")).strip()
            url = str(record.get(url_key, "")).strip()
            if constructor_ref and url:
                return constructor_ref, url
    return None


def repair_reference_fields_from_ranked_entity(
    *,
    question: str,
    task_dir: Path | None,
    prediction_path: Path,
) -> bool:
    if task_dir is None:
        return False
    race_name = _question_grand_prix_name(question)
    if not race_name:
        return False
    race_id = _find_race_id_in_docs(task_dir, race_name, _question_year(question))
    if race_id is None:
        race = _find_race_record(task_dir, question)
        race_id = str(race["race_id"]) if race else None
    if race_id is None:
        return False
    constructor_id = _winning_constructor_id(task_dir, race_id)
    if constructor_id is None:
        return False
    record = _constructor_reference_record(task_dir, constructor_id)
    if record is None:
        return False
    constructor_ref, url = record
    with prediction_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["constructorRef", "url"])
        writer.writerow([constructor_ref, url])
    return True
