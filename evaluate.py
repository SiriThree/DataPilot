#!/usr/bin/env python3
"""Local evaluator for KDD Cup 2026 DataAgent-Bench public demo dataset.

Matches prediction.csv against gold.csv using the competition's
column-signature scoring logic:

- Column names are ignored; column order is ignored.
- Numeric values are normalized to 2 decimal places.
- Null / missing / empty values are collapsed to "".
- Each gold column is matched to at most one prediction column by
  maximum Jaccard similarity of (normalized) value sets.
- Extra (redundant) prediction columns incur a penalty.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "public"


def _normalize_value(val: Any) -> str:
    """Normalize a cell value per competition rules."""
    if val is None or val == "":
        return ""
    # Try to parse as float and round to 2 decimal places
    try:
        num = float(str(val))
        # Use the same format as Python's round with 2 decimals
        return f"{num:.2f}"
    except (ValueError, TypeError):
        return str(val).strip()


def _read_csv_columns(path: Path) -> list[set[str]]:
    """Read a CSV and return a list of column value sets (one set per column)."""
    rows: list[list[str]] = []
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        for row in reader:
            rows.append([_normalize_value(cell) for cell in row])

    if not rows:
        return []

    num_cols = len(rows[0])
    columns: list[set[str]] = [set() for _ in range(num_cols)]
    for row in rows:
        for i, val in enumerate(row):
            if i < num_cols:
                columns[i].add(val)
    return columns


def _jaccard(set_a: set[str], set_b: set[str]) -> float:
    """Jaccard similarity between two sets."""
    if not set_a and not set_b:
        return 1.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union > 0 else 0.0


def _match_columns(
    gold_cols: list[set[str]],
    pred_cols: list[set[str]],
) -> tuple[list[tuple[int, int, float]], list[int], list[int]]:
    """Greedy match prediction columns to gold columns by Jaccard similarity.

    Returns:
        matches: list of (gold_idx, pred_idx, similarity)
        unmatched_gold: list of gold column indices without a match
        unmatched_pred: list of prediction column indices without a match
    """
    # Build all candidate pairs sorted by Jaccard
    candidates: list[tuple[int, int, float]] = []
    for gi, gset in enumerate(gold_cols):
        for pi, pset in enumerate(pred_cols):
            sim = _jaccard(gset, pset)
            if sim > 0:
                candidates.append((gi, pi, sim))
    candidates.sort(key=lambda x: x[2], reverse=True)

    used_gold: set[int] = set()
    used_pred: set[int] = set()
    matches: list[tuple[int, int, float]] = []

    for gi, pi, sim in candidates:
        if gi not in used_gold and pi not in used_pred:
            matches.append((gi, pi, sim))
            used_gold.add(gi)
            used_pred.add(pi)

    unmatched_gold = [i for i in range(len(gold_cols)) if i not in used_gold]
    unmatched_pred = [i for i in range(len(pred_cols)) if i not in used_pred]

    return matches, unmatched_gold, unmatched_pred


def evaluate(prediction_path: Path, gold_path: Path) -> dict[str, Any]:
    """Evaluate a single prediction against gold.

    Returns a dict with:
        - score: float (0.0 to 1.0)
        - gold_columns: int
        - pred_columns: int
        - matched_columns: int
        - redundant_columns: int
        - missing_columns: int
        - match_details: list of per-match info
    """
    if not prediction_path.exists():
        return {
            "score": 0.0,
            "error": f"prediction file not found: {prediction_path}",
        }

    gold_cols = _read_csv_columns(gold_path)
    pred_cols = _read_csv_columns(prediction_path)

    if not gold_cols:
        return {
            "score": 1.0 if not pred_cols else 0.0,
            "error": "gold has no columns",
            "gold_columns": 0,
            "pred_columns": len(pred_cols),
        }

    matches, unmatched_gold, unmatched_pred = _match_columns(gold_cols, pred_cols)

    # Score: matched columns minus penalty for redundant columns
    # Each matched column contributes 1/len(gold). Each redundant column subtracts a penalty.
    gold_count = len(gold_cols)
    matched_count = len(matches)
    redundant_count = len(unmatched_pred)

    # Redundant columns penalty: each extra column reduces score
    # Based on competition FAQ: redundant columns get penalised
    redundancy_penalty = 0.05 * redundant_count
    base_score = matched_count / gold_count
    score = max(0.0, min(1.0, base_score - redundancy_penalty))

    # ── Flat value overlap (column-agnostic) ──
    gold_flat: set[str] = set()
    for gcol in gold_cols:
        gold_flat |= gcol
    pred_flat: set[str] = set()
    for pcol in pred_cols:
        pred_flat |= pcol
    value_overlap = len(gold_flat & pred_flat) / len(gold_flat) if gold_flat else 0.0

    # ── Row count match ──
    gold_rows = sum(1 for _ in open(gold_path, encoding="utf-8-sig")) - 1
    pred_rows = sum(1 for _ in open(prediction_path, encoding="utf-8-sig")) - 1
    row_match = 1.0 - min(abs(gold_rows - pred_rows) / max(gold_rows, 1), 1.0)

    return {
        "score": score,
        "gold_columns": gold_count,
        "pred_columns": len(pred_cols),
        "matched_columns": matched_count,
        "redundant_columns": redundant_count,
        "missing_columns": len(unmatched_gold),
        "value_overlap": round(value_overlap, 4),
        "gold_rows": gold_rows,
        "pred_rows": pred_rows,
        "row_match": round(row_match, 4),
        "composite": round(0.4 * score + 0.3 * value_overlap + 0.3 * row_match, 4),
        "match_details": [
            {
                "gold_col": gi,
                "pred_col": pi,
                "jaccard": round(sim, 4),
            }
            for gi, pi, sim in matches
        ],
    }


def evaluate_all(
    input_dir: Path,
    output_dir: Path,
    predictions_dir: Path,
    *,
    only_existing: bool = False,
) -> dict[str, Any]:
    """Evaluate all tasks in the dataset."""
    task_ids = sorted(
        [d.name for d in input_dir.iterdir() if d.is_dir() and d.name.startswith("task_")]
    )

    results: dict[str, Any] = {}
    total_score = 0.0
    by_difficulty: dict[str, list[float]] = {}
    failures: list[dict[str, Any]] = []
    skipped_missing_predictions: list[str] = []

    for task_id in task_ids:
        task_json_path = input_dir / task_id / "task.json"
        gold_path = output_dir / task_id / "gold.csv"
        pred_path = predictions_dir / task_id / "prediction.csv"

        if not gold_path.exists():
            continue
        if only_existing and not pred_path.exists():
            skipped_missing_predictions.append(task_id)
            continue

        # Read difficulty
        difficulty = "unknown"
        if task_json_path.exists():
            task_data = json.loads(task_json_path.read_text(encoding="utf-8"))
            difficulty = task_data.get("difficulty", "unknown")

        result = evaluate(pred_path, gold_path)
        result["task_id"] = task_id
        result["difficulty"] = difficulty

        total_score += result["score"]
        by_difficulty.setdefault(difficulty, []).append(result["score"])

        if result["score"] < 0.5:
            failures.append(result)

        results[task_id] = result

    overall = {
        "total_tasks": len(results),
        "skipped_missing_predictions": len(skipped_missing_predictions),
        "overall_score": round(total_score / len(results), 4) if results else 0.0,
        "by_difficulty": {
            diff: {
                "count": len(scores),
                "avg_score": round(sum(scores) / len(scores), 4) if scores else 0.0,
            }
            for diff, scores in sorted(by_difficulty.items())
        },
        "failures": failures,
    }

    return {
        "tasks": results,
        "summary": overall,
        "skipped_missing_prediction_task_ids": skipped_missing_predictions,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate predictions against gold.csv")
    sub = parser.add_subparsers(dest="command")

    single = sub.add_parser("single", help="Evaluate a single prediction")
    single.add_argument("--pred", required=True, help="Path to prediction.csv")
    single.add_argument("--gold", required=True, help="Path to gold.csv")

    batch = sub.add_parser("batch", help="Evaluate all tasks")
    batch.add_argument("--input-dir", default=str(DEFAULT_DATA_DIR / "input"))
    batch.add_argument("--output-dir", default=str(DEFAULT_DATA_DIR / "output"))
    batch.add_argument("--predictions-dir", required=True, help="Path to run output directory (e.g. artifacts/runs/<run_id>)")
    batch.add_argument(
        "--only-existing",
        action="store_true",
        help="Only evaluate tasks that already have prediction.csv in the run output directory.",
    )

    args = parser.parse_args()

    if args.command == "single":
        result = evaluate(Path(args.pred), Path(args.gold))
        print(json.dumps(result, indent=2))
    elif args.command == "batch":
        result = evaluate_all(
            Path(args.input_dir),
            Path(args.output_dir),
            Path(args.predictions_dir),
            only_existing=args.only_existing,
        )
        print(json.dumps(result["summary"], indent=2))
        # Also write to file
        out_path = Path(args.predictions_dir) / "evaluation.json"
        out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"\nFull results written to: {out_path}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
