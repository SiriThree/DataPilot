"""Ground threshold/range wording against context docs and observed data."""

from __future__ import annotations

import csv
import math
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Any


ENTITY_ID_RE = re.compile(
    r"(?:Patient|Case ID|subject identified as|subject|identified as|ID)\s+(?:registered under Case ID\s+)?(\d{3,})",
    re.IGNORECASE,
)
NUMBER_RE = re.compile(r"[-+]?\d+(?:\.\d+)?")
THRESHOLD_WORD_RE = re.compile(
    r"\b(normal|abnormal|range|threshold|between|above|below|greater|less|over|under|within|outside|level)\b",
    re.IGNORECASE,
)

KNOWN_TERM_ALIASES = {
    "white blood cell": ["wbc"],
    "white blood cells": ["wbc"],
    "fibrinogen": ["fg"],
    "platelet": ["plt"],
    "platelets": ["plt"],
    "uric acid": ["ua"],
    "lactate dehydrogenase": ["ldh"],
    "creatinine": ["cre"],
    "glucose": ["glu"],
}

POPULATION_ALIASES = {
    "male": {"male", "m"},
    "males": {"male", "m"},
    "man": {"male", "m"},
    "men": {"male", "m"},
    "female": {"female", "f"},
    "females": {"female", "f"},
    "woman": {"female", "f"},
    "women": {"female", "f"},
}


def _norm(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def _acronym(value: str) -> str:
    parts = re.findall(r"[A-Za-z]+", value)
    if len(parts) <= 1:
        return ""
    return "".join(part[0] for part in parts).lower()


def _term_variants(term: str) -> set[str]:
    lower = term.lower().strip()
    variants = {lower, _norm(lower), _acronym(lower)}
    for key, aliases in KNOWN_TERM_ALIASES.items():
        if key in lower or lower in aliases:
            variants.update(aliases)
            variants.add(_norm(key))
    return {variant for variant in variants if variant}


def _population_values(population_terms: list[str]) -> set[str]:
    values: set[str] = set()
    for term in population_terms:
        lower = term.lower().strip()
        values.add(lower)
        values.update(POPULATION_ALIASES.get(lower, set()))
    return {value for value in values if value}


def _sentence_contains_population_value(sentence: str, wanted: set[str]) -> bool:
    lower = sentence.lower()
    for value in wanted:
        if len(value) <= 1:
            continue
        if re.search(rf"\b{re.escape(value)}\b", lower):
            return True
    return False


def _as_float(value: str) -> float | None:
    if value is None:
        return None
    text = str(value).strip().replace(",", "")
    if not text:
        return None
    try:
        value_float = float(text)
    except ValueError:
        return None
    if math.isnan(value_float) or math.isinf(value_float):
        return None
    return value_float


def _quantile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return ordered[lo]
    frac = pos - lo
    return ordered[lo] * (1 - frac) + ordered[hi] * frac


def _round(value: float) -> float:
    return round(value, 6)


def _read_csv_dicts(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            return list(reader.fieldnames or []), [dict(row) for row in reader]
    except (OSError, UnicodeDecodeError, csv.Error):
        return [], []


def _match_columns(context_root: Path, target_terms: list[str]) -> list[dict[str, Any]]:
    target_variants = {term: _term_variants(term) for term in target_terms}
    matches: list[dict[str, Any]] = []
    for path in sorted(context_root.rglob("*.csv")):
        header, rows = _read_csv_dicts(path)
        if not header:
            continue
        for column in header:
            col_variants = {_norm(column), column.lower(), _acronym(column)}
            matched_terms = [
                term for term, variants in target_variants.items()
                if variants & col_variants or _norm(term) == _norm(column)
            ]
            if not matched_terms:
                continue
            numeric_values = [
                parsed for row in rows
                if (parsed := _as_float(row.get(column, ""))) is not None
            ]
            if not numeric_values:
                continue
            matches.append({
                "path": path.relative_to(context_root).as_posix(),
                "column": column,
                "matched_terms": matched_terms,
                "row_count": len(rows),
                "numeric_count": len(numeric_values),
                "missing_count": len(rows) - len(numeric_values),
            })
    return matches


def _threshold_snippets(context_root: Path, target_terms: list[str]) -> list[dict[str, Any]]:
    variants = set()
    for term in target_terms:
        variants.update(_term_variants(term))
    snippets: list[dict[str, Any]] = []
    for path in sorted(context_root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in {".md", ".txt"}:
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        paragraphs = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
        for index, paragraph in enumerate(paragraphs):
            normalized = _norm(paragraph)
            if not any(variant and variant in normalized for variant in variants):
                continue
            if not THRESHOLD_WORD_RE.search(paragraph) or not NUMBER_RE.search(paragraph):
                continue
            snippets.append({
                "path": path.relative_to(context_root).as_posix(),
                "paragraph_index": index,
                "numbers": NUMBER_RE.findall(paragraph)[:8],
                "excerpt": paragraph[:700],
            })
            if len(snippets) >= 10:
                return snippets
    return snippets


def _population_sources(context_root: Path, population_terms: list[str]) -> list[dict[str, Any]]:
    wanted = _population_values(population_terms)
    if not wanted:
        return []
    sources: list[dict[str, Any]] = []

    for path in sorted(context_root.rglob("*.csv")):
        header, rows = _read_csv_dicts(path)
        lower_to_header = {name.lower(): name for name in header}
        id_col = lower_to_header.get("id")
        pop_col = next((lower_to_header[name] for name in ("sex", "gender") if name in lower_to_header), None)
        if not id_col or not pop_col:
            continue
        ids = {
            str(row[id_col]).strip()
            for row in rows
            if str(row.get(pop_col, "")).strip().lower() in wanted and str(row.get(id_col, "")).strip()
        }
        if ids:
            sources.append({
                "source_type": "csv_population",
                "path": path.relative_to(context_root).as_posix(),
                "population_column": pop_col,
                "entity_key": id_col,
                "entity_count": len(ids),
                "sample_ids": sorted(ids)[:10],
                "_ids": ids,
            })

    for path in sorted(context_root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in {".md", ".txt"}:
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        ids: set[str] = set()
        paragraphs = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
        for paragraph in paragraphs:
            if not _sentence_contains_population_value(paragraph, wanted):
                continue
            paragraph_ids = set(ENTITY_ID_RE.findall(paragraph))
            if len(paragraph_ids) == 1:
                ids.update(paragraph_ids)
                continue
            for sentence in re.split(r"(?<=[.!?])\s+", paragraph):
                if _sentence_contains_population_value(sentence, wanted):
                    ids.update(ENTITY_ID_RE.findall(sentence))
        if ids:
            sources.append({
                "source_type": "doc_population",
                "path": path.relative_to(context_root).as_posix(),
                "population_terms": sorted(wanted),
                "entity_count": len(ids),
                "sample_ids": sorted(ids)[:10],
                "_ids": ids,
            })

    return sources


def _column_values(context_root: Path, path: str, column: str) -> list[dict[str, Any]]:
    header, rows = _read_csv_dicts(context_root / path)
    lower_to_header = {name.lower(): name for name in header}
    id_col = lower_to_header.get("id")
    values: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        parsed = _as_float(row.get(column, ""))
        if parsed is None:
            continue
        entity_id = str(row.get(id_col, "")).strip() if id_col else str(index)
        values.append({"entity_id": entity_id, "row_index": index, "value": parsed})
    return values


def _profile_values(values: list[float]) -> dict[str, Any]:
    if not values:
        return {}
    avg = mean(values)
    std = pstdev(values) if len(values) > 1 else 0.0
    return {
        "count": len(values),
        "min": _round(min(values)),
        "max": _round(max(values)),
        "mean": _round(avg),
        "std": _round(std),
        "p05": _round(_quantile(values, 0.05)),
        "p10": _round(_quantile(values, 0.10)),
        "p25": _round(_quantile(values, 0.25)),
        "p75": _round(_quantile(values, 0.75)),
        "p90": _round(_quantile(values, 0.90)),
        "p95": _round(_quantile(values, 0.95)),
    }


def _normal_candidates(profile: dict[str, Any]) -> list[dict[str, Any]]:
    if not profile:
        return []
    candidates = [
        ("observed_p05_p95", profile["p05"], profile["p95"]),
        ("observed_p10_p90", profile["p10"], profile["p90"]),
        ("observed_iqr", profile["p25"], profile["p75"]),
        ("all_observed_non_null", profile["min"], profile["max"]),
    ]
    if profile.get("std", 0) > 0:
        candidates.append(("mean_plus_minus_1std", profile["mean"] - profile["std"], profile["mean"] + profile["std"]))
    return [
        {"name": name, "low": _round(float(low)), "high": _round(float(high))}
        for name, low, high in candidates
        if low <= high
    ]


def _abnormal_candidates(profile: dict[str, Any]) -> list[dict[str, Any]]:
    candidates = [{"name": "non_null_as_abnormal_candidate", "mode": "non_null"}]
    for normal in _normal_candidates(profile):
        if normal["name"] == "all_observed_non_null":
            continue
        candidates.append({
            "name": f"outside_{normal['name']}",
            "mode": "outside_interval",
            "low": normal["low"],
            "high": normal["high"],
        })
    return candidates


def _passes_normal(value: float, candidate: dict[str, Any]) -> bool:
    return float(candidate["low"]) <= value <= float(candidate["high"])


def _passes_abnormal(value: float, candidate: dict[str, Any]) -> bool:
    if candidate.get("mode") == "non_null":
        return True
    return value < float(candidate["low"]) or value > float(candidate["high"])


def _candidate_probes(
    *,
    context_root: Path,
    matches: list[dict[str, Any]],
    population_sources: list[dict[str, Any]],
    max_candidates: int,
) -> list[dict[str, Any]]:
    if len(matches) < 2:
        return []
    first, second = matches[0], matches[1]
    if first["path"] != second["path"]:
        return []

    first_values = _column_values(context_root, first["path"], first["column"])
    second_values = _column_values(context_root, second["path"], second["column"])
    first_profile = _profile_values([item["value"] for item in first_values])
    second_profile = _profile_values([item["value"] for item in second_values])
    normal_candidates = _normal_candidates(first_profile)
    abnormal_candidates = _abnormal_candidates(second_profile)
    if not normal_candidates or not abnormal_candidates:
        return []

    by_entity_first: dict[str, list[tuple[int, float]]] = defaultdict(list)
    by_entity_second: dict[str, list[tuple[int, float]]] = defaultdict(list)
    by_row_first: dict[tuple[str, int], float] = {}
    by_row_second: dict[tuple[str, int], float] = {}
    for item in first_values:
        by_entity_first[item["entity_id"]].append((item["row_index"], item["value"]))
        by_row_first[(item["entity_id"], item["row_index"])] = item["value"]
    for item in second_values:
        by_entity_second[item["entity_id"]].append((item["row_index"], item["value"]))
        by_row_second[(item["entity_id"], item["row_index"])] = item["value"]

    sources = population_sources or [{
        "source_type": "all_entities_in_data",
        "path": first["path"],
        "entity_count": len(set(by_entity_first) | set(by_entity_second)),
        "_ids": set(by_entity_first) | set(by_entity_second),
    }]

    probes: list[dict[str, Any]] = []
    for source in sources:
        source_ids = set(source.get("_ids", set()))
        for normal in normal_candidates:
            normal_entities = {
                entity for entity, values in by_entity_first.items()
                if entity in source_ids and any(_passes_normal(value, normal) for _row, value in values)
            }
            normal_rows = {
                (entity, row)
                for entity, values in by_entity_first.items()
                if entity in source_ids
                for row, value in values
                if _passes_normal(value, normal)
            }
            for abnormal in abnormal_candidates:
                abnormal_entities = {
                    entity for entity, values in by_entity_second.items()
                    if entity in source_ids and any(_passes_abnormal(value, abnormal) for _row, value in values)
                }
                abnormal_rows = {
                    (entity, row)
                    for entity, values in by_entity_second.items()
                    if entity in source_ids
                    for row, value in values
                    if _passes_abnormal(value, abnormal)
                }
                same_row_ids = {entity for entity, row in normal_rows & abnormal_rows}
                entity_ids = normal_entities & abnormal_entities
                probes.append({
                    "population_source": {
                        key: value for key, value in source.items() if not key.startswith("_")
                    },
                    "normal_column": first["column"],
                    "normal_candidate": normal,
                    "abnormal_column": second["column"],
                    "abnormal_candidate": abnormal,
                    "same_row_distinct_entity_count": len(same_row_ids),
                    "entity_level_distinct_count": len(entity_ids),
                    "sample_entity_ids": sorted(entity_ids)[:10],
                    "note": (
                        "entity_level counts allow the normal and abnormal measurements "
                        "to occur on different rows for the same entity"
                    ),
                })

    probes.sort(
        key=lambda item: (
            item["population_source"].get("source_type") != "doc_population",
            item["abnormal_candidate"].get("mode") != "non_null",
            -item["entity_level_distinct_count"],
            -item["same_row_distinct_entity_count"],
        )
    )
    return probes[:max_candidates]


def ground_thresholds(
    context_root: Path,
    target_terms: list[str],
    *,
    population_terms: list[str] | None = None,
    max_candidates: int = 20,
) -> dict[str, Any]:
    """Return explicit and data-derived threshold evidence for target terms."""
    terms = [term.strip() for term in target_terms if term and term.strip()]
    populations = [term.strip() for term in (population_terms or []) if term and term.strip()]
    matches = _match_columns(context_root, terms)
    snippets = _threshold_snippets(context_root, terms)
    sources = _population_sources(context_root, populations)

    column_profiles = []
    for match in matches:
        values = _column_values(context_root, match["path"], match["column"])
        profile = _profile_values([item["value"] for item in values])
        column_profiles.append({
            **match,
            "profile": profile,
            "normal_candidates": _normal_candidates(profile),
            "abnormal_candidates": _abnormal_candidates(profile)[:6],
        })

    probes = _candidate_probes(
        context_root=context_root,
        matches=matches,
        population_sources=sources,
        max_candidates=max_candidates,
    )

    public_sources = [
        {key: value for key, value in source.items() if not key.startswith("_")}
        for source in sources
    ]
    recommended_probe = probes[0] if probes else None
    source_counts = [
        {
            "source_type": source["source_type"],
            "path": source.get("path"),
            "entity_count": source["entity_count"],
        }
        for source in public_sources
    ]
    source_counts.sort(key=lambda item: int(item["entity_count"]), reverse=True)
    recommended_summary = None
    if recommended_probe:
        source_type = recommended_probe["population_source"].get("source_type")
        source_path = recommended_probe["population_source"].get("path")
        entity_count = recommended_probe["entity_level_distinct_count"]
        same_row_count = recommended_probe["same_row_distinct_entity_count"]
        recommended_summary = {
            "entity_level_candidate_count": entity_count,
            "candidate_count": entity_count,
            "same_row_count": same_row_count,
            "population_source": recommended_probe["population_source"],
            "normal_rule": recommended_probe["normal_candidate"],
            "abnormal_rule": recommended_probe["abnormal_candidate"],
            "sample_entity_ids": recommended_probe["sample_entity_ids"],
            "replication_plan": [
                f"Use the population ids from {source_type}:{source_path}.",
                (
                    "Build the normal-measurement entity set from any row satisfying "
                    f"{recommended_probe['normal_column']} within the normal_rule."
                ),
                (
                    "Build the abnormal-measurement entity set from any row satisfying "
                    f"{recommended_probe['abnormal_column']} with the abnormal_rule."
                ),
                "Return the distinct entity count in the intersection of those two entity sets.",
            ],
            "anti_patterns": [
                "Do not require the normal and abnormal measurements to appear on the same row.",
                "Do not join only to a smaller helper CSV when population_source_comparison shows a richer doc population.",
                "Do not replace the candidate rules with outside domain defaults.",
            ],
            "decision_note": (
                "For patient/member/entity count questions, candidate_count is the entity-level answer candidate. "
                "Prefer it over same_row_count unless the question explicitly requires the same measurement row."
            ),
        }
    return {
        "target_terms": terms,
        "population_terms": populations,
        "recommended_probe_summary": recommended_summary,
        "population_source_comparison": source_counts,
        "population_sources": public_sources,
        "explicit_threshold_snippets": snippets,
        "candidate_probes": probes,
        "matched_numeric_columns": column_profiles,
        "guidance": [
            "Prefer explicit_threshold_snippets when they directly define the target term.",
            (
                "If explicit_threshold_snippets is empty, do not use outside or memorized thresholds; "
                "compare candidate_probes instead."
            ),
            "When recommended_probe_summary is present, it is the most compact context-only candidate to verify first.",
            (
                "For entity questions, prefer entity_level_distinct_count over same-row count when the wording "
                "says the entity has multiple properties."
            ),
            (
                "When multiple population sources exist, inspect source coverage; docs may contain entity "
                "filters absent from a small helper CSV."
            ),
            (
                "non_null_as_abnormal_candidate is appropriate when no explicit abnormal threshold exists "
                "and the target column is sparse or appears to store only recorded abnormal results."
            ),
        ],
    }
