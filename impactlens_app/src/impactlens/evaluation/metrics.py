from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from impactlens.evaluation.goldset import GoldFact, get_path


@dataclass(frozen=True)
class EvalSummary:
    n_facts: int
    n_matched: int
    accuracy: float


def _normalize_number(x: Any) -> float | None:
    if x is None:
        return None
    if isinstance(x, (int, float)):
        return float(x)
    try:
        return float(str(x).replace(",", "").strip())
    except ValueError:
        return None


def _normalize_date(x: Any) -> str | None:
    if x is None:
        return None
    s = str(x).strip()
    if not s:
        return None
    # Keep ISO date prefix if present
    return s[:10] if len(s) >= 10 else s


def evaluate_facts(extractions_by_project: dict[str, dict[str, Any]], gold: list[GoldFact]) -> EvalSummary:
    matched = 0
    for fact in gold:
        pred_obj = extractions_by_project.get(fact.project_id)
        if pred_obj is None:
            continue
        pred_val = get_path(pred_obj, fact.field_path)

        if fact.value_type == "number":
            if _normalize_number(pred_val) == _normalize_number(fact.value):
                matched += 1
        elif fact.value_type == "date":
            if _normalize_date(pred_val) == _normalize_date(fact.value):
                matched += 1
        else:
            if (pred_val is not None) and (str(pred_val).strip() == str(fact.value).strip()):
                matched += 1

    n = len(gold)
    acc = matched / n if n else 0.0
    return EvalSummary(n_facts=n, n_matched=matched, accuracy=acc)

