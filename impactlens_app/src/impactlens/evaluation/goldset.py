from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class GoldFact:
    project_id: str
    field_path: str  # dot path with optional [i] indices
    value_type: str  # "number" | "date" | "string"
    value: Any


def load_gold_facts(path: Path) -> list[GoldFact]:
    """Load gold facts from a JSONL file (one JSON object per line).

    Each line:
      { "project_id": "P...", "field_path": "objective.text", "value_type": "string", "value": "..." }
    """
    facts: list[GoldFact] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        facts.append(
            GoldFact(
                project_id=str(obj["project_id"]),
                field_path=str(obj["field_path"]),
                value_type=str(obj["value_type"]),
                value=obj.get("value"),
            )
        )
    return facts


def get_path(obj: Any, field_path: str) -> Any:
    """Resolve a simple dot path with optional [index] selectors."""
    cur: Any = obj
    for part in field_path.split("."):
        if part.endswith("]") and "[" in part:
            name, idx_s = part[:-1].split("[", 1)
            if name:
                cur = cur.get(name) if isinstance(cur, dict) else None
            try:
                idx = int(idx_s)
            except ValueError:
                return None
            if not isinstance(cur, list) or idx >= len(cur):
                return None
            cur = cur[idx]
        else:
            cur = cur.get(part) if isinstance(cur, dict) else None
    return cur

