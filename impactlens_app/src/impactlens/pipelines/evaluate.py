from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import polars as pl

from impactlens.evaluation.goldset import load_gold_facts
from impactlens.evaluation.metrics import EvalSummary, evaluate_facts
from impactlens.storage import read_parquet, write_manifest, write_parquet
from impactlens.utils import utc_now


@dataclass(frozen=True)
class EvaluateResult:
    dataset_version_id: str
    summary: EvalSummary
    output_path: str


def evaluate(
    *,
    dataset_version_id: str,
    gold_facts_path: Path,
) -> EvaluateResult:
    gold = load_gold_facts(gold_facts_path)

    ex_df = read_parquet(layer="gold", name="extractions", dataset_version_id=dataset_version_id)
    # Keep latest per project_id
    ex_df = ex_df.sort("created_at").group_by("project_id").tail(1)

    extractions_by_project: dict[str, dict] = {}
    for row in ex_df.iter_rows(named=True):
        extractions_by_project[str(row["project_id"])] = json.loads(row["output_json"])

    summary = evaluate_facts(extractions_by_project, gold)

    out_df = pl.from_dicts(
        [
            {
                "dataset_version_id": dataset_version_id,
                "created_at": utc_now().isoformat(),
                "gold_facts_path": str(gold_facts_path),
                "n_facts": summary.n_facts,
                "n_matched": summary.n_matched,
                "accuracy": summary.accuracy,
            }
        ]
    )
    path = write_parquet(out_df, layer="gold", name="eval_runs", dataset_version_id=dataset_version_id)

    write_manifest(
        dataset_version_id,
        {
            "pipeline": "evaluate",
            "gold_facts_path": str(gold_facts_path),
            "metrics": {"accuracy": summary.accuracy, "n_facts": summary.n_facts},
        },
    )

    return EvaluateResult(dataset_version_id=dataset_version_id, summary=summary, output_path=str(path))

