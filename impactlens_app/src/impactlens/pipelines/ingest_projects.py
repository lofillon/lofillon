from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import polars as pl

from impactlens.clients.http import HttpConfig
from impactlens.clients.worldbank import WorldBankClient
from impactlens.settings import settings
from impactlens.storage import write_manifest, write_parquet
from impactlens.utils import utc_now


@dataclass(frozen=True)
class IngestProjectsResult:
    dataset_version_id: str
    n_projects: int


async def ingest_projects(
    *,
    dataset_version_id: str,
    rows: int = 100,
    max_pages: int = 1,
) -> IngestProjectsResult:
    cfg = HttpConfig(timeout_s=settings.http_timeout_s)
    wb = WorldBankClient.from_config(cfg)
    records: list[dict[str, Any]] = []

    try:
        async for payload in wb.iter_projects(rows=rows, max_pages=max_pages):
            project_id = str(payload.get("id") or "").strip()
            if not project_id:
                continue
            records.append(
                {
                    "project_id": project_id,
                    "snapshot_date": dataset_version_id,
                    "ingested_at": utc_now().isoformat(),
                    "payload_json": json.dumps(payload, ensure_ascii=False),
                }
            )
    finally:
        await wb.aclose()

    df = pl.from_dicts(records) if records else pl.DataFrame(
        {"project_id": [], "snapshot_date": [], "ingested_at": [], "payload_json": []}
    )
    write_parquet(df, layer="bronze", name="projects_raw", dataset_version_id=dataset_version_id)

    write_manifest(
        dataset_version_id,
        {
            "pipeline": "ingest_projects",
            "source": "https://search.worldbank.org/api/v2/projects",
            "params": {"rows": rows, "max_pages": max_pages},
            "counts": {"projects": int(df.height)},
        },
    )
    return IngestProjectsResult(dataset_version_id=dataset_version_id, n_projects=int(df.height))

