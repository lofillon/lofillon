from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Iterable

import polars as pl

from impactlens.clients.http import HttpConfig
from impactlens.clients.worldbank import WorldBankClient
from impactlens.settings import settings
from impactlens.storage import write_manifest, write_parquet
from impactlens.utils import utc_now


@dataclass(frozen=True)
class IngestDocumentsResult:
    dataset_version_id: str
    n_documents: int


def _iter_project_ids(project_ids: list[str] | None) -> Iterable[str]:
    if not project_ids:
        return []
    return [p.strip() for p in project_ids if p.strip()]


async def ingest_documents_for_projects(
    *,
    dataset_version_id: str,
    project_ids: list[str],
    rows: int = 100,
    max_pages: int = 1,
    project_docs_only: bool = True,
    docty_exact: list[str] | None = None,
    fl: str | None = "id,repnb,docdt,lang,display_title,pdfurl,docty,majdocty,seccl,url,projectid,proid,projn,count,countcode",
) -> IngestDocumentsResult:
    """Ingest WDS documents for projects (WDS).

    Retrieval strategy:
    1) Field query: `projectid=<ID>` (more precise)
    2) Fallback: `qterm=<ID>` (broader)
    """
    cfg = HttpConfig(timeout_s=settings.http_timeout_s)
    wb = WorldBankClient.from_config(cfg)
    records: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    try:
        for project_id in _iter_project_ids(project_ids):
            base_filters: dict[str, Any] = {"sort": "docdt", "order": "desc"}
            if project_docs_only:
                base_filters["majdocty_exact"] = "Project Documents"
            if docty_exact:
                base_filters["docty_exact"] = "^".join(docty_exact)

            for mode, qterm, extra in [
                ("field_projectid", None, {"projectid": project_id, **base_filters}),
                ("fallback_qterm", project_id, base_filters),
            ]:
                async for doc in wb.iter_wds_documents(qterm=qterm, rows=rows, max_pages=max_pages, fl=fl, extra_params=extra):
                    doc_id = str(doc.get("id") or doc.get("repnb") or "").strip()
                    if not doc_id:
                        continue
                    key = (project_id, doc_id)
                    if key in seen:
                        continue
                    seen.add(key)
                    records.append(
                        {
                            "project_id": project_id,
                            "doc_id": doc_id,
                            "query_mode": mode,
                            "snapshot_date": dataset_version_id,
                            "ingested_at": utc_now().isoformat(),
                            "payload_json": json.dumps(doc, ensure_ascii=False),
                        }
                    )
    finally:
        await wb.aclose()

    df = pl.from_dicts(records) if records else pl.DataFrame(
        {
            "project_id": [],
            "doc_id": [],
            "query_mode": [],
            "snapshot_date": [],
            "ingested_at": [],
            "payload_json": [],
        }
    )
    write_parquet(df, layer="bronze", name="documents_raw", dataset_version_id=dataset_version_id)
    write_manifest(
        dataset_version_id,
        {
            "pipeline": "ingest_documents_for_projects",
            "source": "https://search.worldbank.org/api/v3/wds",
            "params": {
                "rows": rows,
                "max_pages": max_pages,
                "project_docs_only": project_docs_only,
                "docty_exact": docty_exact,
                "sort": "docdt",
                "order": "desc",
                "fl": fl,
            },
            "counts": {"documents": int(df.height)},
        },
    )
    return IngestDocumentsResult(dataset_version_id=dataset_version_id, n_documents=int(df.height))

