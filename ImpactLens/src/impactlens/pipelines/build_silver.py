from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import polars as pl

from impactlens.parsing import parse_datetime, parse_money_usd
from impactlens.documents.selection import normalize_doc_class, score_document
from impactlens.storage import read_parquet, write_manifest, write_parquet
from impactlens.utils import utc_now


@dataclass(frozen=True)
class BuildSilverResult:
    dataset_version_id: str
    n_projects: int


@dataclass(frozen=True)
class BuildSilverDocumentsResult:
    dataset_version_id: str
    n_documents: int


def _first(x: Any) -> Any:
    if isinstance(x, list) and len(x) > 0:
        return x[0]
    return x


def build_silver_projects(*, dataset_version_id: str) -> BuildSilverResult:
    bronze = read_parquet(layer="bronze", name="projects_raw", dataset_version_id=dataset_version_id)
    out: list[dict[str, Any]] = []

    for row in bronze.iter_rows(named=True):
        payload = json.loads(row["payload_json"])

        project_id = str(payload.get("id") or "").strip()
        if not project_id:
            continue

        country_code = _first(payload.get("countrycode"))
        country_name = _first(payload.get("countryname"))

        boardapprovaldate = parse_datetime(payload.get("boardapprovaldate"))
        closingdate = parse_datetime(payload.get("closingdate"))
        updated = parse_datetime(payload.get("p2a_updated_date"))

        out.append(
            {
                "project_id": project_id,
                "project_name": payload.get("project_name") or payload.get("proj_title"),
                "status": payload.get("status") or payload.get("projectstatusdisplay"),
                "regionname": payload.get("regionname"),
                "country_iso2": country_code,
                "country_name": country_name or payload.get("countryshortname"),
                "approval_fy": payload.get("approvalfy"),
                "board_approval_date": boardapprovaldate.isoformat() if boardapprovaldate else None,
                "closing_date": closingdate.isoformat() if closingdate else None,
                "total_commitment_usd": parse_money_usd(payload.get("totalamt") or payload.get("totalcommamt")),
                "lending_instrument": payload.get("lendinginstr"),
                "prodline": payload.get("prodline"),
                "updated_at": updated.isoformat() if updated else None,
                "source_url": payload.get("url"),
                "ingested_at": row.get("ingested_at"),
                "dataset_version_id": dataset_version_id,
            }
        )

    df = pl.from_dicts(out) if out else pl.DataFrame(
        {
            "project_id": [],
            "project_name": [],
            "status": [],
            "regionname": [],
            "country_iso2": [],
            "country_name": [],
            "approval_fy": [],
            "board_approval_date": [],
            "closing_date": [],
            "total_commitment_usd": [],
            "lending_instrument": [],
            "prodline": [],
            "updated_at": [],
            "source_url": [],
            "ingested_at": [],
            "dataset_version_id": [],
        }
    )
    write_parquet(df, layer="silver", name="projects", dataset_version_id=dataset_version_id)
    write_manifest(
        dataset_version_id,
        {
            "pipeline": "build_silver_projects",
            "created_at": utc_now().isoformat(),
            "inputs": [{"layer": "bronze", "name": "projects_raw"}],
            "outputs": [{"layer": "silver", "name": "projects"}],
            "counts": {"projects": int(df.height)},
        },
    )
    return BuildSilverResult(dataset_version_id=dataset_version_id, n_projects=int(df.height))


def build_silver_documents(*, dataset_version_id: str) -> BuildSilverDocumentsResult:
    bronze = read_parquet(layer="bronze", name="documents_raw", dataset_version_id=dataset_version_id)
    out: list[dict[str, Any]] = []

    for row in bronze.iter_rows(named=True):
        payload = json.loads(row["payload_json"])
        project_id = str(row.get("project_id") or "").strip()
        doc_id = str(payload.get("id") or payload.get("repnb") or row.get("doc_id") or "").strip()
        if not doc_id:
            continue

        docdt = parse_datetime(payload.get("docdt"))
        stored = parse_datetime(payload.get("datestored"))
        disclosure = parse_datetime(payload.get("disclosure_date"))

        title = payload.get("display_title") or payload.get("repnme", {}).get("repnme")
        lang = payload.get("lang") or payload.get("available_in")
        docty = payload.get("docty")
        majdocty = payload.get("majdocty")
        seccl = payload.get("seccl")
        pdf_url = payload.get("pdfurl")
        has_pdf = bool(pdf_url)
        is_public = (seccl or "").lower() == "public"
        doc_class = normalize_doc_class(title, docty, majdocty)
        doc_score = score_document(
            doc_class=doc_class,
            has_pdf=has_pdf,
            is_public=is_public,
            title=title,
            docty=docty,
            majdocty=majdocty,
        )

        out.append(
            {
                "doc_id": doc_id,
                "project_id": project_id or None,
                "doc_type": docty or majdocty,
                "doc_class": doc_class,
                "doc_score": float(doc_score),
                "language": lang,
                "title": title,
                "publication_date": docdt.isoformat() if docdt else None,
                "stored_date": stored.isoformat() if stored else None,
                "disclosure_date": disclosure.isoformat() if disclosure else None,
                "pdf_url": pdf_url,
                "txt_url": payload.get("txturl"),
                "url": payload.get("url"),
                "security_classification": seccl,
                "is_public": bool(is_public),
                "has_pdf": bool(has_pdf),
                "dataset_version_id": dataset_version_id,
                "ingested_at": row.get("ingested_at"),
            }
        )

    df = pl.from_dicts(out) if out else pl.DataFrame(
        {
            "doc_id": [],
            "project_id": [],
            "doc_type": [],
            "doc_class": [],
            "doc_score": [],
            "language": [],
            "title": [],
            "publication_date": [],
            "stored_date": [],
            "disclosure_date": [],
            "pdf_url": [],
            "txt_url": [],
            "url": [],
            "security_classification": [],
            "is_public": [],
            "has_pdf": [],
            "dataset_version_id": [],
            "ingested_at": [],
        }
    )
    write_parquet(df, layer="silver", name="documents", dataset_version_id=dataset_version_id)
    write_manifest(
        dataset_version_id,
        {
            "pipeline": "build_silver_documents",
            "created_at": utc_now().isoformat(),
            "inputs": [{"layer": "bronze", "name": "documents_raw"}],
            "outputs": [{"layer": "silver", "name": "documents"}],
            "counts": {"documents": int(df.height)},
        },
    )
    return BuildSilverDocumentsResult(dataset_version_id=dataset_version_id, n_documents=int(df.height))

