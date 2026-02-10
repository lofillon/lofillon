from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import polars as pl

from impactlens.documents.download import download_file
from impactlens.documents.ocr import ocr_image_bytes
from impactlens.documents.pdf_text import extract_text_per_page, render_page_png_bytes
from impactlens.documents.text_source import download_text
from impactlens.documents.quality import alpha_ratio, digit_ratio, non_alnum_ratio, section_hint_score, text_length
from impactlens.storage import artifact_dir, read_parquet, write_manifest, write_parquet
from impactlens.utils import utc_now


@dataclass(frozen=True)
class ProcessDocumentsResult:
    dataset_version_id: str
    n_docs_attempted: int
    n_docs_downloaded_ok: int
    n_chunks: int


async def process_documents(
    *,
    dataset_version_id: str,
    project_id: str | None = None,
    limit: int | None = None,
    top_k_per_project: int = 5,
    process_all_docs: bool = False,
    ocr_min_total_chars: int = 500,
    ocr_dpi: int = 200,
    ocr_min_page_chars: int = 40,
) -> ProcessDocumentsResult:
    """Download PDFs and build `silver/document_text_chunks`.

    Strategy:
    - Try text extraction via PyMuPDF.
    - If total extracted chars is low, run OCR per page (EasyOCR).
    """
    # Prefer a frozen selection if available, unless we explicitly want all docs.
    if process_all_docs:
        docs = read_parquet(layer="silver", name="documents", dataset_version_id=dataset_version_id)
        input_table = "documents"
    else:
        try:
            docs = read_parquet(layer="silver", name="documents_selected", dataset_version_id=dataset_version_id)
            input_table = "documents_selected"
        except Exception:  # noqa: BLE001
            docs = read_parquet(layer="silver", name="documents", dataset_version_id=dataset_version_id)
            input_table = "documents"
    # keep only public docs with a PDF
    if "is_public" in docs.columns:
        docs = docs.filter(pl.col("is_public") == True)  # noqa: E712
    # Some sources provide a landing-page `url` instead of a direct `pdf_url`.
    # We can still process those by resolving the landing page to a PDF at download-time.
    if "pdf_url" in docs.columns and "url" in docs.columns:
        docs = docs.filter(pl.col("pdf_url").is_not_null() | pl.col("url").is_not_null())
    elif "pdf_url" in docs.columns:
        docs = docs.filter(pl.col("pdf_url").is_not_null())
    if project_id is not None:
        docs = docs.filter(pl.col("project_id") == project_id)

    # prioritize best docs per project unless we explicitly want to process everything
    if (not process_all_docs) and "doc_score" in docs.columns and docs.height > 0:
        docs = (
            docs.sort(["project_id", "doc_score"], descending=[False, True])
            .group_by("project_id")
            .head(top_k_per_project)
        )
    if limit is not None:
        docs = docs.head(limit)

    pdf_dir = artifact_dir("pdfs", dataset_version_id)
    artifacts: list[dict[str, Any]] = []
    chunks: list[dict[str, Any]] = []

    n_ok = 0
    for row in docs.iter_rows(named=True):
        doc_id = str(row.get("doc_id") or "").strip()
        # Prefer direct PDF URL, else fall back to landing page URL (resolver handles World Bank documentdetail pages).
        pdf_url = row.get("pdf_url") or row.get("url")
        txt_url = row.get("txt_url")
        if not doc_id or not pdf_url:
            continue

        pdf_path = (pdf_dir / f"{doc_id}.pdf").resolve()
        dl = await download_file(str(pdf_url), pdf_path)
        artifacts.append(
            {
                "doc_id": doc_id,
                "project_id": row.get("project_id"),
                "pdf_url": pdf_url,
                "path": str(pdf_path),
                "ok": dl.ok,
                "status_code": dl.status_code,
                "bytes": dl.bytes,
                "sha256": dl.sha256,
                "downloaded_at": utc_now().isoformat(),
                "error": dl.error,
                "dataset_version_id": dataset_version_id,
            }
        )
        if not dl.ok:
            continue
        n_ok += 1

        # Prefer WDS-provided text when available (often better for tables/annexes).
        pages = extract_text_per_page(str(pdf_path))
        total_chars = sum(len(p.text) for p in pages)
        method = "pdf_text"

        if txt_url:
            txt_path = (pdf_dir / f"{doc_id}.txt").resolve()
            dl_txt = await download_text(str(txt_url), txt_path)
            if dl_txt.ok:
                try:
                    text = txt_path.read_text(encoding="utf-8", errors="ignore").strip()
                    # Treat as a single "page" chunk (page=1) to provide strong searchable evidence.
                    pages = [type(pages[0]) (page_number=1, text=text)] if pages else []
                    total_chars = len(text)
                    method = "wds_txt"
                except Exception:  # noqa: BLE001
                    # Fall back to PDF text/OCR below.
                    pass

        if total_chars < ocr_min_total_chars:
            # OCR fallback: render each page and OCR it.
            method = "ocr"
            pages_ocr = []
            for p in pages:
                img_bytes = render_page_png_bytes(str(pdf_path), p.page_number, dpi=ocr_dpi)
                text = ocr_image_bytes(img_bytes)
                pages_ocr.append(type(p)(page_number=p.page_number, text=text.strip()))
            pages = pages_ocr
        else:
            # Per-page OCR fallback: if some pages are essentially empty/garbled, OCR only those pages.
            pages_fixed = []
            for p in pages:
                txt = (p.text or "").strip()
                if len(txt) < ocr_min_page_chars:
                    img_bytes = render_page_png_bytes(str(pdf_path), p.page_number, dpi=ocr_dpi)
                    txt = ocr_image_bytes(img_bytes).strip()
                    if txt:
                        method = "pdf_text+ocr_pages"
                pages_fixed.append(type(p)(page_number=p.page_number, text=txt))
            pages = pages_fixed

        for p in pages:
            chunk_id = f"{doc_id}_p{p.page_number}"
            txt = p.text or ""
            chunks.append(
                {
                    "chunk_id": chunk_id,
                    "doc_id": doc_id,
                    "project_id": row.get("project_id"),
                    "page_start": p.page_number,
                    "page_end": p.page_number,
                    "text": txt,
                    "text_length": text_length(txt),
                    "digit_ratio": digit_ratio(txt),
                    "non_alnum_ratio": non_alnum_ratio(txt),
                    "alpha_ratio": alpha_ratio(txt),
                    "section_hint_score": section_hint_score(txt),
                    "extraction_method": method,
                    "dataset_version_id": dataset_version_id,
                    "created_at": utc_now().isoformat(),
                }
            )

    df_artifacts = pl.from_dicts(artifacts) if artifacts else pl.DataFrame(
        {
            "doc_id": [],
            "project_id": [],
            "pdf_url": [],
            "path": [],
            "ok": [],
            "status_code": [],
            "bytes": [],
            "sha256": [],
            "downloaded_at": [],
            "error": [],
            "dataset_version_id": [],
        }
    )
    df_chunks = pl.from_dicts(chunks) if chunks else pl.DataFrame(
        {
            "chunk_id": [],
            "doc_id": [],
            "project_id": [],
            "page_start": [],
            "page_end": [],
            "text": [],
            "text_length": [],
            "digit_ratio": [],
            "non_alnum_ratio": [],
            "alpha_ratio": [],
            "section_hint_score": [],
            "extraction_method": [],
            "dataset_version_id": [],
            "created_at": [],
        }
    )

    write_parquet(df_artifacts, layer="bronze", name="pdf_artifacts", dataset_version_id=dataset_version_id)
    write_parquet(df_chunks, layer="silver", name="document_text_chunks", dataset_version_id=dataset_version_id)

    write_manifest(
        dataset_version_id,
        {
            "pipeline": "process_documents",
            "inputs": [{"layer": "silver", "name": input_table}],
            "outputs": [
                {"layer": "bronze", "name": "pdf_artifacts"},
                {"layer": "silver", "name": "document_text_chunks"},
            ],
            "params": {
                "project_id": project_id,
                "limit": limit,
                "top_k_per_project": top_k_per_project,
                "ocr_min_total_chars": ocr_min_total_chars,
                "ocr_dpi": ocr_dpi,
                "ocr_min_page_chars": ocr_min_page_chars,
            },
            "counts": {"docs_attempted": int(docs.height), "docs_ok": n_ok, "chunks": int(df_chunks.height)},
        },
    )

    return ProcessDocumentsResult(
        dataset_version_id=dataset_version_id,
        n_docs_attempted=int(docs.height),
        n_docs_downloaded_ok=n_ok,
        n_chunks=int(df_chunks.height),
    )

