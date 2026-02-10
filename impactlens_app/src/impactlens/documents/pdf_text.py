from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import fitz  # PyMuPDF
from PIL import Image


@dataclass(frozen=True)
class PageText:
    page_number: int  # 1-indexed
    text: str


def extract_text_per_page(pdf_path: str) -> list[PageText]:
    doc = fitz.open(pdf_path)
    pages: list[PageText] = []
    try:
        for i in range(doc.page_count):
            page = doc.load_page(i)
            text = page.get_text("text") or ""
            pages.append(PageText(page_number=i + 1, text=text.strip()))
    finally:
        doc.close()
    return pages


def render_page_png_bytes(pdf_path: str, page_number: int, *, dpi: int = 200) -> bytes:
    """Render a PDF page (1-indexed) to PNG bytes."""
    doc = fitz.open(pdf_path)
    try:
        page = doc.load_page(page_number - 1)
        zoom = dpi / 72.0
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        from io import BytesIO

        buf = BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()
    finally:
        doc.close()

