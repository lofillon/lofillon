from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import httpx

from impactlens.clients.http import HttpConfig, build_async_client, get_json
from impactlens.settings import settings
from impactlens.utils import sha256_file
import re


_WB_DOCUMENTDETAIL_RE = re.compile(
    r"^https?://documents\.worldbank\.org/(?:[a-z]{2}/)?en/publication/documents-reports/documentdetail/\d+",
    re.IGNORECASE,
)


def _looks_like_worldbank_documentdetail(url: str) -> bool:
    return bool(_WB_DOCUMENTDETAIL_RE.match((url or "").strip()))


def _extract_documentdetail_token(url: str) -> str | None:
    """Extract the trailing numeric token from documentdetail URLs."""
    u = (url or "").strip().rstrip("/")
    token = u.rsplit("/", 1)[-1] if u else ""
    if re.fullmatch(r"\d{12,}", token):
        return token
    return None


def _extract_pdf_urls_from_html(html: str) -> list[str]:
    """Best-effort extraction of direct PDF links from WB landing pages."""
    if not html:
        return []
    # Absolute URLs
    abs_urls = re.findall(r"https?://[^\\s\"'<>]+?\\.pdf(?:\\?[^\\s\"'<>]+)?", html, flags=re.IGNORECASE)
    # href="...pdf"
    href_urls = re.findall(r'href\\s*=\\s*["\\\']([^"\\\']+?\\.pdf(?:\\?[^"\\\']+)?)["\\\']', html, flags=re.IGNORECASE)
    urls = list(dict.fromkeys([u.strip() for u in (abs_urls + href_urls) if isinstance(u, str) and u.strip()]))

    def _rank(u: str) -> tuple[int, int]:
        # Prefer curated/documents1 style links that are direct downloads.
        s = u.lower()
        score = 0
        if "documents1.worldbank.org" in s:
            score += 3
        if "/curated/" in s:
            score += 2
        if "download" in s:
            score += 1
        return (-score, len(u))

    return sorted(urls, key=_rank)


async def _maybe_resolve_worldbank_pdf_url(client: httpx.AsyncClient, url: str) -> str:
    """If `url` is a WB document landing page, resolve to a direct PDF URL."""
    if not _looks_like_worldbank_documentdetail(url):
        return url
    token = _extract_documentdetail_token(url)
    if token:
        # The landing page itself is mostly JS. Use WDS API to resolve to pdfurl/txturl.
        try:
            wds_url = "https://search.worldbank.org/api/v3/wds"
            params = {
                "format": "json",
                "qterm": token,
                "rows": 5,
                "fl": "id,repnb,pdfurl,txturl,url,display_title,docdt,seccl,docty,majdocty,projectid",
            }
            data = await get_json(client, wds_url, params=params)
            docs = (data or {}).get("documents") if isinstance(data, dict) else None
            if isinstance(docs, dict):
                candidates: list[dict[str, str]] = []
                for _, d in docs.items():
                    if not isinstance(d, dict):
                        continue
                    pdf = str(d.get("pdfurl") or "").strip()
                    u = str(d.get("url") or "").strip()
                    if pdf:
                        candidates.append({"pdfurl": pdf, "url": u})
                # Prefer the candidate whose canonical `url` contains the token.
                for c in candidates:
                    if token in (c.get("url") or ""):
                        return c["pdfurl"]
                if candidates:
                    return candidates[0]["pdfurl"]
        except Exception:  # noqa: BLE001
            pass
    try:
        r = await client.get(url)
        r.raise_for_status()
        ctype = (r.headers.get("content-type") or "").lower()
        # If we already got a PDF, keep it.
        if "pdf" in ctype:
            return url
        # If HTML, try to find a PDF link inside.
        if "html" in ctype or "<html" in (r.text or "")[:500].lower():
            candidates = _extract_pdf_urls_from_html(r.text or "")
            if candidates:
                return candidates[0]
    except Exception:  # noqa: BLE001
        return url
    return url


@dataclass(frozen=True)
class DownloadResult:
    url: str
    path: Path
    ok: bool
    status_code: int | None
    bytes: int | None
    sha256: str | None
    error: str | None


async def download_file(url: str, dest: Path, *, overwrite: bool = False) -> DownloadResult:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and not overwrite:
        return DownloadResult(
            url=url,
            path=dest,
            ok=True,
            status_code=None,
            bytes=dest.stat().st_size,
            sha256=sha256_file(dest),
            error=None,
        )

    cfg = HttpConfig(timeout_s=settings.http_timeout_s)
    client = build_async_client(cfg)
    r: httpx.Response | None = None
    try:
        resolved_url = await _maybe_resolve_worldbank_pdf_url(client, str(url))
        r = await client.get(resolved_url)
        r.raise_for_status()
        dest.write_bytes(r.content)
        return DownloadResult(
            url=resolved_url,
            path=dest,
            ok=True,
            status_code=r.status_code,
            bytes=len(r.content),
            sha256=sha256_file(dest),
            error=None,
        )
    except Exception as e:  # noqa: BLE001
        return DownloadResult(
            url=url,
            path=dest,
            ok=False,
            status_code=(r.status_code if r is not None else None),
            bytes=None,
            sha256=None,
            error=str(e),
        )
    finally:
        await client.aclose()

