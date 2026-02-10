from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from impactlens.documents.download import DownloadResult, download_file


@dataclass(frozen=True)
class DownloadTextResult:
    url: str
    path: Path
    ok: bool
    error: str | None


async def download_text(url: str, dest: Path, *, overwrite: bool = False) -> DownloadTextResult:
    """Download a text artifact and store it locally.

    Uses the same HTTP client config/retries as `download_file`.
    """
    r: DownloadResult = await download_file(url, dest, overwrite=overwrite)
    return DownloadTextResult(url=url, path=r.path, ok=r.ok, error=r.error)

