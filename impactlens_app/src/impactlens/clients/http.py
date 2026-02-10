from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential_jitter


class TransientHttpError(RuntimeError):
    pass


@dataclass(frozen=True)
class HttpConfig:
    timeout_s: float = 60.0
    user_agent: str = "ImpactLens/0.1"


def _is_retryable_status(code: int) -> bool:
    return code in {408, 409, 425, 429, 500, 502, 503, 504}


@retry(
    retry=retry_if_exception_type((httpx.RequestError, TransientHttpError)),
    wait=wait_exponential_jitter(initial=0.5, max=10.0),
    stop=stop_after_attempt(5),
    reraise=True,
)
async def get_json(
    client: httpx.AsyncClient, url: str, *, params: Mapping[str, Any] | None = None
) -> dict[str, Any] | list[Any]:
    r = await client.get(url, params=params)
    if _is_retryable_status(r.status_code):
        raise TransientHttpError(f"Retryable HTTP status {r.status_code} for {url}")
    r.raise_for_status()
    return r.json()


def build_async_client(cfg: HttpConfig) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        timeout=httpx.Timeout(cfg.timeout_s),
        headers={"User-Agent": cfg.user_agent},
        follow_redirects=True,
    )

