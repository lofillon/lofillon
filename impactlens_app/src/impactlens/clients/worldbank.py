from __future__ import annotations

from dataclasses import dataclass
from typing import Any, AsyncIterator, Mapping

import httpx

from impactlens.clients.http import HttpConfig, build_async_client, get_json


@dataclass(frozen=True)
class WorldBankClient:
    """Client for World Bank public APIs used by ImpactLens.

    Sources:
    - Projects Search API v2: https://search.worldbank.org/api/v2/projects
    - Documents & Reports (WDS) API v3: https://search.worldbank.org/api/v3/wds
    - Indicators API v2: https://api.worldbank.org/v2/...
    """

    http: httpx.AsyncClient

    @classmethod
    def from_config(cls, cfg: HttpConfig | None = None) -> "WorldBankClient":
        return cls(http=build_async_client(cfg or HttpConfig()))

    async def aclose(self) -> None:
        await self.http.aclose()

    async def iter_projects(
        self,
        *,
        rows: int = 100,
        page_start: int = 1,
        max_pages: int | None = None,
        extra_params: Mapping[str, Any] | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """Yield raw project payloads.

        Response shape (example):
        {
          "rows": 1,
          "page": "1",
          "total": "22729",
          "projects": { "P505244": { ... } }
        }
        """
        url = "https://search.worldbank.org/api/v2/projects"
        page = page_start
        seen_pages = 0

        while True:
            params: dict[str, Any] = {"format": "json", "rows": rows, "page": page}
            if extra_params:
                params.update(extra_params)
            data = await get_json(self.http, url, params=params)
            projects = (data or {}).get("projects") or {}
            if not isinstance(projects, dict) or len(projects) == 0:
                return

            for _, payload in projects.items():
                if isinstance(payload, dict):
                    yield payload

            page += 1
            seen_pages += 1
            if max_pages is not None and seen_pages >= max_pages:
                return

    async def search_wds(
        self,
        *,
        qterm: str | None = None,
        rows: int = 100,
        os: int = 0,
        fl: str | None = None,
        extra_params: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Search World Bank Documents & Reports (WDS) and return the raw response."""
        url = "https://search.worldbank.org/api/v3/wds"
        params: dict[str, Any] = {"format": "json", "rows": rows, "os": os}
        if qterm:
            params["qterm"] = qterm
        if fl:
            params["fl"] = fl
        if extra_params:
            params.update(extra_params)
        data = await get_json(self.http, url, params=params)
        if not isinstance(data, dict):
            raise TypeError("Unexpected WDS response type")
        return data

    async def iter_wds_documents(
        self,
        *,
        qterm: str | None = None,
        rows: int = 100,
        max_pages: int | None = None,
        fl: str | None = None,
        extra_params: Mapping[str, Any] | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """Yield raw WDS document payloads with offset pagination (os)."""
        os = 0
        seen_pages = 0
        while True:
            data = await self.search_wds(qterm=qterm, rows=rows, os=os, fl=fl, extra_params=extra_params)
            docs = (data or {}).get("documents") or {}
            if not isinstance(docs, dict) or len(docs) == 0:
                return
            for _, payload in docs.items():
                if isinstance(payload, dict):
                    yield payload

            os += rows
            seen_pages += 1
            if max_pages is not None and seen_pages >= max_pages:
                return

    async def fetch_indicator_series(
        self,
        *,
        country: str,
        indicator: str,
        per_page: int = 1000,
        page: int = 1,
    ) -> list[Any]:
        """Fetch one page of an indicator series.

        World Bank API shape is typically:
        [
          { "page": 1, "pages": 1, ... },
          [ { "indicator": {...}, "country": {...}, "date": "2022", "value": 123 }, ... ]
        ]
        """
        url = f"https://api.worldbank.org/v2/country/{country}/indicator/{indicator}"
        params = {"format": "json", "per_page": per_page, "page": page}
        data = await get_json(self.http, url, params=params)
        if not isinstance(data, list):
            raise TypeError("Unexpected indicator response type")
        return data

    async def iter_indicator_observations(
        self,
        *,
        country: str,
        indicator: str,
        per_page: int = 1000,
        max_pages: int | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """Yield observations from the Indicators API (all pages)."""
        page = 1
        seen_pages = 0
        while True:
            data = await self.fetch_indicator_series(
                country=country, indicator=indicator, per_page=per_page, page=page
            )
            if len(data) < 2 or not isinstance(data[1], list) or len(data[1]) == 0:
                return
            for obs in data[1]:
                if isinstance(obs, dict):
                    yield obs
            page += 1
            seen_pages += 1
            if max_pages is not None and seen_pages >= max_pages:
                return

