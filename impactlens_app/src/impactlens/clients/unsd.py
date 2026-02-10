from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import httpx

from impactlens.clients.http import HttpConfig, build_async_client, get_json


@dataclass(frozen=True)
class UnsdsdgClient:
    """Client for the UNSD SDG API (public)."""

    http: httpx.AsyncClient
    base_url: str = "https://unstats.un.org/SDGAPI/v1/sdg"

    @classmethod
    def from_config(cls, cfg: HttpConfig | None = None) -> "UnsdsdgClient":
        return cls(http=build_async_client(cfg or HttpConfig()))

    async def aclose(self) -> None:
        await self.http.aclose()

    async def list_goals(self) -> list[dict[str, Any]]:
        url = f"{self.base_url}/Goal/List"
        data = await get_json(self.http, url)
        if not isinstance(data, list):
            raise TypeError("Unexpected goals response type")
        return [d for d in data if isinstance(d, dict)]

    async def list_targets(self, goal_code: str) -> list[dict[str, Any]]:
        url = f"{self.base_url}/Target/List"
        params: Mapping[str, Any] = {"goal": goal_code}
        data = await get_json(self.http, url, params=params)
        if not isinstance(data, list):
            raise TypeError("Unexpected targets response type")
        return [d for d in data if isinstance(d, dict)]

    async def list_indicators(self, goal_code: str | None = None) -> list[dict[str, Any]]:
        url = f"{self.base_url}/Indicator/List"
        params: dict[str, Any] = {}
        if goal_code is not None:
            params["goal"] = goal_code
        data = await get_json(self.http, url, params=params or None)
        if not isinstance(data, list):
            raise TypeError("Unexpected indicators response type")
        return [d for d in data if isinstance(d, dict)]

