from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import httpx

from impactlens.settings import settings


@dataclass(frozen=True)
class OllamaResponse:
    model: str
    content: str
    raw: dict[str, Any]


class OllamaClient:
    def __init__(self, base_url: str | None = None, model: str | None = None) -> None:
        self.base_url = (base_url or settings.ollama_base_url).rstrip("/")
        self.model = model or settings.ollama_model
        # LLM generation can take significantly longer than typical HTTP requests.
        # Use a dedicated (longer) timeout so we don't fail with ReadTimeout on bigger prompts.
        self._http = httpx.AsyncClient(timeout=httpx.Timeout(settings.ollama_timeout_s))

    async def aclose(self) -> None:
        await self._http.aclose()

    async def chat_json(self, *, system: str, user: str, temperature: float = 0.0) -> OllamaResponse:
        """Call Ollama chat API requesting a JSON response."""
        url = f"{self.base_url}/api/chat"
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
            "format": "json",
            "options": {"temperature": temperature},
        }
        r = await self._http.post(url, json=payload)
        r.raise_for_status()
        data = r.json()
        content = (data.get("message") or {}).get("content") or ""
        return OllamaResponse(model=data.get("model") or self.model, content=str(content), raw=data)


def extract_json_object(text: str) -> dict[str, Any]:
    """Parse a JSON object from text (robust to extra whitespace)."""
    s = text.strip()
    try:
        obj = json.loads(s)
        if not isinstance(obj, dict):
            raise TypeError("Expected a JSON object")
        return obj
    except Exception:  # noqa: BLE001
        # Best-effort: find the first {...} block
        start = s.find("{")
        end = s.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise
        obj = json.loads(s[start : end + 1])
        if not isinstance(obj, dict):
            raise TypeError("Expected a JSON object")
        return obj

