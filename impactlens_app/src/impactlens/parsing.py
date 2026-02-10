from __future__ import annotations

from datetime import datetime


def parse_money_usd(value: str | int | float | None) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    s = str(value).strip()
    if s == "" or s.lower() in {"na", "n/a", "null", "none"}:
        return None
    # Example: "200,000,000"
    s = s.replace(",", "")
    try:
        return float(s)
    except ValueError:
        return None


def parse_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    s = value.strip()

    # Examples:
    # - "2024-12-20T00:00:00Z"
    # - "2024-12-22 00:00:00.0"
    # - "12/20/2025 12:00:00 AM"
    for fmt in (
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%d %H:%M:%S.%f",
        "%m/%d/%Y %I:%M:%S %p",
        "%Y-%m-%d",
    ):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    return None

