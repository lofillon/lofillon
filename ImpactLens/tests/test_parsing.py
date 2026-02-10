from __future__ import annotations

from impactlens.parsing import parse_datetime, parse_money_usd


def test_parse_money_usd() -> None:
    assert parse_money_usd("200,000,000") == 200000000.0
    assert parse_money_usd("") is None
    assert parse_money_usd(None) is None


def test_parse_datetime_formats() -> None:
    assert parse_datetime("2024-12-20T00:00:00Z") is not None
    assert parse_datetime("2024-12-22 00:00:00.0") is not None
    assert parse_datetime("12/20/2025 12:00:00 AM") is not None

