from __future__ import annotations

import re


def text_length(text: str) -> int:
    return len(text or "")


def digit_ratio(text: str) -> float:
    if not text:
        return 0.0
    digits = sum(c.isdigit() for c in text)
    return digits / max(1, len(text))


def non_alnum_ratio(text: str) -> float:
    if not text:
        return 0.0
    non_alnum = sum(not c.isalnum() and not c.isspace() for c in text)
    return non_alnum / max(1, len(text))


def alpha_ratio(text: str) -> float:
    if not text:
        return 0.0
    alpha = sum(c.isalpha() for c in text)
    return alpha / max(1, len(text))


SECTION_HINTS = [
    re.compile(r"\bresults\b", re.IGNORECASE),
    re.compile(r"\bindicator\b", re.IGNORECASE),
    re.compile(r"\bbaseline\b", re.IGNORECASE),
    re.compile(r"\btarget\b", re.IGNORECASE),
]


def section_hint_score(text: str) -> int:
    if not text:
        return 0
    return sum(1 for r in SECTION_HINTS if r.search(text))

