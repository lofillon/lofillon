from __future__ import annotations

import re


_ICR_PATTERNS = [
    re.compile(r"\bimplementation completion and results report\b", re.IGNORECASE),
    re.compile(r"\bicr\b", re.IGNORECASE),
    re.compile(r"\bimplementation completion report\b", re.IGNORECASE),
]
_PAD_PATTERNS = [
    re.compile(r"\bproject appraisal document\b", re.IGNORECASE),
    re.compile(r"\bprogram appraisal document\b", re.IGNORECASE),
    re.compile(r"\bpad\b", re.IGNORECASE),
]
_ISR_PATTERNS = [
    re.compile(r"\bimplementation status and results report\b", re.IGNORECASE),
    re.compile(r"\bisr\b", re.IGNORECASE),
]
_RESULTS_FRAMEWORK_PATTERNS = [
    re.compile(r"\bresults framework\b", re.IGNORECASE),
    re.compile(r"\bresults framework and monitoring\b", re.IGNORECASE),
    re.compile(r"\blogframe\b", re.IGNORECASE),
]
_RESTRUCTURING_PATTERNS = [
    re.compile(r"\brestructuring\b", re.IGNORECASE),
]

_BAD_MAJOR_TYPES = {
    "Publications & Research",
}
_BAD_TYPES = {
    "Working Paper",
}


def normalize_doc_class(title: str | None, docty: str | None, majdocty: str | None) -> str:
    t = (title or "") + " " + (docty or "") + " " + (majdocty or "")
    if any(p.search(t) for p in _ICR_PATTERNS):
        return "ICR"
    if any(p.search(t) for p in _PAD_PATTERNS):
        return "PAD"
    if any(p.search(t) for p in _ISR_PATTERNS):
        return "ISR"
    if any(p.search(t) for p in _RESULTS_FRAMEWORK_PATTERNS):
        return "RESULTS_FRAMEWORK"
    if any(p.search(t) for p in _RESTRUCTURING_PATTERNS):
        return "RESTRUCTURING"
    return "OTHER"


def score_document(
    *,
    doc_class: str,
    has_pdf: bool,
    is_public: bool,
    title: str | None,
    docty: str | None,
    majdocty: str | None,
) -> float:
    """Heuristic score to prioritize "project documents" (ICR/PAD/ISR) with PDFs."""
    score = 0.0
    if has_pdf:
        score += 5.0
    if is_public:
        score += 3.0

    if doc_class == "ICR":
        score += 20.0
    elif doc_class == "PAD":
        score += 15.0
    elif doc_class == "ISR":
        score += 12.0
    elif doc_class == "RESULTS_FRAMEWORK":
        score += 14.0
    elif doc_class == "RESTRUCTURING":
        score += 10.0
    else:
        score += 1.0

    if (majdocty or "") in _BAD_MAJOR_TYPES:
        score -= 10.0
    if (docty or "") in _BAD_TYPES:
        score -= 10.0

    # Small bonus for likely "project-related" titles.
    t = (title or "").lower()
    if "project" in t:
        score += 1.0
    if "report" in t:
        score += 0.5

    return score

