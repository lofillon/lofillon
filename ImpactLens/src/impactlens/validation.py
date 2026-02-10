from __future__ import annotations

from typing import Any

from impactlens.schemas.extraction import (
    Citation,
    ExtractionOutput,
    NumericFact,
    ResultsIndicator,
    ScalarFact,
    TextWithCitations,
    ValidationIssue,
)


def _norm(s: str) -> str:
    return " ".join((s or "").split())


def _quote_supported(c: Citation, chunks_by_id: dict[str, str]) -> bool:
    txt = chunks_by_id.get(c.chunk_id)
    if not txt:
        return False
    return _norm(c.quote) in _norm(txt)


def _validate_text(field_path: str, obj: TextWithCitations | None, chunks_by_id: dict[str, str]) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    if obj is None:
        return issues
    if len(obj.citations) == 0:
        issues.append(
            ValidationIssue(
                severity="error",
                issue_type="missing_citation",
                field_path=field_path,
                message="Non-null text field must have at least one citation.",
            )
        )
        return issues
    for i, c in enumerate(obj.citations):
        if not _quote_supported(c, chunks_by_id):
            issues.append(
                ValidationIssue(
                    severity="error",
                    issue_type="citation_not_supported",
                    field_path=f"{field_path}.citations[{i}]",
                    message="Citation quote not found in referenced chunk text.",
                )
            )
    return issues


def _validate_scalar(field_path: str, obj: ScalarFact | None, chunks_by_id: dict[str, str]) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    if obj is None:
        return issues
    if not (obj.value or "").strip():
        return issues
    if len(obj.citations) == 0:
        issues.append(
            ValidationIssue(
                severity="error",
                issue_type="missing_citation",
                field_path=field_path,
                message="Non-null scalar field must have at least one citation.",
            )
        )
        return issues
    for i, c in enumerate(obj.citations):
        if not _quote_supported(c, chunks_by_id):
            issues.append(
                ValidationIssue(
                    severity="error",
                    issue_type="citation_not_supported",
                    field_path=f"{field_path}.citations[{i}]",
                    message="Citation quote not found in referenced chunk text.",
                )
            )
    return issues


def _validate_numeric(field_path: str, obj: NumericFact | None, chunks_by_id: dict[str, str]) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    if obj is None:
        return issues
    if obj.value is None and obj.year is None:
        return issues
    if len(obj.citations) == 0:
        issues.append(
            ValidationIssue(
                severity="error",
                issue_type="unsupported_numeric_or_date",
                field_path=field_path,
                message="Numeric/date fact must have at least one citation.",
            )
        )
        return issues
    for i, c in enumerate(obj.citations):
        if not _quote_supported(c, chunks_by_id):
            issues.append(
                ValidationIssue(
                    severity="error",
                    issue_type="citation_not_supported",
                    field_path=f"{field_path}.citations[{i}]",
                    message="Citation quote not found in referenced chunk text.",
                )
            )
    return issues


def _validate_indicator(
    idx: int, ind: ResultsIndicator, chunks_by_id: dict[str, str]
) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    base = f"results_indicators[{idx}]"
    if ind.citations:
        for i, c in enumerate(ind.citations):
            if not _quote_supported(c, chunks_by_id):
                issues.append(
                    ValidationIssue(
                        severity="error",
                        issue_type="citation_not_supported",
                        field_path=f"{base}.citations[{i}]",
                        message="Citation quote not found in referenced chunk text.",
                    )
                )
    issues.extend(_validate_numeric(f"{base}.baseline", ind.baseline, chunks_by_id))
    issues.extend(_validate_numeric(f"{base}.target", ind.target, chunks_by_id))
    issues.extend(_validate_numeric(f"{base}.achieved", ind.achieved, chunks_by_id))
    return issues


def validate_extraction(output: ExtractionOutput, chunks_by_id: dict[str, str]) -> list[ValidationIssue]:
    """Validate anti-hallucination constraints and citation grounding."""
    issues: list[ValidationIssue] = []

    if output.overview:
        ov = output.overview
        issues.extend(_validate_scalar("overview.project_name", ov.project_name, chunks_by_id))
        issues.extend(_validate_scalar("overview.country", ov.country, chunks_by_id))
        issues.extend(_validate_scalar("overview.region", ov.region, chunks_by_id))
        issues.extend(_validate_scalar("overview.practice_area", ov.practice_area, chunks_by_id))
        issues.extend(_validate_scalar("overview.financing_instrument", ov.financing_instrument, chunks_by_id))
        issues.extend(_validate_scalar("overview.borrower", ov.borrower, chunks_by_id))
        issues.extend(_validate_scalar("overview.implementing_agency", ov.implementing_agency, chunks_by_id))
        issues.extend(_validate_scalar("overview.approval_date", ov.approval_date, chunks_by_id))
        issues.extend(_validate_scalar("overview.effectiveness_date", ov.effectiveness_date, chunks_by_id))
        issues.extend(_validate_scalar("overview.closing_date", ov.closing_date, chunks_by_id))
        issues.extend(_validate_scalar("overview.total_commitment_usd", ov.total_commitment_usd, chunks_by_id))

    if output.ratings:
        rt = output.ratings
        issues.extend(_validate_scalar("ratings.pdo_rating", rt.pdo_rating, chunks_by_id))
        issues.extend(_validate_scalar("ratings.ip_rating", rt.ip_rating, chunks_by_id))
        issues.extend(_validate_scalar("ratings.overall_risk_rating", rt.overall_risk_rating, chunks_by_id))
        issues.extend(_validate_scalar("ratings.outcome_rating", rt.outcome_rating, chunks_by_id))

    issues.extend(_validate_text("objective", output.objective, chunks_by_id))

    for i, t in enumerate(output.theory_of_change):
        issues.extend(_validate_text(f"theory_of_change[{i}]", t, chunks_by_id))

    for i, r in enumerate(output.risks_and_limitations):
        issues.extend(_validate_text(f"risks_and_limitations[{i}]", r, chunks_by_id))

    for i, ind in enumerate(output.results_indicators):
        issues.extend(_validate_indicator(i, ind, chunks_by_id))

    return issues


def has_errors(issues: list[ValidationIssue]) -> bool:
    return any(i.severity == "error" for i in issues)

