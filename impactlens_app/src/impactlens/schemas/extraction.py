from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class Citation(BaseModel):
    doc_id: str
    chunk_id: str
    page: int | None = None
    quote: str = Field(min_length=1)


class TextWithCitations(BaseModel):
    text: str = Field(min_length=1)
    citations: list[Citation] = Field(default_factory=list)


class NumericFact(BaseModel):
    value: float | None = None
    year: int | None = None
    citations: list[Citation] = Field(default_factory=list)


class ResultsIndicator(BaseModel):
    name: str = Field(min_length=1)
    unit: str | None = None
    baseline: NumericFact | None = None
    target: NumericFact | None = None
    achieved: NumericFact | None = None
    citations: list[Citation] = Field(default_factory=list)


class ScalarFact(BaseModel):
    """A single scalar value with evidence."""

    value: str = Field(min_length=1)
    citations: list[Citation] = Field(default_factory=list)


class ProjectOverview(BaseModel):
    """Project-details style overview fields (similar to project portal pages)."""

    project_name: ScalarFact | None = None
    country: ScalarFact | None = None
    region: ScalarFact | None = None
    practice_area: ScalarFact | None = None
    financing_instrument: ScalarFact | None = None

    borrower: ScalarFact | None = None
    implementing_agency: ScalarFact | None = None

    approval_date: ScalarFact | None = None
    effectiveness_date: ScalarFact | None = None
    closing_date: ScalarFact | None = None

    total_commitment_usd: ScalarFact | None = None
    world_bank_financing_usd: ScalarFact | None = None
    borrower_contribution_usd: ScalarFact | None = None
    total_project_cost_usd: ScalarFact | None = None


class ProjectRatings(BaseModel):
    """High-level ratings commonly found in ISR/ICR documents."""

    pdo_rating: ScalarFact | None = None
    ip_rating: ScalarFact | None = None
    overall_risk_rating: ScalarFact | None = None
    outcome_rating: ScalarFact | None = None


class ExtractionOutput(BaseModel):
    project_id: str = Field(min_length=1)

    overview: ProjectOverview | None = None
    ratings: ProjectRatings | None = None

    objective: TextWithCitations | None = None
    theory_of_change: list[TextWithCitations] = Field(default_factory=list)
    results_indicators: list[ResultsIndicator] = Field(default_factory=list)
    risks_and_limitations: list[TextWithCitations] = Field(default_factory=list)

    insufficient_evidence: list[str] = Field(default_factory=list)


class ValidationIssue(BaseModel):
    severity: Literal["error", "warn"]
    issue_type: str
    field_path: str
    message: str

