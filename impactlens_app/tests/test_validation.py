from __future__ import annotations

from impactlens.schemas.extraction import Citation, ExtractionOutput, NumericFact, ResultsIndicator
from impactlens.validation import has_errors, validate_extraction


def test_numeric_requires_citation() -> None:
    out = ExtractionOutput(
        project_id="P1",
        results_indicators=[
            ResultsIndicator(
                name="Indicator A",
                baseline=NumericFact(value=1.0, year=2020, citations=[]),
            )
        ],
    )
    issues = validate_extraction(out, chunks_by_id={})
    assert has_errors(issues)


def test_citation_quote_must_be_supported() -> None:
    out = ExtractionOutput(
        project_id="P1",
        results_indicators=[
            ResultsIndicator(
                name="Indicator A",
                baseline=NumericFact(
                    value=1.0,
                    year=2020,
                    citations=[
                        Citation(doc_id="D1", chunk_id="c1", page=1, quote="not present"),
                    ],
                ),
            )
        ],
    )
    issues = validate_extraction(out, chunks_by_id={"c1": "some other text"})
    assert has_errors(issues)

