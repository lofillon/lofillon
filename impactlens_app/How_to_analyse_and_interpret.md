## ImpactLens — Analysis method and limitations (evidence-grounded LLM + OCR)

This document explains:

- What the “analysis” is (what steps the pipeline performs conceptually)
- What the output represents (an evidence-linked “fact set” with provenance)
- What the system guarantees vs. what it does not
- Key limitations (data drift, OCR noise, chunk selection, schema rigidity, and validation boundaries)

---

## Data sources and why “mismatches” happen

This project has been coded around World Bank project records, which are generally trustworthy. Still, it is normal to encounter fields that look “wrong” for analysis—not because they’re intentionally false, but because they are stale, incomplete, or inconsistent across sources.

Common causes:

- Update lag: operational systems and public endpoints refresh on different schedules, so statuses/dates/amounts can lag.
- Lifecycle revisions: project scope, financing, indicators, and even titles can change over time; older documents reflect earlier versions.
- Cross-source drift: the Projects API, project portal pages, and Documents & Reports (WDS) can be generated from different pipelines or versions of the same upstream data.
- Data quality/formatting issues: missing values, duplicates, coding changes (sector/country), and parsing/normalization artifacts can make fields appear incorrect.
- Narrative vs. structured fields: PDFs often contain qualitative claims or targets that later get revised, superseded, or interpreted differently in structured reporting.

Because teams may not keep every system perfectly aligned day-to-day, occasional mismatches between structured APIs, the project webpage, and PDF narratives are expected. The report helps surface these mismatches so you can review them explicitly rather than silently averaging them away.

---

## What ImpactLens produces (the analytical output)

ImpactLens produces an evidence-linked summary of a World Bank project by combining:

- structured project/document metadata (what exists, where it came from), and
- claims extracted from PDFs (what the documents say), each anchored with citations.

The output is a traceable evidence map:

source documents → extracted claims → citations → report

---

## Core analytical object: a “fact set” with provenance

ImpactLens extracts a structured JSON object (schema-enforced) where key fields are paired with citations. Conceptually:

- Scalar facts (names, countries, ratings, etc.)
- Numeric/date facts (values + optional year)
- Narrative text (objective, theory of change, risks/limitations)
- Results indicators (baseline/target/achieved)
- Insufficient evidence signals (explicit gaps)

In the codebase this is modeled via a strict Pydantic schema (`ExtractionOutput` in `src/impactlens/schemas/extraction.py`) to keep the output stable across runs and easy to display in reports/UI.

Example (simplified):

```python
class ExtractionOutput(BaseModel):
    project_id: str
    overview: ProjectOverview | None
    ratings: ProjectRatings | None
    objective: TextWithCitations | None
    theory_of_change: list[TextWithCitations]
    results_indicators: list[ResultsIndicator]
    risks_and_limitations: list[TextWithCitations]
    insufficient_evidence: list[str]
```

---

## What makes it “robust” (analytically)

ImpactLens is built around an anti-hallucination discipline:

- Numeric values and dates are only considered valid if they have evidence (citations).
- Citations are designed to be checkable: the quote should appear in the referenced extracted text chunk.
- If evidence is missing or ambiguous, the pipeline should prefer null / not confirmed and record the gap in `insufficient_evidence`, rather than inventing values.

This turns the output into something you can treat as:

- a structured dataset (indicators, values, years, units), and
- a linked audit trail (citations) for QA and reproducibility.

---

## The pipeline’s actual “analysis” steps (conceptual)

1. Scope
   
   Pick a project (or set of projects) and assemble its relevant public documents.

2. Rank/select documents
   
   Prefer high-signal doc types (implementation reports, results frameworks, PAD/ISR/ICR-style docs). The goal is to reduce noise and focus on documents likely to contain indicators and performance reporting. But you can retrieve all files from a project.

3. Extract evidence chunks from PDFs
   
   Convert PDFs into page-level (or chunk-level) text, using OCR fallback for scanned content. This creates an “evidence corpus” per project.

4. Prioritize informative chunks
   
   Focus on pages likely to contain indicator tables, numeric targets, key dates, and “results section” content (conceptually: information density).

5. LLM-based structured extraction
   
   Use a locally hosted LLM to populate a strict schema using only the provided evidence chunks, producing a JSON fact set.

6. Grounding and consistency checks
   
   Validate that citations are supported by the evidence chunks; flag missing citations or unsupported quotes.

7. Decision-ready report generation
   
   Render the fact set into a readable report for humans, preserving “not confirmed” markings where appropriate.

---

## Grounding by citations: what is validated

ImpactLens uses a grounding rule: a citation is only accepted if the quoted text is found inside the referenced chunk.

The validation logic (`src/impactlens/validation.py`) normalizes whitespace and then performs an inclusion check:

```python
def _norm(s: str) -> str:
    return " ".join((s or "").split())

def _quote_supported(c: Citation, chunks_by_id: dict[str, str]) -> bool:
    txt = chunks_by_id.get(c.chunk_id)
    if not txt:
        return False
    return _norm(c.quote) in _norm(txt)
```

Additionally, the validator enforces “non-null field → at least one citation” for:

- narrative fields (`TextWithCitations`)
- scalar fields (`ScalarFact`)
- numeric/date facts (`NumericFact`)

Downstream report rendering (see `src/impactlens/pipelines/build_report.py`) uses these validation signals plus `insufficient_evidence` to mark items as Not confirmed.

---

## LLM extraction + grounding by citations — strengths vs. limitations (translated and integrated)

### The central challenge

The core challenge is to extract useful facts (numbers, dates, objectives, risks) from a heterogeneous PDF corpus while avoiding the classic failure mode of documentary LLMs: outputs that are “plausible” but not verifiable.

### Strengths (what the system optimizes for)

- Primary strength: verifiability
  - Exact quotes + structured citations + deterministic validation maximize checkability, which is crucial for document-based reporting.
- Structured extraction into a strict schema
  - A stable schema (Pydantic) makes the output predictable for a UI and report generator, and enables automated evaluation (e.g., gold sets).
- Anti-hallucination constraints
  - The rule “every non-null numeric value/date must have at least one citation” turns the LLM from a “generator” into a “pointer/structurer”.
- Hybrid approach: deterministic-first + LLM
  - Heuristics/regex/table-oriented parsing are often more reliable for tables and key-value fields, while the LLM is used for narrative synthesis and to fill gaps.
- Operational resilience
  - Runtime guardrails (thin-corpus gating, timeouts, retries with fewer chunks) reduce total failures and allow partial outputs.

### Limitations (what the system does *not* guarantee)

- Primary limitation: “I can point to the source” is not the same as “it is globally true”
  - Documents can contradict each other, OCR can be wrong, and text can be ambiguous.
  - Coverage depends heavily on corpus quality and which chunks were selected.

---

## Key limitations and failure modes (what to expect)

### 1) Schema rigidity and extraction “fragility”

Why the schema helps:

- forces stable output shape
- simplifies UI/reporting
- enables automated validation and evaluation

What it costs:

- strict schemas can be fragile when model output deviates (requiring repair/coercion)
- some real-world concepts do not fit well (complex indicators, multiple units, footnotes, conditional definitions)

### 2) “Quote present” does not mean “information correct”

The validation guarantees the quote exists in the chunk, but it does not guarantee that:

- the model interpreted the passage correctly,
- the passage maps to the correct field/indicator,
- the passage is not contradicted elsewhere,
- units were handled correctly,
- baseline/target/achieved were not mixed up.

### 3) OCR and parsing noise

- OCR can introduce digit-level errors (e.g., 8 ↔ 3, 0 ↔ O, 1 ↔ I).
- Even when a quote is “supported” (string-match succeeds), the extracted number may still be wrong because OCR produced the wrong characters.
- Conversely, OCR differences can cause false negatives (good citation fails validation because the chunk text differs slightly).

### 4) Chunk selection limits coverage

Grounding can only cite what is present in the evidence corpus and the selected chunk set:

- if the “proof page” is not downloaded, not OCR’d, or not selected, the model cannot cite it
- as a result, fields can be missing even if the evidence exists elsewhere in the document set

### 5) Hybrid “deterministic-first + LLM” trade-offs

Benefits:

- better reliability for tables and regular patterns
- lower LLM cost for fields that can be extracted deterministically

Costs:

- heuristics can break on PDF layout variants → partial extraction
- more maintenance: rules need adjustment as new edge cases appear
- complex semi-structured tables may still require more advanced table extraction tools

### 6) `fast` vs `full` analysis modes

Motivation:

- manage the time/cost/coverage trade-off (in response to smaller projects that contained 1 or 2 files that took the LLM the same amount of time than large projects)
  - `fast`: exploration and thin corpora
  - `full`: deeper coverage when you can afford more passes/context

Limitations:

- `fast` can miss important information
- thin-corpus thresholds are heuristic and can misclassify cases

### 7) Runtime fragility management (thin corpus, timeouts, retries)

Why it exists:

- local LLM calls can time out (hardware/model/context)
- it is usually better to return a partial extraction than to fail completely

Limitations:

- retries with fewer chunks reduce context and can lower recall → more “not confirmed” fields
- hard timeouts can cut off extractions that would have succeeded shortly after
- JSON formatting failures can still occur and require robust parsing/repair

---

## Validation boundaries (what the grounding validation does *not* cover)

### A) Validation is “string matching”, not semantic reasoning

The current normalization compacts whitespace only. It does not robustly handle:

- punctuation differences, ligatures, hyphenation/cross-line breaks
- accent/diacritic variations
- common OCR confusions (0/O, 1/I, 5/S)

This can cause:

- false negatives: correct quote not found due to small text differences
- rare false positives: short quotes matching multiple places (mitigated by requiring specific quotes)

### B) No internal numeric coherence checks

Validation does not enforce domain/business rules such as:

- logical date order (approval < closing)
- plausible magnitudes for financing
- indicator arithmetic or constraints (e.g., achieved ≤ target is not always true, but some checks can still be helpful)

These are possible extensions but are intentionally separate from citation grounding.

### C) No cross-document contradiction resolution

Grounding reduces unsourced hallucinations, but it does not automatically:

- reconcile contradictions across documents (older vs newer)
- pick the “latest” value unless that logic is added upstream
- resolve ambiguous references (e.g., multi-project or multi-component docs)

---

## Why this design is still a good choice

Despite the limits, this grounding approach is:

- cheap (deterministic checks)
- reproducible (same inputs → same validation)
- high-impact against the most critical LLM risk in document analysis: unsourced hallucinations

It also produces actionable downstream signals (e.g., “not confirmed” coverage, citation support rate, and validation issue counts) that can be tracked over time.

---

## Common extensions (if you want to go further)

- OCR-tolerant grounding: fuzzy match, richer normalization, or page-image anchoring for citations
- Numeric coherence checks: lightweight sanity checks and unit consistency warnings
- Cross-document contradiction detection: compare candidate facts across doc types/versions and flag conflicts
- Better table extraction: specialized PDF table extraction to improve indicator recall and accuracy

