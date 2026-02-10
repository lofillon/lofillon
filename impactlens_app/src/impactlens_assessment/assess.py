from __future__ import annotations

import asyncio
import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import polars as pl

from impactlens.evaluation.goldset import load_gold_facts
from impactlens.evaluation.metrics import evaluate_facts
from impactlens.llm.ollama import OllamaClient, extract_json_object
from impactlens.schemas.extraction import Citation, ExtractionOutput, ValidationIssue
from impactlens.storage import read_parquet
from impactlens.validation import validate_extraction


def _utc_now_iso() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def _norm(s: str) -> str:
    return " ".join((s or "").split())


def _quote_supported(c: Citation, chunks_by_id: dict[str, str]) -> bool:
    txt = chunks_by_id.get(c.chunk_id)
    if not txt:
        return False
    return _norm(c.quote) in _norm(txt)


def _iter_citations(obj: Any) -> list[dict[str, Any]]:
    """Best-effort traversal over ExtractionOutput-like dicts to collect citations."""
    out: list[dict[str, Any]] = []
    if isinstance(obj, dict):
        if {"doc_id", "chunk_id", "quote"}.issubset(obj.keys()):
            out.append(obj)
        for v in obj.values():
            out.extend(_iter_citations(v))
    elif isinstance(obj, list):
        for it in obj:
            out.extend(_iter_citations(it))
    return out


@dataclass(frozen=True)
class GroundingMetrics:
    n_citations: int
    n_supported: int
    support_rate: float
    n_unique_docs: int
    n_unique_chunks: int


@dataclass(frozen=True)
class CompletenessMetrics:
    n_insufficient_evidence: int
    n_validation_errors: int
    n_validation_warnings: int


@dataclass(frozen=True)
class StabilityMetrics:
    n_runs: int
    scalar_field_stability: float | None
    indicator_name_stability: float | None


@dataclass(frozen=True)
class CoherenceMetrics:
    """Heuristic report-level coherence/quality metrics (0..1 score)."""

    has_objective_section: bool
    has_achievements_section: bool
    has_data_quality_section: bool
    has_key_results_table: bool
    not_confirmed_rate: float
    heuristic_score: float


@dataclass(frozen=True)
class LlmJudgeCoherence:
    """Optional coherence score from an LLM-as-judge pass."""

    model: str
    coherence_score_1_to_5: int
    overclaiming_risk_1_to_5: int
    notes: list[str]


@dataclass(frozen=True)
class ProjectAssessment:
    project_id: str
    latest_extraction_id: str | None
    latest_status: str | None
    created_at: str | None
    ollama_model: str | None
    grounding: GroundingMetrics
    completeness: CompletenessMetrics
    stability: StabilityMetrics
    coherence: CoherenceMetrics
    llm_judge: LlmJudgeCoherence | None


@dataclass(frozen=True)
class ModelAggregate:
    ollama_model: str
    n_projects: int
    avg_gold_accuracy: float | None
    avg_citation_support_rate: float
    avg_validation_errors: float
    avg_insufficient_evidence: float
    avg_coherence_heuristic: float
    avg_scalar_stability: float | None
    avg_indicator_stability: float | None


@dataclass(frozen=True)
class AssessmentSummary:
    dataset_version_id: str
    created_at: str
    n_projects: int
    gold_accuracy: float | None
    projects: list[ProjectAssessment]
    by_model: list[ModelAggregate]


def _load_chunks_by_id(dataset_version_id: str, project_id: str) -> dict[str, str]:
    df = read_parquet(layer="silver", name="document_text_chunks", dataset_version_id=dataset_version_id)
    if "project_id" in df.columns:
        df = df.filter(pl.col("project_id") == project_id)
    out: dict[str, str] = {}
    for r in df.select(["chunk_id", "text"]).iter_rows(named=True):
        out[str(r["chunk_id"])] = str(r.get("text") or "")
    return out


def _load_latest_report_md(dataset_version_id: str, project_id: str) -> str | None:
    """Load latest report markdown if it exists."""
    try:
        df = read_parquet(layer="gold", name="project_reports", dataset_version_id=dataset_version_id)
    except Exception:
        return None
    if "project_id" in df.columns:
        df = df.filter(pl.col("project_id") == project_id)
    if df.height == 0:
        return None
    if "created_at" in df.columns:
        df = df.sort("created_at").tail(1)
    row = df.to_dicts()[0]
    md = row.get("report_md")
    return str(md) if md is not None else None


def _coherence_metrics(report_md: str | None) -> CoherenceMetrics:
    md = report_md or ""
    has_obj = "## Objective" in md
    has_ach = "## What was achieved" in md
    has_dq = "## Data quality" in md
    has_table = "| Indicator |" in md and "| Evidence |" in md
    lines = [ln for ln in md.splitlines() if ln.strip()]
    if not lines:
        return CoherenceMetrics(
            has_objective_section=False,
            has_achievements_section=False,
            has_data_quality_section=False,
            has_key_results_table=False,
            not_confirmed_rate=1.0,
            heuristic_score=0.0,
        )
    n_not = sum(1 for ln in lines if "not confirmed" in ln.lower())
    not_rate = n_not / len(lines) if lines else 1.0
    # Simple heuristic: reward structure, penalize excessive "not confirmed"
    score = 0.0
    score += 0.2 if has_obj else 0.0
    score += 0.3 if has_ach else 0.0
    score += 0.2 if has_dq else 0.0
    score += 0.1 if has_table else 0.0
    score += max(0.0, 0.2 * (1.0 - min(1.0, not_rate / 0.25)))  # full credit if <=25% not confirmed
    score = max(0.0, min(1.0, score))
    return CoherenceMetrics(
        has_objective_section=has_obj,
        has_achievements_section=has_ach,
        has_data_quality_section=has_dq,
        has_key_results_table=has_table,
        not_confirmed_rate=float(not_rate),
        heuristic_score=float(score),
    )


async def _judge_report_with_ollama(*, report_md: str, model: str | None = None) -> LlmJudgeCoherence:
    client = OllamaClient(model=model)
    try:
        system = """You are grading the coherence and reliability of an analytical report.

Return a single JSON object with:
- coherence_score_1_to_5: integer 1..5 (5=very coherent, well-structured, clear)
- overclaiming_risk_1_to_5: integer 1..5 (5=high risk of overclaiming beyond evidence)
- notes: array of 3-7 short strings (actionable observations)

Be strict: if the report contains many "not confirmed" or missing sections, coherence should be low.
"""
        user = json.dumps({"report_markdown": report_md[:50_000]}, ensure_ascii=False)
        resp = await client.chat_json(system=system, user=user, temperature=0.0)
        obj = extract_json_object(resp.content)
        return LlmJudgeCoherence(
            model=resp.model,
            coherence_score_1_to_5=int(obj.get("coherence_score_1_to_5") or 0),
            overclaiming_risk_1_to_5=int(obj.get("overclaiming_risk_1_to_5") or 0),
            notes=[str(x) for x in (obj.get("notes") or []) if isinstance(x, (str, int, float))],
        )
    finally:
        await client.aclose()


def _run_judge_sync(*, report_md: str, model: str | None) -> LlmJudgeCoherence | None:
    if not report_md.strip():
        return None
    try:
        return asyncio.run(_judge_report_with_ollama(report_md=report_md, model=model))
    except Exception:
        return None


def _grounding_metrics(extraction_obj: dict[str, Any], chunks_by_id: dict[str, str]) -> GroundingMetrics:
    citations_raw = _iter_citations(extraction_obj)
    citations: list[Citation] = []
    for c in citations_raw:
        try:
            citations.append(Citation.model_validate(c))
        except Exception:
            continue
    supported = sum(1 for c in citations if _quote_supported(c, chunks_by_id))
    n = len(citations)
    unique_docs = {c.doc_id for c in citations if c.doc_id}
    unique_chunks = {c.chunk_id for c in citations if c.chunk_id}
    return GroundingMetrics(
        n_citations=n,
        n_supported=supported,
        support_rate=(supported / n) if n else 0.0,
        n_unique_docs=len(unique_docs),
        n_unique_chunks=len(unique_chunks),
    )


def _completeness_metrics(issues: list[ValidationIssue], extraction_obj: dict[str, Any]) -> CompletenessMetrics:
    errs = sum(1 for i in issues if i.severity == "error")
    warns = sum(1 for i in issues if i.severity == "warn")
    ie = extraction_obj.get("insufficient_evidence")
    n_ie = len(ie) if isinstance(ie, list) else 0
    return CompletenessMetrics(n_insufficient_evidence=n_ie, n_validation_errors=errs, n_validation_warnings=warns)


def _scalar_signature(extraction_obj: dict[str, Any]) -> dict[str, Any]:
    """Pick a stable subset of scalar fields for stability comparisons."""
    ov = extraction_obj.get("overview") if isinstance(extraction_obj, dict) else None
    rt = extraction_obj.get("ratings") if isinstance(extraction_obj, dict) else None

    def _v(x: Any) -> Any:
        if isinstance(x, dict) and "value" in x:
            return str(x.get("value") or "").strip() or None
        return None

    out = {}
    if isinstance(ov, dict):
        for k in [
            "project_name",
            "country",
            "region",
            "practice_area",
            "financing_instrument",
            "borrower",
            "implementing_agency",
            "approval_date",
            "effectiveness_date",
            "closing_date",
            "total_commitment_usd",
        ]:
            out[f"overview.{k}"] = _v(ov.get(k))
    if isinstance(rt, dict):
        for k in ["pdo_rating", "ip_rating", "overall_risk_rating", "outcome_rating"]:
            out[f"ratings.{k}"] = _v(rt.get(k))
    return out


def _indicator_name_set(extraction_obj: dict[str, Any]) -> set[str]:
    inds = extraction_obj.get("results_indicators") if isinstance(extraction_obj, dict) else None
    if not isinstance(inds, list):
        return set()
    out: set[str] = set()
    for it in inds:
        if not isinstance(it, dict):
            continue
        name = str(it.get("name") or "").strip()
        unit = str(it.get("unit") or "").strip()
        key = f"{name} ({unit})" if unit else name
        if key:
            out.add(key)
    return out


def _jaccard(a: set[str], b: set[str]) -> float | None:
    if not a and not b:
        return None
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else None


def _stability_metrics(extractions: list[dict[str, Any]]) -> StabilityMetrics:
    # Expect extractions sorted newest->oldest; compare consecutive runs.
    n_runs = len(extractions)
    if n_runs < 2:
        return StabilityMetrics(n_runs=n_runs, scalar_field_stability=None, indicator_name_stability=None)

    scalar_scores: list[float] = []
    ind_scores: list[float] = []
    for a, b in zip(extractions, extractions[1:], strict=False):
        sa = _scalar_signature(a)
        sb = _scalar_signature(b)
        keys = set(sa.keys()) | set(sb.keys())
        if keys:
            same = sum(1 for k in keys if (sa.get(k) == sb.get(k)) and (sa.get(k) is not None))
            denom = sum(1 for k in keys if (sa.get(k) is not None or sb.get(k) is not None))
            scalar_scores.append(same / denom if denom else 0.0)
        ind_scores.append(_jaccard(_indicator_name_set(a), _indicator_name_set(b)) or 0.0)

    return StabilityMetrics(
        n_runs=n_runs,
        scalar_field_stability=sum(scalar_scores) / len(scalar_scores) if scalar_scores else None,
        indicator_name_stability=sum(ind_scores) / len(ind_scores) if ind_scores else None,
    )


def run_assessment(
    *,
    dataset_version_id: str,
    gold_facts_path: Path | None = None,
    project_ids: list[str] | None = None,
    stability_last_k: int = 3,
    judge_with_ollama: bool = False,
    judge_model: str | None = None,
) -> AssessmentSummary:
    """Run assessment over the latest extractions.

    - **Accuracy**: optional gold facts exact-match (existing evaluation logic)
    - **Coherence**: approximated via structural + internal-consistency proxies (captured in completeness + stability)
    - **Robustness**: citation grounding support rate + stability across multiple runs when available
    """
    ex_df = read_parquet(layer="gold", name="extractions", dataset_version_id=dataset_version_id)
    if project_ids:
        ex_df = ex_df.filter(pl.col("project_id").is_in(project_ids))

    # Keep all runs; we'll need multiple per project for stability.
    ex_df = ex_df.sort("created_at", descending=True)

    by_project: dict[str, list[dict[str, Any]]] = {}
    for row in ex_df.iter_rows(named=True):
        pid = str(row.get("project_id") or "")
        by_project.setdefault(pid, []).append(row)

    projects: list[ProjectAssessment] = []
    for pid, rows in by_project.items():
        latest = rows[0]
        latest_obj = json.loads(latest["output_json"])
        chunks_by_id = _load_chunks_by_id(dataset_version_id, pid)
        report_md = _load_latest_report_md(dataset_version_id, pid) or ""

        # Recompute issues deterministically to get coherence/robustness signals even if stored issues are missing.
        issues: list[ValidationIssue] = []
        try:
            ex = ExtractionOutput.model_validate(latest_obj)
            issues = validate_extraction(ex, chunks_by_id)
        except Exception:
            issues = []

        grounding = _grounding_metrics(latest_obj, chunks_by_id)
        completeness = _completeness_metrics(issues, latest_obj)
        coherence = _coherence_metrics(report_md)
        llm_judge = _run_judge_sync(report_md=report_md, model=judge_model) if judge_with_ollama else None

        # Stability: compare multiple recent runs (if they exist)
        recent = rows[: max(1, int(stability_last_k))]
        recent_objs: list[dict[str, Any]] = []
        for r in recent:
            try:
                recent_objs.append(json.loads(r["output_json"]))
            except Exception:
                continue
        stability = _stability_metrics(recent_objs)

        projects.append(
            ProjectAssessment(
                project_id=pid,
                latest_extraction_id=str(latest.get("extraction_id") or "") or None,
                latest_status=str(latest.get("status") or "") or None,
                created_at=str(latest.get("created_at") or "") or None,
                ollama_model=str(latest.get("ollama_model") or "") or None,
                grounding=grounding,
                completeness=completeness,
                stability=stability,
                coherence=coherence,
                llm_judge=llm_judge,
            )
        )

    projects = sorted(projects, key=lambda p: p.project_id)

    gold_accuracy: float | None = None
    if gold_facts_path is not None:
        gold = load_gold_facts(gold_facts_path)
        # latest per project_id dict
        latest_by_project = {p.project_id: json.loads(by_project[p.project_id][0]["output_json"]) for p in projects}
        gold_accuracy = evaluate_facts(latest_by_project, gold).accuracy

    # Aggregate by model (based on latest run per project)
    by_model: dict[str, list[ProjectAssessment]] = {}
    for p in projects:
        key = p.ollama_model or "unknown"
        by_model.setdefault(key, []).append(p)

    model_aggs: list[ModelAggregate] = []
    for model_name, plist in sorted(by_model.items(), key=lambda kv: kv[0]):
        n = len(plist)
        avg_support = sum(pp.grounding.support_rate for pp in plist) / n if n else 0.0
        avg_errs = sum(pp.completeness.n_validation_errors for pp in plist) / n if n else 0.0
        avg_ie = sum(pp.completeness.n_insufficient_evidence for pp in plist) / n if n else 0.0
        avg_coh = sum(pp.coherence.heuristic_score for pp in plist) / n if n else 0.0
        scalars = [pp.stability.scalar_field_stability for pp in plist if pp.stability.scalar_field_stability is not None]
        inds = [pp.stability.indicator_name_stability for pp in plist if pp.stability.indicator_name_stability is not None]
        model_aggs.append(
            ModelAggregate(
                ollama_model=model_name,
                n_projects=n,
                avg_gold_accuracy=None,  # placeholder (gold is fact-level across projects, not model-stratified yet)
                avg_citation_support_rate=float(avg_support),
                avg_validation_errors=float(avg_errs),
                avg_insufficient_evidence=float(avg_ie),
                avg_coherence_heuristic=float(avg_coh),
                avg_scalar_stability=(sum(scalars) / len(scalars)) if scalars else None,
                avg_indicator_stability=(sum(inds) / len(inds)) if inds else None,
            )
        )

    return AssessmentSummary(
        dataset_version_id=dataset_version_id,
        created_at=_utc_now_iso(),
        n_projects=len(projects),
        gold_accuracy=gold_accuracy,
        projects=projects,
        by_model=model_aggs,
    )


def render_assessment_report_md(summary: AssessmentSummary) -> str:
    lines: list[str] = []
    lines.append(f"# ImpactLens Assessment Report — dataset_version={summary.dataset_version_id}")
    lines.append("")
    lines.append(f"- Generated at: {summary.created_at}")
    lines.append(f"- Projects assessed: **{summary.n_projects}**")
    if summary.gold_accuracy is not None:
        lines.append(f"- Gold fact accuracy (exact match): **{summary.gold_accuracy:.3f}**")
    lines.append("")

    if summary.by_model:
        lines.append("## Benchmark by LLM model (latest run per project)")
        lines.append("| Model | Projects | Citation support rate | Avg validation errors | Avg insufficient evidence | Avg coherence (heuristic) |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        for m in summary.by_model:
            lines.append(
                "| "
                + " | ".join(
                    [
                        m.ollama_model,
                        str(m.n_projects),
                        f"{m.avg_citation_support_rate:.3f}",
                        f"{m.avg_validation_errors:.2f}",
                        f"{m.avg_insufficient_evidence:.2f}",
                        f"{m.avg_coherence_heuristic:.3f}",
                    ]
                )
                + " |"
            )
        lines.append("")

    lines.append("## Quantitative benchmark (per project)")
    lines.append("| Project | Model | Citations | Citation support rate | Unique docs | Insufficient evidence | Validation errors | Coherence (heuristic) | Stability (scalars) | Stability (indicators) |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for p in summary.projects:
        g = p.grounding
        c = p.completeness
        coh = p.coherence
        s = p.stability
        lines.append(
            "| "
            + " | ".join(
                [
                    p.project_id,
                    (p.ollama_model or "unknown"),
                    str(g.n_citations),
                    f"{g.support_rate:.3f}",
                    str(g.n_unique_docs),
                    str(c.n_insufficient_evidence),
                    str(c.n_validation_errors),
                    f"{coh.heuristic_score:.3f}",
                    (f"{s.scalar_field_stability:.3f}" if s.scalar_field_stability is not None else "—"),
                    (f"{s.indicator_name_stability:.3f}" if s.indicator_name_stability is not None else "—"),
                ]
            )
            + " |"
        )
    lines.append("")

    lines.append("## Interpretation guide")
    lines.append("- **Gold fact accuracy**: exact-match vs your curated `gold_facts.jsonl` (strict; best for regression).")
    lines.append("- **Citation support rate**: fraction of citations whose quote is found in the referenced chunk text (should be near 1.0).")
    lines.append("- **Insufficient evidence**: how many fields the extraction explicitly flagged as missing evidence.")
    lines.append("- **Validation errors**: grounding/format issues detected by ImpactLens validators (higher = less reliable).")
    lines.append("- **Coherence (heuristic)**: report-structure score (sections present) penalized by high 'not confirmed' rate.")
    lines.append("- **Stability**: agreement across multiple recent runs (higher = more robust).")
    lines.append("")
    return "\n".join(lines)


def write_assessment_outputs(*, out_dir: Path, summary: AssessmentSummary) -> dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "assessment_results.json"
    md_path = out_dir / "assessment_report.md"

    payload = asdict(summary)
    # dataclasses -> nested dicts; keep it readable
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(render_assessment_report_md(summary), encoding="utf-8")
    return {"json": str(json_path), "md": str(md_path)}

