from __future__ import annotations

import json
from dataclasses import dataclass

import polars as pl

from impactlens.schemas.extraction import Citation, ExtractionOutput, ValidationIssue
from impactlens.storage import append_parquet, read_parquet
from impactlens.utils import utc_now


def _fmt_citations(citations: list[Citation] | None) -> str:
    if not citations:
        return ""
    parts = []
    for c in citations[:3]:
        page = f"p{c.page}" if c.page else ""
        parts.append(f"[{c.doc_id} {page}] “{c.quote}”")
    return "  \n".join(parts)


def _fmt_citations_table_links(citations: list[Citation] | None, doc_pdf_urls: dict[str, str] | None) -> str:
    """Format citations for markdown tables with clickable doc links.

    Uses `<br>` to keep multiple citations within a single table cell.
    """
    if not citations:
        return ""
    doc_pdf_urls = doc_pdf_urls or {}
    parts: list[str] = []
    for c in citations[:2]:
        doc_id = c.doc_id
        page = c.page
        label = f"{doc_id}" + (f" p{page}" if page else "")
        url = doc_pdf_urls.get(doc_id)
        if url:
            # Jump-to-page works in many PDF viewers.
            if page and "#" not in url:
                url = f"{url}#page={page}"
            link = f"[{label}]({url})"
        else:
            link = f"`{label}`"
        # Intentionally omit verbatim quotes in tables (keeps the report concise and avoids
        # embedding long source text in the evidence column).
        parts.append(link)
    return "<br>".join(parts)


def _normalize_field_path(path: str) -> str:
    """Normalize validation issue paths to the owning field.

    Example:
    - `objective.citations[0]` -> `objective`
    - `results_indicators[3].baseline.citations[0]` -> `results_indicators[3].baseline`
    """
    p = (path or "").strip()
    if not p:
        return ""
    if ".citations[" in p:
        return p.split(".citations[", 1)[0]
    return p


def _build_unconfirmed_paths(ex: ExtractionOutput, issues: list[ValidationIssue] | None) -> set[str]:
    """Return field paths that should be treated as not confirmed."""
    def _has_scalar(sf) -> bool:
        return bool(sf and getattr(sf, "value", None))

    def _path_is_missing(path: str) -> bool:
        """Best-effort: only treat paths as unconfirmed if the field is actually missing.

        This prevents over-broad `insufficient_evidence` lists from marking present fields
        (e.g. overview.project_name) as "not confirmed".
        """
        p = (path or "").strip()
        if not p:
            return False
        if p == "objective":
            return not (ex.objective and ex.objective.text)
        if p == "theory_of_change":
            return len(ex.theory_of_change or []) == 0
        if p == "risks_and_limitations":
            return len(ex.risks_and_limitations or []) == 0
        if p.startswith("overview.") and ex.overview:
            key = p.split(".", 1)[1]
            sf = getattr(ex.overview, key, None)
            return not _has_scalar(sf)
        if p.startswith("ratings.") and ex.ratings:
            key = p.split(".", 1)[1]
            sf = getattr(ex.ratings, key, None)
            return not _has_scalar(sf)
        # Default: if we can't resolve it, keep it (safer).
        return True

    out: set[str] = set()
    for i in issues or []:
        # Only treat hard errors as "not confirmed" signals.
        if getattr(i, "severity", None) == "error":
            out.add(_normalize_field_path(i.field_path))
    for p in ex.insufficient_evidence or []:
        if isinstance(p, str) and p.strip():
            pp = p.strip()
            if _path_is_missing(pp):
                out.add(pp)
    return out


def _mark_if_unconfirmed(value: str, *, unconfirmed: bool) -> str:
    NOT_CONFIRMED = "*Not confirmed.*"
    if not value:
        return NOT_CONFIRMED
    return f"{value} ({NOT_CONFIRMED})" if unconfirmed else value


def _iter_all_citations(ex: ExtractionOutput) -> list[Citation]:
    out: list[Citation] = []

    if ex.overview:
        ov = ex.overview
        for sf in [
            ov.project_name,
            ov.country,
            ov.region,
            ov.practice_area,
            ov.financing_instrument,
            ov.borrower,
            ov.implementing_agency,
            ov.approval_date,
            ov.effectiveness_date,
            ov.closing_date,
            ov.total_commitment_usd,
            ov.world_bank_financing_usd,
            ov.borrower_contribution_usd,
            ov.total_project_cost_usd,
        ]:
            if sf and sf.citations:
                out.extend(sf.citations)

    if ex.ratings:
        rt = ex.ratings
        for sf in [rt.pdo_rating, rt.ip_rating, rt.overall_risk_rating, rt.outcome_rating]:
            if sf and sf.citations:
                out.extend(sf.citations)

    if ex.objective:
        out.extend(ex.objective.citations or [])

    for t in ex.theory_of_change or []:
        out.extend(t.citations or [])
    for r in ex.risks_and_limitations or []:
        out.extend(r.citations or [])

    for ind in ex.results_indicators or []:
        if ind.citations:
            out.extend(ind.citations)
        for f in [ind.baseline, ind.target, ind.achieved]:
            if f and f.citations:
                out.extend(f.citations)

    return out


def _project_links(project_id: str) -> tuple[str, str, str]:
    pid = (project_id or "").strip()
    project_page = f"https://projects.worldbank.org/en/projects-operations/project-detail/{pid}"
    wds_docs = (
        "https://search.worldbank.org/api/v3/wds?format=json&"
        f"projectid={pid}&majdocty_exact=Project%20Documents&sort=docdt&order=desc"
    )
    docs_portal = "https://documents.worldbank.org/en/publication/documents-reports"
    return project_page, wds_docs, docs_portal


def _achievement_bucket(achieved: float | None, target: float | None) -> str | None:
    if achieved is None or target is None:
        return None
    if target == 0:
        return None
    ratio = achieved / target
    if ratio >= 1.0:
        return "met_or_exceeded"
    if ratio >= 0.5:
        return "partially_met"
    return "not_met"


def render_report_md(
    ex: ExtractionOutput,
    *,
    issues: list[ValidationIssue] | None = None,
    extraction_status: str | None = None,
    n_chunks_used: int | None = None,
    n_selected_docs: int | None = None,
    doc_pdf_urls: dict[str, str] | None = None,
) -> str:
    unconfirmed_paths = _build_unconfirmed_paths(ex, issues)

    def _is_unconfirmed(path: str) -> bool:
        return (path or "") in unconfirmed_paths

    def _fmt_scalar_value(sf, path: str) -> str:
        if not sf or not getattr(sf, "value", ""):
            return "*Not confirmed.*"
        return _mark_if_unconfirmed(str(sf.value), unconfirmed=_is_unconfirmed(path))

    lines: list[str] = []
    lines.append(f"# ImpactLens Report — {ex.project_id}")
    lines.append("")

    # Header / sources
    # (URLs intentionally omitted for report cleanliness.)

    # Brief context (mark missing/uncertain as not confirmed)
    if ex.overview:
        lines.append(f"**Project**: {_fmt_scalar_value(ex.overview.project_name, 'overview.project_name')}")
        lines.append(f"**Country**: {_fmt_scalar_value(ex.overview.country, 'overview.country')}")
        lines.append("")

    # Objective is useful context for interpreting "achieved"
    lines.append("## Objective (from project documents)")
    if ex.objective and ex.objective.text:
        obj_text = ex.objective.text
        lines.append(_mark_if_unconfirmed(obj_text, unconfirmed=_is_unconfirmed("objective")))
        cit = _fmt_citations(ex.objective.citations)
        if cit:
            lines.append("")
            lines.append("**Evidence**")
            lines.append(cit)
    else:
        lines.append("*Not confirmed.*")
    lines.append("")

    # Achievements-focused summary
    lines.append("## What was achieved (from project documents)")
    if not ex.results_indicators and not (ex.ratings and ex.ratings.outcome_rating):
        lines.append("*Not confirmed.*")
        lines.append("")
    else:
        # Ratings/outcome first (often the closest thing to an overall achievement statement)
        if ex.ratings:
            rt = ex.ratings
            parts = [
                f"Outcome rating: **{_fmt_scalar_value(rt.outcome_rating, 'ratings.outcome_rating')}**",
                f"PDO progress: **{_fmt_scalar_value(rt.pdo_rating, 'ratings.pdo_rating')}**",
                f"Implementation progress: **{_fmt_scalar_value(rt.ip_rating, 'ratings.ip_rating')}**",
            ]
            # Only show if at least one is present (even if unconfirmed).
            if any("not confirmed" not in p.lower() for p in parts) or any("not confirmed" in p.lower() for p in parts):
                lines.append("- " + "; ".join(parts))

        inds = ex.results_indicators or []
        with_achieved = [i for i in inds if i.achieved and i.achieved.value is not None]
        with_target = [i for i in inds if i.target and i.target.value is not None]
        lines.append(f"- Indicators extracted: **{len(inds)}** (with achieved values: **{len(with_achieved)}**)")

        # Bucket indicators where both achieved & target exist
        met = partial = not_met = 0
        for ind in inds:
            b = _achievement_bucket(
                ind.achieved.value if ind.achieved else None, ind.target.value if ind.target else None
            )
            if b == "met_or_exceeded":
                met += 1
            elif b == "partially_met":
                partial += 1
            elif b == "not_met":
                not_met += 1
        if met + partial + not_met > 0:
            lines.append(
                f"- Target attainment (where achieved & target were found): "
                f"met/exceeded **{met}**, partially met **{partial}**, not met **{not_met}**"
            )
        elif with_target:
            lines.append("- Target attainment: _not computed_ (targets exist but achieved values were missing).")

        lines.append("")
        lines.append("### Key results (evidence-backed)")
        # Show up to 8 indicators that have achieved values; fall back to those with targets.
        def _score(ind) -> int:
            s = 0
            if ind.achieved and ind.achieved.value is not None:
                s += 5
            if ind.target and ind.target.value is not None:
                s += 3
            if ind.baseline and ind.baseline.value is not None:
                s += 1
            return s

        show = sorted(inds, key=_score, reverse=True)[:8]
        if not show:
            lines.append("_No results indicators extracted._")
        else:
            ind_index = {id(obj): i for i, obj in enumerate(inds)}

            def _fmt_fact(fact, *, path: str) -> str:
                if not fact or (fact.value is None and fact.year is None):
                    return "*Not confirmed.*"
                v = fact.value if fact.value is not None else ""
                val = f"{v} ({fact.year})" if fact.year else f"{v}"
                return _mark_if_unconfirmed(val, unconfirmed=_is_unconfirmed(path))

            def _best_evidence(ind, *, base_path: str) -> str:
                # Prefer achieved evidence, then target, then baseline, then indicator-level citations.
                if ind.achieved and ind.achieved.citations:
                    ev = _fmt_citations_table_links(ind.achieved.citations, doc_pdf_urls)
                    return _mark_if_unconfirmed(ev, unconfirmed=_is_unconfirmed(f"{base_path}.achieved"))
                if ind.target and ind.target.citations:
                    ev = _fmt_citations_table_links(ind.target.citations, doc_pdf_urls)
                    return _mark_if_unconfirmed(ev, unconfirmed=_is_unconfirmed(f"{base_path}.target"))
                if ind.baseline and ind.baseline.citations:
                    ev = _fmt_citations_table_links(ind.baseline.citations, doc_pdf_urls)
                    return _mark_if_unconfirmed(ev, unconfirmed=_is_unconfirmed(f"{base_path}.baseline"))
                if ind.citations:
                    ev = _fmt_citations_table_links(ind.citations, doc_pdf_urls)
                    return _mark_if_unconfirmed(ev, unconfirmed=_is_unconfirmed(base_path))
                return "*Not confirmed.*"

            # Markdown table
            lines.append("| Indicator | Unit | Baseline | Target | Achieved | Evidence |")
            lines.append("|---|---|---:|---:|---:|---|")
            for ind in show:
                idx = ind_index.get(id(ind))
                base_path = f"results_indicators[{idx}]" if isinstance(idx, int) else "results_indicators[?]"
                name = (ind.name or "").replace("\n", " ").strip()
                unit = (ind.unit or "").replace("\n", " ").strip()
                baseline = _fmt_fact(ind.baseline, path=f"{base_path}.baseline").replace("\n", " ").strip()
                target = _fmt_fact(ind.target, path=f"{base_path}.target").replace("\n", " ").strip()
                achieved = _fmt_fact(ind.achieved, path=f"{base_path}.achieved").replace("\n", " ").strip()
                ev = _best_evidence(ind, base_path=base_path).replace("\n", " ").replace("  ", " ").strip()
                # escape pipes to avoid breaking table
                name = name.replace("|", "\\|")
                unit = unit.replace("|", "\\|")
                ev = ev.replace("|", "\\|")
                lines.append(f"| {name} | {unit} | {baseline} | {target} | {achieved} | {ev} |")
        lines.append("")

    lines.append("## Risks & limitations")
    if ex.risks_and_limitations:
        for i, r in enumerate(ex.risks_and_limitations):
            path = f"risks_and_limitations[{i}]"
            txt = r.text if r and r.text else ""
            lines.append(f"- {_mark_if_unconfirmed(txt, unconfirmed=_is_unconfirmed(path))}")
            cit = _fmt_citations(r.citations)
            if cit:
                lines.append(f"  - Evidence: {cit}")
    else:
        lines.append("*Not confirmed.*")
    lines.append("")

    # Data quality / discrepancies / exhaustivity
    lines.append("## Data quality, discrepancies, and coverage notes")
    if extraction_status:
        lines.append(f"- Extraction status: **{extraction_status}**")
    if n_chunks_used is not None:
        lines.append(f"- Evidence chunks used for extraction: **{n_chunks_used}**")
    if n_selected_docs is not None:
        lines.append(f"- Selected project documents available (top-ranked): **{n_selected_docs}**")

    n_ie = len(ex.insufficient_evidence or [])
    if n_ie:
        lines.append(f"- Missing/insufficient evidence fields flagged: **{n_ie}**")

    issues = issues or []
    errs = [i for i in issues if i.severity == "error"]
    warns = [i for i in issues if i.severity == "warn"]
    if issues:
        lines.append(f"- Validation issues: **{len(errs)} errors**, **{len(warns)} warnings**")
        # Show a small sample to make “discrepancies” concrete.
        sample = (errs + warns)[:6]
        for it in sample:
            lines.append(f"  - `{it.issue_type}` at `{it.field_path}`: {it.message}")
    else:
        lines.append("- Validation issues: _none recorded_")

    # Citation coverage heuristics (accuracy risk proxy)
    citations = _iter_all_citations(ex)
    unique_docs = {c.doc_id for c in citations if c.doc_id}
    lines.append(f"- Evidence references: **{len(citations)}** citations across **{len(unique_docs)}** documents")
    if not citations and (ex.objective or ex.results_indicators or ex.overview or ex.ratings):
        lines.append(
            "- **Accuracy risk**: extracted fields exist but few/no citations were attached."
        )
    if n_ie or errs:
        lines.append(
            "- **Exhaustivity risk**: some fields are missing evidence or have validation errors; "
            "treat reported numbers as incomplete until reviewed against the source PDFs."
        )

    lines.append("")

    if ex.insufficient_evidence:
        lines.append("## Insufficient evidence")
        for p in ex.insufficient_evidence:
            lines.append(f"- `{p}`")
        lines.append("")

    # Evidence base: which documents were actually cited
    if citations:
        lines.append("## Evidence base (most-cited documents)")
        counts: dict[str, int] = {}
        for c in citations:
            counts[c.doc_id] = counts.get(c.doc_id, 0) + 1
        top = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:8]
        for doc_id, n in top:
            lines.append(f"- `{doc_id}` — {n} citations")
        lines.append("")

    lines.append(f"_Generated at {utc_now().isoformat()}_")
    return "\n".join(lines)


@dataclass(frozen=True)
class BuildReportResult:
    dataset_version_id: str
    project_id: str
    report_path: str


def build_report(*, dataset_version_id: str, project_id: str) -> BuildReportResult:
    ex_df = read_parquet(layer="gold", name="extractions", dataset_version_id=dataset_version_id)
    ex_df = ex_df.filter(pl.col("project_id") == project_id)
    if ex_df.height == 0:
        raise ValueError(f"No extractions found for project_id={project_id}")

    # latest by created_at
    ex_df = ex_df.sort("created_at").tail(1)
    row = ex_df.to_dicts()[0]
    output = ExtractionOutput.model_validate(json.loads(row["output_json"]))

    issues_raw = row.get("validation_issues_json")
    issues: list[ValidationIssue] = []
    try:
        parsed = json.loads(issues_raw) if issues_raw else []
        if isinstance(parsed, list):
            issues = [ValidationIssue.model_validate(x) for x in parsed if isinstance(x, dict)]
    except Exception:  # noqa: BLE001
        issues = []

    # Optional coverage metadata (best-effort; avoid hard failures)
    n_selected_docs: int | None = None
    doc_pdf_urls: dict[str, str] = {}
    try:
        docs_sel = read_parquet(layer="silver", name="documents_selected", dataset_version_id=dataset_version_id)
        if "project_id" in docs_sel.columns:
            docs_sel = docs_sel.filter(pl.col("project_id") == project_id)
        n_selected_docs = int(docs_sel.select(pl.col("doc_id").n_unique()).item())
        if "pdf_url" in docs_sel.columns:
            cols = ["doc_id", "pdf_url"] + (["url"] if "url" in docs_sel.columns else [])
            for r in docs_sel.select(cols).iter_rows(named=True):
                doc_id = str(r.get("doc_id") or "").strip()
                pdf_url = str((r.get("pdf_url") or r.get("url") or "")).strip()
                if doc_id and pdf_url:
                    doc_pdf_urls[doc_id] = pdf_url
    except Exception:  # noqa: BLE001
        n_selected_docs = None
        doc_pdf_urls = {}

    # Fall back to full documents table for PDF urls (if selected-docs missing/incomplete)
    if not doc_pdf_urls:
        try:
            docs = read_parquet(layer="silver", name="documents", dataset_version_id=dataset_version_id)
            docs = docs.filter(pl.col("project_id") == project_id)
            if "pdf_url" in docs.columns:
                cols = ["doc_id", "pdf_url"] + (["url"] if "url" in docs.columns else [])
                for r in docs.select(cols).iter_rows(named=True):
                    doc_id = str(r.get("doc_id") or "").strip()
                    pdf_url = str((r.get("pdf_url") or r.get("url") or "")).strip()
                    if doc_id and pdf_url:
                        doc_pdf_urls[doc_id] = pdf_url
        except Exception:  # noqa: BLE001
            doc_pdf_urls = {}

    report_md = render_report_md(
        output,
        issues=issues,
        extraction_status=str(row.get("status") or ""),
        n_chunks_used=int(row.get("n_chunks_used") or 0) if row.get("n_chunks_used") is not None else None,
        n_selected_docs=n_selected_docs,
        doc_pdf_urls=doc_pdf_urls,
    )

    out_df = pl.from_dicts(
        [
            {
                "project_id": project_id,
                "dataset_version_id": dataset_version_id,
                "created_at": utc_now().isoformat(),
                "report_md": report_md,
                "source_extraction_id": row.get("extraction_id"),
            }
        ]
    )
    path = append_parquet(out_df, layer="gold", name="project_reports", dataset_version_id=dataset_version_id)
    return BuildReportResult(dataset_version_id=dataset_version_id, project_id=project_id, report_path=str(path))

