from __future__ import annotations

import asyncio
import json
from datetime import date
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

import polars as pl
import streamlit as st

from impactlens.pipelines.build_report import build_report
from impactlens.pipelines.build_silver import build_silver_documents
from impactlens.pipelines.extract_project import extract_project
from impactlens.pipelines.ingest_documents import ingest_documents_for_projects
from impactlens.pipelines.process_documents import process_documents
from impactlens.pipelines.select_documents import select_documents
from impactlens.schemas.extraction import ExtractionOutput, ValidationIssue
from impactlens.settings import settings
from impactlens.storage import read_parquet
from impactlens_assessment.assess import run_assessment, write_assessment_outputs


def _run_async(coro):
    """Run an async coroutine safely inside Streamlit."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def _project_webpage_url(project_id: str) -> str:
    pid = (project_id or "").strip()
    return f"https://projects.worldbank.org/en/projects-operations/project-detail/{pid}"


def _project_documents_wds_url(project_id: str) -> str:
    """Direct link to WDS API query for one project's docs (JSON)."""
    pid = (project_id or "").strip()
    params = {
        "format": "json",
        "projectid": pid,
        "majdocty_exact": "Project Documents",
        "sort": "docdt",
        "order": "desc",
        "rows": 100,
        "os": 0,
        "fl": "id,docdt,lang,display_title,pdfurl,docty,majdocty,seccl,url,projectid,proid,projn",
    }
    return f"https://search.worldbank.org/api/v3/wds?{urlencode(params)}"


def _link_button(label: str, url: str) -> None:
    """Use Streamlit link_button when available; fallback to markdown."""
    if hasattr(st, "link_button"):
        st.link_button(label, url, use_container_width=True)
    else:
        st.markdown(f"[{label}]({url})")


def _render_project_links(project_id: str) -> None:
    c1, c2 = st.columns(2)
    with c1:
        _link_button("Open project webpage", _project_webpage_url(project_id))
    with c2:
        _link_button("Open project documents (WDS)", _project_documents_wds_url(project_id))


def _assessment_out_dir(dataset_version_id: str) -> Path:
    return settings.data_dir / "assessments" / dataset_version_id


def _load_assessment_report_md(dataset_version_id: str) -> str | None:
    path = _assessment_out_dir(dataset_version_id) / "assessment_report.md"
    try:
        return path.read_text(encoding="utf-8")
    except Exception:  # noqa: BLE001
        return None


def _load_selected_docs(dataset_version_id: str, project_id: str) -> pl.DataFrame | None:
    try:
        df = read_parquet(layer="silver", name="documents_selected", dataset_version_id=dataset_version_id)
    except Exception:  # noqa: BLE001
        return None
    if "project_id" in df.columns:
        df = df.filter(pl.col("project_id") == project_id)
    return df


def _load_latest_extraction(dataset_version_id: str, project_id: str) -> dict[str, Any] | None:
    try:
        df = read_parquet(layer="gold", name="extractions", dataset_version_id=dataset_version_id)
    except Exception:  # noqa: BLE001
        return None
    df = df.filter(pl.col("project_id") == project_id)
    if df.height == 0:
        return None
    row = df.sort("created_at").tail(1).to_dicts()[0]
    out = json.loads(row["output_json"])
    issues = json.loads(row.get("validation_issues_json") or "[]")
    return {"row": row, "output": out, "issues": issues}


def _load_latest_report_md(dataset_version_id: str, project_id: str) -> str | None:
    try:
        df = read_parquet(layer="gold", name="project_reports", dataset_version_id=dataset_version_id)
    except Exception:  # noqa: BLE001
        return None
    df = df.filter(pl.col("project_id") == project_id)
    if df.height == 0:
        return None
    row = df.sort("created_at").tail(1).to_dicts()[0]
    return str(row["report_md"])


def _citations_to_df(citations: list[dict[str, Any]] | None) -> pl.DataFrame:
    rows: list[dict[str, Any]] = []
    for c in citations or []:
        if not isinstance(c, dict):
            continue
        rows.append(
            {
                "doc_id": c.get("doc_id"),
                "page": c.get("page"),
                "chunk_id": c.get("chunk_id"),
                "quote": c.get("quote"),
            }
        )
    return pl.from_dicts(rows) if rows else pl.DataFrame({"doc_id": [], "page": [], "chunk_id": [], "quote": []})


def _doc_index(docs_df: pl.DataFrame | None) -> dict[str, dict[str, Any]]:
    """Map doc_id -> metadata (title, pdf_url) from selected docs."""
    if docs_df is None or docs_df.height == 0:
        return {}
    out: dict[str, dict[str, Any]] = {}
    for r in docs_df.iter_rows(named=True):
        doc_id = str(r.get("doc_id") or "").strip()
        if not doc_id:
            continue
        out[doc_id] = {"title": r.get("title"), "pdf_url": r.get("pdf_url")}
    return out


def _local_pdf_path(dataset_version_id: str, doc_id: str) -> Path:
    # Mirrors artifact_dir("pdfs", dataset_version_id) / f"{doc_id}.pdf"
    return Path(settings.data_dir) / "artifacts" / "pdfs" / f"dataset_version={dataset_version_id}" / f"{doc_id}.pdf"


def _render_citation_popover(
    *,
    citations: list[dict[str, Any]] | None,
    docs_by_id: dict[str, dict[str, Any]],
    dataset_version_id: str,
    popover_label: str = "i",
) -> None:
    cdf = _citations_to_df(citations)
    if cdf.height == 0:
        st.caption("No citations provided.")
        return

    st.caption(f"Evidence ({cdf.height} citations)")
    st.dataframe(cdf.to_pandas(), use_container_width=True, height=min(260, 35 + 35 * min(6, cdf.height)))

    # Offer per-document open/download actions
    doc_ids = [d for d in cdf.get_column("doc_id").unique().to_list() if d]
    for doc_id in doc_ids[:5]:
        meta = docs_by_id.get(str(doc_id)) or {}
        title = meta.get("title")
        pdf_url = meta.get("pdf_url")
        st.markdown(f"**Source**: `{doc_id}`" + (f" — {title}" if title else ""))
        cols = st.columns(2)
        with cols[0]:
            if pdf_url:
                _link_button("Open PDF URL", str(pdf_url))
        with cols[1]:
            p = _local_pdf_path(dataset_version_id, str(doc_id))
            if p.exists():
                try:
                    data = p.read_bytes()
                    st.download_button(
                        "Download PDF",
                        data=data,
                        file_name=p.name,
                        mime="application/pdf",
                        use_container_width=True,
                        key=f"dl_{dataset_version_id}_{doc_id}_{popover_label}",
                    )
                except Exception:  # noqa: BLE001
                    st.caption("Could not read local PDF.")


def _render_scalar_fact_row(
    *,
    label: str,
    fact: dict[str, Any] | None,
    docs_by_id: dict[str, dict[str, Any]],
    dataset_version_id: str,
    key: str,
) -> None:
    value = ""
    citations = None
    if isinstance(fact, dict):
        value = str(fact.get("value") or "").strip()
        citations = fact.get("citations")

    c1, c2, c3 = st.columns([0.28, 0.62, 0.10], vertical_alignment="top")
    with c1:
        st.write(f"**{label}**")
    with c2:
        st.write(value if value else "—")
    with c3:
        if citations:
            if hasattr(st, "popover"):
                # Use a circled "i" to mimic common evidence/info iconography.
                with st.popover("ⓘ", help="Evidence", use_container_width=True):
                    _render_citation_popover(
                        citations=citations,
                        docs_by_id=docs_by_id,
                        dataset_version_id=dataset_version_id,
                        popover_label=key,
                    )
            else:
                with st.expander("ⓘ Evidence", expanded=False):
                    _render_citation_popover(
                        citations=citations,
                        docs_by_id=docs_by_id,
                        dataset_version_id=dataset_version_id,
                        popover_label=key,
                    )


def _render_text_with_citations(title: str, obj: dict[str, Any] | None) -> None:
    if not obj:
        st.write("_Not available._")
        return
    text = str(obj.get("text") or "").strip()
    if text:
        st.write(text)
    else:
        st.write("_Empty text._")
    cdf = _citations_to_df(obj.get("citations") if isinstance(obj, dict) else None)
    if cdf.height > 0:
        st.caption(f"Evidence ({cdf.height} citations)")
        st.dataframe(cdf.to_pandas(), use_container_width=True, height=min(220, 35 + 35 * min(5, cdf.height)))


def _render_project_details_table(
    *,
    project_id: str,
    output: dict[str, Any],
    docs_by_id: dict[str, dict[str, Any]],
    dataset_version_id: str,
) -> None:
    """Render a World Bank-like 'Project Details' view (label/value rows)."""
    ov = output.get("overview") if isinstance(output, dict) else None
    rt = output.get("ratings") if isinstance(output, dict) else None

    st.subheader(f"Project details — {project_id}")
    st.caption("Main view — formatted like the World Bank project webpage “Project Details” table.")

    with st.container(border=True):
        # Only label the evidence column (requested: remove "Field" and "Value" headers).
        _, _, h3 = st.columns([0.28, 0.62, 0.10], vertical_alignment="bottom")
        with h3:
            st.caption("Evidence")
        st.divider()

        if isinstance(ov, dict):
            def _first_fact(*keys: str) -> dict[str, Any] | None:
                for k in keys:
                    v = ov.get(k)
                    if isinstance(v, dict) and str(v.get("value") or "").strip():
                        return v
                return None

            for label, key in [
                ("Project name", "project_name"),
                ("Country", "country"),
                ("Region", "region"),
                ("Practice area", "practice_area"),
                ("Financing instrument", "financing_instrument"),
                ("Borrower", "borrower"),
                ("Implementing agency", "implementing_agency"),
                ("Approval date", "approval_date"),
                ("Effectiveness date", "effectiveness_date"),
                ("Closing date", "closing_date"),
            ]:
                _render_scalar_fact_row(
                    label=label,
                    fact=ov.get(key),
                    docs_by_id=docs_by_id,
                    dataset_version_id=dataset_version_id,
                    key=f"details_overview_{key}",
                )

            # Financials (show the “best available” facts)
            _render_scalar_fact_row(
                label="World Bank financing (USD)",
                fact=_first_fact("world_bank_financing_usd", "total_commitment_usd"),
                docs_by_id=docs_by_id,
                dataset_version_id=dataset_version_id,
                key="details_overview_world_bank_financing_usd",
            )
            _render_scalar_fact_row(
                label="Borrower contribution (USD)",
                fact=ov.get("borrower_contribution_usd"),
                docs_by_id=docs_by_id,
                dataset_version_id=dataset_version_id,
                key="details_overview_borrower_contribution_usd",
            )
            _render_scalar_fact_row(
                label="Total project cost (USD)",
                fact=ov.get("total_project_cost_usd"),
                docs_by_id=docs_by_id,
                dataset_version_id=dataset_version_id,
                key="details_overview_total_project_cost_usd",
            )
        else:
            st.write("_No overview extracted yet._")

        st.divider()
        st.markdown("**Ratings**")
        if isinstance(rt, dict):
            for label, key in [
                ("PDO rating", "pdo_rating"),
                ("Implementation progress (IP)", "ip_rating"),
                ("Overall risk rating", "overall_risk_rating"),
                ("Outcome rating", "outcome_rating"),
            ]:
                _render_scalar_fact_row(
                    label=label,
                    fact=rt.get(key),
                    docs_by_id=docs_by_id,
                    dataset_version_id=dataset_version_id,
                    key=f"details_ratings_{key}",
                )
        else:
            st.write("_No ratings extracted yet._")


def _render_validation_summary(issues: list[dict[str, Any]]) -> None:
    errs = [i for i in issues if isinstance(i, dict) and i.get("severity") == "error"]
    warns = [i for i in issues if isinstance(i, dict) and i.get("severity") == "warn"]
    c1, c2, c3 = st.columns(3)
    c1.metric("Errors", len(errs))
    c2.metric("Warnings", len(warns))
    c3.metric("Total issues", len(issues))

    if not issues:
        st.success("No validation issues.")
        return

    df = pl.from_dicts([i for i in issues if isinstance(i, dict)])
    if "issue_type" in df.columns:
        counts = df.group_by(["severity", "issue_type"]).len().sort(["severity", "len"], descending=[False, True])
        st.caption("Issue counts")
        st.dataframe(counts.to_pandas(), use_container_width=True, height=200)

    with st.expander("Show all validation issues", expanded=False):
        st.json(issues)


def main() -> None:
    st.set_page_config(page_title="ImpactLens", layout="wide")
    st.title("ImpactLens — World Bank project extraction")

    with st.sidebar:
        st.link_button("Help", settings.github_readme_url, use_container_width=True)
        st.caption("Opens the analysis & interpretation guide on GitHub.")
        st.divider()

        st.subheader("Run configuration")
        project_id = st.text_input("Project ID", value="P131765")
        dataset_version = st.text_input("Dataset version (YYYY-MM-DD)", value=date.today().isoformat())

        st.divider()
        st.caption("WDS ingestion")
        docs_rows = st.number_input("rows", min_value=10, max_value=500, value=100, step=10)
        docs_max_pages = st.number_input("max_pages", min_value=1, max_value=10, value=2, step=1)

        st.divider()
        st.caption("Selection & processing")
        top_k_docs = st.number_input("top_k_docs", min_value=1, max_value=20, value=5, step=1)
        docs_limit = st.number_input("docs_limit (download/OCR cap)", min_value=1, max_value=20, value=5, step=1)
        process_all_docs = st.checkbox("Process all project documents (slower, most complete)", value=False)
        ocr_min_total_chars = st.number_input("ocr_min_total_chars", min_value=0, max_value=5000, value=500, step=100)

        st.divider()
        st.caption("LLM extraction")
        max_chunks = st.number_input("max_chunks", min_value=5, max_value=80, value=30, step=5)
        analysis_mode = st.selectbox("Analysis mode", options=["fast", "full"], index=0)

        st.divider()
        st.caption("Local settings")
        st.code(
            f"Ollama: {settings.ollama_base_url}\nModel: {settings.ollama_model}\nTimeout: {settings.ollama_timeout_s}s\nData dir: {settings.data_dir}",
            language="text",
        )

        run_demo = st.button("Run case analysis", type="primary", use_container_width=True)

        st.divider()
        st.caption("Model assessment")
        gen_assessment = st.button("Generate model assessment report", use_container_width=True)
        if gen_assessment:
            status = st.status("Generating assessment report…", expanded=True)
            try:
                status.write("Assessing latest extraction + citations…")
                summary = run_assessment(
                    dataset_version_id=dataset_version,
                    gold_facts_path=None,
                    project_ids=[project_id],
                    stability_last_k=3,
                    judge_with_ollama=False,
                    judge_model=None,
                )
                out_dir = _assessment_out_dir(dataset_version)
                paths = write_assessment_outputs(out_dir=out_dir, summary=summary)
                st.session_state["impactlens_assessment_md"] = Path(paths["md"]).read_text(encoding="utf-8")
                status.update(label="Assessment ready", state="complete", expanded=False)
            except Exception as e:  # noqa: BLE001
                status.update(label="Assessment failed", state="error", expanded=True)
                st.exception(e)

        # Download control (we don't render the assessment in the main UI)
        assess_md = st.session_state.get("impactlens_assessment_md") or _load_assessment_report_md(dataset_version)
        if assess_md:
            st.download_button(
                "Download model assessment report",
                data=str(assess_md).encode("utf-8"),
                file_name=f"impactlens_assessment_{project_id}_{dataset_version}.md",
                mime="text/markdown",
                use_container_width=True,
                help="Downloads assessment_report.md",
            )

    if not project_id.strip():
        st.warning("Enter a project ID to continue.")
        return

    if run_demo:
        status = st.status("Running pipeline…", expanded=True)
        try:
            status.write("1) Ingesting WDS document metadata…")
            res_docs = _run_async(
                ingest_documents_for_projects(
                    dataset_version_id=dataset_version,
                    project_ids=[project_id],
                    rows=int(docs_rows),
                    max_pages=int(docs_max_pages),
                    project_docs_only=True,
                )
            )
            status.write(f"   → {res_docs.n_documents} documents ingested.")

            status.write("2) Building silver/documents…")
            build_silver_documents(dataset_version_id=dataset_version)
            status.write("   → done.")

            status.write("3) Selecting best documents (silver/documents_selected)…")
            res_sel = select_documents(
                dataset_version_id=dataset_version,
                project_id=project_id,
                top_k_per_project=int(top_k_docs),
            )
            status.write(f"   → selected {res_sel.n_selected} documents.")

            status.write("4) Download + PDF text + OCR fallback + chunking…")
            res_proc = _run_async(
                process_documents(
                    dataset_version_id=dataset_version,
                    project_id=project_id,
                    limit=int(docs_limit),
                    top_k_per_project=int(top_k_docs),
                    process_all_docs=bool(process_all_docs),
                    ocr_min_total_chars=int(ocr_min_total_chars),
                )
            )
            status.write(
                f"   → processed {res_proc.n_docs_downloaded_ok}/{res_proc.n_docs_attempted} docs, "
                f"{res_proc.n_chunks} chunks."
            )

            status.write("5) Ollama extraction (JSON + citations + validation)…")
            res_ext = _run_async(
                extract_project(
                    dataset_version_id=dataset_version,
                    project_id=project_id,
                    max_chunks=int(max_chunks),
                    analysis_mode=str(analysis_mode),
                )
            )
            status.write(
                f"   → status={res_ext.status} validation_errors={res_ext.n_validation_errors}."
            )

            status.write("6) Building report…")
            res_rep = build_report(dataset_version_id=dataset_version, project_id=project_id)
            status.write(f"   → wrote to {res_rep.report_path}")

            status.write("7) Generating model assessment report…")
            try:
                summary = run_assessment(
                    dataset_version_id=dataset_version,
                    gold_facts_path=None,
                    project_ids=[project_id],
                    stability_last_k=3,
                    judge_with_ollama=False,
                    judge_model=None,
                )
                out_dir = _assessment_out_dir(dataset_version)
                paths = write_assessment_outputs(out_dir=out_dir, summary=summary)
                st.session_state["impactlens_assessment_md"] = Path(paths["md"]).read_text(encoding="utf-8")
                status.write("   → assessment_report.md ready (download from sidebar).")
            except Exception as e:  # noqa: BLE001
                status.write("   → assessment failed (see error below).")
                st.exception(e)

            status.update(label="Done", state="complete", expanded=False)
        except Exception as e:  # noqa: BLE001
            status.update(label="Failed", state="error", expanded=True)
            st.exception(e)

    # Display outputs (even if not run, show what's available)
    # Load once for analysis + raw + details
    ext = _load_latest_extraction(dataset_version, project_id)
    docs_df = _load_selected_docs(dataset_version, project_id)
    docs_by_id = _doc_index(docs_df)

    # Project details: full width
    if not ext:
        st.info("No extraction found yet. Run the demo to generate one.")
    else:
        st.caption(f"status: `{ext['row'].get('status')}` — extraction_id: `{ext['row'].get('extraction_id')}`")
        output_raw = ext["output"]
        try:
            output = ExtractionOutput.model_validate(output_raw).model_dump()
        except Exception:  # noqa: BLE001
            output = output_raw if isinstance(output_raw, dict) else {"project_id": project_id}
            output.setdefault("project_id", project_id)

        _render_project_details_table(
            project_id=project_id,
            output=output,
            docs_by_id=docs_by_id,
            dataset_version_id=dataset_version,
        )

        # Quick, webpage-like snapshot: indicators table (if any)
        inds = output.get("results_indicators") if isinstance(output, dict) else None
        inds = inds or []
        if inds:
            st.markdown("### Results indicators (snapshot)")
            rows = []
            for ind in inds:
                if not isinstance(ind, dict):
                    continue

                def _fmt_fact(f: dict[str, Any] | None) -> str:
                    if not isinstance(f, dict):
                        return ""
                    v = f.get("value")
                    y = f.get("year")
                    if v is None and y is None:
                        return ""
                    return f"{v}" + (f" ({y})" if y else "")

                rows.append(
                    {
                        "name": ind.get("name"),
                        "unit": ind.get("unit"),
                        "baseline": _fmt_fact(ind.get("baseline")),
                        "target": _fmt_fact(ind.get("target")),
                        "achieved": _fmt_fact(ind.get("achieved")),
                    }
                )
            st.dataframe(pl.from_dicts(rows).to_pandas(), use_container_width=True, height=220)

    st.caption("Project links")
    _render_project_links(project_id)

    st.divider()

    tab_report, tab_raw, tab_docs = st.tabs(["Report", "Raw JSON", "Selected documents"])

    with tab_report:
        st.subheader("Report (Markdown)")
        report_md = _load_latest_report_md(dataset_version, project_id)
        if report_md:
            st.markdown(report_md)
            st.download_button(
                "Download report.md",
                data=report_md.encode("utf-8"),
                file_name=f"impactlens_report_{project_id}_{dataset_version}.md",
                mime="text/markdown",
                use_container_width=True,
            )
        else:
            st.info("No report found yet. Run the demo to generate one.")

    with tab_raw:
        st.subheader("Latest extraction (JSON)")
        if ext:
            st.caption(
                f"status: `{ext['row'].get('status')}` — extraction_id: `{ext['row'].get('extraction_id')}`"
            )
            st.download_button(
                "Download extraction.json",
                data=json.dumps(ext["output"], ensure_ascii=False, indent=2).encode("utf-8"),
                file_name=f"impactlens_extraction_{project_id}_{dataset_version}.json",
                mime="application/json",
                use_container_width=True,
            )
            st.json(ext["output"])
            if ext["issues"]:
                st.subheader("Validation issues")
                st.download_button(
                    "Download validation_issues.json",
                    data=json.dumps(ext["issues"], ensure_ascii=False, indent=2).encode("utf-8"),
                    file_name=f"impactlens_validation_issues_{project_id}_{dataset_version}.json",
                    mime="application/json",
                    use_container_width=True,
                )
                st.json(ext["issues"])
        else:
            st.info("No extraction found yet. Run the demo to generate one.")

    with tab_docs:
        st.subheader("Selected documents")
        if docs_df is not None and docs_df.height > 0:
            show_cols = [
                c
                for c in [
                    "doc_id",
                    "doc_class",
                    "doc_score",
                    "publication_date",
                    "doc_type",
                    "language",
                    "title",
                    "pdf_url",
                ]
                if c in docs_df.columns
            ]
            st.dataframe(docs_df.select(show_cols).to_pandas(), use_container_width=True, height=350)
        else:
            st.info("No selected docs table yet (silver/documents_selected).")


if __name__ == "__main__":
    main()

