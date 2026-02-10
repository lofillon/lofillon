from __future__ import annotations

import asyncio
import logging
import sys
from datetime import date

import typer

from impactlens.api import create_app
from impactlens.logging import configure_logging
from impactlens.storage import get_layout
from impactlens.pipelines.build_silver import build_silver_documents, build_silver_projects
from impactlens.pipelines.ingest_documents import ingest_documents_for_projects
from impactlens.pipelines.ingest_projects import ingest_projects
from impactlens.pipelines.process_documents import process_documents
from impactlens.pipelines.extract_project import extract_project
from impactlens.pipelines.build_report import build_report
from impactlens.pipelines.evaluate import evaluate
from impactlens.pipelines.select_documents import select_documents

app = typer.Typer(add_completion=False, no_args_is_help=True)


@app.command()
def init() -> None:
    """Initialize local data folders."""
    layout = get_layout()
    typer.echo(f"Initialized data layout at: {layout.root.resolve()}")


@app.command()
def api(host: str = "127.0.0.1", port: int = 8000, log_level: str = "info") -> None:
    """Run the FastAPI server."""
    import uvicorn

    configure_logging(getattr(logging, log_level.upper(), logging.INFO))
    uvicorn.run(create_app(), host=host, port=port, log_level=log_level)


@app.command("ui")
def ui_cmd(host: str = "127.0.0.1", port: int = 8501) -> None:
    """Launch the Streamlit UI (requires `pip install -e '.[ui]'`)."""
    from pathlib import Path
    import subprocess

    app_path = Path(__file__).resolve().parent / "ui" / "streamlit_app.py"
    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(app_path),
        "--server.address",
        host,
        "--server.port",
        str(port),
    ]
    raise typer.Exit(subprocess.call(cmd))


@app.command("ingest-projects")
def ingest_projects_cmd(
    dataset_version: str = typer.Option(default_factory=lambda: date.today().isoformat()),
    rows: int = 100,
    max_pages: int = 1,
) -> None:
    """Ingest raw projects from the World Bank Projects Search API into bronze."""
    configure_logging()
    res = asyncio.run(ingest_projects(dataset_version_id=dataset_version, rows=rows, max_pages=max_pages))
    typer.echo(f"Ingested {res.n_projects} projects into bronze (dataset_version={res.dataset_version_id}).")


@app.command("ingest-docs")
def ingest_docs_cmd(
    dataset_version: str = typer.Option(default_factory=lambda: date.today().isoformat()),
    project_id: list[str] = typer.Option(None, help="Repeatable. Example: --project-id P505244"),
    rows: int = 100,
    max_pages: int = 1,
    project_docs_only: bool = True,
    docty_exact: list[str] = typer.Option(
        None,
        help="Repeatable exact docty filter. Example: --docty-exact 'Implementation Status and Results Report'",
    ),
) -> None:
    """Ingest raw document metadata from WDS into bronze by searching each project_id."""
    configure_logging()
    if not project_id:
        raise typer.BadParameter("Provide at least one --project-id.")
    res = asyncio.run(
        ingest_documents_for_projects(
            dataset_version_id=dataset_version,
            project_ids=project_id,
            rows=rows,
            max_pages=max_pages,
            project_docs_only=project_docs_only,
            docty_exact=docty_exact or None,
        )
    )
    typer.echo(
        f"Ingested {res.n_documents} documents into bronze (dataset_version={res.dataset_version_id})."
    )


@app.command("build-silver")
def build_silver_cmd(
    dataset_version: str = typer.Option(default_factory=lambda: date.today().isoformat()),
    projects: bool = True,
    documents: bool = True,
) -> None:
    """Build normalized silver tables from bronze datasets."""
    configure_logging()
    if projects:
        res_p = build_silver_projects(dataset_version_id=dataset_version)
        typer.echo(f"Built silver/projects: {res_p.n_projects} rows.")
    if documents:
        res_d = build_silver_documents(dataset_version_id=dataset_version)
        typer.echo(f"Built silver/documents: {res_d.n_documents} rows.")


@app.command("process-docs")
def process_docs_cmd(
    dataset_version: str = typer.Option(default_factory=lambda: date.today().isoformat()),
    project_id: str | None = typer.Option(None, help="If set, only process docs for this project."),
    limit: int | None = typer.Option(None, help="Limit number of documents processed."),
    top_k_per_project: int = 5,
    process_all_docs: bool = typer.Option(False, help="If set, process all project docs (ignore top_k_per_project)."),
    ocr_min_total_chars: int = 500,
) -> None:
    """Download PDFs and build silver/document_text_chunks (OCR fallback)."""
    configure_logging()
    res = asyncio.run(
        process_documents(
            dataset_version_id=dataset_version,
            project_id=project_id,
            limit=limit,
            top_k_per_project=top_k_per_project,
            process_all_docs=process_all_docs,
            ocr_min_total_chars=ocr_min_total_chars,
        )
    )
    typer.echo(
        f"Processed {res.n_docs_downloaded_ok}/{res.n_docs_attempted} docs, produced {res.n_chunks} chunks."
    )


@app.command("select-docs")
def select_docs_cmd(
    dataset_version: str = typer.Option(default_factory=lambda: date.today().isoformat()),
    project_id: str | None = typer.Option(None, help="If set, select docs only for this project."),
    top_k_per_project: int = 5,
) -> None:
    """Select and freeze the best documents into silver/documents_selected."""
    configure_logging()
    res = select_documents(
        dataset_version_id=dataset_version,
        project_id=project_id,
        top_k_per_project=top_k_per_project,
    )
    typer.echo(f"Selected {res.n_selected} documents into silver/documents_selected.")


@app.command("extract-project")
def extract_project_cmd(
    project_id: str = typer.Option(..., help="Example: P505244"),
    dataset_version: str = typer.Option(default_factory=lambda: date.today().isoformat()),
    max_chunks: int = 30,
    analysis_mode: str = typer.Option("full", help="fast: fail-fast for thin corpora; full: deeper extraction"),
) -> None:
    """Run Ollama extraction for a project (requires silver/document_text_chunks)."""
    configure_logging()
    res = asyncio.run(
        extract_project(
            dataset_version_id=dataset_version,
            project_id=project_id,
            max_chunks=max_chunks,
            analysis_mode=analysis_mode,
        )
    )
    typer.echo(
        f"Extraction {res.extraction_id} status={res.status} chunks={res.n_chunks_used} "
        f"validation_errors={res.n_validation_errors}"
    )


@app.command("build-report")
def build_report_cmd(
    project_id: str = typer.Option(..., help="Example: P505244"),
    dataset_version: str = typer.Option(default_factory=lambda: date.today().isoformat()),
) -> None:
    """Build a decision-ready Markdown report from the latest extraction."""
    configure_logging()
    res = build_report(dataset_version_id=dataset_version, project_id=project_id)
    typer.echo(f"Wrote report to: {res.report_path}")


@app.command("evaluate")
def evaluate_cmd(
    gold_facts_path: str = typer.Option("assets/goldset/gold_facts.sample.jsonl"),
    dataset_version: str = typer.Option(default_factory=lambda: date.today().isoformat()),
) -> None:
    """Evaluate latest extractions against a gold facts JSONL file."""
    configure_logging()
    from pathlib import Path

    res = evaluate(dataset_version_id=dataset_version, gold_facts_path=Path(gold_facts_path))
    typer.echo(
        f"Eval accuracy={res.summary.accuracy:.3f} matched={res.summary.n_matched}/{res.summary.n_facts} "
        f"output={res.output_path}"
    )


@app.command("demo")
def demo_cmd(
    project_id: str = typer.Option(..., help="Example: P505244"),
    dataset_version: str = typer.Option(default_factory=lambda: date.today().isoformat()),
    docs_rows: int = 100,
    docs_max_pages: int = 2,
    docs_limit: int = 5,
    top_k_docs: int = 5,
    max_chunks: int = 30,
) -> None:
    """Run an end-to-end demo for one project (docs → OCR → Ollama → report)."""
    configure_logging()
    get_layout()  # ensure folders

    typer.echo(f"[demo] dataset_version={dataset_version} project_id={project_id}")

    # 1) Ingest document metadata
    res_docs = asyncio.run(
        ingest_documents_for_projects(
            dataset_version_id=dataset_version,
            project_ids=[project_id],
            rows=docs_rows,
            max_pages=docs_max_pages,
        )
    )
    typer.echo(f"[demo] ingested docs: {res_docs.n_documents}")

    # 2) Normalize to silver/documents
    build_silver_documents(dataset_version_id=dataset_version)
    typer.echo("[demo] built silver/documents")

    # 2b) Freeze best docs for the project
    res_sel = select_documents(
        dataset_version_id=dataset_version,
        project_id=project_id,
        top_k_per_project=top_k_docs,
    )
    typer.echo(f"[demo] selected docs: {res_sel.n_selected}")

    # 3) Download + extract chunks (OCR fallback)
    res_proc = asyncio.run(
        process_documents(
            dataset_version_id=dataset_version,
            project_id=project_id,
            limit=docs_limit,
            top_k_per_project=top_k_docs,
        )
    )
    typer.echo(f"[demo] processed docs ok={res_proc.n_docs_downloaded_ok} chunks={res_proc.n_chunks}")

    # 4) Extract with Ollama
    res_ext = asyncio.run(
        extract_project(dataset_version_id=dataset_version, project_id=project_id, max_chunks=max_chunks)
    )
    typer.echo(
        f"[demo] extraction status={res_ext.status} validation_errors={res_ext.n_validation_errors}"
    )

    # 5) Build report
    res_rep = build_report(dataset_version_id=dataset_version, project_id=project_id)
    typer.echo(f"[demo] report: {res_rep.report_path}")


@app.command()
def version() -> None:
    """Print the current version."""
    from impactlens import __version__

    typer.echo(__version__)


if __name__ == "__main__":
    app()

