from __future__ import annotations

from datetime import date

from fastapi import FastAPI
from fastapi.responses import PlainTextResponse

from impactlens import __version__
from impactlens.pipelines.build_report import build_report
from impactlens.pipelines.extract_project import extract_project


def create_app() -> FastAPI:
    app = FastAPI(title="ImpactLens", version=__version__)

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "version": __version__}

    @app.get("/version")
    def version() -> dict:
        return {"version": __version__}

    @app.post("/projects/{project_id}/extract")
    async def extract(project_id: str, dataset_version: str | None = None, max_chunks: int = 30) -> dict:
        dataset_version_id = dataset_version or date.today().isoformat()
        res = await extract_project(
            dataset_version_id=dataset_version_id, project_id=project_id, max_chunks=max_chunks
        )
        return res.__dict__

    @app.get("/projects/{project_id}/report", response_class=PlainTextResponse)
    def report(project_id: str, dataset_version: str | None = None) -> str:
        dataset_version_id = dataset_version or date.today().isoformat()
        res = build_report(dataset_version_id=dataset_version_id, project_id=project_id)
        # Return the markdown content (stored in Parquet as well).
        import polars as pl

        df = pl.read_parquet(res.report_path)
        return str(df.to_dicts()[0]["report_md"])

    return app

