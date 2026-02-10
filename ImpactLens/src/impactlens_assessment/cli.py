from __future__ import annotations

from pathlib import Path

import typer

from impactlens.settings import settings
from impactlens_assessment.assess import run_assessment, write_assessment_outputs


app = typer.Typer(add_completion=False, no_args_is_help=True)


@app.command("run")
def run_cmd(
    dataset_version: str = typer.Option(..., help="Dataset version id (YYYY-MM-DD)"),
    gold_facts_path: Path | None = typer.Option(
        None, exists=True, dir_okay=False, help="Optional JSONL gold facts file for exact-match accuracy"
    ),
    project_id: list[str] | None = typer.Option(
        None, "--project-id", help="Repeatable. If set, assess only these project IDs."
    ),
    stability_last_k: int = typer.Option(
        3, help="How many recent extraction runs per project to use for stability metrics."
    ),
    judge: bool = typer.Option(
        False, help="If set, run an Ollama LLM-as-judge pass to score report coherence."
    ),
    judge_model: str | None = typer.Option(
        None, help="Optional Ollama model name for the judge pass (defaults to ImpactLens config)."
    ),
    out_dir: Path | None = typer.Option(
        None, help="Output directory. Default: <data_dir>/assessments/<dataset_version>/"
    ),
) -> None:
    """Run external assessment for ImpactLens outputs."""
    out = out_dir or (settings.data_dir / "assessments" / dataset_version)
    summary = run_assessment(
        dataset_version_id=dataset_version,
        gold_facts_path=gold_facts_path,
        project_ids=project_id or None,
        stability_last_k=stability_last_k,
        judge_with_ollama=judge,
        judge_model=judge_model,
    )
    paths = write_assessment_outputs(out_dir=out, summary=summary)
    typer.echo(f"Wrote assessment report: {paths['md']}")
    typer.echo(f"Wrote assessment metrics: {paths['json']}")


if __name__ == "__main__":
    app()

