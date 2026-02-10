from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="IMPACTLENS_", env_file=".env", extra="ignore")

    data_dir: Path = Path("data")

    # Ollama
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.1:8b"
    ollama_timeout_s: float = 600.0

    # Execution / defaults
    http_timeout_s: float = 60.0

    # UI / docs
    # Override with IMPACTLENS_GITHUB_README_URL if your doc URL differs.
    github_readme_url: str = (
        "https://github.com/lofillon/lofillon/blob/main/impactlens_app/How_to_analyse_and_interpret.md"
        "#impactlens--analysis-method-and-limitations-evidence-grounded-llm--ocr"
    )


settings = Settings()

