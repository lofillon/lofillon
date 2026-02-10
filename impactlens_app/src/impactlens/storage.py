from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import polars as pl

from impactlens.settings import settings
from impactlens.utils import dump_json, utc_now

Layer = Literal["bronze", "silver", "gold", "artifacts", "manifests", "logs"]


@dataclass(frozen=True)
class DataLayout:
    root: Path

    @property
    def bronze(self) -> Path:
        return self.root / "bronze"

    @property
    def silver(self) -> Path:
        return self.root / "silver"

    @property
    def gold(self) -> Path:
        return self.root / "gold"

    @property
    def artifacts(self) -> Path:
        return self.root / "artifacts"

    @property
    def manifests(self) -> Path:
        return self.root / "manifests"

    @property
    def logs(self) -> Path:
        return self.root / "logs"

    def ensure(self) -> None:
        for p in [self.bronze, self.silver, self.gold, self.artifacts, self.manifests, self.logs]:
            p.mkdir(parents=True, exist_ok=True)

    def layer_dir(self, layer: Layer) -> Path:
        return getattr(self, layer)


def get_layout() -> DataLayout:
    layout = DataLayout(settings.data_dir)
    layout.ensure()
    return layout


def dataset_path(layer: Layer, name: str, dataset_version_id: str) -> Path:
    layout = get_layout()
    return layout.layer_dir(layer) / name / f"dataset_version={dataset_version_id}"


def artifact_dir(kind: str, dataset_version_id: str) -> Path:
    """Directory for binary artifacts (PDFs, images, etc.)."""
    layout = get_layout()
    out = layout.artifacts / kind / f"dataset_version={dataset_version_id}"
    out.mkdir(parents=True, exist_ok=True)
    return out


def write_parquet(
    df: pl.DataFrame, *, layer: Layer, name: str, dataset_version_id: str, filename: str = "data.parquet"
) -> Path:
    out_dir = dataset_path(layer, name, dataset_version_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / filename
    df.write_parquet(out_path)
    return out_path


def append_parquet(
    df: pl.DataFrame, *, layer: Layer, name: str, dataset_version_id: str, filename: str = "data.parquet"
) -> Path:
    """Append rows to a Parquet file (small/medium scale convenience)."""
    out_dir = dataset_path(layer, name, dataset_version_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / filename
    if out_path.exists():
        existing = pl.read_parquet(out_path)
        combined = pl.concat([existing, df], how="vertical_relaxed")
        combined.write_parquet(out_path)
    else:
        df.write_parquet(out_path)
    return out_path


def read_parquet(*, layer: Layer, name: str, dataset_version_id: str, filename: str = "data.parquet") -> pl.DataFrame:
    path = dataset_path(layer, name, dataset_version_id) / filename
    return pl.read_parquet(path)


def write_manifest(dataset_version_id: str, payload: dict[str, Any]) -> Path:
    layout = get_layout()
    manifest = {
        "dataset_version_id": dataset_version_id,
        "created_at": utc_now().isoformat(),
        **payload,
    }
    path = layout.manifests / f"{dataset_version_id}.json"
    dump_json(path, manifest)
    return path

