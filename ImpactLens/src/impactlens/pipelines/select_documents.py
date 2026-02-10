from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from impactlens.storage import read_parquet, write_manifest, write_parquet
from impactlens.utils import utc_now


@dataclass(frozen=True)
class SelectDocumentsResult:
    dataset_version_id: str
    n_selected: int


def select_documents(
    *,
    dataset_version_id: str,
    project_id: str | None = None,
    top_k_per_project: int = 5,
) -> SelectDocumentsResult:
    """Freeze a reproducible selection of the best documents into `silver/documents_selected`."""
    docs = read_parquet(layer="silver", name="documents", dataset_version_id=dataset_version_id)

    if "has_pdf" in docs.columns:
        # Some records may only provide a landing-page `url` (e.g., WB documentdetail pages).
        # We can still process them because download-time resolution can fetch the real PDF.
        if "url" in docs.columns:
            docs = docs.filter((pl.col("has_pdf") == True) | pl.col("url").is_not_null())  # noqa: E712
        else:
            docs = docs.filter(pl.col("has_pdf") == True)  # noqa: E712
    if "is_public" in docs.columns:
        docs = docs.filter(pl.col("is_public") == True)  # noqa: E712
    if project_id is not None:
        docs = docs.filter(pl.col("project_id") == project_id)

    if docs.height > 0 and "doc_score" in docs.columns:
        # For exhaustivity, ensure we include a mix of doc types (ISR/PAD/ICR/etc.)
        # otherwise top-k is dominated by ICR/PAD and we miss ISR indicator tables.
        sort_cols = ["project_id", "doc_score"]
        desc = [False, True]
        if "publication_date" in docs.columns:
            sort_cols.append("publication_date")
            desc.append(True)
        base_sorted = docs.sort(sort_cols, descending=desc)

        if "doc_class" in base_sorted.columns:
            # Per project: take at least some ISR + one PAD + one ICR when available.
            def _take(doc_class: str, n: int) -> pl.DataFrame:
                return (
                    base_sorted.filter(pl.col("doc_class") == doc_class)
                    .group_by("project_id")
                    .head(n)
                )

            n_isr = max(1, min(3, top_k_per_project // 2))
            picked = [
                _take("ISR", n_isr),
                _take("ICR", 1),
                _take("PAD", 1),
                _take("RESULTS_FRAMEWORK", 1),
                _take("RESTRUCTURING", 1),
            ]
            picked_nonempty = [p for p in picked if p.height > 0]
            if not picked_nonempty:
                # Nothing matched the must-include classes; fall back to pure score/recency top-k.
                docs = base_sorted.group_by("project_id").head(top_k_per_project)
            else:
                combined = pl.concat(picked_nonempty, how="vertical_relaxed")
                combined = combined.unique(subset=["project_id", "doc_id"], keep="first").with_columns(
                    pl.lit(1).alias("must_include")
                )

                # Fill remaining slots by best score (non-mandatory)
                key_struct = combined.select(
                    pl.struct([pl.col("project_id").alias("project_id"), pl.col("doc_id").alias("doc_id")]).alias("k")
                ).get_column("k")
                remaining = base_sorted.filter(
                    ~pl.struct([pl.col("project_id").alias("project_id"), pl.col("doc_id").alias("doc_id")]).is_in(key_struct)
                ).with_columns(
                    pl.lit(0).alias("must_include")
                )

                # IMPORTANT: do not drop mandatory docs when trimming to top_k_per_project.
                # Sort with must_include first, then score/recency.
                sort_cols2 = ["project_id", "must_include", "doc_score"]
                desc2 = [False, True, True]
                if "publication_date" in base_sorted.columns:
                    sort_cols2.append("publication_date")
                    desc2.append(True)

                # Ensure identical column order for concat (Polars can be strict here).
                all_cols = list(base_sorted.columns) + ["must_include"]
                combined = combined.select(all_cols)
                remaining = remaining.select(all_cols)

                docs = (
                    pl.concat([combined, remaining], how="vertical_relaxed")
                    .sort(sort_cols2, descending=desc2)
                    .group_by("project_id")
                    .head(top_k_per_project)
                    .drop("must_include")
                )
        else:
            docs = base_sorted.group_by("project_id").head(top_k_per_project)

    # Persist selection
    write_parquet(docs, layer="silver", name="documents_selected", dataset_version_id=dataset_version_id)
    write_manifest(
        dataset_version_id,
        {
            "pipeline": "select_documents",
            "created_at": utc_now().isoformat(),
            "params": {"project_id": project_id, "top_k_per_project": top_k_per_project},
            "inputs": [{"layer": "silver", "name": "documents"}],
            "outputs": [{"layer": "silver", "name": "documents_selected"}],
            "counts": {"documents_selected": int(docs.height)},
        },
    )
    return SelectDocumentsResult(dataset_version_id=dataset_version_id, n_selected=int(docs.height))

