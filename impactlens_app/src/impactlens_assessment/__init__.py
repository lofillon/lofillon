"""External assessment module for ImpactLens.

This package reads ImpactLens outputs (gold/silver Parquet + manifests) and produces:
- quantitative benchmark metrics (accuracy, grounding, completeness, stability)
- a human-readable markdown assessment report
"""

from __future__ import annotations

