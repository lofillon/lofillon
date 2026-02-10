from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np
import pandas as pd


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Fichier introuvable: {path}")
    return pd.read_csv(path)


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _topk_nodes_table(
    df_metrics_all_nodes: pd.DataFrame,
    metrics: List[str],
    ks: List[int],
    node_col: str = "node",
) -> pd.DataFrame:
    rows = []
    n = len(df_metrics_all_nodes)
    for met in metrics:
        if met not in df_metrics_all_nodes.columns:
            continue
        for k in ks:
            kk = min(int(k), n)
            if kk <= 0:
                continue
            top = (
                df_metrics_all_nodes[[node_col, met]]
                .dropna()
                .sort_values(met, ascending=False)
                .head(kk)[node_col]
                .astype(str)
                .tolist()
            )
            for rank, node in enumerate(top, start=1):
                rows.append({"metric": met, "k": kk, "rank": rank, "node": node})
    return pd.DataFrame(rows)


def _winner_table_spearman(df_spearman: pd.DataFrame, context: str) -> pd.DataFrame:
    """
    df_spearman attendu: colonnes ['metric','spearman_rho'] (+ optionnel 'graph' / 'n' etc.)
    """
    if df_spearman.empty:
        return pd.DataFrame()
    if "spearman_rho" not in df_spearman.columns:
        return pd.DataFrame()

    # moyenne par métrique si plusieurs graphes (jouets)
    g = (
        df_spearman.groupby("metric", as_index=False)
        .agg(spearman_mean=("spearman_rho", "mean"), spearman_std=("spearman_rho", "std"))
        .sort_values("spearman_mean", ascending=False)
        .reset_index(drop=True)
    )
    g.insert(0, "context", context)
    g["rank"] = np.arange(1, len(g) + 1)
    return g


def _pivot_spearman(df_spearman: pd.DataFrame) -> Optional[pd.DataFrame]:
    """
    Pour les jouets: pivot graph x metric -> spearman_rho si 'graph' existe.
    """
    if df_spearman.empty:
        return None
    if "graph" not in df_spearman.columns:
        return None
    if "metric" not in df_spearman.columns or "spearman_rho" not in df_spearman.columns:
        return None
    return (
        df_spearman.pivot_table(index="graph", columns="metric", values="spearman_rho", aggfunc="mean")
        .sort_index()
        .reset_index()
    )


def main() -> None:
    p = argparse.ArgumentParser(
        description="Synthèse CSV: jouets + réel (nœuds). Produit des tableaux de comparaison (Spearman/top-k) et des CSV de synthèse."
    )
    p.add_argument("--toy-dir", type=str, default="toy_like_real_outputs")
    p.add_argument("--real-dir", type=str, default="real_ports_outputs")
    p.add_argument("--outdir", type=str, default="summary_tables")
    p.add_argument("--topks", type=str, default="25,50,100", help="Top-k à exporter pour les classements (réel).")
    args = p.parse_args()

    toy_dir = Path(args.toy_dir)
    real_dir = Path(args.real_dir)
    outdir = Path(args.outdir)
    _ensure_dir(outdir)

    ks = [int(x.strip()) for x in str(args.topks).split(",") if x.strip()]

    # -------------------------
    # Jouets: Spearman vs delta_eff (déjà agrégé par script)
    # -------------------------
    toy_spearman_path = toy_dir / "toy_all_spearman_vs_delta_eff.csv"
    toy_spearman = _read_csv(toy_spearman_path)
    toy_spearman.to_csv(outdir / "toy_spearman_vs_delta_eff.csv", index=False)

    toy_pivot = _pivot_spearman(toy_spearman)
    if toy_pivot is not None:
        toy_pivot.to_csv(outdir / "toy_spearman_pivot_graph_x_metric.csv", index=False)

    toy_winners = _winner_table_spearman(toy_spearman, context="toys_vs_delta_eff")
    if not toy_winners.empty:
        toy_winners.to_csv(outdir / "toy_winners_by_spearman.csv", index=False)

    # -------------------------
    # Réel: Spearman/top-k + top nodes per metric
    # -------------------------
    real_spearman = _read_csv(real_dir / "spearman_vs_delta_eff.csv")
    real_topk = _read_csv(real_dir / "topk_overlap_vs_delta_eff.csv")
    real_metrics_all = _read_csv(real_dir / "metrics_all_nodes.csv")

    real_spearman.to_csv(outdir / "real_spearman_vs_delta_eff.csv", index=False)
    real_topk.to_csv(outdir / "real_topk_overlap_vs_delta_eff.csv", index=False)

    # Résumé "winner" réel (juste tri décroissant)
    if "spearman_rho" in real_spearman.columns and "metric" in real_spearman.columns:
        real_rank = real_spearman.sort_values("spearman_rho", ascending=False).reset_index(drop=True).copy()
        real_rank.insert(0, "rank", np.arange(1, len(real_rank) + 1))
        real_rank.to_csv(outdir / "real_ranked_by_spearman.csv", index=False)

    # Top nodes par métrique (réel)
    metrics_of_interest = ["degree", "strength", "betweenness", "eigenvector", "cf_closeness"]
    top_nodes = _topk_nodes_table(real_metrics_all, metrics_of_interest, ks=ks, node_col="node")
    top_nodes.to_csv(outdir / "real_top_nodes_by_metric.csv", index=False)

    # Tableau comparatif final (concat "winners" toys + real)
    real_winners = _winner_table_spearman(real_spearman, context="real_vs_delta_eff")
    combined = pd.concat([toy_winners, real_winners], ignore_index=True)
    combined.to_csv(outdir / "combined_winners_by_spearman.csv", index=False)

    print("Wrote summary tables to:", outdir)
    print(" - toy_spearman_vs_delta_eff.csv")
    print(" - toy_winners_by_spearman.csv")
    print(" - real_spearman_vs_delta_eff.csv")
    print(" - real_ranked_by_spearman.csv")
    print(" - real_topk_overlap_vs_delta_eff.csv")
    print(" - real_top_nodes_by_metric.csv")
    print(" - combined_winners_by_spearman.csv")


if __name__ == "__main__":
    main()


