from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import pandas as pd


def _load_analysis_module(path: Path):
    """
    Charge Analysis-metrics-resistance.py même si le nom contient des tirets.
    """
    spec = importlib.util.spec_from_file_location("analysis_metrics_resistance", str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Impossible de charger le module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    p = argparse.ArgumentParser(
        description="Compare les graphes jouets avec les mêmes métriques que le graphe réel (classiques + cf_closeness via eigh + neutral GT)."
    )
    p.add_argument("--outdir", type=str, default="toy_like_real_outputs", help="Dossier de sortie (créé si besoin).")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-sources", type=int, default=20, help="Nb de sources pour estimer efficiency/aspl (neutral).")
    p.add_argument(
        "--shortest-path-mode",
        choices=["hops", "cost"],
        default="cost",
        help="Neutral GT: shortest paths en hops (non pondéré) ou cost (pondéré).",
    )
    args = p.parse_args()

    analysis_path = Path(__file__).parent / "Analysis-metrics-resistance.py"
    m = _load_analysis_module(analysis_path)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    toys = m.make_toy_graphs()
    all_spearman = []

    for name, G in toys.items():
        # Assure cond/cost cohérents
        G = m.to_undirected_resistive_graph(G, m.MetricConfig(), sym_method="sum")

        # Métriques classiques (mêmes colonnes que le script réel)
        df_classic = m.compute_classic_metrics(
            G, conductance_attr="cond", resistance_attr="cost", betw_k_sample=None, seed=int(args.seed)
        )

        # Métriques résistance (exactes mais rapides via eigh)
        cf, Kbar = m.resistance_metrics_via_eigh(G, cond_attr="cond", giant_component_only=True)
        df_res = pd.DataFrame({"node": list(cf.keys()), "cf_closeness": list(cf.values())})
        df_res["Kbar"] = float(Kbar)

        df_metrics = df_classic.merge(df_res, on="node", how="left")

        # Ground truth neutral (sur tous les nœuds ; jouets => petit)
        sources = m.sample_sources(G, n_sources=min(int(args.n_sources), G.number_of_nodes()), seed=int(args.seed))
        sp_weight = None if str(args.shortest_path_mode) == "hops" else "cost"
        df_gt = m.neutral_fragility_for_nodes(
            G, nodes_subset=list(G.nodes()), sources=sources, shortest_path_weight=sp_weight
        )
        df_ref = df_metrics.merge(df_gt, on="node", how="inner")

        # Comparaison Spearman + top-k overlap (mêmes métriques que réel)
        metric_cols = ["degree", "strength", "betweenness", "eigenvector", "cf_closeness"]
        sp_eff, ov_eff = m.evaluate_rankings(df_ref, metric_cols, target_col="delta_eff")

        # Exports par graphe
        df_metrics.to_csv(outdir / f"{name}_metrics.csv", index=False)
        df_ref.to_csv(outdir / f"{name}_metrics_with_neutral_gt.csv", index=False)
        sp_eff.to_csv(outdir / f"{name}_spearman_vs_delta_eff.csv", index=False)
        ov_eff.to_csv(outdir / f"{name}_topk_vs_delta_eff.csv", index=False)

        # Table globale
        sp_eff = sp_eff.copy()
        sp_eff.insert(0, "graph", name)
        sp_eff.insert(1, "n", int(G.number_of_nodes()))
        all_spearman.append(sp_eff)

    if all_spearman:
        pd.concat(all_spearman, ignore_index=True).to_csv(outdir / "toy_all_spearman_vs_delta_eff.csv", index=False)

    print("Wrote outputs to:", outdir)
    print(" - *_metrics.csv")
    print(" - *_metrics_with_neutral_gt.csv")
    print(" - *_spearman_vs_delta_eff.csv")
    print(" - *_topk_vs_delta_eff.csv")
    print(" - toy_all_spearman_vs_delta_eff.csv")


if __name__ == "__main__":
    main()


