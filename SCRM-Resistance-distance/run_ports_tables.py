from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
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
        description="Build le graphe réel (ports) + calcule métriques classiques et résistance (eigh) + évaluation neutral, puis export CSV."
    )
    p.add_argument(
        "--csv",
        type=str,
        default="/Users/lapin/Desktop/Graph theory/SCRM-Resistance-distance/Global port supply-chains/Port_to_port_network/port_trade_network.csv",
        help="Chemin ABSOLU vers port_trade_network.csv",
    )
    p.add_argument("--top-n-ports", type=int, default=2000, help="Garde seulement les top-N ports (<=0 pour tout).")
    p.add_argument("--min-edge-flow", type=float, default=0.0, help="Ignore edges avec flow <= seuil.")
    p.add_argument("--weight-basis", choices=["tonnes", "value"], default="tonnes")

    # Classic metrics
    p.add_argument("--betw-k-sample", type=int, default=300, help="Approx betweenness: nombre de sources k (<=0 pour exact).")
    p.add_argument("--seed", type=int, default=0)

    # Neutral GT sampling
    p.add_argument("--ref-top-degree", type=int, default=200)
    p.add_argument("--ref-top-betw", type=int, default=200)
    p.add_argument("--ref-max", type=int, default=300, help="Taille max du sous-ensemble de nœuds à tester en suppression.")
    p.add_argument(
        "--ref-include-articulation",
        action="store_true",
        help="Ajoute les points d'articulation (cut-nodes) à l'ensemble de nœuds testés, si le graphe en possède.",
    )
    p.add_argument(
        "--ref-only-articulation",
        action="store_true",
        help="Ne teste que les points d'articulation (utile pour rendre delta_conn_pairs informatif).",
    )
    p.add_argument("--n-sources", type=int, default=20, help="Nb de sources pour estimer efficiency/aspl (neutral).")
    p.add_argument(
        "--shortest-path-mode",
        choices=["hops", "cost"],
        default="cost",
        help="Neutral GT: shortest paths en hops (non pondéré) ou cost (pondéré).",
    )

    # Output
    p.add_argument("--outdir", type=str, default="real_ports_outputs", help="Dossier de sortie (créé si besoin).")

    args = p.parse_args()

    analysis_path = Path(__file__).parent / "Analysis-metrics-resistance.py"
    m = _load_analysis_module(analysis_path)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # 1) Build graph
    df = m.load_port_trade_network(Path(args.csv))
    keep_top = None if int(args.top_n_ports) <= 0 else int(args.top_n_ports)
    G = m.build_port_to_port_graph_global(
        df,
        weight_basis=str(args.weight_basis),
        keep_top_n_ports=keep_top,
        port_locations_weight_csv=None,
        min_edge_flow=float(args.min_edge_flow),
    )

    print("nodes =", G.number_of_nodes())
    print("edges =", G.number_of_edges())

    # Work on giant component (important for Laplacian / eigen)
    if G.number_of_nodes() > 0 and not m.nx.is_connected(G):
        cc = max(m.nx.connected_components(G), key=len)
        Gc = G.subgraph(cc).copy()
        print("giant component nodes =", Gc.number_of_nodes(), "edges =", Gc.number_of_edges())
    else:
        Gc = G

    # Ensure consistent weights
    Gc = m.to_undirected_resistive_graph(Gc, m.MetricConfig(), sym_method="sum")

    # 2) Classic metrics table
    betw_k = None if int(args.betw_k_sample) > 0 else None
    # compute_classic_metrics expects Optional[int]; we pass k if >0 else None
    betw_k = int(args.betw_k_sample) if int(args.betw_k_sample) > 0 else None
    df_classic = m.compute_classic_metrics(
        Gc,
        conductance_attr="cond",
        resistance_attr="cost",
        betw_k_sample=betw_k,
        seed=int(args.seed),
    )

    # 3) Resistance metrics via eigh (exact but fast)
    cf_dict, Kbar = m.resistance_metrics_via_eigh(Gc, cond_attr="cond", giant_component_only=False)
    df_res = pd.DataFrame({"node": list(cf_dict.keys()), "cf_closeness": list(cf_dict.values())})
    df_res["Kbar"] = float(Kbar)

    df_metrics = df_classic.merge(df_res, on="node", how="left")

    # 4) Neutral ground truth on subset
    top_deg = df_metrics.sort_values("degree", ascending=False).head(int(args.ref_top_degree))["node"].tolist()
    top_betw = df_metrics.sort_values("betweenness", ascending=False).head(int(args.ref_top_betw))["node"].tolist()
    ref_nodes = list(set(top_deg) | set(top_betw))

    # Option: include / focus on articulation points (cut-nodes)
    if bool(args.ref_include_articulation) or bool(args.ref_only_articulation):
        try:
            art = list(m.nx.articulation_points(Gc))
        except Exception:
            art = []
        if art:
            if bool(args.ref_only_articulation):
                ref_nodes = art
            else:
                ref_nodes = list(set(ref_nodes) | set(art))
            print("articulation_points =", len(art))
        else:
            print("articulation_points = 0")
    if int(args.ref_max) > 0:
        ref_nodes = ref_nodes[: int(args.ref_max)]

    sources = m.sample_sources(Gc, n_sources=int(args.n_sources), seed=int(args.seed))
    sp_weight = None if str(args.shortest_path_mode) == "hops" else "cost"
    df_gt = m.neutral_fragility_for_nodes(Gc, nodes_subset=ref_nodes, sources=sources, shortest_path_weight=sp_weight)

    df_ref = df_metrics[df_metrics["node"].isin(ref_nodes)].merge(df_gt, on="node", how="inner")

    # 5) Compare rankings (classic + resistance cf_closeness)
    metric_cols = ["degree", "strength", "betweenness", "eigenvector", "cf_closeness"]
    sp_eff, ov_eff = m.evaluate_rankings(df_ref, metric_cols, target_col="delta_eff")
    sp_conn, ov_conn = m.evaluate_rankings(df_ref, metric_cols, target_col="delta_conn_pairs")

    # Helpful diagnostics: if delta_conn_pairs is constant, Spearman is undefined (NaN)
    if "delta_conn_pairs" in df_ref.columns and len(df_ref) > 0:
        y = pd.to_numeric(df_ref["delta_conn_pairs"], errors="coerce").to_numpy()
        y = y[np.isfinite(y)]
        if y.size > 0 and float(np.nanmax(y) - np.nanmin(y)) == 0.0:
            const_val = float(y[0])
            print(
                "\n[info] delta_conn_pairs is constant on the tested node subset "
                f"(all = {const_val}). Spearman vs delta_conn_pairs will be NaN.\n"
                "       This typically means removing any single tested node does not disconnect the giant component.\n"
                "       If you want a connectivity-based target to be informative, try one of:\n"
                "         - add articulation points to the tested set: --ref-include-articulation\n"
                "         - test only articulation points (if any):   --ref-only-articulation\n"
                "         - make the graph sparser: lower --top-n-ports and/or increase --min-edge-flow\n"
                "       Otherwise, use delta_eff as the main structural impact target.\n"
            )

    df_metrics.to_csv(outdir / "metrics_all_nodes.csv", index=False)
    df_ref.to_csv(outdir / "metrics_ref_with_neutral_gt.csv", index=False)
    sp_eff.to_csv(outdir / "spearman_vs_delta_eff.csv", index=False)
    ov_eff.to_csv(outdir / "topk_overlap_vs_delta_eff.csv", index=False)
    sp_conn.to_csv(outdir / "spearman_vs_delta_conn_pairs.csv", index=False)
    ov_conn.to_csv(outdir / "topk_overlap_vs_delta_conn_pairs.csv", index=False)

    print("\nWrote outputs to:", outdir)
    print(" - metrics_all_nodes.csv")
    print(" - metrics_ref_with_neutral_gt.csv")
    print(" - spearman_vs_delta_eff.csv")
    print(" - topk_overlap_vs_delta_eff.csv")
    print(" - spearman_vs_delta_conn_pairs.csv")
    print(" - topk_overlap_vs_delta_conn_pairs.csv")


if __name__ == "__main__":
    main()


