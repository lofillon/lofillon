from __future__ import annotations

import argparse
import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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


@dataclass(frozen=True)
class Scenario:
    name: str
    top_n_ports: int
    min_edge_flow: float
    ref_mode: str  # "default" | "include_art" | "only_art"


def _is_constant_numeric(col: pd.Series) -> Tuple[bool, Optional[float]]:
    y = pd.to_numeric(col, errors="coerce").to_numpy(dtype=float)
    y = y[np.isfinite(y)]
    if y.size == 0:
        return True, None
    if float(np.nanmax(y) - np.nanmin(y)) == 0.0:
        return True, float(y[0])
    return False, None


def run_one(
    *,
    m: Any,
    df: pd.DataFrame,
    scenario: Scenario,
    weight_basis: str,
    betw_k_sample: Optional[int],
    seed: int,
    n_sources: int,
    shortest_path_mode: str,
    outdir: Path,
    ref_top_degree: int,
    ref_top_betw: int,
    ref_max: int,
) -> pd.DataFrame:
    """
    Exécute un scénario et retourne un tableau long-format:
      (scenario, target, metric) -> spearman_rho + meta (n, m, etc.)
    """
    # 1) Build graph (df is pre-loaded once in main)
    keep_top = None if int(scenario.top_n_ports) <= 0 else int(scenario.top_n_ports)
    G = m.build_port_to_port_graph_global(
        df,
        weight_basis=str(weight_basis),
        keep_top_n_ports=keep_top,
        port_locations_weight_csv=None,
        min_edge_flow=float(scenario.min_edge_flow),
    )

    # Giant component
    if G.number_of_nodes() > 0 and not m.nx.is_connected(G):
        cc = max(m.nx.connected_components(G), key=len)
        Gc = G.subgraph(cc).copy()
    else:
        Gc = G

    # Ensure weights
    Gc = m.to_undirected_resistive_graph(Gc, m.MetricConfig(), sym_method="sum")

    # 2) Metrics
    df_classic = m.compute_classic_metrics(
        Gc,
        conductance_attr="cond",
        resistance_attr="cost",
        betw_k_sample=betw_k_sample,
        seed=int(seed),
    )
    cf_dict, Kbar = m.resistance_metrics_via_eigh(Gc, cond_attr="cond", giant_component_only=False)
    df_res = pd.DataFrame({"node": list(cf_dict.keys()), "cf_closeness": list(cf_dict.values())})
    df_res["Kbar"] = float(Kbar)
    df_metrics = df_classic.merge(df_res, on="node", how="left")

    # 3) Reference nodes subset (bounded)
    top_deg = df_metrics.sort_values("degree", ascending=False).head(int(ref_top_degree))["node"].tolist()
    top_betw = df_metrics.sort_values("betweenness", ascending=False).head(int(ref_top_betw))["node"].tolist()
    ref_nodes = list(set(top_deg) | set(top_betw))

    art_count = 0
    if scenario.ref_mode in {"include_art", "only_art"}:
        try:
            art = list(m.nx.articulation_points(Gc))
        except Exception:
            art = []
        art_count = len(art)
        if scenario.ref_mode == "only_art":
            # If there are no articulation points, this scenario is not informative for connectivity
            # and we keep it empty to keep the run fast.
            ref_nodes = art if art else []
        elif art:
            ref_nodes = list(set(ref_nodes) | set(art))

    ref_nodes = ref_nodes[: int(ref_max)]

    # 4) Neutral GT on subset
    sources = m.sample_sources(Gc, n_sources=int(n_sources), seed=int(seed))
    sp_weight = None if str(shortest_path_mode) == "hops" else "cost"
    df_gt = m.neutral_fragility_for_nodes(Gc, nodes_subset=ref_nodes, sources=sources, shortest_path_weight=sp_weight)
    df_ref = df_metrics[df_metrics["node"].isin(ref_nodes)].merge(df_gt, on="node", how="inner")

    const_eff, _ = _is_constant_numeric(df_ref["delta_eff"]) if "delta_eff" in df_ref.columns else (True, None)
    const_conn, _ = (
        _is_constant_numeric(df_ref["delta_conn_pairs"]) if "delta_conn_pairs" in df_ref.columns else (True, None)
    )

    # 5) Ranking comparisons
    metric_cols = ["degree", "strength", "betweenness", "eigenvector", "cf_closeness"]
    sp_eff, _ov_eff = m.evaluate_rankings(df_ref, metric_cols, target_col="delta_eff")
    sp_conn, _ov_conn = m.evaluate_rankings(df_ref, metric_cols, target_col="delta_conn_pairs")

    # 6) Exports per scenario
    sdir = outdir / scenario.name
    sdir.mkdir(parents=True, exist_ok=True)
    df_metrics.to_csv(sdir / "metrics_all_nodes.csv", index=False)
    df_ref.to_csv(sdir / "metrics_ref_with_neutral_gt.csv", index=False)
    sp_eff.to_csv(sdir / "spearman_vs_delta_eff.csv", index=False)
    sp_conn.to_csv(sdir / "spearman_vs_delta_conn_pairs.csv", index=False)

    # 7) Long-format summary rows
    rows: List[Dict[str, Any]] = []
    for _, r in sp_eff.iterrows():
        rows.append(
            {
                "scenario": scenario.name,
                "target": "delta_eff",
                "metric": r["metric"],
                "spearman_rho": r["spearman_rho"],
                "n_nodes": int(Gc.number_of_nodes()),
                "n_edges": int(Gc.number_of_edges()),
                "ref_size": int(len(df_ref)),
                "articulation_points": int(art_count),
                "delta_eff_constant": bool(const_eff),
                "delta_conn_pairs_constant": bool(const_conn),
                "top_n_ports": int(scenario.top_n_ports),
                "min_edge_flow": float(scenario.min_edge_flow),
                "ref_mode": scenario.ref_mode,
            }
        )
    for _, r in sp_conn.iterrows():
        rows.append(
            {
                "scenario": scenario.name,
                "target": "delta_conn_pairs",
                "metric": r["metric"],
                "spearman_rho": r["spearman_rho"],
                "n_nodes": int(Gc.number_of_nodes()),
                "n_edges": int(Gc.number_of_edges()),
                "ref_size": int(len(df_ref)),
                "articulation_points": int(art_count),
                "delta_eff_constant": bool(const_eff),
                "delta_conn_pairs_constant": bool(const_conn),
                "top_n_ports": int(scenario.top_n_ports),
                "min_edge_flow": float(scenario.min_edge_flow),
                "ref_mode": scenario.ref_mode,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    p = argparse.ArgumentParser(
        description="Stress-test suite: articulation focus + sparsification on the real port graph, with aggregated Spearman tables."
    )
    p.add_argument(
        "--csv",
        type=str,
        default="/Users/lapin/Desktop/Graph theory/SCRM-Resistance-distance/Global port supply-chains/Port_to_port_network/port_trade_network.csv",
        help="Chemin ABSOLU vers port_trade_network.csv",
    )
    p.add_argument("--weight-basis", choices=["tonnes", "value"], default="tonnes")
    p.add_argument("--betw-k-sample", type=int, default=120, help="Approx betweenness: nombre de sources k (<=0 pour exact).")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-sources", type=int, default=10)
    p.add_argument("--shortest-path-mode", choices=["hops", "cost"], default="cost")
    p.add_argument("--outdir", type=str, default="real_ports_stress_suite", help="Dossier de sortie.")
    p.add_argument(
        "--full-suite",
        action="store_true",
        help="Run the full (slower) 4-scenario suite. By default, runs a reduced suite aimed at < 1 hour.",
    )
    p.add_argument(
        "--scenario",
        type=str,
        default="",
        help="Run exactly ONE scenario by name (e.g., 'sparse_500_flow500'). Overrides --full-suite and the default suite.",
    )
    p.add_argument(
        "--list-scenarios",
        action="store_true",
        help="Print available scenario names and exit.",
    )
    p.add_argument("--ref-top-degree", type=int, default=120, help="Nb de top-degree nœuds à inclure dans le sous-ensemble testé.")
    p.add_argument("--ref-top-betw", type=int, default=120, help="Nb de top-betweenness nœuds à inclure dans le sous-ensemble testé.")
    p.add_argument("--ref-max", type=int, default=160, help="Taille max du sous-ensemble de nœuds testé (neutral GT).")
    args = p.parse_args()

    analysis_path = Path(__file__).parent / "Analysis-metrics-resistance.py"
    m = _load_analysis_module(analysis_path)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    betw_k = int(args.betw_k_sample) if int(args.betw_k_sample) > 0 else None
    csv_path = Path(args.csv)
    # Load once (saves time when running multiple scenarios)
    df = m.load_port_trade_network(csv_path)

    scenarios: Dict[str, Scenario] = {
        "articulation_only": Scenario(name="articulation_only", top_n_ports=2000, min_edge_flow=0.0, ref_mode="only_art"),
        "sparse_1200_flow0": Scenario(name="sparse_1200_flow0", top_n_ports=1200, min_edge_flow=0.0, ref_mode="include_art"),
        "sparse_800_flow100": Scenario(name="sparse_800_flow100", top_n_ports=800, min_edge_flow=100.0, ref_mode="include_art"),
        "sparse_500_flow500": Scenario(name="sparse_500_flow500", top_n_ports=500, min_edge_flow=500.0, ref_mode="include_art"),
    }

    if bool(args.list_scenarios):
        print("Available scenarios:")
        for k in sorted(scenarios.keys()):
            print(" -", k)
        return

    # Suite selection
    if str(args.scenario).strip():
        key = str(args.scenario).strip()
        if key not in scenarios:
            raise ValueError(f"Unknown --scenario '{key}'. Use --list-scenarios to see options.")
        suite = [scenarios[key]]
    elif bool(args.full_suite):
        suite = [scenarios[k] for k in ["articulation_only", "sparse_1200_flow0", "sparse_800_flow100", "sparse_500_flow500"]]
    else:
        suite = [scenarios[k] for k in ["sparse_800_flow100", "sparse_500_flow500"]]

    all_rows: List[pd.DataFrame] = []
    for sc in suite:
        df_rows = run_one(
            m=m,
            df=df,
            scenario=sc,
            weight_basis=str(args.weight_basis),
            betw_k_sample=betw_k,
            seed=int(args.seed),
            n_sources=int(args.n_sources),
            shortest_path_mode=str(args.shortest_path_mode),
            outdir=outdir,
            ref_top_degree=int(args.ref_top_degree),
            ref_top_betw=int(args.ref_top_betw),
            ref_max=int(args.ref_max),
        )
        all_rows.append(df_rows)
        print(f"[done] {sc.name} -> {outdir/sc.name}")

    df_all = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()
    df_all.to_csv(outdir / "stress_suite_spearman_summary.csv", index=False)
    print("\nWrote:")
    print(" -", outdir / "stress_suite_spearman_summary.csv")


if __name__ == "__main__":
    main()


