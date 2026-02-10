from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import networkx as nx

try:
    # Optionnel (utilisé par le pipeline "neutral").
    from scipy.stats import spearmanr  # type: ignore
except Exception:  # pragma: no cover
    spearmanr = None  # type: ignore

try:
    # Optionnel (utile pour métriques résistance "exactes mais rapides" via eigen-decomp)
    from scipy.linalg import eigh  # type: ignore
except Exception:  # pragma: no cover
    eigh = None  # type: ignore


# -----------------------------
# Configuration / conventions
# -----------------------------


@dataclass(frozen=True)
class MetricConfig:
    # Attributs d'arêtes
    cost_attr: str = "cost"  # coût / résistance (shortest-path minimisent)
    cond_attr: str = "cond"  # conductance (Laplacien pondéré)

    # Stabilité numérique pour cond = 1/(cost + eps)
    eps: float = 1e-9

    # Déconnexion : distance "pénalisée" entre composantes
    # Si None, on dérive une pénalité automatiquement.
    disconnection_penalty: Optional[float] = None
    penalty_factor: float = 10.0  # utilisé si disconnection_penalty=None

    # Centralités optionnelles
    compute_pagerank: bool = False

    # Garde-fous calculatoires (pinv est O(n^3))
    max_n_for_pinv: int = 600


# -----------------------------
# Préparation du graphe
# -----------------------------


def to_undirected_resistive_graph(
    G: Union[nx.Graph, nx.DiGraph],
    cfg: MetricConfig,
    sym_method: str = "sum",
    drop_self_loops: bool = True,
) -> nx.Graph:
    """
    Retourne un graphe NON orienté avec 2 attributs par arête :
      - cfg.cond_attr (conductance)
      - cfg.cost_attr (coût = 1/(cond+eps) si non fourni)
    Si G est orienté, on symétrise les conductances (sum/avg/max).
    """
    if isinstance(G, nx.DiGraph):
        H = nx.Graph()
        H.add_nodes_from(G.nodes())

        def combine(a: float, b: float) -> float:
            if sym_method == "sum":
                return a + b
            if sym_method == "avg":
                return 0.5 * (a + b)
            if sym_method == "max":
                return max(a, b)
            raise ValueError("sym_method must be one of: 'sum', 'avg', 'max'")

        # On récupère cond/cost sur les arcs ; si cond absent mais cost présent, on dérive cond.
        for u, v, data in G.edges(data=True):
            if u == v and drop_self_loops:
                continue

            cost = data.get(cfg.cost_attr, None)
            cond = data.get(cfg.cond_attr, None)

            if cond is None:
                if cost is None:
                    # baseline
                    cost = 1.0
                cond = 1.0 / (float(cost) + cfg.eps)
            else:
                cond = float(cond)

            # On agrège (u,v) et (v,u)
            if H.has_edge(u, v):
                prev = H[u][v][cfg.cond_attr]
                H[u][v][cfg.cond_attr] = combine(prev, cond)
            else:
                H.add_edge(u, v, **{cfg.cond_attr: cond})

        # Déduire cost depuis cond (cohérence Laplacien/shortest-path)
        for u, v, data in H.edges(data=True):
            cond = float(data[cfg.cond_attr])
            data[cfg.cost_attr] = 1.0 / (cond + cfg.eps)

        return H

    # Déjà non orienté : on s'assure que cost/cond existent et sont cohérents
    H = G.copy()
    if drop_self_loops:
        H.remove_edges_from(list(nx.selfloop_edges(H)))

    for u, v, data in H.edges(data=True):
        cost = data.get(cfg.cost_attr, None)
        cond = data.get(cfg.cond_attr, None)

        if cost is None and cond is None:
            cost = 1.0
            cond = 1.0 / (cost + cfg.eps)
        elif cond is None and cost is not None:
            cond = 1.0 / (float(cost) + cfg.eps)
        elif cost is None and cond is not None:
            cost = 1.0 / (float(cond) + cfg.eps)
        else:
            cost = float(cost)
            cond = float(cond)

        data[cfg.cost_attr] = float(cost)
        data[cfg.cond_attr] = float(cond)

    return H


def default_disconnection_penalty(G: nx.Graph, cfg: MetricConfig) -> float:
    """
    Choix simple et stable d'une pénalité pour paires déconnectées.
    On utilise : penalty_factor * median(cost) * n
    """
    costs = []
    for _, _, data in G.edges(data=True):
        c = float(data.get(cfg.cost_attr, 1.0))
        if np.isfinite(c) and c > 0:
            costs.append(c)
    med = float(np.median(costs)) if costs else 1.0
    return float(cfg.penalty_factor * med * max(2, G.number_of_nodes()))


# -----------------------------
# Résistance effective (R), Kbar, current-flow closeness
# -----------------------------


def resistance_matrix_and_Kbar(
    G: nx.Graph,
    cfg: MetricConfig,
    return_R: bool = True,
) -> Tuple[list, Optional[np.ndarray], float]:
    """
    Calcule:
      - R(u,v) (matrice) via L^+ (par composante), et pénalise les paires déconnectées
      - Kbar = moyenne de R(u,v) sur u!=v (avec pénalité pour composantes différentes)

    Retour:
      nodes: liste ordonnée des nœuds
      R: matrice (n,n) si return_R=True sinon None
      Kbar: float
    """
    nodes = list(G.nodes())
    n = len(nodes)
    if n <= 1:
        return nodes, (np.zeros((n, n)) if return_R else None), 0.0
    if n > cfg.max_n_for_pinv:
        raise ValueError(
            f"n={n} trop grand pour pinv (max_n_for_pinv={cfg.max_n_for_pinv}). "
            "Pour >~600, il faut des approximations (sampling/solveurs laplaciens)."
        )

    penalty = cfg.disconnection_penalty
    if penalty is None:
        penalty = default_disconnection_penalty(G, cfg)

    idx = {v: i for i, v in enumerate(nodes)}
    R = np.full((n, n), penalty, dtype=float) if return_R else None

    # On calcule R intra-composante uniquement (Laplacien sur chaque composante)
    for comp in nx.connected_components(G):
        comp = list(comp)
        m = len(comp)
        if m == 1:
            if return_R:
                i = idx[comp[0]]
                R[i, i] = 0.0
            continue

        H = G.subgraph(comp).copy()
        L = nx.laplacian_matrix(H, weight=cfg.cond_attr).astype(float).toarray()
        L_pinv = np.linalg.pinv(L)

        diag = np.diag(L_pinv)
        R_sub = diag[:, None] + diag[None, :] - 2.0 * L_pinv
        np.fill_diagonal(R_sub, 0.0)

        if return_R:
            for a, u in enumerate(comp):
                iu = idx[u]
                for b, v in enumerate(comp):
                    iv = idx[v]
                    R[iu, iv] = float(R_sub[a, b])

    if return_R:
        np.fill_diagonal(R, 0.0)
        Kbar = float(R.sum() / (n * (n - 1)))
        return nodes, R, Kbar

    # Si on ne retourne pas R, on doit quand même calculer Kbar.
    # Ici, on recalcule Kbar sans matrice complète en accumulant :
    # - paires intra-composantes via R_sub.sum()
    # - paires inter-composantes via penalty
    intra_sum = 0.0
    intra_pairs = 0

    for comp in nx.connected_components(G):
        m = len(comp)
        if m <= 1:
            continue
        H = G.subgraph(comp).copy()
        L = nx.laplacian_matrix(H, weight=cfg.cond_attr).astype(float).toarray()
        L_pinv = np.linalg.pinv(L)
        diag = np.diag(L_pinv)
        R_sub = diag[:, None] + diag[None, :] - 2.0 * L_pinv
        np.fill_diagonal(R_sub, 0.0)
        intra_sum += float(R_sub.sum())
        intra_pairs += m * (m - 1)

    total_pairs = n * (n - 1)
    inter_pairs = total_pairs - intra_pairs
    total_sum = intra_sum + float(inter_pairs) * penalty
    Kbar = float(total_sum / total_pairs)
    return nodes, None, Kbar


def current_flow_closeness_from_R(nodes: Sequence, R: np.ndarray) -> Dict:
    """
    C_cf-clo(v) = 1 / sum_{u != v} R(u,v)
    (R inclut une pénalité pour paires déconnectées si nécessaire)
    """
    cf = {}
    for i, v in enumerate(nodes):
        denom = float(R[i, :].sum())  # diag=0
        cf[v] = 1.0 / denom if denom > 0 else 0.0
    return cf


def resistance_metrics_via_eigh(
    G: nx.Graph,
    cond_attr: str = "cond",
    eps0: float = 1e-12,
    giant_component_only: bool = True,
) -> Tuple[Dict[Any, float], float]:
    """
    Calcule des métriques résistance EXACTES sans pseudo-inverse dense (pinv),
    via diagonalisation du Laplacien symétrique (eigh).

    Retour:
      - cf_closeness: dict node -> 1 / sum_u R(u,v)
      - Kbar: moyenne des résistances effectives sur u!=v

    Identités (graphe connexe):
      sum_u R(v,u) = n*(L^+)_{vv} + trace(L^+)
      Kbar = 2*trace(L^+) / (n-1)
    """
    if eigh is None:
        raise ImportError("scipy n'est pas disponible (eigh). Installe scipy pour resistance_metrics_via_eigh().")

    H = G
    if giant_component_only and H.number_of_nodes() > 0 and not nx.is_connected(H):
        cc = max(nx.connected_components(H), key=len)
        H = H.subgraph(cc).copy()

    nodes = list(H.nodes())
    n = len(nodes)
    if n <= 1:
        return {v: 0.0 for v in nodes}, 0.0

    L = nx.laplacian_matrix(H, weight=cond_attr).astype(float).toarray()
    w, U = eigh(L)

    # retirer le(s) mode(s) zéro (lambda≈0)
    mask = w > eps0
    w = w[mask]
    U = U[:, mask]
    if w.size == 0:
        # graphe sans arêtes (ou tous poids ~0)
        return {v: 0.0 for v in nodes}, float("inf")

    inv_w = 1.0 / w
    trace_Lplus = float(inv_w.sum())

    # diag(L^+) = sum_k U[i,k]^2 / lambda_k
    diag_Lplus = (U * U) @ inv_w

    denom = n * diag_Lplus + trace_Lplus
    cf = 1.0 / denom

    Kbar = float(2.0 * trace_Lplus / (n - 1))
    return {nodes[i]: float(cf[i]) for i in range(n)}, Kbar


# -----------------------------
# Référence indépendante : efficience globale (shortest-path)
# -----------------------------


def global_efficiency_cost(G: nx.Graph, cfg: MetricConfig) -> float:
    """
    Eff(G) = (1/(n(n-1))) * sum_{i!=j} 1/d(i,j), où d est shortest-path sur cost.
    Si i,j déconnectés => contribution 0.
    """
    nodes = list(G.nodes())
    n = len(nodes)
    if n <= 1:
        return 0.0

    # all_pairs_dijkstra_path_length renvoie seulement les nœuds atteignables
    lengths = dict(nx.all_pairs_dijkstra_path_length(G, weight=cfg.cost_attr))

    s = 0.0
    for i in nodes:
        li = lengths.get(i, {})
        for j in nodes:
            if i == j:
                continue
            dij = li.get(j, np.inf)
            if np.isfinite(dij) and dij > 0:
                s += 1.0 / float(dij)
            # sinon +0
    return float(s / (n * (n - 1)))


# -----------------------------
# Centralités classiques (cohérentes avec cost vs cond)
# -----------------------------


def classic_centralities(G: nx.Graph, cfg: MetricConfig) -> Dict[str, Dict]:
    """
    - betweenness/closeness : shortest-path sur cost
    - eigenvector/pagerank/weighted_degree : sur cond
    - degree_centrality : non pondéré (NetworkX)
    """
    res: Dict[str, Dict] = {}
    nodes = list(G.nodes())

    res["deg"] = nx.degree_centrality(G)
    res["deg_w"] = {v: float(d) for v, d in G.degree(weight=cfg.cond_attr)}

    res["betweenness"] = nx.betweenness_centrality(G, weight=cfg.cost_attr, normalized=True)

    # closeness sur coûts (sinon tu mesures une autre chose)
    res["closeness"] = nx.closeness_centrality(G, distance=cfg.cost_attr)

    # eigenvector centrality : peut être instable si composantes / poids extrêmes
    try:
        res["eigenvector"] = nx.eigenvector_centrality_numpy(G, weight=cfg.cond_attr)
    except Exception:
        res["eigenvector"] = {v: 0.0 for v in nodes}

    if cfg.compute_pagerank:
        try:
            res["pagerank"] = nx.pagerank(G, weight=cfg.cond_attr)
        except Exception:
            res["pagerank"] = {v: 0.0 for v in nodes}

    return res


# -----------------------------
# Impacts par suppression (delta_eff, resK, delta_Kbar)
# -----------------------------


def node_removal_impacts(
    G: nx.Graph,
    cfg: MetricConfig,
    nodes_to_remove: Optional[Iterable] = None,
    compute_internal_kirchhoff: bool = True,
) -> Dict[str, Dict]:
    """
    Calcule:
      - delta_eff(v) = Eff(G) - Eff(G\\ v)  (référence indépendante)
      - resK(v)      = Kbar(G\\ v) - Kbar(G) (métrique candidate)
      - delta_Kbar(v)= idem que resK, si compute_internal_kirchhoff=True
        (utile comme "référence interne" Kirchhoff)

    Retourne un dict de dicts, clé -> {node -> value}.
    """
    nodes = list(G.nodes())
    if nodes_to_remove is None:
        nodes_to_remove = nodes
    nodes_to_remove = list(nodes_to_remove)

    # Base
    base_eff = global_efficiency_cost(G, cfg)
    _, base_R, base_Kbar = resistance_matrix_and_Kbar(G, cfg, return_R=True)

    impacts: Dict[str, Dict] = {
        "delta_eff": {},
        "resK": {},
    }
    if compute_internal_kirchhoff:
        impacts["delta_Kbar"] = {}

    for v in nodes_to_remove:
        H = G.copy()
        if v not in H:
            continue
        H.remove_node(v)

        # Eff après suppression
        eff_minus = global_efficiency_cost(H, cfg) if H.number_of_nodes() > 1 else 0.0
        impacts["delta_eff"][v] = float(base_eff - eff_minus)

        # Kbar après suppression (avec pénalité si déconnecté)
        if H.number_of_nodes() <= 1:
            Kbar_minus = 0.0
        else:
            _, _, Kbar_minus = resistance_matrix_and_Kbar(H, cfg, return_R=False)

        resK = float(Kbar_minus - base_Kbar)
        impacts["resK"][v] = resK
        if compute_internal_kirchhoff:
            impacts["delta_Kbar"][v] = resK

    return impacts


# -----------------------------
# Pipeline principal : DataFrame métriques + références
# -----------------------------


def compute_metrics_dataframe(
    G_in: Union[nx.Graph, nx.DiGraph],
    cfg: MetricConfig = MetricConfig(),
    sym_method: str = "sum",
    nodes_to_remove: Optional[Iterable] = None,
    compute_internal_kirchhoff: bool = True,
) -> pd.DataFrame:
    """
    Produit un DataFrame par nœud avec :
      - centralités classiques (deg, deg_w, betweenness, closeness, eigenvector, +pagerank optionnel)
      - cf_closeness (current-flow) via R(u,v)
      - Kbar global
      - delta_eff (référence indépendante) / resK (Kirchhoff par suppression) / delta_Kbar (optionnel)

    Remarque : G_in peut être orienté ; on symétrise pour les métriques résistance-distance.
    """
    G = to_undirected_resistive_graph(G_in, cfg, sym_method=sym_method)

    nodes, R, Kbar = resistance_matrix_and_Kbar(G, cfg, return_R=True)
    cf = current_flow_closeness_from_R(nodes, R)

    centr = classic_centralities(G, cfg)
    impacts = node_removal_impacts(
        G, cfg, nodes_to_remove=nodes_to_remove, compute_internal_kirchhoff=compute_internal_kirchhoff
    )

    rows = []
    for v in nodes:
        row = {
            "node": v,
            "Kbar": float(Kbar),
            "deg": float(centr["deg"].get(v, 0.0)),
            "deg_w": float(centr["deg_w"].get(v, 0.0)),
            "betweenness": float(centr["betweenness"].get(v, 0.0)),
            "closeness": float(centr["closeness"].get(v, 0.0)),
            "eigenvector": float(centr["eigenvector"].get(v, 0.0)),
            "cf_closeness": float(cf.get(v, 0.0)),
            # Références / impacts par suppression
            "delta_eff": float(impacts["delta_eff"].get(v, np.nan)),
            "resK": float(impacts["resK"].get(v, np.nan)),
        }
        if cfg.compute_pagerank:
            row["pagerank"] = float(centr["pagerank"].get(v, 0.0))
        if compute_internal_kirchhoff:
            row["delta_Kbar"] = float(impacts["delta_Kbar"].get(v, np.nan))
        rows.append(row)

    return pd.DataFrame(rows)


# -----------------------------
# Comparaisons : rangs, top-k
# -----------------------------


def rank_correlations(
    df: pd.DataFrame,
    reference_col: str,
    metric_cols: Sequence[str],
    method: str = "spearman",
) -> pd.DataFrame:
    """
    Corrélations de rang entre une référence (ex. delta_eff) et des métriques.
    """
    cols = list(metric_cols) + [reference_col]
    sub = df[cols].dropna()
    if sub.empty:
        return pd.DataFrame({"metric": metric_cols, f"{method}_rho": [np.nan] * len(metric_cols)})

    corr = sub.corr(method=method)
    out = corr[reference_col].drop(reference_col)
    return out.to_frame(name=f"{method}_rho").reset_index().rename(columns={"index": "metric"})


def topk_overlap(df: pd.DataFrame, reference_col: str, metric_col: str, k: int) -> float:
    """
    Overlap top-k entre la référence et une métrique.
    Score élevé => la métrique retrouve les mêmes nœuds critiques.
    """
    sub = df[[reference_col, metric_col, "node"]].dropna()
    if sub.empty:
        return float("nan")

    ref_top = set(sub.sort_values(reference_col, ascending=False).head(k)["node"])
    met_top = set(sub.sort_values(metric_col, ascending=False).head(k)["node"])
    return float(len(ref_top & met_top) / max(1, k))


def evaluate_metrics(
    df: pd.DataFrame,
    reference_col: str = "delta_eff",
    metric_cols: Optional[Sequence[str]] = None,
    corr_methods: Sequence[str] = ("spearman", "kendall"),
    topk_list: Sequence[int] = (5, 10, 20),
) -> pd.DataFrame:
    """
    Renvoie un tableau synthétique :
      - corrélations de rang (Spearman/Kendall)
      - overlaps top-k
    """
    if metric_cols is None:
        metric_cols = [
            "deg",
            "deg_w",
            "betweenness",
            "closeness",
            "eigenvector",
            "cf_closeness",
            "resK",
        ]
        if "pagerank" in df.columns:
            metric_cols = list(metric_cols) + ["pagerank"]

    # Corrélations
    corr_frames = [rank_correlations(df, reference_col, metric_cols, method=m) for m in corr_methods]
    corr_df = corr_frames[0]
    for other in corr_frames[1:]:
        corr_df = corr_df.merge(other, on="metric", how="outer")

    # Top-k overlaps
    for k in topk_list:
        corr_df[f"top{k}_overlap"] = [topk_overlap(df, reference_col, met, k) for met in corr_df["metric"]]

    sort_col = "spearman_rho" if "spearman_rho" in corr_df.columns else corr_df.columns[1]
    return corr_df.sort_values(sort_col, ascending=False).reset_index(drop=True)


# -----------------------------
# Petit helper : normalisation min-max (optionnel)
# -----------------------------


def minmax_normalize_columns(df: pd.DataFrame, cols: Sequence[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c not in out.columns:
            continue
        x = out[c].astype(float)
        lo, hi = float(np.nanmin(x)), float(np.nanmax(x))
        if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
            out[c] = (x - lo) / (hi - lo)
        else:
            out[c] = 0.0
    return out


# -----------------------------
# Graphes jouets
# -----------------------------


def make_toy_graphs() -> Dict[str, nx.Graph]:
    toys: Dict[str, nx.Graph] = {}

    # 1) Deux triangles reliés par un pont
    G1 = nx.Graph()
    G1.add_edges_from(
        [
            ("A", "B"),
            ("B", "C"),
            ("C", "A"),
            ("D", "E"),
            ("E", "F"),
            ("F", "D"),
            ("C", "D"),
        ]
    )
    toys["two_triangles_bridge"] = G1

    # 2) Carré + diagonales (hyper redondant)
    G2 = nx.Graph()
    G2.add_edges_from([(1, 2), (2, 3), (3, 4), (4, 1), (1, 3), (2, 4)])
    toys["square_with_diagonals"] = G2

    # 3) Chaîne (line)
    G3 = nx.path_graph(10)
    toys["path_10"] = G3

    # 4) Étoile
    G4 = nx.star_graph(9)  # centre=0, feuilles 1..9
    toys["star_10"] = G4

    # 5) Deux communautés denses + quelques ponts
    G5 = nx.Graph()
    left = [f"L{i}" for i in range(6)]
    right = [f"R{i}" for i in range(6)]
    for i in range(len(left)):
        for j in range(i + 1, len(left)):
            G5.add_edge(left[i], left[j])
    for i in range(len(right)):
        for j in range(i + 1, len(right)):
            G5.add_edge(right[i], right[j])
    G5.add_edges_from([("L0", "R0"), ("L1", "R1")])  # 2 ponts
    toys["two_cliques_two_bridges"] = G5

    return toys


# -----------------------------
# Graphes synthétiques
# -----------------------------


@dataclass(frozen=True)
class SyntheticSpec:
    family: str
    n: int
    params: Dict[str, Any]


def generate_synthetic_graph(spec: SyntheticSpec, seed: int) -> nx.Graph:
    fam = spec.family.lower()
    n = int(spec.n)
    p = spec.params

    if fam == "er":  # Erdős–Rényi
        prob = float(p.get("p", 0.05))
        return nx.erdos_renyi_graph(n=n, p=prob, seed=seed)

    if fam == "ws":  # Watts–Strogatz
        k = int(p.get("k", 6))
        beta = float(p.get("beta", 0.2))
        return nx.watts_strogatz_graph(n=n, k=k, p=beta, seed=seed)

    if fam == "ba":  # Barabási–Albert
        m = int(p.get("m", 3))
        return nx.barabasi_albert_graph(n=n, m=m, seed=seed)

    if fam == "tree":  # arbre équilibré approx n via hauteur
        r = int(p.get("r", 2))
        h = int(p.get("h", 6))
        return nx.balanced_tree(r=r, h=h)

    raise ValueError(f"Unknown family: {spec.family}")


# -----------------------------
# Pondération optionnelle (temps/risque proxy)
# -----------------------------


def assign_random_costs(
    G: nx.Graph,
    cost_attr: str = "cost",
    mode: str = "lognormal",
    seed: int = 0,
    normalize_median: bool = True,
    eps: float = 1e-9,
) -> nx.Graph:
    """
    Ajoute un coût positif par arête (cost_attr). La conductance sera dérivée plus tard
    par to_undirected_resistive_graph() si besoin.
    """
    rng = np.random.default_rng(seed)
    H = G.copy()

    costs: List[float] = []
    for u, v in H.edges():
        if mode == "lognormal":
            c = float(rng.lognormal(mean=0.0, sigma=0.5))
        elif mode == "uniform":
            c = float(rng.uniform(0.5, 1.5))
        elif mode == "exp":
            c = float(rng.exponential(scale=1.0) + eps)
        else:
            raise ValueError("mode must be one of: lognormal, uniform, exp")
        H[u][v][cost_attr] = c
        costs.append(c)

    if normalize_median and costs:
        med = float(np.median(costs))
        if med > 0:
            for u, v in H.edges():
                H[u][v][cost_attr] = float(H[u][v][cost_attr] / med)

    return H


# -----------------------------
# Choix des nœuds à supprimer (pour limiter le coût)
# -----------------------------


def choose_nodes_to_remove(
    G: nx.Graph,
    mode: str = "all",
    k: int = 100,
    seed: int = 0,
    weight_for_degree: Optional[str] = None,
) -> Optional[List[Any]]:
    """
    Retourne une liste de nœuds à supprimer pour calculer delta_eff/resK.
    - "all" : tous les nœuds (coûteux dès que n grand)
    - "random_k" : échantillon aléatoire de taille k
    - "top_degree_k" : top-k par degré (pondéré si weight_for_degree donné)
    - "top_cond_degree_k" : raccourci pour top-k sur degré pondéré par 'cond'
    """
    nodes = list(G.nodes())
    n = len(nodes)
    if mode == "all":
        return None  # convention: compute_metrics_dataframe => tous les nœuds

    if mode == "random_k":
        rng = np.random.default_rng(seed)
        kk = min(k, n)
        return list(rng.choice(nodes, size=kk, replace=False))

    if mode == "top_degree_k":
        degs = dict(G.degree(weight=weight_for_degree))
        return sorted(nodes, key=lambda v: degs.get(v, 0.0), reverse=True)[: min(k, n)]

    if mode == "top_cond_degree_k":
        degs = dict(G.degree(weight="cond"))
        return sorted(nodes, key=lambda v: degs.get(v, 0.0), reverse=True)[: min(k, n)]

    raise ValueError("mode must be one of: all, random_k, top_degree_k, top_cond_degree_k")


# -----------------------------
# Exécution : 1 graphe => df + table evaluation
# -----------------------------


def run_one_graph(
    G: nx.Graph,
    name: str,
    cfg: MetricConfig,
    reference_col: str = "delta_eff",
    nodes_to_remove: Optional[Iterable[Any]] = None,
    compute_internal_kirchhoff: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Retourne:
      - df_node : DataFrame par nœud
      - df_eval : synthèse par métrique (corrélations + top-k overlap)
    """
    df_node = compute_metrics_dataframe(
        G,
        cfg=cfg,
        nodes_to_remove=nodes_to_remove,
        compute_internal_kirchhoff=compute_internal_kirchhoff,
    )

    metric_cols = [
        "deg",
        "deg_w",
        "betweenness",
        "closeness",
        "eigenvector",
        "cf_closeness",
        "resK",
    ]
    if "pagerank" in df_node.columns:
        metric_cols.append("pagerank")

    df_eval = evaluate_metrics(
        df_node,
        reference_col=reference_col,
        metric_cols=metric_cols,
        corr_methods=("spearman", "kendall"),
        topk_list=(5, 10, 20),
    )

    df_eval.insert(0, "graph_name", name)
    df_eval.insert(1, "n", int(df_node.shape[0]))
    return df_node, df_eval


# -----------------------------
# Suite : jouets
# -----------------------------


def run_toy_suite(cfg: MetricConfig) -> Tuple[Dict[str, pd.DataFrame], pd.DataFrame]:
    toys = make_toy_graphs()
    dfs_nodes: List[Tuple[str, pd.DataFrame]] = []
    evals: List[pd.DataFrame] = []

    for name, G in toys.items():
        t0 = time.perf_counter()
        df_node, df_eval = run_one_graph(
            G,
            name=name,
            cfg=cfg,
            reference_col="delta_eff",
            nodes_to_remove=None,  # tous (petit n)
            compute_internal_kirchhoff=True,
        )
        dt = time.perf_counter() - t0
        df_eval["runtime_s"] = dt
        evals.append(df_eval)
        dfs_nodes.append((name, df_node))

    df_eval_all = pd.concat(evals, ignore_index=True)
    df_nodes_dict = {name: df for name, df in dfs_nodes}
    return df_nodes_dict, df_eval_all


# -----------------------------
# Suite : synthétiques + agrégation
# -----------------------------


def run_synthetic_suite(
    specs: List[SyntheticSpec],
    cfg: MetricConfig,
    seeds: List[int],
    reference_col: str = "delta_eff",
    weight_mode: str = "baseline",  # "baseline" ou "random_costs"
    random_cost_mode: str = "lognormal",
    nodes_to_remove_mode: str = "all",
    nodes_to_remove_k: int = 100,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Retourne:
      - per_run: une ligne par (spec, seed, metric)
      - agg: agrégation par (family, n, params, metric)
    """
    per_run_rows: List[Dict[str, Any]] = []

    for spec in specs:
        for seed in seeds:
            G = generate_synthetic_graph(spec, seed=seed)

            # pondération optionnelle
            if weight_mode == "random_costs":
                G = assign_random_costs(
                    G,
                    cost_attr=cfg.cost_attr,
                    mode=random_cost_mode,
                    seed=seed,
                    normalize_median=True,
                    eps=cfg.eps,
                )

            # IMPORTANT: nodes_to_remove se choisit après passage par to_undirected_resistive_graph,
            # car on peut utiliser 'cond' pour top_cond_degree_k.
            G_prepared = to_undirected_resistive_graph(G, cfg, sym_method="sum")

            nodes_to_remove = choose_nodes_to_remove(
                G_prepared,
                mode=nodes_to_remove_mode,
                k=nodes_to_remove_k,
                seed=seed,
                weight_for_degree=None,
            )

            t0 = time.perf_counter()
            df_node = compute_metrics_dataframe(
                G_prepared,
                cfg=cfg,
                nodes_to_remove=nodes_to_remove,
                compute_internal_kirchhoff=True,
            )
            df_eval = evaluate_metrics(
                df_node,
                reference_col=reference_col,
                metric_cols=[
                    "deg",
                    "deg_w",
                    "betweenness",
                    "closeness",
                    "eigenvector",
                    "cf_closeness",
                    "resK",
                ]
                + (["pagerank"] if "pagerank" in df_node.columns else []),
                corr_methods=("spearman", "kendall"),
                topk_list=(5, 10, 20),
            )
            dt = time.perf_counter() - t0

            for _, row in df_eval.iterrows():
                per_run_rows.append(
                    {
                        "family": spec.family,
                        "n": spec.n,
                        "params": dict(spec.params),
                        "seed": seed,
                        "metric": row["metric"],
                        "spearman_rho": row.get("spearman_rho", np.nan),
                        "kendall_rho": row.get("kendall_rho", np.nan),
                        "top5_overlap": row.get("top5_overlap", np.nan),
                        "top10_overlap": row.get("top10_overlap", np.nan),
                        "top20_overlap": row.get("top20_overlap", np.nan),
                        "runtime_s": dt,
                        "reference": reference_col,
                        "weight_mode": weight_mode,
                        "nodes_removed_mode": nodes_to_remove_mode,
                        "nodes_removed_k": (None if nodes_to_remove is None else len(nodes_to_remove)),
                    }
                )

    per_run = pd.DataFrame(per_run_rows)

    def params_to_str(p: Dict[str, Any]) -> str:
        items = sorted(p.items(), key=lambda x: x[0])
        return ",".join([f"{k}={v}" for k, v in items])

    per_run["params_str"] = per_run["params"].apply(params_to_str)

    agg = (
        per_run.groupby(["family", "n", "params_str", "metric"], as_index=False)
        .agg(
            spearman_mean=("spearman_rho", "mean"),
            spearman_std=("spearman_rho", "std"),
            kendall_mean=("kendall_rho", "mean"),
            kendall_std=("kendall_rho", "std"),
            top5_mean=("top5_overlap", "mean"),
            top5_std=("top5_overlap", "std"),
            top10_mean=("top10_overlap", "mean"),
            top10_std=("top10_overlap", "std"),
            top20_mean=("top20_overlap", "mean"),
            top20_std=("top20_overlap", "std"),
            runtime_mean=("runtime_s", "mean"),
        )
        .sort_values(["family", "n", "params_str", "spearman_mean"], ascending=[True, True, True, False])
        .reset_index(drop=True)
    )

    return per_run, agg


# -----------------------------
# Exemple d'utilisation
# -----------------------------


def example_run_all():
    cfg = MetricConfig(
        compute_pagerank=False,
        # Doit être >= au plus grand graphe de la suite (ex: tree r=2,h=7 => 255 nœuds).
        # pinv est O(n^3) : augmenter cette valeur augmente vite le temps.
        max_n_for_pinv=300,
    )

    toy_nodes, toy_eval = run_toy_suite(cfg)
    print("=== Toy suite (evaluation) ===")
    print(toy_eval.head(30))

    specs = [
        SyntheticSpec("er", 100, {"p": 0.05}),
        SyntheticSpec("er", 200, {"p": 0.03}),
        SyntheticSpec("ws", 100, {"k": 6, "beta": 0.2}),
        SyntheticSpec("ws", 200, {"k": 8, "beta": 0.1}),
        SyntheticSpec("ba", 100, {"m": 3}),
        SyntheticSpec("ba", 200, {"m": 3}),
        # Arbre: taille via h (ex: r=2,h=7 => 255 nœuds)
        SyntheticSpec("tree", 255, {"r": 2, "h": 7}),
    ]
    seeds = [0, 1, 2, 3, 4]

    per_run, agg = run_synthetic_suite(
        specs=specs,
        cfg=cfg,
        seeds=seeds,
        reference_col="delta_eff",
        weight_mode="baseline",  # ou "random_costs"
        random_cost_mode="lognormal",  # si weight_mode="random_costs"
        nodes_to_remove_mode="random_k",
        nodes_to_remove_k=80,
    )

    print("\n=== Synthetic suite (aggregated) ===")
    print(agg.head(40))
    return toy_nodes, toy_eval, per_run, agg


# -----------------------------
# Cas réel (ports) : builder port -> port (MRIO-like)
# -----------------------------


def load_port_trade_network(csv_path: Union[str, "Path"]) -> pd.DataFrame:
    """
    Charge le CSV "port_trade_network.csv" (format de ton dataset), et fait un nettoyage minimal.
    """
    from pathlib import Path as _Path

    csv_path = _Path(csv_path)
    df = pd.read_csv(csv_path)

    # Harmonise quelques types (évite les surprises)
    for col in [
        "q_sea_flow",
        "v_sea_flow",
        "q_sea_flow_sum",
        "v_sea_flow_sum",
        "q_share_port",
        "v_share_port",
        "Industries",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Colonnes minimales attendues
    required = {"id", "flow", "iso3_O", "iso3_D", "Industries", "q_sea_flow", "q_sea_flow_sum"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV incomplet : colonnes manquantes = {sorted(missing)}")

    df = df.dropna(subset=["id", "flow", "iso3_O", "iso3_D", "Industries", "q_sea_flow", "q_sea_flow_sum"])
    df = df[df["q_sea_flow_sum"] > 0]
    df["id"] = df["id"].astype(str)
    return df


def attach_port_attributes(
    G: nx.Graph,
    port_locations_weight_csv: Optional[Union[str, "Path"]] = None,
    port_locations_value_csv: Optional[Union[str, "Path"]] = None,
    port_import_multiplier_csv: Optional[Union[str, "Path"]] = None,
    port_output_multiplier_csv: Optional[Union[str, "Path"]] = None,
) -> None:
    """
    Ajoute des attributs (optionnels) aux noeuds du graphe depuis différents CSV.
    Les CSV doivent contenir une colonne 'id' correspondant aux noeuds de G.
    """
    from pathlib import Path as _Path

    def _safe_merge_node_attrs(node_df: pd.DataFrame, key: str = "id") -> None:
        if key not in node_df.columns:
            return
        node_df = node_df.copy()
        node_df[key] = node_df[key].astype(str)
        node_df = node_df.drop_duplicates(subset=[key]).set_index(key)

        common = set(G.nodes).intersection(node_df.index)
        mapping = {}
        for n in common:
            attrs = node_df.loc[n].to_dict()
            attrs = {k: (None if (isinstance(v, float) and np.isnan(v)) else v) for k, v in attrs.items()}
            mapping[n] = attrs
        if mapping:
            nx.set_node_attributes(G, mapping)

    for maybe_path in [
        port_locations_weight_csv,
        port_locations_value_csv,
        port_import_multiplier_csv,
        port_output_multiplier_csv,
    ]:
        if maybe_path:
            df_ = pd.read_csv(_Path(maybe_path))
            _safe_merge_node_attrs(df_, key="id")


def build_port_to_port_graph_global(
    df: pd.DataFrame,
    weight_basis: str = "tonnes",  # "tonnes" | "value"
    keep_top_n_ports: Optional[int] = 2000,
    top_n_by: str = "throughput",  # "throughput" | "export" | "import" | fallback
    port_locations_weight_csv: Optional[Union[str, "Path"]] = None,
    eps: float = 1e-12,
    min_edge_flow: float = 0.0,
) -> nx.Graph:
    """
    Construit un graphe non-orienté port-port agrégé globalement (toutes industries, tous OD).

    Principe :
      Pour chaque (iso3_O, iso3_D, Industries), on prend :
        - ports export : flow == "port_export"
        - ports import : flow == "port_import"
      On normalise leurs parts, puis on "apparie" export x import :
        edge_flow(p,q) += total_OD_flow * share_export(p) * share_import(q)

    Attributs d'arêtes (compatibles avec MetricConfig):
      - cond : conductance (ici = flow)
      - cost : résistance (ici = 1/(flow+eps))
      - flow : flow agrégé (tonnes ou valeur)
    """
    if weight_basis == "tonnes":
        flow_col = "q_sea_flow"
        total_col = "q_sea_flow_sum"
    elif weight_basis == "value":
        flow_col = "v_sea_flow"
        total_col = "v_sea_flow_sum"
        if total_col not in df.columns or flow_col not in df.columns:
            raise ValueError("Colonnes valeur manquantes (v_sea_flow / v_sea_flow_sum).")
    else:
        raise ValueError("weight_basis must be one of: 'tonnes', 'value'")

    work = df.copy()
    work = work.dropna(subset=[flow_col, total_col])
    work = work[(work[total_col] > 0) & (work[flow_col] >= 0)]

    # Option : filtrer aux top-N ports pour rester exploitable
    if keep_top_n_ports is not None:
        if port_locations_weight_csv is None:
            agg = work.groupby("id", as_index=True)[flow_col].sum().sort_values(ascending=False)
            top_ports = set(agg.head(keep_top_n_ports).index.astype(str))
        else:
            from pathlib import Path as _Path

            stats = pd.read_csv(_Path(port_locations_weight_csv))
            if "id" not in stats.columns:
                raise ValueError("port_locations_weight_csv doit contenir une colonne 'id'.")
            stats["id"] = stats["id"].astype(str)

            if top_n_by == "throughput" and "throughput" in stats.columns:
                s = pd.to_numeric(stats["throughput"], errors="coerce").fillna(0.0)
                stats = stats.assign(_score=s)
            elif top_n_by in stats.columns:
                s = pd.to_numeric(stats[top_n_by], errors="coerce").fillna(0.0)
                stats = stats.assign(_score=s)
            else:
                agg = work.groupby("id", as_index=False)[flow_col].sum().rename(columns={flow_col: "_score"})
                stats = stats.merge(agg, on="id", how="left")
                stats["_score"] = pd.to_numeric(stats["_score"], errors="coerce").fillna(0.0)

            top_ports = set(stats.sort_values("_score", ascending=False)["id"].head(keep_top_n_ports))

        work = work[work["id"].astype(str).isin(top_ports)].copy()

    edge_accum: Dict[Tuple[str, str], float] = {}
    group_cols = ["iso3_O", "iso3_D", "Industries"]

    for _, g in work.groupby(group_cols, sort=False):
        total_flow = float(g[total_col].iloc[0])
        if not np.isfinite(total_flow) or total_flow <= 0:
            continue

        exp = g[g["flow"] == "port_export"][["id", flow_col]].copy()
        imp = g[g["flow"] == "port_import"][["id", flow_col]].copy()
        if exp.empty or imp.empty:
            continue

        exp["share"] = exp[flow_col] / (exp[flow_col].sum() + eps)
        imp["share"] = imp[flow_col] / (imp[flow_col].sum() + eps)

        exp_ids = exp["id"].astype(str).to_numpy()
        imp_ids = imp["id"].astype(str).to_numpy()
        exp_sh = exp["share"].to_numpy()
        imp_sh = imp["share"].to_numpy()

        flow_mat = total_flow * np.outer(exp_sh, imp_sh)

        for i, p in enumerate(exp_ids):
            for j, q in enumerate(imp_ids):
                w = float(flow_mat[i, j])
                if w <= min_edge_flow:
                    continue
                a, b = (p, q) if p <= q else (q, p)  # non-orienté : clé canonique
                edge_accum[(a, b)] = edge_accum.get((a, b), 0.0) + w

    G = nx.Graph()
    ports = work["id"].astype(str).unique().tolist()
    G.add_nodes_from(ports)

    for (u, v), w in edge_accum.items():
        if u == v or w <= min_edge_flow:
            continue
        cond = float(w)
        cost = float(1.0 / (cond + eps))
        G.add_edge(u, v, flow=cond, cond=cond, cost=cost)

    return G


# -----------------------------
# Pipeline "neutral" (ground-truth par retrait, via échantillonnage de sources)
# -----------------------------


def connected_pairs_fraction(G: nx.Graph) -> float:
    """
    Fraction de paires (non ordonnées) connectées:
      sum_c |c|(|c|-1)/2  /  (n(n-1)/2)
    """
    n = G.number_of_nodes()
    if n <= 1:
        return 1.0
    total_pairs = n * (n - 1) / 2
    conn_pairs = 0.0
    for cc in nx.connected_components(G):
        s = len(cc)
        conn_pairs += s * (s - 1) / 2
    return float(conn_pairs / total_pairs)


def sample_sources(G: nx.Graph, n_sources: int = 40, seed: int = 0) -> List[Any]:
    rng = np.random.default_rng(seed)
    nodes = list(G.nodes())
    if not nodes:
        return []
    k = min(n_sources, len(nodes))
    return rng.choice(nodes, size=k, replace=False).tolist()


def estimate_efficiency_and_aspl(
    G: nx.Graph,
    sources: List[Any],
    weight: Optional[str] = None,
) -> Tuple[float, float]:
    """
    Estime (global efficiency, average shortest path length among reachable pairs)
    à partir d'un échantillon de sources.

    - efficiency: moyenne de 1/d(s,t) sur t != s, unreachable => 0
    - aspl: moyenne de d(s,t) sur paires atteignables (sur l'échantillon)
    """
    n = G.number_of_nodes()
    if n <= 1 or len(sources) == 0:
        return 0.0, float("nan")

    eff_sum = 0.0
    eff_count = 0  # compte (s,t), t!=s (unreachable comptés mais ajout 0)
    dist_sum = 0.0
    dist_count = 0  # compte uniquement des paires atteignables

    nodes = list(G.nodes())
    for s in sources:
        # Important: dans le pipeline "neutral", on évalue aussi G\\v.
        # Si v faisait partie des sources, alors s peut ne plus exister dans G.
        # On ignore simplement ces sources (sinon NetworkX lève NodeNotFound).
        if s not in G:
            continue
        if weight is None:
            d = nx.single_source_shortest_path_length(G, s)
        else:
            d = nx.single_source_dijkstra_path_length(G, s, weight=weight)

        for t in nodes:
            if t == s:
                continue
            eff_count += 1
            if t in d:
                dist = d[t]
                if dist > 0:
                    eff_sum += 1.0 / float(dist)
                dist_sum += float(dist)
                dist_count += 1

    eff = eff_sum / max(eff_count, 1)
    aspl = (dist_sum / dist_count) if dist_count > 0 else float("nan")
    return float(eff), float(aspl)


def neutral_fragility_for_nodes(
    G: nx.Graph,
    nodes_subset: Iterable[Any],
    sources: List[Any],
    shortest_path_weight: Optional[str] = None,
) -> pd.DataFrame:
    """
    Calcule, pour chaque v:
      - delta_conn_pairs = ConnPairsFrac(G) - ConnPairsFrac(G\\v)   (plus grand => plus critique)
      - delta_eff        = Eff(G) - Eff(G\\v)                      (plus grand => plus critique)
      - delta_aspl       = ASPL(G\\v) - ASPL(G)                    (plus grand => plus critique)
    """
    base_conn = connected_pairs_fraction(G)
    base_eff, base_aspl = estimate_efficiency_and_aspl(G, sources, weight=shortest_path_weight)

    rows = []
    for v in nodes_subset:
        if v not in G:
            continue
        H = G.copy()
        H.remove_node(v)

        conn = connected_pairs_fraction(H)
        eff, aspl = estimate_efficiency_and_aspl(H, sources, weight=shortest_path_weight)

        rows.append(
            {
                "node": v,
                "delta_conn_pairs": base_conn - conn,
                "delta_eff": base_eff - eff,
                "delta_aspl": (aspl - base_aspl) if (np.isfinite(aspl) and np.isfinite(base_aspl)) else np.nan,
            }
        )

    return pd.DataFrame(rows)


def evaluate_rankings(
    df: pd.DataFrame,
    metric_cols: List[str],
    target_col: str,
    top_ks: List[int] = [25, 50, 100],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Spearman + top-k overlap (comme avant), mais généralisé à n’importe quelle cible.
    """
    if spearmanr is None:
        raise ImportError("scipy n'est pas disponible (spearmanr). Installe scipy pour utiliser evaluate_rankings().")

    def _is_effectively_constant(a: np.ndarray) -> bool:
        a = np.asarray(a, dtype=float)
        a = a[np.isfinite(a)]
        if a.size <= 1:
            return True
        return bool(np.nanmax(a) - np.nanmin(a) == 0.0)

    # Si la cible est constante, toute comparaison de ranking est non-informative :
    # - Spearman est indéfini
    # - top-k overlap dépend arbitrairement de l'ordre de tri en cas d'égalité
    y_all = df[target_col].to_numpy()
    target_is_constant = _is_effectively_constant(y_all)

    spearman_rows = []
    for m in metric_cols:
        x = df[m].to_numpy()
        # Spearman n'est pas défini si x ou y est constant (ou s'il n'y a pas assez de points).
        if target_is_constant or _is_effectively_constant(x):
            rho = np.nan
        else:
            y = y_all
            rho = spearmanr(x, y, nan_policy="omit").correlation
        spearman_rows.append((m, float(rho) if rho is not None else np.nan))
    spearman_df = (
        pd.DataFrame(spearman_rows, columns=["metric", "spearman_rho"])
        .sort_values("spearman_rho", ascending=False)
        .reset_index(drop=True)
    )

    overlap_rows = []
    if target_is_constant:
        for m in metric_cols:
            for k in top_ks:
                kk = min(int(k), len(df))
                overlap_rows.append((m, kk, np.nan))
        overlap_df = pd.DataFrame(overlap_rows, columns=["metric", "k", "topk_overlap"])
    else:
        gt_rank = df.sort_values(target_col, ascending=False)["node"].to_list()
        for m in metric_cols:
            pred_rank = df.sort_values(m, ascending=False)["node"].to_list()
            for k in top_ks:
                kk = min(int(k), len(df))
                a = set(gt_rank[:kk])
                b = set(pred_rank[:kk])
                overlap_rows.append((m, kk, (len(a & b) / kk) if kk > 0 else np.nan))
        overlap_df = pd.DataFrame(overlap_rows, columns=["metric", "k", "topk_overlap"])

    return spearman_df, overlap_df


def _ensure_edge_weights(
    G: nx.Graph,
    conductance_attr: str = "cond",
    resistance_attr: str = "cost",
    eps: float = 1e-12,
) -> nx.Graph:
    """
    Assure qu'un graphe NON orienté possède des attributs cond/cost cohérents.
    Si le graphe a déjà cond/cost, ne fait que normaliser les types.
    """
    cfg = MetricConfig(cond_attr=conductance_attr, cost_attr=resistance_attr, eps=eps)
    return to_undirected_resistive_graph(G, cfg, sym_method="sum")


def compute_classic_metrics(
    G: nx.Graph,
    conductance_attr: str = "cond",
    resistance_attr: str = "cost",
    betw_k_sample: Optional[int] = 400,
    seed: int = 0,
) -> pd.DataFrame:
    """
    Métriques "classiques" compatibles avec le pipeline neutral.
    - degree : degré non pondéré
    - strength : degré pondéré par conductance_attr
    - betweenness : approx si betw_k_sample est fourni (shortest-path sur resistance_attr)
    - eigenvector : sur conductance_attr
    """
    nodes = list(G.nodes())

    degree = dict(G.degree())
    strength = dict(G.degree(weight=conductance_attr))

    k = None
    if betw_k_sample is not None:
        k = min(int(betw_k_sample), len(nodes))
        if k <= 0:
            k = None

    betw = nx.betweenness_centrality(
        G,
        k=k,
        seed=seed,
        weight=resistance_attr,
        normalized=True,
    )

    try:
        eig = nx.eigenvector_centrality_numpy(G, weight=conductance_attr)
    except Exception:
        eig = {v: 0.0 for v in nodes}

    return pd.DataFrame(
        {
            "node": nodes,
            "degree": [float(degree.get(v, 0.0)) for v in nodes],
            "strength": [float(strength.get(v, 0.0)) for v in nodes],
            "betweenness": [float(betw.get(v, 0.0)) for v in nodes],
            "eigenvector": [float(eig.get(v, 0.0)) for v in nodes],
        }
    )


def compute_resistance_metrics(
    G: nx.Graph,
    conductance_attr: str = "cond",
    resistance_attr: str = "cost",
    n_samples: int = 80,
    seed: int = 0,
    rtol: float = 1e-6,
    max_n_for_pinv: int = 600,
) -> pd.DataFrame:
    """
    Calcule des métriques résistance-distance.

    Implémentation actuelle: cf_closeness EXACT via L^+ si n <= max_n_for_pinv.
    (Les paramètres n_samples/rtol sont réservés à une future approximation.)
    """
    _ = (n_samples, seed, rtol)  # placeholders pour API stable
    cfg = MetricConfig(cond_attr=conductance_attr, cost_attr=resistance_attr, max_n_for_pinv=max_n_for_pinv)
    nodes, R, _Kbar = resistance_matrix_and_Kbar(G, cfg, return_R=True)
    cf = current_flow_closeness_from_R(nodes, R)
    return pd.DataFrame({"node": list(nodes), "cf_closeness": [float(cf.get(v, 0.0)) for v in nodes]})


def select_reference_nodes(
    G: nx.Graph,
    k_degree: int = 150,
    k_betw: int = 150,
    k_cf: int = 150,
    betw_k_sample: int = 400,
    seed: int = 0,
    conductance_attr: str = "cond",
    resistance_attr: str = "cost",
) -> List[Any]:
    """
    Sélectionne un sous-ensemble de nœuds "candidats" à tester en ground truth:
    union des top-k sur degree, betweenness, cf_closeness.
    """
    df_classic = compute_classic_metrics(
        G,
        conductance_attr=conductance_attr,
        resistance_attr=resistance_attr,
        betw_k_sample=betw_k_sample,
        seed=seed,
    )
    df_res = compute_resistance_metrics(G, conductance_attr=conductance_attr, resistance_attr=resistance_attr)

    df = df_classic.merge(df_res, on="node", how="left")
    nodes = df["node"].tolist()

    def topk(col: str, k: int) -> List[Any]:
        k = min(int(k), len(nodes))
        if k <= 0:
            return []
        return df.sort_values(col, ascending=False).head(k)["node"].tolist()

    s = set(topk("degree", k_degree)) | set(topk("betweenness", k_betw)) | set(topk("cf_closeness", k_cf))
    return list(s)


def run_minimal_experiment_neutral(
    G: nx.Graph,
    giant_component_only: bool = True,
    seed: int = 0,
    n_sources: int = 40,
    shortest_path_mode: str = "hops",  # "hops" | "resistance"
) -> Dict[str, pd.DataFrame]:
    """
    Pipeline minimal “moins biaisé”:
      - métriques classiques + cf_closeness
      - ground truth neutre: delta_conn_pairs / delta_eff / delta_aspl
      - comparaison Spearman + top-k
    """
    H = _ensure_edge_weights(G, conductance_attr="cond", resistance_attr="cost")

    if giant_component_only and H.number_of_nodes() > 0:
        cc = max(nx.connected_components(H), key=len)
        H = H.subgraph(cc).copy()

    df_classic = compute_classic_metrics(H, betw_k_sample=400, seed=seed)
    df_res = compute_resistance_metrics(H, conductance_attr="cond", resistance_attr="cost", n_samples=80, seed=seed)

    df = df_classic.merge(df_res[["node", "cf_closeness"]], on="node", how="left")

    ref_nodes = select_reference_nodes(
        H, k_degree=150, k_betw=150, k_cf=150, betw_k_sample=400, seed=seed, conductance_attr="cond", resistance_attr="cost"
    )

    sources = sample_sources(H, n_sources=n_sources, seed=seed)
    weight = None if shortest_path_mode == "hops" else "cost"

    df_gt = neutral_fragility_for_nodes(H, nodes_subset=ref_nodes, sources=sources, shortest_path_weight=weight)
    df_ref = df[df["node"].isin(ref_nodes)].merge(df_gt, on="node", how="inner")

    metric_cols = ["degree", "strength", "betweenness", "eigenvector", "cf_closeness"]

    sp_conn, ov_conn = evaluate_rankings(df_ref, metric_cols, target_col="delta_conn_pairs")
    sp_eff, ov_eff = evaluate_rankings(df_ref, metric_cols, target_col="delta_eff")
    sp_aspl, ov_aspl = evaluate_rankings(df_ref.dropna(subset=["delta_aspl"]), metric_cols, target_col="delta_aspl")

    return {
        "metrics_ref_nodes_with_neutral_gt": df_ref.sort_values("delta_eff", ascending=False),
        "spearman_vs_delta_conn_pairs": sp_conn,
        "topk_overlap_vs_delta_conn_pairs": ov_conn,
        "spearman_vs_delta_eff": sp_eff,
        "topk_overlap_vs_delta_eff": ov_eff,
        "spearman_vs_delta_aspl": sp_aspl,
        "topk_overlap_vs_delta_aspl": ov_aspl,
    }
if __name__ == "__main__":
    # Lance la suite complète (jouets + synthétiques).
    # Attention: la partie "node_removal_impacts" peut être coûteuse si tu augmentes
    # n / le nombre de seeds / le nombre de specs, car elle recalcule des métriques
    # après suppression de nœuds.
    example_run_all()


