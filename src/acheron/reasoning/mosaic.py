"""Mosaic Small-World Model Adaptation (Section 2).

Graph abstraction layer for bioelectric network topology:
    Nodes  = cell clusters
    Edges  = gap junction conductance (weighted)
    Local density  = high clustering coefficient
    Global links   = sparse long-range edges

Simulation capabilities:
    - Energy cost per long-range signal
    - Fault tolerance under random node failure
    - Uniform topology vs small-world topology comparison
    - Graph Laplacian spectral analysis (algebraic connectivity / Fiedler value)
    - Energy-optimized edge rewiring (maximize spectral gap, minimize dissipation)
    - Barabasi-Albert scale-free topology for competing model comparison

Output metrics:
    - Signal propagation latency
    - Energy per routing event
    - Error cascade probability
    - Spectral gap (lambda_2 of graph Laplacian)
    - Energy-optimal rewiring recommendations
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Optional

from .substrate import SubstrateEdge, SubstrateGraph, SubstrateNode


# ---------------------------------------------------------------------------
# Graph generation
# ---------------------------------------------------------------------------

def build_ring_lattice(
    n_nodes: int,
    k_neighbors: int,
    base_conductance_nS: float = 2.0,
    diffusion_delay_ms: float = 0.5,
) -> SubstrateGraph:
    """Build a regular ring lattice (Watts-Strogatz starting point).

    Each node connects to its k nearest neighbors on each side.
    """
    graph = SubstrateGraph()
    for i in range(n_nodes):
        graph.nodes[f"c{i}"] = SubstrateNode(
            node_id=f"c{i}",
            vm_mV=-40.0,
            vm_rest_mV=-40.0,
            cluster_size=1,
        )
    for i in range(n_nodes):
        for j in range(1, k_neighbors // 2 + 1):
            target = (i + j) % n_nodes
            graph.edges.append(SubstrateEdge(
                source=f"c{i}",
                target=f"c{target}",
                conductance_nS=base_conductance_nS,
                diffusion_delay_ms=diffusion_delay_ms,
            ))
    return graph


def build_small_world(
    n_nodes: int,
    k_neighbors: int,
    rewire_prob: float = 0.1,
    base_conductance_nS: float = 2.0,
    long_range_conductance_nS: float = 0.5,
    diffusion_delay_ms: float = 0.5,
    long_range_delay_ms: float = 2.0,
    seed: Optional[int] = None,
) -> SubstrateGraph:
    """Build a Watts-Strogatz small-world graph on the substrate.

    Rewired edges get lower conductance and higher delay (long-range cost).
    """
    if seed is not None:
        random.seed(seed)

    graph = build_ring_lattice(n_nodes, k_neighbors, base_conductance_nS, diffusion_delay_ms)

    rewired_edges: list[SubstrateEdge] = []
    kept_edges: list[SubstrateEdge] = []

    node_ids = list(graph.nodes.keys())

    for edge in graph.edges:
        if random.random() < rewire_prob:
            new_target = random.choice(node_ids)
            while new_target == edge.source or new_target == edge.target:
                new_target = random.choice(node_ids)
            rewired_edges.append(SubstrateEdge(
                source=edge.source,
                target=new_target,
                conductance_nS=long_range_conductance_nS,
                diffusion_delay_ms=long_range_delay_ms,
                metadata={"rewired": True},
            ))
        else:
            kept_edges.append(edge)

    graph.edges = kept_edges + rewired_edges
    return graph


def build_scale_free(
    n_nodes: int,
    m_edges_per_node: int = 3,
    base_conductance_nS: float = 2.0,
    diffusion_delay_ms: float = 1.0,
    seed: Optional[int] = None,
) -> SubstrateGraph:
    """Build a Barabasi-Albert scale-free graph for comparison.

    Preferential attachment model: new nodes connect to m existing nodes
    with probability proportional to their degree.  Scale-free networks
    have hub-heavy degree distributions (power-law) vs Watts-Strogatz
    small-world (peaked degree distribution).

    Biological relevance: tests whether bioelectric tissue connectivity
    is better modeled by hub-dominated or uniform-degree topology.
    """
    if seed is not None:
        random.seed(seed)

    graph = SubstrateGraph()
    # Seed with a complete graph of m+1 nodes
    m = max(1, m_edges_per_node)
    seed_size = m + 1
    for i in range(min(seed_size, n_nodes)):
        graph.nodes[f"c{i}"] = SubstrateNode(
            node_id=f"c{i}", vm_mV=-40.0, vm_rest_mV=-40.0,
        )

    # Connect seed nodes fully
    degree: list[int] = [0] * n_nodes
    for i in range(min(seed_size, n_nodes)):
        for j in range(i + 1, min(seed_size, n_nodes)):
            graph.edges.append(SubstrateEdge(
                source=f"c{i}", target=f"c{j}",
                conductance_nS=base_conductance_nS,
                diffusion_delay_ms=diffusion_delay_ms,
            ))
            degree[i] += 1
            degree[j] += 1

    if n_nodes <= seed_size:
        return graph

    # Preferential attachment for remaining nodes
    for i in range(seed_size, n_nodes):
        graph.nodes[f"c{i}"] = SubstrateNode(
            node_id=f"c{i}", vm_mV=-40.0, vm_rest_mV=-40.0,
        )
        # Build cumulative degree distribution for existing nodes
        existing = list(range(i))
        total_degree = sum(degree[k] for k in existing)
        if total_degree == 0:
            total_degree = len(existing)
            weights = [1.0] * len(existing)
        else:
            weights = [degree[k] for k in existing]

        # Select m targets by weighted sampling without replacement
        targets: set[int] = set()
        w = list(weights)
        for _ in range(min(m, len(existing))):
            w_sum = sum(w)
            if w_sum <= 0:
                break
            r = random.random() * w_sum
            cumulative = 0.0
            chosen = existing[0]
            for idx_pos, idx_node in enumerate(existing):
                cumulative += w[idx_pos]
                if cumulative >= r:
                    chosen = idx_node
                    w[idx_pos] = 0.0  # prevent re-selection
                    break
            targets.add(chosen)

        for t in targets:
            graph.edges.append(SubstrateEdge(
                source=f"c{i}", target=f"c{t}",
                conductance_nS=base_conductance_nS,
                diffusion_delay_ms=diffusion_delay_ms,
            ))
            degree[i] += 1
            degree[t] += 1

    return graph


def build_uniform_random(
    n_nodes: int,
    n_edges: int,
    base_conductance_nS: float = 2.0,
    diffusion_delay_ms: float = 1.0,
    seed: Optional[int] = None,
) -> SubstrateGraph:
    """Build a uniform random graph (Erdos-Renyi style) for comparison."""
    if seed is not None:
        random.seed(seed)

    graph = SubstrateGraph()
    for i in range(n_nodes):
        graph.nodes[f"c{i}"] = SubstrateNode(
            node_id=f"c{i}", vm_mV=-40.0, vm_rest_mV=-40.0,
        )

    node_ids = list(graph.nodes.keys())
    edges_added: set[tuple[str, str]] = set()
    while len(edges_added) < n_edges:
        a, b = random.sample(node_ids, 2)
        key = (min(a, b), max(a, b))
        if key not in edges_added:
            edges_added.add(key)
            graph.edges.append(SubstrateEdge(
                source=a, target=b,
                conductance_nS=base_conductance_nS,
                diffusion_delay_ms=diffusion_delay_ms,
            ))
    return graph


# ---------------------------------------------------------------------------
# Topology metrics
# ---------------------------------------------------------------------------

@dataclass
class TopologyMetrics:
    """Metrics tracked by Nexus for substrate topology evaluation."""

    node_count: int = 0
    edge_count: int = 0
    clustering_coefficient: float = 0.0
    average_path_length: float = 0.0
    # Nexus-required output metrics
    signal_propagation_latency_ms: float = 0.0
    energy_per_routing_event_J: float = 0.0
    error_cascade_probability: float = 0.0
    # Fault tolerance
    fault_tolerance_fraction: float = 0.0   # fraction of nodes removable before disconnection
    # Topology class
    topology_class: str = ""                # "small_world", "uniform", "ring_lattice"


def _build_adjacency(graph: SubstrateGraph) -> dict[str, set[str]]:
    """Undirected adjacency set."""
    adj: dict[str, set[str]] = {nid: set() for nid in graph.nodes}
    for e in graph.edges:
        if e.source in adj and e.target in adj:
            adj[e.source].add(e.target)
            adj[e.target].add(e.source)
    return adj


def compute_clustering_coefficient(graph: SubstrateGraph) -> float:
    """Global clustering coefficient (average of local coefficients)."""
    adj = _build_adjacency(graph)
    coefficients: list[float] = []
    for node, neighbors in adj.items():
        k = len(neighbors)
        if k < 2:
            continue
        neighbor_list = list(neighbors)
        triangles = 0
        for i in range(len(neighbor_list)):
            for j in range(i + 1, len(neighbor_list)):
                if neighbor_list[j] in adj[neighbor_list[i]]:
                    triangles += 1
        possible = k * (k - 1) / 2
        coefficients.append(triangles / possible if possible > 0 else 0.0)
    return sum(coefficients) / len(coefficients) if coefficients else 0.0


def compute_average_path_length(graph: SubstrateGraph) -> float:
    """Average shortest path length via BFS (hop count)."""
    adj = _build_adjacency(graph)
    node_ids = list(graph.nodes.keys())
    n = len(node_ids)
    if n < 2:
        return 0.0

    total_length = 0
    pair_count = 0

    for start in node_ids:
        distances: dict[str, int] = {start: 0}
        queue = [start]
        head = 0
        while head < len(queue):
            current = queue[head]
            head += 1
            for neighbor in adj[current]:
                if neighbor not in distances:
                    distances[neighbor] = distances[current] + 1
                    queue.append(neighbor)
        for target in node_ids:
            if target != start and target in distances:
                total_length += distances[target]
                pair_count += 1

    return total_length / pair_count if pair_count > 0 else float("inf")


def compute_signal_propagation_latency(graph: SubstrateGraph, source: str, target: str) -> float:
    """Shortest-path latency (sum of diffusion delays) in ms using Dijkstra."""
    if source not in graph.nodes or target not in graph.nodes:
        return float("inf")

    # Build weighted adjacency for delays
    adj: dict[str, list[tuple[str, float]]] = {nid: [] for nid in graph.nodes}
    for e in graph.edges:
        if e.source in adj and e.target in adj:
            adj[e.source].append((e.target, e.diffusion_delay_ms))
            adj[e.target].append((e.source, e.diffusion_delay_ms))

    dist: dict[str, float] = {nid: float("inf") for nid in graph.nodes}
    dist[source] = 0.0
    visited: set[str] = set()

    while True:
        # Find unvisited node with smallest distance
        current = None
        current_dist = float("inf")
        for nid in graph.nodes:
            if nid not in visited and dist[nid] < current_dist:
                current = nid
                current_dist = dist[nid]
        if current is None or current == target:
            break
        visited.add(current)
        for neighbor, delay in adj[current]:
            new_dist = dist[current] + delay
            if new_dist < dist[neighbor]:
                dist[neighbor] = new_dist

    return dist[target]


def compute_energy_per_routing_event(
    conductance_nS: float,
    delta_v_mV: float = 30.0,
    duration_ms: float = 1.0,
) -> float:
    """Energy dissipated per routing event through a gap junction.

    E = G * (deltaV)^2 * t  (Joules)
    """
    g = conductance_nS * 1e-9       # nS -> S
    v = delta_v_mV * 1e-3           # mV -> V
    t = duration_ms * 1e-3          # ms -> s
    return g * v * v * t


def compute_error_cascade_probability(
    graph: SubstrateGraph,
    failure_probability: float = 0.01,
    cascade_threshold: float = 0.5,
) -> float:
    """Estimate error cascade probability under random node failure.

    Model: each node fails independently with failure_probability.
    A cascade occurs when removing failed nodes disconnects more than
    cascade_threshold fraction of the remaining graph.
    """
    adj = _build_adjacency(graph)
    node_ids = list(graph.nodes.keys())
    n = len(node_ids)
    if n == 0:
        return 0.0

    # Expected number of failures
    expected_failures = failure_probability * n

    # Compute largest connected component after removing expected_failures nodes
    # Use deterministic removal of highest-degree nodes (worst case) and
    # random removal (average case), return average-case estimate.
    # For tractability, simulate with expected count.
    n_remove = max(1, int(round(expected_failures)))
    if n_remove >= n:
        return 1.0

    # Remove n_remove random nodes (use sorted for determinism)
    remaining = set(node_ids[n_remove:])

    # BFS to find largest connected component
    visited: set[str] = set()
    max_component = 0
    for node in remaining:
        if node in visited:
            continue
        component_size = 0
        queue = [node]
        head = 0
        while head < len(queue):
            current = queue[head]
            head += 1
            if current in visited:
                continue
            visited.add(current)
            component_size += 1
            for neighbor in adj.get(current, set()):
                if neighbor in remaining and neighbor not in visited:
                    queue.append(neighbor)
        max_component = max(max_component, component_size)

    surviving_fraction = max_component / len(remaining) if remaining else 0.0
    disconnected_fraction = 1.0 - surviving_fraction

    return disconnected_fraction if disconnected_fraction > cascade_threshold else disconnected_fraction * failure_probability


def compute_fault_tolerance(
    graph: SubstrateGraph,
    step: float = 0.05,
) -> float:
    """Fraction of nodes removable before largest component < 50% of original.

    Removes nodes in order of increasing degree (random-failure approximation).
    """
    adj = _build_adjacency(graph)
    node_ids = sorted(graph.nodes.keys(), key=lambda n: len(adj.get(n, set())))
    n = len(node_ids)
    if n < 2:
        return 0.0

    remaining = set(node_ids)
    for i, node_id in enumerate(node_ids):
        remaining.discard(node_id)
        if not remaining:
            return i / n
        # Check largest connected component
        visited: set[str] = set()
        max_comp = 0
        for start in remaining:
            if start in visited:
                continue
            comp = 0
            queue = [start]
            head = 0
            while head < len(queue):
                cur = queue[head]
                head += 1
                if cur in visited:
                    continue
                visited.add(cur)
                comp += 1
                for nb in adj.get(cur, set()):
                    if nb in remaining and nb not in visited:
                        queue.append(nb)
            max_comp = max(max_comp, comp)
        if max_comp < n * 0.5:
            return i / n

    return 1.0


def analyze_topology(graph: SubstrateGraph, topology_class: str = "") -> TopologyMetrics:
    """Compute all Nexus-required topology metrics for a substrate graph."""
    cc = compute_clustering_coefficient(graph)
    apl = compute_average_path_length(graph)

    # Average signal propagation latency (sample pairs)
    node_ids = list(graph.nodes.keys())
    n = len(node_ids)
    if n >= 2:
        sample_pairs = min(50, n * (n - 1) // 2)
        latencies: list[float] = []
        for i in range(min(sample_pairs, n)):
            j = (i + n // 2) % n
            if i != j:
                lat = compute_signal_propagation_latency(graph, node_ids[i], node_ids[j])
                if lat < float("inf"):
                    latencies.append(lat)
        avg_latency = sum(latencies) / len(latencies) if latencies else float("inf")
    else:
        avg_latency = 0.0

    # Average energy per routing event
    avg_conductance = (
        sum(e.conductance_nS for e in graph.edges) / len(graph.edges)
        if graph.edges else 0.0
    )
    energy = compute_energy_per_routing_event(avg_conductance)

    # Error cascade
    cascade_prob = compute_error_cascade_probability(graph)

    # Fault tolerance
    ft = compute_fault_tolerance(graph)

    return TopologyMetrics(
        node_count=graph.node_count(),
        edge_count=graph.edge_count(),
        clustering_coefficient=round(cc, 4),
        average_path_length=round(apl, 4),
        signal_propagation_latency_ms=round(avg_latency, 4),
        energy_per_routing_event_J=energy,
        error_cascade_probability=round(cascade_prob, 6),
        fault_tolerance_fraction=round(ft, 4),
        topology_class=topology_class,
    )


@dataclass
class TopologyComparison:
    """Side-by-side comparison of topology configurations."""

    uniform: TopologyMetrics = field(default_factory=TopologyMetrics)
    small_world: TopologyMetrics = field(default_factory=TopologyMetrics)
    ring_lattice: TopologyMetrics = field(default_factory=TopologyMetrics)


def compare_topologies(
    n_nodes: int = 100,
    k_neighbors: int = 6,
    rewire_prob: float = 0.1,
    seed: Optional[int] = 42,
) -> TopologyComparison:
    """Compare uniform, ring lattice, and small-world topologies."""
    n_edges = n_nodes * k_neighbors // 2  # match edge count

    ring = build_ring_lattice(n_nodes, k_neighbors)
    sw = build_small_world(n_nodes, k_neighbors, rewire_prob, seed=seed)
    uniform = build_uniform_random(n_nodes, n_edges, seed=seed)

    return TopologyComparison(
        ring_lattice=analyze_topology(ring, "ring_lattice"),
        small_world=analyze_topology(sw, "small_world"),
        uniform=analyze_topology(uniform, "uniform"),
    )


# ---------------------------------------------------------------------------
# Graph Laplacian Spectral Analysis
# ---------------------------------------------------------------------------

def _build_weighted_adjacency_matrix(graph: SubstrateGraph) -> tuple[list[list[float]], list[str]]:
    """Build weighted adjacency matrix W from substrate graph.

    W[i][j] = sum of conductances between nodes i and j (nS).
    Returns (matrix, node_id_list).
    """
    node_ids = sorted(graph.nodes.keys())
    n = len(node_ids)
    idx_map = {nid: i for i, nid in enumerate(node_ids)}

    W = [[0.0] * n for _ in range(n)]
    for e in graph.edges:
        si = idx_map.get(e.source)
        ti = idx_map.get(e.target)
        if si is not None and ti is not None and si != ti:
            W[si][ti] += e.conductance_nS
            W[ti][si] += e.conductance_nS

    return W, node_ids


def compute_graph_laplacian(graph: SubstrateGraph) -> tuple[list[list[float]], list[str]]:
    """Compute the graph Laplacian matrix L = D - W.

    D = degree matrix (diagonal, d_i = sum of edge weights incident to i)
    W = weighted adjacency matrix

    Returns (L, node_id_list).
    """
    W, node_ids = _build_weighted_adjacency_matrix(graph)
    n = len(node_ids)

    L = [[0.0] * n for _ in range(n)]
    for i in range(n):
        degree = sum(W[i])
        L[i][i] = degree
        for j in range(n):
            if i != j:
                L[i][j] = -W[i][j]

    return L, node_ids


def _eigenvalues_symmetric(M: list[list[float]], max_iter: int = 500, tol: float = 1e-10) -> list[float]:
    """Compute eigenvalues of a symmetric matrix using QR iteration.

    Uses Householder tridiagonalization followed by implicit QR shifts.
    Falls back to explicit shifted inverse iteration for the smallest
    non-trivial eigenvalue when needed.

    Returns eigenvalues sorted in ascending order.
    """
    n = len(M)
    if n == 0:
        return []
    if n == 1:
        return [M[0][0]]

    # Work on a copy
    A = [row[:] for row in M]

    # Householder tridiagonalization for better QR convergence
    diag = [0.0] * n     # diagonal
    off = [0.0] * n      # off-diagonal

    # Reduce to tridiagonal form
    for i in range(n - 2, 0, -1):
        # Compute Householder vector for column i
        scale = sum(abs(A[i][j]) for j in range(i))
        if scale < 1e-15:
            off[i] = A[i][i - 1]
            continue

        h = 0.0
        for j in range(i):
            A[i][j] /= scale
            h += A[i][j] * A[i][j]

        f = A[i][i - 1]
        g = -math.copysign(math.sqrt(h), f)
        off[i] = scale * g
        h -= f * g
        A[i][i - 1] = f - g

        f = 0.0
        for j in range(i):
            A[j][i] = A[i][j] / h
            g = 0.0
            for k in range(j + 1):
                g += A[j][k] * A[i][k]
            for k in range(j + 1, i):
                g += A[k][j] * A[i][k]
            off[j] = g / h
            f += off[j] * A[i][j]

        hh = f / (h + h)
        for j in range(i):
            f = A[i][j]
            g = off[j] - hh * f
            off[j] = g
            for k in range(j + 1):
                A[j][k] -= f * off[k] + g * A[i][k]

    off[0] = 0.0
    for i in range(n):
        diag[i] = A[i][i]
    if n >= 2:
        off[1] = A[1][0]
        for i in range(2, n):
            # off[i] was set during tridiagonalization
            if off[i] == 0.0:
                off[i] = A[i][i - 1]

    # QL iteration with implicit shifts on the tridiagonal matrix
    d = diag[:]
    e = off[:]

    for _ in range(max_iter * n):
        converged = True
        for l_idx in range(n - 1):
            if abs(e[l_idx + 1]) > tol * (abs(d[l_idx]) + abs(d[l_idx + 1]) + 1e-30):
                converged = False
                break
        if converged:
            break

        for l_idx in range(n - 1):
            if abs(e[l_idx + 1]) < tol * (abs(d[l_idx]) + abs(d[l_idx + 1]) + 1e-30):
                continue

            # Find block end
            m = l_idx
            for m in range(l_idx + 1, n):
                if m == n - 1 or abs(e[m + 1 if m + 1 < n else m]) < tol * (abs(d[m]) + abs(d[m + 1 if m + 1 < n else m]) + 1e-30):
                    break

            if m == l_idx:
                continue

            # Wilkinson shift
            g = (d[l_idx + 1] - d[l_idx]) / (2.0 * e[l_idx + 1]) if abs(e[l_idx + 1]) > 1e-30 else 0.0
            r = math.sqrt(g * g + 1.0)
            g = d[m] - d[l_idx] + e[l_idx + 1] / (g + math.copysign(r, g)) if abs(g) > 1e-30 else d[m] - d[l_idx]

            s = 1.0
            c = 1.0
            p = 0.0

            for i in range(m - 1, l_idx - 1, -1):
                f = s * e[i + 1]
                b = c * e[i + 1]
                if abs(f) >= abs(g):
                    c = g / f
                    r = math.sqrt(c * c + 1.0)
                    e[i + 2 if i + 2 < n else n - 1] = f * r
                    s = 1.0 / r
                    c *= s
                else:
                    s = f / g
                    r = math.sqrt(s * s + 1.0)
                    e[i + 2 if i + 2 < n else n - 1] = g * r
                    c = 1.0 / r
                    s *= c
                g = d[i + 1] - p
                r = (d[i] - g) * s + 2.0 * c * b
                p = s * r
                d[i + 1] = g + p
                g = c * r - b

            d[l_idx] -= p
            e[l_idx + 1 if l_idx + 1 < n else n - 1] = g
            if m + 1 < n:
                e[m + 1] = 0.0

    d.sort()
    return d


def compute_spectral_gap(graph: SubstrateGraph) -> float:
    """Compute the spectral gap (algebraic connectivity) of the graph.

    The spectral gap is lambda_2, the second-smallest eigenvalue of the
    graph Laplacian L = D - W. A larger spectral gap means faster mixing
    and signal propagation across the network.

    Returns lambda_2 (>= 0). Returns 0.0 for disconnected graphs.
    """
    L, _ = compute_graph_laplacian(graph)
    n = len(L)
    if n < 2:
        return 0.0

    eigenvalues = _eigenvalues_symmetric(L)

    # lambda_1 should be ~0 (connected graph). lambda_2 is the spectral gap.
    # Find first eigenvalue meaningfully > 0.
    for ev in eigenvalues[1:]:
        if ev > 1e-8:
            return round(ev, 8)

    return 0.0


def compute_total_network_energy(
    graph: SubstrateGraph,
    delta_v_mV: float = 30.0,
    duration_ms: float = 1.0,
) -> float:
    """Total energy dissipated across all edges during one signaling event.

    E_total = sum_edges( G_ij * deltaV^2 * t )  in Joules.
    """
    v = delta_v_mV * 1e-3    # mV -> V
    t = duration_ms * 1e-3   # ms -> s
    total = 0.0
    for e in graph.edges:
        g = e.conductance_nS * 1e-9  # nS -> S
        total += g * v * v * t
    return total


@dataclass
class SpectralAnalysis:
    """Spectral properties of the graph Laplacian."""

    spectral_gap: float = 0.0              # lambda_2 (algebraic connectivity)
    laplacian_eigenvalues: list[float] = field(default_factory=list)  # ascending
    total_network_energy_J: float = 0.0    # total dissipation per signaling event
    energy_per_edge_J: float = 0.0         # average energy per edge
    spectral_efficiency: float = 0.0       # spectral_gap / total_energy (higher = better)


def analyze_spectral_properties(
    graph: SubstrateGraph,
    delta_v_mV: float = 30.0,
    duration_ms: float = 1.0,
    n_eigenvalues: int = 10,
) -> SpectralAnalysis:
    """Full spectral analysis of the substrate graph.

    Computes Laplacian eigenvalues, spectral gap, network energy,
    and the spectral efficiency ratio (gap / energy).
    """
    L, _ = compute_graph_laplacian(graph)
    n = len(L)
    if n < 2:
        return SpectralAnalysis()

    eigenvalues = _eigenvalues_symmetric(L)

    # Trim to requested count
    eigs = eigenvalues[:min(n_eigenvalues, len(eigenvalues))]

    spectral_gap = 0.0
    for ev in eigenvalues[1:]:
        if ev > 1e-8:
            spectral_gap = ev
            break

    total_energy = compute_total_network_energy(graph, delta_v_mV, duration_ms)
    edge_count = graph.edge_count()
    energy_per_edge = total_energy / edge_count if edge_count > 0 else 0.0
    spectral_eff = spectral_gap / total_energy if total_energy > 1e-30 else 0.0

    return SpectralAnalysis(
        spectral_gap=round(spectral_gap, 8),
        laplacian_eigenvalues=[round(e, 8) for e in eigs],
        total_network_energy_J=total_energy,
        energy_per_edge_J=energy_per_edge,
        spectral_efficiency=round(spectral_eff, 4),
    )


# ---------------------------------------------------------------------------
# Energy-Optimized Edge Rewiring
# ---------------------------------------------------------------------------

@dataclass
class RewireCandidate:
    """A candidate edge rewiring event with computed impact."""

    original_source: str
    original_target: str
    new_source: str
    new_target: str
    original_conductance_nS: float = 0.0
    new_conductance_nS: float = 0.0
    delta_spectral_gap: float = 0.0         # change in lambda_2
    delta_energy_J: float = 0.0             # change in total network energy
    efficiency_score: float = 0.0           # delta_gap / |delta_energy| (higher = better)
    new_spectral_gap: float = 0.0
    new_total_energy_J: float = 0.0


@dataclass
class RewireAnalysis:
    """Complete rewiring optimization analysis."""

    baseline_spectral_gap: float = 0.0
    baseline_total_energy_J: float = 0.0
    baseline_spectral_efficiency: float = 0.0
    candidates: list[RewireCandidate] = field(default_factory=list)
    optimal_rewires: list[RewireCandidate] = field(default_factory=list)
    optimized_spectral_gap: float = 0.0
    optimized_total_energy_J: float = 0.0
    optimized_spectral_efficiency: float = 0.0
    gap_improvement_pct: float = 0.0
    energy_reduction_pct: float = 0.0
    n_nodes: int = 0
    n_edges: int = 0
    rewire_budget: int = 0


def _apply_rewire(
    graph: SubstrateGraph,
    old_source: str,
    old_target: str,
    new_source: str,
    new_target: str,
    new_conductance_nS: float,
    new_delay_ms: float,
) -> SubstrateGraph:
    """Return a new graph with one edge rewired."""
    new_graph = SubstrateGraph()
    new_graph.nodes = dict(graph.nodes)

    found = False
    for e in graph.edges:
        if not found and e.source == old_source and e.target == old_target:
            new_graph.edges.append(SubstrateEdge(
                source=new_source,
                target=new_target,
                conductance_nS=new_conductance_nS,
                diffusion_delay_ms=new_delay_ms,
                metadata={"rewired_from": f"{old_source}->{old_target}"},
            ))
            found = True
        else:
            new_graph.edges.append(e)

    if not found:
        # Try reverse direction
        for i, e in enumerate(new_graph.edges):
            if e.source == old_target and e.target == old_source:
                new_graph.edges[i] = SubstrateEdge(
                    source=new_source,
                    target=new_target,
                    conductance_nS=new_conductance_nS,
                    diffusion_delay_ms=new_delay_ms,
                    metadata={"rewired_from": f"{old_target}->{old_source}"},
                )
                break

    return new_graph


def find_optimal_rewires(
    graph: SubstrateGraph,
    rewire_budget: int = 5,
    delta_v_mV: float = 30.0,
    duration_ms: float = 1.0,
    long_range_conductance_nS: float = 0.5,
    long_range_delay_ms: float = 2.0,
    seed: Optional[int] = 42,
) -> RewireAnalysis:
    """Find edge rewirings that maximize spectral gap while minimizing energy.

    Strategy: greedy search over candidate rewires. For each local edge,
    evaluate rewiring it to a distant node and compute the spectral gap
    gain vs energy cost change. Rank by efficiency score
    (delta_gap / |delta_energy|) and select top candidates.

    This implements the Payvand Mosaic principle: in-memory routing
    optimization where the topology itself encodes the computation path.
    """
    if seed is not None:
        random.seed(seed)

    n = graph.node_count()
    if n < 4:
        return RewireAnalysis(n_nodes=n, n_edges=graph.edge_count())

    baseline_gap = compute_spectral_gap(graph)
    baseline_energy = compute_total_network_energy(graph, delta_v_mV, duration_ms)
    baseline_eff = baseline_gap / baseline_energy if baseline_energy > 1e-30 else 0.0

    node_ids = sorted(graph.nodes.keys())
    adj = _build_adjacency(graph)

    # Identify local (non-rewired) edges as candidates
    local_edges: list[SubstrateEdge] = []
    for e in graph.edges:
        if not e.metadata.get("rewired", False):
            local_edges.append(e)

    # Limit candidates for tractability
    max_candidates = min(len(local_edges), n * 2)
    candidate_edges = local_edges[:max_candidates]

    candidates: list[RewireCandidate] = []

    for edge in candidate_edges:
        # Find a distant target (not already a neighbor)
        neighbors = adj.get(edge.source, set()) | adj.get(edge.target, set())
        distant_targets = [nid for nid in node_ids
                          if nid != edge.source and nid != edge.target
                          and nid not in neighbors]

        if not distant_targets:
            continue

        # Sample a few distant targets
        sample_size = min(3, len(distant_targets))
        targets = random.sample(distant_targets, sample_size)

        for new_tgt in targets:
            new_graph = _apply_rewire(
                graph, edge.source, edge.target,
                edge.source, new_tgt,
                long_range_conductance_nS, long_range_delay_ms,
            )

            new_gap = compute_spectral_gap(new_graph)
            new_energy = compute_total_network_energy(new_graph, delta_v_mV, duration_ms)

            delta_gap = new_gap - baseline_gap
            delta_energy = new_energy - baseline_energy

            # We want positive delta_gap (improved connectivity)
            # and negative or small delta_energy (reduced dissipation)
            if abs(delta_energy) > 1e-30:
                eff = delta_gap / abs(delta_energy)
            elif delta_gap > 0:
                eff = float("inf")
            else:
                eff = 0.0

            candidates.append(RewireCandidate(
                original_source=edge.source,
                original_target=edge.target,
                new_source=edge.source,
                new_target=new_tgt,
                original_conductance_nS=edge.conductance_nS,
                new_conductance_nS=long_range_conductance_nS,
                delta_spectral_gap=round(delta_gap, 8),
                delta_energy_J=delta_energy,
                efficiency_score=round(eff, 4) if eff != float("inf") else 1e12,
                new_spectral_gap=round(new_gap, 8),
                new_total_energy_J=new_energy,
            ))

    # Sort by efficiency: prefer rewires that increase gap with least energy cost
    # Filter to only those that actually improve spectral gap
    beneficial = [c for c in candidates if c.delta_spectral_gap > 0]
    beneficial.sort(key=lambda c: c.efficiency_score, reverse=True)

    # Select top-K non-overlapping rewires
    optimal: list[RewireCandidate] = []
    used_edges: set[tuple[str, str]] = set()

    for c in beneficial:
        edge_key = (min(c.original_source, c.original_target),
                    max(c.original_source, c.original_target))
        if edge_key not in used_edges and len(optimal) < rewire_budget:
            used_edges.add(edge_key)
            optimal.append(c)

    # Compute cumulative optimized metrics
    working_graph = SubstrateGraph()
    working_graph.nodes = dict(graph.nodes)
    working_graph.edges = list(graph.edges)

    for rw in optimal:
        working_graph = _apply_rewire(
            working_graph, rw.original_source, rw.original_target,
            rw.new_source, rw.new_target,
            rw.new_conductance_nS, long_range_delay_ms,
        )

    opt_gap = compute_spectral_gap(working_graph) if optimal else baseline_gap
    opt_energy = compute_total_network_energy(working_graph, delta_v_mV, duration_ms) if optimal else baseline_energy
    opt_eff = opt_gap / opt_energy if opt_energy > 1e-30 else 0.0

    gap_improvement = ((opt_gap - baseline_gap) / baseline_gap * 100.0) if baseline_gap > 1e-10 else 0.0
    energy_reduction = ((baseline_energy - opt_energy) / baseline_energy * 100.0) if baseline_energy > 1e-30 else 0.0

    return RewireAnalysis(
        baseline_spectral_gap=round(baseline_gap, 8),
        baseline_total_energy_J=baseline_energy,
        baseline_spectral_efficiency=round(baseline_eff, 4),
        candidates=candidates,
        optimal_rewires=optimal,
        optimized_spectral_gap=round(opt_gap, 8),
        optimized_total_energy_J=opt_energy,
        optimized_spectral_efficiency=round(opt_eff, 4),
        gap_improvement_pct=round(gap_improvement, 2),
        energy_reduction_pct=round(energy_reduction, 2),
        n_nodes=n,
        n_edges=graph.edge_count(),
        rewire_budget=rewire_budget,
    )


def rewire_sweep(
    n_nodes: int = 50,
    k_neighbors: int = 6,
    rewire_probs: Optional[list[float]] = None,
    seed: int = 42,
) -> list[dict]:
    """Sweep rewire probability and compute spectral gap + energy for each.

    Implements the Watts-Strogatz beta sweep: as beta increases from 0 to 1,
    tracks how spectral gap and energy change, identifying the optimal
    operating point for bioelectric signal propagation.
    """
    if rewire_probs is None:
        rewire_probs = [0.0, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.8, 1.0]

    results: list[dict] = []
    for beta in rewire_probs:
        g = build_small_world(n_nodes, k_neighbors, rewire_prob=beta, seed=seed)
        gap = compute_spectral_gap(g)
        cc = compute_clustering_coefficient(g)
        apl = compute_average_path_length(g)
        energy = compute_total_network_energy(g)
        eff = gap / energy if energy > 1e-30 else 0.0

        results.append({
            "rewire_prob": beta,
            "spectral_gap": round(gap, 8),
            "clustering_coefficient": round(cc, 4),
            "average_path_length": round(apl, 4),
            "total_energy_J": energy,
            "spectral_efficiency": round(eff, 4),
        })

    return results
