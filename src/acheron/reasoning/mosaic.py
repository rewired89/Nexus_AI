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

Output metrics:
    - Signal propagation latency
    - Energy per routing event
    - Error cascade probability
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
