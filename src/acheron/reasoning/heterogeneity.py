"""Heterogeneity-Aware Computation Layer (Section 4).

Variability modeling:
    Vm_rest variance  = +/- X mV
    Gap junction variance = +/- Y%
    Ion expression variability factor = Z

Simulation targets:
    - Uniform substrate vs heterogeneous substrate
    - Error tolerance comparison
    - Recovery after perturbation
    - Attractor basin depth

Stored result: "Natural disorder may increase distributed resilience."
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Optional

from .substrate import SubstrateGraph, SubstrateNode, SubstrateEdge


@dataclass
class HeterogeneityParams:
    """Parameters defining substrate heterogeneity."""

    vm_rest_variance_mV: float = 5.0            # +/- X mV around mean
    gap_junction_variance_pct: float = 20.0     # +/- Y% of nominal conductance
    ion_expression_variability: float = 0.3     # Z: coefficient of variation (0-1)
    # Source tracking
    evidence_status: str = "[DATA GAP]"


@dataclass
class SubstrateConfig:
    """Configuration for a substrate instance (uniform or heterogeneous)."""

    label: str = ""
    is_heterogeneous: bool = False
    heterogeneity: HeterogeneityParams = field(default_factory=HeterogeneityParams)
    n_nodes: int = 100
    k_neighbors: int = 6
    base_vm_mV: float = -40.0
    base_conductance_nS: float = 2.0
    seed: Optional[int] = None


def build_substrate(config: SubstrateConfig) -> SubstrateGraph:
    """Build a substrate graph with optional heterogeneity applied."""
    if config.seed is not None:
        random.seed(config.seed)

    graph = SubstrateGraph()

    # Create nodes
    for i in range(config.n_nodes):
        if config.is_heterogeneous:
            vm = config.base_vm_mV + random.gauss(0, config.heterogeneity.vm_rest_variance_mV)
            ion_var = {
                "expression_factor": max(0.1, random.gauss(1.0, config.heterogeneity.ion_expression_variability))
            }
        else:
            vm = config.base_vm_mV
            ion_var = {"expression_factor": 1.0}

        graph.nodes[f"n{i}"] = SubstrateNode(
            node_id=f"n{i}",
            vm_mV=vm,
            vm_rest_mV=vm,
            ion_channel_expression=ion_var,
        )

    # Create ring-lattice edges with optional variance
    for i in range(config.n_nodes):
        for j in range(1, config.k_neighbors // 2 + 1):
            target_idx = (i + j) % config.n_nodes
            if config.is_heterogeneous:
                var_frac = config.heterogeneity.gap_junction_variance_pct / 100.0
                g = config.base_conductance_nS * max(0.1, random.gauss(1.0, var_frac))
            else:
                g = config.base_conductance_nS

            graph.edges.append(SubstrateEdge(
                source=f"n{i}",
                target=f"n{target_idx}",
                conductance_nS=g,
            ))

    return graph


# ---------------------------------------------------------------------------
# Perturbation and recovery simulation
# ---------------------------------------------------------------------------

@dataclass
class PerturbationResult:
    """Result of a perturbation-recovery simulation."""

    # Error tolerance
    error_tolerance_fraction: float = 0.0       # fraction of flipped nodes tolerated
    # Recovery
    recovery_steps: int = 0                     # steps to return to equilibrium
    recovery_complete: bool = False              # whether full recovery occurred
    residual_error_mV: float = 0.0              # mean |Vm - Vm_rest| after recovery
    # Attractor basin
    attractor_basin_depth_mV: float = 0.0       # max perturbation before state flip
    # Label
    substrate_label: str = ""


def _mean_field_step(graph: SubstrateGraph, coupling_strength: float = 0.1) -> float:
    """One step of mean-field relaxation toward attractor states.

    Returns mean absolute deviation from rest after the step.
    """
    # Build adjacency
    adj: dict[str, list[tuple[str, float]]] = {nid: [] for nid in graph.nodes}
    for e in graph.edges:
        if e.source in adj and e.target in adj:
            adj[e.source].append((e.target, e.conductance_nS))
            adj[e.target].append((e.source, e.conductance_nS))

    updates: dict[str, float] = {}
    for nid, node in graph.nodes.items():
        # Leak toward resting potential
        leak = -coupling_strength * (node.vm_mV - node.vm_rest_mV)
        # Coupling from neighbors
        neighbor_drive = 0.0
        total_g = 0.0
        for nb_id, g in adj[nid]:
            nb = graph.nodes[nb_id]
            neighbor_drive += g * (nb.vm_mV - node.vm_mV)
            total_g += g
        if total_g > 0:
            neighbor_drive /= total_g
            neighbor_drive *= coupling_strength
        updates[nid] = node.vm_mV + leak + neighbor_drive

    deviation = 0.0
    for nid, new_vm in updates.items():
        graph.nodes[nid].vm_mV = new_vm
        deviation += abs(new_vm - graph.nodes[nid].vm_rest_mV)

    return deviation / len(graph.nodes) if graph.nodes else 0.0


def simulate_perturbation_recovery(
    graph: SubstrateGraph,
    perturbation_mV: float = 20.0,
    fraction_perturbed: float = 0.3,
    max_steps: int = 500,
    convergence_threshold_mV: float = 0.5,
    seed: Optional[int] = None,
) -> PerturbationResult:
    """Simulate perturbation and recovery on a substrate graph.

    Perturbs a fraction of nodes, then runs mean-field relaxation.
    """
    if seed is not None:
        random.seed(seed)

    node_ids = list(graph.nodes.keys())
    n_perturb = max(1, int(len(node_ids) * fraction_perturbed))
    perturbed = random.sample(node_ids, n_perturb)

    # Apply perturbation
    for nid in perturbed:
        graph.nodes[nid].vm_mV += perturbation_mV

    # Relaxation loop
    for step in range(max_steps):
        deviation = _mean_field_step(graph)
        if deviation < convergence_threshold_mV:
            # Compute residual
            residual = sum(
                abs(n.vm_mV - n.vm_rest_mV) for n in graph.nodes.values()
            ) / len(graph.nodes)
            return PerturbationResult(
                error_tolerance_fraction=fraction_perturbed,
                recovery_steps=step + 1,
                recovery_complete=True,
                residual_error_mV=round(residual, 4),
                attractor_basin_depth_mV=perturbation_mV,
                substrate_label=graph.nodes.get(node_ids[0], SubstrateNode(node_id="")).metadata.get("label", ""),
            )

    # Did not converge
    residual = sum(
        abs(n.vm_mV - n.vm_rest_mV) for n in graph.nodes.values()
    ) / len(graph.nodes)
    return PerturbationResult(
        error_tolerance_fraction=fraction_perturbed,
        recovery_steps=max_steps,
        recovery_complete=False,
        residual_error_mV=round(residual, 4),
        attractor_basin_depth_mV=perturbation_mV,
    )


def find_attractor_basin_depth(
    config: SubstrateConfig,
    max_perturbation_mV: float = 50.0,
    steps: int = 20,
    seed: Optional[int] = 42,
) -> float:
    """Find the maximum perturbation that still allows full recovery.

    Binary search over perturbation amplitude.
    """
    lo, hi = 0.0, max_perturbation_mV
    for _ in range(steps):
        mid = (lo + hi) / 2
        graph = build_substrate(config)
        result = simulate_perturbation_recovery(graph, perturbation_mV=mid, seed=seed)
        if result.recovery_complete:
            lo = mid
        else:
            hi = mid
    return round(lo, 2)


# ---------------------------------------------------------------------------
# Comparison: uniform vs heterogeneous
# ---------------------------------------------------------------------------

@dataclass
class HeterogeneityComparison:
    """Side-by-side comparison of uniform vs heterogeneous substrate."""

    uniform_result: PerturbationResult = field(default_factory=PerturbationResult)
    heterogeneous_result: PerturbationResult = field(default_factory=PerturbationResult)
    uniform_basin_depth_mV: float = 0.0
    heterogeneous_basin_depth_mV: float = 0.0
    # Stored result
    conclusion: str = "Natural disorder may increase distributed resilience."


def compare_substrates(
    n_nodes: int = 100,
    k_neighbors: int = 6,
    heterogeneity: Optional[HeterogeneityParams] = None,
    perturbation_mV: float = 15.0,
    seed: int = 42,
) -> HeterogeneityComparison:
    """Compare uniform and heterogeneous substrates under perturbation."""
    if heterogeneity is None:
        heterogeneity = HeterogeneityParams()

    uniform_config = SubstrateConfig(
        label="uniform",
        is_heterogeneous=False,
        n_nodes=n_nodes,
        k_neighbors=k_neighbors,
        seed=seed,
    )
    hetero_config = SubstrateConfig(
        label="heterogeneous",
        is_heterogeneous=True,
        heterogeneity=heterogeneity,
        n_nodes=n_nodes,
        k_neighbors=k_neighbors,
        seed=seed,
    )

    # Build and perturb uniform
    g_uniform = build_substrate(uniform_config)
    r_uniform = simulate_perturbation_recovery(g_uniform, perturbation_mV, seed=seed)
    r_uniform.substrate_label = "uniform"

    # Build and perturb heterogeneous
    g_hetero = build_substrate(hetero_config)
    r_hetero = simulate_perturbation_recovery(g_hetero, perturbation_mV, seed=seed)
    r_hetero.substrate_label = "heterogeneous"

    # Basin depths
    basin_uniform = find_attractor_basin_depth(uniform_config, seed=seed)
    basin_hetero = find_attractor_basin_depth(hetero_config, seed=seed)

    return HeterogeneityComparison(
        uniform_result=r_uniform,
        heterogeneous_result=r_hetero,
        uniform_basin_depth_mV=basin_uniform,
        heterogeneous_basin_depth_mV=basin_hetero,
    )
