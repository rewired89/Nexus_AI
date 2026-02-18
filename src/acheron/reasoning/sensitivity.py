"""Parameter Sensitivity Analysis (Section 11).

Meta-layer that wraps existing computation functions with perturbed
parameters to quantify how sensitive outputs are to modeling assumptions.

Design principle: READ-ONLY over existing layers.
    - Calls existing functions with varied inputs
    - Never modifies existing computation paths
    - Reports which assumptions matter most (high sensitivity)
    - Identifies robust conclusions (low sensitivity)

Key analyses:
    1. Spectral gap sensitivity to conductance variance
    2. Stability sensitivity to leak rate
    3. Topology metrics sensitivity to rewire probability
    4. Energy sensitivity to conductance scale
    5. Cross-model comparison (WS vs BA vs ER)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

from .substrate import SubstrateGraph
from .mosaic import (
    build_small_world,
    build_scale_free,
    build_uniform_random,
    compute_spectral_gap,
    compute_total_network_energy,
    analyze_spectral_properties,
    compute_clustering_coefficient,
    compute_average_path_length,
    SpectralAnalysis,
)
from .state_space import (
    build_system_matrices,
    analyze_stability,
    StabilityAnalysis,
)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ParameterSweepPoint:
    """Single point in a parameter sweep."""

    parameter_name: str
    parameter_value: float
    output_name: str
    output_value: float


@dataclass
class SensitivityResult:
    """Result of a single sensitivity analysis."""

    parameter_name: str
    baseline_value: float
    output_name: str
    sweep_points: list[ParameterSweepPoint] = field(default_factory=list)
    sensitivity_index: float = 0.0  # normalized: |d(output)/d(param)| * param/output
    robust: bool = True             # True if sensitivity_index < threshold


@dataclass
class ModelComparisonResult:
    """Cross-model comparison for a single metric."""

    metric_name: str
    watts_strogatz: float = 0.0
    barabasi_albert: float = 0.0
    erdos_renyi: float = 0.0
    agreement: float = 0.0         # 0-1: how close the models agree
    dominant_model: str = ""        # which model predicts highest value


@dataclass
class SensitivityReport:
    """Complete sensitivity analysis report."""

    parameter_sensitivities: list[SensitivityResult] = field(default_factory=list)
    model_comparisons: list[ModelComparisonResult] = field(default_factory=list)
    high_sensitivity_params: list[str] = field(default_factory=list)
    robust_conclusions: list[str] = field(default_factory=list)
    assumptions_at_risk: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Core sweep engine
# ---------------------------------------------------------------------------

def _sweep_parameter(
    baseline: float,
    perturbation_fractions: list[float],
    compute_fn,
    param_name: str,
    output_name: str,
) -> SensitivityResult:
    """Sweep a single parameter and compute sensitivity index.

    perturbation_fractions: e.g., [-0.5, -0.2, 0.0, 0.2, 0.5]
        meaning baseline * (1 + fraction).

    compute_fn(value) -> float: function that takes the parameter value
        and returns the output metric.
    """
    points: list[ParameterSweepPoint] = []
    outputs: list[float] = []

    for frac in perturbation_fractions:
        val = baseline * (1.0 + frac)
        if val <= 0:
            continue
        try:
            out = compute_fn(val)
        except Exception:
            continue
        points.append(ParameterSweepPoint(
            parameter_name=param_name,
            parameter_value=round(val, 8),
            output_name=output_name,
            output_value=round(out, 8),
        ))
        outputs.append(out)

    # Compute normalized sensitivity: |d(output)/d(param)| * param/output
    # Use finite difference at baseline
    baseline_idx = None
    for i, p in enumerate(points):
        if abs(p.parameter_value - baseline) < 1e-12:
            baseline_idx = i
            break

    sensitivity_index = 0.0
    if baseline_idx is not None and len(outputs) >= 2:
        baseline_out = outputs[baseline_idx]
        if abs(baseline_out) > 1e-15:
            # Compute average slope across sweep
            slopes: list[float] = []
            for i, p in enumerate(points):
                if i == baseline_idx:
                    continue
                dp = p.parameter_value - baseline
                do = outputs[i] - baseline_out
                if abs(dp) > 1e-15:
                    slopes.append(abs(do / dp) * baseline / abs(baseline_out))
            if slopes:
                sensitivity_index = sum(slopes) / len(slopes)

    return SensitivityResult(
        parameter_name=param_name,
        baseline_value=baseline,
        output_name=output_name,
        sweep_points=points,
        sensitivity_index=round(sensitivity_index, 4),
        robust=sensitivity_index < 0.5,
    )


# ---------------------------------------------------------------------------
# Specific sensitivity analyses
# ---------------------------------------------------------------------------

PERTURBATION_FRACS = [-0.5, -0.3, -0.1, 0.0, 0.1, 0.3, 0.5]


def sensitivity_conductance_to_spectral_gap(
    n_nodes: int = 30,
    k_neighbors: int = 4,
    baseline_conductance_nS: float = 2.0,
    seed: int = 42,
) -> SensitivityResult:
    """How sensitive is spectral gap to gap junction conductance?"""

    def compute(cond_nS: float) -> float:
        g = build_small_world(n_nodes, k_neighbors, seed=seed,
                              base_conductance_nS=cond_nS)
        return compute_spectral_gap(g)

    return _sweep_parameter(
        baseline_conductance_nS, PERTURBATION_FRACS,
        compute, "base_conductance_nS", "spectral_gap",
    )


def sensitivity_leak_rate_to_stability(
    n_nodes: int = 20,
    k_neighbors: int = 4,
    baseline_leak_rate: float = 0.1,
    seed: int = 42,
) -> SensitivityResult:
    """How sensitive is system stability to leak rate?"""
    graph = build_small_world(n_nodes, k_neighbors, seed=seed)

    def compute(leak: float) -> float:
        matrices = build_system_matrices(graph, leak_rate=leak)
        stab = analyze_stability(matrices)
        return stab.dominant_eigenvalue

    return _sweep_parameter(
        baseline_leak_rate, PERTURBATION_FRACS,
        compute, "leak_rate", "dominant_eigenvalue",
    )


def sensitivity_rewire_prob_to_clustering(
    n_nodes: int = 30,
    k_neighbors: int = 4,
    baseline_rewire_prob: float = 0.1,
    seed: int = 42,
) -> SensitivityResult:
    """How sensitive is clustering coefficient to rewire probability?"""

    def compute(beta: float) -> float:
        beta = max(0.0, min(1.0, beta))
        g = build_small_world(n_nodes, k_neighbors, rewire_prob=beta, seed=seed)
        return compute_clustering_coefficient(g)

    return _sweep_parameter(
        baseline_rewire_prob, PERTURBATION_FRACS,
        compute, "rewire_prob", "clustering_coefficient",
    )


def sensitivity_conductance_scale_to_energy(
    n_nodes: int = 20,
    k_neighbors: int = 4,
    baseline_delta_v_mV: float = 30.0,
    seed: int = 42,
) -> SensitivityResult:
    """How sensitive is total energy to voltage difference?"""
    graph = build_small_world(n_nodes, k_neighbors, seed=seed)

    def compute(dv: float) -> float:
        return compute_total_network_energy(graph, delta_v_mV=dv)

    return _sweep_parameter(
        baseline_delta_v_mV, PERTURBATION_FRACS,
        compute, "delta_v_mV", "total_network_energy_J",
    )


# ---------------------------------------------------------------------------
# Cross-model topology comparison
# ---------------------------------------------------------------------------

def compare_topology_models(
    n_nodes: int = 30,
    k_neighbors: int = 4,
    seed: int = 42,
) -> list[ModelComparisonResult]:
    """Compare Watts-Strogatz, Barabasi-Albert, and Erdos-Renyi topologies.

    Computes spectral gap, clustering coefficient, average path length,
    and total energy for each model.  Returns agreement scores.
    """
    n_edges = n_nodes * k_neighbors // 2
    m_per_node = max(1, k_neighbors // 2)

    ws = build_small_world(n_nodes, k_neighbors, rewire_prob=0.1, seed=seed)
    ba = build_scale_free(n_nodes, m_edges_per_node=m_per_node, seed=seed)
    er = build_uniform_random(n_nodes, n_edges, seed=seed)

    metrics: list[tuple[str, float, float, float]] = []

    # Spectral gap
    gap_ws = compute_spectral_gap(ws)
    gap_ba = compute_spectral_gap(ba)
    gap_er = compute_spectral_gap(er)
    metrics.append(("spectral_gap", gap_ws, gap_ba, gap_er))

    # Clustering coefficient
    cc_ws = compute_clustering_coefficient(ws)
    cc_ba = compute_clustering_coefficient(ba)
    cc_er = compute_clustering_coefficient(er)
    metrics.append(("clustering_coefficient", cc_ws, cc_ba, cc_er))

    # Average path length
    apl_ws = compute_average_path_length(ws)
    apl_ba = compute_average_path_length(ba)
    apl_er = compute_average_path_length(er)
    metrics.append(("average_path_length", apl_ws, apl_ba, apl_er))

    # Total energy
    e_ws = compute_total_network_energy(ws)
    e_ba = compute_total_network_energy(ba)
    e_er = compute_total_network_energy(er)
    metrics.append(("total_network_energy_J", e_ws, e_ba, e_er))

    results: list[ModelComparisonResult] = []
    for name, v_ws, v_ba, v_er in metrics:
        vals = [v_ws, v_ba, v_er]
        mean = sum(vals) / 3.0
        if mean > 1e-15:
            spread = max(vals) - min(vals)
            agreement = max(0.0, 1.0 - spread / mean)
        else:
            agreement = 1.0

        names = ["watts_strogatz", "barabasi_albert", "erdos_renyi"]
        max_idx = vals.index(max(vals))
        dominant = names[max_idx]

        results.append(ModelComparisonResult(
            metric_name=name,
            watts_strogatz=round(v_ws, 8),
            barabasi_albert=round(v_ba, 8),
            erdos_renyi=round(v_er, 8),
            agreement=round(agreement, 4),
            dominant_model=dominant,
        ))

    return results


# ---------------------------------------------------------------------------
# Full sensitivity report
# ---------------------------------------------------------------------------

SENSITIVITY_THRESHOLD = 0.5  # above this, parameter is "high sensitivity"

_ASSUMPTION_REGISTRY: list[tuple[str, str, str]] = [
    ("Linear Laplacian coupling",
     "Gap junctions modeled as ohmic resistors with constant conductance",
     "conductance_to_spectral_gap"),
    ("Homogeneous leak rate",
     "All cells assumed to have same leak rate constant",
     "leak_rate_to_stability"),
    ("Watts-Strogatz topology",
     "Tissue connectivity assumed to follow small-world model",
     "rewire_prob_to_clustering"),
    ("Ohmic energy model",
     "Energy = G * V^2 * t (purely resistive dissipation)",
     "conductance_scale_to_energy"),
]


def run_sensitivity_analysis(
    n_nodes: int = 30,
    k_neighbors: int = 4,
    seed: int = 42,
) -> SensitivityReport:
    """Run complete sensitivity analysis across all modeling assumptions.

    This is the main entry point. Wraps existing computation functions
    with perturbed parameters — purely additive, never modifies existing paths.
    """
    report = SensitivityReport()

    # Parameter sensitivities
    s1 = sensitivity_conductance_to_spectral_gap(n_nodes, k_neighbors, seed=seed)
    s2 = sensitivity_leak_rate_to_stability(min(n_nodes, 20), k_neighbors, seed=seed)
    s3 = sensitivity_rewire_prob_to_clustering(n_nodes, k_neighbors, seed=seed)
    s4 = sensitivity_conductance_scale_to_energy(min(n_nodes, 20), k_neighbors, seed=seed)
    report.parameter_sensitivities = [s1, s2, s3, s4]

    # Cross-model comparison
    report.model_comparisons = compare_topology_models(n_nodes, k_neighbors, seed=seed)

    # Classify
    for s in report.parameter_sensitivities:
        if not s.robust:
            report.high_sensitivity_params.append(s.parameter_name)

    # Map assumptions to risk
    for assumption_name, description, analysis_key in _ASSUMPTION_REGISTRY:
        matching = [s for s in report.parameter_sensitivities
                    if analysis_key.startswith(s.parameter_name.split("_")[0])]
        # Simpler: check by index
        idx_map = {
            "conductance_to_spectral_gap": 0,
            "leak_rate_to_stability": 1,
            "rewire_prob_to_clustering": 2,
            "conductance_scale_to_energy": 3,
        }
        idx = idx_map.get(analysis_key)
        if idx is not None and idx < len(report.parameter_sensitivities):
            s = report.parameter_sensitivities[idx]
            if not s.robust:
                report.assumptions_at_risk.append(
                    f"{assumption_name}: sensitivity={s.sensitivity_index} "
                    f"({description})"
                )
            else:
                report.robust_conclusions.append(
                    f"{assumption_name}: sensitivity={s.sensitivity_index} — ROBUST"
                )

    # Model agreement conclusions
    for mc in report.model_comparisons:
        if mc.agreement > 0.7:
            report.robust_conclusions.append(
                f"{mc.metric_name}: models agree ({mc.agreement:.0%}) — TOPOLOGY-INDEPENDENT"
            )
        else:
            report.assumptions_at_risk.append(
                f"{mc.metric_name}: models disagree ({mc.agreement:.0%}) — "
                f"TOPOLOGY-DEPENDENT (dominant: {mc.dominant_model})"
            )

    return report


def format_sensitivity_report(report: SensitivityReport) -> str:
    """Format sensitivity report as structured text."""
    lines: list[str] = []
    lines.append("=" * 72)
    lines.append("PARAMETER SENSITIVITY ANALYSIS — MODEL VALIDATION META-LAYER")
    lines.append("=" * 72)
    lines.append("")

    lines.append("A. PARAMETER SENSITIVITIES")
    lines.append("-" * 72)
    for s in report.parameter_sensitivities:
        status = "ROBUST" if s.robust else "HIGH SENSITIVITY"
        lines.append(f"  {s.parameter_name} → {s.output_name}")
        lines.append(f"    Baseline: {s.baseline_value}")
        lines.append(f"    Sensitivity Index: {s.sensitivity_index} [{status}]")
        lines.append(f"    Sweep Points: {len(s.sweep_points)}")
    lines.append("")

    lines.append("B. CROSS-MODEL TOPOLOGY COMPARISON")
    lines.append("-" * 72)
    for mc in report.model_comparisons:
        lines.append(f"  {mc.metric_name}:")
        lines.append(f"    Watts-Strogatz:  {mc.watts_strogatz}")
        lines.append(f"    Barabasi-Albert: {mc.barabasi_albert}")
        lines.append(f"    Erdos-Renyi:     {mc.erdos_renyi}")
        lines.append(f"    Agreement: {mc.agreement:.0%}, Dominant: {mc.dominant_model}")
    lines.append("")

    if report.robust_conclusions:
        lines.append("C. ROBUST CONCLUSIONS")
        lines.append("-" * 72)
        for c in report.robust_conclusions:
            lines.append(f"  + {c}")
        lines.append("")

    if report.assumptions_at_risk:
        lines.append("D. ASSUMPTIONS AT RISK")
        lines.append("-" * 72)
        for a in report.assumptions_at_risk:
            lines.append(f"  ! {a}")
        lines.append("")

    lines.append("=" * 72)
    return "\n".join(lines)
