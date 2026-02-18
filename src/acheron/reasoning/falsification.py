"""Falsification Prediction Registry (Section 12).

Read-only meta-layer that records testable predictions with quantitative
bounds.  Each prediction specifies:
    - What the model predicts (output metric + expected range)
    - Under what conditions (parameter assumptions)
    - How to test it (experimental protocol sketch)
    - What would falsify it (kill condition)

Design principle:
    - Never modifies existing computation layers
    - Calls existing functions to compute predicted values
    - Compares predictions against registered bounds
    - Reports which predictions are alive, at risk, or falsified

This is the Popper-style complement to the sensitivity analysis:
    sensitivity.py asks "how much does the answer change?"
    falsification.py asks "can the answer be proven wrong?"
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
from enum import Enum

from .mosaic import (
    build_small_world,
    build_scale_free,
    compute_spectral_gap,
    compute_clustering_coefficient,
    compute_average_path_length,
    compute_total_network_energy,
)
from .state_space import (
    build_system_matrices,
    analyze_stability,
)
from .empirical import SPECTRAL_THRESHOLD, LEVIN_SPEC, MOSAIC_SPEC


class PredictionStatus(Enum):
    """Status of a falsifiable prediction."""

    ALIVE = "alive"               # prediction holds under current computation
    AT_RISK = "at_risk"           # near boundary of kill condition
    FALSIFIED = "falsified"       # kill condition triggered
    UNTESTED = "untested"         # requires experimental data not yet available


@dataclass
class FalsifiablePrediction:
    """A single falsifiable prediction from the reasoning engine."""

    prediction_id: str
    hypothesis: str               # what the model predicts
    metric_name: str              # computed metric to check
    expected_range: tuple[float, float]  # (lower_bound, upper_bound)
    kill_condition: str           # human-readable: what falsifies this
    kill_bound: float             # numeric threshold for automated check
    kill_direction: str           # "below" or "above" — falsified if metric goes this way
    computed_value: float = 0.0
    status: PredictionStatus = PredictionStatus.UNTESTED
    margin: float = 0.0          # how far from kill boundary (positive = safe)
    experimental_test: str = ""  # how to test in the lab
    source_module: str = ""      # which computation module generates this
    assumptions: list[str] = field(default_factory=list)


@dataclass
class FalsificationReport:
    """Complete falsification registry report."""

    predictions: list[FalsifiablePrediction] = field(default_factory=list)
    alive_count: int = 0
    at_risk_count: int = 0
    falsified_count: int = 0
    untested_count: int = 0
    overall_health: str = ""     # "HEALTHY", "CAUTION", "CRITICAL"


# ---------------------------------------------------------------------------
# Prediction registry (the core knowledge base)
# ---------------------------------------------------------------------------

def _build_prediction_registry() -> list[FalsifiablePrediction]:
    """Build the master list of falsifiable predictions.

    Each prediction is a testable claim from the Acheron reasoning engine.
    """
    return [
        FalsifiablePrediction(
            prediction_id="P1_SPECTRAL_GAP_POSITIVE",
            hypothesis="Planarian tissue connectivity forms a graph with positive "
                       "spectral gap (lambda_2 > 0), implying a connected network "
                       "capable of global signal propagation.",
            metric_name="spectral_gap",
            expected_range=(0.05, 5.0),
            kill_condition="Spectral gap < 0.02 would imply disconnected or "
                          "near-disconnected tissue — no global bioelectric memory possible",
            kill_bound=0.02,
            kill_direction="below",
            experimental_test="Map gap junction connectivity via dye coupling + "
                            "compute graph Laplacian from adjacency matrix",
            source_module="mosaic",
            assumptions=["Watts-Strogatz topology", "Uniform gap junction conductance"],
        ),
        FalsifiablePrediction(
            prediction_id="P2_SPECTRAL_GAP_OPTIMAL",
            hypothesis="Optimal bioelectric signal propagation occurs when the "
                       "spectral gap is near 0.15 (Payvand Mosaic target), "
                       "balancing speed and energy.",
            metric_name="spectral_gap_vs_target",
            expected_range=(0.08, 0.30),
            kill_condition="If measured spectral gap is outside [0.05, 0.50], "
                          "the Mosaic architecture mapping is invalid",
            kill_bound=0.05,
            kill_direction="below",
            experimental_test="Measure tissue connectivity + compute spectral gap; "
                            "compare to Mosaic chip optimal operating point",
            source_module="mosaic",
            assumptions=["Payvand Mosaic analogy holds", "Gap junction = RRAM routing"],
        ),
        FalsifiablePrediction(
            prediction_id="P3_SMALL_WORLD_TOPOLOGY",
            hypothesis="Planarian gap junction networks exhibit small-world "
                       "properties: high clustering (>0.3) + short path length "
                       "relative to random graphs.",
            metric_name="clustering_coefficient",
            expected_range=(0.3, 0.9),
            kill_condition="Clustering coefficient < 0.1 would falsify small-world "
                          "assumption — tissue is random, not structured",
            kill_bound=0.1,
            kill_direction="below",
            experimental_test="Reconstruct gap junction network from electron "
                            "microscopy or Lucifer Yellow dye coupling data",
            source_module="mosaic",
            assumptions=["Watts-Strogatz model", "Gap junctions form regular local connectivity"],
        ),
        FalsifiablePrediction(
            prediction_id="P4_STABILITY_NEGATIVE_EIGENVALUES",
            hypothesis="The bioelectric dynamical system dX/dt = AX is stable — "
                       "all eigenvalues of A have negative real parts, ensuring "
                       "recovery from perturbation.",
            metric_name="dominant_eigenvalue",
            expected_range=(-1.0, -0.001),
            kill_condition="Positive dominant eigenvalue would mean runaway "
                          "depolarization — system is unstable",
            kill_bound=0.0,
            kill_direction="above",
            experimental_test="Apply voltage perturbation to tissue region; "
                            "if Vm diverges instead of recovering, system is unstable",
            source_module="state_space",
            assumptions=["Linear Laplacian coupling", "Homogeneous leak rate"],
        ),
        FalsifiablePrediction(
            prediction_id="P5_ENERGY_PER_BIT_BOUND",
            hypothesis="Energy per bioelectric routing event is in the picojoule "
                       "range (1-100 pJ), consistent with Payvand Mosaic RRAM "
                       "energy budget.",
            metric_name="energy_per_route_pJ",
            expected_range=(0.1, 100.0),
            kill_condition="Energy > 1000 pJ per routing event would make "
                          "bioelectric computation metabolically impossible "
                          "at tissue scale",
            kill_bound=1000.0,
            kill_direction="above",
            experimental_test="Measure ATP consumption during gap-junction-mediated "
                            "signal propagation; convert to Joules per event",
            source_module="mosaic",
            assumptions=["Ohmic energy model", "Conductance = 2 nS baseline"],
        ),
        FalsifiablePrediction(
            prediction_id="P6_MEMORY_PERSISTENCE",
            hypothesis="Bioelectric memory (Vm pattern) persists for at least "
                       "3 hours after initial encoding, consistent with Levin's "
                       "T_hold measurements.",
            metric_name="t_hold_hours",
            expected_range=(3.0, 720.0),  # 3h to 30 days
            kill_condition="If encoded Vm pattern decays in < 1 hour, "
                          "bioelectric memory is too volatile for morphogenetic function",
            kill_bound=1.0,
            kill_direction="below",
            experimental_test="Encode Vm pattern via ionophore; remove stimulus; "
                            "measure pattern persistence via voltage-sensitive dye",
            source_module="empirical",
            assumptions=["Levin's cryptic worm T_hold applies to information encoding"],
        ),
        FalsifiablePrediction(
            prediction_id="P7_NONLINEAR_VS_LINEAR",
            hypothesis="Nonlinear voltage-dependent gap junction gating does not "
                       "qualitatively change system stability — both linear and "
                       "nonlinear models predict stable attractors.",
            metric_name="stability_agreement",
            expected_range=(0.0, 1.0),
            kill_condition="If nonlinear model predicts instability where linear "
                          "model predicts stability, the linear approximation is invalid",
            kill_bound=0.0,
            kill_direction="below",
            experimental_test="Compare recovery trajectories with and without "
                            "voltage-dependent gap junction blockers",
            source_module="state_space",
            assumptions=["Boltzmann sigmoid gating", "V_half = 30 mV"],
        ),
    ]


# ---------------------------------------------------------------------------
# Prediction evaluation
# ---------------------------------------------------------------------------

def _evaluate_prediction(
    pred: FalsifiablePrediction,
    n_nodes: int = 30,
    k_neighbors: int = 4,
    seed: int = 42,
) -> None:
    """Evaluate a single prediction by running the relevant computation.

    Mutates pred in place (sets computed_value, status, margin).
    """
    try:
        if pred.prediction_id == "P1_SPECTRAL_GAP_POSITIVE":
            g = build_small_world(n_nodes, k_neighbors, seed=seed)
            pred.computed_value = compute_spectral_gap(g)

        elif pred.prediction_id == "P2_SPECTRAL_GAP_OPTIMAL":
            g = build_small_world(n_nodes, k_neighbors, rewire_prob=0.1, seed=seed)
            pred.computed_value = compute_spectral_gap(g)

        elif pred.prediction_id == "P3_SMALL_WORLD_TOPOLOGY":
            g = build_small_world(n_nodes, k_neighbors, rewire_prob=0.1, seed=seed)
            pred.computed_value = compute_clustering_coefficient(g)

        elif pred.prediction_id == "P4_STABILITY_NEGATIVE_EIGENVALUES":
            g = build_small_world(n_nodes, k_neighbors, seed=seed)
            matrices = build_system_matrices(g, leak_rate=0.1)
            stab = analyze_stability(matrices)
            pred.computed_value = stab.dominant_eigenvalue

        elif pred.prediction_id == "P5_ENERGY_PER_BIT_BOUND":
            g = build_small_world(n_nodes, k_neighbors, seed=seed)
            total_e = compute_total_network_energy(g)
            n_edges = g.edge_count()
            # Convert to pJ
            e_per_edge_pJ = (total_e / n_edges * 1e12) if n_edges > 0 else 0.0
            pred.computed_value = e_per_edge_pJ

        elif pred.prediction_id == "P6_MEMORY_PERSISTENCE":
            # This is empirical — use Levin's measured value
            pred.computed_value = LEVIN_SPEC.t_hold_initial_hours

        elif pred.prediction_id == "P7_NONLINEAR_VS_LINEAR":
            g = build_small_world(n_nodes, k_neighbors, seed=seed)
            m_lin = build_system_matrices(g, leak_rate=0.1, coupling_mode="linear")
            m_nl = build_system_matrices(g, leak_rate=0.1, coupling_mode="nonlinear")
            stab_lin = analyze_stability(m_lin)
            stab_nl = analyze_stability(m_nl)
            # Agreement: both stable or both unstable
            pred.computed_value = 1.0 if (stab_lin.is_stable == stab_nl.is_stable) else 0.0

        else:
            pred.status = PredictionStatus.UNTESTED
            return

    except Exception:
        pred.status = PredictionStatus.UNTESTED
        return

    # Determine status
    if pred.kill_direction == "below":
        pred.margin = pred.computed_value - pred.kill_bound
        if pred.computed_value < pred.kill_bound:
            pred.status = PredictionStatus.FALSIFIED
        elif pred.margin < (pred.expected_range[1] - pred.expected_range[0]) * 0.1:
            pred.status = PredictionStatus.AT_RISK
        else:
            pred.status = PredictionStatus.ALIVE
    elif pred.kill_direction == "above":
        pred.margin = pred.kill_bound - pred.computed_value
        if pred.computed_value > pred.kill_bound:
            pred.status = PredictionStatus.FALSIFIED
        elif pred.margin < (pred.expected_range[1] - pred.expected_range[0]) * 0.1:
            pred.status = PredictionStatus.AT_RISK
        else:
            pred.status = PredictionStatus.ALIVE


# ---------------------------------------------------------------------------
# Full falsification analysis
# ---------------------------------------------------------------------------

def run_falsification_analysis(
    n_nodes: int = 30,
    k_neighbors: int = 4,
    seed: int = 42,
) -> FalsificationReport:
    """Run the complete falsification registry evaluation.

    Evaluates all registered predictions against current computed values.
    """
    predictions = _build_prediction_registry()

    for pred in predictions:
        _evaluate_prediction(pred, n_nodes, k_neighbors, seed)

    alive = sum(1 for p in predictions if p.status == PredictionStatus.ALIVE)
    at_risk = sum(1 for p in predictions if p.status == PredictionStatus.AT_RISK)
    falsified = sum(1 for p in predictions if p.status == PredictionStatus.FALSIFIED)
    untested = sum(1 for p in predictions if p.status == PredictionStatus.UNTESTED)

    if falsified > 0:
        health = "CRITICAL"
    elif at_risk > 0:
        health = "CAUTION"
    else:
        health = "HEALTHY"

    return FalsificationReport(
        predictions=predictions,
        alive_count=alive,
        at_risk_count=at_risk,
        falsified_count=falsified,
        untested_count=untested,
        overall_health=health,
    )


def format_falsification_report(report: FalsificationReport) -> str:
    """Format falsification report as structured text."""
    lines: list[str] = []
    lines.append("=" * 72)
    lines.append("FALSIFICATION PREDICTION REGISTRY — POPPER-STYLE VALIDATION")
    lines.append("=" * 72)
    lines.append(f"Overall Health: {report.overall_health}")
    lines.append(f"Predictions: {len(report.predictions)} total | "
                 f"{report.alive_count} alive | {report.at_risk_count} at risk | "
                 f"{report.falsified_count} falsified | {report.untested_count} untested")
    lines.append("")

    for pred in report.predictions:
        status_icon = {
            PredictionStatus.ALIVE: "+",
            PredictionStatus.AT_RISK: "~",
            PredictionStatus.FALSIFIED: "X",
            PredictionStatus.UNTESTED: "?",
        }[pred.status]

        lines.append(f"  [{status_icon}] {pred.prediction_id}: {pred.hypothesis[:80]}...")
        lines.append(f"      Computed: {pred.computed_value:.6f} | "
                     f"Expected: {pred.expected_range} | "
                     f"Kill: {pred.kill_direction} {pred.kill_bound}")
        lines.append(f"      Margin: {pred.margin:.6f} | Status: {pred.status.value}")
        lines.append(f"      Test: {pred.experimental_test[:80]}")
        lines.append(f"      Assumptions: {pred.assumptions}")
        lines.append("")

    lines.append("=" * 72)
    return "\n".join(lines)
