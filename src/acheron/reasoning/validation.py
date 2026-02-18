"""Validation Report Generator (Section 9).

Generates internal validation report after integration:
    - Updated modeling layers
    - New variables introduced
    - Cross-layer interaction map
    - Simulation readiness status
    - Missing experimental data required

Structured technical output only.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any, Optional

from .substrate import SubstrateGraph
from .mosaic import (
    TopologyMetrics,
    TopologyComparison,
    SpectralAnalysis,
    RewireAnalysis,
    build_small_world,
    compare_topologies,
    analyze_topology,
    analyze_spectral_properties,
    find_optimal_rewires,
)
from .empirical import (
    MOSAIC_SPEC,
    LEVIN_SPEC,
    SPECTRAL_THRESHOLD,
    EMPIRICAL_PARAMS,
    format_cross_species_table,
)
from .sensitivity import (
    SensitivityReport,
    run_sensitivity_analysis,
    format_sensitivity_report,
)
from .falsification import (
    FalsificationReport,
    run_falsification_analysis,
    format_falsification_report,
    PredictionStatus,
)
from .denram import (
    DelayEncodingMetrics,
    EncodingMode,
    analyze_delay_encoding,
)
from .heterogeneity import (
    HeterogeneityComparison,
    HeterogeneityParams,
    compare_substrates,
)
from .homeostatic import (
    HomeostaticRecoveryResult,
    HomeostaticParams,
    simulate_adversarial_recovery,
    simulate_repeated_perturbation,
)
from .state_space import (
    StateSpaceAnalysis,
    analyze_state_space,
)
from .ribozyme import (
    QT45Analysis,
    QT45Parameters,
    analyze_qt45,
)
from .freeze_thaw import (
    FreezeThawAnalysis,
    FreezeThawParams,
    analyze_freeze_thaw,
)


# ---------------------------------------------------------------------------
# Modeling layer registry
# ---------------------------------------------------------------------------

@dataclass
class ModelingLayer:
    """Description of a single modeling layer in the reasoning engine."""

    layer_id: str
    name: str
    section: str                                # e.g., "Section 1"
    module: str                                 # Python module name
    description: str
    variables_introduced: list[str] = field(default_factory=list)
    simulation_ready: bool = False
    missing_experimental_data: list[str] = field(default_factory=list)
    depends_on: list[str] = field(default_factory=list)     # layer_ids
    feeds_into: list[str] = field(default_factory=list)     # layer_ids


MODELING_LAYERS: list[ModelingLayer] = [
    ModelingLayer(
        layer_id="substrate",
        name="Substrate=Algorithm Core",
        section="Section 1",
        module="acheron.reasoning.substrate",
        description="Topology and physical state transitions as computational primitives. "
                    "Connectivity graph mutations = memory writes. Gap junction conductance "
                    "modulation = routing reconfiguration. Physical delay = temporal encoding.",
        variables_introduced=[
            "TopologyEventType (6 primitives)",
            "ComputationalPrimitive (4 semantics)",
            "SubstrateNode.vm_mV",
            "SubstrateNode.vm_rest_mV",
            "SubstrateNode.ion_channel_expression",
            "SubstrateEdge.conductance_nS",
            "SubstrateEdge.diffusion_delay_ms",
        ],
        simulation_ready=True,
        missing_experimental_data=[
            "Gap junction conductance measurements per cell type (nS)",
            "Signal propagation delay measurements through gap junctions (ms)",
            "Ion channel expression profiles per neoblast",
        ],
        feeds_into=["mosaic", "denram", "state_space"],
    ),
    ModelingLayer(
        layer_id="mosaic",
        name="Mosaic Small-World + Spectral Analysis",
        section="Section 2",
        module="acheron.reasoning.mosaic",
        description="Watts-Strogatz small-world graph abstraction with graph Laplacian spectral "
                    "analysis and energy-optimized edge rewiring. Implements Payvand Mosaic "
                    "in-memory routing principle. Nodes = cell clusters, Edges = weighted gap "
                    "junction conductance. Computes spectral gap (lambda_2), energy-delay product, "
                    "and optimal rewiring events.",
        variables_introduced=[
            "TopologyMetrics.clustering_coefficient",
            "TopologyMetrics.average_path_length",
            "TopologyMetrics.signal_propagation_latency_ms",
            "TopologyMetrics.energy_per_routing_event_J",
            "TopologyMetrics.error_cascade_probability",
            "TopologyMetrics.fault_tolerance_fraction",
            "SpectralAnalysis.spectral_gap (lambda_2)",
            "SpectralAnalysis.laplacian_eigenvalues",
            "SpectralAnalysis.total_network_energy_J",
            "SpectralAnalysis.spectral_efficiency",
            "RewireAnalysis.optimal_rewires",
            "RewireAnalysis.gap_improvement_pct",
            "RewireAnalysis.energy_reduction_pct",
            "rewire_prob (Watts-Strogatz beta)",
            "long_range_conductance_nS",
            "long_range_delay_ms",
        ],
        simulation_ready=True,
        missing_experimental_data=[
            "Actual gap junction topology map for planarian tissue",
            "Measured clustering coefficient in vivo",
            "Long-range gap junction conductance values",
        ],
        depends_on=["substrate"],
        feeds_into=["heterogeneity", "state_space", "empirical"],
    ),
    ModelingLayer(
        layer_id="denram",
        name="DenRAM Bioelectric Delay Encoding",
        section="Section 3",
        module="acheron.reasoning.denram",
        description="Temporal encoding via static attractors (Vm steady state) or "
                    "phase-delay encoding (oscillatory timing). Ion channel kinetics as "
                    "delay variables, gap junction diffusion as delay constants.",
        variables_introduced=[
            "EncodingMode (static_attractor | phase_delay)",
            "ChannelKinetics.tau_activation_ms",
            "ChannelKinetics.tau_inactivation_ms",
            "GapJunctionDelay.diffusion_time_ms",
            "AttractorState.basin_width_mV",
            "AttractorState.energy_barrier_kT",
            "PhaseEncoder.frequency_Hz",
            "PhaseEncoder.n_phase_bins",
            "DelayEncodingMetrics.phase_stability_score",
            "DelayEncodingMetrics.cross_talk_probability",
        ],
        simulation_ready=True,
        missing_experimental_data=[
            "Ion channel activation/inactivation kinetics in planarian cells",
            "Oscillation frequency measurements in planarian tissue",
            "Phase coherence measurements between cell clusters",
        ],
        depends_on=["substrate"],
        feeds_into=["heterogeneity", "state_space"],
    ),
    ModelingLayer(
        layer_id="heterogeneity",
        name="Heterogeneity-Aware Computation",
        section="Section 4",
        module="acheron.reasoning.heterogeneity",
        description="Variability modeling: Vm_rest variance, gap junction variance, "
                    "ion expression variability. Compares uniform vs heterogeneous substrate "
                    "for error tolerance, recovery, and attractor basin depth.",
        variables_introduced=[
            "HeterogeneityParams.vm_rest_variance_mV",
            "HeterogeneityParams.gap_junction_variance_pct",
            "HeterogeneityParams.ion_expression_variability",
            "PerturbationResult.error_tolerance_fraction",
            "PerturbationResult.recovery_steps",
            "PerturbationResult.attractor_basin_depth_mV",
        ],
        simulation_ready=True,
        missing_experimental_data=[
            "Cell-to-cell Vm_rest variance in planarian neoblasts (mV)",
            "Gap junction conductance variance across tissue (%)",
            "Ion channel expression variability factor across neoblasts",
        ],
        depends_on=["substrate", "mosaic"],
        feeds_into=["homeostatic", "state_space"],
    ),
    ModelingLayer(
        layer_id="homeostatic",
        name="Homeostatic Recovery Model (BESS)",
        section="Section 5",
        module="acheron.reasoning.homeostatic",
        description="Feedback scaling: downscale if activity > threshold, upscale if "
                    "below basal range. Simulates adversarial recovery, repeated "
                    "perturbation stability, and energy cost of correction.",
        variables_introduced=[
            "HomeostaticParams.activity_threshold_upper",
            "HomeostaticParams.activity_threshold_lower",
            "HomeostaticParams.scaling_factor_down",
            "HomeostaticParams.scaling_factor_up",
            "NodeExcitability.excitability",
            "HomeostaticRecoveryResult.recovery_steps",
            "HomeostaticRecoveryResult.total_energy_J",
            "HomeostaticRecoveryResult.activity_variance",
        ],
        simulation_ready=True,
        missing_experimental_data=[
            "Homeostatic scaling timescale in planarian tissue",
            "Basal activity range for planarian neoblasts",
            "ATP cost of homeostatic correction per cell",
        ],
        depends_on=["substrate", "heterogeneity"],
        feeds_into=["state_space"],
    ),
    ModelingLayer(
        layer_id="state_space",
        name="State-Space Dynamical Framework",
        section="Section 6",
        module="acheron.reasoning.state_space",
        description="dX/dt = A*X + B*U dynamical system. X = bioelectric state vector, "
                    "A = connectivity matrix, U = external modulation. Eigenvalue stability "
                    "analysis, attractor identification, perturbation recovery trajectories.",
        variables_introduced=[
            "StateVector (Vm per node)",
            "SystemMatrices.A (connectivity + leak)",
            "SystemMatrices.B (input coupling)",
            "StabilityAnalysis.eigenvalues",
            "StabilityAnalysis.is_stable",
            "StabilityAnalysis.spectral_gap",
            "Attractor.basin_radius_mV",
            "RecoveryTrajectory.trajectory_norms",
        ],
        simulation_ready=True,
        missing_experimental_data=[
            "Leak rate constants per cell type",
            "Conductance-to-coupling scale factor (empirical)",
            "Measured attractor states in planarian tissue",
        ],
        depends_on=["substrate", "mosaic", "denram", "heterogeneity", "homeostatic"],
    ),
    ModelingLayer(
        layer_id="ribozyme",
        name="QT45 Ribozyme Integration",
        section="Section 7",
        module="acheron.reasoning.ribozyme",
        description="Molecular storage: 45 nt ribozyme, fidelity 0.941/nt, 72-day copy "
                    "cycle, 0.2% yield. Redundancy calculation, error correction overhead, "
                    "cross-chiral security model with 13 triplets + 1 hexamer.",
        variables_introduced=[
            "QT45Parameters.fidelity_per_nt (0.941)",
            "QT45Parameters.copy_cycle_days (72)",
            "QT45Parameters.yield_fraction (0.002)",
            "RedundancyAnalysis.copies_needed",
            "ErrorCorrectionOverhead.parity_nucleotides_needed",
            "CorruptionProbability.p_corruption_after_N",
            "CrossChiralSubstrate.triplets (13)",
            "CrossChiralSubstrate.hexamer (GCGCGC)",
            "MolecularAccessModel.degradation_resistance",
        ],
        simulation_ready=True,
        missing_experimental_data=[
            "Measured fidelity per nucleotide for QT45 in vivo",
            "Actual yield under controlled conditions",
            "Cross-chiral triplet synthesis efficiency",
        ],
        feeds_into=["freeze_thaw"],
    ),
    ModelingLayer(
        layer_id="freeze_thaw",
        name="Freeze-Thaw Kinetic Model",
        section="Section 8",
        module="acheron.reasoning.freeze_thaw",
        description="Environmental kinetics at -7C eutectic ice. Strand separation "
                    "probability, triplet kinetic trapping, replication stall modeling, "
                    "optimal freeze-thaw interval determination.",
        variables_introduced=[
            "FreezeThawParams.freeze_temp_C (-7)",
            "FreezeThawParams.triplet_binding_energy_kJ_per_mol",
            "FreezeThawParams.activation_energy_kJ_per_mol",
            "TripletTrapping.trapping_probability",
            "stall_probability",
            "strand_separation_probability",
            "eutectic_concentration_factor",
            "FreezeThawInterval.optimal_freeze_hours",
            "FreezeThawInterval.optimal_thaw_hours",
        ],
        simulation_ready=True,
        missing_experimental_data=[
            "Measured strand separation probability at eutectic temperature",
            "Triplet binding energy measurements (kJ/mol)",
            "Activation energy for QT45 replication (kJ/mol)",
            "Eutectic ice pocket concentration factor (measured)",
        ],
        depends_on=["ribozyme"],
    ),
    ModelingLayer(
        layer_id="empirical",
        name="Empirical Grounding Layer",
        section="Section 10",
        module="acheron.reasoning.empirical",
        description="Certified reference parameters from Payvand Mosaic architecture and "
                    "Levin bioelectric research. Cross-species comparison table (Planarian, "
                    "Xenopus, Physarum, Organoid). Spectral threshold hypothesis model with "
                    "kill conditions. Fills UNKNOWN slots with [MEASURED], [BOUNDED-INFERENCE], "
                    "or [TRANSFER] tagged values.",
        variables_introduced=[
            "MosaicArchitectureSpec (energy_per_route, spectral_gap_target)",
            "BioelectricMemorySpec (t_hold, vmem, gap_junction_conductance)",
            "CrossSpeciesEntry (4 organisms with full parameter sets)",
            "SpectralThresholdModel (theta_critical, theta_optimal, kill_condition)",
            "EmpiricalParameters (aggregated certified values)",
        ],
        simulation_ready=True,
        missing_experimental_data=[
            "Direct spectral gap measurement from planarian tissue connectivity",
            "In-vivo energy-per-routing-event measurement",
            "Innexin single-channel conductance in planarian cells",
        ],
        depends_on=["mosaic", "state_space"],
    ),
    ModelingLayer(
        layer_id="sensitivity",
        name="Parameter Sensitivity Analysis",
        section="Section 11",
        module="acheron.reasoning.sensitivity",
        description="Meta-layer that wraps existing computation functions with perturbed "
                    "parameters to quantify modeling assumption sensitivity. Cross-model "
                    "comparison (Watts-Strogatz vs Barabasi-Albert vs Erdos-Renyi). "
                    "Identifies robust conclusions and assumptions at risk.",
        variables_introduced=[
            "SensitivityResult.sensitivity_index",
            "SensitivityResult.robust (bool)",
            "ModelComparisonResult.agreement (0-1)",
            "ModelComparisonResult.dominant_model",
            "SensitivityReport.high_sensitivity_params",
            "SensitivityReport.assumptions_at_risk",
        ],
        simulation_ready=True,
        missing_experimental_data=[
            "Measured parameter ranges for gap junction conductance in vivo",
            "Actual topology type of planarian tissue (WS vs BA vs ER)",
        ],
        depends_on=["mosaic", "state_space", "empirical"],
    ),
    ModelingLayer(
        layer_id="falsification",
        name="Falsification Prediction Registry",
        section="Section 12",
        module="acheron.reasoning.falsification",
        description="Popper-style prediction registry with testable quantitative bounds. "
                    "7 falsifiable predictions with kill conditions, experimental tests, "
                    "and automated margin-of-safety evaluation. Reports overall model "
                    "health (HEALTHY/CAUTION/CRITICAL).",
        variables_introduced=[
            "FalsifiablePrediction.computed_value",
            "FalsifiablePrediction.margin",
            "FalsifiablePrediction.status (alive/at_risk/falsified/untested)",
            "FalsificationReport.overall_health",
            "FalsificationReport.alive_count",
            "FalsificationReport.falsified_count",
        ],
        simulation_ready=True,
        missing_experimental_data=[
            "Experimental validation of any prediction in the registry",
            "In-vivo spectral gap measurement",
            "Measured energy per routing event",
        ],
        depends_on=["mosaic", "state_space", "empirical", "sensitivity"],
    ),
]


# ---------------------------------------------------------------------------
# Cross-layer interaction map
# ---------------------------------------------------------------------------

@dataclass
class CrossLayerInteraction:
    """A directed interaction between two modeling layers."""

    source_layer: str
    target_layer: str
    interaction_type: str               # "data_flow", "constraint", "parameter_sharing"
    variables_shared: list[str] = field(default_factory=list)
    description: str = ""


def build_interaction_map() -> list[CrossLayerInteraction]:
    """Build cross-layer interaction map from layer dependency declarations."""
    interactions: list[CrossLayerInteraction] = []

    interaction_defs = [
        ("substrate", "mosaic", "data_flow",
         ["SubstrateGraph", "SubstrateNode", "SubstrateEdge"],
         "Substrate graph provides node/edge structure for small-world topology"),
        ("substrate", "denram", "data_flow",
         ["SubstrateEdge.diffusion_delay_ms", "SubstrateNode.vm_mV"],
         "Substrate delays and voltages feed delay encoding parameters"),
        ("substrate", "state_space", "data_flow",
         ["SubstrateGraph -> SystemMatrices.A"],
         "Substrate connectivity builds system dynamics matrix A"),
        ("mosaic", "heterogeneity", "parameter_sharing",
         ["TopologyMetrics", "SubstrateGraph"],
         "Mosaic topology provides graph structure for heterogeneity testing"),
        ("mosaic", "state_space", "data_flow",
         ["clustering_coefficient", "average_path_length"],
         "Topology metrics inform dynamical system structure"),
        ("denram", "state_space", "constraint",
         ["EncodingMode", "ChannelKinetics.tau_*"],
         "Delay encoding constrains temporal dynamics in state-space model"),
        ("heterogeneity", "homeostatic", "data_flow",
         ["PerturbationResult", "SubstrateGraph (with variance)"],
         "Heterogeneous substrate provides perturbed state for homeostatic recovery"),
        ("heterogeneity", "state_space", "parameter_sharing",
         ["vm_rest_variance_mV", "gap_junction_variance_pct"],
         "Heterogeneity parameters modulate system matrix A entries"),
        ("homeostatic", "state_space", "constraint",
         ["HomeostaticParams.scaling_factor", "activity_thresholds"],
         "Homeostatic bounds constrain valid operating region in state space"),
        ("ribozyme", "freeze_thaw", "data_flow",
         ["QT45Parameters (length, fidelity, gc_fraction)"],
         "QT45 physical parameters feed freeze-thaw kinetic model"),
    ]

    for src, tgt, itype, vars_shared, desc in interaction_defs:
        interactions.append(CrossLayerInteraction(
            source_layer=src,
            target_layer=tgt,
            interaction_type=itype,
            variables_shared=vars_shared,
            description=desc,
        ))

    return interactions


# ---------------------------------------------------------------------------
# Validation report
# ---------------------------------------------------------------------------

@dataclass
class SimulationReadiness:
    """Simulation readiness status per layer."""

    layer_id: str
    ready: bool = False
    blocker: str = ""                   # what prevents simulation
    notes: str = ""


@dataclass
class ValidationReport:
    """Internal validation report after neuromorphic reasoning engine integration."""

    timestamp: str = ""
    engine_version: str = "1.0.0"
    # Layer inventory
    modeling_layers: list[ModelingLayer] = field(default_factory=list)
    total_layers: int = 0
    # Variables
    total_new_variables: int = 0
    variables_by_layer: dict[str, list[str]] = field(default_factory=dict)
    # Cross-layer interactions
    interactions: list[CrossLayerInteraction] = field(default_factory=list)
    # Simulation readiness
    readiness: list[SimulationReadiness] = field(default_factory=list)
    all_simulations_ready: bool = False
    # Missing data
    all_missing_data: list[dict[str, Any]] = field(default_factory=list)
    total_missing_data_items: int = 0
    # Simulation results (populated when run_simulations=True)
    topology_comparison: Optional[TopologyComparison] = None
    spectral_analysis: Optional[SpectralAnalysis] = None
    rewire_analysis: Optional[RewireAnalysis] = None
    delay_encoding_phase: Optional[DelayEncodingMetrics] = None
    delay_encoding_static: Optional[DelayEncodingMetrics] = None
    heterogeneity_comparison: Optional[HeterogeneityComparison] = None
    homeostatic_adversarial: Optional[HomeostaticRecoveryResult] = None
    homeostatic_repeated: Optional[HomeostaticRecoveryResult] = None
    state_space: Optional[StateSpaceAnalysis] = None
    qt45: Optional[QT45Analysis] = None
    freeze_thaw: Optional[FreezeThawAnalysis] = None
    sensitivity: Optional[SensitivityReport] = None
    falsification: Optional[FalsificationReport] = None


def generate_validation_report(run_simulations: bool = True, seed: int = 42) -> ValidationReport:
    """Generate the full internal validation report.

    If run_simulations=True, executes all simulation layers and includes results.
    """
    report = ValidationReport(
        timestamp=datetime.now(timezone.utc).isoformat(),
        modeling_layers=MODELING_LAYERS,
        total_layers=len(MODELING_LAYERS),
    )

    # Variables inventory
    all_vars: list[str] = []
    for layer in MODELING_LAYERS:
        report.variables_by_layer[layer.layer_id] = layer.variables_introduced
        all_vars.extend(layer.variables_introduced)
    report.total_new_variables = len(all_vars)

    # Cross-layer interactions
    report.interactions = build_interaction_map()

    # Simulation readiness
    for layer in MODELING_LAYERS:
        report.readiness.append(SimulationReadiness(
            layer_id=layer.layer_id,
            ready=layer.simulation_ready,
            blocker="" if layer.simulation_ready else "Missing dependencies",
            notes=f"{len(layer.missing_experimental_data)} experimental data gaps",
        ))
    report.all_simulations_ready = all(r.ready for r in report.readiness)

    # Missing experimental data
    for layer in MODELING_LAYERS:
        for item in layer.missing_experimental_data:
            report.all_missing_data.append({
                "layer": layer.layer_id,
                "data_required": item,
            })
    report.total_missing_data_items = len(report.all_missing_data)

    # Run simulations if requested
    if run_simulations:
        _run_all_simulations(report, seed)

    return report


def _run_all_simulations(report: ValidationReport, seed: int = 42) -> None:
    """Execute all simulation layers and attach results to report."""
    # Section 2: Topology comparison + Spectral analysis
    report.topology_comparison = compare_topologies(n_nodes=50, k_neighbors=6, seed=seed)

    # Section 2b: Spectral analysis + edge rewiring optimization
    graph_for_spectral = build_small_world(50, 6, seed=seed)
    report.spectral_analysis = analyze_spectral_properties(graph_for_spectral)
    report.rewire_analysis = find_optimal_rewires(graph_for_spectral, rewire_budget=5, seed=seed)

    # Section 3: Delay encoding (both modes)
    report.delay_encoding_phase = analyze_delay_encoding(
        encoding_mode=EncodingMode.PHASE_DELAY, seed=seed,
    )
    report.delay_encoding_static = analyze_delay_encoding(
        encoding_mode=EncodingMode.STATIC_ATTRACTOR, seed=seed,
    )

    # Section 4: Heterogeneity comparison
    report.heterogeneity_comparison = compare_substrates(n_nodes=50, seed=seed)

    # Section 5: Homeostatic recovery
    graph_for_homeo = build_small_world(50, 6, seed=seed)
    report.homeostatic_adversarial = simulate_adversarial_recovery(
        graph_for_homeo, bias_mV=30.0, seed=seed,
    )
    graph_for_repeated = build_small_world(50, 6, seed=seed)
    report.homeostatic_repeated = simulate_repeated_perturbation(
        graph_for_repeated, n_perturbations=5, seed=seed,
    )

    # Section 6: State-space analysis
    graph_for_ss = build_small_world(30, 4, seed=seed)
    report.state_space = analyze_state_space(graph_for_ss)

    # Section 7: QT45 ribozyme
    report.qt45 = analyze_qt45()

    # Section 8: Freeze-thaw
    report.freeze_thaw = analyze_freeze_thaw()

    # Section 11: Sensitivity analysis
    report.sensitivity = run_sensitivity_analysis(n_nodes=30, k_neighbors=4, seed=seed)

    # Section 12: Falsification registry
    report.falsification = run_falsification_analysis(n_nodes=30, k_neighbors=4, seed=seed)


def format_report(report: ValidationReport) -> str:
    """Format validation report as structured text output."""
    lines: list[str] = []
    sep = "=" * 72

    lines.append(sep)
    lines.append("NEUROMORPHIC REASONING ENGINE — INTERNAL VALIDATION REPORT")
    lines.append(sep)
    lines.append(f"Timestamp: {report.timestamp}")
    lines.append(f"Engine Version: {report.engine_version}")
    lines.append("")

    # Section A: Modeling Layers
    lines.append(f"A. MODELING LAYERS ({report.total_layers} layers)")
    lines.append("-" * 72)
    for layer in report.modeling_layers:
        lines.append(f"  [{layer.layer_id}] {layer.name} ({layer.section})")
        lines.append(f"    Module: {layer.module}")
        lines.append(f"    Description: {layer.description}")
        lines.append(f"    Depends on: {layer.depends_on or 'none'}")
        lines.append(f"    Feeds into: {layer.feeds_into or 'none'}")
        lines.append("")

    # Section B: New Variables
    lines.append(f"B. NEW VARIABLES INTRODUCED ({report.total_new_variables} total)")
    lines.append("-" * 72)
    for layer_id, vars_list in report.variables_by_layer.items():
        lines.append(f"  [{layer_id}]")
        for v in vars_list:
            lines.append(f"    - {v}")
    lines.append("")

    # Section C: Cross-Layer Interactions
    lines.append(f"C. CROSS-LAYER INTERACTION MAP ({len(report.interactions)} interactions)")
    lines.append("-" * 72)
    for ix in report.interactions:
        lines.append(f"  {ix.source_layer} --[{ix.interaction_type}]--> {ix.target_layer}")
        lines.append(f"    Variables: {ix.variables_shared}")
        lines.append(f"    {ix.description}")
    lines.append("")

    # Section D: Simulation Readiness
    lines.append("D. SIMULATION READINESS STATUS")
    lines.append("-" * 72)
    for r in report.readiness:
        status = "READY" if r.ready else f"BLOCKED: {r.blocker}"
        lines.append(f"  [{r.layer_id}] {status} | {r.notes}")
    lines.append(f"  ALL READY: {report.all_simulations_ready}")
    lines.append("")

    # Section E: Missing Experimental Data
    lines.append(f"E. MISSING EXPERIMENTAL DATA ({report.total_missing_data_items} items)")
    lines.append("-" * 72)
    for item in report.all_missing_data:
        lines.append(f"  [{item['layer']}] {item['data_required']}")
    lines.append("")

    # Section F: Simulation Results
    if report.topology_comparison:
        lines.append("F. SIMULATION RESULTS")
        lines.append("-" * 72)
        lines.append("")
        lines.append("  F.1 Topology Comparison (Section 2)")
        for label, metrics in [
            ("Ring Lattice", report.topology_comparison.ring_lattice),
            ("Small World", report.topology_comparison.small_world),
            ("Uniform Random", report.topology_comparison.uniform),
        ]:
            lines.append(f"    {label}:")
            lines.append(f"      Nodes: {metrics.node_count}, Edges: {metrics.edge_count}")
            lines.append(f"      Clustering Coefficient: {metrics.clustering_coefficient}")
            lines.append(f"      Avg Path Length: {metrics.average_path_length}")
            lines.append(f"      Signal Latency: {metrics.signal_propagation_latency_ms} ms")
            lines.append(f"      Energy/Route: {metrics.energy_per_routing_event_J:.2e} J")
            lines.append(f"      Error Cascade P: {metrics.error_cascade_probability}")
            lines.append(f"      Fault Tolerance: {metrics.fault_tolerance_fraction}")
        lines.append("")

    if report.spectral_analysis:
        lines.append("  F.1b Spectral Analysis (Section 2b)")
        sa = report.spectral_analysis
        lines.append(f"    Spectral Gap (lambda_2): {sa.spectral_gap}")
        lines.append(f"    Laplacian Eigenvalues: {sa.laplacian_eigenvalues[:6]}")
        lines.append(f"    Total Network Energy: {sa.total_network_energy_J:.4e} J")
        lines.append(f"    Energy per Edge: {sa.energy_per_edge_J:.4e} J")
        lines.append(f"    Spectral Efficiency: {sa.spectral_efficiency}")
        stall = SPECTRAL_THRESHOLD.is_stall_risk(sa.spectral_gap)
        optimal = SPECTRAL_THRESHOLD.is_optimal(sa.spectral_gap)
        lines.append(f"    Status: {'OPTIMAL' if optimal else 'STALL RISK' if stall else 'SUB-OPTIMAL'}")
        lines.append("")

    if report.rewire_analysis and report.rewire_analysis.optimal_rewires:
        lines.append("  F.1c Edge Rewiring Optimization (Section 2c)")
        ra = report.rewire_analysis
        lines.append(f"    Baseline Spectral Gap: {ra.baseline_spectral_gap}")
        lines.append(f"    Optimized Spectral Gap: {ra.optimized_spectral_gap}")
        lines.append(f"    Improvement: +{ra.gap_improvement_pct}%")
        lines.append(f"    Energy Reduction: {ra.energy_reduction_pct}%")
        lines.append(f"    Optimal Rewires: {len(ra.optimal_rewires)}")
        for i, rw in enumerate(ra.optimal_rewires[:3], 1):
            lines.append(f"      #{i}: {rw.original_source}→{rw.original_target} → "
                        f"{rw.new_source}→{rw.new_target} (Δλ₂=+{rw.delta_spectral_gap})")
        lines.append("")

    if report.delay_encoding_phase:
        lines.append("  F.2 Delay Encoding — Phase Mode (Section 3)")
        de = report.delay_encoding_phase
        lines.append(f"    Phase Stability Score: {de.phase_stability_score}")
        lines.append(f"    Mean Persistence: {de.mean_persistence_ms} ms")
        lines.append(f"    Noise Tolerance: {de.noise_tolerance_mV} mV")
        lines.append(f"    Metabolic Fluctuation Tol: {de.metabolic_fluctuation_tolerance}")
        lines.append(f"    Cross-Talk P: {de.cross_talk_probability}")
        lines.append(f"    Min Cluster Separation: {de.min_cluster_separation_um} um")
        lines.append("")

    if report.delay_encoding_static:
        lines.append("  F.3 Delay Encoding — Static Attractor Mode (Section 3)")
        de = report.delay_encoding_static
        lines.append(f"    Stability Score: {de.phase_stability_score}")
        lines.append(f"    Mean Persistence: {de.mean_persistence_ms} ms")
        lines.append(f"    Cross-Talk P: {de.cross_talk_probability}")
        lines.append("")

    if report.heterogeneity_comparison:
        lines.append("  F.4 Heterogeneity Comparison (Section 4)")
        hc = report.heterogeneity_comparison
        lines.append(f"    Uniform — Recovery Steps: {hc.uniform_result.recovery_steps}, "
                     f"Complete: {hc.uniform_result.recovery_complete}, "
                     f"Residual: {hc.uniform_result.residual_error_mV} mV")
        lines.append(f"    Heterogeneous — Recovery Steps: {hc.heterogeneous_result.recovery_steps}, "
                     f"Complete: {hc.heterogeneous_result.recovery_complete}, "
                     f"Residual: {hc.heterogeneous_result.residual_error_mV} mV")
        lines.append(f"    Basin Depth (Uniform): {hc.uniform_basin_depth_mV} mV")
        lines.append(f"    Basin Depth (Heterogeneous): {hc.heterogeneous_basin_depth_mV} mV")
        lines.append(f"    Conclusion: {hc.conclusion}")
        lines.append("")

    if report.homeostatic_adversarial:
        lines.append("  F.5 Homeostatic Recovery — Adversarial (Section 5)")
        hr = report.homeostatic_adversarial
        lines.append(f"    Recovery Steps: {hr.recovery_steps}")
        lines.append(f"    Recovery Complete: {hr.recovery_complete}")
        lines.append(f"    Final Activity: {hr.final_activity}")
        lines.append(f"    Total Energy: {hr.total_energy_J:.2e} J")
        lines.append(f"    Scaling Events: {hr.total_scaling_events}")
        lines.append("")

    if report.homeostatic_repeated:
        lines.append("  F.6 Homeostatic Recovery — Repeated Perturbation (Section 5)")
        hr = report.homeostatic_repeated
        lines.append(f"    Perturbations Applied: {hr.perturbation_count}")
        lines.append(f"    Stability Maintained: {hr.stability_maintained}")
        lines.append(f"    Activity Variance: {hr.activity_variance}")
        lines.append(f"    Total Energy: {hr.total_energy_J:.2e} J")
        lines.append("")

    if report.state_space:
        lines.append("  F.7 State-Space Analysis (Section 6)")
        ss = report.state_space
        lines.append(f"    System Dimension: {ss.system_dimension}")
        lines.append(f"    Stable: {ss.stability.is_stable}")
        lines.append(f"    Dominant Eigenvalue: {ss.stability.dominant_eigenvalue}")
        lines.append(f"    Eigenvalues (top 5): {ss.stability.eigenvalues}")
        lines.append(f"    Spectral Gap: {ss.stability.spectral_gap}")
        lines.append(f"    Convergence Rate: {ss.stability.convergence_rate}")
        if ss.attractors:
            lines.append(f"    Attractors Found: {len(ss.attractors)}")
            lines.append(f"    Attractor Convergence Steps: {ss.attractors[0].convergence_steps}")
        if ss.recovery:
            lines.append(f"    Perturbation Recovery Steps: {ss.recovery.recovery_steps}")
            lines.append(f"    Recovery Converged: {ss.recovery.converged}")
        lines.append("")

    if report.qt45:
        lines.append("  F.8 QT45 Ribozyme Analysis (Section 7)")
        qt = report.qt45
        lines.append(f"    Per-Copy Fidelity: {qt.parameters.per_copy_fidelity:.6f}")
        lines.append(f"    Per-Copy Error Rate: {qt.parameters.per_copy_error_rate:.6f}")
        lines.append(f"    Copies/Year: {qt.parameters.copies_per_year:.2f}")
        lines.append(f"    Redundancy for 1yr @ 99%: {qt.redundancy.copies_needed} copies")
        lines.append(f"    Survival Probability: {qt.redundancy.survival_probability_actual}")
        lines.append(f"    ECC Overhead: {qt.error_correction.overhead_fraction} "
                     f"({qt.error_correction.parity_nucleotides_needed} parity nt)")
        lines.append(f"    Corruption after 1yr (single): {qt.corruption_1yr.p_corruption_single_copy}")
        lines.append(f"    Corruption after 1yr (redundant): {qt.corruption_1yr.p_corruption_with_redundancy}")
        lines.append(f"    Cross-Chiral Key Space: {qt.access_model.key_space_size}")
        lines.append(f"    Enzymatic Resistance: {qt.enzymatic_resistance.overall_resistance:.4f}")
        lines.append("")

    if report.freeze_thaw:
        lines.append("  F.9 Freeze-Thaw Kinetics (Section 8)")
        ft = report.freeze_thaw
        lines.append(f"    Stall Probability: {ft.stall_probability}")
        lines.append(f"    Strand Separation @ Freeze: {ft.strand_separation_at_freeze}")
        lines.append(f"    Strand Separation @ Thaw: {ft.strand_separation_at_thaw}")
        lines.append(f"    Concentration Factor: {ft.concentration_factor}")
        lines.append(f"    Optimal Freeze: {ft.optimal_interval.optimal_freeze_hours} h")
        lines.append(f"    Optimal Thaw: {ft.optimal_interval.optimal_thaw_hours} h")
        lines.append(f"    Optimal Cycle: {ft.optimal_interval.optimal_cycle_hours} h")
        lines.append(f"    Replication Efficiency: {ft.optimal_interval.replication_efficiency}")
        lines.append(f"    Triplet Trapping P: {ft.triplet_trapping.trapping_probability:.6f}")
        lines.append(f"    Stalled Triplets: {ft.triplet_trapping.stalled_triplets}/{ft.triplet_trapping.n_triplets}")
        lines.append("")

    if report.sensitivity:
        lines.append("  F.10 Parameter Sensitivity Analysis (Section 11)")
        for s in report.sensitivity.parameter_sensitivities:
            status = "ROBUST" if s.robust else "HIGH SENSITIVITY"
            lines.append(f"    {s.parameter_name} → {s.output_name}: "
                        f"S={s.sensitivity_index} [{status}]")
        if report.sensitivity.model_comparisons:
            lines.append("    Cross-Model Comparison:")
            for mc in report.sensitivity.model_comparisons:
                lines.append(f"      {mc.metric_name}: WS={mc.watts_strogatz} "
                            f"BA={mc.barabasi_albert} ER={mc.erdos_renyi} "
                            f"(agreement={mc.agreement:.0%})")
        lines.append("")

    if report.falsification:
        lines.append("  F.11 Falsification Registry (Section 12)")
        lines.append(f"    Overall Health: {report.falsification.overall_health}")
        lines.append(f"    Alive: {report.falsification.alive_count} | "
                    f"At Risk: {report.falsification.at_risk_count} | "
                    f"Falsified: {report.falsification.falsified_count}")
        for pred in report.falsification.predictions:
            icon = {
                PredictionStatus.ALIVE: "+",
                PredictionStatus.AT_RISK: "~",
                PredictionStatus.FALSIFIED: "X",
                PredictionStatus.UNTESTED: "?",
            }[pred.status]
            lines.append(f"    [{icon}] {pred.prediction_id}: "
                        f"val={pred.computed_value:.4f} margin={pred.margin:.4f}")
        lines.append("")

    lines.append(sep)
    lines.append("END OF VALIDATION REPORT")
    lines.append(sep)

    return "\n".join(lines)


def report_to_dict(report: ValidationReport) -> dict[str, Any]:
    """Convert report to JSON-serializable dict."""
    d: dict[str, Any] = {
        "timestamp": report.timestamp,
        "engine_version": report.engine_version,
        "total_layers": report.total_layers,
        "total_new_variables": report.total_new_variables,
        "all_simulations_ready": report.all_simulations_ready,
        "total_missing_data_items": report.total_missing_data_items,
        "layers": [],
        "interactions": [],
        "readiness": [],
        "missing_data": report.all_missing_data,
    }
    for layer in report.modeling_layers:
        d["layers"].append({
            "layer_id": layer.layer_id,
            "name": layer.name,
            "section": layer.section,
            "module": layer.module,
            "variables_count": len(layer.variables_introduced),
            "simulation_ready": layer.simulation_ready,
            "missing_data_count": len(layer.missing_experimental_data),
            "depends_on": layer.depends_on,
            "feeds_into": layer.feeds_into,
        })
    for ix in report.interactions:
        d["interactions"].append({
            "source": ix.source_layer,
            "target": ix.target_layer,
            "type": ix.interaction_type,
            "variables": ix.variables_shared,
        })
    for r in report.readiness:
        d["readiness"].append({
            "layer_id": r.layer_id,
            "ready": r.ready,
            "notes": r.notes,
        })
    return d
