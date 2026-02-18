"""Neuromorphic Reasoning Engine for Acheron.

Integrates substrate-driven computation principles derived from:
    - Mosaic small-world architectures (Payvand)
    - Graph Laplacian spectral analysis & energy-optimized rewiring
    - DenRAM delay encoding
    - State-space dynamical models
    - Homeostatic scaling (BESS)
    - Heterogeneity-aware encoding
    - QT45 ribozyme dynamics
    - Empirical grounding (Levin bioelectric + cross-species)

Modules:
    substrate       Section 1: Substrate=Algorithm core abstraction
    mosaic          Section 2: Mosaic small-world + spectral analysis + rewiring
    denram          Section 3: DenRAM bioelectric delay encoding
    heterogeneity   Section 4: Heterogeneity-aware computation
    homeostatic     Section 5: Homeostatic recovery model (BESS upgrade)
    state_space     Section 6: State-space dynamical framework
    ribozyme        Section 7: QT45 ribozyme integration layer
    freeze_thaw     Section 8: Freeze-thaw kinetic model
    validation      Section 9: Validation report generator
    empirical       Section 10: Empirical grounding layer (Payvand/Levin/cross-species)
    sensitivity     Section 11: Parameter sensitivity analysis meta-layer
    falsification   Section 12: Falsification prediction registry
"""

from .substrate import (
    ComputationalPrimitive,
    SubstrateEdge,
    SubstrateGraph,
    SubstrateNode,
    TopologyEvent,
    TopologyEventType,
)
from .mosaic import (
    RewireAnalysis,
    RewireCandidate,
    SpectralAnalysis,
    TopologyComparison,
    TopologyMetrics,
    analyze_spectral_properties,
    analyze_topology,
    build_ring_lattice,
    build_scale_free,
    build_small_world,
    build_uniform_random,
    compare_topologies,
    compute_graph_laplacian,
    compute_spectral_gap,
    compute_total_network_energy,
    find_optimal_rewires,
    rewire_sweep,
)
from .empirical import (
    CROSS_SPECIES_TABLE,
    EMPIRICAL_PARAMS,
    LEVIN_SPEC,
    MOSAIC_SPEC,
    SPECTRAL_THRESHOLD,
    BioelectricMemorySpec,
    CrossSpeciesEntry,
    EmpiricalParameters,
    MosaicArchitectureSpec,
    SpectralThresholdModel,
    format_cross_species_table,
    get_cross_species_table,
    get_empirical_context,
)
from .denram import (
    AttractorState,
    ChannelKinetics,
    DelayEncodingMetrics,
    EncodingMode,
    GapJunctionDelay,
    PhaseEncoder,
    StaticAttractorModel,
    analyze_delay_encoding,
)
from .heterogeneity import (
    HeterogeneityComparison,
    HeterogeneityParams,
    PerturbationResult,
    compare_substrates,
)
from .homeostatic import (
    HomeostaticParams,
    HomeostaticRecoveryResult,
    HomeostaticState,
    simulate_adversarial_recovery,
    simulate_repeated_perturbation,
)
from .state_space import (
    Attractor,
    RecoveryTrajectory,
    StabilityAnalysis,
    StateSpaceAnalysis,
    StateVector,
    SystemMatrices,
    analyze_state_space,
    analyze_stability,
    build_system_matrices,
)
from .ribozyme import (
    CrossChiralSubstrate,
    ErrorCorrectionOverhead,
    MolecularAccessModel,
    QT45Analysis,
    QT45Parameters,
    RedundancyAnalysis,
    analyze_qt45,
)
from .freeze_thaw import (
    FreezeThawAnalysis,
    FreezeThawParams,
    analyze_freeze_thaw,
)
from .sensitivity import (
    ModelComparisonResult,
    ParameterSweepPoint,
    SensitivityReport,
    SensitivityResult,
    compare_topology_models,
    format_sensitivity_report,
    run_sensitivity_analysis,
)
from .falsification import (
    FalsifiablePrediction,
    FalsificationReport,
    PredictionStatus,
    format_falsification_report,
    run_falsification_analysis,
)
from .validation import (
    ValidationReport,
    format_report,
    generate_validation_report,
    report_to_dict,
)

__all__ = [
    # Substrate (S1)
    "ComputationalPrimitive",
    "SubstrateEdge",
    "SubstrateGraph",
    "SubstrateNode",
    "TopologyEvent",
    "TopologyEventType",
    # Mosaic (S2)
    "RewireAnalysis",
    "RewireCandidate",
    "SpectralAnalysis",
    "TopologyComparison",
    "TopologyMetrics",
    "analyze_spectral_properties",
    "analyze_topology",
    "build_ring_lattice",
    "build_scale_free",
    "build_small_world",
    "build_uniform_random",
    "compare_topologies",
    "compute_graph_laplacian",
    "compute_spectral_gap",
    "compute_total_network_energy",
    "find_optimal_rewires",
    "rewire_sweep",
    # Empirical Grounding (S10)
    "CROSS_SPECIES_TABLE",
    "EMPIRICAL_PARAMS",
    "LEVIN_SPEC",
    "MOSAIC_SPEC",
    "SPECTRAL_THRESHOLD",
    "BioelectricMemorySpec",
    "CrossSpeciesEntry",
    "EmpiricalParameters",
    "MosaicArchitectureSpec",
    "SpectralThresholdModel",
    "format_cross_species_table",
    "get_cross_species_table",
    "get_empirical_context",
    # DenRAM (S3)
    "AttractorState",
    "ChannelKinetics",
    "DelayEncodingMetrics",
    "EncodingMode",
    "GapJunctionDelay",
    "PhaseEncoder",
    "StaticAttractorModel",
    "analyze_delay_encoding",
    # Heterogeneity (S4)
    "HeterogeneityComparison",
    "HeterogeneityParams",
    "PerturbationResult",
    "compare_substrates",
    # Homeostatic (S5)
    "HomeostaticParams",
    "HomeostaticRecoveryResult",
    "HomeostaticState",
    "simulate_adversarial_recovery",
    "simulate_repeated_perturbation",
    # State-space (S6)
    "Attractor",
    "RecoveryTrajectory",
    "StabilityAnalysis",
    "StateSpaceAnalysis",
    "StateVector",
    "SystemMatrices",
    "analyze_state_space",
    "analyze_stability",
    "build_system_matrices",
    # Ribozyme (S7)
    "CrossChiralSubstrate",
    "ErrorCorrectionOverhead",
    "MolecularAccessModel",
    "QT45Analysis",
    "QT45Parameters",
    "RedundancyAnalysis",
    "analyze_qt45",
    # Freeze-thaw (S8)
    "FreezeThawAnalysis",
    "FreezeThawParams",
    "analyze_freeze_thaw",
    # Sensitivity (S11)
    "ModelComparisonResult",
    "ParameterSweepPoint",
    "SensitivityReport",
    "SensitivityResult",
    "compare_topology_models",
    "format_sensitivity_report",
    "run_sensitivity_analysis",
    # Falsification (S12)
    "FalsifiablePrediction",
    "FalsificationReport",
    "PredictionStatus",
    "format_falsification_report",
    "run_falsification_analysis",
    # Validation (S9)
    "ValidationReport",
    "format_report",
    "generate_validation_report",
    "report_to_dict",
]
