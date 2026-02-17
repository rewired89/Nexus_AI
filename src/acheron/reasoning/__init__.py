"""Neuromorphic Reasoning Engine for Acheron.

Integrates substrate-driven computation principles derived from:
    - Mosaic small-world architectures
    - DenRAM delay encoding
    - State-space dynamical models
    - Homeostatic scaling (BESS)
    - Heterogeneity-aware encoding
    - QT45 ribozyme dynamics

Modules:
    substrate       Section 1: Substrate=Algorithm core abstraction
    mosaic          Section 2: Mosaic small-world model adaptation
    denram          Section 3: DenRAM bioelectric delay encoding
    heterogeneity   Section 4: Heterogeneity-aware computation
    homeostatic     Section 5: Homeostatic recovery model (BESS upgrade)
    state_space     Section 6: State-space dynamical framework
    ribozyme        Section 7: QT45 ribozyme integration layer
    freeze_thaw     Section 8: Freeze-thaw kinetic model
    validation      Section 9: Validation report generator
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
    TopologyComparison,
    TopologyMetrics,
    analyze_topology,
    build_ring_lattice,
    build_small_world,
    build_uniform_random,
    compare_topologies,
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
    "TopologyComparison",
    "TopologyMetrics",
    "analyze_topology",
    "build_ring_lattice",
    "build_small_world",
    "build_uniform_random",
    "compare_topologies",
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
    # Validation (S9)
    "ValidationReport",
    "format_report",
    "generate_validation_report",
    "report_to_dict",
]
