"""QT45 Ribozyme Integration Layer (Section 7).

Molecular storage module parameters:
    - Length: 45 nt
    - Fidelity per nucleotide: 0.941
    - Copy cycle: 72 days
    - Yield: 0.2%

Nexus calculations:
    - Redundancy factor for 1-year data stability
    - Error correction overhead required
    - Probability of corruption after N cycles

Cross-chiral security model:
    - Mirror-RNA storage requires specific triplet set
      (13 defined triplets + 1 hexamer)
    - Incomplete substrate -> replication failure

Simulation:
    - Molecular key-based access model
    - Resistance to standard enzymatic degradation
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# QT45 Physical Parameters
# ---------------------------------------------------------------------------

QT45_LENGTH_NT = 45
QT45_FIDELITY_PER_NT = 0.941
QT45_COPY_CYCLE_DAYS = 72
QT45_YIELD_FRACTION = 0.002        # 0.2%

# Cross-chiral triplet set
CROSS_CHIRAL_TRIPLETS = [
    "GCG", "CGC", "GAC", "GUC", "CGA", "CGU",
    "ACG", "UGC", "GCA", "UCG", "CAG", "CUG", "AGC",
]
CROSS_CHIRAL_HEXAMER = "GCGCGC"
REQUIRED_SUBSTRATE_COUNT = len(CROSS_CHIRAL_TRIPLETS) + 1  # 13 triplets + 1 hexamer = 14


@dataclass
class QT45Parameters:
    """Physical parameters for the QT45 ribozyme system."""

    length_nt: int = QT45_LENGTH_NT
    fidelity_per_nt: float = QT45_FIDELITY_PER_NT
    copy_cycle_days: float = QT45_COPY_CYCLE_DAYS
    yield_fraction: float = QT45_YIELD_FRACTION
    evidence_status: str = "[SIMULATION-DERIVED]"

    @property
    def per_copy_fidelity(self) -> float:
        """Probability of perfect copy for entire strand."""
        return self.fidelity_per_nt ** self.length_nt

    @property
    def per_copy_error_rate(self) -> float:
        """Probability of at least one error per copy."""
        return 1.0 - self.per_copy_fidelity

    @property
    def copies_per_year(self) -> float:
        return 365.25 / self.copy_cycle_days


# ---------------------------------------------------------------------------
# Redundancy and error correction calculations
# ---------------------------------------------------------------------------

@dataclass
class RedundancyAnalysis:
    """Redundancy factor analysis for data stability."""

    target_duration_days: float = 365.25        # 1 year
    target_survival_probability: float = 0.99   # P(at least one intact copy)
    copies_needed: int = 0
    redundancy_factor: float = 0.0              # copies_needed / 1
    survival_probability_actual: float = 0.0
    n_copy_cycles: int = 0


def compute_redundancy_for_stability(
    params: QT45Parameters = QT45Parameters(),
    target_duration_days: float = 365.25,
    target_survival_prob: float = 0.99,
) -> RedundancyAnalysis:
    """Calculate redundancy factor for target duration data stability.

    Model: each copy independently has P(corrupt after N cycles).
    Need enough copies that P(all corrupt) < 1 - target_survival_prob.
    """
    n_cycles = math.ceil(target_duration_days / params.copy_cycle_days)
    # P(single copy survives N cycles) = fidelity^(length * N_cycles)
    # But yield limits effective copies per cycle
    p_survive_one_cycle = params.per_copy_fidelity * params.yield_fraction
    # Over N cycles, survival of lineage = product
    # Each cycle: copy with fidelity, with yield
    # P(lineage survives) = (per_copy_fidelity * yield)^n_cycles
    # This is extremely small, so we need massive redundancy
    # Better model: each cycle produces yield * current_copies new copies
    # P(at least one perfect copy after N cycles from M starting copies)

    # Per-copy survival through all cycles
    p_single_lineage = params.per_copy_fidelity ** n_cycles

    if p_single_lineage <= 0:
        # Fidelity too low for any copies to survive
        return RedundancyAnalysis(
            target_duration_days=target_duration_days,
            target_survival_probability=target_survival_prob,
            copies_needed=-1,  # infeasible
            n_copy_cycles=n_cycles,
        )

    # P(all M copies corrupt) = (1 - p_single_lineage)^M < (1 - target)
    # M > log(1 - target) / log(1 - p_single_lineage)
    p_fail_target = 1.0 - target_survival_prob
    if p_single_lineage >= 1.0:
        copies_needed = 1
    else:
        log_denominator = math.log(1.0 - p_single_lineage)
        if log_denominator >= 0:
            copies_needed = -1  # infeasible
        else:
            copies_needed = math.ceil(math.log(p_fail_target) / log_denominator)

    p_actual = 1.0 - (1.0 - p_single_lineage) ** copies_needed if copies_needed > 0 else 0.0

    return RedundancyAnalysis(
        target_duration_days=target_duration_days,
        target_survival_probability=target_survival_prob,
        copies_needed=copies_needed,
        redundancy_factor=float(copies_needed),
        survival_probability_actual=round(p_actual, 8),
        n_copy_cycles=n_cycles,
    )


@dataclass
class ErrorCorrectionOverhead:
    """Error correction requirements for QT45 storage."""

    raw_error_rate_per_nt: float = 0.0
    hamming_distance_required: int = 0
    parity_nucleotides_needed: int = 0
    overhead_fraction: float = 0.0              # parity_nt / total_nt
    effective_payload_nt: int = 0
    correctable_errors_per_copy: int = 0


def compute_error_correction_overhead(
    params: QT45Parameters = QT45Parameters(),
    target_correctable_errors: int = 3,
) -> ErrorCorrectionOverhead:
    """Calculate error correction overhead for QT45 ribozyme storage.

    Uses BCH/Hamming-style analysis adapted for quaternary alphabet (A,C,G,U).
    """
    raw_error_per_nt = 1.0 - params.fidelity_per_nt

    # For t-error-correcting code over GF(4):
    # Need 2*t check symbols minimum
    parity_nt = 2 * target_correctable_errors
    # Hamming distance = 2*t + 1
    hamming_d = 2 * target_correctable_errors + 1

    effective_payload = params.length_nt - parity_nt
    overhead = parity_nt / params.length_nt if params.length_nt > 0 else 0.0

    return ErrorCorrectionOverhead(
        raw_error_rate_per_nt=round(raw_error_per_nt, 6),
        hamming_distance_required=hamming_d,
        parity_nucleotides_needed=parity_nt,
        overhead_fraction=round(overhead, 4),
        effective_payload_nt=max(0, effective_payload),
        correctable_errors_per_copy=target_correctable_errors,
    )


@dataclass
class CorruptionProbability:
    """Probability of data corruption after N copy cycles."""

    n_cycles: int = 0
    p_corruption_single_copy: float = 0.0
    p_corruption_with_redundancy: float = 0.0
    redundancy: int = 1
    expected_intact_copies: float = 0.0


def compute_corruption_after_n_cycles(
    n_cycles: int,
    params: QT45Parameters = QT45Parameters(),
    redundancy: int = 10,
) -> CorruptionProbability:
    """Probability of corruption after N copy cycles."""
    # P(single copy survives N cycles with no error)
    p_survive = params.per_copy_fidelity ** n_cycles
    p_corrupt_single = 1.0 - p_survive

    # P(all copies corrupt) = (1 - p_survive)^redundancy
    p_all_corrupt = p_corrupt_single ** redundancy if redundancy > 0 else 1.0
    expected_intact = redundancy * p_survive

    return CorruptionProbability(
        n_cycles=n_cycles,
        p_corruption_single_copy=round(p_corrupt_single, 10),
        p_corruption_with_redundancy=round(p_all_corrupt, 10),
        redundancy=redundancy,
        expected_intact_copies=round(expected_intact, 4),
    )


# ---------------------------------------------------------------------------
# Cross-chiral security model
# ---------------------------------------------------------------------------

@dataclass
class CrossChiralSubstrate:
    """Cross-chiral mirror-RNA substrate definition."""

    triplets: list[str] = field(default_factory=lambda: list(CROSS_CHIRAL_TRIPLETS))
    hexamer: str = CROSS_CHIRAL_HEXAMER
    is_complete: bool = True

    def validate(self) -> tuple[bool, list[str]]:
        """Validate substrate completeness. Incomplete = replication failure."""
        missing: list[str] = []
        for required in CROSS_CHIRAL_TRIPLETS:
            if required not in self.triplets:
                missing.append(required)
        if self.hexamer != CROSS_CHIRAL_HEXAMER:
            missing.append(f"hexamer:{CROSS_CHIRAL_HEXAMER}")
        self.is_complete = len(missing) == 0
        return self.is_complete, missing


@dataclass
class MolecularAccessModel:
    """Molecular key-based access model for cross-chiral storage."""

    substrate: CrossChiralSubstrate = field(default_factory=CrossChiralSubstrate)
    # Security properties
    replication_possible: bool = False
    degradation_resistance: float = 0.0         # 0-1 scale
    key_space_size: int = 0                     # combinatorial key space

    def evaluate(self) -> None:
        """Evaluate access model security properties."""
        is_valid, missing = self.substrate.validate()
        self.replication_possible = is_valid

        # Degradation resistance: mirror-RNA resists standard RNases
        # Full resistance if substrate is pure L-RNA (mirror)
        self.degradation_resistance = 0.99 if is_valid else 0.0

        # Key space: number of possible triplet subsets from 14 elements
        # Any subset missing even 1 element = replication failure
        # Attacker must have all 14 substrates
        self.key_space_size = 2 ** REQUIRED_SUBSTRATE_COUNT  # 2^14 = 16384 combinations


@dataclass
class EnzymaticResistance:
    """Resistance profile against standard enzymatic degradation."""

    rnase_a_resistance: float = 0.99            # mirror-RNA invisible to RNase A
    rnase_h_resistance: float = 0.99            # no DNA:L-RNA hybrid recognition
    exonuclease_resistance: float = 0.95        # reduced but not zero
    protease_resistance: float = 1.0            # not a protein
    overall_resistance: float = 0.0

    def __post_init__(self):
        self.overall_resistance = (
            self.rnase_a_resistance *
            self.rnase_h_resistance *
            self.exonuclease_resistance *
            self.protease_resistance
        )


# ---------------------------------------------------------------------------
# Full QT45 analysis
# ---------------------------------------------------------------------------

@dataclass
class QT45Analysis:
    """Complete QT45 ribozyme integration analysis for Nexus."""

    parameters: QT45Parameters = field(default_factory=QT45Parameters)
    redundancy: RedundancyAnalysis = field(default_factory=RedundancyAnalysis)
    error_correction: ErrorCorrectionOverhead = field(default_factory=ErrorCorrectionOverhead)
    corruption_1yr: CorruptionProbability = field(default_factory=CorruptionProbability)
    corruption_5yr: CorruptionProbability = field(default_factory=CorruptionProbability)
    access_model: MolecularAccessModel = field(default_factory=MolecularAccessModel)
    enzymatic_resistance: EnzymaticResistance = field(default_factory=EnzymaticResistance)


def analyze_qt45(
    params: Optional[QT45Parameters] = None,
    target_survival_prob: float = 0.99,
    target_correctable_errors: int = 3,
) -> QT45Analysis:
    """Run complete QT45 ribozyme analysis."""
    if params is None:
        params = QT45Parameters()

    redundancy = compute_redundancy_for_stability(params, target_survival_prob=target_survival_prob)
    ecc = compute_error_correction_overhead(params, target_correctable_errors)

    n_cycles_1yr = math.ceil(365.25 / params.copy_cycle_days)
    n_cycles_5yr = math.ceil(5 * 365.25 / params.copy_cycle_days)

    corruption_1yr = compute_corruption_after_n_cycles(
        n_cycles_1yr, params, redundancy=max(1, redundancy.copies_needed),
    )
    corruption_5yr = compute_corruption_after_n_cycles(
        n_cycles_5yr, params, redundancy=max(1, redundancy.copies_needed),
    )

    access = MolecularAccessModel()
    access.evaluate()

    resistance = EnzymaticResistance()

    return QT45Analysis(
        parameters=params,
        redundancy=redundancy,
        error_correction=ecc,
        corruption_1yr=corruption_1yr,
        corruption_5yr=corruption_5yr,
        access_model=access,
        enzymatic_resistance=resistance,
    )
