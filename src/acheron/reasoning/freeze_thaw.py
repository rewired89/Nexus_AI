"""Freeze-Thaw Kinetic Model (Section 8).

Environmental state parameters:
    - Temperature range: -7C eutectic ice
    - Strand separation probability
    - Triplet kinetic trapping

Nexus must:
    - Model replication stall probability
    - Determine optimal freeze-thaw interval
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------

R_GAS = 8.314              # J/(mol*K), gas constant
K_BOLTZMANN = 1.381e-23    # J/K

# Eutectic ice parameters
EUTECTIC_TEMP_C = -7.0
EUTECTIC_TEMP_K = 273.15 + EUTECTIC_TEMP_C     # 266.15 K


@dataclass
class FreezeThawParams:
    """Parameters for the freeze-thaw kinetic model."""

    # Temperature
    freeze_temp_C: float = EUTECTIC_TEMP_C          # eutectic ice temperature
    thaw_temp_C: float = 20.0                       # thaw / reaction temperature
    # Strand separation
    strand_length_nt: int = 45                      # QT45 ribozyme length
    gc_fraction: float = 0.55                       # GC content fraction
    # Kinetic trapping
    triplet_binding_energy_kJ_per_mol: float = 15.0 # per-triplet binding energy
    # Arrhenius parameters for replication rate
    activation_energy_kJ_per_mol: float = 80.0      # Ea for polymerase activity
    pre_exponential_factor: float = 1e10             # A in k = A * exp(-Ea/RT)
    # Cycle parameters
    freeze_duration_hours: float = 12.0
    thaw_duration_hours: float = 12.0
    evidence_status: str = "[SIMULATION-DERIVED]"


# ---------------------------------------------------------------------------
# Strand separation model
# ---------------------------------------------------------------------------

def strand_separation_probability(
    temp_C: float,
    strand_length_nt: int = 45,
    gc_fraction: float = 0.55,
) -> float:
    """Probability of strand separation at a given temperature.

    Simplified nearest-neighbor thermodynamic model.
    Tm approx = 64.9 + 41 * (gc_fraction - 16.4) / strand_length_nt  (empirical, short RNA)
    P(separation) = sigmoid around Tm.
    """
    # Empirical melting temperature for short RNA
    tm_C = 64.9 + 41.0 * (gc_fraction * strand_length_nt - 16.4) / strand_length_nt

    # Sigmoid: probability of separation increases with temperature
    # Width parameter based on strand length (shorter = sharper transition)
    width = max(1.0, 40.0 / math.sqrt(strand_length_nt))
    x = (temp_C - tm_C) / width
    # Clamp to avoid overflow
    x = max(-50.0, min(50.0, x))
    return 1.0 / (1.0 + math.exp(-x))


def eutectic_concentration_factor(
    temp_C: float = EUTECTIC_TEMP_C,
    initial_concentration_uM: float = 1.0,
) -> float:
    """Concentration enhancement in eutectic ice pockets.

    As water freezes, solutes concentrate in liquid veins.
    Factor depends on temperature below freezing.
    """
    if temp_C >= 0.0:
        return 1.0
    # Empirical: ~10-100x concentration at eutectic
    # Roughly linear with degrees below freezing
    return min(100.0, 1.0 + abs(temp_C) * 10.0)


# ---------------------------------------------------------------------------
# Kinetic trapping model
# ---------------------------------------------------------------------------

@dataclass
class TripletTrapping:
    """Triplet kinetic trapping state during freeze-thaw cycling."""

    n_triplets: int = 15                        # number of triplets in strand
    binding_energy_per_triplet_kJ: float = 15.0
    temp_K: float = EUTECTIC_TEMP_K
    # Computed
    trapping_probability: float = 0.0           # P(triplet kinetically trapped)
    stalled_triplets: int = 0                   # expected number of stalled triplets

    def compute(self) -> None:
        """Compute trapping probability at current temperature."""
        # Boltzmann factor: P(trapped) = 1 - exp(-E_bind / RT)
        energy_J_per_mol = self.binding_energy_per_triplet_kJ * 1000.0
        rt = R_GAS * self.temp_K
        if rt > 0:
            boltzmann = math.exp(-energy_J_per_mol / rt)
            self.trapping_probability = 1.0 - boltzmann
        else:
            self.trapping_probability = 1.0
        self.stalled_triplets = int(round(self.n_triplets * self.trapping_probability))


# ---------------------------------------------------------------------------
# Replication kinetics
# ---------------------------------------------------------------------------

def replication_rate_constant(
    temp_C: float,
    activation_energy_kJ: float = 80.0,
    pre_exponential: float = 1e10,
) -> float:
    """Arrhenius rate constant for replication at given temperature.

    k = A * exp(-Ea / RT)
    """
    temp_K = temp_C + 273.15
    if temp_K <= 0:
        return 0.0
    ea_J = activation_energy_kJ * 1000.0
    exponent = -ea_J / (R_GAS * temp_K)
    exponent = max(-500.0, exponent)  # clamp to avoid underflow
    return pre_exponential * math.exp(exponent)


def replication_stall_probability(
    params: FreezeThawParams = FreezeThawParams(),
) -> float:
    """Probability that replication stalls during a freeze phase.

    Stall occurs when:
    1. Temperature drops below kinetic threshold, AND
    2. Triplet trapping locks incomplete replication intermediates
    """
    # Rate at freeze temp vs thaw temp
    k_freeze = replication_rate_constant(
        params.freeze_temp_C,
        params.activation_energy_kJ_per_mol,
        params.pre_exponential_factor,
    )
    k_thaw = replication_rate_constant(
        params.thaw_temp_C,
        params.activation_energy_kJ_per_mol,
        params.pre_exponential_factor,
    )

    if k_thaw <= 0:
        return 1.0

    rate_ratio = k_freeze / k_thaw

    # Triplet trapping contribution
    trapping = TripletTrapping(
        n_triplets=params.strand_length_nt // 3,
        binding_energy_per_triplet_kJ=params.triplet_binding_energy_kJ_per_mol,
        temp_K=params.freeze_temp_C + 273.15,
    )
    trapping.compute()

    # Stall probability: combination of slow kinetics and trapping
    # P(stall) = 1 - (1 - P(kinetic_slow)) * (1 - P(trapped_enough))
    p_kinetic_slow = 1.0 - rate_ratio   # approaches 1 when freeze rate << thaw rate
    p_kinetic_slow = max(0.0, min(1.0, p_kinetic_slow))
    p_trapped = trapping.trapping_probability

    p_stall = 1.0 - (1.0 - p_kinetic_slow) * (1.0 - p_trapped)
    return round(min(1.0, max(0.0, p_stall)), 6)


# ---------------------------------------------------------------------------
# Optimal freeze-thaw interval
# ---------------------------------------------------------------------------

@dataclass
class FreezeThawInterval:
    """Result of optimal interval determination."""

    optimal_freeze_hours: float = 0.0
    optimal_thaw_hours: float = 0.0
    optimal_cycle_hours: float = 0.0
    replication_efficiency: float = 0.0         # fraction of cycle yielding replication
    stall_probability: float = 0.0
    concentration_factor: float = 0.0


def optimize_freeze_thaw_interval(
    params: FreezeThawParams = FreezeThawParams(),
    min_freeze_hours: float = 2.0,
    max_freeze_hours: float = 48.0,
    min_thaw_hours: float = 2.0,
    max_thaw_hours: float = 48.0,
    steps: int = 20,
) -> FreezeThawInterval:
    """Determine optimal freeze-thaw interval for replication fidelity.

    Balances:
    - Longer freeze: better strand separation + concentration
    - Longer thaw: more replication time
    - But: longer freeze increases trapping risk
    """
    best_score = -1.0
    best_result = FreezeThawInterval()

    for fi in range(steps):
        freeze_h = min_freeze_hours + (max_freeze_hours - min_freeze_hours) * fi / max(1, steps - 1)
        for ti in range(steps):
            thaw_h = min_thaw_hours + (max_thaw_hours - min_thaw_hours) * ti / max(1, steps - 1)

            test_params = FreezeThawParams(
                freeze_temp_C=params.freeze_temp_C,
                thaw_temp_C=params.thaw_temp_C,
                strand_length_nt=params.strand_length_nt,
                gc_fraction=params.gc_fraction,
                triplet_binding_energy_kJ_per_mol=params.triplet_binding_energy_kJ_per_mol,
                activation_energy_kJ_per_mol=params.activation_energy_kJ_per_mol,
                pre_exponential_factor=params.pre_exponential_factor,
                freeze_duration_hours=freeze_h,
                thaw_duration_hours=thaw_h,
            )

            p_stall = replication_stall_probability(test_params)
            conc_factor = eutectic_concentration_factor(params.freeze_temp_C)

            # Separation probability (at thaw temp, strands should separate during freeze)
            p_sep = strand_separation_probability(
                params.thaw_temp_C,
                params.strand_length_nt,
                params.gc_fraction,
            )

            cycle_h = freeze_h + thaw_h
            replication_frac = thaw_h / cycle_h

            # Score: maximize replication efficiency while minimizing stall
            # and leveraging concentration
            score = replication_frac * (1.0 - p_stall) * math.log1p(conc_factor) * p_sep

            if score > best_score:
                best_score = score
                best_result = FreezeThawInterval(
                    optimal_freeze_hours=round(freeze_h, 2),
                    optimal_thaw_hours=round(thaw_h, 2),
                    optimal_cycle_hours=round(cycle_h, 2),
                    replication_efficiency=round(replication_frac * (1.0 - p_stall), 4),
                    stall_probability=p_stall,
                    concentration_factor=round(conc_factor, 2),
                )

    return best_result


# ---------------------------------------------------------------------------
# Full freeze-thaw analysis
# ---------------------------------------------------------------------------

@dataclass
class FreezeThawAnalysis:
    """Complete freeze-thaw kinetic analysis for Nexus."""

    params: FreezeThawParams = field(default_factory=FreezeThawParams)
    stall_probability: float = 0.0
    strand_separation_at_freeze: float = 0.0
    strand_separation_at_thaw: float = 0.0
    concentration_factor: float = 0.0
    optimal_interval: FreezeThawInterval = field(default_factory=FreezeThawInterval)
    triplet_trapping: TripletTrapping = field(default_factory=TripletTrapping)


def analyze_freeze_thaw(
    params: Optional[FreezeThawParams] = None,
) -> FreezeThawAnalysis:
    """Run complete freeze-thaw kinetic analysis."""
    if params is None:
        params = FreezeThawParams()

    p_stall = replication_stall_probability(params)
    p_sep_freeze = strand_separation_probability(
        params.freeze_temp_C, params.strand_length_nt, params.gc_fraction,
    )
    p_sep_thaw = strand_separation_probability(
        params.thaw_temp_C, params.strand_length_nt, params.gc_fraction,
    )
    conc = eutectic_concentration_factor(params.freeze_temp_C)
    optimal = optimize_freeze_thaw_interval(params)

    trapping = TripletTrapping(
        n_triplets=params.strand_length_nt // 3,
        binding_energy_per_triplet_kJ=params.triplet_binding_energy_kJ_per_mol,
        temp_K=params.freeze_temp_C + 273.15,
    )
    trapping.compute()

    return FreezeThawAnalysis(
        params=params,
        stall_probability=p_stall,
        strand_separation_at_freeze=round(p_sep_freeze, 6),
        strand_separation_at_thaw=round(p_sep_thaw, 6),
        concentration_factor=round(conc, 2),
        optimal_interval=optimal,
        triplet_trapping=trapping,
    )
