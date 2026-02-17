"""DenRAM -> Bioelectric Delay Encoding (Section 3).

Temporal encoding model for bioelectric state representation:

State encoding modes:
    A) Static attractor (Vm steady state)
    B) Phase-delay encoding (oscillatory timing)

Delay parameters:
    - Ion channel kinetics as delay variable
    - Gap junction diffusion time as delay constant

Simulation targets:
    - Stability of phase-encoded data
    - Noise tolerance under metabolic fluctuation
    - Cross-talk probability between adjacent clusters
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class EncodingMode(str, Enum):
    """State encoding mode selection."""

    STATIC_ATTRACTOR = "static_attractor"       # Vm steady state
    PHASE_DELAY = "phase_delay"                 # oscillatory timing encoding


@dataclass
class ChannelKinetics:
    """Ion channel kinetic parameters serving as delay variables.

    tau_activation and tau_inactivation define the temporal dynamics
    of the channel's contribution to encoding.
    """

    channel_type: str = "generic"                   # e.g., "Kv2.1", "Nav1.5"
    tau_activation_ms: float = 1.0                  # activation time constant
    tau_inactivation_ms: float = 5.0                # inactivation time constant
    v_half_activation_mV: float = -20.0             # half-activation voltage
    slope_factor_mV: float = 10.0                   # Boltzmann slope
    max_conductance_nS: float = 10.0                # peak conductance
    evidence_status: str = "[DATA GAP]"


@dataclass
class GapJunctionDelay:
    """Gap junction diffusion parameters as delay constants."""

    junction_type: str = "innexin"                  # innexin | connexin
    diffusion_time_ms: float = 0.5                  # signal propagation delay
    conductance_nS: float = 2.0                     # electrical coupling strength
    distance_um: float = 10.0                       # inter-cluster distance
    # Derived: effective propagation velocity
    velocity_um_per_ms: float = 0.0

    def __post_init__(self):
        if self.diffusion_time_ms > 0:
            self.velocity_um_per_ms = self.distance_um / self.diffusion_time_ms


# ---------------------------------------------------------------------------
# Static attractor model
# ---------------------------------------------------------------------------

@dataclass
class AttractorState:
    """A stable Vm attractor (static encoding mode A)."""

    state_id: str
    vm_target_mV: float = -40.0             # target membrane potential
    basin_width_mV: float = 10.0            # half-width of attraction basin
    stability_tau_ms: float = 100.0         # time constant for return to attractor
    energy_barrier_kT: float = 5.0          # barrier height in kT units


@dataclass
class StaticAttractorModel:
    """Static attractor encoding: Vm steady states as data carriers."""

    attractors: list[AttractorState] = field(default_factory=list)
    noise_amplitude_mV: float = 2.0         # metabolic noise amplitude
    temperature_K: float = 293.15           # operating temperature

    def escape_probability(self, attractor: AttractorState, observation_window_ms: float = 1000.0) -> float:
        """Kramers escape rate: probability of leaving attractor basin.

        P_escape ≈ 1 - exp(-t / tau_escape)
        tau_escape ≈ tau_stability * exp(E_barrier / kT_noise)
        """
        if attractor.energy_barrier_kT <= 0:
            return 1.0
        # Effective noise in kT units
        k_B = 1.381e-23  # J/K
        noise_energy_J = 0.5 * (self.noise_amplitude_mV * 1e-3) ** 2 * 1e-12  # approx capacitive
        thermal_energy_J = k_B * self.temperature_K
        effective_kT = max(thermal_energy_J, noise_energy_J)
        barrier_J = attractor.energy_barrier_kT * thermal_energy_J

        tau_escape_ms = attractor.stability_tau_ms * math.exp(barrier_J / effective_kT)
        # Clamp to avoid overflow
        tau_escape_ms = min(tau_escape_ms, 1e15)

        return 1.0 - math.exp(-observation_window_ms / tau_escape_ms)

    def state_stability(self, attractor: AttractorState) -> float:
        """Stability metric: expected persistence time in ms."""
        if attractor.energy_barrier_kT <= 0:
            return 0.0
        k_B = 1.381e-23
        thermal_energy_J = k_B * self.temperature_K
        barrier_J = attractor.energy_barrier_kT * thermal_energy_J
        noise_energy_J = 0.5 * (self.noise_amplitude_mV * 1e-3) ** 2 * 1e-12
        effective_kT = max(thermal_energy_J, noise_energy_J)
        return attractor.stability_tau_ms * math.exp(barrier_J / effective_kT)


# ---------------------------------------------------------------------------
# Phase-delay encoding model
# ---------------------------------------------------------------------------

@dataclass
class PhaseEncoder:
    """Phase-delay encoding: oscillatory timing as data carrier (mode B).

    Information is encoded in the phase relationship between oscillating
    cell clusters, with ion channel kinetics defining delay.
    """

    frequency_Hz: float = 1.0                       # oscillation frequency
    n_phase_bins: int = 8                           # number of distinguishable phases
    channel_kinetics: ChannelKinetics = field(default_factory=ChannelKinetics)
    gap_junction_delay: GapJunctionDelay = field(default_factory=GapJunctionDelay)

    @property
    def period_ms(self) -> float:
        return 1000.0 / self.frequency_Hz if self.frequency_Hz > 0 else float("inf")

    @property
    def phase_resolution_ms(self) -> float:
        """Minimum resolvable phase difference in ms."""
        return self.period_ms / self.n_phase_bins

    def encode_value(self, value: int) -> float:
        """Encode an integer value as a phase delay in ms.

        value must be in [0, n_phase_bins).
        """
        if value < 0 or value >= self.n_phase_bins:
            raise ValueError(f"Value {value} outside [0, {self.n_phase_bins})")
        return value * self.phase_resolution_ms

    def decode_phase(self, delay_ms: float) -> int:
        """Decode a phase delay back to integer value."""
        if self.phase_resolution_ms <= 0:
            return 0
        raw = delay_ms / self.phase_resolution_ms
        return int(round(raw)) % self.n_phase_bins


# ---------------------------------------------------------------------------
# Noise and cross-talk simulation
# ---------------------------------------------------------------------------

@dataclass
class DelayEncodingMetrics:
    """Results from delay encoding simulation."""

    encoding_mode: EncodingMode = EncodingMode.STATIC_ATTRACTOR
    # Stability
    phase_stability_score: float = 0.0      # 0-1, fraction of correctly decoded values
    mean_persistence_ms: float = 0.0        # expected state lifetime
    # Noise tolerance
    noise_tolerance_mV: float = 0.0         # max noise before decoding error
    metabolic_fluctuation_tolerance: float = 0.0  # fractional tolerance
    # Cross-talk
    cross_talk_probability: float = 0.0     # P(adjacent cluster interference)
    min_cluster_separation_um: float = 0.0  # minimum separation for < 1% cross-talk


def simulate_phase_stability(
    encoder: PhaseEncoder,
    noise_amplitude_mV: float = 2.0,
    n_trials: int = 1000,
    seed: Optional[int] = None,
) -> float:
    """Simulate phase encoding stability under noise.

    Returns fraction of correctly decoded values after adding
    Gaussian noise to phase delays.
    """
    import random as _rng
    if seed is not None:
        _rng.seed(seed)

    correct = 0
    for _ in range(n_trials):
        value = _rng.randint(0, encoder.n_phase_bins - 1)
        phase_ms = encoder.encode_value(value)
        # Add noise proportional to channel kinetics jitter
        jitter_ms = _rng.gauss(0, noise_amplitude_mV * encoder.channel_kinetics.tau_activation_ms / 10.0)
        noisy_phase = phase_ms + jitter_ms
        decoded = encoder.decode_phase(noisy_phase)
        if decoded == value:
            correct += 1

    return correct / n_trials


def compute_cross_talk_probability(
    separation_um: float,
    conductance_nS: float = 2.0,
    length_constant_um: float = 50.0,
) -> float:
    """Probability of cross-talk between adjacent clusters.

    Model: exponential decay of coupling with distance.
    P_crosstalk = G * exp(-d / lambda)
    Normalized to [0, 1].
    """
    # Normalize conductance contribution
    g_factor = min(1.0, conductance_nS / 10.0)
    spatial_decay = math.exp(-separation_um / length_constant_um)
    return g_factor * spatial_decay


def compute_noise_tolerance(
    encoder: PhaseEncoder,
    target_accuracy: float = 0.99,
    max_noise_mV: float = 20.0,
    steps: int = 100,
    seed: Optional[int] = 42,
) -> float:
    """Find maximum noise amplitude that maintains target decoding accuracy."""
    for i in range(steps):
        noise = max_noise_mV * (i + 1) / steps
        accuracy = simulate_phase_stability(encoder, noise, n_trials=500, seed=seed)
        if accuracy < target_accuracy:
            return max_noise_mV * i / steps
    return max_noise_mV


def analyze_delay_encoding(
    encoding_mode: EncodingMode = EncodingMode.PHASE_DELAY,
    frequency_Hz: float = 1.0,
    n_phase_bins: int = 8,
    noise_amplitude_mV: float = 2.0,
    cluster_separation_um: float = 20.0,
    seed: Optional[int] = 42,
) -> DelayEncodingMetrics:
    """Run full delay encoding analysis for Nexus."""
    encoder = PhaseEncoder(
        frequency_Hz=frequency_Hz,
        n_phase_bins=n_phase_bins,
    )

    if encoding_mode == EncodingMode.PHASE_DELAY:
        stability = simulate_phase_stability(encoder, noise_amplitude_mV, seed=seed)
        noise_tol = compute_noise_tolerance(encoder, seed=seed)
        cross_talk = compute_cross_talk_probability(cluster_separation_um)
        # Minimum separation for < 1% cross-talk
        length_constant = 50.0
        min_sep = -length_constant * math.log(0.01 / min(1.0, 0.2))  # G_factor ~ 0.2

        return DelayEncodingMetrics(
            encoding_mode=encoding_mode,
            phase_stability_score=round(stability, 4),
            mean_persistence_ms=encoder.period_ms * 100,  # ~100 cycles
            noise_tolerance_mV=round(noise_tol, 2),
            metabolic_fluctuation_tolerance=round(noise_tol / 40.0, 4),  # fraction of Vm range
            cross_talk_probability=round(cross_talk, 6),
            min_cluster_separation_um=round(min_sep, 2),
        )
    else:
        # Static attractor mode
        model = StaticAttractorModel(
            attractors=[
                AttractorState(state_id="depol", vm_target_mV=-20.0, basin_width_mV=10.0, energy_barrier_kT=5.0),
                AttractorState(state_id="hyperpol", vm_target_mV=-60.0, basin_width_mV=10.0, energy_barrier_kT=5.0),
            ],
            noise_amplitude_mV=noise_amplitude_mV,
        )
        persistence = min(model.state_stability(a) for a in model.attractors)
        escape = max(model.escape_probability(a) for a in model.attractors)

        return DelayEncodingMetrics(
            encoding_mode=encoding_mode,
            phase_stability_score=round(1.0 - escape, 4),
            mean_persistence_ms=round(persistence, 2),
            noise_tolerance_mV=round(noise_amplitude_mV * 2.0, 2),
            metabolic_fluctuation_tolerance=round(noise_amplitude_mV / 40.0, 4),
            cross_talk_probability=round(compute_cross_talk_probability(cluster_separation_um), 6),
            min_cluster_separation_um=round(-50.0 * math.log(0.01 / 0.2), 2),
        )
