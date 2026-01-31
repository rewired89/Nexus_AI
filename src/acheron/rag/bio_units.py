"""Biological Information Module (BIM) quantitative framework.

Provides calculators for proving the existence of the "Biological Bit":
  1. Nernst equilibrium potential (E_ion)
  2. State Stability: T_half for Vmem persistence against metabolic noise
  3. Switching Energy: ATP cost per bit-flip (E_bit)
  4. Error Rate: Probability of stochastic bit-flip from ion channel noise
  5. Shannon Entropy: Information content of a bioelectric state space
  6. Channel Capacity: Maximum information throughput of a gap junction bus

Hardware Specification Library:
  - CPU: Nav/Kv channel arrays (switching speed, gate logic)
  - RAM: Vmem gradient across syncytium (read/write latency)
  - SSD: Innexin-gated connectivity patterns (anatomical memory)
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# ======================================================================
# Physical constants
# ======================================================================
R = 8.314           # J/(mol·K), gas constant
F = 96485.0         # C/mol, Faraday constant
K_B = 1.381e-23     # J/K, Boltzmann constant
E_CHARGE = 1.602e-19  # C, elementary charge
ATP_ENERGY = 5.0e-20  # J, free energy per ATP hydrolysis (~30.5 kJ/mol)
ROOM_TEMP_K = 298.15  # K, 25°C
PLANARIAN_TEMP_K = 293.15  # K, 20°C (planarian culture temperature)


# ======================================================================
# Nernst Equation
# ======================================================================
def nernst_potential(
    z: int,
    conc_out: float,
    conc_in: float,
    temp_k: float = PLANARIAN_TEMP_K,
) -> float:
    """Calculate the Nernst equilibrium potential E_ion in volts.

    E_ion = (RT / zF) * ln([Ion]_out / [Ion]_in)

    Args:
        z: Ion valence (+1 for K+/Na+, -1 for Cl-, +2 for Ca2+).
        conc_out: Extracellular concentration (mM).
        conc_in: Intracellular concentration (mM).
        temp_k: Temperature in Kelvin (default: 20°C for planarians).

    Returns:
        E_ion in volts. Multiply by 1000 for mV.
    """
    if conc_in <= 0 or conc_out <= 0:
        raise ValueError("Concentrations must be positive")
    if z == 0:
        raise ValueError("Ion valence z must be non-zero")
    return (R * temp_k) / (z * F) * math.log(conc_out / conc_in)


def nernst_mv(
    z: int,
    conc_out: float,
    conc_in: float,
    temp_k: float = PLANARIAN_TEMP_K,
) -> float:
    """Nernst potential in millivolts (convenience wrapper)."""
    return nernst_potential(z, conc_out, conc_in, temp_k) * 1000.0


# ======================================================================
# Standard ion concentrations
# ======================================================================
@dataclass
class IonProfile:
    """Intracellular/extracellular ion concentrations for a cell type."""

    organism: str
    cell_type: str
    # Concentrations in mM
    k_in: float
    k_out: float
    na_in: float
    na_out: float
    cl_in: float
    cl_out: float
    ca_in: float = 0.0001  # 100 nM resting [Ca2+]_in
    ca_out: float = 1.5
    temp_k: float = PLANARIAN_TEMP_K
    source: str = ""

    def e_k(self) -> float:
        """K+ equilibrium potential in mV."""
        return nernst_mv(+1, self.k_out, self.k_in, self.temp_k)

    def e_na(self) -> float:
        """Na+ equilibrium potential in mV."""
        return nernst_mv(+1, self.na_out, self.na_in, self.temp_k)

    def e_cl(self) -> float:
        """Cl- equilibrium potential in mV."""
        return nernst_mv(-1, self.cl_out, self.cl_in, self.temp_k)

    def e_ca(self) -> float:
        """Ca2+ equilibrium potential in mV."""
        if self.ca_in <= 0 or self.ca_out <= 0:
            return 0.0
        return nernst_mv(+2, self.ca_out, self.ca_in, self.temp_k)


# Known / estimated profiles
XENOPUS_OOCYTE = IonProfile(
    organism="Xenopus laevis",
    cell_type="oocyte",
    k_in=120.0, k_out=2.5,
    na_in=10.0, na_out=110.0,
    cl_in=30.0, cl_out=110.0,
    temp_k=295.15,  # 22°C
    source="Hodgkin & Huxley (adapted), textbook values",
)

PLANARIAN_NEOBLAST_HEURISTIC = IonProfile(
    organism="Schmidtea mediterranea",
    cell_type="neoblast [HEURISTIC]",
    k_in=120.0, k_out=3.0,    # freshwater, extrapolated from Xenopus
    na_in=12.0, na_out=5.0,   # low-Na freshwater environment
    cl_in=20.0, cl_out=4.0,   # freshwater
    temp_k=PLANARIAN_TEMP_K,
    source="[HEURISTIC] Extrapolated from Xenopus, adjusted for freshwater",
)

PHYSARUM = IonProfile(
    organism="Physarum polycephalum",
    cell_type="plasmodium [HEURISTIC]",
    k_in=100.0, k_out=1.0,    # very low external K in pond water
    na_in=10.0, na_out=2.0,
    cl_in=15.0, cl_out=2.0,
    temp_k=295.15,
    source="[HEURISTIC] Extrapolated from known protist values",
)


# ======================================================================
# Biological Bit metrics
# ======================================================================
@dataclass
class BiologicalBit:
    """Quantitative proof for the existence of a biological bit.

    A biological bit is defined as a stable, switchable, readable
    unit of bioelectric information.
    """

    # State Stability
    t_half_seconds: float  # Half-life of Vmem shift before noise dissipation
    t_half_label: str = ""

    # Switching Energy
    e_bit_joules: float = 0.0       # Energy per bit-flip
    atp_per_flip: float = 0.0       # ATP molecules consumed per flip
    ions_per_flip: int = 0          # Ions transported per switching event

    # Error Rate
    bit_flip_probability: float = 0.0  # P(stochastic flip) per unit time
    noise_source: str = ""

    # Information metrics
    shannon_entropy_bits: float = 0.0  # H(X) of the state space
    channel_capacity_bits_per_s: float = 0.0  # C of the signaling channel


def calc_switching_energy(
    delta_v_mv: float,
    ions_per_event: int,
    capacitance_pf: float = 10.0,
) -> tuple[float, float]:
    """Calculate the energy cost of a bioelectric bit-flip.

    E_bit = (1/2) * C * (ΔV)^2   (capacitive energy)
    or
    E_bit = ions_per_event * E_per_ion

    where E_per_ion = e * ΔV for each ion transported.

    Also calculates ATP cost:
    ATP_per_flip = E_bit / ATP_ENERGY

    Args:
        delta_v_mv: Voltage swing for the bit-flip in mV.
        ions_per_event: Number of ions transported per switching event.
        capacitance_pf: Membrane capacitance in picofarads (typical cell ~10 pF).

    Returns:
        (e_bit_joules, atp_per_flip)
    """
    delta_v = delta_v_mv * 1e-3  # Convert mV to V
    cap = capacitance_pf * 1e-12  # Convert pF to F

    # Capacitive energy
    e_cap = 0.5 * cap * delta_v ** 2

    # Ion transport energy
    e_ion = ions_per_event * E_CHARGE * abs(delta_v)

    # Use the larger estimate (more conservative)
    e_bit = max(e_cap, e_ion)
    atp_per_flip = e_bit / ATP_ENERGY

    return e_bit, atp_per_flip


def calc_state_stability(
    membrane_resistance_mohm: float = 500.0,
    capacitance_pf: float = 10.0,
) -> float:
    """Estimate the RC time constant (passive decay) for a Vmem state.

    tau = R_m * C_m

    This gives the time constant for passive voltage decay.
    T_half = tau * ln(2).

    For an actively maintained state (with ion pumps), T_half is
    effectively infinite as long as ATP is available.

    Args:
        membrane_resistance_mohm: Membrane resistance in megaohms.
        capacitance_pf: Membrane capacitance in picofarads.

    Returns:
        T_half in seconds (passive, without active pump maintenance).
    """
    r_ohm = membrane_resistance_mohm * 1e6
    c_farad = capacitance_pf * 1e-12
    tau = r_ohm * c_farad
    return tau * math.log(2)


def calc_channel_noise_error_rate(
    n_channels: int = 100,
    p_open_resting: float = 0.01,
    p_open_threshold: float = 0.10,
    observation_window_s: float = 1.0,
    switching_rate_hz: float = 1000.0,
) -> float:
    """Estimate the probability of a stochastic bit-flip from channel noise.

    Models the probability that enough channels open simultaneously
    (by random fluctuation) to cross the threshold for a state change.

    Uses a binomial approximation:
    P(k >= k_threshold | N, p_open) ≈ exp(-N * D_KL(p_threshold || p_open))

    where D_KL is the Kullback-Leibler divergence.

    Args:
        n_channels: Number of ion channels in the relevant membrane patch.
        p_open_resting: Probability a single channel is open at resting Vmem.
        p_open_threshold: Fraction that must open to trigger a state change.
        observation_window_s: Time window in seconds.
        switching_rate_hz: Channel gating transition rate.

    Returns:
        Probability of a bit-flip per observation window.
    """
    if p_open_resting <= 0 or p_open_resting >= 1:
        return 0.0
    if p_open_threshold <= p_open_resting:
        return 1.0  # threshold already exceeded

    # KL divergence D(p_threshold || p_resting) for Bernoulli
    p = p_open_threshold
    q = p_open_resting
    d_kl = p * math.log(p / q) + (1 - p) * math.log((1 - p) / (1 - q))

    # Probability per gating event
    p_flip_per_event = math.exp(-n_channels * d_kl)

    # Number of independent gating events in the observation window
    n_events = switching_rate_hz * observation_window_s

    # Probability of at least one flip
    p_no_flip = (1 - p_flip_per_event) ** n_events
    return 1 - p_no_flip


def calc_shannon_entropy(
    n_stable_states: int,
    state_probabilities: list[float] | None = None,
) -> float:
    """Calculate Shannon entropy of a bioelectric state space.

    H(X) = -sum(p_i * log2(p_i))

    If state_probabilities is None, assumes uniform distribution.

    Args:
        n_stable_states: Number of distinguishable stable Vmem states.
        state_probabilities: Optional probability for each state.

    Returns:
        Shannon entropy in bits.
    """
    if n_stable_states <= 1:
        return 0.0

    if state_probabilities is None:
        # Uniform distribution: H = log2(N)
        return math.log2(n_stable_states)

    entropy = 0.0
    for p in state_probabilities:
        if p > 0:
            entropy -= p * math.log2(p)
    return entropy


def calc_channel_capacity(
    bandwidth_hz: float,
    snr_linear: float,
) -> float:
    """Shannon-Hartley channel capacity for a biological signaling channel.

    C = B * log2(1 + SNR)

    Args:
        bandwidth_hz: Signal bandwidth in Hz (e.g., gap junction bandwidth).
        snr_linear: Signal-to-noise ratio (linear, not dB).

    Returns:
        Channel capacity in bits/second.
    """
    if bandwidth_hz <= 0 or snr_linear <= 0:
        return 0.0
    return bandwidth_hz * math.log2(1 + snr_linear)


def build_biological_bit(
    delta_v_mv: float = 30.0,
    ions_per_event: int = 10000,
    capacitance_pf: float = 10.0,
    membrane_resistance_mohm: float = 500.0,
    n_channels: int = 100,
    p_open_resting: float = 0.01,
    p_open_threshold: float = 0.10,
    n_stable_states: int = 4,
    gj_bandwidth_hz: float = 100.0,
    gj_snr_linear: float = 10.0,
) -> BiologicalBit:
    """Build a complete BiologicalBit specification from parameters.

    Default values are for a planarian neoblast [HEURISTIC].
    """
    e_bit, atp = calc_switching_energy(delta_v_mv, ions_per_event, capacitance_pf)
    t_half = calc_state_stability(membrane_resistance_mohm, capacitance_pf)
    error = calc_channel_noise_error_rate(
        n_channels, p_open_resting, p_open_threshold
    )
    entropy = calc_shannon_entropy(n_stable_states)
    capacity = calc_channel_capacity(gj_bandwidth_hz, gj_snr_linear)

    return BiologicalBit(
        t_half_seconds=round(t_half, 4),
        t_half_label=_format_time(t_half),
        e_bit_joules=e_bit,
        atp_per_flip=round(atp, 2),
        ions_per_flip=ions_per_event,
        bit_flip_probability=error,
        noise_source="Stochastic ion channel gating",
        shannon_entropy_bits=round(entropy, 3),
        channel_capacity_bits_per_s=round(capacity, 2),
    )


def _format_time(seconds: float) -> str:
    """Human-readable time label."""
    if seconds < 0.001:
        return f"{seconds * 1e6:.1f} μs"
    if seconds < 1:
        return f"{seconds * 1000:.1f} ms"
    if seconds < 60:
        return f"{seconds:.2f} s"
    if seconds < 3600:
        return f"{seconds / 60:.1f} min"
    return f"{seconds / 3600:.1f} h"


# ======================================================================
# Hardware Specification Library
# ======================================================================
@dataclass
class HardwareComponent:
    """A certified biological hardware component for Acheron."""

    name: str
    digital_equivalent: str
    biological_basis: str
    read_latency: str
    write_latency: str
    persistence: str
    key_molecules: list[str]
    organism_availability: str
    evidence_level: str  # [EVIDENCED], [INFERRED], [HEURISTIC]


CPU_NAV_KV = HardwareComponent(
    name="Nav/Kv Channel Array",
    digital_equivalent="CPU — switching logic gates",
    biological_basis=(
        "Voltage-gated Na+ and K+ channels form the core switching "
        "elements. Nav channels open at depolarization threshold (~-40 mV), "
        "producing rapid inward current (bit SET). Kv channels activate with "
        "delay, producing outward current (bit RESET). Together they "
        "implement a NOT gate: depolarization → Nav opens → further "
        "depolarization → Kv opens → repolarization (inversion)."
    ),
    read_latency="~1 ms (action potential propagation speed)",
    write_latency="~0.5 ms (channel gating transition)",
    persistence="Volatile — requires continuous ion flux to maintain state",
    key_molecules=[
        "SCN (Nav1.x family) — voltage-gated sodium channels",
        "KCNA/KCNB (Kv1/Kv2) — voltage-gated potassium channels",
        "KCNK (K2P) — leak channels setting resting potential",
    ],
    organism_availability="Universal eukaryotic; planarian homologs confirmed",
    evidence_level="[EVIDENCED] — electrophysiology in multiple phyla",
)

RAM_VMEM_GRADIENT = HardwareComponent(
    name="Vmem Gradient Across Syncytium",
    digital_equivalent="RAM — volatile read/write memory",
    biological_basis=(
        "The membrane potential (Vmem) across a tissue forms a spatial "
        "voltage map. Each cell's Vmem encodes a state (depolarized / "
        "hyperpolarized). Gap junctions (innexins in planarians) couple "
        "cells into a syncytium, allowing voltage signals to propagate "
        "and form stable patterns. The pattern is actively maintained "
        "by ion pump activity (H+/K+-ATPase, V-ATPase)."
    ),
    read_latency="~30 s (DiBAC4(3) equilibration) to ~1 ms (electrode)",
    write_latency="~seconds (pharmacological) to ~ms (optogenetic)",
    persistence=(
        "Semi-volatile — persists as long as pumps are active (~hours "
        "without ATP). In planarians, the target morphology pattern "
        "persists indefinitely through regeneration, suggesting a "
        "non-volatile backup exists."
    ),
    key_molecules=[
        "H+/K+-ATPase — sets anterior hyperpolarization",
        "V-ATPase — proton pump contributing to Vmem",
        "Innexin-7, Innexin-13 — gap junction coupling (planarian)",
        "Connexin-43 — gap junction coupling (vertebrate equivalent)",
    ],
    organism_availability=(
        "Planarian: innexin-based; Xenopus: connexin-based; "
        "Physarum: actin-tube-coupled oscillation"
    ),
    evidence_level="[EVIDENCED] — DiBAC imaging, pharmacological manipulation",
)

SSD_INNEXIN_CONNECTIVITY = HardwareComponent(
    name="Innexin-Gated Connectivity Pattern",
    digital_equivalent="SSD — non-volatile anatomical memory",
    biological_basis=(
        "The pattern of gap junction connectivity (which cells are "
        "electrically coupled) encodes long-term structural information. "
        "This 'wiring diagram' is determined by innexin isoform "
        "expression and is maintained through cell division. In "
        "planarians, two-headed animals maintain their altered "
        "connectivity pattern through unlimited regeneration cycles, "
        "demonstrating true non-volatile storage. The target "
        "morphology acts as a checksum — cells compare their local "
        "state against the stored pattern and rebuild until consistency "
        "is achieved."
    ),
    read_latency="~minutes (dye coupling assay) to ~hours (gene expression)",
    write_latency="~24-48h (RNAi knockdown of specific innexins)",
    persistence=(
        "Non-volatile — persists through amputation and regeneration "
        "indefinitely (>18 months demonstrated in 2-headed planarians)"
    ),
    key_molecules=[
        "Innexin-7 — primary gap junction protein in planarian epidermis",
        "Innexin-13 — neoblast-specific gap junction",
        "Beta-catenin — downstream effector of stored polarity",
        "Wnt / Notum — pathway controlled by connectivity pattern",
    ],
    organism_availability=(
        "Planarian: innexin family (>15 isoforms); not directly "
        "transferable to vertebrate connexin system without mapping"
    ),
    evidence_level=(
        "[EVIDENCED] for persistence; [INFERRED] for encoding mechanism"
    ),
)

HARDWARE_LIBRARY = {
    "cpu": CPU_NAV_KV,
    "ram": RAM_VMEM_GRADIENT,
    "ssd": SSD_INNEXIN_CONNECTIVITY,
}


def format_hardware_library() -> str:
    """Format the hardware library as a readable text block."""
    lines = ["ACHERON HARDWARE SPECIFICATION LIBRARY", "=" * 50]
    for key, comp in HARDWARE_LIBRARY.items():
        lines.extend([
            "",
            f"[{key.upper()}] {comp.name}",
            f"  Digital Equivalent: {comp.digital_equivalent}",
            f"  Biological Basis: {comp.biological_basis}",
            f"  Read Latency: {comp.read_latency}",
            f"  Write Latency: {comp.write_latency}",
            f"  Persistence: {comp.persistence}",
            f"  Key Molecules: {', '.join(comp.key_molecules)}",
            f"  Organism Availability: {comp.organism_availability}",
            f"  Evidence Level: {comp.evidence_level}",
        ])
    return "\n".join(lines)
