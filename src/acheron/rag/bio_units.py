"""Biological Information Module (BIM) — quantitative specification framework.

Provides calculators for specifying the measurable parameters of a "Biological Bit":
  1. Nernst equilibrium potential (E_ion) — pure physics, universal constants
  2. State Stability: T_half for Vmem persistence (RC time constant formula)
  3. Switching Energy: energy cost per bit-flip (capacitive + ion transport)
  4. Error Rate: stochastic bit-flip probability from ion channel noise model
  5. Shannon Entropy: information content of a bioelectric state space
  6. Channel Capacity: Shannon-Hartley for gap junction signaling bandwidth

IMPORTANT — NO-NUMERIC-INVENTION POLICY:
  These are CALCULATORS, not oracles.  Output is valid only when inputs are:
    (a) directly cited from a published source, OR
    (b) derived purely from universal physical constants with stated assumptions.
  If input parameters are unmeasured, the output must be labeled [DATA GAP]
  and accompanied by an experimental measurement plan.

Hardware Specification Library:
  - CPU: Nav/Kv channel arrays (switching logic)
  - RAM: Vmem gradient across syncytium (bioelectric memory)
  - SSD: Innexin-gated connectivity patterns (anatomical memory)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

# ======================================================================
# Physical constants (universal — not species-specific)
# ======================================================================
R = 8.314           # J/(mol·K), gas constant
F = 96485.0         # C/mol, Faraday constant
K_B = 1.381e-23     # J/K, Boltzmann constant
E_CHARGE = 1.602e-19  # C, elementary charge
ATP_ENERGY = 5.0e-20  # J, free energy per ATP hydrolysis (~30.5 kJ/mol)
ROOM_TEMP_K = 298.15  # K, 25°C
PLANARIAN_TEMP_K = 293.15  # K, 20°C (planarian culture temperature)


# ======================================================================
# Nernst Equation — pure physics bound
# ======================================================================
def nernst_potential(
    z: int,
    conc_out: float,
    conc_in: float,
    temp_k: float = PLANARIAN_TEMP_K,
) -> float:
    """Calculate the Nernst equilibrium potential E_ion in volts.

    E_ion = (RT / zF) * ln([Ion]_out / [Ion]_in)

    This is a pure physics calculation using universal constants.
    The result is valid ONLY if the input concentrations are measured.
    If concentrations are estimated, the output must be labeled [DATA GAP].

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
    """Intracellular/extracellular ion concentrations for a cell type.

    IMPORTANT: The ``evidence_status`` field must accurately reflect
    whether these concentrations were directly measured or extrapolated.
    Nernst potentials calculated from unmeasured concentrations must be
    labeled [DATA GAP] in any output.
    """

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
    evidence_status: str = "[DATA GAP]"  # [EVIDENCED] or [DATA GAP]

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


# --- Profiles with known citation status ---

XENOPUS_OOCYTE = IonProfile(
    organism="Xenopus laevis",
    cell_type="oocyte",
    k_in=120.0, k_out=2.5,
    na_in=10.0, na_out=110.0,
    cl_in=30.0, cl_out=110.0,
    temp_k=295.15,  # 22°C
    source="Hodgkin & Huxley (adapted), textbook values",
    evidence_status="[EVIDENCED]",
)

PLANARIAN_NEOBLAST_HEURISTIC = IonProfile(
    organism="Schmidtea mediterranea",
    cell_type="neoblast",
    k_in=120.0, k_out=3.0,    # extrapolated from Xenopus for freshwater
    na_in=12.0, na_out=5.0,   # estimated for freshwater environment
    cl_in=20.0, cl_out=4.0,   # estimated for freshwater
    temp_k=PLANARIAN_TEMP_K,
    source=(
        "[DATA GAP] No direct ion concentration measurements exist for "
        "S. mediterranea neoblasts. Values extrapolated from Xenopus oocyte "
        "and adjusted for freshwater osmolarity. All Nernst potentials "
        "derived from this profile are UNMEASURED ESTIMATES."
    ),
    evidence_status="[DATA GAP]",
)

PHYSARUM = IonProfile(
    organism="Physarum polycephalum",
    cell_type="plasmodium",
    k_in=100.0, k_out=1.0,    # estimated for pond water
    na_in=10.0, na_out=2.0,
    cl_in=15.0, cl_out=2.0,
    temp_k=295.15,
    source=(
        "[DATA GAP] No direct ion concentration measurements exist for "
        "Physarum plasmodium cytoplasm. Values extrapolated from known "
        "protist literature. All Nernst potentials derived from this "
        "profile are UNMEASURED ESTIMATES."
    ),
    evidence_status="[DATA GAP]",
)


# ======================================================================
# Biological Bit metrics — CALCULATORS (not assertions)
# ======================================================================
@dataclass
class BiologicalBit:
    """Quantitative specification for a candidate biological bit.

    A biological bit is defined as a stable, switchable, readable
    unit of bioelectric information.

    IMPORTANT: This is a SPECIFICATION, not a proof.
    Proof requires experimental measurement of every parameter.
    The ``evidence_status`` field indicates whether values are from
    cited measurements or from unmeasured estimates requiring validation.
    """

    # State Stability
    t_half_seconds: float  # Half-life of Vmem shift (RC passive decay)
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

    # Evidence tracking
    evidence_status: str = "[DATA GAP]"
    measurement_plan: list[str] = field(default_factory=list)


def calc_switching_energy(
    delta_v_mv: float,
    ions_per_event: int,
    capacitance_pf: float = 10.0,
) -> tuple[float, float]:
    """Calculate the energy cost of a bioelectric bit-flip.

    Pure physics formula — valid only when inputs are measured.

    E_bit = max( (1/2)*C*(ΔV)^2 ,  ions * e * ΔV )
    ATP_per_flip = E_bit / ATP_ENERGY

    Args:
        delta_v_mv: Voltage swing for the bit-flip in mV.
        ions_per_event: Number of ions transported per switching event.
        capacitance_pf: Membrane capacitance in picofarads.

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
    """Calculate the RC time constant (passive decay) for a Vmem state.

    Pure physics formula: T_half = R_m * C_m * ln(2)

    This gives the PASSIVE decay half-life. Active ion pumps extend
    persistence indefinitely while ATP is available.

    IMPORTANT: R_m and C_m must be measured for the specific cell type.
    If estimated, label output [DATA GAP].

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
    """Estimate probability of stochastic bit-flip from channel noise.

    Uses a binomial + KL divergence model:
    P(k >= k_threshold | N, p_open) ≈ exp(-N * D_KL(p_threshold || p_open))

    IMPORTANT: n_channels, p_open_resting, p_open_threshold, and
    switching_rate must all be measured for the specific cell type.
    If any are estimated, label output [DATA GAP].

    Args:
        n_channels: Number of ion channels in the relevant membrane patch.
        p_open_resting: Probability a single channel is open at resting Vmem.
        p_open_threshold: Fraction that must open to trigger state change.
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

    Pure information theory: H(X) = -sum(p_i * log2(p_i))

    IMPORTANT: n_stable_states must be experimentally determined
    (e.g., via voltage clamp + bistability assay). If estimated,
    label output [DATA GAP].

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

    Pure information theory: C = B * log2(1 + SNR)

    IMPORTANT: bandwidth_hz and snr_linear must be measured from
    actual gap junction recordings. If estimated, label [DATA GAP].

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
    evidence_status: str = "[DATA GAP]",
) -> BiologicalBit:
    """Build a complete BiologicalBit specification from parameters.

    IMPORTANT: Unless every input parameter is from a cited measurement,
    the resulting BiologicalBit must carry evidence_status="[DATA GAP]"
    and a measurement plan listing what needs to be experimentally determined.
    """
    e_bit, atp = calc_switching_energy(delta_v_mv, ions_per_event, capacitance_pf)
    t_half = calc_state_stability(membrane_resistance_mohm, capacitance_pf)
    error = calc_channel_noise_error_rate(
        n_channels, p_open_resting, p_open_threshold
    )
    entropy = calc_shannon_entropy(n_stable_states)
    capacity = calc_channel_capacity(gj_bandwidth_hz, gj_snr_linear)

    measurement_plan = [
        "Measure R_m and C_m via patch clamp on target cell type",
        "Determine n_stable_states via voltage clamp bistability assay",
        "Count n_channels per membrane patch via single-channel recording",
        "Measure p_open at resting Vmem via single-channel analysis",
        "Determine threshold p_open for state transition",
        "Measure gap junction bandwidth via paired-cell voltage clamp",
        "Measure gap junction SNR from Vmem recordings",
        "Measure delta_V for state transition via current injection",
        "Count ions_per_event via ion-sensitive fluorescence during switching",
    ]

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
        evidence_status=evidence_status,
        measurement_plan=measurement_plan,
    )


def _format_time(seconds: float) -> str:
    """Human-readable time label."""
    if seconds < 0.001:
        return f"{seconds * 1e6:.1f} \u03bcs"
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
    """A candidate biological hardware component for Acheron.

    Each component maps a biological structure to a digital-computing
    equivalent.  The ``evidence_level`` and ``data_gaps`` fields enforce
    the no-numeric-invention policy.
    """

    name: str
    digital_equivalent: str
    biological_basis: str
    read_method: str          # how to read the state (instrument/technique)
    write_method: str         # how to write/set the state
    timescale_class: str      # qualitative: "ms", "seconds", "hours", "days+"
    persistence: str
    key_molecules: list[str]
    organism_availability: str
    evidence_level: str  # [EVIDENCED], [INFERRED], [DATA GAP]
    data_gaps: list[str] = field(default_factory=list)


CPU_NAV_KV = HardwareComponent(
    name="Nav/Kv Channel Array",
    digital_equivalent="CPU — switching logic gates",
    biological_basis=(
        "Voltage-gated Na+ and K+ channels form the core switching "
        "elements. Nav channels open at depolarization threshold, producing "
        "rapid inward current (bit SET). Kv channels activate with delay, "
        "producing outward current (bit RESET). Together they can implement "
        "signal inversion (NOT gate equivalent)."
    ),
    read_method="Patch clamp electrophysiology; voltage-sensitive dyes",
    write_method="Current injection; optogenetic depolarization; ionophore application",
    timescale_class="ms-scale (textbook electrophysiology, multiple phyla)",
    persistence="Volatile — requires continuous ion flux to maintain state",
    key_molecules=[
        "SCN (Nav1.x family) — voltage-gated sodium channels",
        "KCNA/KCNB (Kv1/Kv2) — voltage-gated potassium channels",
        "KCNK (K2P) — leak channels setting resting potential",
    ],
    organism_availability="Universal eukaryotic; planarian homologs confirmed",
    evidence_level="[EVIDENCED] — electrophysiology in multiple phyla",
    data_gaps=[
        "Exact Nav/Kv channel counts per planarian neoblast — UNKNOWN",
        "Single-channel conductance in planarian cells — UNKNOWN",
        "Gating kinetics (activation/inactivation τ) in planarian — UNKNOWN",
    ],
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
    read_method=(
        "DiBAC4(3) voltage-sensitive dye (equilibration ~30 s); "
        "sharp microelectrode (ms resolution); "
        "genetically encoded voltage indicators (future)"
    ),
    write_method=(
        "Pharmacological: SCH28080 (H+/K+-ATPase inhibitor), "
        "ivermectin (GluCl activator), concanamycin (V-ATPase inhibitor); "
        "Optogenetic: channelrhodopsin (future in planarians)"
    ),
    timescale_class=(
        "Write: seconds (pharmacological); "
        "Read: ms (electrode) to ~30 s (dye equilibration)"
    ),
    persistence=(
        "Semi-volatile — persists as long as pumps are active. "
        "In planarians, the target morphology pattern persists "
        "indefinitely through regeneration, suggesting a "
        "non-volatile backup exists (Oviedo et al. 2010, PMID:20160860)."
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
    evidence_level=(
        "[EVIDENCED] — DiBAC imaging of planarian Vmem patterns "
        "(Beane et al. 2011, PMID:21554866; Oviedo et al. 2010)"
    ),
    data_gaps=[
        "Exact resting Vmem of planarian neoblasts (mV) — UNKNOWN; requires patch clamp",
        "Spatial resolution of Vmem map (cell-by-cell vs tissue-average) — UNKNOWN",
        "Number of distinguishable stable Vmem states per cell — UNKNOWN; "
        "requires bistability assay",
        "Gap junction bandwidth (Hz) and SNR — UNKNOWN; requires paired-cell recording",
    ],
)

SSD_INNEXIN_CONNECTIVITY = HardwareComponent(
    name="Innexin-Gated Connectivity Pattern",
    digital_equivalent="SSD — non-volatile anatomical memory",
    biological_basis=(
        "The pattern of gap junction connectivity (which cells are "
        "electrically coupled) encodes long-term structural information. "
        "In planarians, Innexin-7 RNAi produces two-headed animals that "
        "maintain their altered phenotype through unlimited regeneration "
        "cycles, demonstrating non-volatile storage "
        "(Oviedo et al. 2010, PMID:20160860)."
    ),
    read_method=(
        "Dye coupling assay (Lucifer Yellow, neurobiotin); "
        "in situ hybridization for innexin expression"
    ),
    write_method=(
        "RNAi knockdown of specific innexin isoforms; "
        "pharmacological: octanol, heptanol (gap junction blockers)"
    ),
    timescale_class="Write: 24-48 h (RNAi); Persistence: indefinite (>18 months demonstrated)",
    persistence=(
        "Non-volatile — persists through amputation and regeneration "
        "indefinitely (>18 months in 2-headed planarians, "
        "Oviedo et al. 2010)"
    ),
    key_molecules=[
        "Innexin-7 — primary gap junction protein in planarian epidermis",
        "Innexin-13 — neoblast-enriched gap junction",
        "Beta-catenin — downstream effector of stored polarity",
        "Wnt / Notum — pathway controlled by connectivity pattern",
    ],
    organism_availability=(
        "Planarian: innexin family (>15 isoforms); not directly "
        "transferable to vertebrate connexin system without mapping"
    ),
    evidence_level=(
        "[EVIDENCED] for persistence of altered phenotype; "
        "[DATA GAP] for encoding mechanism (how connectivity encodes morphology)"
    ),
    data_gaps=[
        "Complete innexin isoform expression map per cell type — UNKNOWN",
        "Single gap junction channel conductance in planarian — UNKNOWN",
        "Information capacity of the connectivity pattern (bits) — UNKNOWN; "
        "requires systematic connectivity mapping + perturbation",
        "Mechanism by which connectivity pattern is read during regeneration — UNKNOWN",
    ],
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
            f"  Read Method: {comp.read_method}",
            f"  Write Method: {comp.write_method}",
            f"  Timescale Class: {comp.timescale_class}",
            f"  Persistence: {comp.persistence}",
            f"  Key Molecules: {', '.join(comp.key_molecules)}",
            f"  Organism Availability: {comp.organism_availability}",
            f"  Evidence Level: {comp.evidence_level}",
        ])
        if comp.data_gaps:
            lines.append("  DATA GAPS:")
            for gap in comp.data_gaps:
                lines.append(f"    - {gap}")
    return "\n".join(lines)
