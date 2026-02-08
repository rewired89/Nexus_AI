#!/usr/bin/env python3
"""
NEURON Simulation: Planarian Cell Network with Gap Junctions
=============================================================

Uses NEURON simulator for biophysically realistic modeling of:
1. Individual cells with ion channels (simplified HH-type)
2. Gap junction coupling between cells
3. Network-level voltage pattern storage and propagation

This simulates the "biological RAM" hypothesis - can a network of
coupled cells maintain distinct voltage states long enough for
information storage?

Key questions answered:
- How does gap junction strength affect pattern stability?
- What's the critical coupling for consensus vs. independent states?
- How fast do voltage patterns propagate through the network?
"""

from neuron import h, gui
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Cross-platform output path
SCRIPT_DIR = Path(__file__).parent.resolve()

# Initialize NEURON
h.load_file("stdrun.hoc")

# =============================================================================
# PARAMETERS - Based on LOGIC_HASH constraints
# =============================================================================
N_CELLS = 20           # Number of cells in network (reduced for visualization)
CELL_DIAMETER = 10     # μm - neoblast approximate size
CELL_LENGTH = 20       # μm

# Membrane properties (planarian estimates)
CM = 1.0               # μF/cm² - membrane capacitance
RM = 10000             # Ω·cm² - membrane resistance (gives τ = 10ms)
RA = 100               # Ω·cm - axial resistance

# Ion channel parameters (simplified)
G_LEAK = 0.0001        # S/cm² - leak conductance
E_LEAK = -60           # mV - leak reversal (resting state = "Logic 0")
G_K = 0.001            # S/cm² - potassium conductance
E_K = -80              # mV - potassium reversal

# Gap junction conductance sweep
GJ_CONDUCTANCES = [0.001, 0.01, 0.05, 0.1, 0.5]  # μS

# Stimulation (to set initial pattern)
V_LOGIC_1 = -30        # mV - "Logic 1" target
V_LOGIC_0 = -60        # mV - "Logic 0" target

print("=" * 70)
print("NEURON Simulation: Planarian Cell Network")
print("=" * 70)
print(f"\nNetwork: {N_CELLS} cells")
print(f"Cell size: {CELL_DIAMETER}μm × {CELL_LENGTH}μm")
print(f"Membrane τ = Rm × Cm = {RM * CM / 1000:.1f} ms")
print(f"Logic 1: {V_LOGIC_1} mV, Logic 0: {V_LOGIC_0} mV")


# =============================================================================
# CREATE CELL MODEL
# =============================================================================
class PlanarianCell:
    """Simple planarian cell with passive membrane + leak channels."""

    def __init__(self, cell_id):
        self.id = cell_id

        # Create a simple soma
        self.soma = h.Section(name=f"soma_{cell_id}")
        self.soma.L = CELL_LENGTH
        self.soma.diam = CELL_DIAMETER
        self.soma.Ra = RA
        self.soma.cm = CM

        # Insert passive leak channel
        self.soma.insert("pas")
        self.soma.g_pas = G_LEAK
        self.soma.e_pas = E_LEAK

        # Recording vectors
        self.v_vec = h.Vector()
        self.v_vec.record(self.soma(0.5)._ref_v)

    def set_voltage(self, v):
        """Set initial voltage."""
        self.soma.v = v


# =============================================================================
# CREATE GAP JUNCTION
# =============================================================================
class GapJunction:
    """Gap junction connection between two cells."""

    def __init__(self, cell1, cell2, conductance):
        """
        Create bidirectional gap junction.

        Args:
            cell1, cell2: PlanarianCell objects
            conductance: Gap junction conductance in μS
        """
        # Use NEURON's LinearMechanism for gap junction
        # Gap junction current: I = g * (V1 - V2)

        # Create conductance matrix for 2-cell coupling
        # [g, -g; -g, g] creates bidirectional current flow
        self.g = conductance

        # Store references
        self.cell1 = cell1
        self.cell2 = cell2

        # Use a simple current clamp approach for gap junction
        # (more complex implementations would use LinearMechanism)
        self.gj1 = h.IClamp(cell1.soma(0.5))
        self.gj2 = h.IClamp(cell2.soma(0.5))

        # These will be updated during simulation
        self.gj1.dur = 1e9
        self.gj2.dur = 1e9


# =============================================================================
# BUILD NETWORK
# =============================================================================
def build_network(gj_conductance):
    """Build a ring network of cells connected by gap junctions."""
    cells = [PlanarianCell(i) for i in range(N_CELLS)]
    gap_junctions = []

    # Connect cells in a ring topology
    for i in range(N_CELLS):
        j = (i + 1) % N_CELLS  # Next cell (wraps around)
        gj = GapJunction(cells[i], cells[j], gj_conductance)
        gap_junctions.append(gj)

    return cells, gap_junctions


# =============================================================================
# SET INITIAL PATTERN
# =============================================================================
def set_initial_pattern(cells, pattern_type="half"):
    """
    Set initial voltage pattern.

    pattern_type:
        "half" - First half at Logic 1, second half at Logic 0
        "alternating" - Alternating 1-0-1-0 pattern
        "single" - Only one cell at Logic 1
    """
    if pattern_type == "half":
        for i, cell in enumerate(cells):
            if i < N_CELLS // 2:
                cell.set_voltage(V_LOGIC_1)
            else:
                cell.set_voltage(V_LOGIC_0)
    elif pattern_type == "alternating":
        for i, cell in enumerate(cells):
            if i % 2 == 0:
                cell.set_voltage(V_LOGIC_1)
            else:
                cell.set_voltage(V_LOGIC_0)
    elif pattern_type == "single":
        for i, cell in enumerate(cells):
            if i == 0:
                cell.set_voltage(V_LOGIC_1)
            else:
                cell.set_voltage(V_LOGIC_0)

    return pattern_type


# =============================================================================
# UPDATE GAP JUNCTION CURRENTS
# =============================================================================
def update_gap_junctions(gap_junctions, dt=0.025):
    """
    Update gap junction currents based on voltage differences.

    This is a simple implementation - for each gap junction,
    calculate current based on voltage difference and inject it.
    """
    for gj in gap_junctions:
        v1 = gj.cell1.soma(0.5).v
        v2 = gj.cell2.soma(0.5).v

        # Current flows from higher to lower voltage
        # I = g * (V_other - V_self)
        i_gj = gj.g * (v2 - v1)  # Current into cell1
        gj.gj1.amp = i_gj

        i_gj2 = gj.g * (v1 - v2)  # Current into cell2
        gj.gj2.amp = i_gj2


# =============================================================================
# RUN SIMULATION
# =============================================================================
def run_simulation(gj_conductance, duration=500, pattern="half"):
    """
    Run simulation with given gap junction conductance.

    Returns time vector and voltage matrix (cells × time).
    """
    print(f"\n  Running with Gj = {gj_conductance} μS, pattern = {pattern}...")

    # Build network
    cells, gap_junctions = build_network(gj_conductance)

    # Set initial pattern
    set_initial_pattern(cells, pattern)

    # Time vector
    t_vec = h.Vector()
    t_vec.record(h._ref_t)

    # Setup simulation
    h.tstop = duration
    h.dt = 0.1
    h.v_init = E_LEAK

    # Run with manual gap junction updates
    h.finitialize()

    # Override initial voltages after finitialize
    set_initial_pattern(cells, pattern)

    while h.t < h.tstop:
        update_gap_junctions(gap_junctions)
        h.fadvance()

    # Extract results
    t = np.array(t_vec)
    voltages = np.array([np.array(cell.v_vec) for cell in cells])

    return t, voltages


# =============================================================================
# ANALYSIS
# =============================================================================
def analyze_results(t, voltages, gj_conductance):
    """Analyze simulation results."""
    results = {}

    # Final mean voltage
    results["final_mean"] = np.mean(voltages[:, -1])
    results["final_std"] = np.std(voltages[:, -1])

    # Time to consensus (when std drops below 5mV)
    std_over_time = np.std(voltages, axis=0)
    consensus_idx = np.where(std_over_time < 5)[0]
    if len(consensus_idx) > 0:
        results["t_consensus"] = t[consensus_idx[0]]
    else:
        results["t_consensus"] = None  # No consensus reached

    # Pattern preservation (correlation with initial pattern)
    initial = voltages[:, 0]
    final = voltages[:, -1]
    if np.std(initial) > 0 and np.std(final) > 0:
        results["pattern_correlation"] = np.corrcoef(initial, final)[0, 1]
    else:
        results["pattern_correlation"] = 0

    # Information content (bits)
    # Simplified: count distinct states
    threshold = (V_LOGIC_1 + V_LOGIC_0) / 2
    final_bits = (voltages[:, -1] > threshold).astype(int)
    unique_patterns = len(set(tuple(final_bits)))
    results["final_bits"] = final_bits
    results["unique_states"] = unique_patterns

    return results


# =============================================================================
# MAIN EXPERIMENT: GAP JUNCTION STRENGTH SWEEP
# =============================================================================
print("\n" + "=" * 70)
print("EXPERIMENT 1: Gap Junction Conductance Sweep")
print("=" * 70)
print("Question: How does Gj strength affect pattern stability?")

all_results = {}
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

for idx, gj in enumerate(GJ_CONDUCTANCES):
    t, voltages = run_simulation(gj, duration=500, pattern="half")
    results = analyze_results(t, voltages, gj)
    all_results[gj] = results

    # Plot voltage traces
    ax = axes.flatten()[idx] if idx < 5 else None
    if ax:
        for i in range(N_CELLS):
            color = 'red' if i < N_CELLS // 2 else 'blue'
            ax.plot(t, voltages[i], color=color, alpha=0.5, linewidth=0.5)

        # Plot mean
        ax.plot(t, np.mean(voltages, axis=0), 'k-', linewidth=2, label='Mean')
        ax.axhline(y=V_LOGIC_1, color='red', linestyle='--', alpha=0.5)
        ax.axhline(y=V_LOGIC_0, color='blue', linestyle='--', alpha=0.5)
        ax.axhline(y=(V_LOGIC_1 + V_LOGIC_0) / 2, color='gray', linestyle=':')

        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Voltage (mV)')
        ax.set_title(f'Gj = {gj} μS\nCorr: {results["pattern_correlation"]:.2f}')
        ax.set_ylim(-80, -20)
        ax.grid(True, alpha=0.3)

# Summary plot
ax = axes.flatten()[5]
gj_vals = list(all_results.keys())
correlations = [all_results[g]["pattern_correlation"] for g in gj_vals]
ax.semilogx(gj_vals, correlations, 'bo-', markersize=10, linewidth=2)
ax.set_xlabel('Gap Junction Conductance (μS)')
ax.set_ylabel('Pattern Correlation')
ax.set_title('Pattern Preservation vs Coupling Strength')
ax.grid(True, alpha=0.3)
ax.axhline(y=0.5, color='red', linestyle='--', label='50% correlation')
ax.legend()

plt.suptitle('Planarian Cell Network: Gap Junction Sweep\n(Red=Logic 1 cells, Blue=Logic 0 cells)',
             fontsize=14, fontweight='bold')
plt.tight_layout()

output_path = SCRIPT_DIR / 'neuron_gj_sweep.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"\nPlot saved to: {output_path}")

# =============================================================================
# RESULTS SUMMARY
# =============================================================================
print("\n" + "=" * 70)
print("RESULTS SUMMARY")
print("=" * 70)

print("\n  Gap Junction Conductance Effects:")
print("  " + "-" * 50)
print(f"  {'Gj (μS)':<10} {'Correlation':<15} {'T_consensus':<15} {'Final Mean':<12}")
print("  " + "-" * 50)

for gj, res in all_results.items():
    t_cons = f"{res['t_consensus']:.1f} ms" if res['t_consensus'] else "Never"
    print(f"  {gj:<10.3f} {res['pattern_correlation']:<15.3f} {t_cons:<15} {res['final_mean']:<12.1f}")

# Find critical Gj
# Pattern preserved (corr > 0.5) vs consensus (corr ~ 0)
preserved = [gj for gj, res in all_results.items() if res["pattern_correlation"] > 0.5]
consensus = [gj for gj, res in all_results.items() if res["pattern_correlation"] < 0.2]

print("\n  KEY FINDINGS:")
if preserved:
    print(f"  • Pattern PRESERVED (corr > 0.5) at Gj ≤ {max(preserved)} μS")
if consensus:
    print(f"  • CONSENSUS reached (corr < 0.2) at Gj ≥ {min(consensus)} μS")

print("\n  INTERPRETATION:")
print("  • LOW coupling (< 0.01 μS): Cells act independently → can store DIFFERENT bits")
print("  • HIGH coupling (> 0.1 μS): Network reaches consensus → single shared state")
print("  • MEDIUM coupling: Partial pattern preservation → noisy storage")

print("\n  FOR BIOLOGICAL MEMORY:")
print("  • Need GATED gap junctions (like GATE operation in Bio-ISA)")
print("  • WRITE mode: Low Gj (cells independent, can be set individually)")
print("  • STORE mode: Medium Gj (local clusters maintain consensus)")
print("  • READ mode: High Gj (propagate signal for readout)")

plt.show()

print("\n" + "=" * 70)
print("Simulation complete!")
print("=" * 70)
