#!/usr/bin/env python3
"""
Signal Propagation Speed Simulation
====================================

Measures how fast a voltage signal propagates through a planarian-like
cell network connected by gap junctions.

Key questions:
1. What is the propagation velocity (mm/s)?
2. How does signal strength decay with distance?
3. What gap junction strength is needed for reliable propagation?

This informs the I/O speed specifications for biological memory.
"""

from brian2 import *
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.resolve()

# =============================================================================
# PARAMETERS
# =============================================================================
N_CELLS = 50       # Linear chain of cells
CELL_SPACING = 10  # μm between cells

# Cell properties
CM = 10*pF
TAU = 20*ms

# Voltage states
V_REST = -60*mV    # Resting potential
V_STIM = -20*mV    # Stimulation level (strong depolarization)
V_THRESHOLD = -40*mV  # Detection threshold

# Gap junction conductance to test
GJ_VALUES = [0.05*nS, 0.1*nS, 0.2*nS, 0.5*nS, 1*nS]

# Noise
NOISE_STD = 2*mV

DURATION = 200*ms

print("=" * 70)
print("Signal Propagation Speed Simulation")
print("=" * 70)
print(f"\nChain length: {N_CELLS} cells × {CELL_SPACING} μm = {N_CELLS * CELL_SPACING} μm")
print(f"Stimulus: {V_STIM/mV:.0f} mV applied to cell 0")
print(f"Detection threshold: {V_THRESHOLD/mV:.0f} mV")


def run_propagation_test(gj_conductance):
    """Run propagation test for given gap junction conductance."""
    start_scope()

    # Passive cell model with gap junction coupling
    eqs = '''
    dv/dt = (-g_leak*(v - E_leak) + gj_current + noise) / Cm : volt
    g_leak : siemens (constant)
    E_leak : volt (constant)
    Cm : farad (constant)
    gj_current : amp
    noise : amp
    '''

    cells = NeuronGroup(N_CELLS, eqs, method='euler', dt=0.1*ms)
    cells.g_leak = CM / TAU  # ~0.5 nS
    cells.E_leak = V_REST
    cells.Cm = CM
    cells.v = V_REST

    # Stimulus to first cell
    stim = cells[:1]
    stim.v = V_STIM

    # Build neighbor connections (linear chain)
    gj_g = float(gj_conductance / nS)

    @network_operation(dt=0.5*ms)
    def update_gj():
        v_arr = np.array(cells.v / mV)
        gj_curr = np.zeros(N_CELLS)

        # Chain connections
        for i in range(N_CELLS - 1):
            # Current from i to i+1
            i_gj = gj_g * (v_arr[i] - v_arr[i+1])  # nA
            gj_curr[i] -= i_gj
            gj_curr[i+1] += i_gj

        cells.gj_current = gj_curr * nA

        # Noise
        cells.noise = (NOISE_STD / TAU) * CM * np.random.randn(N_CELLS)

        # Maintain stimulus on first cell
        cells.v[0] = V_STIM

    # Monitor
    mon = StateMonitor(cells, 'v', record=True, dt=0.5*ms)

    run(DURATION, report=None)

    return np.array(mon.t / ms), np.array(mon.v / mV)


def analyze_propagation(t, v_array, gj_conductance):
    """Analyze signal propagation."""
    threshold_mv = float(V_THRESHOLD / mV)

    # Find time when each cell crosses threshold
    arrival_times = []
    for i in range(N_CELLS):
        above = np.where(v_array[i] > threshold_mv)[0]
        if len(above) > 0:
            arrival_times.append(t[above[0]])
        else:
            arrival_times.append(None)

    # Calculate velocity from arrival times
    distances = np.arange(N_CELLS) * CELL_SPACING  # μm

    # Find cells that received signal
    valid = [(d, at) for d, at in zip(distances, arrival_times) if at is not None]

    if len(valid) > 2:
        d_arr = np.array([v[0] for v in valid])
        t_arr = np.array([v[1] for v in valid])

        # Linear fit for velocity
        # d = v * t → slope = velocity
        if len(t_arr) > 1 and t_arr[-1] > t_arr[0]:
            velocity = (d_arr[-1] - d_arr[1]) / (t_arr[-1] - t_arr[1])  # μm/ms
            velocity_mm_s = velocity * 1000 / 1000  # Convert to mm/s
        else:
            velocity_mm_s = None
    else:
        velocity_mm_s = None

    # Signal attenuation
    final_voltages = v_array[:, -1]
    attenuation = (final_voltages[0] - final_voltages[-1]) / final_voltages[0] if final_voltages[0] != 0 else 0

    # Propagation distance (how far signal traveled)
    reached = [i for i, at in enumerate(arrival_times) if at is not None]
    max_distance = max(reached) * CELL_SPACING if reached else 0

    return {
        'arrival_times': arrival_times,
        'velocity_mm_s': velocity_mm_s,
        'max_distance_um': max_distance,
        'attenuation': attenuation,
        'final_voltages': final_voltages
    }


# =============================================================================
# RUN EXPERIMENTS
# =============================================================================
print("\nRunning propagation tests for different gap junction strengths...")

results = {}
for gj in GJ_VALUES:
    print(f"  Testing Gj = {gj/nS:.2f} nS...")
    t, v_array = run_propagation_test(gj)
    analysis = analyze_propagation(t, v_array, gj)
    analysis['t'] = t
    analysis['v_array'] = v_array
    results[float(gj/nS)] = analysis

# =============================================================================
# VISUALIZATION
# =============================================================================
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# 1-5: Voltage traces for each Gj
for idx, gj in enumerate(GJ_VALUES):
    ax = axes.flatten()[idx]
    gj_val = float(gj/nS)
    res = results[gj_val]

    # Plot every 5th cell for clarity
    for i in range(0, N_CELLS, 5):
        color = plt.cm.viridis(i / N_CELLS)
        ax.plot(res['t'], res['v_array'][i], color=color, linewidth=1,
                label=f'Cell {i}' if i % 10 == 0 else None)

    ax.axhline(y=float(V_THRESHOLD/mV), color='red', linestyle='--', alpha=0.5)
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Voltage (mV)')

    vel = res['velocity_mm_s']
    vel_str = f"{vel:.1f} mm/s" if vel else "No propagation"
    ax.set_title(f'Gj = {gj_val:.2f} nS\nVelocity: {vel_str}')
    ax.set_ylim(-70, -10)
    ax.grid(True, alpha=0.3)
    if idx == 0:
        ax.legend(loc='right', fontsize=8)

# 6: Summary - velocity vs Gj
ax = axes.flatten()[5]
gj_vals = [float(gj/nS) for gj in GJ_VALUES]
velocities = [results[g]['velocity_mm_s'] for g in gj_vals]
# Replace None with 0 for plotting
velocities_plot = [v if v is not None else 0 for v in velocities]

ax.bar(range(len(gj_vals)), velocities_plot, color='steelblue')
ax.set_xticks(range(len(gj_vals)))
ax.set_xticklabels([f'{g:.2f}' for g in gj_vals])
ax.set_xlabel('Gap Junction Conductance (nS)')
ax.set_ylabel('Propagation Velocity (mm/s)')
ax.set_title('Signal Speed vs Gap Junction Strength')
ax.grid(True, alpha=0.3, axis='y')

plt.suptitle('Signal Propagation Through Cell Chain\n(Stimulus at cell 0, measuring spread)',
             fontsize=14, fontweight='bold')
plt.tight_layout()

output_path = SCRIPT_DIR / 'signal_propagation.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"\nPlot saved to: {output_path}")

# =============================================================================
# RESULTS SUMMARY
# =============================================================================
print("\n" + "=" * 70)
print("RESULTS SUMMARY")
print("=" * 70)

print("\n  Signal Propagation Speed:")
print("  " + "-" * 50)
print(f"  {'Gj (nS)':<10} {'Velocity (mm/s)':<18} {'Max Distance (μm)':<20}")
print("  " + "-" * 50)

for gj in GJ_VALUES:
    gj_val = float(gj/nS)
    res = results[gj_val]
    vel = f"{res['velocity_mm_s']:.2f}" if res['velocity_mm_s'] else "N/A"
    print(f"  {gj_val:<10.2f} {vel:<18} {res['max_distance_um']:<20.0f}")

print("\n  KEY FINDINGS:")
# Find optimal Gj
valid_results = [(g, r['velocity_mm_s']) for g, r in results.items() if r['velocity_mm_s']]
if valid_results:
    best_gj, best_vel = max(valid_results, key=lambda x: x[1])
    print(f"  • Fastest propagation: {best_vel:.1f} mm/s at Gj = {best_gj:.2f} nS")

    # Biological context
    planarian_size = 10  # mm typical
    transit_time = planarian_size / best_vel * 1000 if best_vel > 0 else float('inf')
    print(f"  • Time to cross 10mm planarian: {transit_time:.0f} ms")

print("\n  BIOLOGICAL IMPLICATIONS:")
print("  • Higher Gj = faster signaling but less isolation")
print("  • Need to GATE gap junctions for read/write operations")
print("  • Propagation speed affects I/O bandwidth")

plt.show()

print("\n" + "=" * 70)
print("Simulation complete!")
print("=" * 70)
