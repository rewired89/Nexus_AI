#!/usr/bin/env python3
"""
Bistable Memory Test - Brian2 Simulation
=========================================

Tests whether a bistable ion channel model can maintain stable bit storage.
This addresses the issue found in stochastic_consensus_test.py where
passive leak conductance caused voltage drift.

Key insight: Biological memory requires BISTABILITY, not just gap junctions.
Here we model a simplified bistable system using a voltage-gated conductance
that creates two stable states.

Two stable states:
  - Logic 0: V ≈ -60 mV (hyperpolarized, resting)
  - Logic 1: V ≈ -30 mV (depolarized, active)
  - Unstable threshold: -45 mV
"""

from brian2 import *
import matplotlib.pyplot as plt
from pathlib import Path

# Cross-platform output path
SCRIPT_DIR = Path(__file__).parent.resolve()

# =============================================================================
# 1. PARAMETERS
# =============================================================================
N = 100                         # Number of neoblasts
Cm = 10*pF                      # Membrane Capacitance

# Bistable dynamics parameters
V_low = -60*mV                  # Stable state 0 (rest)
V_high = -30*mV                 # Stable state 1 (active)
V_threshold = -45*mV            # Unstable threshold
tau = 20*ms                     # Time constant

# Gap junction coupling
g_gap = 2*nS                    # Coupling strength

# Noise
noise_std = 3*mV                # Voltage noise standard deviation

print("=" * 60)
print("Bistable Memory Test - 100 Neoblast Cluster")
print("=" * 60)
print(f"\nParameters:")
print(f"  N cells:           {N}")
print(f"  V_low (Logic 0):   {V_low}")
print(f"  V_high (Logic 1):  {V_high}")
print(f"  V_threshold:       {V_threshold}")
print(f"  Time constant:     {tau}")
print(f"  Gap junction g:    {g_gap}")
print(f"  Noise std:         {noise_std}")

# =============================================================================
# 2. BISTABLE EQUATIONS
# =============================================================================
# The key is the bistable term: pushes V toward V_low or V_high depending
# on which side of V_threshold it's on.

eqs = '''
dv/dt = (bistable_drive + coupling_current + noise_current)/Cm : volt
# Bistable drive: cubic-like function with two stable fixed points
bistable_drive = -g_bistable * (v - V_low) * (v - V_threshold) * (v - V_high) / (20*mV**2) : amp
g_bistable : siemens (constant)
# Gap junction coupling to mean field
coupling_current = g_gap_cell * (v_mean - v) : amp
g_gap_cell : siemens (constant)
# Noise
noise_current : amp
# Shared mean voltage
v_mean : volt (shared)
'''

# =============================================================================
# 3. CREATE NEURON GROUP
# =============================================================================
cluster = NeuronGroup(N, eqs, method='euler', dt=0.1*ms)

# Set parameters
cluster.g_bistable = 1*nS
cluster.g_gap_cell = g_gap

# Initialize: Set all cells to Logic 1 (V_high = -30mV)
cluster.v = V_high
print(f"\nInitial state: All cells at Logic 1 ({V_high})")

# =============================================================================
# 4. UPDATE FUNCTIONS
# =============================================================================
@network_operation(dt=0.5*ms)
def update_dynamics():
    """Update mean voltage and inject noise."""
    cluster.v_mean = mean(cluster.v)
    # Add Gaussian noise
    cluster.noise_current = (noise_std / tau) * Cm * np.random.randn(N)

# =============================================================================
# 5. MONITORING
# =============================================================================
state_mon = StateMonitor(cluster, 'v', record=True)

# =============================================================================
# 6. RUN SIMULATION
# =============================================================================
duration = 500*ms
print(f"\nRunning simulation for {duration}...")
run(duration, report='text')

# =============================================================================
# 7. ANALYSIS
# =============================================================================
print("\n" + "=" * 60)
print("RESULTS")
print("=" * 60)

v_array = state_mon.v / mV
mean_v = np.mean(v_array, axis=0)
final_mean = mean_v[-1]
final_std = np.std(v_array[:, -1])

# Bit readout at threshold -45mV
threshold_mv = float(V_threshold/mV)
final_votes_for_1 = np.sum(v_array[:, -1] > threshold_mv)
bit_read = 1 if final_votes_for_1 > N/2 else 0

print(f"\n  Initial voltage:         {float(V_high/mV):.1f} mV (Logic 1)")
print(f"  Final consensus voltage: {final_mean:.2f} mV")
print(f"  Final std across cells:  {final_std:.2f} mV")
print(f"  Cells voting '1' (>{threshold_mv} mV): {final_votes_for_1}/{N}")
print(f"  Bit readout: {bit_read}")
print(f"  Status: {'✓ BIT STABLE' if bit_read == 1 else '✗ BIT CORRUPTED'}")

# Stability analysis
drift = final_mean - float(V_high/mV)
print(f"\n  Drift from Logic 1 target: {drift:.2f} mV")

# Time spent near each attractor
near_high = np.mean(v_array > threshold_mv)
near_low = np.mean(v_array < threshold_mv)
print(f"  Time near Logic 1: {near_high*100:.1f}%")
print(f"  Time near Logic 0: {near_low*100:.1f}%")

# =============================================================================
# 8. VISUALIZATION
# =============================================================================
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# Plot 1: Individual traces
ax1 = axes[0, 0]
for i in range(min(20, N)):  # Plot subset for clarity
    ax1.plot(state_mon.t/ms, v_array[i], color='gray', alpha=0.3, linewidth=0.5)
ax1.plot(state_mon.t/ms, mean_v, color='blue', linewidth=2, label='Consensus')
ax1.axhline(y=float(V_high/mV), color='red', linestyle='--', label=f'Logic 1 ({V_high})')
ax1.axhline(y=float(V_low/mV), color='green', linestyle='--', label=f'Logic 0 ({V_low})')
ax1.axhline(y=float(V_threshold/mV), color='orange', linestyle=':', label=f'Threshold ({V_threshold})')
ax1.set_ylabel('Membrane Potential (mV)')
ax1.set_xlabel('Time (ms)')
ax1.set_title('Bistable Memory: Voltage Traces')
ax1.legend(loc='best', fontsize=8)
ax1.grid(True, alpha=0.3)

# Plot 2: Final voltage distribution
ax2 = axes[0, 1]
ax2.hist(v_array[:, -1], bins=30, color='steelblue', edgecolor='black', alpha=0.7)
ax2.axvline(x=float(V_high/mV), color='red', linestyle='--', label='Logic 1')
ax2.axvline(x=float(V_low/mV), color='green', linestyle='--', label='Logic 0')
ax2.axvline(x=float(V_threshold/mV), color='orange', linestyle=':', label='Threshold')
ax2.set_xlabel('Final Voltage (mV)')
ax2.set_ylabel('Number of Cells')
ax2.set_title('Final Voltage Distribution')
ax2.legend()

# Plot 3: Consensus over time
ax3 = axes[1, 0]
ax3.plot(state_mon.t/ms, mean_v, color='blue', linewidth=1.5)
ax3.fill_between(state_mon.t/ms,
                  np.percentile(v_array, 25, axis=0),
                  np.percentile(v_array, 75, axis=0),
                  color='blue', alpha=0.2, label='25-75 percentile')
ax3.axhline(y=float(V_high/mV), color='red', linestyle='--')
ax3.axhline(y=float(V_threshold/mV), color='orange', linestyle=':')
ax3.set_xlabel('Time (ms)')
ax3.set_ylabel('Consensus Voltage (mV)')
ax3.set_title('Consensus Stability Over Time')
ax3.grid(True, alpha=0.3)

# Plot 4: Phase portrait (bistable dynamics visualization)
ax4 = axes[1, 1]
v_range = np.linspace(-70, -20, 100)
# dv/dt from bistable term only (simplified)
g_bi = 1  # nS
dv_dt = -g_bi * (v_range - (-60)) * (v_range - (-45)) * (v_range - (-30)) / (20**2)
ax4.plot(v_range, dv_dt, 'b-', linewidth=2)
ax4.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
ax4.axvline(x=-60, color='green', linestyle='--', label='Stable (Logic 0)')
ax4.axvline(x=-30, color='red', linestyle='--', label='Stable (Logic 1)')
ax4.axvline(x=-45, color='orange', linestyle=':', label='Unstable (Threshold)')
ax4.set_xlabel('Voltage (mV)')
ax4.set_ylabel('dV/dt (a.u.)')
ax4.set_title('Bistable Phase Portrait')
ax4.legend(fontsize=8)
ax4.grid(True, alpha=0.3)
ax4.set_xlim(-70, -20)

plt.suptitle(f'Bistable Memory Test: N={N} cells, Bit Read = {bit_read}', fontsize=14, fontweight='bold')
plt.tight_layout()

# Save
output_path = SCRIPT_DIR / 'bistable_memory_test.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"\nPlot saved to: {output_path}")

try:
    plt.show()
except:
    pass

print("\n" + "=" * 60)
print("Simulation complete!")
print("=" * 60)
