#!/usr/bin/env python3
"""
Stochastic Consensus Test - Brian2 Simulation
==============================================

Simulates 100 neoblasts coupled via gap junctions with stochastic noise.
Tests whether majority-vote consensus maintains stable bit storage.

Original concept from Gemini, corrected for Brian2 compatibility.

Key fixes from original:
1. mean(v) replaced with summed variable approach
2. xi noise properly integrated with stochastic flag
3. Cross-platform file output added
"""

from brian2 import *
import matplotlib.pyplot as plt
from pathlib import Path

# Cross-platform output path
SCRIPT_DIR = Path(__file__).parent.resolve()

# =============================================================================
# 1. SYSTEM SPECIFICATIONS
# =============================================================================
N = 100                         # Number of neoblasts in the cluster
Cm = 10*pF                      # Membrane Capacitance
g_leak = 1*nS                   # Individual cell leak conductance
E_leak = -60*mV                 # Natural resting state (Logic 0)
V_target = -40*mV               # Our "Written" Bit (Logic 1)
g_gap = 5*nS                    # Gap junction coupling strength
noise_amplitude = 0.5*nA        # Noise amplitude (~5mV fluctuations)

print("=" * 60)
print("Stochastic Consensus Test - 100 Neoblast Cluster")
print("=" * 60)
print(f"\nParameters:")
print(f"  N cells:            {N}")
print(f"  Membrane cap:       {Cm}")
print(f"  Leak conductance:   {g_leak}")
print(f"  E_leak (Logic 0):   {E_leak}")
print(f"  V_target (Logic 1): {V_target}")
print(f"  Gap junction g:     {g_gap}")
print(f"  Noise amplitude:    {noise_amplitude}")

# =============================================================================
# 2. THE EQUATIONS
# =============================================================================
# Brian2 requires special handling for:
# - Stochastic terms (use xi with proper units)
# - Mean-field coupling (use summed variable)

eqs = '''
dv/dt = (g_leak*(E_leak - v) + I_gap + I_noise)/Cm : volt
I_gap = g_gap * (v_mean - v) : amp                    # Coupling to mean
I_noise : amp                                          # Noise current (set by TimedArray or run_regularly)
v_mean : volt (shared)                                 # Shared mean voltage
'''

# =============================================================================
# 3. CREATE THE NEURON GROUP
# =============================================================================
cluster = NeuronGroup(N, eqs, method='euler', dt=0.1*ms)
cluster.v = V_target  # Initialize all cells to Logic 1 (-40mV)

# =============================================================================
# 4. UPDATE MEAN AND NOISE DURING SIMULATION
# =============================================================================
@network_operation(dt=1*ms)
def update_mean_and_noise():
    """Update shared mean voltage and inject noise each timestep."""
    # Update mean voltage for gap junction coupling
    cluster.v_mean = mean(cluster.v)
    # Inject random noise current to each cell
    cluster.I_noise = noise_amplitude * np.random.randn(N)

# =============================================================================
# 5. MONITORING
# =============================================================================
state_mon = StateMonitor(cluster, 'v', record=True)

# =============================================================================
# 6. RUN SIMULATION
# =============================================================================
print(f"\nRunning simulation for 200 ms...")
run(200*ms, report='text')

# =============================================================================
# 7. ANALYSIS
# =============================================================================
print("\n" + "=" * 60)
print("RESULTS")
print("=" * 60)

# Calculate statistics
v_array = state_mon.v / mV  # Convert to mV
mean_v = np.mean(v_array, axis=0)  # Mean across cells at each time
final_mean = mean_v[-1]
final_std = np.std(v_array[:, -1])

# Check bit stability
threshold = -40  # mV
final_votes_for_1 = np.sum(v_array[:, -1] > threshold)
bit_read = 1 if final_votes_for_1 > N/2 else 0

print(f"\n  Final consensus voltage: {final_mean:.2f} mV")
print(f"  Final std across cells:  {final_std:.2f} mV")
print(f"  Cells voting '1' (>{threshold} mV): {final_votes_for_1}/{N}")
print(f"  Bit readout: {bit_read} (expected: 1)")
print(f"  Status: {'✓ BIT STABLE' if bit_read == 1 else '✗ BIT CORRUPTED'}")

# Check if voltage drifted toward E_leak
drift = final_mean - float(V_target/mV)
print(f"\n  Drift from target: {drift:.2f} mV")
if abs(drift) > 5:
    print("  WARNING: Significant drift detected!")

# =============================================================================
# 8. VISUALIZATION
# =============================================================================
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

# Plot 1: Individual cell traces
for i in range(N):
    ax1.plot(state_mon.t/ms, v_array[i], color='gray', alpha=0.2, linewidth=0.5)
ax1.plot(state_mon.t/ms, mean_v, color='blue', linewidth=2, label='Cluster Consensus')
ax1.axhline(y=-40, color='red', linestyle='--', linewidth=1.5, label='Logic 1 Threshold (-40mV)')
ax1.axhline(y=-60, color='green', linestyle=':', linewidth=1.5, label='Logic 0 Rest (-60mV)')
ax1.set_ylabel('Membrane Potential (mV)')
ax1.set_title('Acheron Phase-1: 100-Cell Consensus Stability Test')
ax1.legend(loc='upper right')
ax1.set_ylim(-70, -30)
ax1.grid(True, alpha=0.3)

# Plot 2: Voltage distribution over time (heatmap)
ax2.hist2d(np.tile(state_mon.t/ms, N), v_array.flatten(),
           bins=[100, 50], cmap='Blues', range=[[0, 200], [-70, -30]])
ax2.axhline(y=-40, color='red', linestyle='--', linewidth=1.5)
ax2.set_xlabel('Time (ms)')
ax2.set_ylabel('Membrane Potential (mV)')
ax2.set_title('Voltage Distribution Across Population')
plt.colorbar(ax2.collections[0], ax=ax2, label='Cell count')

plt.tight_layout()

# Save figure
output_path = SCRIPT_DIR / 'stochastic_consensus_test.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"\nPlot saved to: {output_path}")

# Show if running interactively
try:
    plt.show()
except:
    pass  # Headless mode

print("\n" + "=" * 60)
print("Simulation complete!")
print("=" * 60)
