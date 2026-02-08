#!/usr/bin/env python3
"""
Acheron Phase-0: Neoblast Vmem Stability Test

Simulates a single planarian neoblast cell relaxing from a depolarized
state (-20 mV) to resting potential (-60 mV).

This tests the basic T_hold characteristic: how quickly does the membrane
potential stabilize after perturbation?

Based on Nexus v1 Hardware Baselines:
- Dugesia japonica resting Vmem: -20 to -60 mV [MEASURED]
"""

from brian2 import *
import matplotlib.pyplot as plt
from pathlib import Path

# Get the directory where this script is located
SCRIPT_DIR = Path(__file__).parent.resolve()

# --- 1. PHYSICAL PARAMETERS (Nexus v1 Specifications) ---
area = 20000 * umetre**2
Cm = 1 * ufarad/cm**2 * area  # Membrane Capacitance
g_leak = 5 * nsiemens         # Leak Conductance (Low-Power Mode)
E_leak = -60 * mvolt          # Target Resting Potential (Nexus Spec)

# --- 2. THE BIOLOGICAL EQUATION ---
# This is the "Code" for the cell's voltage behavior
# dv/dt = (g_leak * (E_leak - v) + I_ext) / Cm
# This is a simple RC circuit: voltage decays toward E_leak
eqs = '''
dv/dt = (g_leak * (E_leak - v) + I_ext) / Cm : volt
I_ext : amp  # External stimulation (Our "Write" command)
'''

# --- 3. INITIALIZE THE CELL ---
cell = NeuronGroup(1, eqs, method='exact')
cell.v = -20 * mvolt  # Start at a depolarized state (Nexus Spec)

# --- 4. THE EXPERIMENT (The "Write" Operation) ---
# Monitor the voltage over 100ms
state_mon = StateMonitor(cell, 'v', record=True)
run(100 * msecond)

# --- 5. CALCULATE KEY METRICS ---
v_final = state_mon.v[0][-1] / mV
v_initial = -20  # mV
v_target = -60   # mV

# Time constant tau = Cm / g_leak
tau = (Cm / g_leak) / ms
print(f"\n=== Acheron Phase-0 Simulation Results ===")
print(f"Initial Vmem: {v_initial} mV")
print(f"Final Vmem: {v_final:.2f} mV")
print(f"Target Vmem: {v_target} mV")
print(f"Time constant (tau): {tau:.2f} ms")
print(f"Time to 63% relaxation: {tau:.2f} ms")
print(f"Time to 95% relaxation: {3*tau:.2f} ms")
print(f"Time to 99% relaxation: {5*tau:.2f} ms")

# --- 6. VISUALIZE THE RESULT ---
plt.figure(figsize=(10, 6))
plt.plot(state_mon.t/ms, state_mon.v[0]/mV, 'b-', linewidth=2, label='Vmem')
plt.xlabel('Time (ms)', fontsize=12)
plt.ylabel('Membrane Potential (mV)', fontsize=12)
plt.title('Acheron Phase-0: Neoblast Vmem Stability Test', fontsize=14)
plt.axhline(y=-60, color='r', linestyle='--', linewidth=1.5, label='Target Baseline (-60 mV)')
plt.axhline(y=-20, color='g', linestyle=':', linewidth=1.5, label='Initial State (-20 mV)')

# Mark tau
plt.axvline(x=tau, color='orange', linestyle='-.', alpha=0.7, label=f'τ = {tau:.1f} ms')

plt.legend(loc='right')
plt.grid(True, alpha=0.3)
plt.ylim(-70, -10)
plt.xlim(0, 100)

# Save the figure (cross-platform path)
output_path = SCRIPT_DIR / 'neoblast_vmem_stability.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"\nPlot saved to: {output_path}")
plt.show()
