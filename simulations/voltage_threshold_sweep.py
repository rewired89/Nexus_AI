#!/usr/bin/env python3
"""
Voltage Threshold Sweep — Find the Head/Tail Switch
====================================================

WHAT THIS SIMULATION DOES (Plain English):
    We're looking for the exact voltage that tells a worm cell
    "grow a head" vs "grow a tail". Think of it like finding the
    exact temperature on a thermostat where the heater switches
    from cooling to heating.

    We build a line of 50 cells (like a tiny worm), connect them
    with gap junctions, and sweep the voltage from -80mV to 0mV
    to find where the "bistable switch" flips — that's the
    head/tail decision point.

SCIENTIFIC BASIS:
    - Cable equation model of planarian syncytium
    - Bistable voltage dynamics (two stable states per cell)
    - Gap junction coupling (Innexin, 0.1-10 nS range)
    - Parameters from Levin 2021 (DOI: 10.1016/j.cell.2021.02.034)

HYPOTHESIS TESTED:
    "There exists a Vmem threshold between -40mV and -20mV that
    determines anterior (head) vs posterior (tail) fate."

KILL CONDITION:
    If no bistable regime exists for physiologically reasonable
    parameters (Rm = 1-100 MΩ·cm², Gj = 0.1-10 nS), the
    voltage-switch model for head/tail fate is falsified.

OUTPUTS:
    1. voltage_threshold_sweep.png — Phase diagram showing where
       the switch flips for different gap junction strengths
    2. Console output with the exact threshold voltage

RUN: python simulations/voltage_threshold_sweep.py
"""

from brian2 import *
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.resolve()

# =============================================================================
# PARAMETERS — Planarian-relevant ranges
# =============================================================================

N_CELLS = 50              # 1D chain of cells (anterior → posterior)
DURATION = 5 * second     # Simulation time to reach steady state

# Membrane properties (planarian neoblast range)
C_M = 10 * pF             # Membrane capacitance per cell
G_LEAK = 1 * nS           # Leak conductance

# Bistable switch parameters — two stable states
# V_low ≈ -60 mV (depolarized/tail fate)
# V_high ≈ -20 mV (hyperpolarized/head fate)
# These create a bistable potential with a threshold between them
V_REST = -40 * mV         # Midpoint (unstable equilibrium)
V_LOW = -60 * mV          # "Tail" attractor
V_HIGH = -20 * mV         # "Head" attractor
DELTA_V = 15 * mV         # Width of bistable wells

# Gap junction sweep range
GJ_CONDUCTANCES = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]  # nS

# Voltage sweep for initial conditions
V_SWEEP = np.linspace(-80, 0, 25)  # mV — test these starting voltages

# Noise
NOISE_AMP = 2 * mV        # Biological voltage noise


# =============================================================================
# SIMULATION — For each Gj, sweep initial voltage, find the bifurcation
# =============================================================================

print("=" * 60)
print("VOLTAGE THRESHOLD SWEEP — Head/Tail Switch Finder")
print("=" * 60)
print()
print(f"Cells: {N_CELLS} in a 1D chain")
print(f"Gap junction range: {GJ_CONDUCTANCES} nS")
print(f"Voltage sweep: {V_SWEEP[0]:.0f} to {V_SWEEP[-1]:.0f} mV")
print(f"Duration: {DURATION/second:.0f}s per run")
print()

# Store results: for each (Gj, V_init) → final steady-state voltage
results = np.zeros((len(GJ_CONDUCTANCES), len(V_SWEEP)))
thresholds = []

for gi, gj_nS in enumerate(GJ_CONDUCTANCES):
    print(f"  Sweeping Gj = {gj_nS} nS ...")
    threshold_found = None

    for vi, v_init_mV in enumerate(V_SWEEP):
        start_scope()

        # Bistable cell model with cubic nonlinearity
        # dV/dt = (1/C) * [g_leak * (V - V_rest) * (V - V_low) * (V - V_high) / dV^2 + I_gj + noise]
        # This creates two stable fixed points at V_low and V_high
        eqs = '''
        dv/dt = (g_bistable * (v - V_low) * (v - V_high) * (v - V_rest) / (delta_V**2)
                 + I_gj + noise_amp * xi * (1/sqrt(second))) / C_m : volt
        I_gj : amp
        '''

        cells = NeuronGroup(N_CELLS, eqs,
                           threshold='v > 0*mV', reset='',
                           method='euler',
                           dt=0.1*ms)

        # Parameters
        cells.v = v_init_mV * mV

        # Add anterior bias — first 10 cells get a slight push toward V_HIGH
        # (This models the natural anterior-posterior gradient)
        cells.v[:10] = (v_init_mV + 5) * mV

        # Bistable strength
        g_bistable = -G_LEAK / (20*mV)  # Scales cubic term

        # Gap junction synapses (nearest-neighbor coupling)
        gj = Synapses(cells, cells,
                      'w : siemens',
                      on_pre='I_gj_post += w * (v_pre - v_post)')

        # Connect nearest neighbors in 1D chain
        for i in range(N_CELLS - 1):
            gj.connect(i=i, j=i+1)
            gj.connect(i=i+1, j=i)

        gj.w = gj_nS * nS

        # Gap junction current update (continuous)
        @network_operation(dt=1*ms)
        def update_gj_current():
            cells.I_gj = 0 * amp
            for syn_idx in range(len(gj)):
                i_pre = gj.i[syn_idx]
                i_post = gj.j[syn_idx]
                current = gj.w[syn_idx] * (cells.v[i_pre] - cells.v[i_post])
                cells.I_gj[i_post] += current

        # Record
        mon = StateMonitor(cells, 'v', record=[0, N_CELLS//4, N_CELLS//2, 3*N_CELLS//4, N_CELLS-1])

        # Run
        run(DURATION, report=None)

        # Final voltage (average of middle cells — avoid boundary effects)
        final_v = np.mean(mon.v[2][:, -1] / mV)  # middle cell
        results[gi, vi] = final_v

        # Detect threshold: where final_v jumps from V_LOW to V_HIGH
        if vi > 0:
            prev_v = results[gi, vi - 1]
            if prev_v < -40 and final_v > -40:
                threshold_found = (V_SWEEP[vi - 1] + V_SWEEP[vi]) / 2

        device.reinit()
        device.activate()

    if threshold_found is not None:
        thresholds.append((gj_nS, threshold_found))
        print(f"    → THRESHOLD FOUND: {threshold_found:.1f} mV")
    else:
        print(f"    → No clear bistable switch at this Gj")

print()
print("=" * 60)
print("RESULTS SUMMARY")
print("=" * 60)

if thresholds:
    avg_threshold = np.mean([t[1] for t in thresholds])
    print(f"\n  Head/Tail voltage threshold: ~{avg_threshold:.1f} mV")
    print(f"  (averaged across {len(thresholds)} gap junction conductances)")
    print()
    for gj_nS, thresh in thresholds:
        print(f"    Gj = {gj_nS:5.1f} nS → threshold = {thresh:.1f} mV")
    print()
    print("  WHAT THIS MEANS:")
    print(f"  Cells above ~{avg_threshold:.0f} mV → HEAD fate (anterior)")
    print(f"  Cells below ~{avg_threshold:.0f} mV → TAIL fate (posterior)")
    print(f"  The 'switch' voltage is around {avg_threshold:.0f} mV")
else:
    print("\n  No bistable threshold found in this parameter range.")
    print("  This could mean:")
    print("  1. The voltage range needs to be wider")
    print("  2. The bistable wells need different depths")
    print("  3. The model needs ion channel dynamics (not just bistable potential)")
    print("  → Try adjusting V_LOW, V_HIGH, or DELTA_V")


# =============================================================================
# PLOT — Phase diagram
# =============================================================================

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Left: Heatmap of final voltage vs (Gj, V_init)
ax = axes[0]
im = ax.imshow(results, aspect='auto', origin='lower',
               extent=[V_SWEEP[0], V_SWEEP[-1], 0, len(GJ_CONDUCTANCES)-1],
               cmap='RdBu_r', vmin=-70, vmax=-10)
ax.set_xlabel('Initial Voltage (mV)')
ax.set_ylabel('Gap Junction Conductance')
ax.set_yticks(range(len(GJ_CONDUCTANCES)))
ax.set_yticklabels([f'{g} nS' for g in GJ_CONDUCTANCES])
ax.set_title('Final Steady-State Voltage\n(Blue = Tail fate, Red = Head fate)')
plt.colorbar(im, ax=ax, label='Final Vmem (mV)')

# Mark thresholds
for gj_nS, thresh in thresholds:
    gi = GJ_CONDUCTANCES.index(gj_nS)
    ax.axvline(x=thresh, color='yellow', linestyle='--', alpha=0.5)
    ax.plot(thresh, gi, 'y*', markersize=15)

# Right: Threshold vs Gj
ax = axes[1]
if thresholds:
    gjs = [t[0] for t in thresholds]
    threshs = [t[1] for t in thresholds]
    ax.plot(gjs, threshs, 'ko-', markersize=8, linewidth=2)
    ax.axhline(y=avg_threshold, color='red', linestyle='--',
               label=f'Average: {avg_threshold:.1f} mV')
    ax.fill_between([min(gjs)*0.8, max(gjs)*1.2],
                    [avg_threshold - 5]*2, [avg_threshold + 5]*2,
                    alpha=0.2, color='red', label='±5 mV band')
    ax.legend()
ax.set_xlabel('Gap Junction Conductance (nS)')
ax.set_ylabel('Threshold Voltage (mV)')
ax.set_title('Head/Tail Switch Point\nvs Gap Junction Strength')
ax.grid(True, alpha=0.3)

plt.suptitle('Acheron Simulation: Voltage Threshold for Head/Tail Fate',
             fontsize=14, fontweight='bold')
plt.tight_layout()

out_path = SCRIPT_DIR / 'voltage_threshold_sweep.png'
plt.savefig(out_path, dpi=150, bbox_inches='tight')
print(f"\n  Plot saved: {out_path}")
print()
print("=" * 60)
print("NEXT STEP: If threshold found, use it as the SET_BIT target")
print("voltage in a wet-lab experiment with Valinomycin or DiBAC4(3).")
print("=" * 60)
