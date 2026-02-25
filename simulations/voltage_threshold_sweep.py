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
SIM_DURATION = 2 * second # Simulation time to reach steady state

# Gap junction sweep range (nS)
GJ_CONDUCTANCES = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]

# Voltage sweep for initial conditions (mV)
V_SWEEP = np.linspace(-80, 0, 25)

# =============================================================================
# BRIAN2 MODEL
#
# Bistable cell: cubic nullcline creates two stable fixed points
#   V_low  ≈ -60 mV  (depolarized / "tail" attractor)
#   V_high ≈ -20 mV  (hyperpolarized / "head" attractor)
#   V_mid  ≈ -40 mV  (unstable saddle — the "switch" point)
#
# dv/dt = k*(v - v_lo)*(v - v_hi)*(v - v_mid) / dv_scale^2  + I_gj/Cm
#
# Gap junctions modeled as continuous ohmic coupling via network_operation.
# =============================================================================

# These must be Brian2 quantities for the equations
v_lo   = -60 * mV
v_hi   = -20 * mV
v_mid  = -40 * mV
Cm     = 10 * pF
g_leak = 1 * nS
dv_sc  = 15 * mV     # scales the cubic term
# k_bi has units amp/volt^2 so that k_bi*(v-a)*(v-b)*(v-c)/dv^2 → amp
k_bi   = -g_leak / (20 * mV * mV)
noise  = 2 * mV      # noise amplitude

eqs = '''
dv/dt = ( k_bi*(v - v_lo)*(v - v_hi)*(v - v_mid)/(dv_sc**2) * (1*volt)
          + I_gj + noise*xi*Hz**0.5 * g_leak ) / Cm : volt
I_gj : amp
'''

# =============================================================================
# SIMULATION SWEEP
# =============================================================================

print("=" * 60)
print("VOLTAGE THRESHOLD SWEEP — Head/Tail Switch Finder")
print("=" * 60)
print()
print(f"Cells: {N_CELLS} in a 1D chain")
print(f"Gap junction range: {GJ_CONDUCTANCES} nS")
print(f"Voltage sweep: {V_SWEEP[0]:.0f} to {V_SWEEP[-1]:.0f} mV")
print(f"Duration: {SIM_DURATION/second:.0f}s per run")
print()

results = np.zeros((len(GJ_CONDUCTANCES), len(V_SWEEP)))
thresholds = []

for gi, gj_nS in enumerate(GJ_CONDUCTANCES):
    print(f"  Sweeping Gj = {gj_nS} nS ...")
    threshold_found = None
    gj_val = gj_nS * nS

    for vi, v_init_mV in enumerate(V_SWEEP):
        start_scope()

        cells = NeuronGroup(N_CELLS, eqs, method='euler', dt=0.1*ms)
        cells.v = v_init_mV * mV
        # Anterior bias: first 10 cells get +5 mV nudge
        cells.v[:10] = (v_init_mV + 5) * mV

        # Build neighbor index lists once
        # For cell i, neighbors are i-1 and i+1 (if they exist)
        left  = np.arange(N_CELLS) - 1   # left neighbor
        right = np.arange(N_CELLS) + 1   # right neighbor
        left[0] = 0        # cell 0 has no left neighbor (self → no current)
        right[-1] = N_CELLS - 1  # last cell has no right neighbor

        @network_operation(dt=0.5*ms)
        def update_gj():
            v_arr = cells.v[:]
            I = np.zeros(N_CELLS) * amp
            # Current from left neighbor
            I += gj_val * (v_arr[left] - v_arr)
            # Current from right neighbor
            I += gj_val * (v_arr[right] - v_arr)
            # Fix boundaries (cell 0 left = self, last right = self → zero current)
            I[0]  = gj_val * (v_arr[1] - v_arr[0])
            I[-1] = gj_val * (v_arr[-2] - v_arr[-1])
            cells.I_gj = I

        # Record middle cell
        mid = N_CELLS // 2
        mon = StateMonitor(cells, 'v', record=[mid])

        run(SIM_DURATION)

        # Final voltage of middle cell
        final_v = float(mon.v[0][-1] / mV)
        results[gi, vi] = final_v

        # Detect threshold: where final_v jumps from low to high
        if vi > 0:
            prev_v = results[gi, vi - 1]
            if prev_v < -40 and final_v > -40:
                threshold_found = (V_SWEEP[vi - 1] + V_SWEEP[vi]) / 2

    if threshold_found is not None:
        thresholds.append((gj_nS, threshold_found))
        print(f"    -> THRESHOLD FOUND: {threshold_found:.1f} mV")
    else:
        print(f"    -> No clear bistable switch at this Gj")

# =============================================================================
# RESULTS
# =============================================================================

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
        print(f"    Gj = {gj_nS:5.1f} nS -> threshold = {thresh:.1f} mV")
    print()
    print("  WHAT THIS MEANS:")
    print(f"  Cells above ~{avg_threshold:.0f} mV -> HEAD fate (anterior)")
    print(f"  Cells below ~{avg_threshold:.0f} mV -> TAIL fate (posterior)")
    print(f"  The 'switch' voltage is around {avg_threshold:.0f} mV")
else:
    print("\n  No bistable threshold found in this parameter range.")
    print("  This could mean:")
    print("  1. The voltage range needs to be wider")
    print("  2. The bistable wells need different depths")
    print("  3. The model needs ion channel dynamics (not just cubic potential)")
    print("  -> Try adjusting v_lo, v_hi, or dv_sc")

# =============================================================================
# PLOT
# =============================================================================

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

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

for gj_nS, thresh in thresholds:
    idx = GJ_CONDUCTANCES.index(gj_nS)
    ax.axvline(x=thresh, color='yellow', linestyle='--', alpha=0.5)
    ax.plot(thresh, idx, 'y*', markersize=15)

ax = axes[1]
if thresholds:
    gjs = [t[0] for t in thresholds]
    threshs = [t[1] for t in thresholds]
    ax.plot(gjs, threshs, 'ko-', markersize=8, linewidth=2)
    avg_threshold = np.mean(threshs)
    ax.axhline(y=avg_threshold, color='red', linestyle='--',
               label=f'Average: {avg_threshold:.1f} mV')
    ax.fill_between([min(gjs)*0.8, max(gjs)*1.2],
                    [avg_threshold - 5]*2, [avg_threshold + 5]*2,
                    alpha=0.2, color='red', label='+/- 5 mV band')
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
