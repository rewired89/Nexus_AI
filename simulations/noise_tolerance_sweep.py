#!/usr/bin/env python3
"""
Noise Tolerance Sweep - Find Maximum Tolerable Noise for BER < 10^-3
=====================================================================

Sweeps noise levels to find the threshold where bit error rate exceeds 10^-3.
This answers: "What is the maximum tolerable noise before BER exceeds 10^-3?"

Uses the same majority-vote model from ber_vs_cell_count.py
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.resolve()

# =============================================================================
# PARAMETERS
# =============================================================================
VMEM_THRESHOLD = -40.0    # mV
VMEM_BIT_0 = -50.0        # mV (10mV below threshold)
VMEM_BIT_1 = -30.0        # mV (10mV above threshold)
N_CELLS = 10              # Number of cells per bit (from previous finding)
N_TRIALS = 50000          # More trials for accuracy at low BER
TARGET_BER = 1e-3

# Noise levels to test (mV standard deviation)
NOISE_LEVELS = np.concatenate([
    np.arange(1, 10, 0.5),    # 1-10 mV in 0.5 steps
    np.arange(10, 20, 1),     # 10-20 mV in 1 step
    np.arange(20, 35, 2),     # 20-35 mV in 2 steps
])

print("=" * 60)
print("Noise Tolerance Sweep")
print(f"Finding maximum noise for BER < {TARGET_BER}")
print("=" * 60)
print(f"\nFixed parameters:")
print(f"  N cells per bit: {N_CELLS}")
print(f"  Threshold:       {VMEM_THRESHOLD} mV")
print(f"  Bit 0 target:    {VMEM_BIT_0} mV")
print(f"  Bit 1 target:    {VMEM_BIT_1} mV")
print(f"  Signal margin:   {abs(VMEM_BIT_1 - VMEM_THRESHOLD)} mV")
print(f"  Trials per noise level: {N_TRIALS:,}")


def calculate_ber(n_cells: int, noise_std: float, n_trials: int) -> float:
    """Calculate BER for given noise level."""
    errors = 0

    # Test bit "0"
    vmem_0 = VMEM_BIT_0 + np.random.normal(0, noise_std, (n_trials, n_cells))
    votes_0 = np.sum(vmem_0 > VMEM_THRESHOLD, axis=1)
    errors += np.sum(votes_0 > n_cells / 2)  # Error if majority reads "1"

    # Test bit "1"
    vmem_1 = VMEM_BIT_1 + np.random.normal(0, noise_std, (n_trials, n_cells))
    votes_1 = np.sum(vmem_1 > VMEM_THRESHOLD, axis=1)
    errors += np.sum(votes_1 <= n_cells / 2)  # Error if majority reads "0"

    return errors / (2 * n_trials)


print(f"\nSweeping noise from {NOISE_LEVELS[0]} to {NOISE_LEVELS[-1]} mV...")
print("-" * 60)

ber_values = []
max_tolerable_noise = None

for noise in NOISE_LEVELS:
    ber = calculate_ber(N_CELLS, noise, N_TRIALS)
    ber_values.append(ber)

    # Track when we cross the threshold
    if max_tolerable_noise is None and ber >= TARGET_BER:
        max_tolerable_noise = noise

    # Print with status
    if ber < TARGET_BER:
        status = "✓ OK"
    elif ber < TARGET_BER * 10:
        status = "⚠ MARGINAL"
    else:
        status = "✗ FAIL"

    print(f"  Noise = {noise:5.1f} mV  |  BER = {ber:.2e}  |  {status}")

print("-" * 60)

# =============================================================================
# RESULTS
# =============================================================================
print("\n" + "=" * 60)
print("RESULTS")
print("=" * 60)

if max_tolerable_noise:
    # Interpolate to find more precise threshold
    idx = np.where(np.array(ber_values) >= TARGET_BER)[0][0]
    if idx > 0:
        # Linear interpolation between last OK and first FAIL
        noise_low = NOISE_LEVELS[idx - 1]
        noise_high = NOISE_LEVELS[idx]
        ber_low = ber_values[idx - 1]
        ber_high = ber_values[idx]

        # Interpolate to find noise at exactly TARGET_BER
        if ber_high != ber_low:
            precise_threshold = noise_low + (noise_high - noise_low) * (TARGET_BER - ber_low) / (ber_high - ber_low)
        else:
            precise_threshold = noise_high
    else:
        precise_threshold = max_tolerable_noise

    print(f"\n  MAXIMUM TOLERABLE NOISE: {precise_threshold:.1f} mV")
    print(f"  (for BER < {TARGET_BER} with {N_CELLS} cells)")

    # Safety margin
    safety_margin = precise_threshold / 5.0  # Our baseline noise
    print(f"\n  Safety margin vs 5mV baseline: {safety_margin:.1f}x")

    # SNR at threshold
    signal = abs(VMEM_BIT_1 - VMEM_THRESHOLD)
    snr_at_threshold = signal / precise_threshold
    print(f"  SNR at threshold: {snr_at_threshold:.2f}")
else:
    print(f"\n  BER remained below {TARGET_BER} for all tested noise levels!")
    print(f"  System is robust up to at least {NOISE_LEVELS[-1]} mV noise")

# Biological context
print("\n  BIOLOGICAL NOISE SOURCES:")
print("  ─────────────────────────")
print("  • Thermal noise (kT):        ~0.5-1 mV")
print("  • Ion channel stochastic:    ~2-5 mV")
print("  • Gap junction variability:  ~3-8 mV")
print("  • Metabolic fluctuations:    ~5-10 mV")
print("  • External perturbations:    ~10-20 mV")

# =============================================================================
# VISUALIZATION
# =============================================================================
plt.figure(figsize=(10, 6))

plt.semilogy(NOISE_LEVELS, ber_values, 'b-o', linewidth=2, markersize=4)
plt.axhline(y=TARGET_BER, color='r', linestyle='--', linewidth=2, label=f'Target BER = {TARGET_BER}')

if max_tolerable_noise:
    plt.axvline(x=precise_threshold, color='g', linestyle=':', linewidth=2,
                label=f'Max noise = {precise_threshold:.1f} mV')
    plt.fill_betweenx([1e-6, 1], 0, precise_threshold, color='green', alpha=0.1, label='Safe zone')
    plt.fill_betweenx([1e-6, 1], precise_threshold, NOISE_LEVELS[-1], color='red', alpha=0.1, label='Failure zone')

# Mark our baseline
plt.axvline(x=5.0, color='blue', linestyle=':', linewidth=1.5, alpha=0.7, label='Baseline (5 mV)')

plt.xlabel('Noise Standard Deviation (mV)', fontsize=12)
plt.ylabel('Bit Error Rate (BER)', fontsize=12)
plt.title(f'Noise Tolerance for {N_CELLS}-Cell Biological Memory\n(Signal margin: {abs(VMEM_BIT_1 - VMEM_THRESHOLD):.0f} mV)', fontsize=14)
plt.legend(loc='lower right', fontsize=10)
plt.grid(True, alpha=0.3, which='both')
plt.xlim(0, NOISE_LEVELS[-1] + 2)
plt.ylim(1e-5, 1)

plt.tight_layout()

output_path = SCRIPT_DIR / 'noise_tolerance_sweep.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"\nPlot saved to: {output_path}")

plt.show()
