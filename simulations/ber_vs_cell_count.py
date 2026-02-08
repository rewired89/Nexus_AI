#!/usr/bin/env python3
"""
BER vs Cell Count Simulation for 4-Bit Biological Memory
=========================================================

Simulates a population of neoblasts storing a single bit via majority-vote.
Each cell has Vmem = -40mV baseline with ±5mV Gaussian noise.

Bit encoding:
  - Bit "0": Vmem target = -50mV (hyperpolarized)
  - Bit "1": Vmem target = -30mV (depolarized)
  - Threshold: -40mV

Error occurs when noise causes majority of cells to cross threshold incorrectly.

Output: BER vs N curve showing minimum cells needed for BER < 10^-3
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Cross-platform output path
SCRIPT_DIR = Path(__file__).parent.resolve()

# =============================================================================
# SIMULATION PARAMETERS (from Nexus recommendation)
# =============================================================================
VMEM_THRESHOLD = -40.0    # mV - decision threshold
VMEM_NOISE_STD = 5.0      # mV - Gaussian noise standard deviation
VMEM_BIT_0 = -50.0        # mV - target for storing "0" (hyperpolarized)
VMEM_BIT_1 = -30.0        # mV - target for storing "1" (depolarized)

N_CELLS_RANGE = np.concatenate([
    np.arange(10, 100, 10),      # 10, 20, 30, ... 90
    np.arange(100, 1001, 50)     # 100, 150, 200, ... 1000
])

N_TRIALS = 10000  # Number of bit storage/retrieval trials per cell count
TARGET_BER = 1e-3  # Target BER < 10^-3


def simulate_bit_storage(n_cells: int, target_vmem: float, n_trials: int) -> np.ndarray:
    """
    Simulate storing a bit across n_cells with Gaussian noise.

    Returns array of measured Vmem values for each cell in each trial.
    Shape: (n_trials, n_cells)
    """
    # Each cell's Vmem = target + Gaussian noise
    vmem_values = target_vmem + np.random.normal(0, VMEM_NOISE_STD, (n_trials, n_cells))
    return vmem_values


def majority_vote_readout(vmem_values: np.ndarray, threshold: float) -> np.ndarray:
    """
    Read bit using majority vote across cells.

    Returns: Array of read bits (0 or 1) for each trial
    """
    # Count cells above threshold (reading as "1")
    cells_above_threshold = vmem_values > threshold
    votes_for_1 = np.sum(cells_above_threshold, axis=1)

    # Majority vote: if more than half vote "1", read as "1"
    n_cells = vmem_values.shape[1]
    read_bits = (votes_for_1 > n_cells / 2).astype(int)

    return read_bits


def calculate_ber(n_cells: int, n_trials: int) -> float:
    """
    Calculate Bit Error Rate for given cell count.

    Tests both bit "0" and bit "1" storage.
    """
    errors = 0
    total_bits = 0

    # Test storing bit "0"
    vmem_0 = simulate_bit_storage(n_cells, VMEM_BIT_0, n_trials)
    read_0 = majority_vote_readout(vmem_0, VMEM_THRESHOLD)
    errors += np.sum(read_0 != 0)  # Error if we read "1" when we stored "0"
    total_bits += n_trials

    # Test storing bit "1"
    vmem_1 = simulate_bit_storage(n_cells, VMEM_BIT_1, n_trials)
    read_1 = majority_vote_readout(vmem_1, VMEM_THRESHOLD)
    errors += np.sum(read_1 != 1)  # Error if we read "0" when we stored "1"
    total_bits += n_trials

    ber = errors / total_bits
    return ber


def run_simulation():
    """Run full BER vs N sweep and plot results."""

    print("=" * 60)
    print("BER vs Cell Count Simulation")
    print("4-Bit Biological Memory - Neoblast Majority Vote")
    print("=" * 60)
    print(f"\nParameters:")
    print(f"  Vmem threshold:     {VMEM_THRESHOLD} mV")
    print(f"  Vmem noise (std):   ±{VMEM_NOISE_STD} mV")
    print(f"  Bit '0' target:     {VMEM_BIT_0} mV")
    print(f"  Bit '1' target:     {VMEM_BIT_1} mV")
    print(f"  Trials per N:       {N_TRIALS:,}")
    print(f"  Target BER:         < {TARGET_BER}")
    print(f"\nSweeping N from {N_CELLS_RANGE[0]} to {N_CELLS_RANGE[-1]} cells...")
    print("-" * 60)

    ber_values = []
    min_n_for_target = None

    for n_cells in N_CELLS_RANGE:
        ber = calculate_ber(n_cells, N_TRIALS)
        ber_values.append(ber)

        # Check if we've reached target BER
        if min_n_for_target is None and ber < TARGET_BER:
            min_n_for_target = n_cells

        # Progress output
        status = "✓ TARGET MET" if ber < TARGET_BER else ""
        print(f"  N = {n_cells:4d} cells  |  BER = {ber:.2e}  {status}")

    print("-" * 60)

    # Results summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)

    if min_n_for_target:
        print(f"\n  MINIMUM CELLS FOR BER < {TARGET_BER}: {min_n_for_target} cells")
        print(f"\n  For 4-bit memory: {min_n_for_target * 4} cells total")
        print(f"  (4 independent cell populations, one per bit)")
    else:
        print(f"\n  WARNING: BER target not achieved within tested range!")
        print(f"  Consider: larger cell populations or better error correction")

    # Theoretical comparison
    print(f"\n  Signal-to-Noise Ratio:")
    signal = abs(VMEM_BIT_1 - VMEM_BIT_0) / 2  # Half the gap from threshold
    snr = signal / VMEM_NOISE_STD
    print(f"    Signal (half-gap): {signal} mV")
    print(f"    Noise (std):       {VMEM_NOISE_STD} mV")
    print(f"    SNR:               {snr:.1f}")

    # Plot
    plt.figure(figsize=(10, 6))

    plt.semilogy(N_CELLS_RANGE, ber_values, 'b-o', linewidth=2, markersize=4, label='Simulated BER')
    plt.axhline(y=TARGET_BER, color='r', linestyle='--', linewidth=2, label=f'Target BER = {TARGET_BER}')

    if min_n_for_target:
        plt.axvline(x=min_n_for_target, color='g', linestyle=':', linewidth=2,
                    label=f'Min N = {min_n_for_target} cells')

    plt.xlabel('Number of Neoblasts (N)', fontsize=12)
    plt.ylabel('Bit Error Rate (BER)', fontsize=12)
    plt.title('BER vs Cell Count for Biological Memory\n(Majority-Vote Error Correction, Vmem ±5mV noise)', fontsize=14)
    plt.legend(loc='upper right', fontsize=10)
    plt.grid(True, alpha=0.3, which='both')
    plt.xlim(0, N_CELLS_RANGE[-1] + 50)
    plt.ylim(1e-5, 1)

    # Add annotation
    if min_n_for_target:
        plt.annotate(f'N = {min_n_for_target}\nBER < 10⁻³',
                     xy=(min_n_for_target, TARGET_BER),
                     xytext=(min_n_for_target + 150, TARGET_BER * 10),
                     fontsize=10,
                     arrowprops=dict(arrowstyle='->', color='green'))

    plt.tight_layout()

    # Save figure
    output_path = SCRIPT_DIR / 'ber_vs_cell_count.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to: {output_path}")

    plt.show()

    return min_n_for_target, ber_values


if __name__ == "__main__":
    np.random.seed(42)  # For reproducibility
    min_n, ber_values = run_simulation()
