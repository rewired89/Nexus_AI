#!/usr/bin/env python3
"""
GOLDEN TEMPLATE SELF-REPAIR TEST
================================

THE QUESTION:
Does the body maintain a "Golden Image" (master template) that can
repair damaged voltage patterns during regeneration?

THE TEST:
1. Start with 100 cells storing [1, 0, 1, 0]
2. At t=5s, DAMAGE 50% of cells (flip to wrong voltage)
3. Measure: Does the network self-correct back to original?

POSSIBLE OUTCOMES:
A) Full self-repair → Network has redundant template (majority vote)
B) Partial repair → Some error correction exists
C) No repair → Damage is permanent, no master copy
D) Corruption spreads → Damaged cells corrupt healthy ones

This answers: Is there a "Golden Template" for biological memory?
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.resolve()

# =============================================================================
# PARAMETERS
# =============================================================================

N_CELLS = 100
N_BITS = 4
CELLS_PER_BIT = N_CELLS // N_BITS

# Biophysics
C_M = 1.0
G_LEAK = 0.1
V_REST = -60.0
V_LOGIC_1 = -30.0
V_LOGIC_0 = -60.0
V_THRESHOLD = -45.0

# Bistability
BISTABLE_STRENGTH = 0.8
V_SCALE = 5.0

# Gap junctions (key for self-repair)
G_GJ = 0.15  # Higher coupling for repair

# Noise
NOISE_STD = 1.5

# Timing
DT = 0.1
T_DAMAGE = 5000      # Damage at 5 seconds
T_TOTAL = 30000      # 30 second simulation

# Damage parameters
DAMAGE_FRACTION = 0.5  # 50% of cells damaged


# =============================================================================
# BISTABILITY MODEL
# =============================================================================

def bistable_current(V):
    sigmoid = 1.0 / (1.0 + np.exp(-(V - V_THRESHOLD) / V_SCALE))
    return BISTABLE_STRENGTH * (2 * sigmoid - 1) * (V_LOGIC_1 - V_LOGIC_0)


def leak_current(V):
    return G_LEAK * (V_REST - V)


def gap_junction_current(V, bit_assignments):
    I_gj = np.zeros_like(V)
    for bit_idx in range(N_BITS):
        mask = bit_assignments == bit_idx
        V_mean = np.mean(V[mask])
        I_gj[mask] = G_GJ * (V_mean - V[mask])
    return I_gj


def initialize_pattern(pattern):
    V = np.zeros(N_CELLS)
    for bit_idx, bit_value in enumerate(pattern):
        start = bit_idx * CELLS_PER_BIT
        end = start + CELLS_PER_BIT
        target_V = V_LOGIC_1 if bit_value == 1 else V_LOGIC_0
        V[start:end] = target_V + np.random.randn(CELLS_PER_BIT) * 2
    return V


def read_pattern(V):
    pattern = []
    for bit_idx in range(N_BITS):
        start = bit_idx * CELLS_PER_BIT
        end = start + CELLS_PER_BIT
        V_mean = np.mean(V[start:end])
        pattern.append(1 if V_mean > V_THRESHOLD else 0)
    return pattern


def apply_damage(V, bit_assignments, damage_fraction=0.5):
    """
    Damage cells by flipping their voltage to the WRONG state.
    Returns indices of damaged cells.
    """
    n_damage = int(N_CELLS * damage_fraction)
    damage_indices = np.random.choice(N_CELLS, size=n_damage, replace=False)

    for idx in damage_indices:
        # Flip to opposite state
        if V[idx] > V_THRESHOLD:
            V[idx] = V_LOGIC_0 + np.random.randn() * 2
        else:
            V[idx] = V_LOGIC_1 + np.random.randn() * 2

    return damage_indices


# =============================================================================
# MAIN SIMULATION
# =============================================================================

def run_golden_template_test():
    print("=" * 70)
    print("GOLDEN TEMPLATE SELF-REPAIR TEST")
    print("=" * 70)

    INITIAL_PATTERN = [1, 0, 1, 0]
    print(f"\nInitial pattern: {INITIAL_PATTERN}")
    print(f"Damage applied at: t = {T_DAMAGE/1000:.1f} seconds")
    print(f"Damage fraction: {DAMAGE_FRACTION * 100:.0f}%")

    # Initialize
    V = initialize_pattern(INITIAL_PATTERN)
    bit_assignments = np.repeat(np.arange(N_BITS), CELLS_PER_BIT)

    # Recording
    n_steps = int(T_TOTAL / DT)
    record_interval = 50
    n_records = n_steps // record_interval

    time_record = np.zeros(n_records)
    V_mean_record = np.zeros((n_records, N_BITS))
    pattern_record = []
    accuracy_record = []

    damage_applied = False
    damage_indices = []
    pattern_at_damage = None
    pattern_after_damage = None

    print("\nRunning simulation...")

    record_idx = 0
    for step in range(n_steps):
        t = step * DT

        # === DAMAGE EVENT ===
        if t >= T_DAMAGE and not damage_applied:
            damage_applied = True
            pattern_at_damage = read_pattern(V)
            print(f"\n>>> DAMAGE APPLIED at t = {t/1000:.1f}s <<<")
            print(f"    Pattern before damage: {pattern_at_damage}")

            damage_indices = apply_damage(V, bit_assignments, DAMAGE_FRACTION)

            pattern_after_damage = read_pattern(V)
            print(f"    Pattern after damage:  {pattern_after_damage}")
            print(f"    Cells damaged: {len(damage_indices)}")

            # Count damaged per bit
            for bit_idx in range(N_BITS):
                start = bit_idx * CELLS_PER_BIT
                end = start + CELLS_PER_BIT
                n_damaged = np.sum((damage_indices >= start) & (damage_indices < end))
                print(f"      Bit {bit_idx}: {n_damaged}/{CELLS_PER_BIT} cells damaged")

        # === CALCULATE CURRENTS ===
        I_leak = leak_current(V)
        I_bistable = bistable_current(V)
        I_gj = gap_junction_current(V, bit_assignments)
        I_noise = np.random.randn(N_CELLS) * NOISE_STD * np.sqrt(DT)

        # Update
        dV = (I_leak + I_bistable + I_gj) * DT / C_M + I_noise
        V = V + dV

        # === RECORD ===
        if step % record_interval == 0 and record_idx < n_records:
            time_record[record_idx] = t / 1000

            for bit_idx in range(N_BITS):
                start = bit_idx * CELLS_PER_BIT
                end = start + CELLS_PER_BIT
                V_mean_record[record_idx, bit_idx] = np.mean(V[start:end])

            current_pattern = read_pattern(V)
            pattern_record.append(current_pattern)

            # Calculate accuracy vs original
            accuracy = sum(1 for a, b in zip(current_pattern, INITIAL_PATTERN) if a == b) / N_BITS
            accuracy_record.append(accuracy)

            record_idx += 1

        # Progress
        if step % (n_steps // 10) == 0:
            pattern = read_pattern(V)
            accuracy = sum(1 for a, b in zip(pattern, INITIAL_PATTERN) if a == b) / N_BITS
            print(f"  t = {t/1000:.1f}s, Pattern: {pattern}, Accuracy: {accuracy*100:.0f}%")

    # ==========================================================================
    # ANALYSIS
    # ==========================================================================

    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    final_pattern = read_pattern(V)
    final_accuracy = sum(1 for a, b in zip(final_pattern, INITIAL_PATTERN) if a == b) / N_BITS

    print(f"\nOriginal pattern:      {INITIAL_PATTERN}")
    print(f"Pattern after damage:  {pattern_after_damage}")
    print(f"Final pattern:         {final_pattern}")
    print(f"Final accuracy:        {final_accuracy * 100:.0f}%")

    # Determine repair outcome
    if final_pattern == INITIAL_PATTERN:
        repair_result = "FULL SELF-REPAIR"
        interpretation = "Golden Template EXISTS - majority vote corrected errors"
    elif final_accuracy > 0.5:
        repair_result = "PARTIAL SELF-REPAIR"
        interpretation = "Some error correction, but not complete"
    elif final_pattern == pattern_after_damage:
        repair_result = "NO REPAIR"
        interpretation = "Damage is permanent - no template exists"
    else:
        repair_result = "CORRUPTION SPREAD"
        interpretation = "Damaged cells corrupted healthy ones!"

    print(f"\nREPAIR RESULT: {repair_result}")
    print(f"Interpretation: {interpretation}")

    # ==========================================================================
    # PLOTTING
    # ==========================================================================

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    colors = ['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4']
    labels = ['Bit 0 (HEAD)', 'Bit 1 (ANT)', 'Bit 2 (POST)', 'Bit 3 (TAIL)']

    # --- Panel 1: Voltage traces ---
    ax1 = axes[0, 0]
    for bit_idx in range(N_BITS):
        ax1.plot(time_record, V_mean_record[:, bit_idx],
                color=colors[bit_idx], linewidth=2, label=labels[bit_idx])
    ax1.axvline(x=T_DAMAGE/1000, color='red', linestyle='--', linewidth=2, label='Damage')
    ax1.axhline(y=V_THRESHOLD, color='gray', linestyle=':', alpha=0.5)
    ax1.set_xlabel('Time (seconds)')
    ax1.set_ylabel('Mean Voltage (mV)')
    ax1.set_title('VOLTAGE TRACES (watch for recovery)')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)

    # --- Panel 2: Accuracy over time ---
    ax2 = axes[0, 1]
    time_acc = time_record[:len(accuracy_record)]
    ax2.plot(time_acc, np.array(accuracy_record) * 100, 'g-', linewidth=2)
    ax2.axvline(x=T_DAMAGE/1000, color='red', linestyle='--', linewidth=2, label='Damage')
    ax2.axhline(y=100, color='green', linestyle=':', alpha=0.5, label='Perfect')
    ax2.axhline(y=50, color='orange', linestyle=':', alpha=0.5, label='Random')
    ax2.set_xlabel('Time (seconds)')
    ax2.set_ylabel('Pattern Accuracy (%)')
    ax2.set_title('SELF-REPAIR: ACCURACY vs TIME')
    ax2.legend(loc='lower right')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 110)

    # --- Panel 3: Bit-by-bit recovery ---
    ax3 = axes[1, 0]

    pattern_array = np.array(pattern_record)
    expected = np.array(INITIAL_PATTERN)

    for bit_idx in range(N_BITS):
        correct = pattern_array[:, bit_idx] == expected[bit_idx]
        ax3.fill_between(time_acc, bit_idx, bit_idx + correct.astype(float),
                        color=colors[bit_idx], alpha=0.7)
        ax3.fill_between(time_acc, bit_idx, bit_idx + (~correct).astype(float) * 0.3,
                        color='red', alpha=0.3)

    ax3.axvline(x=T_DAMAGE/1000, color='red', linestyle='--', linewidth=2)
    ax3.set_xlabel('Time (seconds)')
    ax3.set_ylabel('Bit (colored=correct, red=wrong)')
    ax3.set_title('BIT-BY-BIT CORRECTNESS')
    ax3.set_yticks([0.5, 1.5, 2.5, 3.5])
    ax3.set_yticklabels(['Bit 0', 'Bit 1', 'Bit 2', 'Bit 3'])
    ax3.grid(True, alpha=0.3, axis='x')

    # --- Panel 4: Summary ---
    ax4 = axes[1, 1]
    ax4.axis('off')

    summary = f"""
    GOLDEN TEMPLATE SELF-REPAIR TEST
    {'═' * 40}

    Original Pattern:    {INITIAL_PATTERN}
    Damage Fraction:     {DAMAGE_FRACTION * 100:.0f}%
    Damage Time:         t = {T_DAMAGE/1000:.1f} s

    {'─' * 40}

    Pattern After Damage: {pattern_after_damage}
    Final Pattern:        {final_pattern}
    Final Accuracy:       {final_accuracy * 100:.0f}%

    {'─' * 40}

    RESULT: {repair_result}

    {interpretation}

    {'═' * 40}

    MECHANISM (if repair occurred):
    1. Undamaged cells maintain correct state
    2. Gap junctions couple damaged to healthy
    3. Majority vote pulls damaged cells back
    4. Bistability locks in corrected state

    {'═' * 40}

    BIOLOGICAL IMPLICATION:
    {'The bioelectric field IS the template!' if 'FULL' in repair_result else 'External template may be needed.'}
    {'Regeneration can use majority vote.' if 'FULL' in repair_result else 'Check epigenetic/structural backup.'}
    """

    ax4.text(0.5, 0.5, summary, transform=ax4.transAxes,
             fontsize=10, fontfamily='monospace',
             ha='center', va='center',
             bbox=dict(boxstyle='round', facecolor='#f0f0f0', edgecolor='black'))

    plt.suptitle(f'GOLDEN TEMPLATE TEST: {repair_result}',
                fontsize=14, fontweight='bold')
    plt.tight_layout()

    output_path = SCRIPT_DIR / 'golden_template_results.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved: {output_path}")

    plt.show()

    return repair_result, final_accuracy


if __name__ == "__main__":
    result, accuracy = run_golden_template_test()

    print("\n" + "=" * 70)
    print("WHAT THIS MEANS FOR REGENERATION")
    print("=" * 70)

    if "FULL" in result:
        print("""
    GOLDEN TEMPLATE EXISTS!

    The bioelectric pattern IS the template. Here's why:
    - Each cell "votes" with its neighbors via gap junctions
    - Majority wins: if >50% of a region is correct, it corrects the rest
    - This is ERROR CORRECTION built into the tissue architecture

    For regeneration:
    - When new cells form, they're "taught" by existing tissue
    - The body doesn't need a separate "master copy"
    - The distributed pattern IS the master copy

    In the lab:
    - Damage a region of tissue (laser ablation)
    - Watch the Vmem pattern repair itself
    - If >50% survives, pattern should recover
        """)
    else:
        print("""
    NO AUTOMATIC SELF-REPAIR

    The bioelectric pattern alone may not be sufficient:
    - Damage beyond threshold cannot be corrected
    - External template may be needed (epigenetic, structural)
    - Or: gap junction coupling is too weak for repair

    For regeneration:
    - Need to look for backup storage mechanism
    - Check gene expression patterns
    - Test if structural cues (ECM) provide template
        """)
