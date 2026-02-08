#!/usr/bin/env python3
"""
MAINTENANCE COST TEST: DRAM vs Flash Memory
============================================

THE QUESTION:
Does biological memory require constant energy (like DRAM refresh)?
Or is it a permanent physical state (like Flash)?

THE TEST:
1. Run bistable network storing pattern [1,0,1,0]
2. At t=10s, DISABLE active ion channels (only passive leak remains)
3. Measure: How long until pattern corrupts?

PREDICTION:
- If T_hold < 1 second after disable → DRAM-like (needs constant pump activity)
- If T_hold > 100 seconds → Flash-like (bistability is self-sustaining)
- If T_hold is infinite → True permanent storage (expression-based)

This directly answers: What is the "maintenance cost" of biological memory?
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.resolve()

# =============================================================================
# SIMULATION PARAMETERS
# =============================================================================

N_CELLS = 100          # Total cells (25 per bit)
N_BITS = 4             # 4-bit memory
CELLS_PER_BIT = N_CELLS // N_BITS

# Biophysical parameters
C_M = 1.0              # Membrane capacitance (normalized)
G_LEAK = 0.1           # Passive leak conductance (always present)
V_REST = -60.0         # Leak reversal potential (mV)
V_LOGIC_1 = -30.0      # Depolarized state
V_LOGIC_0 = -60.0      # Hyperpolarized state

# Bistability parameters (active ion channels)
BISTABLE_STRENGTH = 0.5   # Strength of bistable current
V_THRESHOLD = -45.0       # Threshold between states
V_SCALE = 5.0             # Steepness of transition

# Gap junction coupling
G_GJ = 0.05            # Gap junction conductance (within-bit coupling)

# Noise
NOISE_STD = 2.0        # mV noise

# Timing
DT = 0.1               # Time step (ms)
T_TOTAL = 60000        # Total simulation time (60 seconds)
T_DISABLE = 10000      # Time to disable active channels (10 seconds)

# =============================================================================
# BISTABLE CURRENT MODEL
# =============================================================================

def bistable_current(V, active=True):
    """
    Bistable ion channel current.

    When active: provides positive feedback that locks cells into either
    V_LOGIC_0 or V_LOGIC_1 state.

    When disabled: returns 0 (only passive leak remains)
    """
    if not active:
        return np.zeros_like(V)

    # Sigmoid centered at threshold
    # Positive current pushes toward V_LOGIC_1 when V > threshold
    # Negative current pushes toward V_LOGIC_0 when V < threshold
    sigmoid = 1.0 / (1.0 + np.exp(-(V - V_THRESHOLD) / V_SCALE))

    # Current that creates bistability
    I_bistable = BISTABLE_STRENGTH * (2 * sigmoid - 1) * (V_LOGIC_1 - V_LOGIC_0)

    return I_bistable


def leak_current(V):
    """Passive leak current - always present."""
    return G_LEAK * (V_REST - V)


def gap_junction_current(V, bit_assignments):
    """
    Gap junction coupling within each bit-group.
    Cells within the same bit are coupled; cells in different bits are isolated.
    """
    I_gj = np.zeros_like(V)

    for bit_idx in range(N_BITS):
        start = bit_idx * CELLS_PER_BIT
        end = start + CELLS_PER_BIT

        # Mean voltage of this bit's cells
        V_mean = np.mean(V[start:end])

        # Gap junction current pulls toward mean
        I_gj[start:end] = G_GJ * (V_mean - V[start:end])

    return I_gj


# =============================================================================
# INITIALIZE MEMORY PATTERN
# =============================================================================

def initialize_pattern(pattern):
    """Initialize voltage array with given bit pattern."""
    V = np.zeros(N_CELLS)

    for bit_idx, bit_value in enumerate(pattern):
        start = bit_idx * CELLS_PER_BIT
        end = start + CELLS_PER_BIT

        if bit_value == 1:
            V[start:end] = V_LOGIC_1 + np.random.randn(CELLS_PER_BIT) * 2
        else:
            V[start:end] = V_LOGIC_0 + np.random.randn(CELLS_PER_BIT) * 2

    return V


def read_pattern(V):
    """Read the current bit pattern from voltages."""
    pattern = []
    for bit_idx in range(N_BITS):
        start = bit_idx * CELLS_PER_BIT
        end = start + CELLS_PER_BIT
        V_mean = np.mean(V[start:end])
        bit = 1 if V_mean > V_THRESHOLD else 0
        pattern.append(bit)
    return pattern


# =============================================================================
# MAIN SIMULATION
# =============================================================================

def run_maintenance_test():
    """Run the DRAM vs Flash test."""

    print("=" * 70)
    print("MAINTENANCE COST TEST: DRAM vs Flash")
    print("=" * 70)

    # Initial pattern
    INITIAL_PATTERN = [1, 0, 1, 0]
    print(f"\nInitial pattern: {INITIAL_PATTERN}")
    print(f"Active channels disabled at: t = {T_DISABLE/1000:.1f} seconds")
    print(f"Total simulation time: {T_TOTAL/1000:.1f} seconds")

    # Initialize
    V = initialize_pattern(INITIAL_PATTERN)
    bit_assignments = np.repeat(np.arange(N_BITS), CELLS_PER_BIT)

    # Recording
    n_steps = int(T_TOTAL / DT)
    record_interval = 100  # Record every 100 steps
    n_records = n_steps // record_interval

    time_record = np.zeros(n_records)
    V_mean_record = np.zeros((n_records, N_BITS))
    pattern_record = []

    # Track when pattern corrupts
    corruption_time = None
    active_channels = True

    print("\nRunning simulation...")

    record_idx = 0
    for step in range(n_steps):
        t = step * DT

        # Disable active channels at T_DISABLE
        if t >= T_DISABLE and active_channels:
            active_channels = False
            print(f"\n>>> ACTIVE CHANNELS DISABLED at t = {t/1000:.1f}s <<<")
            print(f"    Pattern at disable: {read_pattern(V)}")

        # Calculate currents
        I_leak = leak_current(V)
        I_bistable = bistable_current(V, active=active_channels)
        I_gj = gap_junction_current(V, bit_assignments)
        I_noise = np.random.randn(N_CELLS) * NOISE_STD * np.sqrt(DT)

        # Update voltage
        dV = (I_leak + I_bistable + I_gj) * DT / C_M + I_noise
        V = V + dV

        # Record
        if step % record_interval == 0:
            time_record[record_idx] = t
            for bit_idx in range(N_BITS):
                start = bit_idx * CELLS_PER_BIT
                end = start + CELLS_PER_BIT
                V_mean_record[record_idx, bit_idx] = np.mean(V[start:end])

            current_pattern = read_pattern(V)
            pattern_record.append(current_pattern)

            # Check for corruption (after disable)
            if not active_channels and corruption_time is None:
                if current_pattern != INITIAL_PATTERN:
                    corruption_time = t
                    print(f"\n>>> PATTERN CORRUPTED at t = {t/1000:.2f}s <<<")
                    print(f"    Expected: {INITIAL_PATTERN}")
                    print(f"    Got:      {current_pattern}")

            record_idx += 1

        # Progress
        if step % (n_steps // 10) == 0:
            print(f"  t = {t/1000:.1f}s, Pattern: {read_pattern(V)}, Active: {active_channels}")

    # ==========================================================================
    # ANALYSIS
    # ==========================================================================

    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    time_s = time_record / 1000  # Convert to seconds

    if corruption_time is not None:
        T_hold = (corruption_time - T_DISABLE) / 1000  # seconds after disable
        print(f"\nT_hold (time until corruption after disable): {T_hold:.2f} seconds")

        if T_hold < 1:
            memory_type = "DRAM-LIKE"
            explanation = "Requires constant active maintenance"
        elif T_hold < 100:
            memory_type = "HYBRID"
            explanation = "Semi-stable, benefits from refresh"
        else:
            memory_type = "FLASH-LIKE"
            explanation = "Self-sustaining bistability"
    else:
        T_hold = (T_TOTAL - T_DISABLE) / 1000
        memory_type = "FLASH-LIKE (or better)"
        explanation = f"Pattern survived {T_hold:.1f}s without active channels!"
        print(f"\nPattern NEVER corrupted!")
        print(f"Survived {T_hold:.1f} seconds after active channels disabled")

    print(f"\nMEMORY TYPE: {memory_type}")
    print(f"Explanation: {explanation}")

    # ==========================================================================
    # PLOTTING
    # ==========================================================================

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # --- Panel 1: Voltage traces ---
    ax1 = axes[0, 0]
    colors = ['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4']
    labels = ['Bit 0 (HEAD)', 'Bit 1 (ANT)', 'Bit 2 (POST)', 'Bit 3 (TAIL)']

    for bit_idx in range(N_BITS):
        ax1.plot(time_s, V_mean_record[:, bit_idx],
                color=colors[bit_idx], linewidth=1.5, label=labels[bit_idx])

    ax1.axvline(x=T_DISABLE/1000, color='red', linestyle='--', linewidth=2,
                label='Channels Disabled')
    ax1.axhline(y=V_THRESHOLD, color='gray', linestyle=':', alpha=0.5)
    ax1.axhline(y=V_LOGIC_1, color='green', linestyle=':', alpha=0.3)
    ax1.axhline(y=V_LOGIC_0, color='blue', linestyle=':', alpha=0.3)

    if corruption_time is not None:
        ax1.axvline(x=corruption_time/1000, color='orange', linestyle='--',
                   linewidth=2, label='Pattern Corrupted')

    ax1.set_xlabel('Time (seconds)')
    ax1.set_ylabel('Mean Voltage (mV)')
    ax1.set_title('VOLTAGE TRACES BY BIT REGION')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, T_TOTAL/1000)

    # --- Panel 2: Zoom on disable event ---
    ax2 = axes[0, 1]

    t_zoom_start = T_DISABLE/1000 - 2
    t_zoom_end = min(T_DISABLE/1000 + 20, T_TOTAL/1000)

    mask = (time_s >= t_zoom_start) & (time_s <= t_zoom_end)

    for bit_idx in range(N_BITS):
        ax2.plot(time_s[mask], V_mean_record[mask, bit_idx],
                color=colors[bit_idx], linewidth=2, label=labels[bit_idx])

    ax2.axvline(x=T_DISABLE/1000, color='red', linestyle='--', linewidth=2)
    ax2.axhline(y=V_THRESHOLD, color='gray', linestyle=':', alpha=0.5, label='Threshold')

    ax2.set_xlabel('Time (seconds)')
    ax2.set_ylabel('Mean Voltage (mV)')
    ax2.set_title('ZOOM: MOMENT OF CHANNEL DISABLE')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)

    # --- Panel 3: Pattern stability ---
    ax3 = axes[1, 0]

    pattern_array = np.array(pattern_record)
    pattern_times = time_s[:len(pattern_record)]

    for bit_idx in range(N_BITS):
        y_offset = bit_idx * 1.5
        ax3.fill_between(pattern_times, y_offset, y_offset + pattern_array[:, bit_idx],
                        color=colors[bit_idx], alpha=0.7, label=labels[bit_idx])

    ax3.axvline(x=T_DISABLE/1000, color='red', linestyle='--', linewidth=2)
    ax3.set_xlabel('Time (seconds)')
    ax3.set_ylabel('Bit Value (stacked)')
    ax3.set_title('BIT VALUES OVER TIME')
    ax3.set_yticks([0.5, 2, 3.5, 5])
    ax3.set_yticklabels(['Bit 0', 'Bit 1', 'Bit 2', 'Bit 3'])
    ax3.set_xlim(0, T_TOTAL/1000)
    ax3.grid(True, alpha=0.3, axis='x')

    # --- Panel 4: Summary ---
    ax4 = axes[1, 1]
    ax4.axis('off')

    summary = f"""
    MAINTENANCE COST TEST RESULTS
    {'═' * 40}

    Initial Pattern:     {INITIAL_PATTERN}
    Channels Disabled:   t = {T_DISABLE/1000:.1f} s

    {'─' * 40}

    T_hold (after disable): {T_hold:.2f} seconds

    MEMORY TYPE: {memory_type}

    {explanation}

    {'─' * 40}

    INTERPRETATION:

    {'Pattern maintained by passive bistability alone.' if corruption_time is None else f'Pattern decayed after {T_hold:.2f}s without active channels.'}

    {'Gap junction coupling + bistable dynamics' if corruption_time is None else 'Active ion channel pumping required'}
    {'provide sufficient maintenance current.' if corruption_time is None else 'for long-term pattern stability.'}

    {'═' * 40}

    NEXT EXPERIMENT:
    Test with real planarian tissue using
    TTX + TEA to block active channels.
    Measure actual T_hold with DiBAC4(3).
    """

    ax4.text(0.5, 0.5, summary, transform=ax4.transAxes,
             fontsize=11, fontfamily='monospace',
             ha='center', va='center',
             bbox=dict(boxstyle='round', facecolor='#f0f0f0', edgecolor='black'))

    plt.suptitle(f'MAINTENANCE COST TEST: {memory_type}',
                fontsize=14, fontweight='bold')
    plt.tight_layout()

    output_path = SCRIPT_DIR / 'maintenance_cost_results.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved: {output_path}")

    plt.show()

    return T_hold, memory_type


if __name__ == "__main__":
    T_hold, memory_type = run_maintenance_test()

    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)
    print(f"""
    With current parameters:
    - Gap junction conductance: {G_GJ}
    - Leak conductance: {G_LEAK}
    - Bistable strength: {BISTABLE_STRENGTH}

    The memory is {memory_type}.

    This means in the lab you should expect:
    {'Pattern to persist even after blocking active channels.' if 'FLASH' in memory_type else 'Pattern to decay - need to measure actual T_hold.'}

    Test this with: TTX (blocks Na+) + TEA (blocks K+) application
    while monitoring with DiBAC4(3) fluorescence.
    """)
