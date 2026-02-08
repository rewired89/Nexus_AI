#!/usr/bin/env python3
"""
CELL DIVISION HANDOFF TEST
==========================

THE QUESTION:
When a cell divides, how does the voltage "bit" get copied to daughters?

THE MODEL:
1. Start with 100 cells storing [1, 0, 1, 0]
2. At t=5s, cells "divide" - each becomes 2 cells
3. New daughter cells start at resting potential
4. Parent and daughter are connected by gap junction
5. Measure: Do daughters learn the parent's voltage state?

HYPOTHESES:
A) Gap Junction Teaching: Parent current flows to daughter, sets its state
B) Expression Inheritance: Daughter inherits channel expression (instant match)
C) No Inheritance: Daughter stays at resting potential (bit lost)

This answers: Is biological memory "copied" during cell division?
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.resolve()

# =============================================================================
# PARAMETERS
# =============================================================================

N_CELLS_INITIAL = 100
N_BITS = 4
CELLS_PER_BIT_INITIAL = N_CELLS_INITIAL // N_BITS

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

# Gap junctions
G_GJ_INTRA = 0.1     # Within existing cells
G_GJ_PARENT_DAUGHTER = 0.2   # Parent-daughter (newly divided)

# Noise
NOISE_STD = 1.5

# Timing
DT = 0.1
T_DIVISION = 5000    # Division occurs at 5 seconds
T_TOTAL = 20000      # 20 second simulation


# =============================================================================
# CELL CLASS
# =============================================================================

class Cell:
    def __init__(self, cell_id, bit_assignment, initial_V, is_daughter=False):
        self.cell_id = cell_id
        self.bit_assignment = bit_assignment
        self.V = initial_V
        self.is_daughter = is_daughter
        self.parent_id = None
        self.daughter_ids = []

    def bistable_current(self):
        sigmoid = 1.0 / (1.0 + np.exp(-(self.V - V_THRESHOLD) / V_SCALE))
        return BISTABLE_STRENGTH * (2 * sigmoid - 1) * (V_LOGIC_1 - V_LOGIC_0)

    def leak_current(self):
        return G_LEAK * (V_REST - self.V)


# =============================================================================
# SIMULATION
# =============================================================================

def run_division_test():
    print("=" * 70)
    print("CELL DIVISION HANDOFF TEST")
    print("=" * 70)

    INITIAL_PATTERN = [1, 0, 1, 0]
    print(f"\nInitial pattern: {INITIAL_PATTERN}")
    print(f"Cell division at: t = {T_DIVISION/1000:.1f} seconds")

    # Initialize parent cells
    cells = []
    cell_id = 0
    for bit_idx, bit_value in enumerate(INITIAL_PATTERN):
        for _ in range(CELLS_PER_BIT_INITIAL):
            if bit_value == 1:
                V_init = V_LOGIC_1 + np.random.randn() * 2
            else:
                V_init = V_LOGIC_0 + np.random.randn() * 2
            cells.append(Cell(cell_id, bit_idx, V_init, is_daughter=False))
            cell_id += 1

    print(f"Initial cells: {len(cells)}")

    # Recording
    n_steps = int(T_TOTAL / DT)
    record_interval = 50
    n_records = n_steps // record_interval

    time_record = []
    V_parent_mean = {i: [] for i in range(N_BITS)}
    V_daughter_mean = {i: [] for i in range(N_BITS)}
    n_cells_record = []

    division_happened = False
    n_daughters = 0

    print("\nRunning simulation...")

    for step in range(n_steps):
        t = step * DT

        # === DIVISION EVENT ===
        if t >= T_DIVISION and not division_happened:
            division_happened = True
            print(f"\n>>> CELL DIVISION at t = {t/1000:.1f}s <<<")

            # Each parent spawns a daughter
            new_cells = []
            for parent in cells:
                if not parent.is_daughter:  # Only original cells divide
                    # Daughter starts at resting potential (naive state)
                    daughter = Cell(
                        cell_id=cell_id,
                        bit_assignment=parent.bit_assignment,
                        initial_V=V_REST + np.random.randn() * 2,  # Starts naive
                        is_daughter=True
                    )
                    daughter.parent_id = parent.cell_id
                    parent.daughter_ids.append(daughter.cell_id)
                    new_cells.append(daughter)
                    cell_id += 1
                    n_daughters += 1

            cells.extend(new_cells)
            print(f"    Created {n_daughters} daughter cells")
            print(f"    Total cells now: {len(cells)}")

        # === CALCULATE CURRENTS ===

        # Build neighbor lists for gap junctions
        for cell in cells:
            I_total = cell.leak_current() + cell.bistable_current()

            # Gap junction from parent (if daughter)
            if cell.is_daughter and cell.parent_id is not None:
                parent = cells[cell.parent_id]
                I_gj = G_GJ_PARENT_DAUGHTER * (parent.V - cell.V)
                I_total += I_gj

            # Gap junction to daughters (if parent)
            for daughter_id in cell.daughter_ids:
                if daughter_id < len(cells):
                    daughter = cells[daughter_id]
                    I_gj = G_GJ_PARENT_DAUGHTER * (daughter.V - cell.V)
                    I_total += I_gj

            # Gap junction to same-bit neighbors (simplified: mean field)
            same_bit_cells = [c for c in cells if c.bit_assignment == cell.bit_assignment and c.cell_id != cell.cell_id]
            if same_bit_cells:
                V_mean = np.mean([c.V for c in same_bit_cells])
                I_gj = G_GJ_INTRA * (V_mean - cell.V)
                I_total += I_gj

            # Noise
            I_noise = np.random.randn() * NOISE_STD * np.sqrt(DT)

            # Update voltage
            cell.V += (I_total * DT / C_M) + I_noise

        # === RECORD ===
        if step % record_interval == 0:
            time_record.append(t / 1000)
            n_cells_record.append(len(cells))

            for bit_idx in range(N_BITS):
                # Parents
                parent_cells = [c for c in cells if c.bit_assignment == bit_idx and not c.is_daughter]
                if parent_cells:
                    V_parent_mean[bit_idx].append(np.mean([c.V for c in parent_cells]))
                else:
                    V_parent_mean[bit_idx].append(np.nan)

                # Daughters
                daughter_cells = [c for c in cells if c.bit_assignment == bit_idx and c.is_daughter]
                if daughter_cells:
                    V_daughter_mean[bit_idx].append(np.mean([c.V for c in daughter_cells]))
                else:
                    V_daughter_mean[bit_idx].append(np.nan)

        # Progress
        if step % (n_steps // 10) == 0:
            pattern = read_pattern(cells)
            print(f"  t = {t/1000:.1f}s, Cells: {len(cells)}, Pattern: {pattern}")

    # ==========================================================================
    # ANALYSIS
    # ==========================================================================

    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    # Final pattern check
    final_pattern = read_pattern(cells)
    parent_pattern = read_pattern([c for c in cells if not c.is_daughter])
    daughter_pattern = read_pattern([c for c in cells if c.is_daughter])

    print(f"\nFinal overall pattern: {final_pattern}")
    print(f"Parent cells pattern:  {parent_pattern}")
    print(f"Daughter cells pattern: {daughter_pattern}")

    # Check if daughters learned
    handoff_success = (daughter_pattern == INITIAL_PATTERN)

    if handoff_success:
        print("\n*** HANDOFF SUCCESSFUL ***")
        print("Daughter cells learned the pattern from parents!")
        result_type = "GAP JUNCTION TEACHING WORKS"
    else:
        print("\n*** HANDOFF FAILED ***")
        print("Daughter cells did NOT learn the pattern.")
        result_type = "REQUIRES EXPRESSION INHERITANCE"

    # ==========================================================================
    # PLOTTING
    # ==========================================================================

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    colors = ['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4']
    labels = ['Bit 0 (HEAD)', 'Bit 1 (ANT)', 'Bit 2 (POST)', 'Bit 3 (TAIL)']

    # --- Panel 1: Parent voltages ---
    ax1 = axes[0, 0]
    for bit_idx in range(N_BITS):
        ax1.plot(time_record, V_parent_mean[bit_idx],
                color=colors[bit_idx], linewidth=2, label=labels[bit_idx])
    ax1.axvline(x=T_DIVISION/1000, color='red', linestyle='--', linewidth=2, label='Division')
    ax1.axhline(y=V_THRESHOLD, color='gray', linestyle=':', alpha=0.5)
    ax1.set_xlabel('Time (seconds)')
    ax1.set_ylabel('Mean Voltage (mV)')
    ax1.set_title('PARENT CELL VOLTAGES')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)

    # --- Panel 2: Daughter voltages ---
    ax2 = axes[0, 1]
    for bit_idx in range(N_BITS):
        ax2.plot(time_record, V_daughter_mean[bit_idx],
                color=colors[bit_idx], linewidth=2, label=labels[bit_idx])
    ax2.axvline(x=T_DIVISION/1000, color='red', linestyle='--', linewidth=2, label='Division')
    ax2.axhline(y=V_THRESHOLD, color='gray', linestyle=':', alpha=0.5)
    ax2.axhline(y=V_REST, color='purple', linestyle=':', alpha=0.5, label='V_rest')
    ax2.set_xlabel('Time (seconds)')
    ax2.set_ylabel('Mean Voltage (mV)')
    ax2.set_title('DAUGHTER CELL VOLTAGES (start at V_rest)')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)

    # --- Panel 3: Cell count ---
    ax3 = axes[1, 0]
    ax3.plot(time_record, n_cells_record, 'b-', linewidth=2)
    ax3.axvline(x=T_DIVISION/1000, color='red', linestyle='--', linewidth=2)
    ax3.set_xlabel('Time (seconds)')
    ax3.set_ylabel('Total Cells')
    ax3.set_title('CELL POPULATION')
    ax3.grid(True, alpha=0.3)

    # --- Panel 4: Summary ---
    ax4 = axes[1, 1]
    ax4.axis('off')

    summary = f"""
    CELL DIVISION HANDOFF TEST
    {'═' * 40}

    Initial Pattern:     {INITIAL_PATTERN}
    Division Time:       t = {T_DIVISION/1000:.1f} s
    Cells After Division: {len(cells)}

    {'─' * 40}

    Parent Pattern (final):   {parent_pattern}
    Daughter Pattern (final): {daughter_pattern}

    {'─' * 40}

    RESULT: {result_type}

    {'─' * 40}

    {'Daughters successfully copied parent voltage state' if handoff_success else 'Daughters did NOT inherit voltage pattern'}
    {'via gap junction current flow.' if handoff_success else 'Expression-level inheritance may be required.'}

    {'═' * 40}

    MECHANISM:
    {'1. Parent maintains bistable state' if handoff_success else '1. Gap junction current insufficient'}
    {'2. GJ current flows to naive daughter' if handoff_success else '2. Daughter bistability pulls to nearest attractor'}
    {'3. Daughter adopts same attractor state' if handoff_success else '3. Without expression match, pattern lost'}
    """

    ax4.text(0.5, 0.5, summary, transform=ax4.transAxes,
             fontsize=10, fontfamily='monospace',
             ha='center', va='center',
             bbox=dict(boxstyle='round', facecolor='#f0f0f0', edgecolor='black'))

    plt.suptitle(f'CELL DIVISION HANDOFF: {result_type}',
                fontsize=14, fontweight='bold')
    plt.tight_layout()

    output_path = SCRIPT_DIR / 'cell_division_results.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved: {output_path}")

    plt.show()

    return handoff_success


def read_pattern(cells):
    """Read pattern from cell list."""
    pattern = []
    for bit_idx in range(N_BITS):
        bit_cells = [c for c in cells if c.bit_assignment == bit_idx]
        if bit_cells:
            V_mean = np.mean([c.V for c in bit_cells])
            pattern.append(1 if V_mean > V_THRESHOLD else 0)
        else:
            pattern.append(-1)  # Unknown
    return pattern


if __name__ == "__main__":
    success = run_division_test()

    print("\n" + "=" * 70)
    print("BIOLOGICAL INTERPRETATION")
    print("=" * 70)

    if success:
        print("""
    GAP JUNCTION TEACHING WORKS!

    This means:
    - When a cell divides, it stays connected to its daughter
    - The parent's bistable state creates current flow
    - The daughter is "taught" to adopt the same voltage
    - Pattern is inherited through electrical coupling

    In the lab:
    - Track dividing cells with live imaging
    - Monitor Vmem before/after division
    - Expect daughter to match parent within seconds
        """)
    else:
        print("""
    GAP JUNCTION TEACHING IS INSUFFICIENT

    This means:
    - Voltage alone doesn't reliably copy during division
    - Expression-level inheritance is likely required:
      * Transcription factors partitioned to daughters
      * Channel mRNAs copied during division
      * Epigenetic marks on ion channel genes

    In the lab:
    - Look for asymmetric channel distribution at cytokinesis
    - Test if blocking transcription prevents pattern inheritance
    - Check if daughter channel expression matches parent
        """)
