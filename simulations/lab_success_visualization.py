#!/usr/bin/env python3
"""
Lab Success Visualization
=========================

What would you SEE in the lab when biological memory works?

This simulates the view through a fluorescence microscope with
DiBAC4(3) voltage-sensitive dye, showing:

1. A planarian body with voltage patterns
2. 4-bit memory encoded in different body regions
3. The READ/WRITE process as it would appear

DiBAC4(3) properties:
- BRIGHT (green/yellow) = Depolarized = -30mV = Logic 1
- DIM (dark) = Hyperpolarized = -60mV = Logic 0

This is what success looks like!
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, FancyBboxPatch
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.animation as animation
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.resolve()

# =============================================================================
# PLANARIAN BODY SHAPE
# =============================================================================
def create_planarian_mask(shape=(200, 100)):
    """Create a planarian-shaped mask."""
    h, w = shape
    y, x = np.ogrid[:h, :w]

    # Center coordinates
    cy, cx = h // 2, w // 2

    # Main body - elongated ellipse
    body = ((x - cx) / (w * 0.45))**2 + ((y - cy) / (h * 0.35))**2 < 1

    # Head (wider, rounded) - left side
    head_cx = w * 0.15
    head = ((x - head_cx) / (w * 0.18))**2 + ((y - cy) / (h * 0.42))**2 < 1

    # Tail (tapered) - right side
    tail_cx = w * 0.85
    tail_width = w * 0.12
    tail = ((x - tail_cx) / tail_width)**2 + ((y - cy) / (h * 0.25))**2 < 1

    # Auricles (ear-like projections on head)
    aur1_cy = cy - h * 0.25
    aur2_cy = cy + h * 0.25
    aur_cx = w * 0.08
    aur1 = ((x - aur_cx) / (w * 0.08))**2 + ((y - aur1_cy) / (h * 0.12))**2 < 1
    aur2 = ((x - aur_cx) / (w * 0.08))**2 + ((y - aur2_cy) / (h * 0.12))**2 < 1

    # Combine all parts
    planarian = body | head | tail | aur1 | aur2

    return planarian.astype(float)


def create_voltage_pattern(shape, bit_pattern, noise_level=0.05):
    """
    Create voltage pattern for 4-bit memory.

    Regions:
    - Bit 0: Head (left quarter)
    - Bit 1: Anterior trunk (center-left)
    - Bit 2: Posterior trunk (center-right)
    - Bit 3: Tail (right quarter)
    """
    h, w = shape
    voltage = np.zeros(shape)

    # Define regions (x-coordinate based for linear body plan)
    regions = [
        (0, w // 4),           # Head - Bit 0
        (w // 4, w // 2),      # Anterior - Bit 1
        (w // 2, 3 * w // 4),  # Posterior - Bit 2
        (3 * w // 4, w)        # Tail - Bit 3
    ]

    # Logic levels (normalized 0-1 for display)
    # 1.0 = depolarized (-30mV, bright)
    # 0.0 = hyperpolarized (-60mV, dim)

    for i, (x_start, x_end) in enumerate(regions):
        bit_value = bit_pattern[i]
        base_level = 0.85 if bit_value == 1 else 0.15
        voltage[:, x_start:x_end] = base_level

    # Add biological noise
    noise = np.random.randn(*shape) * noise_level
    voltage = np.clip(voltage + noise, 0, 1)

    return voltage


# =============================================================================
# FLUORESCENCE COLORMAP (DiBAC4(3)-like)
# =============================================================================
def create_dibac_colormap():
    """
    Create colormap mimicking DiBAC4(3) fluorescence.

    Dim (hyperpolarized) -> Bright green/yellow (depolarized)
    """
    colors = [
        (0.02, 0.02, 0.05),    # Very dark (hyperpolarized)
        (0.05, 0.15, 0.05),    # Dark green
        (0.1, 0.4, 0.1),       # Green
        (0.3, 0.7, 0.2),       # Bright green
        (0.6, 0.9, 0.3),       # Yellow-green
        (0.9, 1.0, 0.4),       # Bright yellow (strongly depolarized)
    ]
    return LinearSegmentedColormap.from_list('dibac', colors)


# =============================================================================
# STATIC VISUALIZATION - "SUCCESS" IMAGE
# =============================================================================
def create_success_image(bit_pattern=[1, 0, 1, 0]):
    """Create the 'success' image - what you'd see in the lab."""

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.patch.set_facecolor('black')

    shape = (200, 400)
    mask = create_planarian_mask((shape[0], shape[1]))
    voltage = create_voltage_pattern(shape, bit_pattern)

    # Apply mask
    display = voltage * mask
    display[mask == 0] = np.nan  # Transparent background

    cmap = create_dibac_colormap()

    # --- Panel 1: The planarian with voltage pattern ---
    ax1 = axes[0, 0]
    ax1.set_facecolor('black')
    im = ax1.imshow(display, cmap=cmap, vmin=0, vmax=1, aspect='equal')
    ax1.set_title(f'FLUORESCENCE MICROSCOPY VIEW\nPattern: {bit_pattern}',
                  color='white', fontsize=14, fontweight='bold')
    ax1.axis('off')

    # Add region labels
    region_labels = ['HEAD\n(Bit 0)', 'ANTERIOR\n(Bit 1)', 'POSTERIOR\n(Bit 2)', 'TAIL\n(Bit 3)']
    x_positions = [50, 150, 250, 350]
    for i, (label, x) in enumerate(zip(region_labels, x_positions)):
        color = '#90EE90' if bit_pattern[i] == 1 else '#404040'
        ax1.text(x, 220, label, ha='center', va='top', fontsize=10,
                 color=color, fontweight='bold')

    # --- Panel 2: Voltage trace across body ---
    ax2 = axes[0, 1]
    ax2.set_facecolor('#1a1a2e')

    # Take middle row voltage trace
    mid_row = shape[0] // 2
    x_coords = np.arange(shape[1])
    voltage_trace = voltage[mid_row, :] * mask[mid_row, :]

    # Convert to mV
    voltage_mv = -60 + voltage_trace * 30  # Map 0-1 to -60 to -30 mV

    ax2.fill_between(x_coords, -70, voltage_mv, alpha=0.3, color='lime')
    ax2.plot(x_coords, voltage_mv, 'lime', linewidth=2)
    ax2.axhline(y=-45, color='yellow', linestyle='--', alpha=0.5, label='Threshold')
    ax2.axhline(y=-30, color='red', linestyle=':', alpha=0.5, label='Logic 1')
    ax2.axhline(y=-60, color='blue', linestyle=':', alpha=0.5, label='Logic 0')

    ax2.set_xlim(0, shape[1])
    ax2.set_ylim(-70, -20)
    ax2.set_xlabel('Position (Head → Tail)', color='white')
    ax2.set_ylabel('Membrane Potential (mV)', color='white')
    ax2.set_title('VOLTAGE PROFILE ALONG BODY AXIS', color='white', fontsize=12, fontweight='bold')
    ax2.tick_params(colors='white')
    ax2.legend(loc='upper right', facecolor='#2a2a3e', labelcolor='white')
    ax2.grid(True, alpha=0.2)

    # Region boundaries
    for x in [100, 200, 300]:
        ax2.axvline(x=x, color='white', linestyle='-', alpha=0.3)

    # --- Panel 3: Bit readout display ---
    ax3 = axes[1, 0]
    ax3.set_facecolor('#0a0a1a')
    ax3.axis('off')

    # Create digital display
    display_text = "BIOLOGICAL MEMORY READOUT\n"
    display_text += "═" * 30 + "\n\n"

    for i, bit in enumerate(bit_pattern):
        region_names = ['HEAD', 'ANTERIOR', 'POSTERIOR', 'TAIL']
        status = "██ DEPOLARIZED" if bit == 1 else "░░ HYPERPOLARIZED"
        display_text += f"  BIT {i} ({region_names[i]:>9}): {bit}  {status}\n"

    display_text += "\n" + "═" * 30 + "\n"
    binary_str = ''.join(str(b) for b in bit_pattern)
    display_text += f"\n  BINARY: {binary_str}\n"
    display_text += f"  DECIMAL: {int(binary_str, 2)}\n"
    display_text += f"\n  STATUS: ✓ STABLE PATTERN DETECTED"

    ax3.text(0.5, 0.5, display_text, transform=ax3.transAxes,
             fontsize=12, fontfamily='monospace', color='#00ff00',
             ha='center', va='center',
             bbox=dict(boxstyle='round', facecolor='#0a0a1a', edgecolor='#00ff00', linewidth=2))

    # --- Panel 4: Explanation ---
    ax4 = axes[1, 1]
    ax4.set_facecolor('#1a1a2e')
    ax4.axis('off')

    explanation = """
    WHAT YOU'RE SEEING
    ══════════════════

    This is a planarian viewed through a
    fluorescence microscope with DiBAC4(3)
    voltage-sensitive dye.

    BRIGHT REGIONS (yellow-green):
    • Depolarized cells (~-30 mV)
    • Representing Logic "1"
    • Ion channels OPEN

    DIM REGIONS (dark):
    • Hyperpolarized cells (~-60 mV)
    • Representing Logic "0"
    • Ion channels CLOSED

    THE PATTERN PERSISTS because:
    • Cells are BISTABLE
    • Gap junctions are GATED
    • Each region votes by majority

    This IS biological memory!
    """

    ax4.text(0.5, 0.5, explanation, transform=ax4.transAxes,
             fontsize=11, fontfamily='monospace', color='white',
             ha='center', va='center',
             bbox=dict(boxstyle='round', facecolor='#2a2a3e', edgecolor='#00aaff', linewidth=2))

    plt.suptitle('SUCCESS IN THE LAB: 4-BIT BIOLOGICAL MEMORY IN A PLANARIAN',
                 fontsize=16, fontweight='bold', color='white', y=0.98)

    plt.tight_layout()
    return fig


# =============================================================================
# ANIMATED VISUALIZATION - WRITE/STORE/READ CYCLE
# =============================================================================
def create_memory_cycle_animation():
    """Create animation showing Write → Store → Read cycle."""

    fig, ax = plt.subplots(figsize=(12, 8))
    fig.patch.set_facecolor('black')
    ax.set_facecolor('black')

    shape = (200, 400)
    mask = create_planarian_mask((shape[0], shape[1]))
    cmap = create_dibac_colormap()

    # Animation frames
    n_frames = 120  # 4 seconds at 30 fps

    # Phases:
    # 0-30: Initial state (all 0)
    # 30-60: WRITE operation (pattern appears bit by bit)
    # 60-90: STORE (pattern stable with slight noise)
    # 90-120: READ (voltage sweep for readout)

    def animate(frame):
        ax.clear()
        ax.set_facecolor('black')

        if frame < 30:
            # Initial - all hyperpolarized
            phase = "INITIAL STATE"
            bit_pattern = [0, 0, 0, 0]
            noise = 0.02
        elif frame < 45:
            # Writing bit 0
            phase = "WRITE: Setting HEAD (Bit 0)"
            bit_pattern = [1, 0, 0, 0]
            noise = 0.08
        elif frame < 52:
            # Writing bit 1 (stays 0)
            phase = "WRITE: ANTERIOR (Bit 1) = 0"
            bit_pattern = [1, 0, 0, 0]
            noise = 0.05
        elif frame < 60:
            # Writing bit 2
            phase = "WRITE: Setting POSTERIOR (Bit 2)"
            bit_pattern = [1, 0, 1, 0]
            noise = 0.08
        elif frame < 67:
            # Writing bit 3 (stays 0)
            phase = "WRITE: TAIL (Bit 3) = 0"
            bit_pattern = [1, 0, 1, 0]
            noise = 0.05
        elif frame < 100:
            # Storing - pattern stable
            phase = "STORE: Pattern Stable (Gap Junctions Gated)"
            bit_pattern = [1, 0, 1, 0]
            noise = 0.03 + 0.02 * np.sin(frame / 5)  # Slight biological fluctuation
        else:
            # Reading
            phase = "READ: Scanning Pattern..."
            bit_pattern = [1, 0, 1, 0]
            noise = 0.03

        voltage = create_voltage_pattern(shape, bit_pattern, noise)
        display = voltage * mask
        display[mask == 0] = np.nan

        ax.imshow(display, cmap=cmap, vmin=0, vmax=1, aspect='equal')

        # Phase indicator
        ax.text(200, -15, phase, ha='center', va='bottom',
                fontsize=14, fontweight='bold', color='cyan')

        # Time indicator
        time_s = frame / 30
        ax.text(380, 210, f't = {time_s:.1f}s', ha='right', va='top',
                fontsize=10, color='white')

        # Bit display
        bit_str = ' '.join(str(b) for b in bit_pattern)
        ax.text(200, 220, f'Pattern: [{bit_str}]', ha='center', va='top',
                fontsize=12, color='lime', fontweight='bold')

        # Read indicator (scanning line)
        if frame >= 100:
            scan_x = 20 + (frame - 100) * 18
            ax.axvline(x=scan_x, color='cyan', linewidth=2, alpha=0.7)

        ax.axis('off')
        ax.set_xlim(-10, 410)
        ax.set_ylim(230, -30)

        return []

    anim = animation.FuncAnimation(fig, animate, frames=n_frames,
                                   interval=33, blit=True)
    return fig, anim


# =============================================================================
# MAIN - GENERATE ALL VISUALIZATIONS
# =============================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("LAB SUCCESS VISUALIZATION")
    print("=" * 70)
    print("\nGenerating what you would SEE in the lab...")

    # Generate static success image
    print("\n1. Creating success image...")
    fig1 = create_success_image([1, 0, 1, 0])
    output1 = SCRIPT_DIR / 'lab_success_view.png'
    fig1.savefig(output1, dpi=150, bbox_inches='tight', facecolor='black')
    print(f"   Saved: {output1}")

    # Generate alternate patterns
    print("\n2. Creating alternate patterns...")
    patterns = [
        [0, 0, 0, 0],  # All 0
        [1, 1, 1, 1],  # All 1
        [1, 0, 0, 1],  # Ends bright
        [0, 1, 1, 0],  # Middle bright
    ]

    fig2, axes2 = plt.subplots(2, 2, figsize=(14, 10))
    fig2.patch.set_facecolor('black')

    shape = (200, 400)
    mask = create_planarian_mask((shape[0], shape[1]))
    cmap = create_dibac_colormap()

    for idx, pattern in enumerate(patterns):
        ax = axes2.flatten()[idx]
        ax.set_facecolor('black')

        voltage = create_voltage_pattern(shape, pattern, noise_level=0.04)
        display = voltage * mask
        display[mask == 0] = np.nan

        ax.imshow(display, cmap=cmap, vmin=0, vmax=1, aspect='equal')
        binary_str = ''.join(str(b) for b in pattern)
        ax.set_title(f'Pattern: {pattern}\nBinary: {binary_str} = Decimal {int(binary_str, 2)}',
                     color='white', fontsize=12, fontweight='bold')
        ax.axis('off')

    plt.suptitle('DIFFERENT MEMORY PATTERNS UNDER FLUORESCENCE MICROSCOPY',
                 fontsize=14, fontweight='bold', color='white', y=0.98)
    plt.tight_layout()

    output2 = SCRIPT_DIR / 'lab_pattern_gallery.png'
    fig2.savefig(output2, dpi=150, bbox_inches='tight', facecolor='black')
    print(f"   Saved: {output2}")

    # Summary
    print("\n" + "=" * 70)
    print("WHAT YOU WOULD SEE IN THE LAB")
    print("=" * 70)
    print("""
    Under the fluorescence microscope with DiBAC4(3) dye:

    1. BRIGHT REGIONS (yellow-green glow):
       • These cells are DEPOLARIZED (~-30 mV)
       • Representing binary "1"
       • More dye accumulates in depolarized cells

    2. DIM REGIONS (dark/faint):
       • These cells are HYPERPOLARIZED (~-60 mV)
       • Representing binary "0"
       • Dye is excluded from hyperpolarized cells

    3. THE PATTERN IS STABLE:
       • Watch for several minutes
       • The bright/dim pattern persists!
       • This is BIOLOGICAL MEMORY in action

    4. TO WRITE NEW PATTERN:
       • Apply ion channel modulators to specific regions
       • Watch the fluorescence change in real-time
       • New pattern becomes stable

    5. TO READ THE PATTERN:
       • Simply image with fluorescence microscopy
       • Segment into regions
       • Threshold to determine 0 or 1

    SUCCESS = Stable, distinguishable patterns that persist
              and can be written/read reliably!
    """)

    print("\n" + "=" * 70)
    print("Visualization complete!")
    print("=" * 70)

    plt.show()
