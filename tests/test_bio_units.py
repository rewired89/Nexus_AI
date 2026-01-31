"""Tests for Biological Information Module (BIM) framework.

Covers:
  - Nernst equation calculations
  - Ion profiles and equilibrium potentials
  - Switching energy (E_bit)
  - State stability (T_half)
  - Channel noise error rate
  - Shannon entropy
  - Channel capacity (Shannon-Hartley)
  - BiologicalBit builder
  - Hardware Specification Library
"""

from __future__ import annotations

import math

import pytest

from acheron.rag.bio_units import (
    CPU_NAV_KV,
    HARDWARE_LIBRARY,
    PHYSARUM,
    PLANARIAN_NEOBLAST_HEURISTIC,
    RAM_VMEM_GRADIENT,
    SSD_INNEXIN_CONNECTIVITY,
    XENOPUS_OOCYTE,
    BiologicalBit,
    HardwareComponent,
    IonProfile,
    build_biological_bit,
    calc_channel_capacity,
    calc_channel_noise_error_rate,
    calc_shannon_entropy,
    calc_state_stability,
    calc_switching_energy,
    format_hardware_library,
    nernst_mv,
    nernst_potential,
)


# ======================================================================
# Nernst Equation tests
# ======================================================================
class TestNernst:
    """Tests for Nernst equilibrium potential calculations."""

    def test_nernst_k_typical(self):
        """K+ with 120 mM in, 3 mM out → large negative potential."""
        e_mv = nernst_mv(z=+1, conc_out=3.0, conc_in=120.0)
        # At 20°C: (8.314*293.15)/(1*96485)*ln(3/120)*1000 ≈ -93 mV
        assert -100 < e_mv < -85

    def test_nernst_na_typical(self):
        """Na+ with 12 mM in, 110 mM out → positive potential."""
        e_mv = nernst_mv(z=+1, conc_out=110.0, conc_in=12.0)
        assert e_mv > 50

    def test_nernst_cl_typical(self):
        """Cl- (z=-1) with 4 mM out, 20 mM in → positive result for Cl-."""
        e_mv = nernst_mv(z=-1, conc_out=4.0, conc_in=20.0)
        # E_Cl = -(RT/F)*ln(out/in) = (RT/F)*ln(in/out) > 0
        assert e_mv > 0

    def test_nernst_ca_divalent(self):
        """Ca2+ (z=+2) should give ~half the voltage swing of monovalent."""
        e_ca = nernst_mv(z=+2, conc_out=1.5, conc_in=0.0001)
        # Very large ratio → large positive potential, but divided by 2
        assert e_ca > 100

    def test_nernst_equal_concentrations(self):
        """Equal concentrations → 0 mV."""
        e_mv = nernst_mv(z=+1, conc_out=10.0, conc_in=10.0)
        assert abs(e_mv) < 0.001

    def test_nernst_volts_vs_mv(self):
        """nernst_mv should be 1000x nernst_potential."""
        e_v = nernst_potential(z=+1, conc_out=3.0, conc_in=120.0)
        e_mv = nernst_mv(z=+1, conc_out=3.0, conc_in=120.0)
        assert abs(e_mv - e_v * 1000.0) < 0.001

    def test_nernst_zero_concentration_raises(self):
        """Zero concentration should raise ValueError."""
        with pytest.raises(ValueError, match="positive"):
            nernst_potential(z=+1, conc_out=0.0, conc_in=10.0)
        with pytest.raises(ValueError, match="positive"):
            nernst_potential(z=+1, conc_out=10.0, conc_in=0.0)

    def test_nernst_negative_concentration_raises(self):
        """Negative concentration should raise ValueError."""
        with pytest.raises(ValueError, match="positive"):
            nernst_potential(z=+1, conc_out=-1.0, conc_in=10.0)

    def test_nernst_zero_valence_raises(self):
        """Zero valence should raise ValueError."""
        with pytest.raises(ValueError, match="non-zero"):
            nernst_potential(z=0, conc_out=10.0, conc_in=10.0)

    def test_nernst_temperature_effect(self):
        """Higher temperature should increase the magnitude."""
        e_cold = abs(nernst_mv(z=+1, conc_out=3.0, conc_in=120.0, temp_k=280.0))
        e_warm = abs(nernst_mv(z=+1, conc_out=3.0, conc_in=120.0, temp_k=310.0))
        assert e_warm > e_cold


# ======================================================================
# Ion Profile tests
# ======================================================================
class TestIonProfile:
    """Tests for ion concentration profiles and equilibrium potentials."""

    def test_xenopus_e_k(self):
        """Xenopus oocyte K+ should be strongly negative."""
        e_k = XENOPUS_OOCYTE.e_k()
        assert -110 < e_k < -80

    def test_xenopus_e_na(self):
        """Xenopus oocyte Na+ should be positive."""
        e_na = XENOPUS_OOCYTE.e_na()
        assert e_na > 40

    def test_planarian_heuristic_e_k(self):
        """Planarian neoblast (heuristic) K+ should be negative."""
        e_k = PLANARIAN_NEOBLAST_HEURISTIC.e_k()
        assert e_k < -80

    def test_planarian_freshwater_na(self):
        """In freshwater planarian, Na+ reversal should be low/negative."""
        e_na = PLANARIAN_NEOBLAST_HEURISTIC.e_na()
        # na_out=5 < na_in=12, so E_Na should be negative
        assert e_na < 0

    def test_physarum_e_k(self):
        """Physarum K+ should be strongly negative (high ratio)."""
        e_k = PHYSARUM.e_k()
        assert e_k < -100

    def test_ion_profile_e_ca(self):
        """Ca2+ equilibrium potential should be positive (high out/in)."""
        e_ca = XENOPUS_OOCYTE.e_ca()
        assert e_ca > 100

    def test_ion_profile_e_ca_zero(self):
        """If ca_in or ca_out is 0, should return 0.0 safely."""
        profile = IonProfile(
            organism="test", cell_type="test",
            k_in=120, k_out=3, na_in=10, na_out=110,
            cl_in=20, cl_out=4, ca_in=0, ca_out=0,
        )
        assert profile.e_ca() == 0.0

    def test_profiles_have_sources(self):
        """All standard profiles should have non-empty source strings."""
        assert XENOPUS_OOCYTE.source != ""
        assert PLANARIAN_NEOBLAST_HEURISTIC.source != ""
        assert PHYSARUM.source != ""


# ======================================================================
# Switching Energy tests
# ======================================================================
class TestSwitchingEnergy:
    """Tests for E_bit and ATP cost calculations."""

    def test_positive_energy(self):
        """Switching energy should always be positive."""
        e_bit, atp = calc_switching_energy(30.0, 10000)
        assert e_bit > 0
        assert atp > 0

    def test_larger_voltage_more_energy(self):
        """Larger voltage swing → more energy."""
        e_low, _ = calc_switching_energy(10.0, 10000)
        e_high, _ = calc_switching_energy(50.0, 10000)
        assert e_high > e_low

    def test_more_ions_more_energy(self):
        """More ions per event → more energy (with small capacitance so ion energy dominates)."""
        e_few, _ = calc_switching_energy(30.0, 1000, capacitance_pf=0.001)
        e_many, _ = calc_switching_energy(30.0, 100000, capacitance_pf=0.001)
        assert e_many > e_few

    def test_atp_cost_reasonable(self):
        """ATP per flip should be in a physically reasonable range.

        A 30 mV swing across 10 pF capacitance stores ~4.5e-15 J.
        With ATP_ENERGY ~5e-20 J/ATP, that's ~90k ATP — realistic for
        a full membrane capacitor discharge/recharge cycle.
        """
        _, atp = calc_switching_energy(30.0, 10000, 10.0)
        assert 0.01 < atp < 200000

    def test_zero_voltage_minimal_energy(self):
        """Zero voltage swing → minimal capacitive energy."""
        e_bit, atp = calc_switching_energy(0.0, 0)
        assert e_bit == 0.0
        assert atp == 0.0

    def test_capacitance_effect(self):
        """Larger capacitance → more capacitive energy."""
        e_small, _ = calc_switching_energy(30.0, 100, capacitance_pf=5.0)
        e_large, _ = calc_switching_energy(30.0, 100, capacitance_pf=50.0)
        assert e_large >= e_small


# ======================================================================
# State Stability tests
# ======================================================================
class TestStateStability:
    """Tests for T_half (RC time constant) calculations."""

    def test_positive_t_half(self):
        """T_half should always be positive."""
        t_half = calc_state_stability(500.0, 10.0)
        assert t_half > 0

    def test_higher_resistance_longer(self):
        """Higher membrane resistance → longer T_half."""
        t_low = calc_state_stability(100.0, 10.0)
        t_high = calc_state_stability(1000.0, 10.0)
        assert t_high > t_low

    def test_higher_capacitance_longer(self):
        """Higher membrane capacitance → longer T_half."""
        t_low = calc_state_stability(500.0, 5.0)
        t_high = calc_state_stability(500.0, 50.0)
        assert t_high > t_low

    def test_t_half_formula(self):
        """Verify T_half = R_m * C_m * ln(2)."""
        r_mohm = 500.0
        c_pf = 10.0
        t_half = calc_state_stability(r_mohm, c_pf)
        expected = (r_mohm * 1e6) * (c_pf * 1e-12) * math.log(2)
        assert abs(t_half - expected) < 1e-12

    def test_typical_value_range(self):
        """Typical neuron-like params: T_half should be ~ms range."""
        t_half = calc_state_stability(500.0, 10.0)
        # 500 MOhm * 10 pF = 5e-3 s = 5 ms; T_half = 5ms * ln2 ≈ 3.47 ms
        assert 0.001 < t_half < 0.01  # 1-10 ms range


# ======================================================================
# Channel Noise Error Rate tests
# ======================================================================
class TestChannelNoiseErrorRate:
    """Tests for stochastic bit-flip probability."""

    def test_low_error_rate(self):
        """With many channels and low p_open, error should be small."""
        error = calc_channel_noise_error_rate(
            n_channels=100, p_open_resting=0.01, p_open_threshold=0.10
        )
        assert 0.0 <= error < 0.5

    def test_threshold_below_resting_gives_one(self):
        """If threshold <= resting, a flip is certain."""
        error = calc_channel_noise_error_rate(
            n_channels=100, p_open_resting=0.10, p_open_threshold=0.05
        )
        assert error == 1.0

    def test_more_channels_lower_error(self):
        """More channels → law of large numbers → lower noise error."""
        error_few = calc_channel_noise_error_rate(n_channels=10)
        error_many = calc_channel_noise_error_rate(n_channels=500)
        assert error_many <= error_few

    def test_invalid_p_open_returns_zero(self):
        """p_open <= 0 or >= 1 should return 0.0."""
        assert calc_channel_noise_error_rate(p_open_resting=0.0) == 0.0
        assert calc_channel_noise_error_rate(p_open_resting=1.0) == 0.0

    def test_longer_window_higher_error(self):
        """Longer observation window → higher cumulative error probability."""
        error_short = calc_channel_noise_error_rate(observation_window_s=0.1)
        error_long = calc_channel_noise_error_rate(observation_window_s=10.0)
        assert error_long >= error_short

    def test_error_bounded_zero_one(self):
        """Error rate must always be in [0, 1]."""
        error = calc_channel_noise_error_rate(
            n_channels=5, p_open_resting=0.05, p_open_threshold=0.10,
            observation_window_s=100.0, switching_rate_hz=10000.0,
        )
        assert 0.0 <= error <= 1.0


# ======================================================================
# Shannon Entropy tests
# ======================================================================
class TestShannonEntropy:
    """Tests for information content calculations."""

    def test_binary_states(self):
        """Two equiprobable states → 1 bit."""
        h = calc_shannon_entropy(2)
        assert abs(h - 1.0) < 0.001

    def test_four_states(self):
        """Four equiprobable states → 2 bits."""
        h = calc_shannon_entropy(4)
        assert abs(h - 2.0) < 0.001

    def test_single_state_zero(self):
        """One state → 0 bits (no information)."""
        h = calc_shannon_entropy(1)
        assert h == 0.0

    def test_zero_states_zero(self):
        """Zero states → 0 bits."""
        h = calc_shannon_entropy(0)
        assert h == 0.0

    def test_non_uniform_lower(self):
        """Non-uniform distribution → less than max entropy."""
        h_uniform = calc_shannon_entropy(4)
        h_skewed = calc_shannon_entropy(4, [0.7, 0.1, 0.1, 0.1])
        assert h_skewed < h_uniform

    def test_custom_probabilities(self):
        """Custom probabilities should give known result."""
        # H([0.5, 0.5]) = 1 bit
        h = calc_shannon_entropy(2, [0.5, 0.5])
        assert abs(h - 1.0) < 0.001

    def test_certain_state_zero_entropy(self):
        """One probability = 1.0 → 0 bits."""
        h = calc_shannon_entropy(3, [1.0, 0.0, 0.0])
        assert abs(h) < 0.001


# ======================================================================
# Channel Capacity tests
# ======================================================================
class TestChannelCapacity:
    """Tests for Shannon-Hartley channel capacity."""

    def test_positive_capacity(self):
        """Normal parameters → positive capacity."""
        c = calc_channel_capacity(100.0, 10.0)
        assert c > 0

    def test_higher_bandwidth_more_capacity(self):
        """More bandwidth → higher capacity."""
        c_low = calc_channel_capacity(50.0, 10.0)
        c_high = calc_channel_capacity(200.0, 10.0)
        assert c_high > c_low

    def test_higher_snr_more_capacity(self):
        """Higher SNR → higher capacity."""
        c_low = calc_channel_capacity(100.0, 5.0)
        c_high = calc_channel_capacity(100.0, 50.0)
        assert c_high > c_low

    def test_zero_bandwidth_zero(self):
        """Zero bandwidth → zero capacity."""
        assert calc_channel_capacity(0.0, 10.0) == 0.0

    def test_zero_snr_zero(self):
        """Zero SNR → zero capacity."""
        assert calc_channel_capacity(100.0, 0.0) == 0.0

    def test_formula(self):
        """Verify C = B * log2(1 + SNR)."""
        b = 100.0
        snr = 10.0
        c = calc_channel_capacity(b, snr)
        expected = b * math.log2(1 + snr)
        assert abs(c - expected) < 0.001


# ======================================================================
# BiologicalBit Builder tests
# ======================================================================
class TestBiologicalBitBuilder:
    """Tests for the complete BiologicalBit builder."""

    def test_default_build(self):
        """Default parameters should produce a valid BiologicalBit."""
        bb = build_biological_bit()
        assert isinstance(bb, BiologicalBit)
        assert bb.t_half_seconds > 0
        assert bb.e_bit_joules > 0
        assert bb.atp_per_flip > 0
        assert bb.ions_per_flip == 10000
        assert 0.0 <= bb.bit_flip_probability <= 1.0
        assert bb.shannon_entropy_bits > 0
        assert bb.channel_capacity_bits_per_s > 0
        assert bb.noise_source != ""

    def test_custom_build(self):
        """Custom parameters should be reflected in the result."""
        bb = build_biological_bit(
            delta_v_mv=50.0,
            ions_per_event=20000,
            n_stable_states=8,
        )
        assert bb.ions_per_flip == 20000
        assert bb.shannon_entropy_bits == pytest.approx(3.0, abs=0.01)

    def test_t_half_label_format(self):
        """T_half label should be a human-readable string."""
        bb = build_biological_bit()
        assert bb.t_half_label != ""
        # Should contain a unit
        assert any(u in bb.t_half_label for u in ["μs", "ms", "s", "min", "h"])

    def test_different_resistance_changes_t_half(self):
        """Higher membrane resistance → longer T_half."""
        bb_low = build_biological_bit(membrane_resistance_mohm=100.0)
        bb_high = build_biological_bit(membrane_resistance_mohm=2000.0)
        assert bb_high.t_half_seconds > bb_low.t_half_seconds


# ======================================================================
# Hardware Specification Library tests
# ======================================================================
class TestHardwareLibrary:
    """Tests for the Hardware Specification Library."""

    def test_library_has_three_components(self):
        """Library should contain CPU, RAM, SSD."""
        assert "cpu" in HARDWARE_LIBRARY
        assert "ram" in HARDWARE_LIBRARY
        assert "ssd" in HARDWARE_LIBRARY
        assert len(HARDWARE_LIBRARY) == 3

    def test_cpu_component(self):
        """CPU component should be Nav/Kv channel array."""
        cpu = CPU_NAV_KV
        assert isinstance(cpu, HardwareComponent)
        assert "Nav" in cpu.name
        assert "CPU" in cpu.digital_equivalent
        assert len(cpu.key_molecules) >= 3
        assert cpu.evidence_level != ""

    def test_ram_component(self):
        """RAM component should be Vmem gradient."""
        ram = RAM_VMEM_GRADIENT
        assert isinstance(ram, HardwareComponent)
        assert "Vmem" in ram.name
        assert "RAM" in ram.digital_equivalent
        assert len(ram.key_molecules) >= 3

    def test_ssd_component(self):
        """SSD component should be Innexin connectivity."""
        ssd = SSD_INNEXIN_CONNECTIVITY
        assert isinstance(ssd, HardwareComponent)
        assert "Innexin" in ssd.name
        assert "SSD" in ssd.digital_equivalent
        assert "non-volatile" in ssd.persistence.lower()

    def test_all_components_have_evidence(self):
        """Every component should declare an evidence level."""
        for key, comp in HARDWARE_LIBRARY.items():
            assert comp.evidence_level != "", f"{key} missing evidence_level"

    def test_format_hardware_library(self):
        """format_hardware_library should produce readable text."""
        text = format_hardware_library()
        assert "ACHERON HARDWARE SPECIFICATION LIBRARY" in text
        assert "[CPU]" in text
        assert "[RAM]" in text
        assert "[SSD]" in text
        assert "Digital Equivalent:" in text
        assert "Key Molecules:" in text
