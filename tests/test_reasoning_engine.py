"""Tests for the Neuromorphic Reasoning Engine (acheron.reasoning)."""

from __future__ import annotations

import sys
sys.path.insert(0, "src")

import math

from acheron.reasoning.substrate import (
    ComputationalPrimitive,
    SubstrateEdge,
    SubstrateGraph,
    SubstrateNode,
    TopologyEventType,
)
from acheron.reasoning.mosaic import (
    analyze_topology,
    build_ring_lattice,
    build_small_world,
    build_uniform_random,
    compare_topologies,
    compute_clustering_coefficient,
    compute_average_path_length,
    compute_signal_propagation_latency,
    compute_energy_per_routing_event,
)
from acheron.reasoning.denram import (
    EncodingMode,
    PhaseEncoder,
    StaticAttractorModel,
    AttractorState,
    analyze_delay_encoding,
    simulate_phase_stability,
    compute_cross_talk_probability,
)
from acheron.reasoning.heterogeneity import (
    HeterogeneityParams,
    SubstrateConfig,
    build_substrate,
    compare_substrates,
    simulate_perturbation_recovery,
)
from acheron.reasoning.homeostatic import (
    HomeostaticParams,
    HomeostaticState,
    simulate_adversarial_recovery,
    simulate_repeated_perturbation,
)
from acheron.reasoning.state_space import (
    build_system_matrices,
    state_from_graph,
    analyze_stability,
    analyze_state_space,
    compute_dominant_eigenvalue,
    simulate_perturbation_trajectory,
)
from acheron.reasoning.ribozyme import (
    QT45Parameters,
    analyze_qt45,
    compute_redundancy_for_stability,
    compute_error_correction_overhead,
    compute_corruption_after_n_cycles,
    CrossChiralSubstrate,
    MolecularAccessModel,
)
from acheron.reasoning.freeze_thaw import (
    FreezeThawParams,
    analyze_freeze_thaw,
    strand_separation_probability,
    eutectic_concentration_factor,
    replication_stall_probability,
    optimize_freeze_thaw_interval,
)
from acheron.reasoning.validation import (
    generate_validation_report,
    format_report,
    report_to_dict,
    build_interaction_map,
    MODELING_LAYERS,
)


# ======================================================================
# Section 1: Substrate=Algorithm
# ======================================================================

class TestSubstrate:
    def test_graph_creation(self):
        g = SubstrateGraph()
        n = SubstrateNode(node_id="a", vm_mV=-40.0)
        event = g.add_node(n)
        assert g.node_count() == 1
        assert event.event_type == TopologyEventType.NODE_CREATE
        assert event.computational_primitive == ComputationalPrimitive.MEMORY_WRITE

    def test_edge_creation(self):
        g = SubstrateGraph()
        g.add_node(SubstrateNode(node_id="a"))
        g.add_node(SubstrateNode(node_id="b"))
        e = SubstrateEdge(source="a", target="b", conductance_nS=2.0)
        event = g.add_edge(e)
        assert g.edge_count() == 1
        assert event.computational_primitive == ComputationalPrimitive.MEMORY_WRITE

    def test_edge_modulation(self):
        g = SubstrateGraph()
        g.add_node(SubstrateNode(node_id="a"))
        g.add_node(SubstrateNode(node_id="b"))
        g.add_edge(SubstrateEdge(source="a", target="b", conductance_nS=2.0))
        event = g.modulate_edge("a", "b", 5.0)
        assert event is not None
        assert event.computational_primitive == ComputationalPrimitive.ROUTING_RECONFIG
        assert event.conductance_nS == 5.0

    def test_node_vm_change(self):
        g = SubstrateGraph()
        g.add_node(SubstrateNode(node_id="a", vm_mV=-40.0))
        event = g.set_node_vm("a", -20.0)
        assert event is not None
        assert event.computational_primitive == ComputationalPrimitive.STATE_TRANSITION
        assert event.delta_vm_mV == 20.0

    def test_adjacency_dict(self):
        g = SubstrateGraph()
        g.add_node(SubstrateNode(node_id="a"))
        g.add_node(SubstrateNode(node_id="b"))
        g.add_edge(SubstrateEdge(source="a", target="b", conductance_nS=3.0))
        adj = g.adjacency_dict()
        assert ("b", 3.0) in adj["a"]
        assert ("a", 3.0) in adj["b"]

    def test_event_log(self):
        g = SubstrateGraph()
        g.add_node(SubstrateNode(node_id="a"))
        g.add_node(SubstrateNode(node_id="b"))
        g.add_edge(SubstrateEdge(source="a", target="b"))
        g.set_node_vm("a", -20.0)
        assert len(g.event_log) == 4  # 2 node_create + 1 edge_create + 1 state_change


# ======================================================================
# Section 2: Mosaic Small-World
# ======================================================================

class TestMosaic:
    def test_ring_lattice(self):
        g = build_ring_lattice(20, 4)
        assert g.node_count() == 20
        assert g.edge_count() == 40  # 20 * 4/2

    def test_small_world(self):
        g = build_small_world(50, 6, rewire_prob=0.1, seed=42)
        assert g.node_count() == 50
        assert g.edge_count() == 150

    def test_uniform_random(self):
        g = build_uniform_random(30, 60, seed=42)
        assert g.node_count() == 30
        assert g.edge_count() == 60

    def test_clustering_coefficient(self):
        g = build_ring_lattice(20, 4)
        cc = compute_clustering_coefficient(g)
        assert 0.0 <= cc <= 1.0
        assert cc > 0.3  # ring lattice should have high clustering

    def test_average_path_length(self):
        g = build_ring_lattice(20, 4)
        apl = compute_average_path_length(g)
        assert apl > 0
        assert apl < 20  # can't be longer than diameter

    def test_small_world_vs_ring(self):
        ring = build_ring_lattice(50, 6)
        sw = build_small_world(50, 6, rewire_prob=0.1, seed=42)
        apl_ring = compute_average_path_length(ring)
        apl_sw = compute_average_path_length(sw)
        # Small-world should have shorter average path
        assert apl_sw <= apl_ring

    def test_signal_propagation(self):
        g = build_small_world(20, 4, seed=42)
        lat = compute_signal_propagation_latency(g, "c0", "c10")
        assert lat > 0
        assert lat < float("inf")

    def test_energy_per_routing_event(self):
        e = compute_energy_per_routing_event(2.0, delta_v_mV=30.0, duration_ms=1.0)
        assert e > 0

    def test_topology_metrics(self):
        g = build_small_world(30, 4, seed=42)
        m = analyze_topology(g, "small_world")
        assert m.node_count == 30
        assert m.clustering_coefficient > 0
        assert m.average_path_length > 0
        assert m.signal_propagation_latency_ms > 0
        assert m.topology_class == "small_world"

    def test_compare_topologies(self):
        comp = compare_topologies(n_nodes=30, k_neighbors=4, seed=42)
        assert comp.ring_lattice.node_count == 30
        assert comp.small_world.node_count == 30
        assert comp.uniform.node_count == 30


# ======================================================================
# Section 3: DenRAM Delay Encoding
# ======================================================================

class TestDenRAM:
    def test_phase_encoder(self):
        enc = PhaseEncoder(frequency_Hz=2.0, n_phase_bins=8)
        assert enc.period_ms == 500.0
        assert enc.phase_resolution_ms == 62.5
        for v in range(8):
            delay = enc.encode_value(v)
            decoded = enc.decode_phase(delay)
            assert decoded == v

    def test_phase_stability(self):
        enc = PhaseEncoder(frequency_Hz=1.0, n_phase_bins=4)
        stability = simulate_phase_stability(enc, noise_amplitude_mV=1.0, n_trials=500, seed=42)
        assert 0.0 <= stability <= 1.0
        assert stability > 0.5  # should mostly decode correctly with low noise

    def test_static_attractor_model(self):
        model = StaticAttractorModel(
            attractors=[
                AttractorState(state_id="low", vm_target_mV=-60.0, energy_barrier_kT=5.0),
                AttractorState(state_id="high", vm_target_mV=-20.0, energy_barrier_kT=5.0),
            ],
            noise_amplitude_mV=2.0,
        )
        for a in model.attractors:
            persistence = model.state_stability(a)
            assert persistence > 0
            escape = model.escape_probability(a)
            assert 0.0 <= escape <= 1.0

    def test_cross_talk_probability(self):
        p_close = compute_cross_talk_probability(5.0)
        p_far = compute_cross_talk_probability(100.0)
        assert p_close > p_far  # closer = more cross-talk

    def test_analyze_delay_encoding_phase(self):
        result = analyze_delay_encoding(EncodingMode.PHASE_DELAY, seed=42)
        assert result.encoding_mode == EncodingMode.PHASE_DELAY
        assert result.phase_stability_score >= 0

    def test_analyze_delay_encoding_static(self):
        result = analyze_delay_encoding(EncodingMode.STATIC_ATTRACTOR, seed=42)
        assert result.encoding_mode == EncodingMode.STATIC_ATTRACTOR


# ======================================================================
# Section 4: Heterogeneity
# ======================================================================

class TestHeterogeneity:
    def test_build_uniform_substrate(self):
        config = SubstrateConfig(n_nodes=20, k_neighbors=4, is_heterogeneous=False, seed=42)
        g = build_substrate(config)
        assert g.node_count() == 20
        # All nodes should have same Vm
        vms = [n.vm_mV for n in g.nodes.values()]
        assert len(set(vms)) == 1

    def test_build_heterogeneous_substrate(self):
        config = SubstrateConfig(
            n_nodes=20, k_neighbors=4, is_heterogeneous=True,
            heterogeneity=HeterogeneityParams(vm_rest_variance_mV=5.0),
            seed=42,
        )
        g = build_substrate(config)
        vms = [n.vm_mV for n in g.nodes.values()]
        # Should have variance
        assert len(set(vms)) > 1

    def test_perturbation_recovery(self):
        config = SubstrateConfig(n_nodes=30, k_neighbors=4, seed=42)
        g = build_substrate(config)
        result = simulate_perturbation_recovery(g, perturbation_mV=10.0, seed=42)
        assert result.recovery_steps > 0

    def test_compare_substrates(self):
        comp = compare_substrates(n_nodes=30, seed=42)
        assert comp.conclusion == "Natural disorder may increase distributed resilience."


# ======================================================================
# Section 5: Homeostatic Recovery
# ======================================================================

class TestHomeostatic:
    def test_homeostatic_state(self):
        g = build_small_world(20, 4, seed=42)
        state = HomeostaticState()
        state.initialize_from_graph(g)
        assert len(state.node_states) == 20
        assert 0.0 <= state.global_activity <= 1.0

    def test_adversarial_recovery(self):
        g = build_small_world(30, 4, seed=42)
        result = simulate_adversarial_recovery(g, bias_mV=20.0, seed=42)
        assert result.recovery_steps >= 0
        assert len(result.activity_trajectory) > 0

    def test_repeated_perturbation(self):
        g = build_small_world(30, 4, seed=42)
        result = simulate_repeated_perturbation(g, n_perturbations=3, seed=42)
        assert result.perturbation_count == 3
        assert len(result.activity_trajectory) > 0


# ======================================================================
# Section 6: State-Space Dynamics
# ======================================================================

class TestStateSpace:
    def test_system_matrices(self):
        g = build_small_world(10, 4, seed=42)
        matrices = build_system_matrices(g)
        assert matrices.dimension == 10
        assert len(matrices.A) == 10
        assert len(matrices.B) == 10

    def test_state_vector(self):
        g = build_small_world(10, 4, seed=42)
        X = state_from_graph(g)
        assert X.dimension == 10
        assert X.norm() > 0

    def test_dominant_eigenvalue(self):
        g = build_small_world(10, 4, seed=42)
        matrices = build_system_matrices(g, leak_rate=0.1)
        ev, vec = compute_dominant_eigenvalue(matrices.A)
        assert ev != 0
        assert len(vec) == 10

    def test_stability_analysis(self):
        g = build_small_world(10, 4, seed=42)
        matrices = build_system_matrices(g, leak_rate=0.2)
        result = analyze_stability(matrices)
        assert len(result.eigenvalues) > 0

    def test_state_space_analysis(self):
        g = build_small_world(15, 4, seed=42)
        result = analyze_state_space(g)
        assert result.system_dimension == 15
        assert len(result.stability.eigenvalues) > 0
        assert len(result.attractors) > 0

    def test_perturbation_trajectory(self):
        g = build_small_world(10, 4, seed=42)
        matrices = build_system_matrices(g, leak_rate=0.1)
        perturbation = [10.0] * 3 + [0.0] * 7
        result = simulate_perturbation_trajectory(matrices, g, perturbation)
        assert result.recovery_steps > 0
        assert len(result.trajectory_norms) > 0


# ======================================================================
# Section 7: QT45 Ribozyme
# ======================================================================

class TestRibozyme:
    def test_qt45_parameters(self):
        p = QT45Parameters()
        assert p.length_nt == 45
        assert p.fidelity_per_nt == 0.941
        assert p.copy_cycle_days == 72
        assert p.yield_fraction == 0.002
        assert 0 < p.per_copy_fidelity < 1
        assert p.per_copy_error_rate > 0

    def test_redundancy(self):
        result = compute_redundancy_for_stability()
        assert result.copies_needed > 0
        assert result.n_copy_cycles > 0

    def test_error_correction(self):
        result = compute_error_correction_overhead()
        assert result.parity_nucleotides_needed > 0
        assert result.effective_payload_nt < 45
        assert result.overhead_fraction > 0

    def test_corruption(self):
        result = compute_corruption_after_n_cycles(5, redundancy=10)
        assert 0.0 <= result.p_corruption_single_copy <= 1.0
        assert result.expected_intact_copies >= 0

    def test_cross_chiral(self):
        substrate = CrossChiralSubstrate()
        valid, missing = substrate.validate()
        assert valid is True
        assert len(missing) == 0

    def test_cross_chiral_incomplete(self):
        substrate = CrossChiralSubstrate(triplets=["GCG", "CGC"])
        valid, missing = substrate.validate()
        assert valid is False
        assert len(missing) > 0

    def test_molecular_access(self):
        access = MolecularAccessModel()
        access.evaluate()
        assert access.replication_possible is True
        assert access.degradation_resistance > 0.9
        assert access.key_space_size == 16384

    def test_full_qt45_analysis(self):
        result = analyze_qt45()
        assert result.redundancy.copies_needed > 0
        assert result.error_correction.parity_nucleotides_needed > 0
        assert result.enzymatic_resistance.overall_resistance > 0


# ======================================================================
# Section 8: Freeze-Thaw
# ======================================================================

class TestFreezeThaw:
    def test_strand_separation(self):
        # At high temperature, separation should be higher
        p_cold = strand_separation_probability(-7.0)
        p_warm = strand_separation_probability(90.0)
        assert p_warm > p_cold

    def test_eutectic_concentration(self):
        factor = eutectic_concentration_factor(-7.0)
        assert factor > 1.0
        # Above freezing, no concentration
        factor_warm = eutectic_concentration_factor(20.0)
        assert factor_warm == 1.0

    def test_stall_probability(self):
        p = replication_stall_probability()
        assert 0.0 <= p <= 1.0

    def test_optimize_interval(self):
        result = optimize_freeze_thaw_interval()
        assert result.optimal_freeze_hours > 0
        assert result.optimal_thaw_hours > 0
        assert result.optimal_cycle_hours > 0

    def test_full_freeze_thaw_analysis(self):
        result = analyze_freeze_thaw()
        assert result.stall_probability >= 0
        assert result.concentration_factor > 0
        assert result.triplet_trapping.n_triplets > 0


# ======================================================================
# Section 9: Validation Report
# ======================================================================

class TestValidation:
    def test_modeling_layers_registered(self):
        assert len(MODELING_LAYERS) == 8

    def test_interaction_map(self):
        interactions = build_interaction_map()
        assert len(interactions) == 10
        types = {ix.interaction_type for ix in interactions}
        assert "data_flow" in types
        assert "constraint" in types
        assert "parameter_sharing" in types

    def test_generate_report_no_sim(self):
        report = generate_validation_report(run_simulations=False)
        assert report.total_layers == 8
        assert report.total_new_variables > 0
        assert report.all_simulations_ready is True
        assert report.total_missing_data_items > 0

    def test_generate_report_with_sim(self):
        report = generate_validation_report(run_simulations=True, seed=42)
        assert report.topology_comparison is not None
        assert report.delay_encoding_phase is not None
        assert report.delay_encoding_static is not None
        assert report.heterogeneity_comparison is not None
        assert report.homeostatic_adversarial is not None
        assert report.homeostatic_repeated is not None
        assert report.state_space is not None
        assert report.qt45 is not None
        assert report.freeze_thaw is not None

    def test_format_report(self):
        report = generate_validation_report(run_simulations=False)
        text = format_report(report)
        assert "NEUROMORPHIC REASONING ENGINE" in text
        assert "MODELING LAYERS" in text
        assert "SIMULATION READINESS" in text

    def test_report_to_dict(self):
        report = generate_validation_report(run_simulations=False)
        d = report_to_dict(report)
        assert d["total_layers"] == 8
        assert len(d["layers"]) == 8
        assert len(d["interactions"]) == 10


# ======================================================================
# Section 10: Computation Layer Integration (hypothesis_engine wiring)
# ======================================================================

class TestComputationLayer:
    """Test the reasoning engine computation layer wired into hypothesis_engine."""

    def test_detect_ribozyme_query(self):
        from acheron.rag.hypothesis_engine import detect_computation_query
        q = "Calculate the Bio-RAID redundancy factor for QT45 with 94.1% fidelity"
        modules = detect_computation_query(q)
        assert "ribozyme" in modules

    def test_detect_freeze_thaw_query(self):
        from acheron.rag.hypothesis_engine import detect_computation_query
        modules = detect_computation_query("freeze-thaw optimal interval")
        assert "freeze_thaw" in modules

    def test_detect_mosaic_query(self):
        from acheron.rag.hypothesis_engine import detect_computation_query
        modules = detect_computation_query("small-world topology fault tolerance")
        assert "mosaic" in modules

    def test_detect_state_space_query(self):
        from acheron.rag.hypothesis_engine import detect_computation_query
        modules = detect_computation_query("eigenvalue stability analysis")
        assert "state_space" in modules

    def test_detect_no_computation(self):
        from acheron.rag.hypothesis_engine import detect_computation_query
        modules = detect_computation_query("what is the gap junction conductance")
        assert modules == []

    def test_run_ribozyme_computation(self):
        from acheron.rag.hypothesis_engine import run_computation
        result = run_computation(["ribozyme"], "calculate redundancy for 99.9% integrity one year")
        assert "REASONING ENGINE" in result
        assert "Redundancy Factor" in result
        assert "copies" in result.lower()

    def test_run_computation_extracts_params(self):
        from acheron.rag.hypothesis_engine import run_computation
        result = run_computation(
            ["ribozyme"],
            "Bio-RAID redundancy for 99.9% data integrity for one year with fidelity of 94.1% and 72-day generation cycle"
        )
        assert "0.941" in result
        assert "72" in result
        assert "0.999" in result

    def test_template_injection(self):
        from acheron.rag.hypothesis_engine import get_mode_query_template
        from acheron.models import NexusMode
        query = "Calculate the Bio-RAID redundancy factor for QT45"
        template = get_mode_query_template(NexusMode.DECISION, fast=True, query=query)
        assert "REASONING ENGINE" in template
        assert "Redundancy Factor" in template

    def test_template_no_computation_for_plain_query(self):
        from acheron.rag.hypothesis_engine import get_mode_query_template
        from acheron.models import NexusMode
        query = "Is planarian bioelectric memory viable?"
        template = get_mode_query_template(NexusMode.DECISION, fast=True, query=query)
        assert "REASONING ENGINE" not in template
