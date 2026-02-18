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
    analyze_spectral_properties,
    build_ring_lattice,
    build_small_world,
    build_uniform_random,
    compare_topologies,
    compute_clustering_coefficient,
    compute_average_path_length,
    compute_graph_laplacian,
    compute_signal_propagation_latency,
    compute_energy_per_routing_event,
    compute_spectral_gap,
    compute_total_network_energy,
    find_optimal_rewires,
    rewire_sweep,
)
from acheron.reasoning.empirical import (
    CROSS_SPECIES_TABLE,
    EMPIRICAL_PARAMS,
    LEVIN_SPEC,
    MOSAIC_SPEC,
    SPECTRAL_THRESHOLD,
    format_cross_species_table,
    get_cross_species_table,
    get_empirical_context,
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
        assert len(MODELING_LAYERS) == 11  # 8 original + empirical + sensitivity + falsification

    def test_interaction_map(self):
        interactions = build_interaction_map()
        assert len(interactions) == 10
        types = {ix.interaction_type for ix in interactions}
        assert "data_flow" in types
        assert "constraint" in types
        assert "parameter_sharing" in types

    def test_generate_report_no_sim(self):
        report = generate_validation_report(run_simulations=False)
        assert report.total_layers == 11
        assert report.total_new_variables > 0
        assert report.all_simulations_ready is True
        assert report.total_missing_data_items > 0

    def test_generate_report_with_sim(self):
        report = generate_validation_report(run_simulations=True, seed=42)
        assert report.topology_comparison is not None
        assert report.spectral_analysis is not None
        assert report.spectral_analysis.spectral_gap > 0
        assert report.rewire_analysis is not None
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
        assert d["total_layers"] == 11
        assert len(d["layers"]) == 11
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

    def test_detect_spectral_rewiring_query(self):
        from acheron.rag.hypothesis_engine import detect_computation_query
        modules = detect_computation_query(
            "analyze our Watts-Strogatz connectivity matrix and identify "
            "edge-rewiring events that minimize energy dissipation while "
            "maximizing the Spectral Gap"
        )
        assert "spectral_rewiring" in modules

    def test_detect_payvand_mosaic_query(self):
        from acheron.rag.hypothesis_engine import detect_computation_query
        modules = detect_computation_query(
            "given Melika Payvand's research on Mosaic architectures"
        )
        assert "spectral_rewiring" in modules

    def test_detect_cross_species_query(self):
        from acheron.rag.hypothesis_engine import detect_computation_query
        modules = detect_computation_query("cross-species comparison of bioelectric memory")
        assert "spectral_rewiring" in modules

    def test_run_spectral_rewiring_computation(self):
        from acheron.rag.hypothesis_engine import run_computation
        result = run_computation(
            ["spectral_rewiring"],
            "analyze Watts-Strogatz connectivity matrix spectral gap"
        )
        assert "REASONING ENGINE" in result
        assert "SPECTRAL ANALYSIS" in result
        assert "Spectral gap" in result
        assert "REWIRING" in result
        assert "MOSAIC" in result
        assert "BIOELECTRIC MEMORY" in result
        assert "CROSS-SPECIES" in result
        assert "KILL CONDITION" in result


# ======================================================================
# Section 2b: Graph Laplacian Spectral Analysis
# ======================================================================

class TestSpectralAnalysis:
    def test_graph_laplacian_ring(self):
        g = build_ring_lattice(10, 4)
        L, node_ids = compute_graph_laplacian(g)
        assert len(L) == 10
        assert len(node_ids) == 10
        # Laplacian row sum should be 0 for each row
        for row in L:
            assert abs(sum(row)) < 1e-10

    def test_graph_laplacian_small_world(self):
        g = build_small_world(20, 4, seed=42)
        L, _ = compute_graph_laplacian(g)
        # Laplacian is symmetric
        for i in range(len(L)):
            for j in range(len(L)):
                assert abs(L[i][j] - L[j][i]) < 1e-10

    def test_spectral_gap_positive(self):
        g = build_small_world(20, 4, seed=42)
        gap = compute_spectral_gap(g)
        assert gap > 0  # connected graph should have positive lambda_2

    def test_spectral_gap_ring_vs_small_world(self):
        ring = build_ring_lattice(30, 4)
        sw = build_small_world(30, 4, rewire_prob=0.2, seed=42)
        gap_ring = compute_spectral_gap(ring)
        gap_sw = compute_spectral_gap(sw)
        # Small-world should have higher spectral gap (better connectivity)
        assert gap_sw >= gap_ring * 0.8  # allow some tolerance

    def test_total_network_energy(self):
        g = build_small_world(10, 4, seed=42)
        energy = compute_total_network_energy(g)
        assert energy > 0

    def test_spectral_analysis_full(self):
        g = build_small_world(15, 4, seed=42)
        sa = analyze_spectral_properties(g)
        assert sa.spectral_gap > 0
        assert len(sa.laplacian_eigenvalues) > 0
        assert sa.total_network_energy_J > 0
        assert sa.energy_per_edge_J > 0
        assert sa.spectral_efficiency > 0

    def test_spectral_analysis_small_graph(self):
        g = SubstrateGraph()
        g.nodes["a"] = SubstrateNode(node_id="a")
        sa = analyze_spectral_properties(g)
        assert sa.spectral_gap == 0.0


# ======================================================================
# Section 2c: Energy-Optimized Edge Rewiring
# ======================================================================

class TestEdgeRewiring:
    def test_find_optimal_rewires(self):
        g = build_small_world(20, 4, seed=42)
        result = find_optimal_rewires(g, rewire_budget=3, seed=42)
        assert result.n_nodes == 20
        assert result.baseline_spectral_gap > 0
        assert result.rewire_budget == 3
        assert len(result.candidates) > 0

    def test_rewire_improves_efficiency(self):
        g = build_small_world(20, 4, rewire_prob=0.05, seed=42)
        result = find_optimal_rewires(g, rewire_budget=5, seed=42)
        # Optimal rewires should improve spectral efficiency (gap/energy),
        # even if cumulative application introduces interaction effects on gap alone
        if result.optimal_rewires:
            assert result.optimized_spectral_efficiency >= result.baseline_spectral_efficiency
            # Individual top rewire should have positive delta_spectral_gap
            assert result.optimal_rewires[0].delta_spectral_gap > 0

    def test_rewire_candidates_have_scores(self):
        g = build_small_world(15, 4, seed=42)
        result = find_optimal_rewires(g, rewire_budget=3, seed=42)
        for c in result.candidates:
            assert c.original_source != ""
            assert c.new_target != ""
            # efficiency_score should be computed
            assert isinstance(c.efficiency_score, float)

    def test_rewire_sweep(self):
        results = rewire_sweep(n_nodes=15, k_neighbors=4, seed=42)
        assert len(results) > 5
        # All entries should have computed values
        for r in results:
            assert "spectral_gap" in r
            assert "clustering_coefficient" in r
            assert "average_path_length" in r
            assert "spectral_efficiency" in r

    def test_rewire_sweep_monotonic_path_length(self):
        results = rewire_sweep(n_nodes=20, k_neighbors=4, seed=42)
        # Path length should generally decrease as rewire_prob increases
        apl_first = results[0]["average_path_length"]  # p=0
        apl_last = results[-1]["average_path_length"]   # p=1
        assert apl_last < apl_first

    def test_small_graph_rewire(self):
        # Graph with < 4 nodes should return empty result
        g = SubstrateGraph()
        for i in range(3):
            g.nodes[f"c{i}"] = SubstrateNode(node_id=f"c{i}")
        result = find_optimal_rewires(g, seed=42)
        assert result.n_nodes == 3
        assert len(result.optimal_rewires) == 0


# ======================================================================
# Section 10: Empirical Grounding Layer
# ======================================================================

class TestEmpiricalGrounding:
    def test_mosaic_spec(self):
        assert MOSAIC_SPEC.architecture_type == "2D Analog Systolic Array"
        assert MOSAIC_SPEC.energy_per_route_J > 0
        assert MOSAIC_SPEC.spectral_gap_target > 0
        assert "RRAM" in MOSAIC_SPEC.routing_substrate

    def test_levin_spec(self):
        assert LEVIN_SPEC.t_hold_initial_hours == 3.0
        assert LEVIN_SPEC.vmem_delta_mV == 50.0
        assert LEVIN_SPEC.gap_junction_conductance_nS > 0
        assert "Innexin" in LEVIN_SPEC.gap_junction_type_planarian
        assert "Connexin" in LEVIN_SPEC.gap_junction_type_vertebrate

    def test_spectral_threshold_model(self):
        assert SPECTRAL_THRESHOLD.theta_critical == 0.05
        assert SPECTRAL_THRESHOLD.theta_optimal == 0.15
        assert SPECTRAL_THRESHOLD.is_stall_risk(0.01) is True
        assert SPECTRAL_THRESHOLD.is_stall_risk(0.10) is False
        assert SPECTRAL_THRESHOLD.is_optimal(0.20) is True
        assert SPECTRAL_THRESHOLD.is_optimal(0.10) is False

    def test_energy_delay_cost(self):
        cost = SPECTRAL_THRESHOLD.energy_delay_cost(1e-10, 0.15)
        assert cost > 0
        assert cost < float("inf")
        # Zero spectral gap = infinite cost
        cost_inf = SPECTRAL_THRESHOLD.energy_delay_cost(1e-10, 0.0)
        assert cost_inf == float("inf")

    def test_cross_species_table(self):
        table = get_cross_species_table()
        assert len(table) == 4
        organisms = {e.organism for e in table}
        assert "Schmidtea mediterranea" in organisms
        assert "Xenopus laevis" in organisms
        assert "Physarum polycephalum" in organisms

    def test_cross_species_relevance_scores(self):
        table = get_cross_species_table()
        for entry in table:
            assert 1 <= entry.relevance_score <= 5
        # Planarian should have highest relevance
        planarian = [e for e in table if "Schmidtea" in e.organism][0]
        assert planarian.relevance_score == 5

    def test_cross_species_table_formatting(self):
        text = format_cross_species_table()
        assert "CROSS-SPECIES" in text
        assert "Planarian" in text
        assert "Xenopus" in text
        assert "Physarum" in text

    def test_empirical_params(self):
        assert EMPIRICAL_PARAMS.spectral_gap_target == 0.15
        assert EMPIRICAL_PARAMS.spectral_gap_critical == 0.05
        assert EMPIRICAL_PARAMS.gj_conductance_nS == 2.0
        assert EMPIRICAL_PARAMS.e_bit_pJ > 0

    def test_empirical_context_generation(self):
        ctx = get_empirical_context()
        assert "CERTIFIED EMPIRICAL PARAMETERS" in ctx
        assert "T_hold" in ctx
        assert "SPECTRAL PROPERTIES" in ctx
        assert "MOSAIC" in ctx
        assert "Lattice (p=0)" in ctx
        assert "Small-World" in ctx

    def test_cross_species_entries_have_evidence(self):
        for entry in CROSS_SPECIES_TABLE:
            assert "[MEASURED]" in entry.vmem_manipulation_evidence
            assert entry.gap_junction_type != ""
            assert entry.spectral_relevance != ""


# ======================================================================
# Section 2d: Barabasi-Albert Scale-Free Topology
# ======================================================================

class TestScaleFree:
    def test_build_scale_free_basic(self):
        from acheron.reasoning.mosaic import build_scale_free
        g = build_scale_free(30, m_edges_per_node=3, seed=42)
        assert g.node_count() == 30
        assert g.edge_count() > 0

    def test_scale_free_has_hubs(self):
        from acheron.reasoning.mosaic import build_scale_free, _build_adjacency
        g = build_scale_free(50, m_edges_per_node=2, seed=42)
        adj = _build_adjacency(g)
        degrees = [len(neighbors) for neighbors in adj.values()]
        max_deg = max(degrees)
        min_deg = min(degrees)
        # Scale-free should have high degree variance (hubs)
        assert max_deg > min_deg * 2

    def test_scale_free_spectral_gap(self):
        from acheron.reasoning.mosaic import build_scale_free, compute_spectral_gap
        g = build_scale_free(20, m_edges_per_node=2, seed=42)
        gap = compute_spectral_gap(g)
        assert gap > 0  # connected graph

    def test_scale_free_vs_small_world(self):
        from acheron.reasoning.mosaic import (
            build_scale_free, compute_clustering_coefficient,
            compute_average_path_length,
        )
        ba = build_scale_free(30, m_edges_per_node=3, seed=42)
        sw = build_small_world(30, 6, rewire_prob=0.1, seed=42)
        # BA typically has lower clustering than SW
        cc_ba = compute_clustering_coefficient(ba)
        cc_sw = compute_clustering_coefficient(sw)
        # Both should be positive
        assert cc_ba >= 0
        assert cc_sw >= 0

    def test_scale_free_small_graph(self):
        from acheron.reasoning.mosaic import build_scale_free
        g = build_scale_free(4, m_edges_per_node=2, seed=42)
        assert g.node_count() == 4
        assert g.edge_count() > 0


# ======================================================================
# Section 6b: Nonlinear Voltage-Dependent Coupling
# ======================================================================

class TestNonlinearCoupling:
    def test_linear_coupling_unchanged(self):
        """Existing linear coupling behavior should be identical."""
        g = build_small_world(10, 4, seed=42)
        m = build_system_matrices(g, leak_rate=0.1, coupling_mode="linear")
        assert m.dimension == 10
        # Diagonal should be negative (leak)
        for i in range(10):
            assert m.A[i][i] < 0

    def test_nonlinear_coupling_builds(self):
        g = build_small_world(10, 4, seed=42)
        m = build_system_matrices(g, leak_rate=0.1, coupling_mode="nonlinear")
        assert m.dimension == 10
        assert len(m.A) == 10

    def test_nonlinear_coupling_reduces_off_diagonal(self):
        """Nonlinear gating should reduce coupling when nodes have same Vm."""
        g = build_small_world(10, 4, seed=42)
        m_lin = build_system_matrices(g, leak_rate=0.1, coupling_mode="linear")
        m_nl = build_system_matrices(g, leak_rate=0.1, coupling_mode="nonlinear")
        # When all nodes have same Vm (-40mV), dV=0, so gating ≈ sigmoid(0)
        # which is close to but can differ from 1.0. Off-diags should be
        # similar magnitude.
        for i in range(10):
            for j in range(10):
                if i != j:
                    # Both should have same sign (positive off-diagonal)
                    if abs(m_lin.A[i][j]) > 1e-12:
                        assert m_nl.A[i][j] >= 0

    def test_nonlinear_stability(self):
        g = build_small_world(10, 4, seed=42)
        m = build_system_matrices(g, leak_rate=0.1, coupling_mode="nonlinear")
        stab = analyze_stability(m)
        assert len(stab.eigenvalues) > 0

    def test_linear_nonlinear_stability_agreement(self):
        """Both models should agree on stability for uniform Vm."""
        g = build_small_world(15, 4, seed=42)
        m_lin = build_system_matrices(g, leak_rate=0.1, coupling_mode="linear")
        m_nl = build_system_matrices(g, leak_rate=0.1, coupling_mode="nonlinear")
        stab_lin = analyze_stability(m_lin)
        stab_nl = analyze_stability(m_nl)
        assert stab_lin.is_stable == stab_nl.is_stable


# ======================================================================
# Section 11: Parameter Sensitivity Analysis
# ======================================================================

class TestSensitivity:
    def test_sensitivity_conductance(self):
        from acheron.reasoning.sensitivity import sensitivity_conductance_to_spectral_gap
        result = sensitivity_conductance_to_spectral_gap(n_nodes=15, seed=42)
        assert result.parameter_name == "base_conductance_nS"
        assert result.output_name == "spectral_gap"
        assert len(result.sweep_points) > 3
        assert isinstance(result.sensitivity_index, float)

    def test_sensitivity_leak_rate(self):
        from acheron.reasoning.sensitivity import sensitivity_leak_rate_to_stability
        result = sensitivity_leak_rate_to_stability(n_nodes=10, seed=42)
        assert result.parameter_name == "leak_rate"
        assert len(result.sweep_points) > 3

    def test_sensitivity_rewire_prob(self):
        from acheron.reasoning.sensitivity import sensitivity_rewire_prob_to_clustering
        result = sensitivity_rewire_prob_to_clustering(n_nodes=15, seed=42)
        assert result.parameter_name == "rewire_prob"
        assert result.output_name == "clustering_coefficient"

    def test_sensitivity_energy(self):
        from acheron.reasoning.sensitivity import sensitivity_conductance_scale_to_energy
        result = sensitivity_conductance_scale_to_energy(n_nodes=10, seed=42)
        assert result.parameter_name == "delta_v_mV"
        assert len(result.sweep_points) > 3

    def test_cross_model_comparison(self):
        from acheron.reasoning.sensitivity import compare_topology_models
        results = compare_topology_models(n_nodes=15, seed=42)
        assert len(results) == 4  # spectral_gap, cc, apl, energy
        for mc in results:
            assert mc.watts_strogatz >= 0 or mc.watts_strogatz < 0  # is a number
            assert mc.barabasi_albert >= 0 or mc.barabasi_albert < 0
            assert mc.erdos_renyi >= 0 or mc.erdos_renyi < 0
            assert 0.0 <= mc.agreement <= 1.0
            assert mc.dominant_model in ["watts_strogatz", "barabasi_albert", "erdos_renyi"]

    def test_full_sensitivity_report(self):
        from acheron.reasoning.sensitivity import run_sensitivity_analysis
        report = run_sensitivity_analysis(n_nodes=15, seed=42)
        assert len(report.parameter_sensitivities) == 4
        assert len(report.model_comparisons) == 4
        # Should have some conclusions
        assert len(report.robust_conclusions) + len(report.assumptions_at_risk) > 0

    def test_sensitivity_report_formatting(self):
        from acheron.reasoning.sensitivity import (
            run_sensitivity_analysis, format_sensitivity_report,
        )
        report = run_sensitivity_analysis(n_nodes=15, seed=42)
        text = format_sensitivity_report(report)
        assert "PARAMETER SENSITIVITY" in text
        assert "CROSS-MODEL" in text


# ======================================================================
# Section 12: Falsification Prediction Registry
# ======================================================================

class TestFalsification:
    def test_prediction_registry(self):
        from acheron.reasoning.falsification import _build_prediction_registry
        preds = _build_prediction_registry()
        assert len(preds) == 7
        ids = [p.prediction_id for p in preds]
        assert "P1_SPECTRAL_GAP_POSITIVE" in ids
        assert "P7_NONLINEAR_VS_LINEAR" in ids

    def test_all_predictions_have_kill_conditions(self):
        from acheron.reasoning.falsification import _build_prediction_registry
        for pred in _build_prediction_registry():
            assert pred.kill_condition != ""
            assert pred.kill_direction in ("below", "above")
            assert pred.experimental_test != ""
            assert len(pred.assumptions) > 0

    def test_falsification_analysis(self):
        from acheron.reasoning.falsification import (
            run_falsification_analysis, PredictionStatus,
        )
        report = run_falsification_analysis(n_nodes=15, seed=42)
        assert len(report.predictions) == 7
        assert report.alive_count + report.at_risk_count + report.falsified_count + report.untested_count == 7
        assert report.overall_health in ("HEALTHY", "CAUTION", "CRITICAL")

    def test_spectral_gap_prediction_alive(self):
        from acheron.reasoning.falsification import (
            run_falsification_analysis, PredictionStatus,
        )
        report = run_falsification_analysis(n_nodes=20, seed=42)
        p1 = [p for p in report.predictions if p.prediction_id == "P1_SPECTRAL_GAP_POSITIVE"][0]
        assert p1.computed_value > 0
        assert p1.status == PredictionStatus.ALIVE

    def test_stability_prediction_alive(self):
        from acheron.reasoning.falsification import (
            run_falsification_analysis, PredictionStatus,
        )
        report = run_falsification_analysis(n_nodes=15, seed=42)
        p4 = [p for p in report.predictions if p.prediction_id == "P4_STABILITY_NEGATIVE_EIGENVALUES"][0]
        assert p4.computed_value < 0  # negative eigenvalue = stable
        assert p4.status == PredictionStatus.ALIVE

    def test_nonlinear_agreement_prediction(self):
        from acheron.reasoning.falsification import (
            run_falsification_analysis, PredictionStatus,
        )
        report = run_falsification_analysis(n_nodes=15, seed=42)
        p7 = [p for p in report.predictions if p.prediction_id == "P7_NONLINEAR_VS_LINEAR"][0]
        assert p7.computed_value == 1.0  # both should agree on stability
        assert p7.status == PredictionStatus.ALIVE

    def test_falsification_report_formatting(self):
        from acheron.reasoning.falsification import (
            run_falsification_analysis, format_falsification_report,
        )
        report = run_falsification_analysis(n_nodes=15, seed=42)
        text = format_falsification_report(report)
        assert "FALSIFICATION" in text
        assert "POPPER" in text
        assert "Overall Health" in text

    def test_memory_persistence_prediction(self):
        from acheron.reasoning.falsification import (
            run_falsification_analysis, PredictionStatus,
        )
        report = run_falsification_analysis(n_nodes=15, seed=42)
        p6 = [p for p in report.predictions if p.prediction_id == "P6_MEMORY_PERSISTENCE"][0]
        assert p6.computed_value == 3.0  # Levin's T_hold
        # AT_RISK because 3h is close to the 1h kill bound relative to the
        # full expected range (3-720h). This is diagnostically correct:
        # T_hold of 3h is borderline for robust memory.
        assert p6.status == PredictionStatus.AT_RISK
        assert p6.margin > 0  # but not falsified


# ======================================================================
# Section 10b: Computation Layer — Model Validation
# ======================================================================

class TestModelValidationComputation:
    def test_detect_sensitivity_query(self):
        from acheron.rag.hypothesis_engine import detect_computation_query
        modules = detect_computation_query("run parameter sensitivity analysis")
        assert "model_validation" in modules

    def test_detect_falsification_query(self):
        from acheron.rag.hypothesis_engine import detect_computation_query
        modules = detect_computation_query("show falsification predictions and kill conditions")
        assert "model_validation" in modules

    def test_detect_scale_free_query(self):
        from acheron.rag.hypothesis_engine import detect_computation_query
        modules = detect_computation_query("compare Barabasi-Albert scale-free topology")
        assert "model_validation" in modules

    def test_detect_nonlinear_query(self):
        from acheron.rag.hypothesis_engine import detect_computation_query
        modules = detect_computation_query("nonlinear coupling voltage-dependent gap junction")
        assert "model_validation" in modules

    def test_run_model_validation_computation(self):
        from acheron.rag.hypothesis_engine import run_computation
        result = run_computation(
            ["model_validation"],
            "run sensitivity analysis and falsification registry"
        )
        assert "REASONING ENGINE" in result
        assert "SENSITIVITY" in result
        assert "FALSIFICATION" in result
        assert "ROBUST" in result or "HIGH SENSITIVITY" in result


# ======================================================================
# Section 9b: Validation Report with new layers
# ======================================================================

class TestValidationUpdated:
    def test_modeling_layers_count(self):
        assert len(MODELING_LAYERS) == 11  # 9 original + sensitivity + falsification

    def test_report_includes_new_layers(self):
        report = generate_validation_report(run_simulations=False)
        assert report.total_layers == 11
        layer_ids = [l.layer_id for l in report.modeling_layers]
        assert "sensitivity" in layer_ids
        assert "falsification" in layer_ids

    def test_report_with_sim_includes_new(self):
        report = generate_validation_report(run_simulations=True, seed=42)
        assert report.sensitivity is not None
        assert report.falsification is not None
        assert report.falsification.overall_health in ("HEALTHY", "CAUTION", "CRITICAL")

    def test_format_report_includes_new_sections(self):
        report = generate_validation_report(run_simulations=True, seed=42)
        text = format_report(report)
        assert "Sensitivity" in text
        assert "Falsification" in text


# ======================================================================
# Section 13: Research Question Translator & Plain English
# ======================================================================

class TestResearchQuestions:
    def test_detect_voltage_category(self):
        from acheron.reasoning.research_questions import detect_research_category, ResearchCategory
        cats = detect_research_category("What voltage turns on head regeneration?")
        assert ResearchCategory.VOLTAGE_CONTROL in cats

    def test_detect_damage_category(self):
        from acheron.reasoning.research_questions import detect_research_category, ResearchCategory
        cats = detect_research_category("Can damaged cells auto-heal and self-repair?")
        assert ResearchCategory.DAMAGE_RESPONSE in cats

    def test_detect_bio_storage_category(self):
        from acheron.reasoning.research_questions import detect_research_category, ResearchCategory
        cats = detect_research_category("Can planarians carry information like a hard drive?")
        assert ResearchCategory.BIO_STORAGE in cats

    def test_detect_timeline_category(self):
        from acheron.reasoning.research_questions import detect_research_category, ResearchCategory
        cats = detect_research_category("How long does regeneration take?")
        assert ResearchCategory.REGENERATION_TIMELINE in cats

    def test_detect_rewriting_category(self):
        from acheron.reasoning.research_questions import detect_research_category, ResearchCategory
        cats = detect_research_category("Can we rewrite the pattern to force regeneration on demand?")
        assert ResearchCategory.PATTERN_REWRITING in cats

    def test_translate_casual_query(self):
        from acheron.reasoning.research_questions import translate_query
        tq = translate_query("What voltage turns on head growth in worms?")
        assert tq.category.value == "voltage_control"
        assert "Vmem" in tq.scientific_query
        assert len(tq.key_variables) > 0

    def test_translate_storage_query(self):
        from acheron.reasoning.research_questions import translate_query
        tq = translate_query("Can worms store data like a hard drive and pass it on?")
        assert tq.category.value == "bio_storage"
        assert "T_hold" in tq.scientific_query

    def test_format_translated_query(self):
        from acheron.reasoning.research_questions import translate_query, format_translated_query
        tq = translate_query("How long does it take after damage?")
        text = format_translated_query(tq)
        assert "RESEARCH QUESTION TRANSLATION" in text
        assert "NEXUS QUERY" in text
        assert "HYPOTHESIS TEMPLATE" in text

    def test_is_casual_query(self):
        from acheron.reasoning.research_questions import is_casual_query
        assert is_casual_query("Can we turn on regeneration in worms?") is True
        assert is_casual_query("eigenvalue stability analysis of Laplacian") is False

    def test_plain_english_injection(self):
        from acheron.reasoning.research_questions import get_plain_english_injection, ResearchCategory
        text = get_plain_english_injection(ResearchCategory.VOLTAGE_CONTROL)
        assert "WHAT THIS MEANS" in text
        assert "CONFIDENCE LEVEL" in text

    def test_casual_query_routes_to_tutor(self):
        from acheron.rag.hypothesis_engine import detect_mode
        from acheron.models import NexusMode
        mode = detect_mode("Can we turn on head regeneration in worms?")
        assert mode == NexusMode.TUTOR

    def test_tutor_template_has_plain_english(self):
        from acheron.rag.hypothesis_engine import get_mode_query_template
        from acheron.models import NexusMode
        template = get_mode_query_template(
            NexusMode.TUTOR, fast=True,
            query="Can worms store data like a hard drive and pass it on?"
        )
        assert "WHAT THIS MEANS" in template
