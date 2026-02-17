"""Substrate=Algorithm Core Abstraction (Section 1).

Principle:
    In unconventional substrates, topology and physical state transitions
    ARE the algorithm.  Connectivity graph changes = memory write.
    Gap junction conductance modulation = routing reconfiguration.
    Physical delay = temporal encoding.

Nexus treats structural topology as computational state, not transport layer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class TopologyEventType(str, Enum):
    """Primitive operations on the substrate topology graph."""

    EDGE_WEIGHT_CHANGE = "edge_weight_change"      # conductance modulation
    EDGE_CREATE = "edge_create"                     # new gap junction formation
    EDGE_DESTROY = "edge_destroy"                   # gap junction closure
    NODE_STATE_CHANGE = "node_state_change"         # Vmem transition
    NODE_CREATE = "node_create"                     # cell division
    NODE_DESTROY = "node_destroy"                   # apoptosis


class ComputationalPrimitive(str, Enum):
    """Maps substrate events to computational semantics."""

    MEMORY_WRITE = "memory_write"           # topology change persists state
    ROUTING_RECONFIG = "routing_reconfig"    # conductance change alters signal path
    TEMPORAL_ENCODE = "temporal_encode"      # physical delay encodes information
    STATE_TRANSITION = "state_transition"    # attractor basin switch


@dataclass
class TopologyEvent:
    """A single event on the substrate topology graph.

    Each event carries both physical parameters (conductance, voltage)
    and computational semantics (what operation this event represents
    in the substrate-as-algorithm model).
    """

    event_type: TopologyEventType
    source_node: str = ""
    target_node: str = ""
    # Physical parameters
    conductance_nS: Optional[float] = None          # gap junction conductance
    delta_vm_mV: Optional[float] = None             # voltage change
    delay_ms: Optional[float] = None                # physical propagation delay
    # Computational mapping
    computational_primitive: ComputationalPrimitive = ComputationalPrimitive.STATE_TRANSITION
    metadata: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Topology-to-computation mapping rules
# ---------------------------------------------------------------------------
TOPOLOGY_COMPUTATION_MAP: dict[TopologyEventType, ComputationalPrimitive] = {
    TopologyEventType.EDGE_WEIGHT_CHANGE: ComputationalPrimitive.ROUTING_RECONFIG,
    TopologyEventType.EDGE_CREATE: ComputationalPrimitive.MEMORY_WRITE,
    TopologyEventType.EDGE_DESTROY: ComputationalPrimitive.MEMORY_WRITE,
    TopologyEventType.NODE_STATE_CHANGE: ComputationalPrimitive.STATE_TRANSITION,
    TopologyEventType.NODE_CREATE: ComputationalPrimitive.MEMORY_WRITE,
    TopologyEventType.NODE_DESTROY: ComputationalPrimitive.MEMORY_WRITE,
}


@dataclass
class SubstrateNode:
    """A node in the substrate topology graph (cell cluster)."""

    node_id: str
    vm_mV: float = -40.0                # membrane potential
    vm_rest_mV: float = -40.0           # resting membrane potential
    ion_channel_expression: dict = field(default_factory=dict)  # channel -> expression level
    cluster_size: int = 1               # number of cells in this cluster
    metadata: dict = field(default_factory=dict)


@dataclass
class SubstrateEdge:
    """A weighted edge (gap junction connection) between nodes."""

    source: str
    target: str
    conductance_nS: float = 1.0         # gap junction conductance in nanosiemens
    diffusion_delay_ms: float = 0.5     # signal propagation delay
    junction_type: str = "innexin"      # innexin | connexin
    metadata: dict = field(default_factory=dict)


@dataclass
class SubstrateGraph:
    """The substrate topology graph — computational state of the system.

    This is the core data structure that embodies the substrate=algorithm
    principle.  Mutations to this graph ARE computation.
    """

    nodes: dict[str, SubstrateNode] = field(default_factory=dict)
    edges: list[SubstrateEdge] = field(default_factory=list)
    event_log: list[TopologyEvent] = field(default_factory=list)

    def add_node(self, node: SubstrateNode) -> TopologyEvent:
        self.nodes[node.node_id] = node
        event = TopologyEvent(
            event_type=TopologyEventType.NODE_CREATE,
            source_node=node.node_id,
            computational_primitive=ComputationalPrimitive.MEMORY_WRITE,
        )
        self.event_log.append(event)
        return event

    def remove_node(self, node_id: str) -> TopologyEvent:
        self.nodes.pop(node_id, None)
        self.edges = [
            e for e in self.edges
            if e.source != node_id and e.target != node_id
        ]
        event = TopologyEvent(
            event_type=TopologyEventType.NODE_DESTROY,
            source_node=node_id,
            computational_primitive=ComputationalPrimitive.MEMORY_WRITE,
        )
        self.event_log.append(event)
        return event

    def add_edge(self, edge: SubstrateEdge) -> TopologyEvent:
        self.edges.append(edge)
        event = TopologyEvent(
            event_type=TopologyEventType.EDGE_CREATE,
            source_node=edge.source,
            target_node=edge.target,
            conductance_nS=edge.conductance_nS,
            delay_ms=edge.diffusion_delay_ms,
            computational_primitive=ComputationalPrimitive.MEMORY_WRITE,
        )
        self.event_log.append(event)
        return event

    def modulate_edge(self, source: str, target: str, new_conductance_nS: float) -> Optional[TopologyEvent]:
        """Modulate gap junction conductance = routing reconfiguration."""
        for edge in self.edges:
            if edge.source == source and edge.target == target:
                old = edge.conductance_nS
                edge.conductance_nS = new_conductance_nS
                event = TopologyEvent(
                    event_type=TopologyEventType.EDGE_WEIGHT_CHANGE,
                    source_node=source,
                    target_node=target,
                    conductance_nS=new_conductance_nS,
                    computational_primitive=ComputationalPrimitive.ROUTING_RECONFIG,
                    metadata={"old_conductance_nS": old},
                )
                self.event_log.append(event)
                return event
        return None

    def set_node_vm(self, node_id: str, new_vm_mV: float) -> Optional[TopologyEvent]:
        """Change node voltage state = state transition."""
        node = self.nodes.get(node_id)
        if node is None:
            return None
        old_vm = node.vm_mV
        node.vm_mV = new_vm_mV
        event = TopologyEvent(
            event_type=TopologyEventType.NODE_STATE_CHANGE,
            source_node=node_id,
            delta_vm_mV=new_vm_mV - old_vm,
            computational_primitive=ComputationalPrimitive.STATE_TRANSITION,
            metadata={"old_vm_mV": old_vm, "new_vm_mV": new_vm_mV},
        )
        self.event_log.append(event)
        return event

    def adjacency_dict(self) -> dict[str, list[tuple[str, float]]]:
        """Return adjacency list with conductance weights."""
        adj: dict[str, list[tuple[str, float]]] = {nid: [] for nid in self.nodes}
        for e in self.edges:
            if e.source in adj:
                adj[e.source].append((e.target, e.conductance_nS))
            if e.target in adj:
                adj[e.target].append((e.source, e.conductance_nS))
        return adj

    def node_count(self) -> int:
        return len(self.nodes)

    def edge_count(self) -> int:
        return len(self.edges)
