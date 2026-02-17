"""Homeostatic Recovery Model — BESS Upgrade (Section 5).

Feedback rule:
    If global activity > threshold:  apply scaling factor S
    If activity < basal range:       increase excitability proportionally

Simulation targets:
    - Recovery time after adversarial bias
    - Stability under repeated perturbation
    - Energy cost of homeostatic correction
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

from .substrate import SubstrateGraph, SubstrateNode


@dataclass
class HomeostaticParams:
    """Parameters for the homeostatic scaling feedback loop."""

    activity_threshold_upper: float = 0.7       # global activity above this triggers downscaling
    activity_threshold_lower: float = 0.3       # global activity below this triggers upscaling
    basal_activity: float = 0.5                 # target activity level
    scaling_factor_down: float = 0.9            # multiplicative downscale per step
    scaling_factor_up: float = 1.1              # multiplicative upscale per step
    max_excitability: float = 3.0               # upper bound on excitability multiplier
    min_excitability: float = 0.1               # lower bound on excitability multiplier
    # Energy cost parameters
    atp_per_scaling_event: float = 100.0        # ATP molecules per scaling correction
    energy_per_atp_J: float = 5.0e-20           # ~30.5 kJ/mol per ATP


@dataclass
class NodeExcitability:
    """Per-node excitability state managed by homeostatic feedback."""

    node_id: str
    excitability: float = 1.0                   # multiplicative factor on responsiveness
    activity: float = 0.5                       # current activity level (0-1)


@dataclass
class HomeostaticState:
    """Global homeostatic state for the substrate."""

    params: HomeostaticParams = field(default_factory=HomeostaticParams)
    node_states: dict[str, NodeExcitability] = field(default_factory=dict)
    global_activity: float = 0.5
    step_count: int = 0
    total_energy_J: float = 0.0
    scaling_events: int = 0

    def initialize_from_graph(self, graph: SubstrateGraph) -> None:
        """Initialize node excitability states from substrate graph."""
        for nid, node in graph.nodes.items():
            # Activity derived from Vm relative to resting
            activity = _vm_to_activity(node.vm_mV, node.vm_rest_mV)
            self.node_states[nid] = NodeExcitability(
                node_id=nid,
                excitability=1.0,
                activity=activity,
            )
        self._update_global_activity()

    def _update_global_activity(self) -> None:
        if self.node_states:
            self.global_activity = sum(
                ns.activity for ns in self.node_states.values()
            ) / len(self.node_states)

    def step(self) -> HomeostaticStepResult:
        """Execute one homeostatic feedback step.

        If global activity > threshold: apply scaling factor S (downscale)
        If activity < basal range: increase excitability proportionally
        """
        self.step_count += 1
        p = self.params
        action = "none"
        energy_this_step = 0.0

        if self.global_activity > p.activity_threshold_upper:
            # Downscale: reduce excitability
            action = "downscale"
            for ns in self.node_states.values():
                ns.excitability = max(p.min_excitability, ns.excitability * p.scaling_factor_down)
                ns.activity = min(1.0, ns.activity * ns.excitability)
            self.scaling_events += 1
            energy_this_step = p.atp_per_scaling_event * p.energy_per_atp_J * len(self.node_states)

        elif self.global_activity < p.activity_threshold_lower:
            # Upscale: increase excitability proportionally
            action = "upscale"
            deficit = p.basal_activity - self.global_activity
            scale = p.scaling_factor_up + deficit  # proportional boost
            for ns in self.node_states.values():
                ns.excitability = min(p.max_excitability, ns.excitability * scale)
                ns.activity = min(1.0, ns.activity * ns.excitability)
            self.scaling_events += 1
            energy_this_step = p.atp_per_scaling_event * p.energy_per_atp_J * len(self.node_states)

        self.total_energy_J += energy_this_step
        self._update_global_activity()

        return HomeostaticStepResult(
            step=self.step_count,
            action=action,
            global_activity=round(self.global_activity, 6),
            energy_J=energy_this_step,
        )


@dataclass
class HomeostaticStepResult:
    """Result of a single homeostatic feedback step."""

    step: int = 0
    action: str = "none"            # "upscale", "downscale", "none"
    global_activity: float = 0.0
    energy_J: float = 0.0


def _vm_to_activity(vm_mV: float, vm_rest_mV: float, vm_range_mV: float = 40.0) -> float:
    """Map membrane potential to activity level [0, 1].

    Sigmoid centered at vm_rest, spanning vm_range.
    """
    x = (vm_mV - vm_rest_mV) / (vm_range_mV / 4.0)
    return 1.0 / (1.0 + math.exp(-x))


# ---------------------------------------------------------------------------
# Simulation routines
# ---------------------------------------------------------------------------

@dataclass
class HomeostaticRecoveryResult:
    """Result of a homeostatic recovery simulation."""

    # Recovery time
    recovery_steps: int = 0
    recovery_complete: bool = False
    final_activity: float = 0.0
    # Stability under repeated perturbation
    perturbation_count: int = 0
    stability_maintained: bool = False
    activity_variance: float = 0.0
    # Energy cost
    total_energy_J: float = 0.0
    energy_per_correction_J: float = 0.0
    total_scaling_events: int = 0
    # Trajectory
    activity_trajectory: list[float] = field(default_factory=list)


def simulate_adversarial_recovery(
    graph: SubstrateGraph,
    bias_mV: float = 30.0,
    fraction_biased: float = 0.5,
    max_steps: int = 200,
    params: Optional[HomeostaticParams] = None,
    seed: Optional[int] = None,
) -> HomeostaticRecoveryResult:
    """Simulate recovery after adversarial bias injection.

    Applies a voltage bias to a fraction of nodes, then runs
    homeostatic feedback until activity returns to basal range.
    """
    import random as _rng
    if seed is not None:
        _rng.seed(seed)

    if params is None:
        params = HomeostaticParams()

    # Apply adversarial bias
    node_ids = list(graph.nodes.keys())
    n_bias = max(1, int(len(node_ids) * fraction_biased))
    biased = _rng.sample(node_ids, n_bias)
    for nid in biased:
        graph.nodes[nid].vm_mV += bias_mV

    # Initialize homeostatic state
    state = HomeostaticState(params=params)
    state.initialize_from_graph(graph)

    trajectory: list[float] = [state.global_activity]

    for _ in range(max_steps):
        result = state.step()
        trajectory.append(result.global_activity)
        if params.activity_threshold_lower <= result.global_activity <= params.activity_threshold_upper:
            return HomeostaticRecoveryResult(
                recovery_steps=state.step_count,
                recovery_complete=True,
                final_activity=result.global_activity,
                total_energy_J=state.total_energy_J,
                energy_per_correction_J=(
                    state.total_energy_J / state.scaling_events
                    if state.scaling_events > 0 else 0.0
                ),
                total_scaling_events=state.scaling_events,
                activity_trajectory=trajectory,
            )

    return HomeostaticRecoveryResult(
        recovery_steps=max_steps,
        recovery_complete=False,
        final_activity=state.global_activity,
        total_energy_J=state.total_energy_J,
        energy_per_correction_J=(
            state.total_energy_J / state.scaling_events
            if state.scaling_events > 0 else 0.0
        ),
        total_scaling_events=state.scaling_events,
        activity_trajectory=trajectory,
    )


def simulate_repeated_perturbation(
    graph: SubstrateGraph,
    n_perturbations: int = 10,
    perturbation_mV: float = 15.0,
    fraction_perturbed: float = 0.2,
    steps_between: int = 50,
    params: Optional[HomeostaticParams] = None,
    seed: Optional[int] = None,
) -> HomeostaticRecoveryResult:
    """Simulate stability under repeated perturbation.

    Applies n_perturbations sequential voltage kicks with
    homeostatic recovery between each.
    """
    import random as _rng
    if seed is not None:
        _rng.seed(seed)

    if params is None:
        params = HomeostaticParams()

    state = HomeostaticState(params=params)
    state.initialize_from_graph(graph)

    trajectory: list[float] = [state.global_activity]
    node_ids = list(graph.nodes.keys())

    for p_idx in range(n_perturbations):
        # Apply perturbation
        n_kick = max(1, int(len(node_ids) * fraction_perturbed))
        kicked = _rng.sample(node_ids, n_kick)
        sign = 1.0 if p_idx % 2 == 0 else -1.0
        for nid in kicked:
            graph.nodes[nid].vm_mV += sign * perturbation_mV
            ns = state.node_states.get(nid)
            if ns:
                ns.activity = _vm_to_activity(graph.nodes[nid].vm_mV, graph.nodes[nid].vm_rest_mV)

        state._update_global_activity()

        # Recovery steps
        for _ in range(steps_between):
            result = state.step()
            trajectory.append(result.global_activity)

    # Compute activity variance over trajectory
    mean_act = sum(trajectory) / len(trajectory) if trajectory else 0.0
    variance = sum((a - mean_act) ** 2 for a in trajectory) / len(trajectory) if trajectory else 0.0
    final_in_range = (
        params.activity_threshold_lower <= state.global_activity <= params.activity_threshold_upper
    )

    return HomeostaticRecoveryResult(
        recovery_steps=state.step_count,
        recovery_complete=final_in_range,
        final_activity=state.global_activity,
        perturbation_count=n_perturbations,
        stability_maintained=final_in_range,
        activity_variance=round(variance, 8),
        total_energy_J=state.total_energy_J,
        energy_per_correction_J=(
            state.total_energy_J / state.scaling_events
            if state.scaling_events > 0 else 0.0
        ),
        total_scaling_events=state.scaling_events,
        activity_trajectory=trajectory,
    )
