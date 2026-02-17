"""State-Space Dynamical Framework (Section 6).

Replaces static bit-modeling with dynamical system representation:

    dX/dt = A * X + B * U

Where:
    X = bioelectric state vector (Vm per node)
    A = connectivity matrix (gap junction weighted adjacency + leak)
    U = external modulation vector

Nexus must:
    - Identify stable attractors
    - Compute eigenvalues for stability analysis
    - Simulate perturbation recovery trajectory
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

from .substrate import SubstrateGraph


@dataclass
class StateVector:
    """Bioelectric state vector X: Vm for each node."""

    node_ids: list[str] = field(default_factory=list)
    values: list[float] = field(default_factory=list)     # Vm in mV per node

    @property
    def dimension(self) -> int:
        return len(self.values)

    def get(self, node_id: str) -> float:
        idx = self.node_ids.index(node_id) if node_id in self.node_ids else -1
        return self.values[idx] if idx >= 0 else 0.0

    def set(self, node_id: str, value: float) -> None:
        idx = self.node_ids.index(node_id) if node_id in self.node_ids else -1
        if idx >= 0:
            self.values[idx] = value

    def norm(self) -> float:
        return math.sqrt(sum(v * v for v in self.values))

    def copy(self) -> StateVector:
        return StateVector(node_ids=list(self.node_ids), values=list(self.values))


@dataclass
class SystemMatrices:
    """System matrices A (connectivity) and B (input coupling).

    A[i][j] represents influence of node j on node i.
    B[i][j] represents coupling of external input j to node i.

    Stored as lists-of-lists for zero-dependency operation.
    """

    A: list[list[float]] = field(default_factory=list)   # n x n connectivity
    B: list[list[float]] = field(default_factory=list)   # n x m input coupling
    node_ids: list[str] = field(default_factory=list)
    dimension: int = 0


def build_system_matrices(
    graph: SubstrateGraph,
    leak_rate: float = 0.1,
    conductance_scale: float = 0.01,
) -> SystemMatrices:
    """Build state-space matrices A and B from substrate graph.

    A[i][i] = -leak_rate (self-decay toward rest)
    A[i][j] = conductance_scale * G_ij (coupling from node j)

    B = identity (each node can receive independent external input).
    """
    node_ids = sorted(graph.nodes.keys())
    n = len(node_ids)
    idx_map = {nid: i for i, nid in enumerate(node_ids)}

    # Initialize A with leak on diagonal
    A = [[-leak_rate if i == j else 0.0 for j in range(n)] for i in range(n)]

    # Fill coupling from edges
    for edge in graph.edges:
        si = idx_map.get(edge.source)
        ti = idx_map.get(edge.target)
        if si is not None and ti is not None:
            coupling = conductance_scale * edge.conductance_nS
            A[si][ti] += coupling
            A[ti][si] += coupling

    # B = identity (external modulation per node)
    B = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]

    return SystemMatrices(A=A, B=B, node_ids=node_ids, dimension=n)


def state_from_graph(graph: SubstrateGraph) -> StateVector:
    """Extract current state vector X from substrate graph."""
    node_ids = sorted(graph.nodes.keys())
    values = [graph.nodes[nid].vm_mV for nid in node_ids]
    return StateVector(node_ids=node_ids, values=values)


def rest_state_from_graph(graph: SubstrateGraph) -> StateVector:
    """Extract resting state vector from substrate graph."""
    node_ids = sorted(graph.nodes.keys())
    values = [graph.nodes[nid].vm_rest_mV for nid in node_ids]
    return StateVector(node_ids=node_ids, values=values)


# ---------------------------------------------------------------------------
# Linear algebra operations (no numpy dependency)
# ---------------------------------------------------------------------------

def _mat_vec_mul(M: list[list[float]], v: list[float]) -> list[float]:
    """Matrix-vector multiplication."""
    n = len(v)
    result = [0.0] * n
    for i in range(n):
        s = 0.0
        for j in range(n):
            s += M[i][j] * v[j]
        result[i] = s
    return result


def _vec_add(a: list[float], b: list[float]) -> list[float]:
    return [ai + bi for ai, bi in zip(a, b)]


def _vec_scale(v: list[float], s: float) -> list[float]:
    return [vi * s for vi in v]


def _vec_sub(a: list[float], b: list[float]) -> list[float]:
    return [ai - bi for ai, bi in zip(a, b)]


def _vec_norm(v: list[float]) -> float:
    return math.sqrt(sum(vi * vi for vi in v))


# ---------------------------------------------------------------------------
# Eigenvalue computation (power iteration for dominant eigenvalue)
# ---------------------------------------------------------------------------

def compute_dominant_eigenvalue(
    A: list[list[float]],
    max_iter: int = 1000,
    tol: float = 1e-8,
) -> tuple[float, list[float]]:
    """Compute dominant eigenvalue via power iteration.

    Returns (eigenvalue, eigenvector).
    """
    n = len(A)
    if n == 0:
        return 0.0, []

    # Initial vector
    v = [1.0 / math.sqrt(n)] * n
    eigenvalue = 0.0

    for _ in range(max_iter):
        w = _mat_vec_mul(A, v)
        norm_w = _vec_norm(w)
        if norm_w < 1e-15:
            return 0.0, v
        new_eigenvalue = norm_w
        v_new = _vec_scale(w, 1.0 / norm_w)

        # Check for sign (negative eigenvalue)
        dot = sum(vi * wi for vi, wi in zip(v, w))
        if dot < 0:
            new_eigenvalue = -new_eigenvalue

        if abs(new_eigenvalue - eigenvalue) < tol:
            return new_eigenvalue, v_new
        eigenvalue = new_eigenvalue
        v = v_new

    return eigenvalue, v


def estimate_eigenvalue_spectrum(
    A: list[list[float]],
    n_eigenvalues: int = 5,
    max_iter: int = 500,
) -> list[float]:
    """Estimate top eigenvalues using deflation + power iteration.

    Returns eigenvalues in descending order of magnitude.
    """
    n = len(A)
    if n == 0:
        return []

    # Work on a copy
    M = [row[:] for row in A]
    eigenvalues: list[float] = []

    for _ in range(min(n_eigenvalues, n)):
        ev, vec = compute_dominant_eigenvalue(M, max_iter=max_iter)
        eigenvalues.append(round(ev, 8))

        # Deflate: M = M - ev * v * v^T
        for i in range(n):
            for j in range(n):
                M[i][j] -= ev * vec[i] * vec[j]

    return eigenvalues


# ---------------------------------------------------------------------------
# Stability analysis
# ---------------------------------------------------------------------------

@dataclass
class StabilityAnalysis:
    """Result of eigenvalue-based stability analysis."""

    eigenvalues: list[float] = field(default_factory=list)
    is_stable: bool = False                 # all eigenvalues have negative real part
    dominant_eigenvalue: float = 0.0
    spectral_gap: float = 0.0               # |lambda_1| - |lambda_2|
    convergence_rate: float = 0.0           # determined by dominant eigenvalue


def analyze_stability(matrices: SystemMatrices, n_eigenvalues: int = 5) -> StabilityAnalysis:
    """Perform eigenvalue-based stability analysis on the system."""
    eigenvalues = estimate_eigenvalue_spectrum(matrices.A, n_eigenvalues)

    if not eigenvalues:
        return StabilityAnalysis()

    dominant = eigenvalues[0]
    is_stable = all(ev < 0 for ev in eigenvalues)
    spectral_gap = abs(eigenvalues[0]) - abs(eigenvalues[1]) if len(eigenvalues) > 1 else abs(eigenvalues[0])
    convergence_rate = -dominant if dominant < 0 else 0.0

    return StabilityAnalysis(
        eigenvalues=eigenvalues,
        is_stable=is_stable,
        dominant_eigenvalue=dominant,
        spectral_gap=round(spectral_gap, 8),
        convergence_rate=round(convergence_rate, 8),
    )


# ---------------------------------------------------------------------------
# Attractor identification
# ---------------------------------------------------------------------------

@dataclass
class Attractor:
    """A stable attractor in the state space."""

    attractor_id: str
    state: StateVector = field(default_factory=StateVector)
    basin_radius_mV: float = 0.0
    convergence_steps: int = 0


def find_attractors(
    matrices: SystemMatrices,
    graph: SubstrateGraph,
    dt: float = 0.1,
    max_steps: int = 2000,
    convergence_threshold: float = 0.01,
) -> list[Attractor]:
    """Identify stable attractors by simulating from current state.

    Runs dX/dt = A*X with U=0 until convergence.
    """
    X = state_from_graph(graph)
    X_rest = rest_state_from_graph(graph)

    trajectory: list[StateVector] = [X.copy()]

    for step in range(max_steps):
        # dX/dt = A * (X - X_rest)  (deviation from rest)
        deviation = _vec_sub(X.values, X_rest.values)
        dX = _mat_vec_mul(matrices.A, deviation)
        X.values = _vec_add(X.values, _vec_scale(dX, dt))
        trajectory.append(X.copy())

        # Check convergence
        if _vec_norm(dX) * dt < convergence_threshold:
            return [Attractor(
                attractor_id="attractor_0",
                state=X.copy(),
                convergence_steps=step + 1,
            )]

    return [Attractor(
        attractor_id="attractor_0",
        state=X.copy(),
        convergence_steps=max_steps,
    )]


# ---------------------------------------------------------------------------
# Perturbation recovery trajectory
# ---------------------------------------------------------------------------

@dataclass
class RecoveryTrajectory:
    """Trajectory of recovery from perturbation in state space."""

    initial_state: StateVector = field(default_factory=StateVector)
    final_state: StateVector = field(default_factory=StateVector)
    trajectory_norms: list[float] = field(default_factory=list)   # deviation norm at each step
    recovery_steps: int = 0
    converged: bool = False
    dt: float = 0.1


def simulate_perturbation_trajectory(
    matrices: SystemMatrices,
    graph: SubstrateGraph,
    perturbation: list[float],
    dt: float = 0.1,
    max_steps: int = 2000,
    convergence_threshold: float = 0.01,
) -> RecoveryTrajectory:
    """Simulate recovery trajectory after applying a perturbation vector.

    dX/dt = A * (X - X_rest) + B * U
    where U = perturbation (applied at t=0 only).
    """
    X = state_from_graph(graph)
    X_rest = rest_state_from_graph(graph)
    initial = X.copy()

    # Apply perturbation at t=0
    n = min(len(X.values), len(perturbation))
    for i in range(n):
        X.values[i] += perturbation[i]

    norms: list[float] = [_vec_norm(_vec_sub(X.values, X_rest.values))]

    for step in range(max_steps):
        deviation = _vec_sub(X.values, X_rest.values)
        dX = _mat_vec_mul(matrices.A, deviation)
        X.values = _vec_add(X.values, _vec_scale(dX, dt))
        dev_norm = _vec_norm(_vec_sub(X.values, X_rest.values))
        norms.append(dev_norm)

        if dev_norm < convergence_threshold:
            return RecoveryTrajectory(
                initial_state=initial,
                final_state=X.copy(),
                trajectory_norms=norms,
                recovery_steps=step + 1,
                converged=True,
                dt=dt,
            )

    return RecoveryTrajectory(
        initial_state=initial,
        final_state=X.copy(),
        trajectory_norms=norms,
        recovery_steps=max_steps,
        converged=False,
        dt=dt,
    )


# ---------------------------------------------------------------------------
# Full state-space analysis
# ---------------------------------------------------------------------------

@dataclass
class StateSpaceAnalysis:
    """Complete state-space analysis result for Nexus."""

    stability: StabilityAnalysis = field(default_factory=StabilityAnalysis)
    attractors: list[Attractor] = field(default_factory=list)
    recovery: Optional[RecoveryTrajectory] = None
    system_dimension: int = 0


def analyze_state_space(
    graph: SubstrateGraph,
    leak_rate: float = 0.1,
    conductance_scale: float = 0.01,
    perturbation_mV: float = 20.0,
) -> StateSpaceAnalysis:
    """Run complete state-space analysis on a substrate graph."""
    matrices = build_system_matrices(graph, leak_rate, conductance_scale)
    stability = analyze_stability(matrices)
    attractors = find_attractors(matrices, graph)

    # Perturbation recovery trajectory
    n = matrices.dimension
    perturbation = [perturbation_mV if i < n // 3 else 0.0 for i in range(n)]
    recovery = simulate_perturbation_trajectory(matrices, graph, perturbation)

    return StateSpaceAnalysis(
        stability=stability,
        attractors=attractors,
        recovery=recovery,
        system_dimension=n,
    )
