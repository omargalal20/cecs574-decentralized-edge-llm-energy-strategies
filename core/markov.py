"""
Semi-Markov model for a single edge device with energy harvesting.

Implements the model from Khoshsirat et al. (GLOBECOM 2024), §III–IV, to
compute q_lim — the maximum job arrival rate a device can sustain while keeping
its downtime probability (ξ_lim) below a user-defined risk threshold.

State space
-----------
Each state is the tuple S = (Q, E, γ):
  Q ∈ {0, 1}          — queue occupancy
  E ∈ {0, …, E_max}   — discrete battery level [kJ]
  γ ∈ {0, 1}          — 0 = power-saving, 1 = active

Total states: 2 × (E_max + 1) × 2

Six transition cases (§III)
----------------------------
  Case 1: Q=0, γ=1 → Q=0, γ=1  (no job, active, no arrival)
  Case 2: Q=0, γ=1 → Q=1, γ=1  (no job, active, new arrival)
  Case 3: Q=0, γ=0 → Q=0, γ?   (no job, power-saving, no processing)
  Case 4: Q=1, γ=0 → Q=1, γ?   (job waiting, power-saving, job not processed)
  Case 5: Q=1, γ=1 → Q=0, γ?   (job completes, no new arrival)
  Case 6: Q=1, γ=1 → Q=1, γ?   (job completes, new arrival)

Important: C_E(PM) is subtracted from ΔIE only in Cases 5 and 6 (processing
states). All other cases involve zero energy consumption.

Dwell times (T_S = κ × δ, §III)
---------------------------------
  Q=0, any γ     → κ = 1  (single slot)
  Q=1, γ=0       → κ = 1  (power-saving, job waiting but not processed)
  Q=1, γ=1, PM=1 → κ = 3  (15W: 300s / 100s per slot)
  Q=1, γ=1, PM=2 → κ = 2  (30W: 200s / 100s per slot)
  Q=1, γ=1, PM=3 → κ = 1  (60W: 100s / 100s per slot)

Key equations
-------------
  Eq. 3: ξ_lim = Σ_{S: E≤E_lim} π_S T_S / Σ_S π_S T_S
  Eq. 4: κ̄ = Σ_{S: Q=1,γ=1} π_S κ_S / Σ_{S: Q=1,γ=1} π_S
  Eq. 5: q_lim = min(q_energy_lim, 1/κ̄)
  Eq. 6: r_i = q_lim,i / Σ_j q_lim,j
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq

from core.energy import UniformEnergyModel

# ---------------------------------------------------------------------------
# Power-mode constants (matching device.py)
# ---------------------------------------------------------------------------
PROCESSING_SLOTS: dict[int, int] = {1: 3, 2: 2, 3: 1}
ENERGY_PER_JOB: dict[int, int] = {1: 26, 2: 22, 3: 23}


# ---------------------------------------------------------------------------
# State indexing helpers
# ---------------------------------------------------------------------------

def _build_state_index(E_max: int) -> tuple[dict, list]:
    """
    Build a bijection between (Q, E, γ) tuples and integer row/column indices.

    Returns
    -------
    state_to_idx : dict mapping (Q, E, γ) → int
    idx_to_state : list of (Q, E, γ) tuples indexed by int
    """
    idx_to_state: list[tuple[int, int, int]] = []
    state_to_idx: dict[tuple[int, int, int], int] = {}
    for Q in (0, 1):
        for E in range(E_max + 1):
            for gamma in (0, 1):
                state = (Q, E, gamma)
                state_to_idx[state] = len(idx_to_state)
                idx_to_state.append(state)
    return state_to_idx, idx_to_state


# ---------------------------------------------------------------------------
# Transition matrix builder
# ---------------------------------------------------------------------------

def _build_transition_matrix(
        q: float,
        power_mode: int,
        E_max: int,
        E_th_low: int,
        E_th_high: int,
        energy_model: UniformEnergyModel,
) -> np.ndarray:
    """
    Construct the embedded Markov chain transition matrix for a given job
    arrival rate q and fixed power mode.

    Parameters
    ----------
    q            : job arrival rate (probability of at least one arrival per stage)
    power_mode   : active PM index (1, 2, or 3)
    E_max        : maximum battery [kJ]
    E_th_low     : enter power-saving threshold [kJ]
    E_th_high    : exit power-saving threshold [kJ]
    energy_model : provides the discrete PMF for single-slot energy arrivals

    Returns
    -------
    P : (n_states × n_states) row-stochastic transition matrix
    """
    kappa = PROCESSING_SLOTS[power_mode]
    C_E = ENERGY_PER_JOB[power_mode]

    # p_m = P(≥1 arrival in κ slots) for active states
    # For idle/power-saving states κ=1, so p_m is the same as q.
    p_m_active = 1.0 - (1.0 - q) ** kappa
    p_m_idle = q  # κ=1 for Q=0 and power-saving states

    # Discrete energy arrival PMF on {0, …, E_max}
    e_grid = np.arange(E_max + 1, dtype=int)
    pmf_single = energy_model.pmf(e_grid)

    # PMF for κ i.i.d. draws (convolution of pmf_single with itself κ times)
    pmf_kappa = pmf_single.copy()
    for _ in range(kappa - 1):
        pmf_kappa = np.convolve(pmf_kappa, pmf_single)
    # pmf_kappa may be longer than E_max+1 after convolution; we will address
    # this when indexing by clamping the destination E to [0, E_max].

    state_to_idx, idx_to_state = _build_state_index(E_max)
    n = len(idx_to_state)
    P = np.zeros((n, n))

    def gamma_next(E_next: int, current_gamma: int) -> int:
        """Apply hysteresis rule to determine next γ."""
        if current_gamma == 1:
            # Active: enter power-saving if battery drops below E_th_low
            return 0 if E_next < E_th_low else 1
        else:
            # Power-saving: exit if battery rises above E_th_high
            return 1 if E_next > E_th_high else 0

    for idx, (Q, E, gamma) in enumerate(idx_to_state):

        # ------------------------------------------------------------------
        # Cases 1 & 2: Q=0, γ=1 (idle active)
        # No energy consumed. Stage duration κ=1, so p_m = p_m_idle = q.
        # ------------------------------------------------------------------
        if Q == 0 and gamma == 1:
            for delta_e, prob in enumerate(pmf_single):
                if prob == 0:
                    continue
                E_next = int(np.clip(E + delta_e, 0, E_max))
                gn = gamma_next(E_next, current_gamma=1)

                # Case 1: no arrival → stays Q=0
                dest1 = state_to_idx.get((0, E_next, gn))
                if dest1 is not None:
                    P[idx, dest1] += prob * (1.0 - p_m_idle)

                # Case 2: arrival → Q becomes 1
                # γ_next for the new state is the same (no battery change beyond
                # what we already computed); queue fills but gamma obeys hysteresis.
                dest2 = state_to_idx.get((1, E_next, gn))
                if dest2 is not None:
                    P[idx, dest2] += prob * p_m_idle

        # ------------------------------------------------------------------
        # Case 3: Q=0, γ=0 (idle power-saving)
        # No energy consumed. Job arrivals rejected (γ=0). κ=1.
        # ------------------------------------------------------------------
        elif Q == 0 and gamma == 0:
            for delta_e, prob in enumerate(pmf_single):
                if prob == 0:
                    continue
                E_next = int(np.clip(E + delta_e, 0, E_max))
                gn = gamma_next(E_next, current_gamma=0)
                dest = state_to_idx.get((0, E_next, gn))
                if dest is not None:
                    P[idx, dest] += prob

        # ------------------------------------------------------------------
        # Case 4: Q=1, γ=0 (job waiting, power-saving)
        # No energy consumed. Job not processed. Arrivals rejected. κ=1.
        # ------------------------------------------------------------------
        elif Q == 1 and gamma == 0:
            for delta_e, prob in enumerate(pmf_single):
                if prob == 0:
                    continue
                E_next = int(np.clip(E + delta_e, 0, E_max))
                gn = gamma_next(E_next, current_gamma=0)
                dest = state_to_idx.get((1, E_next, gn))
                if dest is not None:
                    P[idx, dest] += prob

        # ------------------------------------------------------------------
        # Cases 5 & 6: Q=1, γ=1 (processing)
        # C_E is deducted from the energy budget. Stage duration = κ.
        # p_m = p_m_active.
        # ------------------------------------------------------------------
        elif Q == 1 and gamma == 1:
            for delta_e, prob in enumerate(pmf_kappa):
                if prob == 0:
                    continue
                # E_next after harvesting and consuming C_E
                E_next = int(np.clip(E + delta_e - C_E, 0, E_max))
                gn = gamma_next(E_next, current_gamma=1)

                # Case 5: job completes, no new arrival → Q=0
                dest5 = state_to_idx.get((0, E_next, gn))
                if dest5 is not None:
                    P[idx, dest5] += prob * (1.0 - p_m_active)

                # Case 6: job completes, new arrival → Q=1
                dest6 = state_to_idx.get((1, E_next, gn))
                if dest6 is not None:
                    P[idx, dest6] += prob * p_m_active

    # Normalise rows that don't sum to exactly 1.0 due to floating point or
    # edge cases where all delta_e values are out of range.
    row_sums = P.sum(axis=1)
    nonzero = row_sums > 0
    P[nonzero] /= row_sums[nonzero, np.newaxis]

    return P


# ---------------------------------------------------------------------------
# Stationary distribution solver
# ---------------------------------------------------------------------------

def _stationary_distribution(P: np.ndarray) -> np.ndarray:
    """
    Compute the stationary distribution π of an embedded Markov chain.

    Solves π P = π subject to Σ π_i = 1 using the left-eigenvector method.
    Falls back to power iteration if the eigenvector approach is ill-conditioned.

    Returns
    -------
    pi : 1-D array of length n, entries summing to 1.
    """
    n = P.shape[0]

    # Method 1: solve the linear system (I - P^T) π = 0 with Σπ=1 constraint
    A = (np.eye(n) - P.T)
    A[-1, :] = 1.0  # replace last equation with normalisation constraint
    b = np.zeros(n)
    b[-1] = 1.0
    try:
        pi = np.linalg.solve(A, b)
        if np.all(pi >= -1e-10):
            pi = np.maximum(pi, 0.0)
            pi /= pi.sum()
            return pi
    except np.linalg.LinAlgError:
        pass

    # Fallback: power iteration
    pi = np.ones(n) / n
    for _ in range(10_000):
        pi_new = pi @ P
        if np.max(np.abs(pi_new - pi)) < 1e-12:
            break
        pi = pi_new
    pi = np.maximum(pi, 0.0)
    pi /= pi.sum()
    return pi


# ---------------------------------------------------------------------------
# ξ_lim computation (Equation 3)
# ---------------------------------------------------------------------------

def _compute_xi_lim(
        pi: np.ndarray,
        idx_to_state: list[tuple[int, int, int]],
        kappa: int,
        E_lim: int,
) -> float:
    """
    Compute ξ_lim: the fraction of real time the battery spends ≤ E_lim.

    Equation 3:
      ξ_lim = Σ_{S: E≤E_lim} π_S T_S  /  Σ_S π_S T_S

    where T_S is the dwell time of state S:
      - Q=0 or (Q=1, γ=0): T_S = 1 slot (κ=1)
      - Q=1, γ=1:           T_S = κ slots (processing duration)
    """
    numerator = 0.0
    denominator = 0.0
    for i, (Q, E, gamma) in enumerate(idx_to_state):
        T_S = kappa if (Q == 1 and gamma == 1) else 1
        contribution = pi[i] * T_S
        denominator += contribution
        if E <= E_lim:
            numerator += contribution

    return numerator / denominator if denominator > 0 else 0.0


# ---------------------------------------------------------------------------
# κ̄ computation (Equation 4)
# ---------------------------------------------------------------------------

def _compute_kappa_bar(
        pi: np.ndarray,
        idx_to_state: list[tuple[int, int, int]],
        kappa: int,
) -> float:
    """
    Compute κ̄: expected number of slots to process one job.

    Equation 4:
      κ̄ = Σ_{S: Q=1,γ=1} π_S κ_S  /  Σ_{S: Q=1,γ=1} π_S

    For a fixed power mode, κ_S = κ for all processing states.
    """
    pi_sum = 0.0
    pi_kappa_sum = 0.0
    for i, (Q, E, gamma) in enumerate(idx_to_state):  # noqa: B007
        if Q == 1 and gamma == 1:
            pi_sum += pi[i]
            pi_kappa_sum += pi[i] * kappa
    if pi_sum == 0:
        return float(kappa)
    return pi_kappa_sum / pi_sum


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_q_lim(
        energy_model: UniformEnergyModel,
        power_mode: int,
        E_max: int = 100,
        E_th_low: int = 10,
        E_th_high: int = 20,
        target_risk: float = 0.01,
) -> float:
    """
    Compute the maximum sustainable job arrival rate q_lim for a single device.

    Uses Brent's method to find q_energy_lim (the highest arrival rate keeping
    ξ_lim ≤ target_risk), then applies the processing-delay constraint of
    Equation 5 to obtain the final q_lim.

    Parameters
    ----------
    energy_model : UniformEnergyModel
        Energy arrival model for this device.
    power_mode : int
        Active power mode (1=15W, 2=30W, 3=60W).
    E_max : int
        Maximum battery [kJ].
    E_th_low : int
        Enter power-saving threshold [kJ].
    E_th_high : int
        Exit power-saving threshold [kJ].
    target_risk : float
        Maximum acceptable ξ_lim (default 0.01 = 1%).

    Returns
    -------
    q_lim : float
        Maximum sustainable job arrival rate [jobs/slot].
    """
    kappa = PROCESSING_SLOTS[power_mode]
    _, idx_to_state = _build_state_index(E_max)

    def risk_function(q: float) -> float:
        """ξ_lim(q) − target_risk. Root gives q_energy_lim."""
        P = _build_transition_matrix(
            q=q,
            power_mode=power_mode,
            E_max=E_max,
            E_th_low=E_th_low,
            E_th_high=E_th_high,
            energy_model=energy_model,
        )
        pi = _stationary_distribution(P)
        xi = _compute_xi_lim(pi, idx_to_state, kappa, E_lim=E_th_low)
        return xi - target_risk

    # Check boundary behaviour before calling brentq
    risk_at_low = risk_function(1e-4)
    risk_at_high = risk_function(0.999)

    if risk_at_low >= 0:
        # Even at near-zero arrival rate the risk exceeds target → device too
        # energy-poor to meet the constraint; return a near-zero rate.
        return 1e-4

    if risk_at_high <= 0:
        # Even at the maximum rate the risk stays below target → not binding;
        # fall through to apply only the processing-delay constraint.
        q_energy_lim = 0.999
    else:
        # Standard case: root lies in (1e-4, 0.999)
        q_energy_lim = brentq(risk_function, 1e-4, 0.999, xtol=1e-6, rtol=1e-6)

    # Equation 5: also limit by processing throughput
    # Recompute stationary distribution at q_energy_lim for κ̄
    P_final = _build_transition_matrix(
        q=q_energy_lim,
        power_mode=power_mode,
        E_max=E_max,
        E_th_low=E_th_low,
        E_th_high=E_th_high,
        energy_model=energy_model,
    )
    pi_final = _stationary_distribution(P_final)
    kappa_bar = _compute_kappa_bar(pi_final, idx_to_state, kappa)

    q_lim = min(q_energy_lim, 1.0 / kappa_bar)
    return float(q_lim)


def compute_scheduling_weights(
        devices_energy_models: list[UniformEnergyModel],
        power_mode: int,
        E_max: int = 100,
        E_th_low: int = 10,
        E_th_high: int = 20,
        target_risk: float = 0.01,
) -> np.ndarray:
    """
    Compute normalised scheduling weights r_i for a group of devices (Eq. 6).

    Parameters
    ----------
    devices_energy_models : list of UniformEnergyModel
        One model per device in the group.
    power_mode, E_max, E_th_low, E_th_high, target_risk :
        Passed through to compute_q_lim for each device.

    Returns
    -------
    weights : np.ndarray of shape (n_devices,), summing to 1.
    """
    q_lims = np.array([
        compute_q_lim(em, power_mode, E_max, E_th_low, E_th_high, target_risk)
        for em in devices_energy_models
    ])
    total = q_lims.sum()
    if total == 0:
        return np.ones(len(q_lims)) / len(q_lims)
    return q_lims / total
