"""
Energy management strategies for decentralized LLM inference.

Implements all seven strategies compared in the paper:

  Static baselines (fixed power mode, uniform random scheduling):
    S1 — Fixed 15W
    S2 — Fixed 30W
    S3 — Fixed 60W

  Dynamic strategies from Khoshsirat et al. (GLOBECOM 2024), §IV:
    D1 — Long-term scheduling   (Markov-informed static weights)
    D2 — Adaptive scheduling    (D1 + real-time PM1 penalty, Algorithm 1)
    D3 — Dynamic power mode     (per-device threshold-based PM switching)

  Novel contribution (this work):
    D4 — Energy-proportional scheduling (continuous battery-weighted selection)

Class hierarchy
---------------
  BaseScheduler          — abstract interface for all schedulers
    StaticScheduler      — S1, S2, S3
    LongTermScheduler    — D1
    AdaptiveScheduler    — D2
    EnergyProportionalScheduler — D4

  PowerModeController    — D3 (not a scheduler; operates per-device)

Usage
-----
  scheduler = LongTermScheduler(energy_models, config)
  scheduler.update(devices)            # call each tick
  chosen = scheduler.select_device(group, t)
"""

from __future__ import annotations

import abc
from typing import TYPE_CHECKING

import numpy as np
from numpy.random import Generator

if TYPE_CHECKING:
    from core.device import Device
    from core.energy import UniformEnergyModel

from core.markov import compute_scheduling_weights


# ---------------------------------------------------------------------------
# Abstract base class
# ---------------------------------------------------------------------------

class BaseScheduler(abc.ABC):
    """
    Abstract base for all scheduling strategies.

    Subclasses must implement select_device(). They may optionally override
    update() if they need to maintain real-time state.
    """

    def update(self, devices: list[Device]) -> None:
        """
        Called once per time slot before select_device().

        Default implementation is a no-op. Stateful strategies (D2, D4)
        override this to refresh their internal state from device observations.
        """

    @abc.abstractmethod
    def select_device(self, group: list[Device], t: int) -> Device | None:
        """
        Choose which device in `group` should handle the next job at slot t.

        Parameters
        ----------
        group : list of Device
            All devices in this group (available or not).
        t : int
            Current time slot (used by some strategies for logging/diurnal).

        Returns
        -------
        Device or None
            The chosen device, or None if no device is available.
        """

    def _available(self, group: list[Device]) -> list[Device]:
        """Return the subset of devices that can currently accept a job."""
        return [d for d in group if d.is_available()]


# ---------------------------------------------------------------------------
# S1, S2, S3 — Static schedulers
# ---------------------------------------------------------------------------

class StaticScheduler(BaseScheduler):
    """
    Static baseline scheduler (S1=15W, S2=30W, S3=60W).

    Sets every device to the given fixed power mode at construction and never
    changes it. Job selection is uniformly random among available devices.

    Parameters
    ----------
    power_mode : int
        Fixed power mode index: 1 (15W), 2 (30W), or 3 (60W).
    rng : Generator | None
        NumPy random generator for reproducible selection.
    """

    _VALID_MODES = {1, 2, 3}

    def __init__(self, power_mode: int, rng: Generator | None = None) -> None:
        if power_mode not in self._VALID_MODES:
            raise ValueError(f"power_mode must be in {self._VALID_MODES}, got {power_mode}")
        self.power_mode = power_mode
        self._rng = rng if rng is not None else np.random.default_rng()

    def select_device(self, group: list[Device], t: int) -> Device | None:  # noqa: ARG002
        available = self._available(group)
        if not available:
            return None
        idx = self._rng.integers(len(available))
        return available[idx]

    def __repr__(self) -> str:
        from core.device import POWER_WATTS
        return f"StaticScheduler(PM={self.power_mode}, {POWER_WATTS[self.power_mode]}W)"


# ---------------------------------------------------------------------------
# D1 — Long-term scheduler
# ---------------------------------------------------------------------------

class LongTermScheduler(BaseScheduler):
    """
    Long-term optimal scheduling (D1, §IV Static long-term optimal scheduling).

    Computes q_lim for each device offline using the semi-Markov model, then
    uses the normalised q_lim values as static scheduling weights (Eq. 6).
    Weights never change during the simulation run.

    Parameters
    ----------
    energy_models : list of UniformEnergyModel
        One model per device across ALL groups, in the same order as devices
        are stored in network.py.
    power_mode : int
        Active power mode used for the Markov calculation (typically the
        dynamic mode or a fixed mode for comparison).
    E_max, E_th_low, E_th_high, target_risk :
        Passed to compute_q_lim (see markov.py).
    rng : Generator | None
    """

    def __init__(
            self,
            energy_models: list[UniformEnergyModel],
            power_mode: int = 2,
            E_max: int = 100,
            E_th_low: int = 10,
            E_th_high: int = 20,
            target_risk: float = 0.01,
            rng: Generator | None = None,
    ) -> None:
        self._rng = rng if rng is not None else np.random.default_rng()
        self._energy_models = energy_models

        # Compute per-device q_lim offline (once)
        self._weights: np.ndarray = compute_scheduling_weights(
            energy_models,
            power_mode=power_mode,
            E_max=E_max,
            E_th_low=E_th_low,
            E_th_high=E_th_high,
            target_risk=target_risk,
        )
        # Map device_id → weight (set lazily on first select_device call)
        self._device_id_to_weight: dict[int, float] = {}

    def _ensure_weight_map(self, group: list[Device]) -> None:
        """Build device_id → weight mapping from the flat weights array."""
        if not self._device_id_to_weight and group:
            for device in group:
                idx = device.device_id
                if idx < len(self._weights):
                    self._device_id_to_weight[idx] = float(self._weights[idx])
                else:
                    self._device_id_to_weight[idx] = 1.0 / len(self._weights)

    def select_device(self, group: list[Device], t: int) -> Device | None:  # noqa: ARG002
        available = self._available(group)
        if not available:
            return None
        self._ensure_weight_map(group)
        w = np.array([
            self._device_id_to_weight.get(d.device_id, 1.0)
            for d in available
        ])
        total = w.sum()
        if total == 0:
            w = np.ones(len(available))
            total = float(len(available))
        probs = w / total
        idx = self._rng.choice(len(available), p=probs)
        return available[idx]

    @property
    def weights(self) -> np.ndarray:
        """Per-device scheduling weights (q_lim normalised)."""
        return self._weights.copy()

    def __repr__(self) -> str:
        return f"LongTermScheduler(n_devices={len(self._weights)})"


# ---------------------------------------------------------------------------
# D2 — Adaptive scheduler
# ---------------------------------------------------------------------------

class AdaptiveScheduler(LongTermScheduler):
    """
    Adaptive scheduling (D2, §IV Adaptive scheduling, Algorithm 1).

    Extends D1 by penalising devices currently in the lowest active power
    mode (PM=1, 15W) — indicating energy stress. At each tick, a fraction of
    their scheduling probability is redistributed to healthier peers.

    Algorithm 1 pseudocode:
      x ← long_term(q_lim)
      for each device i in PM1:
          z = α / N_layer
          x[i] = x[i] - (1 - z) * x[i]   # reduce by (1-z) fraction
      x ← x / Σ x                          # renormalise

    Parameters
    ----------
    alpha : float
        Tuning parameter α ∈ [0, N_layer]. Recommended: |{devices in PM1}|.
        Higher α means less aggressive penalisation of PM1 devices.
    """

    def __init__(
            self,
            energy_models: list[UniformEnergyModel],
            power_mode: int = 2,
            E_max: int = 100,
            E_th_low: int = 10,
            E_th_high: int = 20,
            target_risk: float = 0.01,
            alpha: float = 1.0,
            rng: Generator | None = None,
    ) -> None:
        super().__init__(
            energy_models=energy_models,
            power_mode=power_mode,
            E_max=E_max,
            E_th_low=E_th_low,
            E_th_high=E_th_high,
            target_risk=target_risk,
            rng=rng,
        )
        self.alpha = alpha
        # Set of device_ids currently observed to be in PM1 (15W)
        self._pm1_devices: set[int] = set()

    def update(self, devices: list[Device]) -> None:
        """Record which devices are currently in PM1 (energy-stressed)."""
        self._pm1_devices = {d.device_id for d in devices if d.power_mode == 1}

    def select_device(self, group: list[Device], t: int) -> Device | None:  # noqa: ARG002
        available = self._available(group)
        if not available:
            return None
        self._ensure_weight_map(group)
        N_layer = len(group)

        # Start from long-term weights
        w = np.array([
            self._device_id_to_weight.get(d.device_id, 1.0)
            for d in available
        ], dtype=float)

        # Penalise PM1 devices (Algorithm 1, lines 22–26)
        for i, device in enumerate(available):
            if device.device_id in self._pm1_devices:
                z = self.alpha / max(N_layer, 1)
                w[i] = w[i] - (1.0 - z) * w[i]

        # Renormalise (Algorithm 1, line 27)
        total = w.sum()
        if total <= 0:
            w = np.ones(len(available))
            total = float(len(available))
        probs = w / total

        idx = self._rng.choice(len(available), p=probs)
        return available[idx]

    def __repr__(self) -> str:
        return f"AdaptiveScheduler(n_devices={len(self._weights)}, alpha={self.alpha})"


# ---------------------------------------------------------------------------
# D3 — Dynamic power mode controller
# ---------------------------------------------------------------------------

class PowerModeController:
    """
    Dynamic power mode switching (D3, §V dynamic power mode).

    This is NOT a scheduler — it is a per-device controller that sets each
    device's active power mode based on its current battery fraction.

    Thresholds (from §V, found by manual exploration):
      battery < 40% → PM=1 (15W)
      40% ≤ battery < 60% → PM=2 (30W)
      battery ≥ 60% → PM=3 (60W)

    These thresholds are separate from the Markov-model hysteresis thresholds
    (E_th_low / E_th_high) and operate only on active devices (γ=1).

    Parameters
    ----------
    th_15_to_30 : float
        Battery fraction below which PM is set to 1 (15W). Default 0.40.
    th_30_to_60 : float
        Battery fraction above which PM is set to 3 (60W). Default 0.60.
    """

    def __init__(
            self,
            th_15_to_30: float = 0.40,
            th_30_to_60: float = 0.60,
    ) -> None:
        if not (0 < th_15_to_30 < th_30_to_60 < 1):
            raise ValueError(
                f"Require 0 < th_15_to_30 < th_30_to_60 < 1, "
                f"got {th_15_to_30}, {th_30_to_60}"
            )
        self.th_15_to_30 = th_15_to_30
        self.th_30_to_60 = th_30_to_60

    def update_device(self, device: Device) -> None:
        """
        Set the power mode on a single device based on its battery fraction.

        Called by the network orchestrator for each device on each tick,
        before scheduling decisions are made.
        Has no effect if the device is in power-saving mode (γ=0).
        """
        if device.is_in_power_saving:
            return
        bf = device.battery_fraction
        if bf < self.th_15_to_30:
            device.set_power_mode(1)
        elif bf < self.th_30_to_60:
            device.set_power_mode(2)
        else:
            device.set_power_mode(3)

    def update(self, devices: list[Device]) -> None:
        """Apply power mode updates to all devices in the list."""
        for d in devices:
            self.update_device(d)

    def __repr__(self) -> str:
        return (
            f"PowerModeController("
            f"th_15→30={self.th_15_to_30:.0%}, "
            f"th_30→60={self.th_30_to_60:.0%})"
        )


# ---------------------------------------------------------------------------
# D4 — Energy-proportional scheduler (novel contribution)
# ---------------------------------------------------------------------------

class EnergyProportionalScheduler(BaseScheduler):
    """
    Energy-proportional scheduling (D4, novel contribution of this work).

    At every scheduling decision, job probability is weighted continuously
    and proportionally to each available device's current battery level.
    Unlike D2, this mechanism is always active — no threshold crossing is
    required — providing smooth, gradient-like load balancing toward
    energy-rich devices.

    Distinction from D2
    -------------------
    D2 is reactive and binary: it reduces weights only when devices enter
    PM1 (15W), otherwise inheriting static D1 weights unchanged.
    D4 is continuous and proportional: weights are recomputed every
    scheduling call from live battery readings, always nudging load toward
    whoever has the most energy to spare.

    This approach has precedent in green data centre scheduling (workload
    shifted toward servers with more renewable energy), but has not
    previously been applied to decentralized LLM inference on EH edge devices.

    Parameters
    ----------
    rng : Generator | None
    """

    def __init__(self, rng: Generator | None = None) -> None:
        self._rng = rng if rng is not None else np.random.default_rng()

    def select_device(self, group: list[Device], t: int) -> Device | None:  # noqa: ARG002
        available = self._available(group)
        if not available:
            return None

        # Weight by current battery level
        w = np.array([float(d.battery) for d in available])
        total = w.sum()
        if total == 0:
            # All batteries at zero — fall back to uniform
            probs = np.ones(len(available)) / len(available)
        else:
            probs = w / total

        idx = self._rng.choice(len(available), p=probs)
        return available[idx]

    def __repr__(self) -> str:
        return "EnergyProportionalScheduler()"
