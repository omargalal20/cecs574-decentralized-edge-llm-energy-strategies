"""
Energy arrival models for edge devices equipped with renewable energy sources.

Both models expose the same interface — sample(t) — so they are interchangeable
in any simulation component without code changes.

UNIT CONVENTION
---------------
All values returned by sample() and stored in low/high/peak/base are in **kJ**
(kilojoules per time slot). This matches the battery capacity (E_max [kJ]) and
job energy consumption (C_E [kJ]) in device.py, so no unit conversion is needed.

Mapping to paper notation (which uses J/slot):
  UniformEnergyModel(low=0.44, high=0.66)  <->  paper [440, 660] J/slot
  ENERGY_MEAN_BASELINE = 0.55 kJ/slot      <->  paper 550 J/slot

Reference: Khoshsirat et al. (GLOBECOM 2024), §III system model.
"""

import math

import numpy as np
from numpy.random import Generator


class UniformEnergyModel:
    """
    i.i.d. uniform energy arrival model (paper's original model, §III).

    Each call to sample() draws independently and uniformly from [low, high]
    (in Joules per time slot). Arrivals in different time slots are statistically
    independent, as assumed by the paper.

    For multi-slot stages (κ > 1) the caller is responsible for convolving κ
    independent draws; this class always returns a single-slot sample.
    """

    def __init__(self, low: float, high: float, rng: Generator | None = None) -> None:
        if low < 0 or high < low:
            raise ValueError(f"Require 0 ≤ low ≤ high, got low={low}, high={high}")
        self.low = low
        self.high = high
        self._rng = rng if rng is not None else np.random.default_rng()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def sample(self, t: int = 0) -> float:  # noqa: ARG002 (t unused but kept for API symmetry)
        """Return energy arriving this slot [kJ]. Time slot t is ignored."""
        return float(self._rng.uniform(self.low, self.high))

    def mean(self) -> float:
        """Expected energy per slot [kJ]."""
        return (self.low + self.high) / 2.0

    def pmf(self, e_values: np.ndarray) -> np.ndarray:
        """
        Probability mass function approximation on a discrete grid.

        The paper discretizes energy arrivals to integer units matching the
        battery discretization. Returns P(ΔIE = e) for each value in e_values.
        The uniform continuous distribution is approximated by dividing the
        probability uniformly across the discrete range [ceil(low), floor(high)].
        """
        lo = math.ceil(self.low)
        hi = math.floor(self.high)
        if hi < lo:
            # Degenerate case: all probability at the single nearest integer
            probs = np.zeros(len(e_values))
            nearest = round((self.low + self.high) / 2)
            probs[e_values == nearest] = 1.0
            return probs
        n_vals = hi - lo + 1
        probs = np.zeros(len(e_values))
        mask = (e_values >= lo) & (e_values <= hi)
        probs[mask] = 1.0 / n_vals
        return probs

    def __repr__(self) -> str:
        return f"UniformEnergyModel(low={self.low}, high={self.high})"


class DiurnalEnergyModel:
    """
    Sinusoidal diurnal energy arrival model (Experiment 3 extension).

    Models realistic solar harvesting: energy peaks at midday and drops to a
    minimum at night. The sinusoid is phase-shifted so that t=0 corresponds
    to midnight (minimum energy).

    Formula: E(t) = base + amplitude * sin(2π * t / period - π/2)
    which gives E(0) = base (midnight minimum) and E(period/4) = peak (noon).

    amplitude = (peak - base) / 2
    vertical shift: base + amplitude = (peak + base) / 2 + (peak - base) / 2 = peak ✓

    The mean energy per slot equals base + amplitude = (peak + base) / 2.
    """

    def __init__(
            self,
            peak: float,
            base: float,
            period_slots: int,
            rng: Generator | None = None,
    ) -> None:
        if base < 0:
            raise ValueError(f"base must be ≥ 0, got {base}")
        if peak < base:
            raise ValueError(f"peak must be ≥ base, got peak={peak}, base={base}")
        if period_slots < 1:
            raise ValueError(f"period_slots must be ≥ 1, got {period_slots}")
        self.peak = peak
        self.base = base
        self.period_slots = period_slots
        self._amplitude = (peak - base) / 2.0
        self._vertical_shift = base + self._amplitude
        self._rng = rng if rng is not None else np.random.default_rng()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def sample(self, t: int = 0) -> float:
        """
        Return energy arriving at slot t [kJ].

        A small Gaussian noise term (std = 5% of amplitude) is added to
        simulate realistic solar irradiance fluctuations.
        """
        deterministic = self._vertical_shift + self._amplitude * math.sin(
            2 * math.pi * t / self.period_slots - math.pi / 2
        )
        noise_std = 0.05 * self._amplitude
        noisy = deterministic + self._rng.normal(0.0, noise_std)
        return float(max(0.0, noisy))

    def deterministic_value(self, t: int) -> float:
        """Return the noise-free sinusoidal value at slot t [kJ]."""
        return self._vertical_shift + self._amplitude * math.sin(
            2 * math.pi * t / self.period_slots - math.pi / 2
        )

    def mean(self) -> float:
        """Expected energy per slot [kJ] (equals vertical shift of sinusoid)."""
        return self._vertical_shift

    def __repr__(self) -> str:
        return (
            f"DiurnalEnergyModel(peak={self.peak}, base={self.base}, "
            f"period_slots={self.period_slots})"
        )
