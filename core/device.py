"""
Edge device model for decentralized LLM inference with energy harvesting.

Implements the stateful device described in Khoshsirat et al. (GLOBECOM 2024),
§II–III. Each Device instance represents one Nvidia Jetson AGX Orin (or
equivalent) running a portion of an LLM, powered by a battery charged via
a renewable energy source (e.g., solar panel).

State at stage m: S_m = (Q_m, E_m, γ_m)
  Q_m  ∈ {0, 1}          — queue occupancy (0 = empty, 1 = job present)
  E_m  ∈ {0, …, E_max}   — discrete battery level [kJ]
  γ_m  ∈ {0, 1}          — mode (0 = power-saving, 1 = active)

Battery update (Equation 1):
  E_{m+1} = clip(E_m + ΔIE_m − C_E(PM), 0, E_max)

C_E(PM) is only subtracted in processing states (Q=1, γ=1). For all other
states the consumed energy is 0 — the device is either idle or power-saving.
"""

import numpy as np

# Power-mode index → wattage mapping (50W excluded per paper §V)
POWER_WATTS: dict[int, int] = {1: 15, 2: 30, 3: 60}

# Power-mode index → processing slots per job
PROCESSING_SLOTS: dict[int, int] = {1: 3, 2: 2, 3: 1}

# Power-mode index → total system energy per job [kJ] (race-to-idle effect)
ENERGY_PER_JOB: dict[int, int] = {1: 26, 2: 22, 3: 23}


class Device:
    """
    Stateful edge device updated one stage at a time.

    Parameters
    ----------
    device_id : int
        Unique identifier (used for logging and reproducibility).
    E_max : int
        Maximum battery capacity [kJ].
    E_th_low : int
        Battery level below which the device enters power-saving mode.
        Corresponds to E_th in the paper (default 10 kJ = 10% of 100 kJ).
    E_th_high : int
        Battery level above which the device exits power-saving mode.
        Corresponds to E'_th in the paper. Must be > E_th_low.
    initial_battery : int | None
        Starting battery level [kJ]. Defaults to E_max (fully charged).
    initial_power_mode : int
        Starting active power mode (1, 2, or 3). Only relevant when the
        device starts in active state (γ=1).
    """

    def __init__(
            self,
            device_id: int,
            E_max: int = 100,
            E_th_low: int = 10,
            E_th_high: int = 20,
            initial_battery: int | None = None,
            initial_power_mode: int = 2,
    ) -> None:
        if E_th_low >= E_th_high:
            raise ValueError(
                f"E_th_low ({E_th_low}) must be < E_th_high ({E_th_high})"
            )
        if initial_power_mode not in POWER_WATTS:
            raise ValueError(f"initial_power_mode must be in {list(POWER_WATTS)}")

        self.device_id = device_id
        self.E_max = E_max
        self.E_th_low = E_th_low
        self.E_th_high = E_th_high

        self._battery: int = E_max if initial_battery is None else int(initial_battery)
        self._power_mode: int = initial_power_mode  # 1, 2, or 3 (active modes)
        self._gamma: int = 1  # 0 = power-saving, 1 = active
        self._queue: int = 0  # 0 = empty, 1 = job present
        self._slots_remaining: int = 0  # slots left to finish current job

    # ------------------------------------------------------------------
    # Core simulation step
    # ------------------------------------------------------------------

    def step(self, harvested: float) -> dict:
        """
        Advance the device by one time slot.

        Called once per slot (not per stage) by the network orchestrator.
        Performs the Equation 1 battery update. C_E is only deducted when
        the device is actively processing (Q=1, γ=1) and this is the last
        slot of the current job (i.e., the stage boundary).

        Parameters
        ----------
        harvested : float
            Energy [kJ] collected from the renewable source this slot.
            The energy model is expected to return values in kJ to match
            the battery (E_max [kJ]) and job consumption (C_E [kJ]) units.

        Returns
        -------
        dict with keys: battery, gamma, power_mode, queue, job_completed
        """
        harvested_kj = harvested  # energy model already returns kJ

        # Determine energy consumed this slot.
        # C_E(PM) is charged only at the end of a processing stage (the slot
        # where the job finishes). Between stage boundaries the device consumes
        # zero additional battery (the paper models energy at stage granularity).
        consumed_kj = 0.0
        job_completed = False

        if self._gamma == 1 and self._queue == 1:
            self._slots_remaining -= 1
            if self._slots_remaining <= 0:
                # Stage boundary: deduct C_E(PM) for the completed job
                consumed_kj = float(ENERGY_PER_JOB[self._power_mode])
                self._queue = 0
                job_completed = True

        # Equation 1 battery update
        new_battery = np.clip(
            self._battery + harvested_kj - consumed_kj,
            0,
            self.E_max,
        )
        self._battery = int(round(new_battery))

        # Hysteresis power-saving logic
        self._update_gamma()

        return {
            "battery": self._battery,
            "gamma": self._gamma,
            "power_mode": self._power_mode if self._gamma == 1 else 0,
            "queue": self._queue,
            "job_completed": job_completed,
        }

    # ------------------------------------------------------------------
    # Job queue management
    # ------------------------------------------------------------------

    def accept_job(self) -> bool:
        """
        Assign a job to this device.

        Returns True if accepted, False if the device is unavailable
        (power-saving or queue already full).
        """
        if not self.is_available():
            return False
        self._queue = 1
        self._slots_remaining = PROCESSING_SLOTS[self._power_mode]
        return True

    def is_available(self) -> bool:
        """
        True if the device can accept a new job right now.

        A device is unavailable when it is in power-saving mode (γ=0)
        or already has a job in its queue (Q=1).
        """
        return self._gamma == 1 and self._queue == 0

    # ------------------------------------------------------------------
    # Power mode control (used by D3 PowerModeController and network)
    # ------------------------------------------------------------------

    def set_power_mode(self, mode: int) -> None:
        """
        Override the active power mode (1, 2, or 3).

        Only effective when γ=1. Has no effect when in power-saving mode.
        """
        if mode not in POWER_WATTS:
            raise ValueError(f"mode must be in {list(POWER_WATTS)}, got {mode}")
        self._power_mode = mode

    # ------------------------------------------------------------------
    # Read-only properties
    # ------------------------------------------------------------------

    @property
    def battery(self) -> int:
        """Current battery level [kJ]."""
        return self._battery

    @property
    def battery_fraction(self) -> float:
        """Current battery level as a fraction of E_max [0.0–1.0]."""
        return self._battery / self.E_max

    @property
    def gamma(self) -> int:
        """Current γ: 0 = power-saving, 1 = active."""
        return self._gamma

    @property
    def power_mode(self) -> int:
        """
        Current power mode index.

        Returns 0 when in power-saving mode (γ=0), otherwise 1, 2, or 3.
        """
        return self._power_mode if self._gamma == 1 else 0

    @property
    def queue(self) -> int:
        """Queue occupancy: 0 = empty, 1 = job present."""
        return self._queue

    @property
    def is_in_power_saving(self) -> bool:
        """True when the device is in power-saving mode (γ=0)."""
        return self._gamma == 0

    def state_tuple(self) -> tuple[int, int, int]:
        """Return the state tuple (Q, E, γ) as used in the semi-Markov model."""
        return (self._queue, self._battery, self._gamma)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _update_gamma(self) -> None:
        """
        Apply hysteresis to γ after each battery update.

        Entering power-saving: E < E_th_low  (strict, paper §III)
        Exiting  power-saving: E > E_th_high (strict, paper §III)
        """
        if self._gamma == 1:
            # Active → check if we should enter power-saving
            if self._battery < self.E_th_low:
                self._gamma = 0
                # Reject any pending job (device goes offline)
                self._queue = 0
                self._slots_remaining = 0
        else:
            # Power-saving → check if we can exit
            if self._battery > self.E_th_high:
                self._gamma = 1

    def __repr__(self) -> str:
        return (
            f"Device(id={self.device_id}, "
            f"battery={self._battery}/{self.E_max} kJ, "
            f"gamma={self._gamma}, "
            f"PM={self.power_mode}, "
            f"queue={self._queue})"
        )
