"""
Network orchestrator for decentralized LLM inference simulation.

Implements the simulation time loop described in Khoshsirat et al.
(GLOBECOM 2024), §II–IV (see Algorithm 1 for the authoritative pseudocode).

Architecture
------------
- 3 groups × 3 devices = 9 total devices (paper §V topology)
- Each device has its own energy model (different solar profiles)
- One scheduler handles job routing across all groups
- An optional PowerModeController (D3) updates device PMs each tick

Tick order (strict, per Algorithm 1)
--------------------------------------
  1. For each device: harvest energy, call device.step()
  2. [D3 only] Update power mode for each device
  3. [D2/D4] Update scheduler state from current device observations
  4. For each group:
       a. Check if an arriving job can be scheduled → select device
       b. Drop jobs when no device is available (queue full or all power-saving)
  5. Record metrics

Metrics recorded per slot
--------------------------
  batteries          : (n_devices,) array of battery levels [kJ]
  gammas             : (n_devices,) array of γ values
  power_modes        : (n_devices,) array of power mode indices
  jobs_completed     : total jobs finished this slot across all groups
  jobs_dropped       : total jobs dropped (no available device)
  jobs_arrived       : total jobs that arrived this slot
  inactive_fraction  : fraction of devices in power-saving mode (γ=0)
"""

from __future__ import annotations

import numpy as np
from numpy.random import Generator

from core.device import Device
from core.energy import UniformEnergyModel, DiurnalEnergyModel
from core.strategies import (
    BaseScheduler,
    PowerModeController,
    StaticScheduler,
)
from experiments.config import SimConfig, DEFAULT_CONFIG

# Type alias for energy models
EnergyModel = UniformEnergyModel | DiurnalEnergyModel


class Network:
    """
    Simulation network: 3 groups × 3 devices running LLM inference.

    Parameters
    ----------
    scheduler : BaseScheduler
        Job routing strategy (S1–S3, D1, D2, or D4).
    power_controller : PowerModeController | None
        D3 power mode controller. Pass None for all other strategies.
    energy_configs : list of (low, high) tuples | None
        Per-device uniform energy arrival bounds [kJ/slot].
        Length must equal n_groups × devices_per_group.
        If None, uses the default from SimConfig (symmetric ±20% spread).
    config : SimConfig
        Simulation parameters (default: DEFAULT_CONFIG).
    rng : Generator | None
        NumPy random generator (controls job arrivals and energy sampling).
    """

    def __init__(
            self,
            scheduler: BaseScheduler,
            power_controller: PowerModeController | None = None,
            energy_configs: list[tuple[float, float]] | None = None,
            config: SimConfig = DEFAULT_CONFIG,
            rng: Generator | None = None,
            e_max_per_device: list[int] | None = None,
    ) -> None:
        self.config = config
        self._scheduler = scheduler
        self._power_controller = power_controller
        self._rng = rng if rng is not None else np.random.default_rng()

        n_total = config.N_GROUPS * config.DEVICES_PER_GROUP

        # Build energy models
        if energy_configs is None:
            energy_configs = _default_energy_configs(config)
        if len(energy_configs) != n_total:
            raise ValueError(
                f"energy_configs length ({len(energy_configs)}) must equal "
                f"n_groups × devices_per_group ({n_total})"
            )
        self._energy_models: list[EnergyModel] = [
            UniformEnergyModel(low, high, rng=self._rng)
            for low, high in energy_configs
        ]

        # Per-device battery capacities (supports heterogeneous Exp 4)
        if e_max_per_device is None:
            e_max_per_device = [config.E_MAX] * n_total
        if len(e_max_per_device) != n_total:
            raise ValueError(
                f"e_max_per_device length ({len(e_max_per_device)}) must equal "
                f"n_groups × devices_per_group ({n_total})"
            )
        self._e_max_per_device = e_max_per_device

        # Build devices (device_id = flat index across all groups)
        pm = _initial_power_mode(scheduler)
        self._devices: list[Device] = [
            Device(
                device_id=i,
                E_max=e_max_per_device[i],
                E_th_low=config.E_TH_LOW,
                E_th_high=config.E_TH_HIGH_MARKOV,
                initial_battery=e_max_per_device[i],
                initial_power_mode=pm,
            )
            for i in range(n_total)
        ]

        # Group structure: groups[g] = list of Device for group g
        dpg = config.DEVICES_PER_GROUP
        self._groups: list[list[Device]] = [
            self._devices[g * dpg: (g + 1) * dpg]
            for g in range(config.N_GROUPS)
        ]

    # ------------------------------------------------------------------
    # Public simulation runners
    # ------------------------------------------------------------------

    def run(self, T: int | None = None, seed: int | None = None) -> dict[str, np.ndarray]:
        """
        Run one simulation of length T slots.

        Parameters
        ----------
        T : int | None
            Number of time slots. Defaults to config.T.
        seed : int | None
            If given, re-seeds the internal RNG before running (enables
            independent replications with different seeds).

        Returns
        -------
        metrics : dict
            Keys and shapes:
              'batteries'         : (T, n_devices)
              'gammas'            : (T, n_devices)
              'power_modes'       : (T, n_devices)
              'jobs_completed'    : (T,)
              'jobs_dropped'      : (T,)
              'jobs_arrived'      : (T,)
              'inactive_fraction' : (T,)
        """
        if seed is not None:
            self._rng = np.random.default_rng(seed)
            for i, em in enumerate(self._energy_models):
                em.reseed(np.random.default_rng(seed + 1000 + i))
            self._scheduler._rng = np.random.default_rng(seed + 2000)
        T = T if T is not None else self.config.T

        n_dev = len(self._devices)
        batteries = np.zeros((T, n_dev), dtype=float)
        gammas = np.zeros((T, n_dev), dtype=int)
        power_modes = np.zeros((T, n_dev), dtype=int)
        completed = np.zeros(T, dtype=int)
        dropped = np.zeros(T, dtype=int)
        arrived = np.zeros(T, dtype=int)

        self._reset()

        for t in range(T):
            # ----------------------------------------------------------
            # Step 1: energy harvest + battery update for every device
            # ----------------------------------------------------------
            slot_completed = 0
            for i, device in enumerate(self._devices):
                harvested = self._energy_models[i].sample(t)
                result = device.step(harvested)
                if result["job_completed"]:
                    slot_completed += 1

            # ----------------------------------------------------------
            # Step 2 (D3 only): update power modes
            # ----------------------------------------------------------
            if self._power_controller is not None:
                self._power_controller.update(self._devices)

            # ----------------------------------------------------------
            # Step 3 (D2 / D4): update scheduler state
            # ----------------------------------------------------------
            self._scheduler.update(self._devices)

            # ----------------------------------------------------------
            # Step 4: job arrivals and scheduling per group
            # ----------------------------------------------------------
            slot_dropped = 0
            slot_arrived = 0

            for group in self._groups:
                # Bernoulli job arrival
                if self._rng.random() < self.config.JOB_ARRIVAL_PROB:
                    slot_arrived += 1
                    chosen = self._scheduler.select_device(group, t)
                    if chosen is not None and chosen.accept_job():
                        pass  # job accepted
                    else:
                        slot_dropped += 1

            completed[t] = slot_completed
            dropped[t] = slot_dropped
            arrived[t] = slot_arrived

            # ----------------------------------------------------------
            # Step 5: record metrics
            # ----------------------------------------------------------
            for i, device in enumerate(self._devices):
                batteries[t, i] = device.battery
                gammas[t, i] = device.gamma
                power_modes[t, i] = device.power_mode

        inactive_frac = (gammas == 0).mean(axis=1)
        return {
            "batteries": batteries,
            "gammas": gammas,
            "power_modes": power_modes,
            "jobs_completed": completed,
            "jobs_dropped": dropped,
            "jobs_arrived": arrived,
            "inactive_fraction": inactive_frac,
        }

    def run_batch(
            self,
            T: int | None = None,
            n_iterations: int | None = None,
            seed_start: int = 0,
    ) -> dict[str, np.ndarray]:
        """
        Run multiple independent replications and return mean ± std metrics.

        Parameters
        ----------
        T : int | None
            Slots per run. Defaults to config.T.
        n_iterations : int | None
            Number of replications. Defaults to config.N_ITERATIONS.
        seed_start : int
            First seed; subsequent runs use seed_start+1, seed_start+2, …

        Returns
        -------
        dict with keys 'mean_<metric>' and 'std_<metric>' for each metric,
        plus 'mean_inactive_fraction' and 'std_inactive_fraction'.
        """
        T = T if T is not None else self.config.T
        n_iter = n_iterations if n_iterations is not None else self.config.N_ITERATIONS

        all_results: list[dict] = []
        for i in range(n_iter):
            result = self.run(T=T, seed=seed_start + i)
            all_results.append(result)

        # Aggregate scalar metrics (summed over T per run, then stats over runs)
        agg: dict[str, np.ndarray] = {}
        scalar_keys = ["jobs_completed", "jobs_dropped", "jobs_arrived"]
        ts_keys = ["inactive_fraction"]

        for key in scalar_keys:
            values = np.array([r[key].sum() for r in all_results])
            agg[f"mean_{key}"] = values.mean()
            agg[f"std_{key}"] = values.std()

        for key in ts_keys:
            values = np.array([r[key].mean() for r in all_results])
            agg[f"mean_{key}"] = values.mean()
            agg[f"std_{key}"] = values.std()

        # Battery level (mean over devices and time, per run)
        batt_means = np.array([r["batteries"].mean() for r in all_results])
        agg["mean_battery"] = float(batt_means.mean())
        agg["std_battery"]  = float(batt_means.std())

        # Normalised throughput: completed / arrived per run
        completed_arr = np.array([r["jobs_completed"].sum() for r in all_results])
        arrived_arr   = np.array([r["jobs_arrived"].sum()   for r in all_results])
        throughput = np.where(arrived_arr > 0, completed_arr / arrived_arr, 0.0)
        agg["mean_throughput"] = float(throughput.mean())
        agg["std_throughput"]  = float(throughput.std())

        return agg

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _reset(self) -> None:
        """Reset all devices to fully charged, active state."""
        pm = _initial_power_mode(self._scheduler)
        for device in self._devices:
            device._battery = device.E_max  # respects per-device capacity
            device._gamma = 1
            device._queue = 0
            device._slots_remaining = 0
            device._power_mode = pm

    @property
    def devices(self) -> list[Device]:
        """All devices in the network."""
        return self._devices

    @property
    def groups(self) -> list[list[Device]]:
        """Devices organised by group."""
        return self._groups

    @property
    def energy_models(self) -> list[EnergyModel]:
        """Per-device energy arrival models."""
        return self._energy_models


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------

def _default_energy_configs(config: SimConfig) -> list[tuple[float, float]]:
    """
    Generate default per-device energy arrival bounds [kJ/slot].

    All devices use the same mean (ENERGY_MEAN_BASELINE) with ±ENERGY_SPREAD
    bounds. Homogeneous by default; pass explicit energy_configs to
    build_network() or Network() for heterogeneous setups.
    """
    n_total = config.N_GROUPS * config.DEVICES_PER_GROUP
    m = config.ENERGY_MEAN_BASELINE  # kJ/slot (e.g. 0.55 = 550 J/slot)
    spread = config.ENERGY_SPREAD
    return [(float(m * (1 - spread)), float(m * (1 + spread)))] * n_total


def _initial_power_mode(scheduler: BaseScheduler) -> int:
    """
    Determine the starting power mode for devices based on scheduler type.

    StaticSchedulers fix the mode at construction; others start at PM=2 (30W).
    """
    if isinstance(scheduler, StaticScheduler):
        return scheduler.power_mode
    return 2


# ---------------------------------------------------------------------------
# Convenience factory: build a complete Network from strategy name
# ---------------------------------------------------------------------------

def build_network(
        strategy_name: str,
        energy_configs: list[tuple[float, float]] | None = None,
        config: SimConfig = DEFAULT_CONFIG,
        seed: int = 42,
        e_max_per_device: list[int] | None = None,
) -> Network:
    """
    Convenience factory to build a Network from a strategy name string.

    Parameters
    ----------
    strategy_name : str
        One of: 'S1', 'S2', 'S3', 'D1', 'D2', 'D3', 'D4'
    energy_configs : list of (low, high) tuples | None
        Per-device energy bounds. If None, uses defaults.
    config : SimConfig
    seed : int
    e_max_per_device : list of int | None
        Per-device battery capacity [kJ]. If None, all devices use config.E_MAX.
        Use this for Exp 4 heterogeneous device setups.

    Returns
    -------
    Network ready to call .run() or .run_batch() on.
    """
    from core.strategies import (
        StaticScheduler,
        LongTermScheduler,
        AdaptiveScheduler,
        EnergyProportionalScheduler,
        PowerModeController,
    )

    rng = np.random.default_rng(seed)

    if energy_configs is None:
        energy_configs = _default_energy_configs(config)

    energy_models = [
        UniformEnergyModel(low, high, rng=np.random.default_rng(seed + i + 1))
        for i, (low, high) in enumerate(energy_configs)
    ]

    pm_controller = None

    if strategy_name == "S1":
        scheduler = StaticScheduler(power_mode=1, rng=rng)
    elif strategy_name == "S2":
        scheduler = StaticScheduler(power_mode=2, rng=rng)
    elif strategy_name == "S3":
        scheduler = StaticScheduler(power_mode=3, rng=rng)
    elif strategy_name == "D1":
        scheduler = LongTermScheduler(
            energy_models=energy_models,
            power_mode=2,
            E_max=config.E_MAX,
            E_th_low=config.E_TH_LOW,
            E_th_high=config.E_TH_HIGH_MARKOV,
            target_risk=config.TARGET_RISK,
            rng=rng,
        )
    elif strategy_name == "D2":
        scheduler = AdaptiveScheduler(
            energy_models=energy_models,
            power_mode=2,
            E_max=config.E_MAX,
            E_th_low=config.E_TH_LOW,
            E_th_high=config.E_TH_HIGH_MARKOV,
            target_risk=config.TARGET_RISK,
            alpha=config.ALPHA,
            rng=rng,
        )
    elif strategy_name == "D3":
        scheduler = LongTermScheduler(
            energy_models=energy_models,
            power_mode=2,
            E_max=config.E_MAX,
            E_th_low=config.E_TH_LOW,
            E_th_high=config.E_TH_HIGH_MARKOV,
            target_risk=config.TARGET_RISK,
            rng=rng,
        )
        pm_controller = PowerModeController(
            th_15_to_30=config.PM_TH_15_TO_30_PCT,
            th_30_to_60=config.PM_TH_30_TO_60_PCT,
        )
    elif strategy_name == "D4":
        scheduler = EnergyProportionalScheduler(rng=rng)
    else:
        raise ValueError(
            f"Unknown strategy '{strategy_name}'. "
            f"Choose from: S1, S2, S3, D1, D2, D3, D4"
        )

    return Network(
        scheduler=scheduler,
        power_controller=pm_controller,
        energy_configs=energy_configs,
        config=config,
        e_max_per_device=e_max_per_device,
        rng=rng,
    )
