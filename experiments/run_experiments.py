"""
Experiment runner for the comparative analysis of energy management strategies.

Four experiments are defined, each sweeping one independent variable across
all 7 strategies (S1, S2, S3, D1, D2, D3, D4) and saving results to CSV.

UNIT CONVENTION
---------------
All energy quantities throughout this module are in kJ (not J).
Paper figures report J/slot on the x-axis; divide by 1000 for kJ/slot used here.
  Paper Exp 1 sweep:  50–600 J/slot  ->  0.05–0.60 kJ/slot
  Paper baseline:    550   J/slot  ->  0.55 kJ/slot
  Diurnal peak:      800   J/slot  ->  0.80 kJ/slot

Experiments
-----------
  Exp 1 — Vary average energy arrival rate (reproduces paper Figs 3a / 4a,
           extended to all 7 strategies).
           X-axis: mean energy [kJ/slot] from 0.05 to 0.60.
           Metrics: inactive_fraction, normalised throughput.

  Exp 2 — Vary job arrival probability (reproduces paper Figs 3b / 4b,
           extended to all 7 strategies).
           X-axis: p from 0.1 to 1.0.
           Metrics: inactive_fraction, jobs_dropped.

  Exp 3 — Diurnal energy model (novel extension).
           Same average energy as Exp 1 baseline but sinusoidal temporal
           structure simulating realistic solar harvesting.
           X-axis: diurnal peak energy level [kJ/slot].
           Metrics: inactive_fraction, normalised throughput.

  Exp 4 — Heterogeneous devices (novel extension).
           Devices have different E_max and solar strengths.
           X-axis: heterogeneity scale factor.
           Metrics: inactive_fraction, normalised throughput.

Output
------
  results/exp1_energy_sweep.csv
  results/exp2_arrival_sweep.csv
  results/exp3_diurnal.csv
  results/exp4_heterogeneous.csv

Each CSV has columns:
  strategy, param_value,
  mean_inactive_fraction, std_inactive_fraction,
  mean_throughput, std_throughput,
  mean_jobs_dropped, std_jobs_dropped,
  mean_jobs_completed, std_jobs_completed,
  mean_battery, std_battery

Run from the project root:
  uv run python -m experiments.run_experiments
  # or
  python -m experiments.run_experiments
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from core.energy import UniformEnergyModel, DiurnalEnergyModel
from core.network import Network, build_network, _default_energy_configs
from core.strategies import (
    StaticScheduler,
    LongTermScheduler,
    AdaptiveScheduler,
    EnergyProportionalScheduler,
    PowerModeController,
)
from experiments.config import SimConfig, DEFAULT_CONFIG

RESULTS_DIR = Path(__file__).parent.parent / "results"
STRATEGIES = ["S1", "S2", "S3", "D1", "D2", "D3", "D4"]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _make_energy_configs(
    mean_energy: float,
    n_total: int,
    spread: float = 0.2,
) -> list[tuple[float, float]]:
    """
    Build per-device uniform energy bounds around a shared mean.

    Devices are given slightly different means (±10% spread around the
    target mean) to simulate heterogeneous solar panels, consistent with
    the paper's description of distinct per-device energy profiles.
    """
    rng = np.random.default_rng(0)  # fixed seed for reproducibility
    means = mean_energy + rng.uniform(-0.1 * mean_energy, 0.1 * mean_energy, size=n_total)
    means = np.clip(means, 1.0, None)
    return [(float(m * (1 - spread)), float(m * (1 + spread))) for m in means]


def _run_strategy(
    strategy: str,
    config: SimConfig,
    energy_configs: list[tuple[float, float]] | None = None,
    seed_start: int = 0,
) -> dict[str, float]:
    """
    Run one strategy for N_ITERATIONS replications and return aggregate stats.

    Returns a flat dict with mean_* and std_* keys.
    """
    net = build_network(
        strategy,
        energy_configs=energy_configs,
        config=config,
        seed=seed_start,
    )
    return net.run_batch(
        T=config.T,
        n_iterations=config.N_ITERATIONS,
        seed_start=seed_start,
    )


def _run_strategy_diurnal(
    strategy: str,
    config: SimConfig,
    peak: float,
    base: float,
    period_slots: int,
    seed_start: int = 0,
) -> dict[str, float]:
    """
    Run one strategy with a DiurnalEnergyModel and return aggregate stats.

    Builds the Network manually since build_network() uses UniformEnergyModel.
    """
    n_total = config.N_GROUPS * config.DEVICES_PER_GROUP
    rng_master = np.random.default_rng(seed_start)

    # Build diurnal energy models — each device gets the same sinusoidal
    # profile but different noise seeds, simulating co-located solar panels.
    energy_models_diurnal = [
        DiurnalEnergyModel(
            peak=peak,
            base=base,
            period_slots=period_slots,
            rng=np.random.default_rng(seed_start + i + 1),
        )
        for i in range(n_total)
    ]

    # Build the equivalent uniform energy configs for the Markov solver
    # (D1/D2 use UniformEnergyModel for q_lim computation; the diurnal model
    # is used at runtime for the actual energy arrivals during simulation).
    mean_energy = (peak + base) / 2.0
    uniform_configs = _make_energy_configs(mean_energy, n_total, spread=0.2)
    uniform_models = [
        UniformEnergyModel(lo, hi, rng=np.random.default_rng(seed_start + i + 100))
        for i, (lo, hi) in enumerate(uniform_configs)
    ]

    # Build the scheduler (same logic as build_network, but we pass uniform
    # models to the Markov solver and override energy at runtime).
    scheduler, pm_controller = _build_scheduler(
        strategy, uniform_models, config, rng_master
    )

    # Collect per-iteration results
    per_iter: list[dict] = []
    for i in range(config.N_ITERATIONS):
        # Re-seed RNG for this iteration
        iter_rng = np.random.default_rng(seed_start + i)

        # Build network but inject diurnal energy models manually
        net = Network(
            scheduler=scheduler,
            power_controller=pm_controller,
            energy_configs=uniform_configs,  # placeholder (overridden below)
            config=config,
            rng=iter_rng,
        )
        # Override the energy models with diurnal ones
        net._energy_models = [
            DiurnalEnergyModel(
                peak=peak,
                base=base,
                period_slots=period_slots,
                rng=np.random.default_rng(seed_start + i * 100 + j),
            )
            for j in range(n_total)
        ]
        metrics = net.run(T=config.T, seed=seed_start + i)
        per_iter.append(metrics)

    return _aggregate(per_iter)


def _build_scheduler(strategy, uniform_models, config, rng):
    """Build a (scheduler, pm_controller) pair — mirrors build_network() logic."""
    pm_controller = None
    if strategy == "S1":
        scheduler = StaticScheduler(power_mode=1, rng=rng)
    elif strategy == "S2":
        scheduler = StaticScheduler(power_mode=2, rng=rng)
    elif strategy == "S3":
        scheduler = StaticScheduler(power_mode=3, rng=rng)
    elif strategy == "D1":
        scheduler = LongTermScheduler(
            energy_models=uniform_models,
            power_mode=2,
            E_max=config.E_MAX,
            E_th_low=config.E_TH_LOW,
            E_th_high=config.E_TH_HIGH_MARKOV,
            target_risk=config.TARGET_RISK,
            rng=rng,
        )
    elif strategy == "D2":
        scheduler = AdaptiveScheduler(
            energy_models=uniform_models,
            power_mode=2,
            E_max=config.E_MAX,
            E_th_low=config.E_TH_LOW,
            E_th_high=config.E_TH_HIGH_MARKOV,
            target_risk=config.TARGET_RISK,
            alpha=config.ALPHA,
            rng=rng,
        )
    elif strategy == "D3":
        scheduler = LongTermScheduler(
            energy_models=uniform_models,
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
    elif strategy == "D4":
        scheduler = EnergyProportionalScheduler(rng=rng)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    return scheduler, pm_controller


def _aggregate(per_iter: list[dict]) -> dict[str, float]:
    """Aggregate a list of per-iteration metric dicts into mean ± std."""
    # Scalar totals over the run
    completed  = np.array([r["jobs_completed"].sum()    for r in per_iter])
    dropped    = np.array([r["jobs_dropped"].sum()      for r in per_iter])
    arrived    = np.array([r["jobs_arrived"].sum()      for r in per_iter])
    inactive   = np.array([r["inactive_fraction"].mean() for r in per_iter])
    battery    = np.array([r["batteries"].mean()         for r in per_iter])

    # Normalised throughput: completed / arrived (avoid divide-by-zero)
    throughput = np.where(arrived > 0, completed / arrived, 0.0)

    return {
        "mean_inactive_fraction": float(inactive.mean()),
        "std_inactive_fraction":  float(inactive.std()),
        "mean_throughput":        float(throughput.mean()),
        "std_throughput":         float(throughput.std()),
        "mean_jobs_dropped":      float(dropped.mean()),
        "std_jobs_dropped":       float(dropped.std()),
        "mean_jobs_completed":    float(completed.mean()),
        "std_jobs_completed":     float(completed.std()),
        "mean_battery":           float(battery.mean()),
        "std_battery":            float(battery.std()),
    }


def _progress(strategy: str, param_name: str, value: float, elapsed: float) -> None:
    print(f"    {strategy:3s}  {param_name}={value:7.1f}  [{elapsed:5.1f}s]")


# ---------------------------------------------------------------------------
# Experiment 1 — Vary average energy arrival rate
# ---------------------------------------------------------------------------

def run_exp1(
    config: SimConfig = DEFAULT_CONFIG,
    energy_means: list[float] | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Reproduce paper Figures 3a / 4a, extended to all 7 strategies.

    Sweeps average energy arrival from sparse (0.05 kJ/slot = 50 J/slot) to
    abundant (0.60 kJ/slot = 600 J/slot). At each value, all 7 strategies run
    for N_ITERATIONS replications. Job arrival probability is held at p=0.3.

    Parameters
    ----------
    config : SimConfig
    energy_means : list of float | None
        X-axis values [kJ/slot]. Defaults to 12 evenly-spaced points 0.05–0.60.
        (Equivalent to 50–600 J/slot as reported in paper Fig 3a.)
    verbose : bool
        Print progress to stdout.

    Returns
    -------
    DataFrame saved to results/exp1_energy_sweep.csv
    """
    if energy_means is None:
        energy_means = list(np.linspace(0.05, 0.60, 12))

    n_total = config.N_GROUPS * config.DEVICES_PER_GROUP

    if verbose:
        print("\n[Exp 1] Energy arrival rate sweep")
        print(f"  strategies: {STRATEGIES}")
        print(f"  energy_means: {[f'{v:.3f}' for v in energy_means]} kJ/slot")
        print(f"  N_ITERATIONS={config.N_ITERATIONS}, T={config.T}")

    rows: list[dict] = []
    for mean_e in energy_means:
        energy_configs = _make_energy_configs(mean_e, n_total, config.ENERGY_SPREAD)
        for strategy in STRATEGIES:
            t0 = time.time()
            agg = _run_strategy(
                strategy,
                config=config,
                energy_configs=energy_configs,
                seed_start=0,
            )
            elapsed = time.time() - t0
            if verbose:
                _progress(strategy, "mean_E", mean_e, elapsed)
            rows.append({
                "strategy":    strategy,
                "param_value": mean_e,
                **agg,
            })

    df = pd.DataFrame(rows)
    _save(df, "exp1_energy_sweep.csv", verbose)
    return df


# ---------------------------------------------------------------------------
# Experiment 2 — Vary job arrival probability
# ---------------------------------------------------------------------------

def run_exp2(
    config: SimConfig = DEFAULT_CONFIG,
    arrival_probs: list[float] | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Reproduce paper Figures 3b / 4b, extended to all 7 strategies.

    Sweeps Bernoulli job arrival probability p from 0.1 to 1.0.
    Average energy is held at the baseline (550 J/slot).

    Parameters
    ----------
    config : SimConfig
    arrival_probs : list of float | None
        X-axis values. Defaults to 10 evenly-spaced points 0.1–1.0.
    verbose : bool
    """
    if arrival_probs is None:
        arrival_probs = list(np.linspace(0.1, 1.0, 10))

    n_total = config.N_GROUPS * config.DEVICES_PER_GROUP
    energy_configs = _make_energy_configs(
        config.ENERGY_MEAN_BASELINE, n_total, config.ENERGY_SPREAD
    )

    if verbose:
        print("\n[Exp 2] Job arrival probability sweep")
        print(f"  strategies: {STRATEGIES}")
        print(f"  arrival_probs: {[f'{v:.2f}' for v in arrival_probs]}")
        print(f"  energy_mean={config.ENERGY_MEAN_BASELINE} J/slot  "
              f"N_ITERATIONS={config.N_ITERATIONS}, T={config.T}")

    rows: list[dict] = []
    for p in arrival_probs:
        sweep_config = SimConfig(
            E_MAX=config.E_MAX,
            E_TH_LOW=config.E_TH_LOW,
            E_TH_HIGH_MARKOV=config.E_TH_HIGH_MARKOV,
            PM_TH_15_TO_30_PCT=config.PM_TH_15_TO_30_PCT,
            PM_TH_30_TO_60_PCT=config.PM_TH_30_TO_60_PCT,
            DELTA=config.DELTA,
            T=config.T,
            N_ITERATIONS=config.N_ITERATIONS,
            JOB_ARRIVAL_PROB=p,  # swept parameter
            TARGET_RISK=config.TARGET_RISK,
            ALPHA=config.ALPHA,
            ENERGY_MEAN_BASELINE=config.ENERGY_MEAN_BASELINE,
            ENERGY_SPREAD=config.ENERGY_SPREAD,
            DIURNAL_PEAK=config.DIURNAL_PEAK,
            DIURNAL_BASE=config.DIURNAL_BASE,
            DIURNAL_PERIOD=config.DIURNAL_PERIOD,
        )
        for strategy in STRATEGIES:
            t0 = time.time()
            agg = _run_strategy(
                strategy,
                config=sweep_config,
                energy_configs=energy_configs,
                seed_start=0,
            )
            elapsed = time.time() - t0
            if verbose:
                _progress(strategy, "p", p, elapsed)
            rows.append({
                "strategy":    strategy,
                "param_value": p,
                **agg,
            })

    df = pd.DataFrame(rows)
    _save(df, "exp2_arrival_sweep.csv", verbose)
    return df


# ---------------------------------------------------------------------------
# Experiment 3 — Diurnal energy model (novel extension)
# ---------------------------------------------------------------------------

def run_exp3(
    config: SimConfig = DEFAULT_CONFIG,
    peak_values: list[float] | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Novel experiment: diurnal (sinusoidal) energy arrivals.

    Replaces the i.i.d. uniform model with a solar-realistic sinusoid that
    peaks at midday and bottoms out at night. Sweeps the peak energy level
    while keeping base=50 J/slot constant, so the mean energy changes with
    the peak. This tests how correlated temporal structure (vs. uncorrelated
    i.i.d.) affects strategy rankings.

    The average energy per slot equals (peak + base) / 2, so the sweep from
    peak=150 to peak=1150 gives means of 100–600 J/slot — matching Exp 1's
    energy range for direct comparison.

    Parameters
    ----------
    config : SimConfig
    peak_values : list of float | None
        Diurnal peak energy [J/slot]. Defaults to 12 values 150–1150.
    verbose : bool
    """
    if peak_values is None:
        # Peak 0.15->1.15 kJ/slot gives mean 0.10->0.60 kJ/slot (matching Exp 1 range)
        peak_values = list(np.linspace(0.15, 1.15, 12))

    if verbose:
        print("\n[Exp 3] Diurnal energy model (novel)")
        print(f"  strategies: {STRATEGIES}")
        print(f"  peak_values: {[f'{v:.3f}' for v in peak_values]} kJ/slot")
        print(f"  base={config.DIURNAL_BASE:.3f} kJ/slot, period={config.DIURNAL_PERIOD} slots (24h)")
        print(f"  N_ITERATIONS={config.N_ITERATIONS}, T={config.T}")

    rows: list[dict] = []
    for peak in peak_values:
        for strategy in STRATEGIES:
            t0 = time.time()
            agg = _run_strategy_diurnal(
                strategy=strategy,
                config=config,
                peak=peak,
                base=config.DIURNAL_BASE,
                period_slots=config.DIURNAL_PERIOD,
                seed_start=0,
            )
            elapsed = time.time() - t0
            if verbose:
                mean_e = (peak + config.DIURNAL_BASE) / 2.0
                _progress(strategy, "peak", peak, elapsed)
            rows.append({
                "strategy":          strategy,
                "param_value":       peak,
                "mean_energy_equiv": (peak + config.DIURNAL_BASE) / 2.0,
                **agg,
            })

    df = pd.DataFrame(rows)
    _save(df, "exp3_diurnal.csv", verbose)
    return df


# ---------------------------------------------------------------------------
# Experiment 4 — Heterogeneous devices (novel extension)
# ---------------------------------------------------------------------------

def run_exp4(
    config: SimConfig = DEFAULT_CONFIG,
    hetero_scales: list[float] | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Novel experiment: heterogeneous devices with different E_max and solar strength.

    In the paper all devices are identical. Here we introduce heterogeneity by
    scaling each device's E_max and energy mean differently. A scale factor of
    1.0 means homogeneous (identical to paper setup). Higher values mean greater
    spread in device capability.

    Device capabilities are drawn from uniform distributions:
      E_max_i ~ Uniform(50, 50 + scale * 100) kJ
      energy_mean_i ~ Uniform(100, 100 + scale * 450) J/slot

    This tests whether strategy rankings change when devices are not identical —
    a condition not addressed by the reference paper.

    Parameters
    ----------
    config : SimConfig
    hetero_scales : list of float | None
        Heterogeneity scale factors. Default: 0.0 (homogeneous) to 1.0.
    verbose : bool
    """
    if hetero_scales is None:
        hetero_scales = list(np.linspace(0.0, 1.0, 6))

    n_total = config.N_GROUPS * config.DEVICES_PER_GROUP

    if verbose:
        print("\n[Exp 4] Heterogeneous devices (novel)")
        print(f"  strategies: {STRATEGIES}")
        print(f"  hetero_scales: {[f'{v:.2f}' for v in hetero_scales]}")
        print(f"  N_ITERATIONS={config.N_ITERATIONS}, T={config.T}")

    rows: list[dict] = []
    for scale in hetero_scales:
        rng_cfg = np.random.default_rng(42)

        if scale == 0.0:
            # Fully homogeneous — all devices identical
            e_max_values = [config.E_MAX] * n_total
            energy_means = [config.ENERGY_MEAN_BASELINE] * n_total
        else:
            e_max_values = rng_cfg.uniform(
                50, 50 + scale * 100, size=n_total
            ).tolist()
            energy_means = rng_cfg.uniform(
                100, 100 + scale * 450, size=n_total
            ).tolist()

        energy_configs = [
            (float(m * (1 - config.ENERGY_SPREAD)),
             float(m * (1 + config.ENERGY_SPREAD)))
            for m in energy_means
        ]

        # Use the mean E_max across devices for the Markov model.
        # Build a fresh SimConfig and override only E_MAX — this avoids
        # issues with dataclass field factories when copying via __dict__.
        mean_e_max = int(round(np.mean(e_max_values)))
        hetero_config = SimConfig(
            E_MAX=mean_e_max,
            E_TH_LOW=config.E_TH_LOW,
            E_TH_HIGH_MARKOV=config.E_TH_HIGH_MARKOV,
            PM_TH_15_TO_30_PCT=config.PM_TH_15_TO_30_PCT,
            PM_TH_30_TO_60_PCT=config.PM_TH_30_TO_60_PCT,
            DELTA=config.DELTA,
            T=config.T,
            N_ITERATIONS=config.N_ITERATIONS,
            JOB_ARRIVAL_PROB=config.JOB_ARRIVAL_PROB,
            TARGET_RISK=config.TARGET_RISK,
            ALPHA=config.ALPHA,
            ENERGY_MEAN_BASELINE=config.ENERGY_MEAN_BASELINE,
            ENERGY_SPREAD=config.ENERGY_SPREAD,
            DIURNAL_PEAK=config.DIURNAL_PEAK,
            DIURNAL_BASE=config.DIURNAL_BASE,
            DIURNAL_PERIOD=config.DIURNAL_PERIOD,
        )

        for strategy in STRATEGIES:
            t0 = time.time()
            agg = _run_strategy(
                strategy,
                config=hetero_config,
                energy_configs=energy_configs,
                seed_start=0,
            )
            elapsed = time.time() - t0
            if verbose:
                _progress(strategy, "scale", scale, elapsed)
            rows.append({
                "strategy":    strategy,
                "param_value": scale,
                "mean_e_max":  float(np.mean(e_max_values)),
                "mean_energy": float(np.mean(energy_means)),
                **agg,
            })

    df = pd.DataFrame(rows)
    _save(df, "exp4_heterogeneous.csv", verbose)
    return df


# ---------------------------------------------------------------------------
# Save helper
# ---------------------------------------------------------------------------

def _save(df: pd.DataFrame, filename: str, verbose: bool = True) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / filename
    df.to_csv(path, index=False)
    if verbose:
        print(f"  --> saved {path} ({len(df)} rows)")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(
    fast: bool = False,
    experiments: list[int] | None = None,
) -> dict[str, pd.DataFrame]:
    """
    Run all four experiments sequentially.

    Parameters
    ----------
    fast : bool
        If True, use a reduced config (T=20, N_ITERATIONS=5, fewer sweep
        points) for quick validation. Results are still saved to CSV.
    experiments : list[int] | None
        Subset of experiments to run, e.g. [1, 3]. Default: all four.

    Returns
    -------
    dict mapping experiment name to DataFrame.
    """
    if experiments is None:
        experiments = [1, 2, 3, 4]

    if fast:
        cfg = SimConfig(T=100, N_ITERATIONS=10)
        energy_means  = [0.10, 0.30, 0.55]   # kJ/slot (= 100, 300, 550 J/slot)
        arrival_probs = [0.3, 0.6, 1.0]
        peak_values   = [0.20, 0.60, 1.10]   # kJ/slot diurnal peak
        hetero_scales = [0.0, 0.5, 1.0]
        print("\n[fast mode] T=100, N_ITERATIONS=10, 3 sweep points per axis")
    else:
        cfg = DEFAULT_CONFIG
        energy_means  = None
        arrival_probs = None
        peak_values   = None
        hetero_scales = None

    t_total = time.time()
    results: dict[str, pd.DataFrame] = {}

    if 1 in experiments:
        results["exp1"] = run_exp1(cfg, energy_means)
    if 2 in experiments:
        results["exp2"] = run_exp2(cfg, arrival_probs)
    if 3 in experiments:
        results["exp3"] = run_exp3(cfg, peak_values)
    if 4 in experiments:
        results["exp4"] = run_exp4(cfg, hetero_scales)

    print(f"\nAll experiments complete. Total time: {time.time() - t_total:.1f}s")
    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run energy strategy comparison experiments."
    )
    parser.add_argument(
        "--fast", action="store_true",
        help="Quick validation run (T=100, N=10, few sweep points)."
    )
    parser.add_argument(
        "--exp", nargs="+", type=int, choices=[1, 2, 3, 4],
        help="Which experiments to run (default: all). E.g. --exp 1 3"
    )
    args = parser.parse_args()

    main(fast=args.fast, experiments=args.exp)
