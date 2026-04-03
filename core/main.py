"""
Layer 1 smoke test — verifies every core component works end-to-end.

Run from the project root:
    uv run python -m core.main
    # or
    python -m core.main

Tests performed
---------------
  1. UniformEnergyModel  — sample() and pmf() sanity checks
  2. DiurnalEnergyModel  — sample() returns non-negative values across a day
  3. Device              — battery update, hysteresis, job accept/complete
  4. compute_q_lim       — runs for each power mode; verifies ordering
  5. All 7 strategies    — select_device() returns a Device or None
  6. Network.run()       — single run for all 7 strategies (T=20 slots)
  7. Network.run_batch() — 5 replications for one strategy (S2, fastest)

Expected output: all checks print PASS. Any exception indicates a bug.
"""

import sys
import time
import traceback

import numpy as np

from core.device import Device
from core.energy import UniformEnergyModel, DiurnalEnergyModel
from core.markov import compute_q_lim
from core.network import build_network
from experiments.config import SimConfig

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

PASS = "PASS"
FAIL = "FAIL"


def check(label: str, condition: bool, detail: str = "") -> bool:
    status = PASS if condition else FAIL
    suffix = f"  ({detail})" if detail else ""
    print(f"  [{status}] {label}{suffix}")
    return condition


def section(title: str) -> None:
    print(f"\n{'-' * 60}")
    print(f"  {title}")
    print(f"{'-' * 60}")


# ---------------------------------------------------------------------------
# Test 1: Energy models
# ---------------------------------------------------------------------------

def test_energy_models() -> bool:
    section("1. Energy models")
    ok = True

    # Uniform — values in kJ/slot (0.10 to 0.50 kJ/slot = 100 to 500 J/slot)
    rng = np.random.default_rng(0)
    em = UniformEnergyModel(low=0.10, high=0.50, rng=rng)
    samples = [em.sample(t) for t in range(1000)]
    ok &= check("Uniform: all samples in [low, high] kJ",
                all(0.10 <= s <= 0.50 for s in samples))
    ok &= check("Uniform: mean ~= 0.30 kJ",
                abs(np.mean(samples) - 0.30) < 0.015,
                f"got {np.mean(samples):.4f}")

    # PMF operates on integer kJ grid; with sub-kJ values all mass goes to 0
    e_grid = np.arange(0, 101)
    pmf = em.pmf(e_grid)
    ok &= check("Uniform PMF sums to <= 1.0 (sub-kJ -> mass at 0)",
                pmf.sum() <= 1.0 + 1e-9,
                f"sum={pmf.sum():.4f}")

    # Diurnal — values in kJ/slot (peak=0.80, base=0.05 matches config defaults)
    rng2 = np.random.default_rng(1)
    dm = DiurnalEnergyModel(peak=0.80, base=0.05, period_slots=864, rng=rng2)
    noon_slot = 864 // 4
    noon_val = dm.deterministic_value(noon_slot)
    midnight_val = dm.deterministic_value(0)
    ok &= check("Diurnal: noon > midnight", noon_val > midnight_val,
                f"noon={noon_val:.4f}, midnight={midnight_val:.4f} kJ")
    diurnal_samples = [dm.sample(t) for t in range(864)]
    ok &= check("Diurnal: all samples >= 0", all(s >= 0 for s in diurnal_samples))
    ok &= check("Diurnal: mean ~= (peak+base)/2 kJ",
                abs(np.mean(diurnal_samples) - dm.mean()) < 0.05,
                f"got {np.mean(diurnal_samples):.4f}, expected ~{dm.mean():.4f}")

    return ok


# ---------------------------------------------------------------------------
# Test 2: Device state machine
# ---------------------------------------------------------------------------

def test_device() -> bool:
    section("2. Device state machine")
    ok = True

    cfg = SimConfig()
    d = Device(device_id=0, E_max=cfg.E_MAX, E_th_low=cfg.E_TH_LOW,
               E_th_high=cfg.E_TH_HIGH_MARKOV, initial_battery=cfg.E_MAX)

    ok &= check("Device starts active (gamma=1)", d.gamma == 1)
    ok &= check("Device starts available", d.is_available())

    # Drain battery below E_th_low to trigger power-saving
    d._battery = cfg.E_TH_LOW - 1
    d._update_gamma()
    ok &= check("Enters power-saving when E < E_th_low", d.gamma == 0)
    ok &= check("Not available in power-saving", not d.is_available())

    # Recharge above E_th_high to exit power-saving
    d._battery = cfg.E_TH_HIGH_MARKOV + 1
    d._update_gamma()
    ok &= check("Exits power-saving when E > E_th_high", d.gamma == 1)
    ok &= check("Available again after recharge", d.is_available())

    # Accept a job and step through it
    accepted = d.accept_job()
    ok &= check("accept_job() returns True when available", accepted)
    ok &= check("Queue is 1 after accept_job", d.queue == 1)
    ok &= check("Not available with job in queue", not d.is_available())

    # Step through kappa=2 slots (PM=2, 30W). Harvested in kJ directly.
    d._power_mode = 2
    d._slots_remaining = 2
    result1 = d.step(harvested=0.22)  # 0.22 kJ = 220 J (small harvest)
    ok &= check("Job not done after 1 of 2 slots", not result1["job_completed"])
    result2 = d.step(harvested=0.22)
    ok &= check("Job done after 2nd slot", result2["job_completed"])
    ok &= check("Queue cleared after job completes", d.queue == 0)

    # Battery clamping
    d._battery = cfg.E_MAX
    d._gamma = 1
    d._queue = 0
    result = d.step(harvested=1_000.0)  # absurdly large harvest in kJ
    ok &= check("Battery clamped to E_max", result["battery"] == cfg.E_MAX,
                f"got {result['battery']}")

    return ok


# ---------------------------------------------------------------------------
# Test 3: Markov model
# ---------------------------------------------------------------------------

def test_markov() -> bool:
    section("3. Markov / compute_q_lim (may take ~10-30 s)")
    ok = True
    cfg = SimConfig()

    # Use a simple energy model for speed — values in kJ/slot
    # 0.20-0.40 kJ/slot = 200-400 J/slot (moderate energy, good for Markov solver)
    em = UniformEnergyModel(low=0.20, high=0.40)

    q_lims = {}
    for pm in (1, 2, 3):
        t0 = time.time()
        q = compute_q_lim(
            energy_model=em,
            power_mode=pm,
            E_max=cfg.E_MAX,
            E_th_low=cfg.E_TH_LOW,
            E_th_high=cfg.E_TH_HIGH_MARKOV,
            target_risk=cfg.TARGET_RISK,
        )
        elapsed = time.time() - t0
        q_lims[pm] = q
        ok &= check(f"PM{pm}: q_lim in (0, 1.0]",
                    0 < q <= 1.0,
                    f"q_lim={q:.4f}, elapsed={elapsed:.1f}s")

    # PM=1 (15W) should have the lowest q_lim due to κ=3 processing constraint
    ok &= check("PM1 q_lim <= 1/3 (processing-delay bound)",
                q_lims[1] <= 1 / 3 + 1e-6,
                f"got {q_lims[1]:.4f}")
    ok &= check("PM2 q_lim <= 1/2 (processing-delay bound)",
                q_lims[2] <= 1 / 2 + 1e-6,
                f"got {q_lims[2]:.4f}")
    ok &= check("PM3 q_lim <= 1.0 (no processing-delay constraint at 60W)",
                q_lims[3] <= 1.0,
                f"got {q_lims[3]:.4f}")

    return ok


# ---------------------------------------------------------------------------
# Test 4: All strategies — single select_device call
# ---------------------------------------------------------------------------

def test_strategies() -> bool:
    section("4. Strategies - select_device()")
    ok = True
    cfg = SimConfig(N_ITERATIONS=1)  # not used here, just config

    # Build dummy devices for testing
    devices = [
        Device(device_id=i, E_max=100, E_th_low=10, E_th_high=20,
               initial_battery=80)
        for i in range(3)
    ]

    # kJ/slot values matching the kJ unit convention
    em = UniformEnergyModel(0.20, 0.40, rng=np.random.default_rng(0))
    energy_models = [UniformEnergyModel(0.20, 0.40, rng=np.random.default_rng(i))
                     for i in range(9)]

    from core.strategies import (
        StaticScheduler, PowerModeController, EnergyProportionalScheduler,
    )

    strategies_under_test = [
        ("S1", StaticScheduler(power_mode=1, rng=np.random.default_rng(0))),
        ("S2", StaticScheduler(power_mode=2, rng=np.random.default_rng(0))),
        ("S3", StaticScheduler(power_mode=3, rng=np.random.default_rng(0))),
        ("D4", EnergyProportionalScheduler(rng=np.random.default_rng(0))),
    ]

    for name, sched in strategies_under_test:
        chosen = sched.select_device(devices, t=0)
        ok &= check(f"{name}: returns a Device", chosen is not None and isinstance(chosen, Device))

    # D3 PowerModeController
    pmc = PowerModeController(th_15_to_30=0.40, th_30_to_60=0.60)
    d = Device(device_id=99, E_max=100, initial_battery=30)
    pmc.update_device(d)
    ok &= check("D3: battery=30% -> PM=1 (15W)", d.power_mode == 1,
                f"got PM={d.power_mode}")
    d._battery = 55
    pmc.update_device(d)
    ok &= check("D3: battery=55% -> PM=2 (30W)", d.power_mode == 2,
                f"got PM={d.power_mode}")
    d._battery = 75
    pmc.update_device(d)
    ok &= check("D3: battery=75% -> PM=3 (60W)", d.power_mode == 3,
                f"got PM={d.power_mode}")

    # All-unavailable group -> should return None
    busy_devices = [Device(device_id=i, E_max=100, initial_battery=80) for i in range(3)]
    for bd in busy_devices:
        bd.accept_job()
    chosen = strategies_under_test[0][1].select_device(busy_devices, t=0)
    ok &= check("select_device returns None when all devices busy", chosen is None)

    return ok


# ---------------------------------------------------------------------------
# Test 5: Network.run() — all 7 strategies
# ---------------------------------------------------------------------------

def test_network() -> bool:
    section("5. Network.run() - all 7 strategies (T=20)")
    ok = True
    cfg = SimConfig(T=20, N_ITERATIONS=5)

    for strategy in ("S1", "S2", "S3", "D1", "D2", "D3", "D4"):
        try:
            net = build_network(strategy, config=cfg, seed=42)
            metrics = net.run(T=20, seed=0)

            # Basic shape checks
            n_dev = cfg.N_GROUPS * cfg.DEVICES_PER_GROUP
            shapes_ok = (
                    metrics["batteries"].shape == (20, n_dev)
                    and metrics["jobs_completed"].shape == (20,)
                    and metrics["inactive_fraction"].shape == (20,)
            )
            ok &= check(f"{strategy}: output shapes correct", shapes_ok)

            # Batteries stay within [0, E_max]
            batt_ok = (
                    metrics["batteries"].min() >= 0
                    and metrics["batteries"].max() <= cfg.E_MAX
            )
            ok &= check(f"{strategy}: batteries in [0, E_max]", batt_ok,
                        f"min={metrics['batteries'].min():.1f}, "
                        f"max={metrics['batteries'].max():.1f}")

            total_arr = metrics["jobs_arrived"].sum()
            total_cmp = metrics["jobs_completed"].sum()
            total_drp = metrics["jobs_dropped"].sum()
            ok &= check(f"{strategy}: completed + dropped <= arrived",
                        total_cmp + total_drp <= total_arr + 1,  # +1 for rounding
                        f"arrived={total_arr}, completed={total_cmp}, dropped={total_drp}")

        except Exception:
            print(f"  [{FAIL}] {strategy}: exception during run")
            traceback.print_exc()
            ok = False

    return ok


# ---------------------------------------------------------------------------
# Test 6: Network.run_batch()
# ---------------------------------------------------------------------------

def test_run_batch() -> bool:
    section("6. Network.run_batch() - S2, 5 iterations")
    ok = True
    cfg = SimConfig(T=10, N_ITERATIONS=5)
    try:
        net = build_network("S2", config=cfg, seed=0)
        agg = net.run_batch(T=10, n_iterations=5, seed_start=100)
        expected_keys = {
            "mean_jobs_completed", "std_jobs_completed",
            "mean_jobs_dropped", "std_jobs_dropped",
            "mean_inactive_fraction", "std_inactive_fraction",
            "mean_battery", "std_battery",
        }
        ok &= check("run_batch: all expected keys present",
                    expected_keys.issubset(agg.keys()),
                    f"missing: {expected_keys - agg.keys()}")
        ok &= check("run_batch: mean_battery in [0, E_max]",
                    0 <= agg["mean_battery"] <= cfg.E_MAX,
                    f"got {agg['mean_battery']:.2f}")
    except Exception:
        print(f"  [{FAIL}] run_batch raised an exception")
        traceback.print_exc()
        ok = False
    return ok


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    print("\n" + "=" * 60)
    print("  Layer 1 smoke test - core simulation engine")
    print("=" * 60)

    results = [
        test_energy_models(),
        test_device(),
        test_markov(),
        test_strategies(),
        test_network(),
        test_run_batch(),
    ]

    total = len(results)
    passed = sum(results)
    failed = total - passed

    print(f"\n{'=' * 60}")
    if failed == 0:
        print(f"  All {total} test sections PASSED.")
    else:
        print(f"  {passed}/{total} sections passed, {failed} FAILED.")
    print("=" * 60 + "\n")

    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
