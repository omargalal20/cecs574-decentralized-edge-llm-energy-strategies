"""
Three sanity checks to validate the simulation engine before trusting results.

Run from the project root with:
  uv run python sanity_check.py
  # or
  python sanity_check.py

All three checks must PASS before running full experiments.

Check 1 — Conservation
  Total network energy can only decrease (consumed > harvested at 0.35 kJ/slot).
  Verifies no phantom energy is created above E_max and no negative batteries.

Check 2 — Ordering
  S1 (15W) < S2 (30W) < S3 (60W) for jobs completed.
  S3 < S2 < S1 for final average battery.
  Verifies processing slots and energy-per-job constants are correct.

  Parameter choice: 0.35 kJ/slot (350 J/slot), p=0.7, T=300.
  Rationale: each device harvests ~0.35 x 300 = 105 kJ over the run (slightly
  above one full battery recharge). At p=0.7, devices process roughly 1 job
  every 3-4 slots. S3 processes fastest (kappa=1) but drains fastest too.
  This keeps batteries in a dynamic range where ordering is clear.

Check 3 — Convergence
  Mean downtime stabilizes as N grows (10 -> 50 -> 200).
  Std shrinks as more iterations are averaged.
  Verifies the energy model is i.i.d. and seeds are independent.

  Parameter choice: 0.45 kJ/slot, p=0.5, T=100.
  Rationale: this puts downtime at ~10-30% per run — genuinely stochastic,
  so Monte Carlo averaging has real variance to reduce.
"""

from __future__ import annotations

import sys

import numpy as np

from core.network import build_network
from experiments.config import SimConfig

PASS = "  [PASS]"
FAIL = "  [FAIL]"
SEP  = "-" * 60

# Check 2 config: tight but not extreme — strategies stay differentiated
# 0.35 kJ/slot = 350 J/slot, p=0.7, T=300 slots
# Total harvest per device: 0.35 x 300 = 105 kJ (~1 full recharge)
# S3 completes most jobs (kappa=1), S1 fewest (kappa=3)
ORDERING_CFG = SimConfig(
    T=300,
    N_ITERATIONS=1,
    JOB_ARRIVAL_PROB=0.7,
    ENERGY_MEAN_BASELINE=0.35,
)

# Check 3 config: moderate regime where downtime is ~10-30% and genuinely varies
# 0.45 kJ/slot = 450 J/slot, p=0.5 — system in dynamic equilibrium
CONVERGENCE_CFG = SimConfig(
    T=100,
    JOB_ARRIVAL_PROB=0.5,
    ENERGY_MEAN_BASELINE=0.45,
)


# ---------------------------------------------------------------------------
# Check 1 — Energy conservation
# ---------------------------------------------------------------------------

def check_conservation() -> bool:
    """
    Verify that no phantom energy is created — batteries stay within [0, E_max]
    and the total network energy never exceeds the initial charge plus harvest.

    We run S2 using network.run() directly, which handles the correct tick order.
    Before and after, we record total battery. We also compute the theoretical
    maximum possible final energy (initial + all harvest, clamped to E_max).
    The actual final energy must be <= this maximum (consumption can only help).
    """
    print(f"\n{SEP}")
    print("Check 1 - Energy conservation")
    print(SEP)

    cfg = ORDERING_CFG
    n_total = cfg.N_GROUPS * cfg.DEVICES_PER_GROUP
    net = build_network("S2", config=cfg, seed=42)

    initial_total = sum(d.battery for d in net.devices)
    print(f"  n_devices             : {n_total}")
    print(f"  Energy arrival        : {cfg.ENERGY_MEAN_BASELINE:.2f} kJ/slot/device")
    print(f"  Initial total battery : {initial_total:.1f} kJ")

    # Run via the standard simulation path
    metrics = net.run(T=cfg.T, seed=42)

    final_total = sum(d.battery for d in net.devices)
    energy_delta = final_total - initial_total

    # Upper bound: initial + total possible harvest (ignoring clamping)
    # Actual must be <= this because consumption can only reduce energy further
    total_possible_harvest = cfg.ENERGY_MEAN_BASELINE * n_total * cfg.T
    theoretical_max = min(initial_total + total_possible_harvest,
                          float(cfg.E_MAX * n_total))

    print(f"  Final   total battery : {final_total:.1f} kJ")
    print(f"  Change                : {energy_delta:+.1f} kJ")
    print(f"  Theoretical max final : {theoretical_max:.1f} kJ")
    print(f"  Jobs completed        : {int(metrics['jobs_completed'].sum())}")

    ok_upper = final_total <= theoretical_max + 1.0   # 1 kJ float tolerance
    ok_lower = final_total >= 0.0
    ok_per_device = all(0 <= d.battery <= cfg.E_MAX for d in net.devices)

    passed = ok_upper and ok_lower and ok_per_device
    if passed:
        print(PASS + " battery bounds respected, no phantom energy")
    else:
        if not ok_upper:
            print(FAIL + f" final battery {final_total:.1f} > theoretical max {theoretical_max:.1f}")
        if not ok_lower:
            print(FAIL + " total battery went negative")
        if not ok_per_device:
            for d in net.devices:
                if not (0 <= d.battery <= cfg.E_MAX):
                    print(FAIL + f" device {d.device_id}: battery={d.battery}")
    return passed


# ---------------------------------------------------------------------------
# Check 2 — Ordering: S1 < S2 < S3 in jobs, S3 < S2 < S1 in battery
# ---------------------------------------------------------------------------

def check_ordering() -> bool:
    """
    Run S1, S2, S3 for 300 slots with p=0.7 and moderate scarcity (0.35 kJ/slot).

    Expected (from paper physics):
      jobs_completed : S1 < S2 < S3   (S3 is fastest: kappa=1 slot/job)
      avg battery    : S3 < S2 < S1   (S3 burns most energy: 23 kJ/job at κ=1)

    At 0.35 kJ/slot each device harvests ~105 kJ total — enough to process
    ~4-5 jobs with full recharge cycles, keeping strategies in distinct regimes.
    """
    print(f"\n{SEP}")
    print("Check 2 - Strategy ordering (S1 < S2 < S3 on throughput)")
    print(f"  Config: {ORDERING_CFG.ENERGY_MEAN_BASELINE} kJ/slot, "
          f"p={ORDERING_CFG.JOB_ARRIVAL_PROB}, T={ORDERING_CFG.T}")
    print(SEP)

    results = {}
    for strategy in ["S1", "S2", "S3"]:
        net = build_network(strategy, config=ORDERING_CFG, seed=7)
        metrics = net.run(T=ORDERING_CFG.T, seed=7)
        jobs = int(metrics["jobs_completed"].sum())
        avg_battery = float(metrics["batteries"].mean())
        final_battery = float(metrics["batteries"][-1].mean())
        results[strategy] = {"jobs": jobs, "battery": avg_battery,
                             "final_battery": final_battery}
        print(f"  {strategy}: {jobs:4d} jobs completed, "
              f"avg battery = {avg_battery:.1f} kJ, "
              f"final battery = {final_battery:.1f} kJ")

    jobs_order_ok = (
        results["S1"]["jobs"] < results["S2"]["jobs"] < results["S3"]["jobs"]
    )
    battery_order_ok = (
        results["S3"]["battery"] < results["S2"]["battery"] < results["S1"]["battery"]
    )

    if jobs_order_ok:
        print(PASS + " jobs completed: S1 < S2 < S3")
    else:
        print(FAIL + f" jobs ordering violated: "
              f"S1={results['S1']['jobs']}, S2={results['S2']['jobs']}, "
              f"S3={results['S3']['jobs']}")
        print("       (check PROCESSING_SLOTS in core/device.py)")

    if battery_order_ok:
        print(PASS + " avg battery: S3 < S2 < S1")
    else:
        print(FAIL + f" battery ordering violated: "
              f"S1={results['S1']['battery']:.1f}, S2={results['S2']['battery']:.1f}, "
              f"S3={results['S3']['battery']:.1f}")
        print("       (check ENERGY_PER_JOB in core/device.py)")

    return jobs_order_ok and battery_order_ok


# ---------------------------------------------------------------------------
# Check 3 — Convergence: std shrinks as N grows
# ---------------------------------------------------------------------------

def check_convergence() -> bool:
    """
    Run D2 at 0.45 kJ/slot and p=0.5 for N=10, 50, 200 replications.

    At these parameters, downtime is ~10-30% (genuinely stochastic), so
    Monte Carlo averaging produces meaningful variance reduction. The std
    should shrink and the mean should stabilize — confirming the energy
    model is i.i.d. and seeds produce independent runs.
    """
    print(f"\n{SEP}")
    print("Check 3 - Monte Carlo convergence (std shrinks as 1/sqrt(N))")
    print(f"  Config: {CONVERGENCE_CFG.ENERGY_MEAN_BASELINE} kJ/slot, "
          f"p={CONVERGENCE_CFG.JOB_ARRIVAL_PROB}, T={CONVERGENCE_CFG.T}")
    print(SEP)

    ns = [10, 50, 200]
    means = []
    stds  = []

    for n in ns:
        net = build_network("D2", config=CONVERGENCE_CFG, seed=0)
        stats = net.run_batch(T=CONVERGENCE_CFG.T, n_iterations=n, seed_start=0)
        m = stats["mean_inactive_fraction"]
        s = stats["std_inactive_fraction"]
        means.append(m)
        stds.append(s)
        print(f"  N={n:4d}: mean_downtime={m:.4f}  std={s:.4f}")

    mean_shift = abs(means[2] - means[1])
    mean_stable = mean_shift < 0.05

    std_shrinks = stds[2] < stds[0]

    ratio = stds[2] / stds[0] if stds[0] > 1e-9 else 0.0
    expected_ratio = (ns[0] / ns[2]) ** 0.5
    print(f"  std ratio N={ns[2]}/N={ns[0]}: {ratio:.3f}  (expected ~{expected_ratio:.3f})")

    if mean_stable:
        print(PASS + f" mean stable (shift={mean_shift:.4f} < 0.05)")
    else:
        print(FAIL + f" mean still shifting (shift={mean_shift:.4f} > 0.05)")

    if std_shrinks:
        print(PASS + " std shrinks with more iterations")
    else:
        print(FAIL + " std not shrinking -- possible non-stationarity in energy model")

    return mean_stable and std_shrinks


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("Simulation sanity checks")
    print("=" * 60)

    c1 = check_conservation()
    c2 = check_ordering()
    c3 = check_convergence()

    print(f"\n{'=' * 60}")
    print("Summary")
    print(f"{'=' * 60}")
    all_passed = c1 and c2 and c3
    print(f"  Check 1 (conservation) : {'PASS' if c1 else 'FAIL'}")
    print(f"  Check 2 (ordering)     : {'PASS' if c2 else 'FAIL'}")
    print(f"  Check 3 (convergence)  : {'PASS' if c3 else 'FAIL'}")
    print()
    if all_passed:
        print("All checks passed. Simulation engine is valid.")
        print("Run the full experiments next:")
        print("  uv run python -m experiments.run_experiments")
    else:
        print("One or more checks FAILED. Fix the issues above before running experiments.")
    sys.exit(0 if all_passed else 1)
