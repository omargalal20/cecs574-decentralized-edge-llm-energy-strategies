# `core/` — Simulation Engine (Layer 1)

This package is the mathematical and computational heart of the project.
It implements the semi-Markov model and all seven energy management strategies
from **Khoshsirat et al., "Decentralized LLM Inference over Edge Networks with
Energy Harvesting," IEEE GLOBECOM 2024**, plus the novel D4 strategy introduced
in this work.

> **This README is referenced by the project-level README.**
> Layer 2 (`experiments/`) and Layer 3 (`app/`) import from `core/` but never
> modify it — all simulation logic lives here.

---

## Files at a glance

| File            | Role                                                                |
|-----------------|---------------------------------------------------------------------|
| `device.py`     | `Device` — stateful edge node (battery, queue, power mode)          |
| `energy.py`     | `UniformEnergyModel`, `DiurnalEnergyModel` — energy arrival models  |
| `markov.py`     | `compute_q_lim()` — semi-Markov solver for max sustainable job rate |
| `strategies.py` | All 7 strategies: S1–S3 (static), D1–D4 (dynamic)                   |
| `network.py`    | `Network` orchestrator + `build_network()` factory                  |
| `main.py`       | Smoke-test runner for Layer 1                                       |
| `__init__.py`   | Public re-exports                                                   |

---

## System model (paper §III)

### State representation

Each device is modelled as a semi-Markov chain with state tuple:

```
S_m = (Q_m, E_m, γ_m)
```

| Variable | Domain        | Meaning                                  |
|----------|---------------|------------------------------------------|
| `Q_m`    | {0, 1}        | Queue occupancy (0=empty, 1=job present) |
| `E_m`    | {0, …, E_max} | Discrete battery level [kJ]              |
| `γ_m`    | {0, 1}        | Mode (0=power-saving, 1=active)          |

### Battery update (Equation 1)

```python
E_next = np.clip(E + harvested_kJ - C_E(PM), 0, E_max)
```

`C_E(PM)` is **only subtracted in processing states** (`Q=1, γ=1`).
All other states consume zero battery.

### Power-saving hysteresis

| Condition              | Effect               |
|------------------------|----------------------|
| `E < E_th_low` (enter) | γ ← 0, queue cleared |
| `E > E_th_high` (exit) | γ ← 1                |

Default thresholds (from `experiments/config.py`):

- `E_th_low = 10 kJ` (10% of 100 kJ battery)
- `E_th_high = 20 kJ` (20% — 2× entry threshold, chosen to enforce hysteresis)

### Empirical power mode parameters (Jetson AGX Orin, paper §V)

| PM index | Power | Processing time | Energy/job | κ (slots) |
|----------|-------|-----------------|------------|-----------|
| 1        | 15 W  | 300 s           | 26 kJ      | 3         |
| 2        | 30 W  | 200 s           | 22 kJ      | 2         |
| 3        | 60 W  | 100 s           | 23 kJ      | 1         |

> **Race-to-idle effect:** 15 W uses *more* energy per job than 30 W because
> fixed-overhead subsystems (memory refresh, network interfaces) run for the
> full 300 s duration, regardless of compute intensity.
> The 50 W mode is excluded: same latency as 30 W but higher energy/job.

---

## Strategy summary

### Static baselines

| ID | Class                | Power mode   | Scheduling     |
|----|----------------------|--------------|----------------|
| S1 | `StaticScheduler(1)` | 15 W (fixed) | Uniform random |
| S2 | `StaticScheduler(2)` | 30 W (fixed) | Uniform random |
| S3 | `StaticScheduler(3)` | 60 W (fixed) | Uniform random |

### Dynamic strategies (paper)

| ID | Class                 | Description                                                   |
|----|-----------------------|---------------------------------------------------------------|
| D1 | `LongTermScheduler`   | Markov-derived q_lim weights, computed offline once           |
| D2 | `AdaptiveScheduler`   | D1 weights + real-time penalty for PM=1 devices (Algorithm 1) |
| D3 | `PowerModeController` | Per-device PM switching at 40%/60% battery thresholds         |

### Novel contribution (this work)

| ID | Class                         | Description                                        |
|----|-------------------------------|----------------------------------------------------|
| D4 | `EnergyProportionalScheduler` | Continuous battery-proportional scheduling weights |

D4 differs from D2 in that it is **always active** (no threshold trigger) and
**fully continuous** (not derived from static Markov weights). Load is
proportionally shifted toward whichever device has the most energy at every
single scheduling decision.

---

## Markov model internals (`markov.py`)

### Six transition cases (paper §III)

| Case | Q_m | γ_m | Q_{m+1} | C_E deducted? | Stage duration   |
|------|-----|-----|---------|---------------|------------------|
| 1    | 0   | 1   | 0       | No            | κ=1              |
| 2    | 0   | 1   | 1       | No            | κ=1              |
| 3    | 0   | 0   | 0       | No            | κ=1              |
| 4    | 1   | 0   | 1       | No            | κ=1              |
| 5    | 1   | 1   | 0       | **Yes**       | κ (PM-dependent) |
| 6    | 1   | 1   | 1       | **Yes**       | κ (PM-dependent) |

### Key equations

**Eq. 3 — ξ_lim (downtime risk):**

```
ξ_lim = Σ_{S: E≤E_lim} π_S T_S  /  Σ_S π_S T_S
```

**Eq. 4 — κ̄ (expected processing slots per job):**

```
κ̄ = Σ_{S: Q=1,γ=1} π_S κ_S  /  Σ_{S: Q=1,γ=1} π_S
```

**Eq. 5 — q_lim (final max arrival rate):**

```
q_lim = min(q_energy_lim, 1/κ̄)
```

**Eq. 6 — scheduling weight for device i in group ℓ:**

```
r_i = q_lim,i / Σ_{j∈ℓ} q_lim,j
```

`q_energy_lim` is found via `scipy.optimize.brentq` on the function
`risk_function(q) = ξ_lim(q) − target_risk`.

---

## Network simulation loop (`network.py`)

Tick order (one time slot = 100 s = δ):

```
for t in 0 … T-1:
    1. For each device: sample energy, call device.step(harvested)
    2. [D3 only] power_controller.update(devices) — set PM by battery level
    3. [D2/D4]   scheduler.update(devices)        — refresh real-time state
    4. For each group:
           if Bernoulli(p) arrival:
               chosen = scheduler.select_device(group, t)
               if chosen: chosen.accept_job()
               else:      drop job
    5. Record metrics
```

---

## Running the smoke test

Install dependencies first (from project root):

```bash
uv sync
```

Then run the Layer 1 tests:

```bash
uv run python -m core.main
```

Expected output: all six test sections print **PASS**.

> **Note on test 3 (Markov):** Computing `compute_q_lim` for all three power
> modes takes approximately 10–30 seconds on a modern laptop because it builds
> a ~200-state transition matrix, solves for the stationary distribution, and
> runs Brent's method. This is a one-time offline cost per device per
> simulation run — it is not incurred during the simulation loop itself.

---

## Quick usage example

```python
from core import build_network
from experiments.config import SimConfig

cfg = SimConfig(T=100, N_ITERATIONS=1000)

# Single run — D4 energy-proportional strategy
net = build_network("D4", config=cfg, seed=42)
metrics = net.run(T=100, seed=0)

print(f"Jobs completed: {metrics['jobs_completed'].sum()}")
print(f"Jobs dropped:   {metrics['jobs_dropped'].sum()}")
print(f"Mean downtime:  {metrics['inactive_fraction'].mean():.1%}")

# Batch run — 1000 replications, returns mean ± std
agg = net.run_batch(T=100, n_iterations=1000, seed_start=0)
print(f"Mean throughput: {agg['mean_jobs_completed']:.1f} ± {agg['std_jobs_completed']:.1f}")
```

---

## Design decisions and known limitations

| Decision                                                            | Rationale                                                                                                                                         |
|---------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------|
| `E_th_high = 20 kJ` for Markov                                      | Not stated in paper; 2× entry threshold chosen for defensible hysteresis. Marked as configurable in `SimConfig`.                                  |
| Energy model PMF uses integer grid                                  | Paper discretizes battery to kJ units; PMF must match this granularity.                                                                           |
| Stationary distribution via linear solve + power iteration fallback | Linear solve is faster; power iteration handles numerically singular matrices.                                                                    |
| `_patch_devices()` in Network                                       | Avoids a second per-device loop to collect step() return values; can be replaced with an explicit results list if monkey-patching is undesirable. |
| D3 uses D1 weights for scheduling                                   | Paper pairs dynamic PM with long-term scheduling as its "combined" strategy.                                                                      |
