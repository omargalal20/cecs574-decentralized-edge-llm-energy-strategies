# `experiments/` — Experiment Runner (Layer 2)

This package defines and runs the four comparative experiments. It sits on top
of `core/` (Layer 1) and produces CSV files in `results/` that feed directly
into the Streamlit dashboard (Layer 3) and the paper's Results section.

> **This README is referenced by the project-level README.**
> Layer 2 imports from `core/` but never modifies it.

---

## Files at a glance

| File | Role |
|---|---|
| `config.py` | `SimConfig` dataclass — all simulation parameters |
| `run_experiments.py` | Four experiment functions + `main()` entry point |
| `main.py` | Fast smoke-test runner for Layer 2 |
| `__init__.py` | Package marker |

---

## Experiments overview

All experiments run all **7 strategies** (S1, S2, S3, D1, D2, D3, D4) and
save one CSV per experiment to `results/`.

### Experiment 1 — Energy arrival rate sweep

**Reproduces paper Figures 3a / 4a, extended to all 7 strategies.**

- X-axis: mean energy arrival [J/slot], 50 → 600 (12 points)
- Fixed: job arrival probability p = 0.3
- Metrics: inactive fraction, normalised throughput

### Experiment 2 — Job arrival probability sweep

**Reproduces paper Figures 3b / 4b, extended to all 7 strategies.**

- X-axis: Bernoulli probability p, 0.1 → 1.0 (10 points)
- Fixed: mean energy = 550 J/slot
- Metrics: inactive fraction, jobs dropped

### Experiment 3 — Diurnal energy model *(novel contribution)*

**First test of correlated temporal energy structure on strategy ranking.**

- Replaces i.i.d. uniform model with sinusoidal solar profile
- X-axis: diurnal peak [J/slot], 150 → 1150 (giving means 100–600 J/slot,
  matching Exp 1's range for direct comparison)
- Fixed: base = 50 J/slot, period = T = 100 slots (one full sinusoidal cycle per run)
  > Note: the physically realistic 24h period is 864 slots at δ=100s, but this makes
  > the diurnal structure invisible within a T=100 run. Period is set to T so every
  > simulation run experiences a complete day-night cycle.
- Metrics: inactive fraction, normalised throughput

### Experiment 4 — Heterogeneous devices *(novel contribution)*

**Tests whether strategy rankings change when devices are not identical.**

- Each device has a different E_max and solar panel strength
- X-axis: heterogeneity scale factor (0.0 = homogeneous, 1.0 = highly varied)
- Metrics: inactive fraction, normalised throughput

---

## Output CSV schema

All CSVs share this column schema:

| Column | Description |
|---|---|
| `strategy` | One of: S1, S2, S3, D1, D2, D3, D4 |
| `param_value` | X-axis value for this row |
| `mean_inactive_fraction` | Mean fraction of devices in power-saving mode |
| `std_inactive_fraction` | Std dev across N_ITERATIONS runs |
| `mean_throughput` | Mean normalised job throughput (completed / arrived) |
| `std_throughput` | Std dev across runs |
| `mean_jobs_dropped` | Mean number of jobs dropped per run |
| `std_jobs_dropped` | Std dev |
| `mean_jobs_completed` | Mean number of jobs completed per run |
| `std_jobs_completed` | Std dev |
| `mean_battery` | Mean battery level [kJ] across all devices and time |
| `std_battery` | Std dev |

Exp 3 adds `mean_energy_equiv` (equivalent mean energy [J/slot] for comparison
with Exp 1). Exp 4 adds `mean_e_max` and `mean_energy`.

---

## Running experiments

### Quick validation (fast mode, ~2-5 min)

```bash
uv run python -m experiments.main
# or
python -m experiments.main
```

Uses T=20, N_ITERATIONS=5, 3 sweep points — validates structure only.

### Single experiment

```bash
uv run python -m experiments.run_experiments --exp 1
uv run python -m experiments.run_experiments --exp 1 3
```

### Full run (all 4 experiments, paper-quality results)

```bash
uv run python -m experiments.run_experiments
```

> **Runtime estimate:** With the default config (T=100, N_ITERATIONS=1000),
> a full run takes approximately 2–6 hours depending on machine speed.
> The Markov solver runs once per (strategy, sweep-point) combination for D1/D2/D3
> and dominates the runtime. S1–S3 and D4 are much faster (no Markov computation).
>
> Recommended workflow: run the fast smoke test first to confirm correctness,
> then kick off the full run overnight.

---

## Key design decisions

### Per-device energy model heterogeneity

Each device's energy bounds are drawn around a shared mean with ±20% spread
(controlled by `SimConfig.ENERGY_SPREAD`). This reflects the paper's description
of "unique energy profiles of individual edge devices" due to different solar
panel models or weather conditions (paper §II).

### Markov solver reuse across replications

`compute_q_lim()` is called once per device during scheduler construction
(before `run_batch()`). The stationary distribution is reused across all
N_ITERATIONS replications — consistent with the paper's statement that
"the stationary distribution only needs to be computed once unless the
network parameters change" (§III).

### Diurnal experiment: Markov solver uses uniform equivalent

D1/D2/D3 require `UniformEnergyModel` to compute q_lim (the Markov PMF
requires a discrete uniform distribution). In Exp 3, the Markov solver
receives a uniform model with the same mean as the diurnal model, then
the actual simulation uses the diurnal model at runtime. This is a
reasonable approximation since the long-term average energy income is
what the Markov model optimises for.

### Normalised throughput definition

`throughput = jobs_completed / jobs_arrived` per run. This matches the
paper's Figure 4a definition ("normalized job throughput [job/slot]").
Zero arrivals (possible for very low p) are handled by returning 0.0.

---

## SimConfig reference

All parameters are in `experiments/config.py`. Key values:

| Parameter | Default | Source |
|---|---|---|
| `E_MAX` | 100 kJ | Paper §V |
| `E_TH_LOW` | 10 kJ | Paper §V (10%) |
| `E_TH_HIGH_MARKOV` | 20 kJ | This work (2x entry threshold) |
| `PM_TH_15_TO_30_PCT` | 0.40 | Paper §V (40%) |
| `PM_TH_30_TO_60_PCT` | 0.60 | Paper §V (60%) |
| `DELTA` | 100 s | Paper §V |
| `T` | 100 | Paper §V |
| `N_ITERATIONS` | 1000 | Paper §V |
| `JOB_ARRIVAL_PROB` | 0.3 | Paper Fig 3a baseline |
| `ENERGY_MEAN_BASELINE` | 0.55 kJ/slot (= 550 J/slot) | Paper Fig 3b baseline |
| `TARGET_RISK` | 0.01 | Paper §IV (1%) |
