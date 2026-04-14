# Comparative Analysis of Energy Management Strategies for Decentralized LLM Inference on Energy-Harvesting Edge Networks

**CECS 574 — Topics in Distributed Computing | CSULB Spring 2026**

This repository implements and compares seven energy management strategies for running large language models across battery-powered edge devices that charge from renewable energy sources. It extends the work of Khoshsirat et al. (IEEE GLOBECOM 2024) with two novel experiments and one novel strategy.

---

## Research Question

> How do different energy utilization strategies — static power mode selection, threshold-based dynamic power switching, and adaptive scheduling — compare in throughput, energy efficiency, and device availability for decentralized LLM inference on energy-harvesting edge networks, and **under what conditions does the added complexity of dynamic approaches yield meaningful gains?**

---

## Background and Paper Context

### The Problem

Large language models (LLMs) are increasingly deployed on **edge devices** — small, battery-powered computers (e.g., Nvidia Jetson AGX Orin) that sit close to users rather than in data centers. Edge deployment reduces latency and preserves data privacy, but introduces two hard constraints:

1. **Memory** — no single edge device can hold a full LLM. The model is split across multiple devices running as a processing pipeline.
2. **Energy** — batteries are charged by renewable harvesters (solar panels). Energy income is variable, uncertain, and correlated with time of day.

### Reference Paper

**Khoshsirat, A., Perin, G., and Rossi, M.** "Decentralized LLM Inference over Edge Networks with Energy Harvesting." *IEEE Global Communications Conference (GLOBECOM)*, 2024. [arXiv:2408.15907](https://arxiv.org/abs/2408.15907)

The paper proposes a **semi-Markov model** to analytically derive the maximum job arrival rate each device can sustain without exceeding a downtime risk threshold. It designs two scheduling strategies (D1 and D2) and one power mode controller (D3), validating them on a 3-group × 3-device network of Jetson AGX Orin nodes.

### Our Contributions

This project **reproduces** the paper's core results (Experiments 1 and 2) and **extends** them with:

- **Explicit static baselines** S1/S2/S3 that the paper only compared against implicitly
- **Experiment 3** — Diurnal (solar) energy model: replaces i.i.d. uniform harvest with a sinusoidal daily pattern that creates extended low-energy periods
- **Experiment 4** — Heterogeneous devices: introduces variation in battery capacity and solar panel strength across devices
- **Strategy D4** — Energy-proportional scheduling: a novel continuous scheduling approach that proportionally routes jobs toward energy-rich devices at every decision, without requiring a threshold crossing

---

## The Seven Strategies

The strategies form a deliberate ladder from simplest to most adaptive:

| ID | Name | Power mode | Scheduling | Source |
|---|---|---|---|---|
| **S1** | Fixed 15W | Always 15W (κ=3 slots/job, 26 kJ/job) | Uniform random | Baseline |
| **S2** | Fixed 30W | Always 30W (κ=2 slots/job, 22 kJ/job) | Uniform random | Baseline |
| **S3** | Fixed 60W | Always 60W (κ=1 slot/job, 23 kJ/job) | Uniform random | Baseline |
| **D1** | Long-term | Starts at 30W | Markov-derived q_lim weights, computed offline once | Paper §IV |
| **D2** | Adaptive | Starts at 30W | D1 weights & real-time penalty for energy-stressed (15W) devices | Paper §IV Algorithm 1 |
| **D3** | Dynamic PM | Threshold-based switching (40%→15W, 60%→30W, ≥60%→60W) | D1 weights | Paper §III/V |
| **D4** | Energy-proportional | Not fixed (per-device proportional) | Continuously weighted by live battery level | **This work** |

> **Race-to-idle effect (confirmed from paper):** S3 at 60W uses only 23 kJ/job despite triple the wattage of S2 (22 kJ/job), because it finishes faster and fixed-overhead subsystems (memory, networking) run for less time. This is not a bug — it is a real physical phenomenon documented by DynamoLLM (HPCA 2025).

### The Scientific Argument

The seven strategies let the paper make three concrete claims:

**Claim 1 — Static strategies fail under variable energy.** The performance gap between S1, S2, S3 widens as energy becomes scarce. No single fixed wattage is universally optimal.

**Claim 2 — Each layer of dynamism adds measurable gain.** D1 > S1–S3. D2 > D1. D3+D2 > D2 alone. The gains are large enough to justify the added complexity at low energy or high load.

**Claim 3 — D4 outperforms threshold-based strategies under correlated energy.** Continuous proportional weighting is smoother than reacting only at a threshold crossing, particularly when energy varies gradually (diurnal pattern) or devices are heterogeneous.

---

## The Four Experiments

| # | What it sweeps | What it proves | Paper equivalent |
|---|---|---|---|
| **Exp 1** | Mean energy arrival (0.05–0.60 kJ/slot = 50–600 J/slot) at fixed p=0.3 | Static strategies diverge first as energy becomes scarce | Figs 3a / 4a (extended) |
| **Exp 2** | Job arrival probability p (0.1–1.0) at fixed energy | Dynamic strategies drop far fewer jobs under heavy load | Figs 3b / 4b (extended) |
| **Exp 3** | Diurnal peak energy (0.15–1.15 kJ/slot) | D4 outperforms D2 under correlated solar energy; D1 degrades worst | Novel |
| **Exp 4** | Heterogeneity scale (0.0–1.0) | D4 adapts naturally to different battery sizes; D2 thresholds do not scale | Novel |

### Expected Results

**Experiments 1 & 2** (validation against paper):

- Downtime ranking at low energy: `D2+D3 < D1 < S2 < S1 < S3`
- Throughput ranking: `D2+D3 > D1 > S2 > S3 > S1`
- At high energy all strategies converge toward zero downtime

**Experiment 3** (novel):

- D4 downtime advantage over D2 should be *larger* under diurnal than i.i.d. conditions
- D1 should degrade most — its offline weights assume i.i.d. energy

**Experiment 4** (novel):

- D4 naturally handles heterogeneous E_max (proportional weighting adjusts automatically)
- D2's fixed 10 kJ threshold hits proportionally earlier on small-battery devices

---

## Project Structure

```
.
├── core/                       # Layer 1 — Simulation engine
│   ├── device.py               #   Device: battery, queue, power mode state machine
│   ├── energy.py               #   UniformEnergyModel, DiurnalEnergyModel
│   ├── markov.py               #   compute_q_lim() — semi-Markov solver
│   ├── strategies.py           #   All 7 strategies (S1–S3, D1–D4)
│   ├── network.py              #   Network orchestrator + build_network() factory
│   ├── main.py                 #   Layer 1 smoke test
│   ├── __init__.py
│   └── README.md               #   Detailed Layer 1 documentation
│
├── experiments/                # Layer 2 — Experiment runner
│   ├── config.py               #   SimConfig dataclass (all parameters)
│   ├── run_experiments.py      #   4 experiment functions + main()
│   ├── main.py                 #   Layer 2 fast smoke test
│   ├── __init__.py
│   └── README.md               #   Detailed Layer 2 documentation
│
├── app/                        # Layer 3 — Streamlit dashboard
│   ├── dashboard.py            #   Main Streamlit app (6 pages)
│   ├── plots.py                #   Plotly chart builders (one per experiment)
│   └── __init__.py
│
├── results/                    # Output CSVs (generated by experiments)
│   ├── exp1_energy_sweep.csv
│   ├── exp2_arrival_sweep.csv
│   ├── exp3_diurnal.csv
│   └── exp4_heterogeneous.csv
│
├── sanity_check.py             # 3 simulation validity checks (run before experiments)
├── pyproject.toml              # Dependencies (managed by uv)
└── README.md                   # This file
```

---

## Energy Unit Convention

All energy values in this codebase are in **kJ** (kilojoules):

| Quantity | Value | Paper equivalent |
|---|---|---|
| Battery capacity (E_max) | 100 kJ | 100 kJ |
| Energy per job — 15W | 26 kJ | 26 kJ |
| Energy per job — 30W | 22 kJ | 22 kJ |
| Energy per job — 60W | 23 kJ | 23 kJ |
| Energy arrival baseline | 0.55 kJ/slot | 550 J/slot |
| Experiment 1 sweep range | 0.05–0.60 kJ/slot | 50–600 J/slot |
| Diurnal peak | 0.80 kJ/slot | 800 J/slot |

The paper reports energy arrival in J/slot. Divide by 1000 to get kJ/slot used here.

---

## Setup

**Prerequisites:** Python 3.12+, [uv](https://docs.astral.sh/uv/)

```bash
# Clone and install dependencies
git clone <repo-url>
cd cecs574-decentralized-edge-llm-energy-strategies
uv sync
```

---

## Running — Step by Step

Run these steps in order. Each step validates the previous one before committing to the longer run.

### Step 1 — Layer 1 smoke test (~2 min)

Verifies the simulation engine components individually.

```bash
uv run python -m core.main
```

Expected: all 6 test sections print `[PASS]`. This tests energy models, the device state machine, the Markov solver, all 7 strategy `select_device()` calls, `Network.run()`, and `Network.run_batch()`.

### Step 2 — Three sanity checks (~10–15 min)

Validates the physics of the full simulation before running experiments.

```bash
uv run python sanity_check.py
```

Expected: all 3 checks print `PASS`.

| Check | What it verifies |
|---|---|
| **Conservation** | No phantom energy created; all batteries stay within [0, E_max] |
| **Ordering** | S1 < S2 < S3 jobs completed; S3 < S2 < S1 average battery |
| **Convergence** | Mean downtime stabilizes and std shrinks as N_ITERATIONS grows |

### Step 3 — Layer 2 fast smoke test (~15 min)

Runs all 4 experiments with reduced parameters (T=100, N=10, 3 sweep points) to verify CSV output schema and basic strategy ordering.

```bash
uv run python -m experiments.main
```

Expected: all checks print `PASS`. This overwrites the `results/` CSVs with fast-mode data — enough to verify the dashboard renders correctly.

### Step 4 — Full experiments (~1–3 hours)

Runs all 4 experiments with paper-quality parameters (T=100, N=1000, full sweep ranges). Recommended to run overnight.

```bash
# All 4 experiments
uv run python -m experiments.run_experiments

# Or run individual experiments
uv run python -m experiments.run_experiments --exp 1 2
uv run python -m experiments.run_experiments --exp 3 4
```

### Step 5 — Launch the Streamlit dashboard

```bash
uv run streamlit run app/dashboard.py
```

Opens at [http://localhost:8501](http://localhost:8501). The dashboard has 6 pages:

| Page | Content |
|---|---|
| **Overview** | Heatmap comparing all strategies across all experiments |
| **Exp 1** | Energy sweep — downtime, throughput, battery level plots |
| **Exp 2** | Arrival sweep — jobs dropped, throughput, downtime plots |
| **Exp 3** | Diurnal — downtime, throughput, bar comparison at mid-peak |
| **Exp 4** | Heterogeneous — downtime, throughput, battery plots |
| **Raw Data** | Browse, filter, and download any CSV |

Use the sidebar checkboxes to show/hide individual strategies and compare subsets.

---

## Key Parameters (SimConfig)

All parameters live in `experiments/config.py` and can be overridden when constructing `SimConfig(...)`.

| Parameter | Default | Meaning |
|---|---|---|
| `E_MAX` | 100 kJ | Battery capacity per device |
| `E_TH_LOW` | 10 kJ | Enter power-saving below this level (10%) |
| `E_TH_HIGH_MARKOV` | 20 kJ | Exit power-saving above this level (20%) |
| `PM_TH_15_TO_30_PCT` | 0.40 | D3: switch to 30W above 40% battery |
| `PM_TH_30_TO_60_PCT` | 0.60 | D3: switch to 60W above 60% battery |
| `DELTA` | 100 s | Time slot duration |
| `T` | 100 | Simulation length (slots per run) |
| `N_ITERATIONS` | 1000 | Monte Carlo replications per experiment |
| `JOB_ARRIVAL_PROB` | 0.3 | Bernoulli job arrival probability |
| `N_GROUPS` | 3 | Number of device groups |
| `DEVICES_PER_GROUP` | 3 | Devices per group (9 total) |
| `TARGET_RISK` | 0.01 | Max acceptable downtime risk ξ_lim (1%) |
| `ENERGY_MEAN_BASELINE` | 0.55 kJ/slot | Baseline energy arrival (= 550 J/slot) |

---

## References

| Paper | Role in this project |
|---|---|
| Khoshsirat, Perin, Rossi. *Decentralized LLM Inference over Edge Networks with Energy Harvesting.* IEEE GLOBECOM 2024. [arXiv:2408.15907](https://arxiv.org/abs/2408.15907) | Primary reference — system model, Markov framework, D1/D2/D3 strategies |
| Griggs et al. *DynamoLLM: Designing LLM Inference Clusters for Performance and Energy Efficiency.* HPCA 2025. | Race-to-idle effect; motivation for dynamic power management |
| Tian et al. *CLONE: Customizing LLMs for Efficient Latency-Aware Inference at the Edge.* USENIX ATC 2025. | Per-device DVFS for LLM inference; motivation for D3/D4 |
| Radovanovic et al. *Spatio-temporal load shifting for truly clean computing.* 2024. | Energy-proportional scheduling precedent for D4 |

---

## Layer Documentation

For deeper technical detail on each layer:

- [`core/README.md`](core/README.md) — Simulation engine: state model, Markov internals, strategy classes, network tick order
- [`experiments/README.md`](experiments/README.md) — Experiment runner: CSV schema, parameter sweep design, runtime estimates
