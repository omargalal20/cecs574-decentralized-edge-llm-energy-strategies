"""
Simulation parameters for the decentralized LLM inference energy management study.

All constants are sourced directly from Khoshsirat et al. (GLOBECOM 2024) unless
noted otherwise. Change values here to reproduce paper experiments or run sweeps.

UNIT CONVENTION
---------------
All energy quantities are in **kJ** throughout the simulation:
  - Battery levels      : kJ  (E_max = 100 kJ)
  - Energy per job      : kJ  (C_E = 22–26 kJ per job)
  - Energy model output : kJ/slot

The paper's x-axis labels say "J/slot" but the Markov model operates on integer
kJ units (E ∈ {0, ..., E_max=100}). The energy arrival values map 1:1 to these
kJ integers: "550 J/slot" on the paper's axis = 550 kJ/slot in our model.
This is confirmed by back-calculating from Figure 2a published results.

  Paper Fig 3a sweep:  100–500  on x-axis  →  100–500 kJ/slot here
  Paper Fig 3b baseline: 550    on x-axis  →  550 kJ/slot here
"""

from dataclasses import dataclass, field


@dataclass
class SimConfig:
    # -------------------------------------------------------------------------
    # Battery / energy storage
    # -------------------------------------------------------------------------
    E_MAX: int = 100          # Maximum battery capacity [kJ]

    # Power-saving mode thresholds (trigger hysteresis on device downtime)
    # Enter power-saving when E < E_TH_LOW; exit when E > E_TH_HIGH_MARKOV.
    E_TH_LOW: int = 10        # Entry threshold [kJ] — 10% of E_MAX (paper §V)
    # Exit threshold for the Markov model [kJ].
    # Not stated explicitly in the paper; 20 kJ (20% of E_MAX) is used here
    # as a defensible choice (2× the entry threshold, enforcing hysteresis).
    E_TH_HIGH_MARKOV: int = 20

    # -------------------------------------------------------------------------
    # D3 dynamic power-mode thresholds (§V, found by manual exploration)
    # These are separate from the Markov-model thresholds above.
    # -------------------------------------------------------------------------
    PM_TH_15_TO_30_PCT: float = 0.40   # Battery fraction: switch 15W -> 30W
    PM_TH_30_TO_60_PCT: float = 0.60   # Battery fraction: switch 30W -> 60W

    # -------------------------------------------------------------------------
    # Time
    # -------------------------------------------------------------------------
    DELTA: int = 100          # Slot duration delta [seconds]
    T: int = 100              # Simulation length [stages/slots]
    N_ITERATIONS: int = 1000  # Monte Carlo iterations per experiment

    # -------------------------------------------------------------------------
    # Job arrival model (Bernoulli)
    # -------------------------------------------------------------------------
    JOB_ARRIVAL_PROB: float = 0.3   # Baseline p (paper Fig 3a sweep baseline)

    # -------------------------------------------------------------------------
    # Power modes — empirical measurements on Nvidia Jetson AGX Orin (§V)
    #
    # PM key: integer power-mode index (1=15W, 2=30W, 3=60W)
    # 50W mode excluded: same latency as 30W but higher energy/job.
    # -------------------------------------------------------------------------
    POWER_WATTS: dict = field(default_factory=lambda: {
        1: 15,
        2: 30,
        3: 60,
    })

    # Processing duration per job [stages] — kappa values
    # 15W -> 3 slots (300s), 30W -> 2 slots (200s), 60W -> 1 slot (100s)
    PROCESSING_SLOTS: dict = field(default_factory=lambda: {
        1: 3,
        2: 2,
        3: 1,
    })

    # Total system energy consumed per job [kJ] (race-to-idle effect accounts
    # for fixed-overhead subsystems running for the full job duration)
    ENERGY_PER_JOB: dict = field(default_factory=lambda: {
        1: 26,   # 15W x 300s + fixed overhead -> 26 kJ
        2: 22,   # 30W x 200s + fixed overhead -> 22 kJ
        3: 23,   # 60W x 100s + fixed overhead -> 23 kJ
    })

    # -------------------------------------------------------------------------
    # Network topology (§V: 3 groups of 3 devices each)
    # -------------------------------------------------------------------------
    N_GROUPS: int = 3
    DEVICES_PER_GROUP: int = 3

    # -------------------------------------------------------------------------
    # Markov model / scheduling
    # -------------------------------------------------------------------------
    TARGET_RISK: float = 0.01    # xi_lim threshold (1% downtime risk, §IV)
    ALPHA: float = 1.0           # D2 adaptive scheduling tuning parameter

    # -------------------------------------------------------------------------
    # Energy arrival model — values in kJ/slot
    #
    # At 550 kJ/slot and PM=2 (22 kJ/job, kappa=2):
    #   harvest over one job = 550 x 2 = 1100 kJ >> 22 kJ consumed
    #   => battery stays near full at low job rates
    #   => at high job rates (p->1.0), devices begin to drain and go offline
    # This matches the paper: Figs 3a/4a sweep from 100-500 on the x-axis.
    # -------------------------------------------------------------------------
    ENERGY_MEAN_BASELINE: float = 550.0  # Baseline mean [kJ/slot] (paper Fig 3b)
    ENERGY_SPREAD: float = 0.2           # +/-20% around mean for [low, high] bounds

    # -------------------------------------------------------------------------
    # Diurnal energy model (experiment 3 extension) — values in kJ/slot
    # -------------------------------------------------------------------------
    DIURNAL_PEAK: float = 800.0   # Peak (noon) energy arrival [kJ/slot]
    DIURNAL_BASE: float = 50.0    # Minimum (night) energy arrival [kJ/slot]
    DIURNAL_PERIOD: int = 864     # Period in slots (24h / 100s per slot)


# Default config instance — import and use directly for standard runs
DEFAULT_CONFIG = SimConfig()
