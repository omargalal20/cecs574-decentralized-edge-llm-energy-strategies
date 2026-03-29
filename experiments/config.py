"""
Simulation parameters for the decentralized LLM inference energy management study.

All constants are sourced directly from Khoshsirat et al. (GLOBECOM 2024) unless
noted otherwise. Change values here to reproduce paper experiments or run sweeps.
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
    PM_TH_15_TO_30_PCT: float = 0.40   # Battery fraction: switch 15W → 30W
    PM_TH_30_TO_60_PCT: float = 0.60   # Battery fraction: switch 30W → 60W

    # -------------------------------------------------------------------------
    # Time
    # -------------------------------------------------------------------------
    DELTA: int = 100          # Slot duration δ [seconds]
    T: int = 100              # Simulation length [stages/slots]
    N_ITERATIONS: int = 1000  # Monte Carlo iterations per experiment

    # -------------------------------------------------------------------------
    # Job arrival model (Bernoulli)
    # -------------------------------------------------------------------------
    JOB_ARRIVAL_PROB: float = 0.3   # Baseline p (Fig 3a sweep baseline)

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

    # Processing duration per job [stages] — κ values
    # 15W → 3 slots (300s), 30W → 2 slots (200s), 60W → 1 slot (100s)
    PROCESSING_SLOTS: dict = field(default_factory=lambda: {
        1: 3,
        2: 2,
        3: 1,
    })

    # Total system energy consumed per job [kJ] (race-to-idle effect accounts
    # for fixed-overhead subsystems running for the full job duration)
    ENERGY_PER_JOB: dict = field(default_factory=lambda: {
        1: 26,   # 15W × 300s + fixed overhead → 26 kJ
        2: 22,   # 30W × 200s + fixed overhead → 22 kJ
        3: 23,   # 60W × 100s + fixed overhead → 23 kJ
    })

    # -------------------------------------------------------------------------
    # Network topology (§V: 3 groups of 3 devices each)
    # -------------------------------------------------------------------------
    N_GROUPS: int = 3
    DEVICES_PER_GROUP: int = 3

    # -------------------------------------------------------------------------
    # Markov model / scheduling
    # -------------------------------------------------------------------------
    TARGET_RISK: float = 0.01    # ξ_lim threshold (1% downtime risk, §IV)
    ALPHA: float = 1.0           # D2 adaptive scheduling tuning parameter

    # -------------------------------------------------------------------------
    # Energy arrival model (baseline for experiments 1–2)
    # Each device gets its own uniform [low, high] bounds (J/slot).
    # The paper uses distinct per-device means; these are representative defaults.
    # -------------------------------------------------------------------------
    ENERGY_MEAN_BASELINE: float = 550.0   # Baseline mean (Fig 3b sweep baseline)
    ENERGY_SPREAD: float = 0.2            # ±20% around mean for [low, high] bounds

    # -------------------------------------------------------------------------
    # Diurnal energy model (experiment 3 extension)
    # -------------------------------------------------------------------------
    DIURNAL_PEAK: float = 800.0      # Peak energy arrival [J/slot]
    DIURNAL_BASE: float = 50.0       # Minimum (night) energy arrival [J/slot]
    DIURNAL_PERIOD: int = 864        # Period in slots (24h / 100s per slot)


# Default config instance — import and use directly for standard runs
DEFAULT_CONFIG = SimConfig()
