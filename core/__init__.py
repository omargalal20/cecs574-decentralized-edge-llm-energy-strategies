"""
core — simulation engine for decentralized LLM inference energy management.

Public API
----------
Energy models:
    UniformEnergyModel   — i.i.d. uniform arrivals (paper original)
    DiurnalEnergyModel   — sinusoidal solar model (Experiment 3 extension)

Device:
    Device               — stateful edge device (battery + queue + power mode)

Markov model:
    compute_q_lim             — max sustainable job arrival rate for one device
    compute_scheduling_weights — normalised D1/D2 weights for a device group

Strategies:
    StaticScheduler             — S1 (15W), S2 (30W), S3 (60W)
    LongTermScheduler           — D1
    AdaptiveScheduler           — D2
    PowerModeController         — D3
    EnergyProportionalScheduler — D4

Network:
    Network      — simulation orchestrator
    build_network — factory: build Network from strategy name string
"""

from core.device import Device
from core.energy import DiurnalEnergyModel, UniformEnergyModel
from core.markov import compute_q_lim, compute_scheduling_weights
from core.network import Network, build_network
from core.strategies import (
    BaseScheduler,
    StaticScheduler,
    LongTermScheduler,
    AdaptiveScheduler,
    PowerModeController,
    EnergyProportionalScheduler,
)

__all__ = [
    # energy models
    "UniformEnergyModel",
    "DiurnalEnergyModel",
    # device
    "Device",
    # markov
    "compute_q_lim",
    "compute_scheduling_weights",
    # strategies
    "BaseScheduler",
    "StaticScheduler",
    "LongTermScheduler",
    "AdaptiveScheduler",
    "PowerModeController",
    "EnergyProportionalScheduler",
    # network
    "Network",
    "build_network",
]
