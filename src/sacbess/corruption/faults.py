"""F1-F6 fault parameters by severity, fitted to the test-site-01 13-month audit.

Baseline (severity 2) rates derive from measured counts over 38,016 intervals:
F1 ~29 gaps/meter -> p_onset 7.6e-4/step, longest 328 h;
F2 14 fault-class zero-runs on PM0_A -> 3.7e-4/step, longest 29.8 h;
F3 ~5 register spikes/raw meter -> 1.3e-4/step, artefacts to 1e6+ in channel units;
F4 3.85% peak-to-trough seasonal IRR divergence -> OU stationary sigma 0.0385.
F5/F6 are held out for evaluation; magnitudes are design defaults pending tariff/CT confirmation.
"""

from __future__ import annotations

from dataclasses import dataclass

DT_H = 0.25
STEPS_PER_H = 4


@dataclass
class F1Params:
    p_onset: float
    med_h: float
    sig_h: float
    cap_h: float


@dataclass
class F2Params:
    p_onset: float
    med_h: float
    sig_h: float
    cap_h: float


@dataclass
class F3Params:
    p_spike: float
    mag_lo: float
    mag_hi: float
    p_sentinel: float
    sentinel_value: float


@dataclass
class F4Params:
    sigma_stat: float
    theta: float


@dataclass
class F5Params:
    drift_frac_30d: float


@dataclass
class F6Params:
    shift_steps: int


SEVERITY_TABLES: dict[str, dict[int, dict]] = {
    "F1": {
        1: F1Params(3.8e-4, 6.0, 1.2, 200.0),
        2: F1Params(7.6e-4, 12.0, 1.3, 330.0),
        3: F1Params(1.9e-3, 24.0, 1.4, 430.0),
    },
    "F2": {
        1: F2Params(1.85e-4, 3.0, 1.0, 24.0),
        2: F2Params(3.7e-4, 6.0, 1.1, 32.0),
        3: F2Params(9.25e-4, 12.0, 1.2, 48.0),
    },
    "F3": {
        1: F3Params(6.6e-5, 1e4, 1e6, 0.15, 888.89),
        2: F3Params(1.3e-4, 1e4, 1e6, 0.15, 888.89),
        3: F3Params(3.3e-4, 1e4, 1e6, 0.15, 888.89),
    },
    "F4": {
        1: F4Params(0.01, 1.0 / 1920.0),
        2: F4Params(0.0385, 1.0 / 1920.0),
        3: F4Params(0.08, 1.0 / 1920.0),
    },
    "F5": {1: F5Params(0.02), 2: F5Params(0.05), 3: F5Params(0.10)},
    "F6": {1: F6Params(1), 2: F6Params(4), 3: F6Params(12)},
}

TRAIN_FAULTS = ["F1", "F2", "F3", "F4"]
HELD_OUT_FAULTS = ["F5", "F6a", "F6b"]
