"""S2 deterministic rule.

Rule inputs: v (hard validity), tau (staleness, solar-gated), physical rate-jump,
and the PHYSICS residuals (rho1 sensor pair, rho2 performance, rho3 clear-sky
ceiling, temp pair, sat-vs-sensor). Meter channels carry no physics residual: no
fixed load-persistence threshold survives load-forecast error, so they rely on
validity/staleness/rate only. Residual exceedance uses a 1.5x envelope deadband
and a soft weight; validity, staleness and rate violations are hard.
"""

from __future__ import annotations

import numpy as np

from .features import TAU_ALARM_STEPS

RHO_DEADBAND = 1.5
RHO_SOFT_WEIGHT = 0.6
RHO_EXCLUDED = {1, 2}


def alarms(feats: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    valid = feats[:, 0] > 0.5
    tau_alarm = (feats[:, 1] * 96.0) >= TAU_ALARM_STEPS
    return valid, tau_alarm


def r_det_from_features(feats: np.ndarray) -> np.ndarray:
    """feats: (8,6) = [v, tau_norm, hampel_z, jump_ratio/2, rho_a/5, rho_b/5]. Returns r in [0,1]."""
    valid, tau_alarm = alarms(feats)
    jump_ratio = feats[:, 3] * 2.0
    rho_ratio = np.maximum(feats[:, 4], feats[:, 5]) * 5.0
    rho_ratio[[1, 2]] = 0.0
    rho_excess = np.clip((rho_ratio - RHO_DEADBAND) / RHO_DEADBAND, 0.0, 1.0)
    jump_excess = np.clip(jump_ratio - 1.0, 0.0, 1.0)
    excess = np.maximum(jump_excess, RHO_SOFT_WEIGHT * rho_excess)
    r = 1.0 - 0.5 * excess
    r = np.where(tau_alarm, 0.1, r)
    r = np.where(~valid, 0.0, r)
    return r
