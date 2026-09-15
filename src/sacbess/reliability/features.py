"""Streaming reliability features per sensor channel: [v, tau, hampel z, rate jump, rho_a, rho_b].

All state is episode-local and causal: the stage sees exactly the corrupted value
stream the policy sees, plus row-aligned site priors (climatology ceiling, daylight
gate, PV performance coefficient, meter profiles) fitted on TRAINING rows only.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

N_SENSOR = 8
FEAT_DIM = 6
WINDOW = 32
TAU_CAP = 96
TAU_ALARM_STEPS = 8
DELTA_Z_ALARM = 6.0

SENSOR_PHYS_LO = np.array([0.0, 0.0, 0.0, 0.0, 0.0, -40.0, -40.0, 0.0])
SENSOR_PHYS_HI = np.array([2000.0, 2000.0, 2000.0, 1500.0, 1500.0, 100.0, 100.0, 1500.0])
SOLAR_GATED = np.array([True, False, False, True, True, True, True, True])
CHANGE_DEADBAND = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.05, 0.05, 0.5])
MAD_FLOOR = np.array([0.4, 0.4, 0.4, 1.0, 1.0, 0.05, 0.05, 1.0])
PHYS_RATE = np.array([600.0, 600.0, 600.0, 600.0, 600.0, 3.0, 3.0, 600.0])
SENTINELS = (-999.0, 888.89)


@dataclass
class SitePriors:
    ceiling: np.ndarray
    daylight: np.ndarray
    k_pv: float
    profile_a: np.ndarray
    profile_b: np.ndarray
    quarter: np.ndarray

    @classmethod
    def fit(cls, frame: pd.DataFrame, train_mask: np.ndarray) -> SitePriors:
        sat = frame["sat_gti_wm2"].to_numpy()
        month = frame.index.month.to_numpy()
        hour = frame.index.hour.to_numpy()
        key = (month - 1) * 24 + hour
        ceiling = np.zeros(len(frame))
        for k in np.unique(key):
            sel = train_mask & (key == k)
            ref = sat[sel] if sel.sum() >= 8 else sat[train_mask]
            ceiling[key == k] = np.quantile(ref, 0.90)
        daylight = ceiling > 20.0
        dl_train = train_mask & daylight
        denom = sat[dl_train].sum()
        k_pv = float(frame["pm1_kw"].to_numpy()[dl_train].sum() / denom) if denom > 0 else 0.0
        quarter = (frame.index.hour * 4 + frame.index.minute // 15).to_numpy()
        dow = frame.index.dayofweek.to_numpy()
        g = dow * 96 + quarter
        profile_a = np.zeros(len(frame))
        profile_b = np.zeros(len(frame))
        for k in np.unique(g):
            sel = train_mask & (g == k)
            ref_a = frame["pm0_a_kw"].to_numpy()[sel]
            ref_b = frame["pm0_b_kw"].to_numpy()[sel]
            profile_a[g == k] = np.median(ref_a) if sel.sum() >= 8 else np.median(frame["pm0_a_kw"].to_numpy()[train_mask])
            profile_b[g == k] = np.median(ref_b) if sel.sum() >= 8 else np.median(frame["pm0_b_kw"].to_numpy()[train_mask])
        return cls(ceiling, daylight, k_pv, profile_a, profile_b, quarter)


class ReliabilityFeatures:
    """Per-episode streaming state; call reset() at episode start, then step(obs, pos)."""

    def __init__(self, priors: SitePriors):
        self.priors = priors
        self._tau = np.zeros(N_SENSOR)
        self._prev = np.full(N_SENSOR, np.nan)
        self._raw_win: list[np.ndarray] = []
        self._valid_win: list[np.ndarray] = []
        self._mu = np.full(N_SENSOR, np.nan)

    def reset(self) -> None:
        self._tau = np.zeros(N_SENSOR)
        self._prev = np.full(N_SENSOR, np.nan)
        self._raw_win = []
        self._valid_win = []
        self._mu = np.full(N_SENSOR, np.nan)

    def step(self, obs: np.ndarray, pos: int) -> tuple[np.ndarray, np.ndarray]:
        """Returns (features (8,6), mu prior (8,)). obs is the corrupted raw vector."""
        x = obs[:N_SENSOR].astype(float)
        daylight = bool(self.priors.daylight[pos])

        valid = np.array(
            [
                (not np.isnan(xi))
                and (SENSOR_PHYS_LO[c] - 1e-9 <= xi <= SENSOR_PHYS_HI[c] + 1e-9)
                and all(abs(xi - s) > 1e-6 for s in SENTINELS)
                for c, xi in enumerate(x)
            ]
        )

        prev = np.where(np.isnan(self._prev), x, self._prev)
        changed = np.abs(x - prev) > CHANGE_DEADBAND
        counts = np.array([daylight if SOLAR_GATED[c] else True for c in range(N_SENSOR)])
        self._tau = np.where(changed, 0.0, self._tau + counts.astype(float))
        self._prev = x.copy()

        self._raw_win.append(x.copy())
        if len(self._raw_win) > WINDOW:
            self._raw_win.pop(0)
        stack = np.stack(self._raw_win)
        if len(self._raw_win) >= 4:
            med = np.median(stack, axis=0)
            mad = np.median(np.abs(stack - med), axis=0)
            z = np.abs(x - med) / (1.4826 * mad + MAD_FLOOR)
        else:
            z = np.zeros(N_SENSOR)
        hampel_norm = np.clip(z / 10.0, 0.0, 1.0)
        jump_ratio = np.minimum(np.abs(x - prev) / PHYS_RATE, 2.0) / 2.0

        self._valid_win.append(np.where(valid, x, np.nan))
        if len(self._valid_win) > WINDOW:
            self._valid_win.pop(0)
        if self._valid_win:
            vw = np.stack(self._valid_win)
            with np.errstate(all="ignore"):
                med_v = np.nanmedian(vw, axis=0) if not np.isnan(vw).all() else np.full(N_SENSOR, np.nan)
            upd = ~np.isnan(med_v)
            self._mu[upd] = med_v[upd]
        mu = np.where(np.isnan(self._mu), x, self._mu)

        rho = self._residuals(x, pos)

        feats = np.zeros((N_SENSOR, FEAT_DIM))
        feats[:, 0] = valid.astype(float)
        feats[:, 1] = np.minimum(self._tau, TAU_CAP) / TAU_CAP
        feats[:, 2] = hampel_norm
        feats[:, 3] = jump_ratio
        feats[:, 4] = rho[:, 0]
        feats[:, 5] = rho[:, 1]
        return feats, mu

    def _residuals(self, x: np.ndarray, pos: int) -> np.ndarray:
        p = self.priors
        rho = np.zeros((N_SENSOR, 2))
        irr1, irr2, sat = x[3], x[4], x[7]
        env_pair = (5.0 + 0.025 * max(irr1, irr2, 0.0)) * np.sqrt(2.0) + 1e-6
        rho1 = min(abs(irr1 - irr2) / env_pair, 5.0) / 5.0
        rho[3, 0] = rho1
        rho[4, 0] = rho1
        ceil = p.ceiling[pos]
        for c, irr in ((3, irr1), (4, irr2)):
            rho[c, 1] = min(max(irr - ceil, 0.0) / 50.0, 5.0) / 5.0
        pv_pred = p.k_pv * irr1
        rho[0, 0] = min(abs(x[0] - pv_pred) / (0.2 * max(pv_pred, 150.0)), 5.0) / 5.0
        for c, prof in ((1, p.profile_a[pos]), (2, p.profile_b[pos])):
            rho[c, 0] = min(abs(x[c] - prof) / (0.25 * max(prof, 80.0)), 5.0) / 5.0
        t_pair = abs(x[5] - x[6]) / 1.4
        rho[5, 0] = min(t_pair, 5.0) / 5.0
        rho[6, 0] = rho[5, 0]
        rho[7, 0] = min(abs(sat - irr1) / (20.0 + 0.1 * max(irr1, 0.0) + 1e-6), 5.0) / 5.0
        return rho

    def alarms(self, feats: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        valid = feats[:, 0] > 0.5
        tau_alarm = (feats[:, 1] * TAU_CAP) >= TAU_ALARM_STEPS
        return valid, tau_alarm
