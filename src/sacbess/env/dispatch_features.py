"""Forward-tariff and temporal dispatch features appended to the scaled observation.

Literature basis for including anticipatory state in storage-dispatch RL:
  - arXiv:2410.20005: adding multi-horizon price forecasts to the state lifted DQN
    arbitrage reward by ~60%, approaching the MPC oracle - agents acting on
    SoC/current-price alone cannot plan ahead.
  - Mohammed et al., Energies 19(5):1233 (2026): temporal state features
    (hour-of-day, day-of-year) let the agent internalise diurnal TOU structure.
  - Kumar, MIT MEng thesis (2021): state = SoC + current price + forecast prices.

The TOU tariff is deterministic administrative data, not a sensor channel: the
forward features are exact by construction. They are appended AFTER the
reliability stage (never corrupted by F1-F5). Like the sin/cos time encodings,
they are index-derived and therefore shift coherently under F6a.
"""

from __future__ import annotations

import gymnasium
import numpy as np

# forecast horizons in 15-min steps: 1h, 2h, 3h, 6h, 12h, 24h (arXiv:2410.20005
# found that combining short/middle/long horizons beats any single horizon)
FORWARD_HORIZON_STEPS = (4, 8, 12, 24, 48, 96)


class DispatchFeatures(gymnasium.Wrapper):
    """Appends forward tariffs (normalized) and the hour-of-day index to the obs."""

    def __init__(self, env):
        super().__init__(env)
        core = env.unwrapped
        self._tariff = core.tariff
        self._n_rows = len(self._tariff)
        self._max_tariff = float(self._tariff.max())
        ts = core.frame.index
        self._hour_idx = (ts.hour * 4 + ts.minute // 15).to_numpy() / 95.0
        n_feat = len(FORWARD_HORIZON_STEPS) + 1
        dim = env.observation_space.shape[0] + n_feat
        lo = np.concatenate([env.observation_space.low, np.zeros(n_feat)])
        hi = np.concatenate([env.observation_space.high, np.ones(n_feat)])
        self.observation_space = gymnasium.spaces.Box(lo, hi, dtype=np.float64)

    def _augment(self, obs: np.ndarray) -> np.ndarray:
        pos = self.env.unwrapped.pos
        fwd = [self._tariff[min(pos + h, self._n_rows - 1)] / self._max_tariff
               for h in FORWARD_HORIZON_STEPS]
        return np.concatenate([obs, fwd, [self._hour_idx[pos]]]).astype(np.float64)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self._augment(obs), reward, terminated, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self._augment(obs), info
