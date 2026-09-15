"""ReliabilityWrapper: S2 (deterministic) / S3 (learned) gate-and-flag stage.

Chain position: ScaleObservation -> ReliabilityWrapper -> CorruptionWrapper -> MicrogridEnv.
Operates on physical corrupted values; appends [r; v; tau] flags per sensor channel
(obs 22 -> 46). The wrapper never reads the ground-truth mask - the mask only
supervises the S3 head offline.
"""

from __future__ import annotations

import gymnasium
import numpy as np

from ..data.channels import OBS_CHANNELS
from ..data.splits import build_splits, row_split_mask
from .deterministic import r_det_from_features
from .features import (
    N_SENSOR,
    SENSOR_PHYS_HI,
    SENSOR_PHYS_LO,
    ReliabilityFeatures,
    SitePriors,
)

REL_FLAG_CHANNELS = (
    [f"{c}__r" for c in OBS_CHANNELS[:N_SENSOR]]
    + [f"{c}__v" for c in OBS_CHANNELS[:N_SENSOR]]
    + [f"{c}__tau" for c in OBS_CHANNELS[:N_SENSOR]]
)
REL_OBS_CHANNELS = OBS_CHANNELS + REL_FLAG_CHANNELS
OBS_DIM_REL = len(REL_OBS_CHANNELS)

_PRIOR_CACHE: dict = {}


def _get_priors(core) -> SitePriors:
    key = str(core.canonical_path)
    if key not in _PRIOR_CACHE:
        weeks = build_splits(core.frame, embargo_steps=96, seed=0)
        train_mask = row_split_mask(core.frame, weeks, "train")
        _PRIOR_CACHE[key] = SitePriors.fit(core.frame, train_mask)
    return _PRIOR_CACHE[key]


class ReliabilityWrapper(gymnasium.ObservationWrapper):
    def __init__(self, env, mode: str = "S2", head_path: str | None = None, alpha: float = 0.2):
        super().__init__(env)
        assert mode in ("S2", "S3")
        self.mode = mode
        self.alpha = alpha
        self.core = env.unwrapped
        self.head = None
        if mode == "S3":
            import torch

            ckpt_kind = torch.load(head_path, map_location="cpu")
            kind = ckpt_kind.get("kind") if isinstance(ckpt_kind, dict) else None
            if kind == "conv":
                from .conv_head import FrozenConvHead

                self.head = FrozenConvHead(head_path)
            else:
                from .learned_head import FrozenHead

                self.head = FrozenHead(head_path)
        self.priors = _get_priors(self.core)
        self.features = ReliabilityFeatures(self.priors)
        lo = np.concatenate([SENSOR_PHYS_LO, self.core.observation_space.low[N_SENSOR:]])
        hi = np.concatenate([SENSOR_PHYS_HI, self.core.observation_space.high[N_SENSOR:]])
        lo = np.concatenate([lo, np.zeros(3 * N_SENSOR)])
        hi = np.concatenate([hi, np.ones(3 * N_SENSOR)])
        self.observation_space = gymnasium.spaces.Box(lo, hi, dtype=np.float64)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        obs, info = self.env.reset(seed=seed, options=options)
        self.features.reset()
        if self.head is not None and hasattr(self.head, "reset"):
            self.head.reset()  # conv head: clear its rolling window per episode
        out = self._process(obs, info)
        info = dict(info)
        info["reliability"] = self.last_r.copy()
        return out, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        out = self._process(obs, info)
        info = dict(info)
        info["reliability"] = self.last_r.copy()
        return out, reward, terminated, truncated, info

    def observation(self, observation: np.ndarray) -> np.ndarray:
        return observation

    def _process(self, obs: np.ndarray, info: dict) -> np.ndarray:
        pos = info.get("obs_pos", self.core.pos)
        feats, mu = self.features.step(obs, pos)
        if self.mode == "S3":
            r = self.head.r_learned(feats)
        else:  # S2
            r = r_det_from_features(feats)
        self.last_r = r
        w = self.alpha + (1.0 - self.alpha) * r
        w = np.where(feats[:, 0] > 0.5, w, 0.0)
        gated = obs[:N_SENSOR] * w + mu * (1.0 - w)
        gated = np.clip(gated, SENSOR_PHYS_LO, SENSOR_PHYS_HI)
        out = obs.copy()
        out[:N_SENSOR] = gated
        return np.concatenate([out, r, feats[:, 0], feats[:, 1]])
