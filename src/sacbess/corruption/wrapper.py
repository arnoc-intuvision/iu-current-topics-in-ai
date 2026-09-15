"""CorruptionWrapper: F1-F6 x severity 0-3 on raw observation channels, ground-truth mask via info.

Sits directly above MicrogridEnv (physical units), below ScaleObservation:
    SB3 -> TimeLimit -> ScaleObservation -> CorruptionWrapper -> MicrogridEnv
Reward and dynamics are untouched: the wrapper perturbs observations only.
"""

from __future__ import annotations

import gymnasium
import numpy as np

from ..data.channels import (
    DATA_IDX,
    IRR_IDX,
    METER_IDX,
    OBS_CHANNELS,
    SENSOR_IDX,
    WEATHER_IDX,
)
from ..env.microgrid_env import MicrogridEnv
from .faults import SEVERITY_TABLES, STEPS_PER_H, TRAIN_FAULTS
from .markov import MarkovChannel

SEED_STREAM_OFFSET = 7919
OU_BIAS_EPS = 1e-3
F5_DRIFT_STEPS = 30 * 24 * STEPS_PER_H


class CorruptionWrapper(gymnasium.ObservationWrapper):
    def __init__(
        self,
        env: MicrogridEnv,
        fault: str | None = "F1",
        severity: int = 2,
        sentinel: float = -999.0,
    ):
        super().__init__(env)
        self.core: MicrogridEnv = env.unwrapped
        self.fault_spec = fault
        self.severity_spec = int(severity)
        self.sentinel = sentinel
        self._rng = np.random.default_rng()
        self._fault = "none"
        self._severity = 0
        self._chains: dict[int, MarkovChannel] = {}
        self._ou_bias: dict[int, float] = {}
        self._f4 = None
        self._f5 = None
        self._f5_steps = 0
        self._shift = 0
        self.observation_space = gymnasium.spaces.Box(
            -1.1e6, 1.1e6, shape=(len(OBS_CHANNELS),), dtype=np.float64
        )

    # ------------------------------------------------------------------ reset/step
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            self._rng = np.random.default_rng(seed + SEED_STREAM_OFFSET)
        obs, info = self.env.reset(seed=seed, options=options)
        self._resolve_fault()
        self._init_state()
        obs, mask = self._apply_fault(obs, info.get("obs_pos", self.core.pos))
        info = dict(info)
        info["corruption_mask"] = mask
        info["fault"] = self._fault
        info["severity"] = self._severity
        info["faulted_channels"] = [OBS_CHANNELS[i] for i in np.flatnonzero(mask)]
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        obs_pos = info.get("obs_pos", self.core.pos)
        obs, mask = self._apply_fault(obs, obs_pos)
        info = dict(info)
        info["corruption_mask"] = mask
        info["fault"] = self._fault
        info["severity"] = self._severity
        info["faulted_channels"] = [OBS_CHANNELS[i] for i in np.flatnonzero(mask)]
        return obs, reward, terminated, truncated, info

    # ------------------------------------------------------------------ internals
    def observation(self, observation: np.ndarray) -> np.ndarray:
        return observation

    def _resolve_fault(self) -> None:
        if self.fault_spec == "MIX_TRAIN":
            self._fault = TRAIN_FAULTS[int(self._rng.integers(len(TRAIN_FAULTS)))]
            self._severity = int(self._rng.integers(1, 4))
        elif self.fault_spec in (None, "none") or self.severity_spec == 0:
            self._fault, self._severity = "none", 0
        elif self.fault_spec == "F6":
            self._fault = "F6a"
            self._severity = self.severity_spec
        else:
            self._fault = self.fault_spec
            self._severity = self.severity_spec

    def _init_state(self) -> None:
        self._chains = {}
        self._ou_bias = {}
        self._f5_steps = 0
        self._shift = 0
        if self._fault in ("none",):
            return
        params = SEVERITY_TABLES.get(self._fault.rstrip("ab") if self._fault.startswith("F6") else self._fault)
        if params is None:
            raise ValueError(f"unknown fault {self._fault}")
        p = params[self._severity] if self._severity > 0 else None
        if self._fault == "F1" and p:
            for ch in SENSOR_IDX:
                self._chains[ch] = MarkovChannel(
                    p.p_onset, p.med_h * STEPS_PER_H, p.sig_h, p.cap_h * STEPS_PER_H
                )
        elif self._fault == "F2" and p:
            for ch in METER_IDX:
                self._chains[ch] = MarkovChannel(
                    p.p_onset, p.med_h * STEPS_PER_H, p.sig_h, p.cap_h * STEPS_PER_H
                )
        elif self._fault == "F4" and p:
            self._f4 = p
            for ch in IRR_IDX:
                self._ou_bias[ch] = float(self._rng.normal(0.0, p.sigma_stat))
        elif self._fault == "F6a" and p or self._fault == "F6b" and p:
            sign = 1 if self._rng.random() < 0.5 else -1
            self._shift = sign * p.shift_steps

    def _apply_fault(self, obs: np.ndarray, obs_pos: int) -> tuple[np.ndarray, np.ndarray]:
        mask = np.zeros(len(OBS_CHANNELS), dtype=bool)
        if self._fault == "none" or self._severity == 0:
            return obs, mask
        obs = obs.copy()

        if self._fault == "F1":
            p = SEVERITY_TABLES["F1"][self._severity]
            for ch, chain in self._chains.items():
                if chain.step(self._rng):
                    obs[ch] = self.sentinel
                    mask[ch] = True

        elif self._fault == "F2":
            for ch, chain in self._chains.items():
                if chain.step(self._rng):
                    obs[ch] = 0.0
                    mask[ch] = True

        elif self._fault == "F3":
            p = SEVERITY_TABLES["F3"][self._severity]
            for ch in SENSOR_IDX:
                if self._rng.random() < p.p_spike:
                    if self._rng.random() < p.p_sentinel:
                        obs[ch] = p.sentinel_value
                    else:
                        mag = 10 ** self._rng.uniform(np.log10(p.mag_lo), np.log10(p.mag_hi))
                        obs[ch] = mag * (1 if self._rng.random() < 0.5 else -1)
                    mask[ch] = True

        elif self._fault == "F4":
            p = self._f4
            for ch in IRR_IDX:
                b = self._ou_bias[ch]
                b = b + p.theta * (0.0 - b) + p.sigma_stat * np.sqrt(2 * p.theta) * self._rng.normal()
                self._ou_bias[ch] = b
                obs[ch] = obs[ch] * (1.0 + b)
                if abs(b) > OU_BIAS_EPS:
                    mask[ch] = True

        elif self._fault == "F5":
            p = SEVERITY_TABLES["F5"][self._severity]
            d = p.drift_frac_30d * self._f5_steps / F5_DRIFT_STEPS
            for ch in (1, 2):
                obs[ch] = obs[ch] * (1.0 + d)
                if d > OU_BIAS_EPS:
                    mask[ch] = True
            self._f5_steps += 1

        elif self._fault == "F6a":
            src = max(obs_pos - self._shift, 0)
            row = self.core.raw_obs_at(src)
            for ch in DATA_IDX:
                mask[ch] = True
                obs[ch] = row[ch]

        elif self._fault == "F6b":
            src = max(obs_pos - self._shift, 0)
            row = self.core.raw_obs_at(src)
            for ch in WEATHER_IDX:
                mask[ch] = True
                obs[ch] = row[ch]

        return obs, mask
