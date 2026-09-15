"""Env assembly: TruncationBootstrap -> TimeLimit -> ScaleObservation -> [Reliability] -> [Corruption] -> core."""

from __future__ import annotations

from pathlib import Path

import gymnasium

from .corruption.wrapper import CorruptionWrapper
from .env.config import EnvConfig
from .env.microgrid_env import MicrogridEnv, ScaleObservation


class TruncationBootstrap(gymnasium.Wrapper):
    """Restores info["TimeLimit.truncated"]: gymnasium >=1.3 stopped emitting it,
    but SB3's replay buffer still keys truncation bootstrapping on it."""

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if truncated and not terminated and "TimeLimit.truncated" not in info:
            info = dict(info)
            info["TimeLimit.truncated"] = True
        return obs, reward, terminated, truncated, info


def build_env(
    canonical_path: str | Path,
    config: EnvConfig | None = None,
    split: str = "train",
    fault: str | None = None,
    severity: int = 0,
    reliability: str | None = None,
    head_path: str | Path | None = None,
    scale: bool = True,
    time_limit: bool = True,
    dispatch_features: bool = False,
) -> gymnasium.Env:
    core = MicrogridEnv(canonical_path, config, split)
    env = core
    if fault is not None and fault != "none":
        env = CorruptionWrapper(env, fault=fault, severity=severity)
    if reliability is not None and reliability != "none":
        from .reliability.wrapper import ReliabilityWrapper

        env = ReliabilityWrapper(env, mode=reliability, head_path=head_path)
    if scale:
        env = ScaleObservation(env)
    if dispatch_features:
        from .env.dispatch_features import DispatchFeatures

        env = DispatchFeatures(env)
    if time_limit:
        env = TruncationBootstrap(
            gymnasium.wrappers.TimeLimit(env, max_episode_steps=core.config.episode_steps)
        )
    return env
