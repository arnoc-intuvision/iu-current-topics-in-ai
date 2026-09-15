"""SAC from Demonstrations for BESS dispatch (SACfD, arXiv:2504.04326).

Plain SAC struggles to learn useful storage-dispatch policies from scratch - the
cited paper reports SAC at zero-action level until demonstrations were added, after
which it peaked within ~7 episodes. Minibatches mix demo and online transitions
with a demo fraction rho(e) = max(0, 1 - e/E) linearly decaying over E steps
(their ablation: linear decay beats exponential decay and uniform merged sampling).
The rule providing demonstrations only needs to be slightly better than nothing.
"""

from __future__ import annotations

import numpy as np
import torch
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.type_aliases import ReplayBufferSamples


def _filled(buf: ReplayBuffer) -> int:
    return buf.buffer_size if buf.full else buf.pos


class DemoMixedReplayBuffer(ReplayBuffer):
    """Online replay buffer mixed with a frozen demonstration buffer.

    sample() draws round(rho * batch) demo transitions and the rest online, with
    rho = max(0, 1 - elapsed/demo_decay); elapsed is updated each training step
    via the companion ProgressCallback.
    """

    def __init__(self, *args, demo_buffer: ReplayBuffer, demo_decay: int = 100_000, **kwargs):
        super().__init__(*args, **kwargs)
        self.demo = demo_buffer
        self.demo_decay = int(demo_decay)
        self._elapsed = 0

    def set_elapsed(self, n: int) -> None:
        self._elapsed = int(n)

    def sample(self, batch_size, env=None):
        rho = max(0.0, 1.0 - self._elapsed / self.demo_decay)
        n_demo = _filled(self.demo)
        n_demo = min(int(round(rho * batch_size)), batch_size, n_demo)
        n_online = min(batch_size - n_demo, _filled(self))
        n_demo = batch_size - n_online
        if n_online <= 0:
            return self.demo.sample(batch_size, env=env)
        if n_demo <= 0:
            return super().sample(batch_size, env=env)
        demo_inds = np.random.randint(0, _filled(self.demo), size=n_demo)
        online_inds = np.random.randint(0, _filled(self), size=n_online)
        d = self.demo._get_samples(demo_inds, env=env)
        o = self._get_samples(online_inds, env=env)
        return ReplayBufferSamples(
            torch.cat([d.observations, o.observations]),
            torch.cat([d.actions, o.actions]),
            torch.cat([d.next_observations, o.next_observations]),
            torch.cat([d.dones, o.dones]),
            torch.cat([d.rewards, o.rewards]),
        )


class ProgressCallback(BaseCallback):
    """Feeds model.num_timesteps to a DemoMixedReplayBuffer each rollout step."""

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        buf = self.model.replay_buffer
        if hasattr(buf, "set_elapsed"):
            buf.set_elapsed(self.num_timesteps)
        return True


def collect_demos(env, policy, n_steps: int, seed: int = 0) -> ReplayBuffer:
    """Fill a replay buffer with transitions from a callable policy (e.g. B0).

    `policy(core_env, obs) -> action` receives the wrapped env and the current
    (scaled/augmented) observation. Truncation timeouts are stored so SAC does
    not bootstrap across episode boundaries.
    """
    buf = ReplayBuffer(
        n_steps, env.observation_space, env.action_space,
        handle_timeout_termination=True,
    )
    obs, _ = env.reset(seed=seed)
    for _ in range(n_steps):
        action = policy(env.unwrapped, obs)
        next_obs, reward, terminated, truncated, _ = env.step(action)
        buf.add(
            np.asarray(obs, dtype=np.float32).reshape(1, -1),         # obs
            np.asarray(next_obs, dtype=np.float32).reshape(1, -1),    # next_obs
            np.asarray(action, dtype=np.float32).reshape(1, -1),      # action
            np.array([reward], dtype=np.float32),
            np.array([terminated], dtype=bool),
            [{"TimeLimit.truncated": bool(truncated and not terminated)}],
        )
        obs = next_obs
        if terminated or truncated:
            obs, _ = env.reset()
    return buf
