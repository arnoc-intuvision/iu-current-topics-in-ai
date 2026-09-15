"""Microgrid Gymnasium environment: grid + solar PV + BESS + site load (test-site-01, 15-min SAST).

Reward and dynamics consume the TRUE canonical series only; observations expose the raw
sensor channels that the corruption wrapper (and later the reliability stage) operates on.
"""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

import gymnasium
import numpy as np
import pandas as pd

from ..data.channels import (
    CHG_HEAD_IDX,
    DIS_HEAD_IDX,
    EXPOSURE_IDX,
    OBS_CHANNELS,
    SOC_IDX,
)
from .battery import Battery
from .config import EnvConfig

OBS_LOW_RAW = np.array(
    [-1000, -1000, -1000, 0, 0, -40, -40, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, -1, -1, -1, -1],
    dtype=np.float64,
)
OBS_HIGH_RAW = np.array(
    [2000, 2000, 2000, 1500, 1500, 100, 100, 1500, 1000, 1000, 1000, 10, 1500, 1, 1000, 1000, 1, 1, 1, 1, 1, 1],
    dtype=np.float64,
)


class MicrogridEnv(gymnasium.Env):
    metadata: ClassVar[dict] = {"render_modes": []}

    def __init__(
        self,
        canonical_path: str | Path,
        config: EnvConfig | None = None,
        split: str = "train",
        scaler_path: str | Path | None = None,
    ):
        super().__init__()
        self.config = config or EnvConfig()
        self.split = split
        self.canonical_path = Path(canonical_path)
        df = pd.read_parquet(self.canonical_path)
        self.frame = df
        self.n_rows = len(df)
        self.dt_h = self.config.dt_h

        starts_path = self.canonical_path.parent / "episode_starts.parquet"
        starts = pd.read_parquet(starts_path)
        self.valid_starts = {
            s: starts.loc[starts["split"] == s, "start_idx"].to_numpy()
            for s in ["train", "val", "test"]
        }
        if len(self.valid_starts[split]) == 0:
            raise ValueError(f"no valid episode starts for split '{split}'")

        self._build_features()

        if scaler_path is None:
            scaler_path = self.canonical_path.parent / "scaler.npz"
        sc = np.load(scaler_path)
        self.obs_lo = sc["obs_lo"]
        self.obs_hi = sc["obs_hi"]
        self.obs_span = np.where(self.obs_hi - self.obs_lo > 0, self.obs_hi - self.obs_lo, 1.0)

        self.battery = Battery(self.config.battery)
        self.action_space = gymnasium.spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float64)
        self.observation_space = gymnasium.spaces.Box(
            OBS_LOW_RAW, OBS_HIGH_RAW, dtype=np.float64
        )
        self._pos = 0
        self._start = 0
        self._steps = 0
        self._pair_kwh = 0.0
        self._running_peak_kw = 0.0
        self._initial_peak_kw = 0.0
        self._episode_energy_cost = 0.0
        self._episode_demand_charge = 0.0

    def _build_features(self) -> None:
        df = self.frame
        n = self.n_rows
        feats = np.zeros((n, len(OBS_CHANNELS)))
        feats[:, 0] = df["pm1_kw"]
        feats[:, 1] = df["pm0_a_kw"]
        feats[:, 2] = df["pm0_b_kw"]
        feats[:, 3] = df["irr1_wm2"]
        feats[:, 4] = df["irr2_wm2"]
        feats[:, 5] = df["irr1_temp_c"]
        feats[:, 6] = df["irr2_temp_c"]
        feats[:, 7] = df["sat_gti_wm2"]
        feats[:, 8] = df["fcst_pv_t1h_kw"]
        feats[:, 9] = df["fcst_pv_t3h_kw"]
        feats[:, 10] = df["fcst_pv_t6h_kw"]
        feats[:, 11] = df["tariff_zar_kwh"]
        ts = df.index
        hod = ts.hour + ts.minute / 60.0
        feats[:, 16] = np.sin(2 * np.pi * hod / 24.0)
        feats[:, 17] = np.cos(2 * np.pi * hod / 24.0)
        dow = ts.dayofweek
        feats[:, 18] = np.sin(2 * np.pi * dow / 7.0)
        feats[:, 19] = np.cos(2 * np.pi * dow / 7.0)
        doy = ts.dayofyear
        feats[:, 20] = np.sin(2 * np.pi * doy / 365.25)
        feats[:, 21] = np.cos(2 * np.pi * doy / 365.25)
        self.features = feats
        self.load_kw = df["load_kw_true"].to_numpy()
        self.pv_kw = df["pv_kw_true"].to_numpy()
        self.tariff = df["tariff_zar_kwh"].to_numpy()
        self.slots = df["tou_slot"].to_numpy()
        self.mtd_peak = df["month_to_date_peak_kw"].to_numpy()

    @property
    def pos(self) -> int:
        return self._pos

    def raw_obs_at(self, pos: int) -> np.ndarray:
        return self.features[pos].copy()

    def _assemble_obs(self) -> np.ndarray:
        obs = self.features[self._pos].copy()
        charge_head, discharge_head = self.battery.headrooms(self.dt_h)
        obs[EXPOSURE_IDX[0]] = self._running_peak_kw
        obs[SOC_IDX[0]] = self.battery.soc
        obs[CHG_HEAD_IDX[0]] = charge_head
        obs[DIS_HEAD_IDX[0]] = discharge_head
        return obs

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        options = options or {}
        if "start_idx" in options:
            start = int(options["start_idx"])
        else:
            starts = self.valid_starts[self.split]
            start = int(starts[self.np_random.integers(len(starts))])
        self._start = start
        self._pos = start
        self._steps = 0
        soc_lo, soc_hi = self.config.initial_soc_range
        self.battery.reset(self.np_random.uniform(soc_lo, soc_hi))
        self._initial_peak_kw = float(self.mtd_peak[start])
        self._running_peak_kw = self._initial_peak_kw
        self._pair_kwh = 0.0
        self._episode_energy_cost = 0.0
        self._episode_demand_charge = 0.0
        info = self._info_dict(0.0, 0.0, action_applied_kw=0.0, clipped=False)
        info["obs_pos"] = self._pos
        return self._assemble_obs(), info

    def step(self, action):
        a = float(np.clip(np.asarray(action).reshape(-1)[0], -1.0, 1.0))
        desired_kw = a * self.config.battery.max_power_kw

        net_pre_kw = self.load_kw[self._pos] - self.pv_kw[self._pos]
        guard_clamped_kw = 0.0
        if self.config.charge_guard and desired_kw < 0.0:
            # keep the 30-min window import at/below the running peak: the charge
            # budget covers BOTH halves of the pair - the banked first-half import
            # (closing half) / the next half's baseline import (opening half) is
            # reserved so the baseline alone cannot breach the bound
            if self._pos % 2 == 0:  # opening half of a fresh pair
                banked_kwh = 0.0
                next_kw = (
                    self.load_kw[self._pos + 1] - self.pv_kw[self._pos + 1]
                    if self._pos + 1 < self.n_rows
                    else net_pre_kw
                )
                reserved_kwh = max(net_pre_kw, 0.0) * self.dt_h + max(next_kw, 0.0) * self.dt_h
            else:  # closing half: first-half import already banked in _pair_kwh
                banked_kwh = self._pair_kwh
                reserved_kwh = max(net_pre_kw, 0.0) * self.dt_h
            bound_kwh = max(
                0.0,
                (self._running_peak_kw - self.config.guard_margin_kw) * 0.5
                - banked_kwh
                - reserved_kwh,
            )
            charge_head, _ = self.battery.headrooms(self.dt_h)
            max_charge_kwh = min(charge_head * self.dt_h, bound_kwh)
            allowed_kw = max_charge_kwh / self.dt_h
            if -desired_kw > allowed_kw:
                guard_clamped_kw = -desired_kw - allowed_kw
                desired_kw = -allowed_kw

        soc_kwh_before = self.battery.soc * self.config.battery.capacity_kwh
        applied_kw, charge_kw, discharge_kw = self.battery.step(desired_kw, self.dt_h)
        clipped = abs(applied_kw - desired_kw) > 1e-9

        load_kw = self.load_kw[self._pos]
        pv_kw = self.pv_kw[self._pos]
        net_kw = load_kw - pv_kw + charge_kw - discharge_kw
        import_kw = max(net_kw, 0.0)
        export_kw = max(-net_kw, 0.0)
        import_kwh = import_kw * self.dt_h
        export_kwh = export_kw * self.dt_h

        rate = self.tariff[self._pos]
        energy_cost = import_kwh * rate - export_kwh * self.config.export_credit_zar_per_kwh

        demand_charge = 0.0
        window_kw = 0.0
        if self._pos % 2 == 0:
            self._pair_kwh = import_kwh
        else:
            self._pair_kwh += import_kwh
            window_kw = self._pair_kwh / 0.5
            if window_kw > self._running_peak_kw:
                demand_charge = (
                    window_kw - self._running_peak_kw
                ) * self.config.demand_charge_rate_zar_per_kw
                self._running_peak_kw = window_kw
            self._pair_kwh = 0.0

        self._episode_energy_cost += energy_cost
        self._episode_demand_charge += demand_charge
        reward = -(energy_cost + demand_charge) * self.config.reward_scale
        if self.config.shaping_lambda_zar_per_kwh > 0.0:
            soc_kwh_after = self.battery.soc * self.config.battery.capacity_kwh
            potential = self.config.shaping_lambda_zar_per_kwh
            reward += (
                self.config.shaping_gamma * soc_kwh_after - soc_kwh_before
            ) * potential * self.config.reward_scale

        info = self._info_dict(
            energy_cost,
            demand_charge,
            action_applied_kw=applied_kw,
            clipped=clipped,
            action_raw=a,
            import_kw=import_kw,
            export_kw=export_kw,
            import_kwh=import_kwh,
            export_kwh=export_kwh,
            window_demand_kw=window_kw,
            charge_kw=charge_kw,
            discharge_kw=discharge_kw,
            desired_kw=desired_kw,
            guard_clamped_kw=guard_clamped_kw,
        )
        info["obs_pos"] = self._pos + 1
        self._pos += 1
        self._steps += 1
        terminated = False
        truncated = False
        return self._assemble_obs(), reward, terminated, truncated, info

    def _info_dict(
        self,
        energy_cost: float,
        demand_charge: float,
        action_applied_kw: float,
        clipped: bool,
        action_raw: float = 0.0,
        import_kw: float = 0.0,
        export_kw: float = 0.0,
        import_kwh: float = 0.0,
        export_kwh: float = 0.0,
        window_demand_kw: float = 0.0,
        charge_kw: float = 0.0,
        discharge_kw: float = 0.0,
        desired_kw: float = 0.0,
        guard_clamped_kw: float = 0.0,
    ) -> dict:
        return {
            "timestamp": str(self.frame.index[self._pos]),
            "pos": int(self._pos),
            "step": int(self._steps),
            "split": self.split,
            "tou_slot": self.slots[self._pos],
            "tariff_zar_kwh": float(self.tariff[self._pos]),
            "load_kw": float(self.load_kw[self._pos]),
            "pv_kw": float(self.pv_kw[self._pos]),
            "import_kw": float(import_kw),
            "export_kw": float(export_kw),
            "import_kwh": float(import_kwh),
            "export_kwh": float(export_kwh),
            "charge_kw": float(charge_kw),
            "discharge_kw": float(discharge_kw),
            "soc": float(self.battery.soc),
            "action_raw": float(action_raw),
            "action_desired_kw": float(desired_kw),
            "action_applied_kw": float(action_applied_kw),
            "clipped": bool(clipped),
            "guard_clamped_kw": float(guard_clamped_kw),
            "energy_cost_zar": float(energy_cost),
            "demand_charge_zar": float(demand_charge),
            "window_demand_kw": float(window_demand_kw),
            "running_peak_kw": float(self._running_peak_kw),
            "initial_peak_kw": float(self._initial_peak_kw),
            "episode_energy_cost_zar": float(self._episode_energy_cost),
            "episode_demand_charge_zar": float(self._episode_demand_charge),
        }

    def scale_obs(self, obs: np.ndarray) -> np.ndarray:
        out = obs.copy()
        idx = list(range(16))
        out[idx] = (obs[idx] - self.obs_lo[idx]) / self.obs_span[idx] * 2.0 - 1.0
        return out


class ScaleObservation(gymnasium.ObservationWrapper):
    """Linear min-max scaling of channels 0..15; flags and time encodings pass through."""

    def __init__(self, env):
        super().__init__(env)
        self.env_ref = env.unwrapped
        dim = env.observation_space.shape[0]
        self.observation_space = gymnasium.spaces.Box(-50.0, 50.0, shape=(dim,), dtype=np.float64)

    def observation(self, obs: np.ndarray) -> np.ndarray:
        return self.env_ref.scale_obs(obs)


def make_env(
    canonical_path: str | Path,
    config: EnvConfig | None = None,
    split: str = "train",
    time_limit: bool = True,
) -> MicrogridEnv:
    env = MicrogridEnv(canonical_path, config, split)
    if time_limit:
        env = gymnasium.wrappers.TimeLimit(env, max_episode_steps=env.config.episode_steps)
    return env
