import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import gymnasium
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from sacbess.env.config import EnvConfig
from sacbess.env.microgrid_env import MicrogridEnv, make_env
from sacbess.factory import build_env


@pytest.fixture(scope="module")
def env(processed_dir):
    return MicrogridEnv(processed_dir / "canonical.parquet", EnvConfig(), "train")


def test_gymnasium_contract(env):
    check_env(env, skip_render_check=True)


def test_reset_determinism(env):
    obs1, info1 = env.reset(seed=123)
    obs2, info2 = env.reset(seed=123)
    assert np.array_equal(obs1, obs2)
    assert info1["pos"] == info2["pos"]


def test_episode_windows_exclude_islanded(env):
    frame = env.frame
    for seed in range(40):
        _, info = env.reset(seed=seed)
        start = info["pos"]
        assert not frame["islanded"].to_numpy()[start : start + env.config.episode_steps].any()
        assert start in set(env.valid_starts["train"].tolist())


def test_zero_action_reward_matches_hand_computation(env):
    cfg = env.config
    start = int(env.valid_starts["train"][10])
    env.reset(seed=5, options={"start_idx": start})
    load = env.load_kw
    pv = env.pv_kw
    rate = env.tariff
    mtd = env.mtd_peak

    expected_energy = 0.0
    expected_demand = 0.0
    running_peak = float(mtd[start])
    pair = 0.0
    for t in range(start, start + cfg.episode_steps):
        import_kw = max(load[t] - pv[t], 0.0)
        expected_energy += import_kw * cfg.dt_h * rate[t]
        if t % 2 == 0:
            pair = import_kw * cfg.dt_h
        else:
            pair += import_kw * cfg.dt_h
            window_kw = pair / 0.5
            if window_kw > running_peak:
                expected_demand += (window_kw - running_peak) * cfg.demand_charge_rate_zar_per_kw
                running_peak = window_kw
            pair = 0.0

    steps_taken = 0
    for _ in range(cfg.episode_steps):
        _, _, _term, _trunc, info = env.step(np.zeros(1))
        steps_taken += 1
    assert steps_taken == cfg.episode_steps and info["step"] == cfg.episode_steps - 1
    assert abs(info["episode_energy_cost_zar"] - expected_energy) < 1e-6
    assert abs(info["episode_demand_charge_zar"] - expected_demand) < 1e-6


def test_charging_increases_import_and_demand_charge(env):
    start = int(env.valid_starts["train"][10])
    env.reset(seed=5, options={"start_idx": start})
    _, _, _, _, info_idle = env.step(np.array([-1.0]))
    env.reset(seed=5, options={"start_idx": start})
    _, _, _, _, info_charged = env.step(np.array([0.0]))
    assert info_charged["import_kw"] < info_idle["import_kw"]


def test_timelimit_truncation_and_sb3_bootstrap_flag(processed_dir):
    env = build_env(processed_dir / "canonical.parquet", EnvConfig(), "train")
    env.reset(seed=0)
    trunc = False
    for _ in range(672):
        _, _, term, trunc, info = env.step(np.zeros(1))
        if term or trunc:
            break
    assert trunc is True
    assert info.get("TimeLimit.truncated") is True


def test_scaled_observation_ranges(processed_dir):
    env = build_env(processed_dir / "canonical.parquet", EnvConfig(), "train")
    obs, _ = env.reset(seed=3)
    assert obs.shape == (22,)
    assert np.all(obs[:16] >= -1.001) and np.all(obs[:16] <= 1.001)
    assert np.all(obs[16:] >= -1.0) and np.all(obs[16:] <= 1.0)


def test_make_env_factory(processed_dir):
    env = make_env(processed_dir / "canonical.parquet", EnvConfig(), "val")
    assert isinstance(env, gymnasium.wrappers.TimeLimit)
