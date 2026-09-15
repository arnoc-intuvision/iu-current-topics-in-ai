import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

from sacbess.corruption.wrapper import CorruptionWrapper
from sacbess.data.channels import DATA_IDX, SENSOR_IDX, TIME_IDX, WEATHER_IDX
from sacbess.env.config import EnvConfig
from sacbess.env.microgrid_env import MicrogridEnv


def make_pair(processed_dir, fault, severity, split="train"):
    core = MicrogridEnv(processed_dir / "canonical.parquet", EnvConfig(), split)
    wrapped = CorruptionWrapper(
        MicrogridEnv(processed_dir / "canonical.parquet", EnvConfig(), split),
        fault=fault,
        severity=severity,
    )
    return core, wrapped


def run_lockstep(core, wrapped, start, seed, steps, action=0.0):
    c_obs, c_info = core.reset(seed=seed, options={"start_idx": start})
    w_obs, w_info = wrapped.reset(seed=seed, options={"start_idx": start})
    a = np.array([action])
    rows = []
    for _ in range(steps):
        c_obs, cr, _, _, c_info = core.step(a)
        w_obs, wr, _, _, w_info = wrapped.step(a)
        rows.append((c_obs, w_obs, cr, wr, c_info, w_info))
    return rows


def test_severity_zero_is_bitwise_passthrough(processed_dir):
    core, wrapped = make_pair(processed_dir, "F1", 0)
    start = int(core.valid_starts["train"][3])
    c_obs, _ = core.reset(seed=11, options={"start_idx": start})
    w_obs, w_info = wrapped.reset(seed=11, options={"start_idx": start})
    assert np.array_equal(c_obs, w_obs)
    assert not w_info["corruption_mask"].any()
    for _ in range(50):
        c_obs, *_ = core.step(np.zeros(1))
        w_obs, _, _, _, w_info = wrapped.step(np.zeros(1))
        assert np.array_equal(c_obs, w_obs)
        assert not w_info["corruption_mask"].any()


def test_f1_sentinel_matches_mask_and_spares_time_encodings(processed_dir):
    core, wrapped = make_pair(processed_dir, "F1", 3)
    start = int(core.valid_starts["train"][5])
    rows = run_lockstep(core, wrapped, start, seed=42, steps=672)
    any_mask = False
    for c_obs, w_obs, _, _, c_info, w_info in rows:
        mask = w_info["corruption_mask"]
        any_mask |= mask.any()
        for ch in SENSOR_IDX:
            if mask[ch]:
                assert w_obs[ch] == wrapped.sentinel
            else:
                assert w_obs[ch] == c_obs[ch]
        assert not mask[12:].any()
        assert np.array_equal(w_obs[TIME_IDX], c_obs[TIME_IDX])
    assert any_mask


def test_reward_runs_on_true_series_under_f4(processed_dir):
    core, wrapped = make_pair(processed_dir, "F4", 2)
    start = int(core.valid_starts["train"][8])
    rows = run_lockstep(core, wrapped, start, seed=9, steps=300)
    for _, _, cr, wr, _, _ in rows:
        assert cr == wr


def test_f2_stuck_runs_are_valued_zero(processed_dir):
    core = MicrogridEnv(processed_dir / "canonical.parquet", EnvConfig(), "train")
    found = False
    for seed in range(60):
        wrapped = CorruptionWrapper(
            MicrogridEnv(processed_dir / "canonical.parquet", EnvConfig(), "train"),
            fault="F2",
            severity=3,
        )
        start = int(core.valid_starts["train"][12])
        rows = run_lockstep(core, wrapped, start, seed=seed, steps=672)
        for _, w_obs, _, _, _, w_info in rows:
            mask = w_info["corruption_mask"]
            for ch in (0, 1, 2):
                if mask[ch]:
                    assert w_obs[ch] == 0.0
                    found = True
    assert found


def test_f5_applies_identical_drift_to_both_pm0_channels(processed_dir):
    core = MicrogridEnv(processed_dir / "canonical.parquet", EnvConfig(), "train")
    wrapped = CorruptionWrapper(
        MicrogridEnv(processed_dir / "canonical.parquet", EnvConfig(), "train"),
        fault="F5",
        severity=2,
    )
    start = int(core.valid_starts["train"][8])
    rows = run_lockstep(core, wrapped, start, seed=17, steps=200)
    for i, (c_obs, w_obs, _, _, c_info, w_info) in enumerate(rows):
        d = 0.05 * (i + 1) / (30 * 24 * 4)
        for ch in (1, 2):
            assert abs(w_obs[ch] - c_obs[ch] * (1 + d)) < 1e-9
        assert not w_info["corruption_mask"][0]


def test_f6a_shifts_whole_data_block_coherently(processed_dir):
    core, wrapped = make_pair(processed_dir, "F6a", 2)
    start = int(core.valid_starts["train"][8])
    c_obs, _ = core.reset(seed=4, options={"start_idx": start})
    w_obs, w_info = wrapped.reset(seed=4, options={"start_idx": start})
    shift = wrapped._shift
    assert shift != 0
    for _ in range(100):
        c_obs, *_ = core.step(np.zeros(1))
        w_obs, _, _, _, w_info = wrapped.step(np.zeros(1))
        pos = w_info["obs_pos"]
        src = max(pos - shift, 0)
        row = core.raw_obs_at(src)
        for ch in DATA_IDX:
            assert w_obs[ch] == row[ch]
        assert w_info["corruption_mask"][DATA_IDX].all()
        assert np.array_equal(w_obs[TIME_IDX], row[TIME_IDX])
        assert np.array_equal(w_obs[12:16], c_obs[12:16])


def test_f6b_shifts_only_weather_block(processed_dir):
    core, wrapped = make_pair(processed_dir, "F6b", 2)
    start = int(core.valid_starts["train"][8])
    c_obs, _ = core.reset(seed=4, options={"start_idx": start})
    w_obs, _ = wrapped.reset(seed=4, options={"start_idx": start})
    shift = wrapped._shift
    assert shift != 0
    for _ in range(100):
        c_obs, *_ = core.step(np.zeros(1))
        w_obs, _, _, _, w_info = wrapped.step(np.zeros(1))
        src = max(w_info["obs_pos"] - shift, 0)
        row = core.raw_obs_at(src)
        for ch in WEATHER_IDX:
            assert w_obs[ch] == row[ch]
        for ch in (0, 1, 2, 3, 4, 5, 6, 11):
            assert w_obs[ch] == c_obs[ch]


def test_mix_train_resolves_to_trained_faults(processed_dir):
    seen = set()
    for seed in range(30):
        wrapped = CorruptionWrapper(
            MicrogridEnv(processed_dir / "canonical.parquet", EnvConfig(), "train"),
            fault="MIX_TRAIN",
            severity=0,
        )
        wrapped.reset(seed=seed)
        seen.add(wrapped._fault)
        assert wrapped._fault in {"F1", "F2", "F3", "F4"}
        assert 1 <= wrapped._severity <= 3
    assert seen == {"F1", "F2", "F3", "F4"}
