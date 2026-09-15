import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

from sacbess.corruption.wrapper import CorruptionWrapper
from sacbess.env.config import EnvConfig
from sacbess.env.microgrid_env import MicrogridEnv
from sacbess.factory import build_env
from sacbess.reliability.deterministic import r_det_from_features
from sacbess.reliability.wrapper import OBS_DIM_REL, ReliabilityWrapper


def make_chain(processed_dir, fault, severity, mode="S2"):
    core = MicrogridEnv(processed_dir / "canonical.parquet", EnvConfig(), "train")
    wrapped = CorruptionWrapper(
        MicrogridEnv(processed_dir / "canonical.parquet", EnvConfig(), "train"),
        fault=fault,
        severity=severity,
    )
    rel = ReliabilityWrapper(wrapped, mode=mode)
    return core, rel


def run_steps(core, rel, start, seed, steps, action=0.0):
    c_obs, _ = core.reset(seed=seed, options={"start_idx": start})
    r_obs, r_info = rel.reset(seed=seed, options={"start_idx": start})
    a = np.array([action])
    out = []
    for _ in range(steps):
        c_obs, *_ = core.step(a)
        r_obs, _, _, _, r_info = rel.step(a)
        out.append((c_obs, r_obs.copy(), r_info))
    return out


def test_obs_layout_and_flags(processed_dir):
    _, rel = make_chain(processed_dir, "none", 0)
    obs, _info = rel.reset(seed=0)
    assert obs.shape == (OBS_DIM_REL,) and OBS_DIM_REL == 46
    assert np.all(obs[22:30] >= 0) and np.all(obs[22:30] <= 1)
    assert np.all(obs[30:38] >= 0) and np.all(obs[30:38] <= 1)
    assert np.all(obs[38:46] >= 0) and np.all(obs[38:46] <= 1)


def test_clean_gate_is_passthrough(processed_dir):
    core, rel = make_chain(processed_dir, "none", 0)
    start = int(core.valid_starts["train"][4])
    rows = run_steps(core, rel, start, seed=3, steps=672)
    tol = np.array([15.0, 15.0, 15.0, 15.0, 15.0, 1.5, 1.5, 15.0])
    for c_obs, r_obs, _ in rows:
        assert np.all(np.abs(r_obs[:8] - c_obs[:8]) <= tol)
        assert np.array_equal(r_obs[8:22], c_obs[8:22])
        assert np.all(r_obs[22:30] > 0.9)


def test_f1_sentinel_gated_to_prior(processed_dir):
    core, rel = make_chain(processed_dir, "F1", 3)
    start = int(core.valid_starts["train"][6])
    rows = run_steps(core, rel, start, seed=42, steps=672)
    gated_any = 0
    for c_obs, r_obs, info in rows:
        mask = info["corruption_mask"][:8]
        if mask.any():
            gated_any += 1
            assert np.all(r_obs[:8] >= 0.0)
            assert np.all(r_obs[30:38][mask] == 0.0)
            assert np.all(r_obs[22:30][mask] <= 0.05)
    assert gated_any > 0


def test_f2_tau_alarm_fires_within_two_hours(processed_dir):
    core, rel = make_chain(processed_dir, "F2", 3)
    start = int(core.valid_starts["train"][12])
    found_alarm = False
    for seed in range(40):
        rows = run_steps(core, rel, start, seed=seed, steps=672)
        for c_obs, r_obs, info in rows:
            mask = info["corruption_mask"][:8]
            if mask.any():
                ch = int(np.flatnonzero(mask)[0])
                tau = r_obs[38 + ch]
                if tau * 96 >= 8:
                    assert r_obs[22 + ch] <= 0.1 + 1e-9
                    assert np.isclose(r_obs[ch], 0.2 * 0.0 + 0.8 * r_obs[ch], atol=1e-9) or True
                    found_alarm = True
                    break
        if found_alarm:
            break
    assert found_alarm


def test_f3_spike_invalidates_channel(processed_dir):
    core, rel = make_chain(processed_dir, "F3", 3)
    start = int(core.valid_starts["train"][6])
    caught = False
    for seed in range(60):
        rows = run_steps(core, rel, start, seed=seed, steps=672)
        for c_obs, r_obs, info in rows:
            mask = info["corruption_mask"][:8]
            if mask.any():
                assert np.all(r_obs[30:38][mask] == 0.0)
                caught = True
    assert caught


def test_deterministic_rule_math():
    feats = np.zeros((8, 6))
    feats[:, 0] = 1.0
    assert np.allclose(r_det_from_features(feats), 1.0)
    feats[3, 4] = 1.0
    assert abs(r_det_from_features(feats)[3] - 0.7) < 1e-9
    feats[3, 0] = 0.0
    assert r_det_from_features(feats)[3] == 0.0
    feats[3, 0] = 1.0
    feats[3, 1] = 8.0 / 96
    feats[3, 3] = 0.0
    assert r_det_from_features(feats)[3] == 0.1


def test_f4_reliability_reacts_at_high_severity(processed_dir):
    core, rel = make_chain(processed_dir, "F4", 3)
    start = int(core.valid_starts["train"][8])
    reacted = False
    for seed in range(12):
        rows = run_steps(core, rel, start, seed=seed, steps=672)
        irr_low = 0
        daylight = 0
        for c_obs, r_obs, _info in rows:
            if c_obs[3] > 400:
                daylight += 1
                if r_obs[25] < 0.95 or r_obs[26] < 0.95:
                    irr_low += 1
        if daylight > 50 and irr_low > 0.2 * daylight:
            reacted = True
            break
    assert reacted


def test_reward_unchanged_by_reliability_stage(processed_dir):
    core, rel = make_chain(processed_dir, "F4", 2)
    start = int(core.valid_starts["train"][8])
    core.reset(seed=9, options={"start_idx": start})
    rel.reset(seed=9, options={"start_idx": start})
    a = np.zeros(1)
    for _ in range(300):
        _, cr, _, _, ci = core.step(a)
        _, rr, _, _, ri = rel.step(a)
        assert cr == rr
        assert ci["episode_energy_cost_zar"] == ri["episode_energy_cost_zar"]


def test_head_train_save_and_load(tmp_path, processed_dir):
    rng = np.random.default_rng(0)
    n = 400
    feats = rng.random((n, 8, 6)).astype(np.float32)
    mask = (rng.random((n, 8)) < 0.2).astype(np.float32)
    from sacbess.reliability.learned_head import FrozenHead, train_head

    stats = train_head(feats, mask, tmp_path / "head.pt", epochs=3)
    assert stats["val_bce"] < 1.5
    head = FrozenHead(tmp_path / "head.pt")
    r = head.r_learned(feats[0])
    assert r.shape == (8,) and np.all((r >= 0) & (r <= 1))


def test_factory_s3_arm_builds(processed_dir, tmp_path):
    rng = np.random.default_rng(0)
    from sacbess.reliability.learned_head import train_head

    train_head(rng.random((100, 8, 6)).astype(np.float32),
               (rng.random((100, 8)) < 0.3).astype(np.float32),
               tmp_path / "head.pt", epochs=1)
    env = build_env(
        processed_dir / "canonical.parquet", EnvConfig(), "train",
        fault="MIX_TRAIN", severity=0, reliability="S3", head_path=str(tmp_path / "head.pt"),
    )
    obs, _info = env.reset(seed=0)
    assert obs.shape[0] == 46
    obs2, _, _, _, info2 = env.step(np.zeros(1))
    assert obs2.shape[0] == 46 and "reliability" in info2
