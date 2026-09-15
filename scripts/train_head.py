"""Pretrain the S3 reliability head on corrupted training episodes.

Collects (stage-1 features, ground-truth mask) pairs from the CorruptionWrapper's
MIX_TRAIN stream (F1-F4 only; F5/F6 held out), trains the shared MLP with
class-weighted BCE, freezes to runs/protocol/reliability_head.pt.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--canonical", type=Path, default=ROOT / "data" / "processed" / "canonical.parquet")
    ap.add_argument("--env-config", type=Path, default=ROOT / "configs" / "env.yaml")
    ap.add_argument("--episodes", type=int, default=24)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--out", type=Path, default=ROOT / "runs" / "protocol" / "reliability_head.pt")
    args = ap.parse_args()

    import numpy as np
    import yaml

    from sacbess.corruption.wrapper import CorruptionWrapper
    from sacbess.data.splits import build_splits, row_split_mask
    from sacbess.env.config import EnvConfig
    from sacbess.env.microgrid_env import MicrogridEnv
    from sacbess.reliability.features import N_SENSOR, ReliabilityFeatures, SitePriors
    from sacbess.reliability.learned_head import train_head

    with open(args.env_config) as f:
        config = EnvConfig.from_dict(yaml.safe_load(f))

    core = MicrogridEnv(args.canonical, config, "train")
    weeks = build_splits(core.frame, embargo_steps=96, seed=0)
    train_mask = row_split_mask(core.frame, weeks, "train")
    priors = SitePriors.fit(core.frame, train_mask)
    starts = core.valid_starts["train"]

    rng = np.random.default_rng(args.seed)
    sel = rng.choice(len(starts), size=min(args.episodes, len(starts)), replace=False)
    feats_all, mask_all = [], []
    n_fault_eps = 0
    for k, i in enumerate(sel):
        fault = "MIX_TRAIN" if rng.random() < 0.6 else "none"
        import gymnasium

        env = gymnasium.wrappers.TimeLimit(
            CorruptionWrapper(
                MicrogridEnv(args.canonical, config, "train"),
                fault=fault,
                severity=0,
            ),
            max_episode_steps=config.episode_steps,
        )
        start = int(starts[i])
        seed = int(rng.integers(1 << 31))
        obs, info = env.reset(seed=seed, options={"start_idx": start})
        stream = ReliabilityFeatures(priors)
        stream.reset()
        ep_f, ep_m = [], []
        done = False
        while not done:
            f, _ = stream.step(obs, info["obs_pos"])
            m = info["corruption_mask"][:N_SENSOR]
            ep_f.append(f)
            ep_m.append(m)
            obs, _, term, trunc, info = env.step(np.zeros(1))
            done = term or trunc
        feats_all.append(np.stack(ep_f))
        mask_all.append(np.stack(ep_m))
        if fault != "none":
            n_fault_eps += 1
        print(f"episode {k + 1}/{len(sel)}: fault={fault} resolved={info['fault']}/s{info['severity']} mask_rate={np.stack(ep_m).mean():.3f}")

    feats = np.stack(feats_all)
    masks = np.stack(mask_all).astype(np.float32)
    stats = train_head(feats, masks, args.out, epochs=args.epochs, seed=args.seed, hidden=args.hidden)
    print(f"head trained: {stats} | fault episodes: {n_fault_eps}/{len(sel)}")


if __name__ == "__main__":
    main()
