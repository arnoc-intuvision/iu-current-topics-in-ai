"""Train a SAC agent (S0 clean / S1 raw-corrupt / S2 deterministic / S3 learned) on the microgrid env.

Usage:
  python scripts/train_sac.py --arm S0 --steps 50000 --seed 0 --out runs/protocol
  python scripts/train_sac.py --arm S3 --steps 50000 --seed 0 --head runs/protocol/reliability_head.pt
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["S0", "S1", "S2", "S3"], default="S0")
    ap.add_argument("--steps", type=int, default=50_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=ROOT / "runs" / "protocol")
    ap.add_argument("--canonical", type=Path, default=ROOT / "data" / "processed" / "canonical.parquet")
    ap.add_argument("--env-config", type=Path, default=ROOT / "configs" / "env.yaml")
    ap.add_argument("--head", type=Path, default=ROOT / "runs" / "protocol" / "reliability_head.pt")
    ap.add_argument("--learning-rate", type=float, default=3e-4)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--ent-coef", default="auto", help="'auto' or fixed entropy coefficient")
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--hidden", default="256,256",
                    help="MLP hidden sizes, comma-separated (applies to actor and critics)")
    ap.add_argument("--dispatch-features", action="store_true",
                    help="append forward-tariff/temporal dispatch features to the obs")
    ap.add_argument("--demo-steps", type=int, default=0,
                    help="collect N B0 demonstration transitions (SACfD, arXiv:2504.04326)")
    ap.add_argument("--demo-decay", type=int, default=100_000,
                    help="linear decay horizon of the demo mix ratio")
    ap.add_argument("--init-from", default=None,
                    help="path to a saved SAC zip: load it and fine-tune on this run's env")
    ap.add_argument("--save-replay-buffer", action="store_true",
                    help="save the replay buffer to {out}/buffer_{tag}.npz at the end")
    ap.add_argument("--load-buffer", default=None,
                    help="path to a saved replay buffer: warm-start fine-tuning with it")
    ap.add_argument("--checkpoint-every", type=int, default=0,
                    help="save a policy checkpoint every N training steps (no buffer)")
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")

    import stable_baselines3 as sb3
    import yaml
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.monitor import Monitor

    from sacbess.env.config import EnvConfig
    from sacbess.factory import build_env

    class EpisodeRewardLogger(BaseCallback):
        """CSV log per episode (672 steps = 1 week at TimeLimit): training reward
        (true economics + guard + shaping) and the env's true cost accounting."""

        def __init__(self, out_path):
            super().__init__()
            self.out_path = out_path
            self.episode = 0

        def _on_training_start(self):
            import csv

            self.f = open(self.out_path, "w", newline="")
            self.w = csv.writer(self.f)
            self.w.writerow(["episode", "timestep", "reward_sum", "reward_mean",
                             "steps", "energy_cost_zar", "demand_charge_zar"])

        def _on_step(self):
            for info in self.locals["infos"]:
                if "episode" in info:
                    self.episode += 1
                    self.w.writerow([
                        self.episode, self.num_timesteps,
                        round(info["episode"]["r"], 4),
                        round(info["episode"]["r"] / info["episode"]["l"], 6),
                        info["episode"]["l"],
                        round(info.get("episode_energy_cost_zar", float("nan")), 2),
                        round(info.get("episode_demand_charge_zar", float("nan")), 2),
                    ])
                    self.f.flush()
            return True

    class CheckpointEvery(BaseCallback):
        """Policy-only checkpoint every `every` steps of THIS learn() call."""

        def __init__(self, out_dir, tag, every: int):
            super().__init__()
            self.out_dir = out_dir
            self.tag = tag
            self.every = every

        def _on_step(self):
            if self.n_calls % self.every == 0:
                self.model.save(self.out_dir / f"ckpt_{self.tag}_s{self.n_calls:06d}.zip")
            return True

    with open(args.env_config) as f:
        config = EnvConfig.from_dict(yaml.safe_load(f))

    fault = None if args.arm == "S0" else "MIX_TRAIN"
    reliability = {"S2": "S2", "S3": "S3"}.get(args.arm)
    head = str(args.head) if args.arm == "S3" else None
    env = build_env(
        args.canonical, config, split="train",
        fault=fault, severity=0,
        reliability=reliability, head_path=head,
        dispatch_features=args.dispatch_features,
    )
    # Monitor must sit outside TimeLimit so truncation ends are recorded per episode
    args.out.mkdir(parents=True, exist_ok=True)
    tag = args.tag or f"{args.arm}_seed{args.seed}"
    episode_log = args.out / f"episodes_{tag}.csv"
    env = Monitor(env)
    episode_logger = EpisodeRewardLogger(episode_log)

    buffer_class = None
    buffer_kwargs = {}
    callbacks = []
    if args.demo_steps > 0:
        from sacbess.env.baselines import B0Policy
        from sacbess.rl.demo_buffer import DemoMixedReplayBuffer, ProgressCallback, collect_demos

        b0 = B0Policy(config.battery.max_power_kw)
        demo_env = build_env(
            args.canonical, config, split="train",
            fault=fault, severity=0,
            reliability=reliability, head_path=head,
            dispatch_features=args.dispatch_features,
        )
        demo_buf = collect_demos(
            demo_env, lambda core, obs: b0.action(core), args.demo_steps, seed=args.seed
        )
        from sacbess.rl.demo_buffer import _filled

        n_demo = _filled(demo_buf)
        buffer_class = DemoMixedReplayBuffer
        buffer_kwargs = {"demo_buffer": demo_buf, "demo_decay": args.demo_decay}
        callbacks = [ProgressCallback()]
        print(f"collected {n_demo} B0 demo transitions (decay {args.demo_decay})")
    callbacks.append(episode_logger)
    if args.checkpoint_every > 0:
        callbacks.append(CheckpointEvery(args.out, tag, args.checkpoint_every))

    ent_coef: float | str = args.ent_coef
    try:
        ent_coef = float(ent_coef)
    except ValueError:
        pass  # keep "auto"
    sac_kwargs = dict(
        seed=args.seed,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        ent_coef=ent_coef,
        gamma=args.gamma,
        buffer_size=300_000,
        learning_starts=2_000,
        train_freq=1,
        gradient_steps=1,
        policy_kwargs={"net_arch": [int(x) for x in args.hidden.split(",")]},
        verbose=0,
    )
    if buffer_class is not None:
        sac_kwargs["replay_buffer_class"] = buffer_class
        sac_kwargs["replay_buffer_kwargs"] = buffer_kwargs
    if args.init_from:
        # fine-tune a pretrained model on this run's env; optionally keep its replay
        # buffer so stage-2 minibatches sample a mix of clean and corrupted transitions
        model = sb3.SAC.load(args.init_from, env=env, seed=args.seed, device="cpu")
        if args.load_buffer:
            model.load_replay_buffer(args.load_buffer)
            print(f"fine-tuning from {args.init_from} with warm replay buffer "
                  f"({model.replay_buffer.pos if not model.replay_buffer.full else model.replay_buffer.buffer_size} "
                  f"transitions loaded)")
        else:
            print(f"fine-tuning from {args.init_from} for {args.steps} steps "
                  f"(fresh buffer, ent_coef={model.ent_coef})")
    else:
        model = sb3.SAC("MlpPolicy", env, **sac_kwargs)
    model.learn(total_timesteps=args.steps, callback=callbacks, log_interval=50)

    if args.save_replay_buffer:
        buf_path = args.out / f"buffer_{tag}.npz"
        model.save_replay_buffer(str(buf_path))
        print(f"saved replay buffer: {buf_path}")

    model_path = args.out / f"sac_{tag}.zip"
    model.save(model_path)
    n_params = sum(p.numel() for p in model.policy.parameters())
    print(f"saved {model_path} | policy params: {n_params}")
    print(f"episode log: {episode_log}")


if __name__ == "__main__":
    main()
