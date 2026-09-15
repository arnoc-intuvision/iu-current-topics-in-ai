"""Evaluate a set of SAC models through the protocol's 22-condition fault matrix,
then compare retained performance against protocol arms (paired per seed).

Works for any model dir whose zips follow sac_{label}_seed{k}.zip and whose agents
consume the raw (no reliability stage) observation + dispatch features.

Usage:
  python scripts/analysis/eval_arm.py --dir <model_dir> --label <label> --episodes 2
  python scripts/analysis/eval_arm.py --dir <model_dir> --label <label> --compare S1 S0
"""

from __future__ import annotations

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
import yaml

CANONICAL = ROOT / "data" / "processed" / "canonical.parquet"
CONFIG = ROOT / "configs" / "env.yaml"
CONDITIONS = [("none", 0)] + [
    (f, s) for f in ["F1", "F2", "F3", "F4", "F5", "F6a", "F6b"] for s in [1, 2, 3]
]


def eval_model(model_path: str, label: str, episodes: int, eval_seed: int,
               split: str = "test", reliability: str | None = None,
               head_path: str | None = None) -> list[dict]:
    from stable_baselines3 import SAC

    from sacbess.env.config import EnvConfig
    from sacbess.factory import build_env
    from sacbess.env.microgrid_env import MicrogridEnv

    with open(CONFIG) as f:
        config = EnvConfig.from_dict(yaml.safe_load(f))
    model = SAC.load(model_path, device="cpu")
    import re

    seed = int(re.search(r"seed(\d+)", str(model_path)).group(1))

    probe = MicrogridEnv(CANONICAL, config, split)
    p_max = config.battery.max_power_kw
    rng = np.random.default_rng(eval_seed)
    starts = rng.choice(probe.valid_starts[split], size=episodes, replace=True)
    seeds = [eval_seed + 1000 + i for i in range(episodes)]
    del probe

    def run(fault, severity, start, s):
        env = build_env(
            CANONICAL, config, split,
            fault=(None if fault in (None, "none") else fault),
            severity=severity, reliability=reliability, head_path=head_path,
            dispatch_features=True,
        )
        obs, info = env.reset(seed=s, options={"start_idx": int(start)})
        actions, steps, clip_any, sat = [], 0, 0, 0
        done = False
        while not done:
            a, _ = model.predict(obs, deterministic=True)
            obs, _, term, trunc, info = env.step(a)
            actions.append(float(np.asarray(a).reshape(-1)[0]))
            steps += 1
            clip_any += int(info["clipped"])
            sat += int(abs(info["action_raw"]) >= 0.99)
            done = term or trunc
        return {
            "cost": info["episode_energy_cost_zar"] + info["episode_demand_charge_zar"],
            "energy": info["episode_energy_cost_zar"],
            "demand": info["episode_demand_charge_zar"],
            "clipped": clip_any / max(steps, 1), "sat": sat / max(steps, 1),
            "actions": np.array(actions),
        }

    clean_actions = None
    rows = []
    for fault, severity in CONDITIONS:
        acts = []
        for start, s in zip(starts, seeds):
            r = run(fault, severity, start, s)
            acts.append(r.pop("actions"))
            rows.append({"arm": label, "seed": seed, "fault": fault, "severity": severity, **r})
        if fault == "none":
            clean_actions = acts
        else:
            dev = float(np.mean([np.abs(a - b).mean() for a, b in zip(acts, clean_actions)]))
            for row in rows[-len(acts):]:
                row["act_dev"] = dev
    for row in rows:
        row.setdefault("act_dev", 0.0)
    return rows


def iqm(x):
    x = np.sort(np.asarray(x))
    lo, hi = int(np.ceil(len(x) * 0.25)), int(np.ceil(len(x) * 0.75))
    return float(x[lo:hi].mean())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--episodes", type=int, default=2)
    ap.add_argument("--eval-seed", type=int, default=4242)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--compare", nargs="*", default=["S1", "S0"])
    args = ap.parse_args()

    models = sorted(args.dir.glob(f"sac_{args.label}_seed*.zip"))
    assert models, f"no models matching sac_{args.label}_seed*.zip in {args.dir}"
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(eval_model, str(m), args.label, args.episodes, args.eval_seed)
                for m in models]
        rows = [r for f in futs for r in f.result()]
    df = pd.DataFrame(rows)
    out = args.dir / f"results_{args.label}.parquet"
    df.to_parquet(out, index=False)
    print(f"wrote {out} ({len(df)} rows)\n")

    # retained performance for the new arm + protocol arms, on the same episodes
    proto = pd.read_parquet(ROOT / "runs" / "protocol" / "results.parquet")
    per_new = df.groupby(["arm", "seed", "fault", "severity"]).agg(cost=("cost", "mean")).reset_index()
    per_old = proto.groupby(["arm", "seed", "fault", "severity"]).agg(cost=("cost", "mean")).reset_index()
    per = pd.concat([per_new, per_old], ignore_index=True)

    def retained(arm):
        sub = per[per.arm == arm]
        cl = sub[sub.fault == "none"].set_index("seed")["cost"]
        out = {}
        for f, s in CONDITIONS[1:]:
            cc = sub[(sub.fault == f) & (sub.severity == s)].set_index("seed")["cost"]
            out[(f, s)] = (cl / cc).reindex(cc.index).dropna()
        return out, float(cl.mean())

    rets = {}
    clean_means = {}
    for arm in [args.label] + args.compare:
        rets[arm], clean_means[arm] = retained(arm)

    print(f"clean mean cost: " + "  ".join(f"{a}={clean_means[a]:,.0f}" for a in rets))
    print("\nretained performance (IQM) and paired diffs vs " + ", ".join(args.compare) + ":")
    rng = np.random.default_rng(0)
    print(f"{'cond':>6} " + " ".join(f"{a:>7}" for a in rets) +
          "   " + " ".join(f"{args.label}-{a}: d [CI]{'':4}" for a in args.compare))
    for f, s in CONDITIONS[1:]:
        cells = []
        for arm in rets:
            cells.append(f"{iqm(rets[arm][(f, s)]):7.3f}")
        diffs = []
        for a in args.compare:
            d = (rets[args.label][(f, s)] - rets[a][(f, s)]).dropna()
            boots = [d[rng.choice(len(d), len(d))].mean() for _ in range(2000)]
            lo, hi = np.percentile(boots, [2.5, 97.5])
            sig = "*" if lo > 0 else ("v" if hi < 0 else " ")
            diffs.append(f"{d.mean():+.3f} [{lo:+.3f},{hi:+.3f}]{sig}")
        print(f"{f + str(s):>6} " + " ".join(cells) + "   " + "   ".join(diffs))
    print("\n(* = new arm significantly better, v = significantly worse, blank = overlap)")


if __name__ == "__main__":
    main()
