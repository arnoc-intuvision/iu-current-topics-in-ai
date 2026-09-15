"""Seasonally stratified re-evaluation of the existing protocol models (no retraining).

8 test weeks: one each from Sep/Dec/Jan/Feb/Mar/Apr 2025-26 + TWO June 2026 (winter,
6.24/1.03 ZAR/kWh tariff) weeks. Evaluates all 40 protocol models (S0-S3 x 10 seeds)
plus idle and B0 across the 22 fault conditions on the IDENTICAL weeks and
initial-SoC seeds, so per-week and per-season comparisons are purely policy.

  python scripts/analysis/eval_seasonal.py [--workers 8]
"""

from __future__ import annotations

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

CANONICAL = ROOT / "data" / "processed" / "canonical.parquet"
CONFIG = ROOT / "configs" / "env.yaml"
CONDITIONS = [("none", 0)] + [
    (f, s) for f in ["F1", "F2", "F3", "F4", "F5", "F6a", "F6b"] for s in [1, 2, 3]
]
SEED = 4243

_W: dict = {}


def _init_worker():
    sys.path.insert(0, str(ROOT / "src"))
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import yaml as _yaml

    from sacbess.env.config import EnvConfig

    with open(CONFIG) as f:
        _W["config"] = EnvConfig.from_dict(_yaml.safe_load(f))


def stratified_starts() -> list[tuple[int, str]]:
    df = pd.read_parquet(CANONICAL)
    st = pd.read_parquet(CANONICAL.parent / "episode_starts.parquet")
    test = st[st["split"] == "test"].start_idx.to_numpy()
    months = df.index[test].to_period("M")
    rng = np.random.default_rng(SEED)
    picks, used = [], set()
    for p in ["2025-09", "2025-12", "2026-01", "2026-02", "2026-03", "2026-04",
              "2026-06", "2026-06"]:  # two winter weeks
        cand = test[months == pd.Period(p, "M")]
        assert len(cand) > 0, p
        while True:
            take = int(cand[rng.integers(len(cand))])
            if take not in used:
                used.add(take)
                picks.append((take, "jun-winter" if p == "2026-06" else p[:7]))
                break
    return picks


def one_task(task):
    """One (policy, week) pair across all 22 conditions; model loaded once."""
    kind, model_path, reliability, head, start, month = task
    config = _W["config"]
    soc_seed = 10_000 + start
    rows = []

    import numpy as np

    from stable_baselines3 import SAC

    from sacbess.env.baselines import B0Policy
    from sacbess.factory import build_env

    model = SAC.load(model_path, device="cpu") if model_path and model_path != "B0" else None
    b0 = B0Policy(config.battery.max_power_kw) if model_path == "B0" else None
    for fault, sev in CONDITIONS:
        env = build_env(CANONICAL, config, "test",
                        fault=(None if fault == "none" else fault),
                        severity=sev, reliability=reliability, head_path=head,
                        dispatch_features=True)
        obs, info = env.reset(seed=soc_seed, options={"start_idx": int(start)})
        for _ in range(config.episode_steps):
            if model is not None:
                a = model.predict(obs, deterministic=True)[0]
            elif b0 is not None:
                a = b0.action(env.unwrapped)
            else:
                a = np.zeros(1)
            obs, _, _, _, info = env.step(a)
        rows.append((kind, fault, sev,
                     info["episode_energy_cost_zar"] + info["episode_demand_charge_zar"],
                     info["episode_energy_cost_zar"], info["episode_demand_charge_zar"]))
    return [{"policy": k, "fault": f, "severity": s, "start": start,
             "month": month, "cost": c, "energy": e, "demand": d}
            for k, f, s, c, e, d in rows]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    picks = stratified_starts()
    print("stratified test weeks (start_idx, month):")
    for s, m in picks:
        print(f"  {s:>6}  {m}")

    tasks = []
    for start, month in picks:
        tasks.append(("idle", None, None, None, start, month))
        tasks.append(("B0", "B0", None, None, start, month))
        for arm in ["S0", "S1", "S2", "S3"]:
            rel = {"S2": "S2", "S3": "S3"}.get(arm)
            head = (str(ROOT / "runs" / "protocol" / "reliability_head.pt")
                    if arm == "S3" else None)
            for seed in range(10):
                tasks.append((arm, str(ROOT / "runs" / "protocol" /
                                       f"sac_{arm}_seed{seed}.zip"), rel, head,
                              start, month))
    print(f"{len(tasks)} tasks x {len(CONDITIONS)} conditions = "
          f"{len(tasks) * len(CONDITIONS)} rollouts")

    rows = []
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init_worker) as ex:
        for i, r in enumerate(ex.map(one_task, tasks), 1):
            rows.extend(r)
            if i % 50 == 0:
                print(f"  {i}/{len(tasks)} tasks")
    df = pd.DataFrame(rows)
    out = ROOT / "runs" / "protocol" / "results_seasonal.parquet"
    df.to_parquet(out, index=False)
    print(f"wrote {out} ({len(df)} rows)")


if __name__ == "__main__":
    main()
