"""Study protocol: 4 arms x 10 seeds - parallel training, robustness evaluation, IQM analysis.

Phases:
  python scripts/run_protocol.py train --steps 50000 --seeds 10 --workers 8
  python scripts/run_protocol.py eval --seeds 10 --episodes 2 --workers 4
  python scripts/run_protocol.py analyze
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

ARMS = ["S0", "S1", "S2", "S3"]
PY = sys.executable
RUNS = ROOT / "runs" / "protocol"
CONDITIONS = [("none", 0)] + [
    (f, s) for f in ["F1", "F2", "F3", "F4", "F5", "F6a", "F6b"] for s in [1, 2, 3]
]


def cmd_train(args) -> None:
    RUNS.mkdir(parents=True, exist_ok=True)
    head_path = RUNS / "reliability_head.pt"
    if not head_path.exists():
        print("pretraining S3 reliability head...")
        subprocess.run(
            [PY, str(ROOT / "scripts" / "train_head.py"), "--out", str(head_path)], check=True
        )
    jobs = []
    for arm in ARMS:
        for seed in range(args.seeds):
            model = RUNS / f"sac_{arm}_seed{seed}.zip"
            if model.exists():
                print(f"skip {model.name} (exists)")
                continue
            jobs.append((arm, seed))
    running: list[tuple[subprocess.Popen, str]] = []
    t0 = time.time()
    idx = 0
    while idx < len(jobs) or running:
        while idx < len(jobs) and len(running) < args.workers:
            arm, seed = jobs[idx]
            log = open(RUNS / f"train_{arm}_seed{seed}.log", "w")  # noqa: SIM115
            env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
            p = subprocess.Popen(
                [PY, str(ROOT / "scripts" / "train_sac.py"),
                 "--arm", arm, "--seed", str(seed), "--steps", str(args.steps),
                 "--out", str(RUNS), "--head", str(head_path),
                 "--dispatch-features", "--demo-steps", str(args.demo_steps),
                 "--demo-decay", str(args.demo_decay), "--hidden", args.hidden],
                stdout=log, stderr=subprocess.STDOUT, env=env,
            )
            running.append((p, f"{arm}_seed{seed}"))
            print(f"[{idx + 1}/{len(jobs)}] launched {arm} seed {seed}")
            idx += 1
        still = []
        for p, tag in running:
            if p.poll() is None:
                still.append((p, tag))
            else:
                print(f"done {tag} (rc={p.returncode}, elapsed {time.time() - t0:.0f}s)")
        running = still
        time.sleep(2)
    print(f"training phase complete in {time.time() - t0:.0f}s ({len(jobs)} runs)")


def _eval_model(arm: str, seed: int, episodes: int, eval_seed: int) -> list[dict]:
    import numpy as np
    import yaml

    from sacbess.env.config import EnvConfig
    from sacbess.factory import build_env

    with open(ROOT / "configs" / "env.yaml") as f:
        config = EnvConfig.from_dict(yaml.safe_load(f))
    canonical = ROOT / "data" / "processed" / "canonical.parquet"
    from stable_baselines3 import SAC

    model = SAC.load(RUNS / f"sac_{arm}_seed{seed}.zip", device="cpu")
    reliability = {"S2": "S2", "S3": "S3"}.get(arm)
    head = str(RUNS / "reliability_head.pt") if arm == "S3" else None

    from sacbess.env.microgrid_env import MicrogridEnv

    probe = MicrogridEnv(canonical, config, "test")
    p_max = config.battery.max_power_kw
    soc_min = config.battery.min_soc
    soc_max = config.battery.max_soc
    rng = np.random.default_rng(eval_seed)
    starts = rng.choice(probe.valid_starts["test"], size=episodes, replace=True)
    seeds = [eval_seed + 1000 + i for i in range(episodes)]
    del probe

    def run(fault, severity, start, s):
        env = build_env(
            canonical, config, "test",
            fault=(None if fault in (None, "none") else fault),
            severity=severity, reliability=reliability, head_path=head,
            dispatch_features=True,
        )
        obs, info = env.reset(seed=s, options={"start_idx": int(start)})
        actions, steps = [], 0
        clip_any = clip_dis = clip_chg = clip_floor = clip_ceil = clip_full = sat = 0
        done = False
        while not done:
            a, _ = model.predict(obs, deterministic=True)
            obs, _, term, trunc, info = env.step(a)
            actions.append(float(np.asarray(a).reshape(-1)[0]))
            steps += 1
            des, app = info["action_desired_kw"], info["action_applied_kw"]
            if info["clipped"]:
                clip_any += 1
                if des > 0:
                    clip_dis += 1
                    if info["soc"] <= soc_min + 1e-9:
                        clip_floor += 1
                else:
                    clip_chg += 1
                    if info["soc"] >= soc_max - 1e-9:
                        clip_ceil += 1
                if abs(des) >= p_max - 0.5:
                    clip_full += 1
            if abs(info["action_raw"]) >= 0.99:
                sat += 1
            done = term or trunc
        n = max(steps, 1)
        return {
            "cost": info["episode_energy_cost_zar"] + info["episode_demand_charge_zar"],
            "energy": info["episode_energy_cost_zar"],
            "demand": info["episode_demand_charge_zar"],
            "clipped": clip_any / n,
            "clip_dis": clip_dis / n,
            "clip_chg": clip_chg / n,
            "clip_floor": clip_floor / n,
            "clip_ceil": clip_ceil / n,
            "clip_full": clip_full / n,
            "sat": sat / n,
            "actions": np.array(actions),
        }

    clean_actions = None
    rows = []
    for fault, severity in CONDITIONS:
        acts = []
        for start, s in zip(starts, seeds):
            r = run(fault, severity, start, s)
            acts.append(r.pop("actions"))
            rows.append({
                "arm": arm, "seed": seed, "fault": fault, "severity": severity,
                "start": int(start), **r,
            })
        if fault == "none":
            clean_actions = acts
        else:
            dev = float(np.mean([np.abs(a - b).mean() for a, b in zip(acts, clean_actions)]))
            for row in rows[-len(acts):]:
                row["act_dev"] = dev
    for row in rows:
        row.setdefault("act_dev", 0.0)
    return rows


def cmd_eval(args) -> None:
    RUNS.mkdir(parents=True, exist_ok=True)
    tasks = [(arm, seed) for arm in ARMS for seed in range(args.seeds)
             if (RUNS / f"sac_{arm}_seed{seed}.zip").exists()]
    all_rows = []
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(_eval_model, a, s, args.episodes, args.eval_seed): (a, s) for a, s in tasks}
        for done_i, fut in enumerate(as_completed(futs), start=1):
            rows = fut.result()
            all_rows.extend(rows)
            a, s = futs[fut]
            print(f"evaluated {a} seed {s} ({done_i}/{len(tasks)}, {time.time() - t0:.0f}s)")
    import pandas as pd

    out = RUNS / "results.parquet"
    pd.DataFrame(all_rows).to_parquet(out, index=False)
    print(f"wrote {out} ({len(all_rows)} rows)")


def iqm(x: np.ndarray) -> float:
    s = np.sort(np.asarray(x, dtype=float))
    n = len(s)
    lo = int(np.ceil(n * 0.25))
    hi = int(np.ceil(n * 0.75))
    return float(np.mean(s[lo:hi])) if hi > lo else float(np.mean(s))


def sig(fault, severity, arm, n_seeds, retained):
    va = retained.get((arm, fault, severity))
    vb = retained.get(("S1", fault, severity))
    if va is None or vb is None or len(va) < n_seeds or len(vb) < n_seeds:
        return "  --  "
    d = va - vb
    lo, hi = np.percentile(d, [2.5, 97.5])
    return "  yes  " if lo > 0 else ("  no   " if hi < 0 else " overlap")


def cmd_analyze(args) -> None:
    import numpy as np
    import pandas as pd

    df = pd.read_parquet(RUNS / "results.parquet")
    per = df.groupby(["arm", "seed", "fault", "severity"]).agg(
        cost=("cost", "mean"), energy=("energy", "mean"), demand=("demand", "mean"),
        clipped=("clipped", "mean"), act_dev=("act_dev", "mean"),
        clip_dis=("clip_dis", "mean"), clip_chg=("clip_chg", "mean"),
        clip_floor=("clip_floor", "mean"), clip_ceil=("clip_ceil", "mean"),
        clip_full=("clip_full", "mean"), sat=("sat", "mean"),
    ).reset_index()

    print("\n=== Clean-data cost (price of robustness anchor), ZAR/week ===")
    for arm in ARMS:
        c = per[(per.arm == arm) & (per.fault == "none")]["cost"].to_numpy()
        print(f"  {arm}: IQM {iqm(c):,.0f}  [{np.percentile(c, 2.5):,.0f}, {np.percentile(c, 97.5):,.0f}] (n={len(c)})")

    retained = {}
    for arm in ARMS:
        sub = per[per.arm == arm]
        cl = sub[sub.fault == "none"].set_index("seed")["cost"]
        for fault, severity in CONDITIONS[1:]:
            cc = sub[(sub.fault == fault) & (sub.severity == severity)].set_index("seed")["cost"]
            retained[(arm, fault, severity)] = (cl / cc).reindex(cc.index).dropna().to_numpy()

    rng = np.random.default_rng(0)
    B = 5000
    n_seeds = args.seeds
    seeds_idx = np.arange(n_seeds)
    boot = {k: np.empty(B) for k in retained}
    for b in range(B):
        idx = rng.choice(seeds_idx, size=n_seeds, replace=True)
        for k, v in retained.items():
            if len(v) == n_seeds:
                boot[k][b] = iqm(v[idx])

    print("\n=== Retained performance (IQM [2.5%, 97.5% bootstrap CI], clean/corrupted cost) ===")
    header = f"{'cond':>8} | " + " | ".join(f"{a:>24}" for a in ARMS) + " | S2-S1 sig | S3-S1 sig"
    print(header)
    print("-" * len(header))
    for fault, severity in CONDITIONS[1:]:
        cells = []
        for arm in ARMS:
            v = retained.get((arm, fault, severity), np.array([]))
            if len(v) != n_seeds:
                cells.append(f"{'--':>24}")
                continue
            bq = boot[(arm, fault, severity)]
            cells.append(f"{iqm(v):.3f} [{np.percentile(bq, 2.5):.3f},{np.percentile(bq, 97.5):.3f}]")
        print(f"{fault + str(severity):>8} | " + " | ".join(cells)
              + f" | {sig(fault, severity, 'S2', n_seeds, retained)}"
              f" | {sig(fault, severity, 'S3', n_seeds, retained)}")

    print("\n=== Companions (arm means across seeds): clip decomposition, action deviation ===")
    print("    clipped = any curtailment of the requested power;")
    print("    floor/ceil = discharge cut at the reserve floor / charge cut at the SoC ceiling;")
    print("    full-req = curtailed while requesting the full inverter rating; sat = |a| >= 0.99")
    for arm in ARMS:
        sub = per[(per.arm == arm) & ((per.severity == 3) | (per.fault == "none"))]
        for fault in ["none", "F1", "F2", "F6a"]:
            s2 = sub[(sub.fault == fault) & (sub.severity == (3 if fault != "none" else 0))]
            if len(s2):
                print(
                    f"  {arm} {fault:>4}: clipped {s2.clipped.mean() * 100:5.1f}% "
                    f"(floor {s2.clip_floor.mean() * 100:5.1f} | ceil {s2.clip_ceil.mean() * 100:4.1f} | "
                    f"full-req {s2.clip_full.mean() * 100:4.1f})  sat {s2.sat.mean() * 100:4.1f}%  "
                    f"act_dev {s2.act_dev.mean():.3f}"
                )

    out = RUNS / "analysis_retained.csv"
    rows = [{"arm": a, "fault": f, "severity": s,
             "iqm": iqm(retained[(a, f, s)]), "ci_lo": np.percentile(boot[(a, f, s)], 2.5),
             "ci_hi": np.percentile(boot[(a, f, s)], 97.5)}
            for (a, f, s) in retained if len(retained[(a, f, s)]) == n_seeds]
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nwrote {out}")

    comp = per.groupby(["arm", "fault", "severity"]).agg(
        cost=("cost", "mean"), energy=("energy", "mean"), demand=("demand", "mean"),
        clipped=("clipped", "mean"), clip_dis=("clip_dis", "mean"), clip_chg=("clip_chg", "mean"),
        clip_floor=("clip_floor", "mean"), clip_ceil=("clip_ceil", "mean"),
        clip_full=("clip_full", "mean"), sat=("sat", "mean"), act_dev=("act_dev", "mean"),
    ).reset_index()
    comp_out = RUNS / "analysis_companions.csv"
    comp.to_csv(comp_out, index=False)
    print(f"wrote {comp_out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="phase", required=True)
    t = sub.add_parser("train")
    t.add_argument("--steps", type=int, default=300_000)
    t.add_argument("--seeds", type=int, default=10)
    t.add_argument("--workers", type=int, default=8)
    t.add_argument("--demo-steps", type=int, default=30_000)
    t.add_argument("--demo-decay", type=int, default=100_000)
    t.add_argument("--hidden", default="1024,1024")
    e = sub.add_parser("eval")
    e.add_argument("--seeds", type=int, default=10)
    e.add_argument("--episodes", type=int, default=2)
    e.add_argument("--workers", type=int, default=4)
    e.add_argument("--eval-seed", type=int, default=4242)
    a = sub.add_parser("analyze")
    a.add_argument("--seeds", type=int, default=10)
    args = ap.parse_args()
    {"train": cmd_train, "eval": cmd_eval, "analyze": cmd_analyze}[args.phase](args)


if __name__ == "__main__":
    main()
