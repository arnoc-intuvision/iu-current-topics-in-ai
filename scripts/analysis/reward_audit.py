"""Reward-function audit: what the cost signal actually correlates with, per action.

Questions this answers on clean (S0) test data:
  1. Can the demand-charge term ever REWARD a discharge?  (idle incidence + slack analysis)
  2. What does an unguarded charge cost, and in what currency (energy vs demand)?
  3. Per-step reward scale/asymmetry: charge steps vs discharge steps under a random
     policy; spike magnitude vs typical reward (SAC critic target variance).
  4. Ratchet dynamics: same action, different penalty by history; where penalties land
     inside 30-min pairs (credit assignment across steps).
  5. How far does a never-charge policy get (LP, ch pinned 0) vs the full oracle -
     i.e. how attractive is the discharge-only local optimum the trained agents sit in?

Usage:
  python scripts/analysis/reward_audit.py [--sample 120] [--workers 8]
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
import yaml

CANONICAL = ROOT / "data" / "processed" / "canonical.parquet"
CONFIG = ROOT / "configs" / "env.yaml"


def load():
    with open(CONFIG) as f:
        config = yaml.safe_load(f)
    from sacbess.env.config import EnvConfig

    config = EnvConfig.from_dict(config)
    df = pd.read_parquet(CANONICAL)
    starts = pd.read_parquet(CANONICAL.parent / "episode_starts.parquet")
    test = starts.loc[starts["split"] == "test", "start_idx"].to_numpy()
    return df, config, test


def window_imports(imp_kwh, start_abs):
    """Replicate the env's 30-min pair rule: pairs anchored on absolute index parity."""
    wins = []
    t, acc = 0, 0.0
    while t < len(imp_kwh):
        if (start_abs + t) % 2 == 0:
            acc = imp_kwh[t]
            if t + 1 >= len(imp_kwh):
                break
            acc += imp_kwh[t + 1]
            wins.append(acc / 0.5)
            t += 2
        else:
            acc = imp_kwh[t]
            wins.append(acc / 0.5)
            t += 1
    return np.array(wins)


def run_policy(start, seed, kind, rng_seed=None):
    """Roll one policy on the raw env. kind in {naive_chg, b0, random}."""
    from sacbess.env.baselines import B0Policy
    from sacbess.env.microgrid_env import MicrogridEnv

    with open(CONFIG) as f:
        config = yaml.safe_load(f)
    from sacbess.env.config import EnvConfig

    config = EnvConfig.from_dict(config)
    env = MicrogridEnv(CANONICAL, config, "test")
    obs, info = env.reset(seed=seed, options={"start_idx": int(start)})
    b0 = B0Policy(config.battery.max_power_kw)
    rng = np.random.default_rng(rng_seed)
    rec = []
    for t in range(config.episode_steps):
        if kind == "naive_chg":
            a = -1.0 if (env.slots[env.pos] == "o" and env.battery.soc < 0.95) else 0.0
        elif kind == "b0":
            a = float(b0.action(env).reshape(-1)[0])
        else:
            a = float(rng.uniform(-1, 1))
        obs, r, _, _, info = env.step(np.array([a]))
        rec.append((a, info["energy_cost_zar"], info["demand_charge_zar"], r,
                    info["action_applied_kw"], env.pos - 1))
    arr = np.array(rec, dtype=float)
    out = {
        "start": int(start), "policy": kind,
        "cost": info["episode_energy_cost_zar"] + info["episode_demand_charge_zar"],
        "energy": info["episode_energy_cost_zar"],
        "demand": info["episode_demand_charge_zar"],
    }
    if kind == "random":
        a, ec, dc, r, app, pos = arr.T
        chg, dis = a < -0.05, a > 0.05
        paid = dc > 1e-9
        out.update({
            "chg_steps": int(chg.sum()), "dis_steps": int(dis.sum()),
            "chg_steps_paid": int((paid & chg).sum()),
            "dis_steps_paid": int((paid & dis).sum()),
            "n_windows_paid": int(paid.sum()),
            "reward_med": float(np.median(r)), "reward_std": float(r.std()),
            "reward_min": float(r.min()),
            "mean_r_chg": float(r[chg].mean()), "mean_r_dis": float(r[dis].mean()),
            "max_demand_zar": float(dc.max()),
            # demand charge can only fire when the env closes a 30-min pair: absolute pos odd
            "paid_on_pairclose_frac": float(np.mean(pos[paid] % 2 == 1)) if paid.any() else 0.0,
        })
    return out


def one_job(args):
    start, seed, kind, rng_seed = args
    try:
        return run_policy(start, seed, kind, rng_seed)
    except Exception as e:  # noqa: BLE001
        return {"start": start, "policy": kind, "error": str(e)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=120)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", type=Path, default=ROOT / "runs" / "protocol" / "reward_audit.csv")
    args = ap.parse_args()

    df, config, test = load()
    n = config.episode_steps

    # ---- Part 1: idle incidence + charge headroom slack (vectorised, all starts) ----
    load_kw = df["load_kw_true"].to_numpy()
    pv_kw = df["pv_kw_true"].to_numpy()
    rate = df["tariff_zar_kwh"].to_numpy()
    mtd = df["month_to_date_peak_kw"].to_numpy()
    net = load_kw - pv_kw
    idle_imp_kwh = np.maximum(net, 0.0) * config.dt_h
    idle_energy = idle_imp_kwh * rate
    inc, slack, idle_demand = [], [], []
    for s in test:
        e = idle_energy[s:s + n].sum()
        wins = window_imports(idle_imp_kwh[s:s + n], s)
        m0 = mtd[s]
        d = config.demand_charge_rate_zar_per_kw * max(0.0, wins.max() - m0) if len(wins) else 0.0
        idle_demand.append(d)
        inc.append(e + d)
        slack.append(m0 - wins.max())
    inc, idle_demand, slack = map(np.array, (inc, idle_demand, slack))
    print("== Part 1: demand-charge term vs a no-BESS site (all %d test starts) ==" % len(test))
    print(f"  episodes where IDLE pays any demand charge: {(idle_demand > 1e-9).sum()}")
    print(f"  idle demand charge: mean {idle_demand.mean():.1f}  max {idle_demand.max():.1f} ZAR/wk")
    print(f"  slack mtd0 - max idle import window: median {np.median(slack):.0f} kW  "
          f"q10 {np.quantile(slack, 0.1):.0f}  min {slack.min():.0f} kW")

    # ---- Part 2: policy rollouts on a stratified sample ----
    rng = np.random.default_rng(0)
    sample = np.sort(rng.choice(test, size=min(args.sample, len(test)), replace=False))
    jobs = []
    for i, s in enumerate(sample):
        seed = 10_000 + int(s)
        jobs.append((int(s), seed, "naive_chg", None))
        jobs.append((int(s), seed, "b0", None))
        jobs.append((int(s), seed, "random", int(s)))
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for r in ex.map(one_job, jobs, chunksize=3):
            rows.append(r)
    pol = pd.DataFrame(rows)
    if "error" in pol:
        errs = pol[pol["error"].notna()]
        if len(errs):
            print(f"  ({len(errs)} rollout errors, e.g. {errs.iloc[0]['error']!r})")
        pol = pol[pol["error"].isna()].drop(columns=["error"])
    if pol["cost"].isna().any():
        pol = pol[pol["cost"].notna()]

    print("\n== Part 2: unguarded off-peak charge vs B0 (same %d starts, matched SoC/seed) ==" % len(sample))
    idle_map = dict(zip(test, inc))
    sample_idle = np.array([idle_map[s] for s in sample])
    for kind in ["naive_chg", "b0"]:
        sub = pol[pol.policy == kind]
        extra = sub["cost"].mean() - sample_idle.mean()
        print(f"  {kind:>10}: cost {sub['cost'].mean():>10,.0f}  energy {sub['energy'].mean():>9,.0f}  "
              f"demand {sub['demand'].mean():>10,.0f}  extra-vs-idle {extra:>9,.0f} ZAR/wk  "
              f"(demand share {100 * sub['demand'].mean() / max(extra, 1):.0f}%)")

    rnd = pol[pol.policy == "random"]
    print("\n== Part 3: random-policy reward structure (%d episodes) ==" % len(rnd))
    if len(rnd) and "reward_std" in rnd:
        for k, v in {
            "episodes hitting any demand charge": (rnd.n_windows_paid > 0).mean(),
            "charge steps penalised": rnd.chg_steps_paid.sum() / max(rnd.chg_steps.sum(), 1),
            "discharge steps penalised": rnd.dis_steps_paid.sum() / max(rnd.dis_steps.sum(), 1),
            "mean per-step reward | charge steps": rnd.mean_r_chg.mean(),
            "mean per-step reward | discharge steps": rnd.mean_r_dis.mean(),
            "reward std (within episode, mean)": rnd.reward_std.mean(),
            "worst single-step reward (mean over eps)": rnd.reward_min.mean(),
            "worst single-step demand ZAR (mean)": rnd.max_demand_zar.mean(),
        }.items():
            print(f"  {k:>42}: {v:,.3f}")

    # ---- Part 4: never-charge LP vs full oracle on the same sample ----
    from oracle_bound import solve_episode

    lp_rows = []
    for s in sample:
        sl = slice(int(s), int(s) + n)
        from sacbess.env.microgrid_env import MicrogridEnv

        probe = MicrogridEnv(CANONICAL, config, "test")
        _, info = probe.reset(seed=10_000 + int(s), options={"start_idx": int(s)})
        soc0 = info["soc"]
        del probe
        full = solve_episode(net[sl], rate[sl], float(mtd[int(s)]), soc0, config, start_abs=int(s))
        nochg = solve_episode(net[sl], rate[sl], float(mtd[int(s)]), soc0, config,
                              allow_charge=False, start_abs=int(s))
        if full is None or nochg is None:
            continue
        lp_rows.append({"start": int(s), "idle": idle_map[int(s)],
                        "oracle": full["oracle_cost"], "never_charge": nochg["oracle_cost"]})
    lp = pd.DataFrame(lp_rows)
    lp["oracle_sav"] = (lp.idle - lp.oracle) / lp.idle
    lp["nochg_sav"] = (lp.idle - lp.never_charge) / lp.idle
    print("\n== Part 4: value of charging at all (LP, perfect foresight, matched SoC) ==")
    print(f"  never-charge savings: mean {lp.nochg_sav.mean()*100:.1f}%  median {lp.nochg_sav.median()*100:.1f}%")
    print(f"  full-oracle savings:  mean {lp.oracle_sav.mean()*100:.1f}%  median {lp.oracle_sav.median()*100:.1f}%")
    print(f"  => fraction of oracle value requiring charge actions: "
          f"{(1 - lp.nochg_sav.mean() / lp.oracle_sav.mean()) * 100:.0f}%")

    pol.to_csv(args.out, index=False)
    lp.to_csv(args.out.with_name("reward_audit_lp.csv"), index=False)
    print(f"\nwrote {args.out} and {args.out.with_name('reward_audit_lp.csv')}")


if __name__ == "__main__":
    main()
