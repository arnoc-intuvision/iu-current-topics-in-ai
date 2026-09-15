"""Perfect-foresight LP oracle: upper bound on achievable weekly savings (S0, clean data).

For every valid test episode start, solves the exact cost-minimisation problem the env
poses (TOU energy cost + marginal 30-min-window demand charge over the month-to-date
peak) with full knowledge of the coming week. Every env constraint is linear in the
decision variables, so the LP optimum is a true upper bound on any causal policy
(RL, rule-based, or otherwise):

  vars per step t: ch_t, dis_t >= 0 (kW), imp_t, exp_t >= 0 (kW); soc_t (kWh); peak P.
  imp_t - exp_t - ch_t + dis_t          = net_t            (load - PV)
  soc_{t+1} - soc_t - dt*eta*ch_t + (dt/eta)*dis_t = 0
  ch_t <= 1000, dis_t <= 1000                     (inverter)
  dt*eta*ch_t + soc_t <= cap                      (SoC-ceiling headroom, battery.headrooms)
  (dt/eta)*dis_t - soc_t <= -cap*min_soc          (reserve-floor headroom)
  P >= (imp_i + imp_j)/2 for every 30-min pair     (env window rule, incl. partial first pair)
  P >= mtd_peak[start]                            (exposure initialises at the no-BESS MTD peak)
  cost = sum rate_t*imp_t*dt + 150*(P - mtd_peak[start]);  export earns nothing (credit = 0)

Simultaneous charge+discharge is never optimal (net replacement is feasible and no
dearer), so the LP optimum is attainable by a signed-power policy like the env's.

Compares, per episode: idle (zero action), B0 rule floor, LP oracle (env-matched
initial SoC, plus 0.4/1.0 brackets), and splits cost into energy vs demand.

Usage:
  python scripts/analysis/oracle_bound.py [--starts 904] [--workers 8] [--out runs/protocol/oracle.csv]
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
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

CANONICAL = ROOT / "data" / "processed" / "canonical.parquet"
CONFIG = ROOT / "configs" / "env.yaml"


def solve_episode(net_kw, rate, mtd0, soc0_frac, cfg, allow_charge: bool = True,
                  start_abs: int = 0) -> dict | None:
    """LP over one episode window. net_kw/rate are absolute-index slices.
    allow_charge=False pins ch=0: upper bound for never-charge policies.
    start_abs sets absolute index parity (the env pairs :15/:45 windows on it)."""
    n = len(net_kw)
    dt = cfg.dt_h
    cap = cfg.battery.capacity_kwh
    pmax = cfg.battery.max_power_kw
    eta = cfg.battery.one_way_efficiency
    soc_min = cap * cfg.battery.min_soc
    soc_max = cap * cfg.battery.max_soc
    drate = cfg.demand_charge_rate_zar_per_kw

    # variable layout: ch(0..n-1) dis(n..2n-1) imp(2n..3n-1) exp(3n..4n-1) soc(4n..4n+n) P(last)
    i_ch = lambda t: t
    i_dis = lambda t: n + t
    i_imp = lambda t: 2 * n + t
    i_exp = lambda t: 3 * n + t
    i_soc = lambda t: 4 * n + t
    iP = 5 * n + 1  # soc occupies 4n .. 5n inclusive (n+1 entries); P must not collide
    nv = 5 * n + 2

    # 30-min pair windows under the env rule (episode-relative step parity mirrors
    # absolute parity because each step advances _pos by 1 from `start`)
    pairs = []  # (t_first, t_second or None)
    t = 0
    while t < n:
        if (t + start_abs) % 2 == 0:
            if t + 1 < n:
                pairs.append((t, t + 1))
            t += 2
        else:
            pairs.append((t, None))  # partial pair: window = imp_kwh/0.5 = imp_kw/2
            t += 1

    rows, cols, vals, beq = [], [], [], []
    r = 0
    for t in range(n):  # net balance: imp - exp - ch + dis = net
        rows += [r, r, r, r]
        cols += [i_imp(t), i_exp(t), i_ch(t), i_dis(t)]
        vals += [1.0, -1.0, -1.0, 1.0]
        beq.append(net_kw[t])
        r += 1
    for t in range(n):  # soc_{t+1} - soc_t - dt*eta*ch + (dt/eta)*dis = 0
        rows += [r, r, r, r]
        cols += [i_soc(t + 1), i_soc(t), i_ch(t), i_dis(t)]
        vals += [1.0, -1.0, -dt * eta, dt / eta]
        beq.append(0.0)
        r += 1
    Aeq = coo_matrix((vals, (rows, cols)), shape=(r, nv))
    beq = np.array(beq)

    rows, cols, vals, bub = [], [], [], []
    r = 0
    for t in range(n):
        rows += [r, r]
        cols += [i_ch(t), i_soc(t)]
        vals += [dt * eta, 1.0]
        bub.append(soc_max)  # dt*eta*ch + soc <= cap
        r += 1
        rows += [r, r]
        cols += [i_dis(t), i_soc(t)]
        vals += [dt / eta, -1.0]
        bub.append(-soc_min)  # (dt/eta)*dis - soc <= -cap*min_soc
        r += 1
    for t1, t2 in pairs:
        if t2 is None:
            rows.append(r); cols.append(i_imp(t1)); vals.append(0.5)
        else:
            rows += [r, r]; cols += [i_imp(t1), i_imp(t2)]; vals += [0.5, 0.5]
        rows.append(r); cols.append(iP); vals.append(-1.0)
        bub.append(0.0)  # pair import avg <= P
        r += 1
    rows.append(r); cols.append(iP); vals.append(-1.0); bub.append(-mtd0)  # P >= mtd0
    r += 1
    Aub = coo_matrix((vals, (rows, cols)), shape=(r, nv))
    bub = np.array(bub)

    c = np.zeros(nv)
    c[2 * n:3 * n] = rate * dt
    c[iP] = drate

    bounds = [(0, pmax if allow_charge else 0)] * n + [(0, pmax)] * n + [(0, None)] * n + [(0, None)] * n
    bounds += [(soc_min, soc_max)] * (n + 1) + [(0, None)]
    bounds[i_soc(0)] = (soc0_frac * cap, soc0_frac * cap)

    res = linprog(c, A_ub=Aub, b_ub=bub, A_eq=Aeq, b_eq=beq, bounds=bounds, method="highs")
    if not res.success:
        return None
    imp = res.x[2 * n:3 * n]
    ch = res.x[:n]
    dis = res.x[n:2 * n]
    energy = float(imp @ rate * dt)
    peak = res.x[iP]
    demand = drate * max(0.0, peak - mtd0)
    surplus_charged = float(np.minimum(ch, np.maximum(-net_kw, 0.0)).sum() * dt)  # kWh from PV surplus
    return {
        "oracle_cost": energy + demand,
        "oracle_energy": energy,
        "oracle_demand": demand,
        "oracle_peak_kw": float(peak),
        "charged_kwh": float(ch.sum() * dt),
        "discharged_kwh": float(dis.sum() * dt),
        "surplus_charged_kwh": surplus_charged,
        "final_soc": float(res.x[i_soc(n)] / cap),
    }


def run_policy(canonical, config, start, seed, kind) -> dict:
    from sacbess.env.baselines import B0Policy
    from sacbess.env.microgrid_env import MicrogridEnv

    env = MicrogridEnv(canonical, config, "test")
    obs, info = env.reset(seed=seed, options={"start_idx": int(start)})
    soc0 = info["soc"]
    b0 = B0Policy(config.battery.max_power_kw)
    n_pos, n_neg = 0, 0
    for _ in range(config.episode_steps):  # raw env has no TimeLimit wrapper
        if kind == "idle":
            a = np.zeros(1)
        else:
            a = b0.action(env)
        obs, _, term, trunc, info = env.step(a)
        if info["action_raw"] > 0:
            n_pos += 1
        elif info["action_raw"] < 0:
            n_neg += 1
    return {
        f"{kind}_cost": info["episode_energy_cost_zar"] + info["episode_demand_charge_zar"],
        f"{kind}_energy": info["episode_energy_cost_zar"],
        f"{kind}_demand": info["episode_demand_charge_zar"],
        f"{kind}_soc0": soc0,
        f"{kind}_dis_steps": n_pos,
        f"{kind}_chg_steps": n_neg,
    }


def one_start(args) -> dict:
    start, seed, soc_variant = args
    import yaml as _yaml

    from sacbess.env.microgrid_env import MicrogridEnv

    with open(CONFIG) as f:
        config = _yaml.safe_load(f)
    from sacbess.env.config import EnvConfig

    config = EnvConfig.from_dict(config)
    df = pd.read_parquet(CANONICAL)
    n = config.episode_steps
    sl = slice(start, start + n)
    net = (df["load_kw_true"] - df["pv_kw_true"]).to_numpy()[sl]
    rate = df["tariff_zar_kwh"].to_numpy()[sl]
    mtd0 = float(df["month_to_date_peak_kw"].to_numpy()[start])

    # env-matched initial SoC for this seed
    probe = MicrogridEnv(CANONICAL, config, "test")
    _, info = probe.reset(seed=seed, options={"start_idx": int(start)})
    soc_matched = info["soc"]
    del probe

    out = {"start": int(start), "month": str(df.index[start].to_period("M")), "mtd0_kw": mtd0,
           "soc0_matched": soc_matched}
    out.update(run_policy(CANONICAL, config, start, seed, "idle"))
    out.update(run_policy(CANONICAL, config, start, seed, "b0"))

    for tag, soc in [("oracle", soc_matched)] + ([("oracle_soc04", 0.4), ("oracle_soc10", 1.0)]
                                                 if soc_variant else []):
        r = solve_episode(net, rate, mtd0, soc, config, start_abs=int(start))
        if r is None:
            return {"start": int(start), "status": "lp_fail"}
        out.update({k if k == "final_soc" else f"{tag}_{k.replace('oracle_', '')}": v for k, v in r.items()})
    out["status"] = "ok"
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--starts", type=int, default=904)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", type=Path, default=ROOT / "runs" / "protocol" / "oracle.csv")
    args = ap.parse_args()

    starts = pd.read_parquet(CANONICAL.parent / "episode_starts.parquet")
    test = starts.loc[starts["split"] == "test", "start_idx"].to_numpy()
    if args.starts < len(test):
        rng = np.random.default_rng(0)
        test = np.sort(rng.choice(test, size=args.starts, replace=False))
    jobs = [(int(s), 10_000 + int(s), True) for s in test]

    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for i, row in enumerate(ex.map(one_start, jobs, chunksize=4), 1):
            rows.append(row)
            if i % 100 == 0:
                print(f"{i}/{len(jobs)}")

    df = pd.DataFrame(rows)
    ok = df[df["status"] == "ok"]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    ok.to_csv(args.out, index=False)
    print(f"wrote {args.out} ({len(ok)} episodes)")

    idle = ok["idle_cost"]
    for col, label in [("b0_cost", "B0"), ("oracle_cost", "oracle(matched SoC)"),
                       ("oracle_soc04_cost", "oracle SoC0=0.4"), ("oracle_soc10_cost", "oracle SoC0=1.0")]:
        if col in ok:
            sav = (idle - ok[col]) / idle
            print(f"{label:>22}: cost IQM {ok[col].mean():>10,.0f}  savings mean {sav.mean()*100:5.1f}%  "
                  f"IQM {sav.quantile(0.75) * 0 + sav.median()*100:5.1f}%  min {sav.min()*100:6.1f}%  "
                  f"p90 {sav.quantile(0.9)*100:5.1f}%")
    print("\nby month (mean savings vs idle):")
    g = ok.assign(b0_sav=(idle - ok.b0_cost) / idle,
                  oracle_sav=(idle - ok.oracle_cost) / idle,
                  idle_demand=ok.idle_demand, oracle_demand=ok.oracle_demand)
    print(g.groupby("month")[["b0_sav", "oracle_sav", "idle_demand", "oracle_demand"]].mean().round(3))


if __name__ == "__main__":
    main()
