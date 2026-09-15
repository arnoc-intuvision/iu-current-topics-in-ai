"""Evaluate stage-2 checkpoint ladders: every ckpt_{tag}_s{steps}.zip through the
22-condition fault matrix; produce the forgetting-vs-robustness frontier.

Usage:
  python scripts/analysis/eval_ladder.py --dir <ckpt_dir> --tag <tag> \
      --episodes 2 --workers 8
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

from eval_arm import eval_model  # noqa: E402  (same dir)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--episodes", type=int, default=2)
    ap.add_argument("--eval-seed", type=int, default=4242)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--split", default="test",
                    help="evaluation split: test (reporting) or val (checkpoint selection)")
    args = ap.parse_args()

    ckpts = sorted(args.dir.glob(f"ckpt_{args.tag}_seed*_s*.zip"))
    ckpts += sorted(args.dir.glob(f"sac_{args.tag}_seed*.zip"))  # final models (100k)
    assert ckpts, f"no checkpoints in {args.dir}"
    labels = [f"{p.stem}|{args.tag}" for p in ckpts]

    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(eval_model, str(p), lbl, args.episodes, args.eval_seed, args.split)
                for p, lbl in zip(ckpts, labels)]
        rows = [r for f in futs for r in f.result()]

    df = pd.DataFrame(rows)
    df = df.rename(columns={"arm": "model"})
    df[["model_name", "tag"]] = df["model"].str.split("|", expand=True)
    df["seed"] = df["model_name"].str.extract(r"seed(\d+)").astype(int)
    step = df["model_name"].str.extract(r"_s(\d{6})$")[0]
    df["step"] = pd.to_numeric(step, errors="coerce").fillna(100000.0)
    suffix = "" if args.split == "test" else f"_{args.split}"
    out = args.dir / f"results_{args.tag}_ladder{suffix}.parquet"
    df.to_parquet(out, index=False)
    print(f"wrote {out} ({len(df)} rows, {df.model_name.nunique()} models, split={args.split})\n")

    per = df.groupby(["model_name", "step", "seed", "fault", "severity"]).agg(
        cost=("cost", "mean")).reset_index()

    # frontier: per checkpoint step, clean cost + absolute costs across conditions
    proto = pd.read_parquet(ROOT / "runs" / "protocol" / "results.parquet")
    pp = proto.groupby(["arm", "seed", "fault", "severity"]).agg(cost=("cost", "mean")).reset_index()
    anchors = {}
    for arm in ["S0", "S1"]:
        cl = pp[(pp.arm == arm) & (pp.fault == "none")].cost.mean()
        f13 = pp[(pp.arm == arm) & (pp.fault == "F1") & (pp.severity == 3)].cost.mean()
        f6a = pp[(pp.arm == arm) & (pp.fault == "F6a") & (pp.severity == 3)].cost.mean()
        anchors[arm] = (cl, f13, f6a)

    print(f"{'step':>7} {'clean':>8} {'meanFault':>9} {'worstFault':>10} {'F1s3':>8} {'F6a3':>8}   anchors: S0 clean {anchors['S0'][0]:,.0f} F1s3 {anchors['S0'][1]:,.0f} F6a3 {anchors['S0'][2]:,.0f} | S1 clean {anchors['S1'][0]:,.0f} F1s3 {anchors['S1'][1]:,.0f} F6a3 {anchors['S1'][2]:,.0f}")
    for st in sorted(per.step.unique()):
        sub = per[per.step == st]
        cl = sub[sub.fault == "none"].groupby("seed").cost.mean()
        f13 = sub[(sub.fault == "F1") & (sub.severity == 3)].groupby("seed").cost.mean()
        f6a = sub[(sub.fault == "F6a") & (sub.severity == 3)].groupby("seed").cost.mean()
        faults = sub[sub.fault != "none"].groupby(["seed", "fault", "severity"]).cost.mean()
        per_seed_mean = faults.groupby("seed").mean()
        per_seed_worst = faults.groupby("seed").max()
        print(f"{st / 1000:>6.0f}k {cl.mean():>8,.0f} {per_seed_mean.mean():>9,.0f} "
              f"{per_seed_worst.mean():>10,.0f} {f13.mean():>8,.0f} {f6a.mean():>8,.0f}")


if __name__ == "__main__":
    main()
