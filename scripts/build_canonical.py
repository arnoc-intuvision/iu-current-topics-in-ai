"""Build canonical parquet, splits, episode starts and scaler from the VCOM export.

Usage: python scripts/build_canonical.py [--data-dir data] [--out-dir data/processed]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import yaml

from sacbess.data.loader import build_canonical
from sacbess.data.splits import build_episode_starts, build_splits, fit_scaler, save_artifacts
from sacbess.env.config import EnvConfig


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=ROOT / "data")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "data" / "processed")
    ap.add_argument("--env-config", type=Path, default=ROOT / "configs" / "env.yaml")
    args = ap.parse_args()

    with open(args.env_config) as f:
        config = EnvConfig.from_dict(yaml.safe_load(f))

    import pandas as pd

    canonical_path = args.out_dir / "canonical.parquet"
    if canonical_path.exists():
        df = pd.read_parquet(canonical_path)
        print(f"canonical exists: {len(df)} rows ({df.index.min()} -> {df.index.max()})")
    else:
        audit = build_canonical(
            args.data_dir / "vcom_data_export.csv",
            args.data_dir / "tou_schedule.csv",
            args.data_dir / "tariff_rates.csv",
            args.out_dir,
        )
        df = pd.read_parquet(canonical_path)
        print(f"canonical built: {audit['index']}")
        print(f"  islanded: {audit['islanded']}")
        print(f"  true series: {audit['true_series']}")

    weeks = build_splits(df, embargo_steps=96, seed=0)
    starts = build_episode_starts(df, weeks, config.episode_steps)
    obs_lo, obs_hi = fit_scaler(df, weeks, config)
    save_artifacts(df, weeks, starts, obs_lo, obs_hi, args.out_dir)

    counts = starts.groupby("split").size().to_dict()
    week_counts = weeks.groupby("split").size().to_dict()
    print(f"splits: {week_counts} weeks; episode starts: {counts}")
    print(f"scaler fitted on train rows only: {len(obs_lo)} channels")
    print(f"artifacts in {args.out_dir}")


if __name__ == "__main__":
    main()
