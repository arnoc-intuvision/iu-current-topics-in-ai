"""Week-blocked month-stratified splits with 24h embargo, valid episode starts, train-fitted scaler."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..env.config import EnvConfig

EMBARGO_STEPS_DEFAULT = 96


def build_splits(
    canonical: pd.DataFrame,
    embargo_steps: int = EMBARGO_STEPS_DEFAULT,
    seed: int = 0,
) -> pd.DataFrame:
    idx = canonical.index
    week_starts = pd.date_range(
        idx.min().normalize() - pd.Timedelta(days=idx.min().weekday()), idx.max(), freq="W-MON"
    )
    rows = []
    for ws in week_starts:
        we = ws + pd.Timedelta(days=7)
        in_range = (idx >= ws) & (idx < we)
        if in_range.sum() != 672:
            continue
        month_of_week = (ws + pd.Timedelta(days=3)).month
        year_of_week = (ws + pd.Timedelta(days=3)).year
        rows.append({"week_start": ws, "week_end": we, "month": month_of_week, "year": year_of_week})
    weeks = pd.DataFrame(rows)
    rng = np.random.default_rng(seed)
    split_of_week = {}
    for _, wk in weeks.groupby(["year", "month"]):
        candidates = wk.index.to_list()
        held = list(rng.choice(candidates, size=min(2, len(candidates)), replace=False))
        if held:
            split_of_week[held[0]] = "val"
        if len(held) > 1:
            split_of_week[held[1]] = "test"
    weeks["split"] = [split_of_week.get(i, "train") for i in weeks.index]

    n = len(idx)
    held_mask = np.zeros(n, dtype=bool)
    for i, row in weeks.iterrows():
        if row["split"] == "train":
            continue
        lo = idx.searchsorted(row["week_start"])
        hi = idx.searchsorted(row["week_end"])
        held_mask[max(0, lo - embargo_steps) : min(n, hi + embargo_steps)] = True
    weeks["embargoed_train_rows"] = int(held_mask.sum())
    weeks.attrs["held_mask"] = held_mask
    return weeks


def row_split_mask(canonical: pd.DataFrame, weeks: pd.DataFrame, split: str) -> np.ndarray:
    n = len(canonical)
    mask = np.zeros(n, dtype=bool)
    for _, row in weeks.iterrows():
        if row["split"] != split:
            continue
        lo = canonical.index.searchsorted(row["week_start"])
        hi = canonical.index.searchsorted(row["week_end"])
        mask[lo:hi] = True
    if split == "train":
        mask &= ~weeks.attrs["held_mask"]
    return mask


def build_episode_starts(
    canonical: pd.DataFrame,
    weeks: pd.DataFrame,
    episode_steps: int = 672,
    embargo_steps: int = EMBARGO_STEPS_DEFAULT,
) -> pd.DataFrame:
    islanded = canonical["islanded"].to_numpy()
    n = len(canonical)
    idx = canonical.index
    other_split_rows = {}
    for split, other in [("val", "test"), ("test", "val")]:
        m = np.zeros(n, dtype=bool)
        for _, row in weeks.iterrows():
            if row["split"] == other:
                lo = idx.searchsorted(row["week_start"])
                hi = idx.searchsorted(row["week_end"])
                m[lo:hi] = True
        other_split_rows[split] = m

    entries = []
    for split in ["train", "val", "test"]:
        mask = np.zeros(n, dtype=bool)
        for _, row in weeks.iterrows():
            if row["split"] != split:
                continue
            lo = idx.searchsorted(row["week_start"])
            hi = idx.searchsorted(row["week_end"])
            if split == "train":
                mask[lo:hi] = True
            else:
                lo_e = max(0, lo - embargo_steps)
                hi_e = min(n, hi + embargo_steps)
                zone = np.zeros(n, dtype=bool)
                zone[lo_e:hi_e] = True
                zone &= ~other_split_rows[split]
                mask |= zone
        if split == "train":
            mask &= ~weeks.attrs["held_mask"]
        mask &= ~islanded
        cs = np.concatenate(([0], np.cumsum(~mask)))
        ok = np.array(
            [cs[i + episode_steps] - cs[i] == 0 for i in range(n - episode_steps)]
        )
        for start in np.flatnonzero(ok):
            entries.append({"split": split, "start_idx": int(start)})
    return pd.DataFrame(entries)


def fit_scaler(canonical: pd.DataFrame, weeks: pd.DataFrame, config: EnvConfig):
    from ..data.channels import CHG_HEAD_IDX, DIS_HEAD_IDX, EXPOSURE_IDX, SOC_IDX

    train_mask = row_split_mask(canonical, weeks, "train")
    cols = [
        "pm1_kw", "pm0_a_kw", "pm0_b_kw", "irr1_wm2", "irr2_wm2",
        "irr1_temp_c", "irr2_temp_c", "sat_gti_wm2",
        "fcst_pv_t1h_kw", "fcst_pv_t3h_kw", "fcst_pv_t6h_kw", "tariff_zar_kwh",
    ]
    sub = canonical.loc[train_mask, cols]
    lo = sub.min().to_numpy(dtype=float)
    hi = sub.max().to_numpy(dtype=float)
    fixed = {
        EXPOSURE_IDX[0]: (0.0, 1200.0),
        SOC_IDX[0]: (0.0, 1.0),
        CHG_HEAD_IDX[0]: (0.0, config.battery.max_power_kw),
        DIS_HEAD_IDX[0]: (0.0, config.battery.max_power_kw),
    }
    obs_lo = np.concatenate([lo, [fixed[i][0] for i in sorted(fixed)]])
    obs_hi = np.concatenate([hi, [fixed[i][1] for i in sorted(fixed)]])
    return obs_lo, obs_hi


def save_artifacts(
    canonical: pd.DataFrame,
    weeks: pd.DataFrame,
    starts: pd.DataFrame,
    obs_lo: np.ndarray,
    obs_hi: np.ndarray,
    out_dir: Path,
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    weeks.drop(columns=["embargoed_train_rows"]).assign(
        week_start=lambda d: d.week_start.astype(str),
        week_end=lambda d: d.week_end.astype(str),
    ).to_csv(out_dir / "splits.csv", index=False)
    starts.to_parquet(out_dir / "episode_starts.parquet", index=False)
    np.savez(out_dir / "scaler.npz", obs_lo=obs_lo, obs_hi=obs_hi)
