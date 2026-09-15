"""Shared fixtures: a small synthetic canonical dataset built through the real pipeline."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
import pytest

from sacbess.data.loader import month_to_date_peak, tariff_series
from sacbess.data.splits import build_episode_starts, build_splits, fit_scaler, save_artifacts
from sacbess.env.config import EnvConfig


def make_toy_canonical(n_days: int = 84) -> pd.DataFrame:
    idx = pd.date_range("2025-01-06 00:15", periods=n_days * 96, freq="15min")
    rng = np.random.default_rng(7)
    hod = idx.hour + idx.minute / 60.0
    load_kw = 600 + 150 * np.sin(2 * np.pi * (hod - 8) / 24.0) + rng.normal(0, 20, len(idx))
    pv_shape = np.clip(np.sin(np.pi * (hod - 6) / 12.0), 0, None)
    season = 0.85 + 0.1 * np.sin(2 * np.pi * idx.dayofyear / 365.25)
    pv_base = 700 * pv_shape * season
    pv_kw = np.clip(pv_base + rng.normal(0, 5, len(idx)), 0, None)
    irr = 1.3 * pv_base + rng.normal(0, 5, len(idx))
    irr = np.clip(irr, 0, None)
    temp = 25 + 10 * pv_shape + rng.normal(0, 0.5, len(idx))
    sat = 0.97 * irr * (1 + 0.01 * rng.normal(0, 1, len(idx)))
    sat = np.clip(sat, 0, None)
    islanded = np.zeros(len(idx), dtype=bool)
    islanded[500:520] = True
    islanded[3000:3010] = True
    zar, slots = tariff_series(idx, ROOT / "data" / "tou_schedule.csv", ROOT / "data" / "tariff_rates.csv")
    mtd = month_to_date_peak(idx, load_kw)
    grid_a = np.maximum(load_kw - pv_kw, 0) * 0.6
    grid_b = np.maximum(load_kw - pv_kw, 0) * 0.4
    df = pd.DataFrame(
        {
            "pm1_kw": pv_kw,
            "pm0_a_kw": grid_a,
            "pm0_b_kw": grid_b,
            "irr1_wm2": irr,
            "irr2_wm2": irr * 1.02,
            "irr1_temp_c": temp,
            "irr2_temp_c": temp + 0.2,
            "sat_gti_wm2": sat,
            "fcst_pv_t1h_kw": np.roll(pv_kw, -4),
            "fcst_pv_t3h_kw": np.roll(pv_kw, -12),
            "fcst_pv_t6h_kw": np.roll(pv_kw, -24),
            "load_kw_true": load_kw,
            "pv_kw_true": pv_kw,
            "tariff_zar_kwh": zar,
            "tou_slot": slots,
            "islanded": islanded,
            "month_to_date_peak_kw": mtd,
        },
        index=idx,
    )
    df.index.name = "ts_sast"
    return df


@pytest.fixture(scope="session")
def processed_dir(tmp_path_factory):
    out = tmp_path_factory.mktemp("processed")
    df = make_toy_canonical()
    df.to_parquet(out / "canonical.parquet")
    weeks = build_splits(df, embargo_steps=96, seed=0)
    starts = build_episode_starts(df, weeks, 672)
    config = EnvConfig()
    obs_lo, obs_hi = fit_scaler(df, weeks, config)
    save_artifacts(df, weeks, starts, obs_lo, obs_hi, out)
    return out
