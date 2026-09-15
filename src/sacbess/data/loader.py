"""VCOM CSV export -> canonical parquet (SAST 15-min index, cleaned registers, causal forecast block)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .channels import RAW_CHANNEL_BOUNDS

REGISTER_COLUMNS = {
    "pm0_a": "pm0_a_grid_active_energy_import",
    "pm0_b": "pm0_b_grid_active_energy_import",
    "pm1": "pm1_solar_active_energy_export",
}

IRRADIANCE_COLUMNS = [
    "irr1_sensor_poa_irradiance",
    "irr2_sensor_poa_irradiance",
    "vcom_satellite_poa_irradiance",
]
TEMP_COLUMNS = [
    "irr1_sensor_temperature",
    "irr2_sensor_temperature",
]

DT_H = 0.25
FCST_HORIZONS_STEPS = [4, 12, 24]
FCST_QUANTILE = 0.90
FCST_MIN_SAMPLES = 4
DAYLIGHT_THRESHOLD_WM2 = 50.0


def _months(spec: str) -> list[int]:
    name_to_num = {
        "Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4, "May": 5, "Jun": 6,
        "Jul": 7, "Aug": 8, "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12,
    }
    parts = spec.split("-")
    if len(parts) == 2:
        a, b = name_to_num[parts[0]], name_to_num[parts[1]]
        if a <= b:
            return list(range(a, b + 1))
        return list(range(a, 13)) + list(range(1, b + 1))
    return [name_to_num[parts[0]]]


def load_tariff(tou_csv: Path, rates_csv: Path):
    tou = pd.read_csv(tou_csv)
    rates = pd.read_csv(rates_csv)
    rate_map = {
        (r.season, r.tou_slot): float(r.rate_c_per_kwh) / 100.0
        for r in rates.itertuples()
    }
    season_map = {}
    for r in rates.itertuples():
        for m in _months(r.months):
            season_map[m] = r.season
    rows = []
    for t in tou.itertuples():
        for h in range(t.hour_start, t.hour_end):
            rows.append({"day_type": t.day_type, "hour": h, "slot": t.tou_time_slot})
    hour_slot = pd.DataFrame(rows).drop_duplicates(["day_type", "hour"])
    slot_lookup = {(r.day_type, r.hour): r.slot for r in hour_slot.itertuples()}
    return rate_map, season_map, slot_lookup


def tariff_series(index: pd.DatetimeIndex, tou_csv: Path, rates_csv: Path):
    rate_map, season_map, slot_lookup = load_tariff(tou_csv, rates_csv)
    dow = index.dayofweek.to_numpy()
    day_type = np.where(dow <= 4, "weekday", np.where(dow == 5, "saturday", "sunday"))
    slots = np.array([slot_lookup[(dt, h)] for dt, h in zip(day_type, index.hour)])
    zar = np.array(
        [rate_map[(season_map[m], s)] for m, s in zip(index.month, slots)]
    )
    return zar, slots


def clean_registers(df: pd.DataFrame):
    out = {}
    audit = {}
    for name, col in REGISTER_COLUMNS.items():
        raw = df[col].astype(float)
        bound = RAW_CHANNEL_BOUNDS[name]
        bad = (raw > bound) | (raw < 0) | raw.isna()
        audit[name] = {
            "register": col,
            "interval_kwh_bound": bound,
            "null_intervals": int(raw.isna().sum()),
            "spike_or_negative_intervals": int(((raw > bound) | (raw < 0)).sum()),
            "max_clean_kwh_per_interval": float(raw.where(~bad).max()),
        }
        cleaned = raw.where(~bad).interpolate(method="time").bfill().ffill()
        out[name] = cleaned / DT_H
    for col in IRRADIANCE_COLUMNS:
        s = df[col].astype(float)
        s = s.where((s >= 0) & (s <= 1500))
        audit[col] = {"out_of_range_or_null": int(s.isna().sum() - df[col].isna().sum())}
        out[col] = s.interpolate(method="time").bfill().ffill()
    for col in TEMP_COLUMNS:
        s = df[col].astype(float)
        s = s.where((s >= -40) & (s <= 100))
        out[col] = s.interpolate(method="time").bfill().ffill()
    return pd.DataFrame(out), audit


def forecast_block(index: pd.DatetimeIndex, pm1_kw: np.ndarray, sat: np.ndarray):
    n = len(index)
    day_id = (index.normalize() - index.normalize()[0]).days.to_numpy()
    n_days = int(day_id[-1]) + 1
    daylight = sat > DAYLIGHT_THRESHOLD_WM2
    pv_cum = np.concatenate(([0.0], np.cumsum(np.where(daylight, pm1_kw, 0.0))))
    sat_cum = np.concatenate(([0.0], np.cumsum(np.where(daylight, sat, 0.0))))
    day_last_row = np.zeros(n_days, dtype=int)
    for d in range(n_days):
        day_last_row[d] = np.searchsorted(day_id, d, side="right") - 1
    kpv_day = np.zeros(n_days + 1)
    for d in range(1, n_days + 1):
        end = day_last_row[d - 1] + 1
        kpv_day[d] = pv_cum[end] / sat_cum[end] if sat_cum[end] > 0 else 0.0

    month = index.month.to_numpy()
    hour = index.hour.to_numpy()
    gkey = (month - 1) * 24 + hour
    groups = {}
    for g in np.unique(gkey):
        rows = np.flatnonzero(gkey == g)
        vals = sat[rows]
        q_inc = []
        for j in range(len(rows)):
            q_inc.append(np.quantile(vals[: j + 1], FCST_QUANTILE))
        groups[g] = (day_id[rows], np.array(q_inc))

    fcst = np.zeros((n, len(FCST_HORIZONS_STEPS)))
    for col, h in enumerate(FCST_HORIZONS_STEPS):
        tgt = np.minimum(np.arange(n) + h, n - 1)
        gk = gkey[tgt]
        for i in range(n):
            d_hist = day_id[i]
            gdays, gq = groups[gk[i]]
            c = int(np.searchsorted(gdays, d_hist, side="left"))
            if c >= FCST_MIN_SAMPLES:
                q = gq[c - 1]
            else:
                c_all = int(np.searchsorted(day_id, d_hist, side="left"))
                q = (
                    np.quantile(sat[:c_all], FCST_QUANTILE)
                    if c_all >= 24
                    else 0.0
                )
            fcst[i, col] = q * kpv_day[d_hist]
    return fcst


def month_to_date_peak(index: pd.DatetimeIndex, load_kw: np.ndarray) -> np.ndarray:
    n = len(index)
    window_kw = np.zeros(n)
    s = (load_kw[0:-1:2] + load_kw[1::2]) / 2.0
    window_kw[0:-1:2] = s
    window_kw[1::2] = s
    peak = np.zeros(n)
    ym = index.year.to_numpy() * 100 + index.month.to_numpy()
    for m in np.unique(ym):
        sel = np.flatnonzero(ym == m)
        running = np.maximum.accumulate(window_kw[sel])
        peak[sel[0]] = 0.0
        peak[sel[1:]] = running[:-1]
    return peak


def _longest_run_hours(flags: np.ndarray) -> float:
    if not flags.any():
        return 0.0
    padded = np.concatenate(([0], flags.astype(int), [0]))
    changes = np.flatnonzero(np.diff(padded))
    starts, ends = changes[0::2], changes[1::2]
    return float((ends - starts).max() * DT_H)


def build_canonical(
    vcom_csv: Path, tou_csv: Path, rates_csv: Path, out_dir: Path
) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(vcom_csv, parse_dates=["entry_time"]).set_index("entry_time").sort_index()
    full = pd.date_range(df.index.min(), df.index.max(), freq="15min")
    assert len(full) == len(df) and not df.index.duplicated().any(), "index not 15-min complete"

    registers, audit = clean_registers(df)
    pm1_kw = registers["pm1"].to_numpy()
    pm0_a_kw = registers["pm0_a"].to_numpy()
    pm0_b_kw = registers["pm0_b"].to_numpy()
    load_kw = pm0_a_kw + pm0_b_kw + pm1_kw
    sat = registers["vcom_satellite_poa_irradiance"].to_numpy()
    fcst = forecast_block(df.index, pm1_kw, sat)
    zar, slots = tariff_series(df.index, Path(tou_csv), Path(rates_csv))
    islanded = (df["ppc_generator_run_status"].fillna(0.0) > 0).to_numpy().astype(bool)
    mtd = month_to_date_peak(df.index, load_kw)

    canonical = pd.DataFrame(
        {
            "pm1_kw": pm1_kw,
            "pm0_a_kw": pm0_a_kw,
            "pm0_b_kw": pm0_b_kw,
            "irr1_wm2": registers["irr1_sensor_poa_irradiance"].to_numpy(),
            "irr2_wm2": registers["irr2_sensor_poa_irradiance"].to_numpy(),
            "irr1_temp_c": registers["irr1_sensor_temperature"].to_numpy(),
            "irr2_temp_c": registers["irr2_sensor_temperature"].to_numpy(),
            "sat_gti_wm2": sat,
            "fcst_pv_t1h_kw": fcst[:, 0],
            "fcst_pv_t3h_kw": fcst[:, 1],
            "fcst_pv_t6h_kw": fcst[:, 2],
            "load_kw_true": load_kw,
            "pv_kw_true": pm1_kw,
            "tariff_zar_kwh": zar,
            "tou_slot": slots,
            "islanded": islanded,
            "month_to_date_peak_kw": mtd,
        },
        index=df.index,
    )
    canonical.index.name = "ts_sast"
    canonical.to_parquet(out_dir / "canonical.parquet")

    audit["index"] = {
        "start": str(df.index.min()),
        "end": str(df.index.max()),
        "rows": len(df),
        "resolution_min": 15,
        "timezone": "SAST (UTC+2, no DST); solar-noon check: satellite POA peaks hour 12",
    }
    audit["islanded"] = {
        "steps": int(islanded.sum()),
        "runs": int((islanded & ~np.concatenate(([False], islanded[:-1]))).sum()),
        "longest_hours": _longest_run_hours(islanded),
    }
    audit["true_series"] = {
        "load_kw": {
            "mean": float(load_kw.mean()),
            "max": float(load_kw.max()),
            "max_30min_demand_kw": float(np.max((load_kw[:-1:2] + load_kw[1::2]) / 2.0)),
        },
        "pv_kw": {"mean": float(pm1_kw.mean()), "max": float(pm1_kw.max())},
    }
    with open(out_dir / "audit_summary.json", "w") as f:
        json.dump(audit, f, indent=2)
    return audit
