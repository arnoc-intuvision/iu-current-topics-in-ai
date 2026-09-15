import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

from sacbess.env.battery import Battery
from sacbess.env.config import BatteryConfig


def test_soc_conservation_under_random_actions():
    cfg = BatteryConfig(
        capacity_kwh=100.0, max_power_kw=50.0, round_trip_efficiency=0.81,
        reserve_frac=0.2, min_soc=0.2, max_soc=1.0,
    )
    eta = cfg.one_way_efficiency
    bat = Battery(cfg)
    bat.reset(0.6)
    rng = np.random.default_rng(0)
    for _ in range(5000):
        p = rng.uniform(-80, 80)
        prev_kwh = bat.soc * cfg.capacity_kwh
        _applied, chg, dis = bat.step(p, 0.25)
        expected = prev_kwh + chg * 0.25 * eta - dis * 0.25 / eta
        assert abs(bat.soc * cfg.capacity_kwh - expected) < 1e-9
        assert cfg.min_soc - 1e-12 <= bat.soc <= cfg.max_soc + 1e-12


def test_headrooms_respect_bounds():
    cfg = BatteryConfig(capacity_kwh=400.0, max_power_kw=250.0, min_soc=0.2, max_soc=1.0)
    bat = Battery(cfg)
    bat.reset(0.2)
    chg, dis = bat.headrooms()
    assert dis == 0.0 and 0 < chg <= cfg.max_power_kw
    bat.reset(1.0)
    chg, dis = bat.headrooms()
    assert chg == 0.0 and 0 < dis <= cfg.max_power_kw
