"""BESS model: SoC bookkeeping, one-way efficiencies, C-rate bounds, backup reserve floor."""

from __future__ import annotations

from .config import BatteryConfig


class Battery:
    def __init__(self, config: BatteryConfig):
        self.cfg = config
        self.soc = config.min_soc

    def reset(self, soc: float) -> None:
        self.soc = float(min(max(soc, self.cfg.min_soc), self.cfg.max_soc))

    def headrooms(self, dt_h: float = 0.25) -> tuple[float, float]:
        c = self.cfg
        charge_kw = min(
            c.max_power_kw,
            (c.capacity_kwh * (c.max_soc - self.soc)) / (dt_h * c.one_way_efficiency),
        )
        discharge_kw = min(
            c.max_power_kw,
            (c.capacity_kwh * (self.soc - c.min_soc)) * c.one_way_efficiency / dt_h,
        )
        return max(charge_kw, 0.0), max(discharge_kw, 0.0)

    def step(self, power_kw: float, dt_h: float = 0.25) -> tuple[float, float, float]:
        """power_kw > 0 discharges, < 0 charges. Returns (applied_kw, charge_kw, discharge_kw)."""
        c = self.cfg
        charge_head, discharge_head = self.headrooms(dt_h)
        applied = max(-charge_head, min(discharge_head, power_kw))
        if applied < 0:
            energy_in_kwh = -applied * dt_h
            self.soc += energy_in_kwh * c.one_way_efficiency / c.capacity_kwh
        else:
            energy_out_kwh = applied * dt_h
            self.soc -= energy_out_kwh / c.one_way_efficiency / c.capacity_kwh
        self.soc = min(max(self.soc, c.min_soc), c.max_soc)
        charge_kw = max(-applied, 0.0)
        discharge_kw = max(applied, 0.0)
        return applied, charge_kw, discharge_kw
