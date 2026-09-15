"""Environment configuration (mirrors configs/env.yaml)."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class BatteryConfig:
    capacity_kwh: float = 500.0
    max_power_kw: float = 250.0
    round_trip_efficiency: float = 0.90
    reserve_frac: float = 0.2
    min_soc: float = 0.2
    max_soc: float = 1.0

    @property
    def one_way_efficiency(self) -> float:
        return self.round_trip_efficiency ** 0.5


@dataclass
class EnvConfig:
    battery: BatteryConfig = field(default_factory=BatteryConfig)
    dt_h: float = 0.25
    episode_steps: int = 672
    demand_charge_rate_zar_per_kw: float = 150.0
    export_credit_zar_per_kwh: float = 0.0
    reward_scale: float = 1e-2
    initial_soc_range: tuple[float, float] = (0.4, 1.0)
    # Charge guard: clamp charging so the 30-min window import cannot exceed the
    # running peak (oracle: never beneficial to exceed on this site). Off by default
    # to preserve the legacy reward/dynamics for exact-replication runs.
    charge_guard: bool = False
    guard_margin_kw: float = 5.0
    # Potential-based shaping on stored-energy value: F = gamma*phi(s') - phi(s),
    # phi = stored_kWh * lambda. Policy-invariant (does not change the optimum).
    shaping_lambda_zar_per_kwh: float = 0.0
    shaping_gamma: float = 0.99

    @classmethod
    def from_dict(cls, d: dict) -> EnvConfig:
        d = dict(d)
        bat = BatteryConfig(**d.pop("battery", {}))
        if "initial_soc_range" in d and isinstance(d["initial_soc_range"], list):
            d["initial_soc_range"] = tuple(d["initial_soc_range"])
        return cls(battery=bat, **d)
