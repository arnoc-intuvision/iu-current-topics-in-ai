"""B0 rule-based heuristic floor: peak-window discharge, demand-exposure cap, offpeak charge."""

from __future__ import annotations

import numpy as np


class B0Policy:
    def __init__(self, max_power_kw: float):
        self.max_power_kw = max_power_kw

    def action(self, env) -> float:
        pos = env.pos
        slot = env.slots[pos]
        net_import_kw = env.load_kw[pos] - env.pv_kw[pos]
        charge_head, discharge_head = env.battery.headrooms(env.dt_h)
        peak_ref = max(env._running_peak_kw, env._initial_peak_kw)

        power_kw = 0.0
        if slot == "p" and discharge_head > 0 and net_import_kw > 0:
            power_kw = min(discharge_head, net_import_kw)
        elif net_import_kw > peak_ref > 0 and discharge_head > 0:
            power_kw = min(discharge_head, net_import_kw - peak_ref)
        elif slot == "o" and charge_head > 0 and env.battery.soc < 0.95:
            headroom_to_peak = peak_ref - net_import_kw if peak_ref > 0 else charge_head
            if headroom_to_peak > 0:
                power_kw = -min(charge_head, headroom_to_peak)

        a = float(np.clip(power_kw / self.max_power_kw, -1.0, 1.0))
        return np.array([a], dtype=np.float64)
