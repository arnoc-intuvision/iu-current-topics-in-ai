"""Per-channel two-state Markov failure process (Rodrigues & Lima Azevedo 2019), fitted to the site audit."""

from __future__ import annotations

import numpy as np


class MarkovChannel:
    def __init__(self, p_onset: float, med_steps: float, sig_steps: float, cap_steps: float):
        self.p_onset = p_onset
        self.med_steps = med_steps
        self.sig_steps = sig_steps
        self.cap_steps = cap_steps
        self._remaining = 0

    def reset(self) -> None:
        self._remaining = 0

    def step(self, rng: np.random.Generator) -> bool:
        if self._remaining > 0:
            self._remaining -= 1
            return True
        if rng.random() < self.p_onset:
            duration = rng.lognormal(np.log(self.med_steps), self.sig_steps)
            self._remaining = int(min(np.ceil(duration), self.cap_steps)) - 1
            return True
        return False

    @property
    def faulted(self) -> bool:
        return self._remaining > 0
