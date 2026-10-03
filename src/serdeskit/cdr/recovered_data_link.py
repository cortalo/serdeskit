"""RecoveredDataLink: wraps a PhaseAdjustableLink so the data d it reports
is what the receiver recovers itself -- a PrbsSync locked onto the known
PRBS from the link's decisions -- instead of the simulator's transmitted
symbols. d is 0 (unknown) until the sync locks.
"""
from __future__ import annotations

import numpy as np
import numpy.typing as npt

from serdeskit.adapt import Received
from serdeskit.cdr.mueller_muller import PhaseAdjustableLink
from serdeskit.prbs import PrbsSync


class RecoveredDataLink:
    """Satisfies PhaseAdjustableLink implicitly."""

    def __init__(self, link: PhaseAdjustableLink, sync: PrbsSync) -> None:
        self._link = link
        self.sync = sync

    @property
    def sampling_phase(self) -> float:
        return self._link.sampling_phase

    def set_sampling_phase(self, phase: float) -> None:
        self._link.set_sampling_phase(phase)

    def respond(self, symbols: npt.NDArray[np.float64]) -> Received:
        rx = self._link.respond(symbols)
        return Received(r=rx.r, decisions=rx.decisions, d=self.sync.feed(rx.decisions))
