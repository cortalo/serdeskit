"""MuellerMuller: the type-A baud-rate timing loop of Mueller & Müller, IEEE
Trans. Commun. 1976, in its linear form with known data d:

    z[n] = r[n] d[n-1] - r[n-1] d[n],    E{z} = h_1 - h_-1
    phase <- phase + gain * mean(z)

h_1 > h_-1 means sampling early, so a positive z moves the phase later.
The loop locks where h_1 = h_-1.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import numpy.typing as npt

from serdeskit.adapt import Received


class PhaseAdjustableLink(Protocol):
    """What the timing loop can see of a link: received samples with the
    data each one carries, and the sampling phase it may move.
    """

    @property
    def sampling_phase(self) -> float: ...  # UI
    def set_sampling_phase(self, phase: float) -> None: ...
    def respond(self, symbols: npt.NDArray[np.float64]) -> Received: ...


@dataclass(frozen=True)
class MuellerMuller:
    gain: float  # UI per volt of block-averaged z

    def run(
        self,
        link: PhaseAdjustableLink,
        data: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Args:
            data: +/-1 symbols, shape (n_iterations, block) -- one block
                per iteration, sent back to back as one stream.

        Returns:
            The sampling phase before the first iteration and after each
            one, shape (n_iterations + 1,).
        """
        phases = [link.sampling_phase]
        for block in data:
            rx = link.respond(block)

            z = 0.0
            for n in range(1, len(rx.d)):
                z += rx.r[n] * rx.d[n - 1] - rx.r[n - 1] * rx.d[n]
            z /= len(rx.d) - 1

            link.set_sampling_phase(link.sampling_phase + self.gain * z)
            phases.append(link.sampling_phase)
        return np.array(phases)
