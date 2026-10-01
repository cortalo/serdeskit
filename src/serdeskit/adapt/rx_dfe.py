"""RxDfe: a receiver decision-feedback equalizer's taps, in volts --
taps[k - 1] times the decision made k UI ago is subtracted from the
received sample. Plain tap data; SymbolRateLink applies it.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True)
class RxDfe:
    taps: npt.NDArray[np.float64]

    def __post_init__(self) -> None:
        self.taps.setflags(write=False)

    def with_taps(self, taps: npt.NDArray[np.float64]) -> RxDfe:
        return RxDfe(taps=taps)
