"""Core domain types shared across the link: the signal representation that
flows between stages.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True, slots=True)
class Signal:
    """A real-valued waveform plus enough metadata to place it in absolute
    time. t0 is what samples[0] corresponds to — carrying it explicitly (
    instead of expecting every caller to track array-index offsets by hand)
    is what prevents the kind of index-misalignment bug that silently
    corrupts a BER estimate.
    """

    samples: npt.NDArray[np.float64]
    fs: float  # sample rate, Hz
    t0: float = 0.0  # absolute time of samples[0], seconds

    def __len__(self) -> int:
        return len(self.samples)
