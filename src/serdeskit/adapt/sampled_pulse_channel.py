"""SampledPulseChannel: a channel known only by its UI-spaced pulse-response
samples (cursors) on the sampling phase -- the symbol-rate view used by
adaptation loops, where nothing between sampling instants matters.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True)
class SampledPulseChannel:
    """`cursors[n_pre]` is the main cursor a_0; entries before it are
    pre-cursors, after it post-cursors.
    """

    cursors: npt.NDArray[np.float64]
    n_pre: int

    def __post_init__(self) -> None:
        if not 0 <= self.n_pre < len(self.cursors):
            raise ValueError(
                f"n_pre must index into cursors (length {len(self.cursors)}), got {self.n_pre}."
            )
        self.cursors.setflags(write=False)
