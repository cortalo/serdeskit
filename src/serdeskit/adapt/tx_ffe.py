"""TxFfe: a symbol-spaced TX FFE -- all tap weights, main cursor included,
plus which one is the main cursor. Plain tap data; SymbolRateLink applies it.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True)
class TxFfe:
    # Ordered by UI offset: weights[i] multiplies d[n - (i - main)].
    weights: npt.NDArray[np.float64]
    main: int  # index of the main-cursor tap in `weights`

    def __post_init__(self) -> None:
        if not 0 <= self.main < len(self.weights):
            raise ValueError(
                f"main must index into weights (length {len(self.weights)}), got {self.main}."
            )
        self.weights.setflags(write=False)

    def with_weights(self, weights: npt.NDArray[np.float64]) -> TxFfe:
        return TxFfe(weights=weights, main=self.main)
