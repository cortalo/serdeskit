"""Prbs: a pseudo-random binary sequence from a two-tap Fibonacci LFSR,

    b[n] = b[n - order] XOR b[n - tap],

so any `order` consecutive bits determine everything after them -- which
is what lets a receiver lock onto it (serdeskit.prbs.PrbsSync).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

# order -> second tap, for the standard polynomials x^order + x^tap + 1
TAPS = {7: 6, 9: 5, 11: 9, 15: 14, 23: 18, 31: 28}


@dataclass(frozen=True)
class Prbs:
    order: int

    def __post_init__(self) -> None:
        if self.order not in TAPS:
            raise ValueError(f"order must be one of {sorted(TAPS)}, got {self.order}.")

    @property
    def period(self) -> int:
        return int(2**self.order - 1)

    def continue_from(self, seed: npt.NDArray[np.int8], n: int) -> npt.NDArray[np.int8]:
        """The n bits that follow `seed`, its last `order` bits (0/1)."""
        if len(seed) < self.order:
            raise ValueError(f"seed needs {self.order} bits, got {len(seed)}.")
        tap = TAPS[self.order]
        bits = np.concatenate([np.asarray(seed[-self.order :], dtype=np.int8), np.zeros(n, dtype=np.int8)])
        for i in range(self.order, len(bits)):
            bits[i] = bits[i - self.order] ^ bits[i - tap]
        return bits[self.order :]

    def bits(self, n: int, start: int = 0) -> npt.NDArray[np.int8]:
        """n bits starting `start` bits into the sequence from the all-ones state."""
        ones = np.ones(self.order, dtype=np.int8)
        return self.continue_from(ones, start + n)[start:]

    def symbols(self, n: int, start: int = 0) -> npt.NDArray[np.float64]:
        """NRZ symbols: bit 1 -> +1, bit 0 -> -1."""
        return np.asarray(2.0 * self.bits(n, start) - 1.0, dtype=np.float64)
