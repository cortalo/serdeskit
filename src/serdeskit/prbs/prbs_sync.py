"""PrbsSync: a receiver locking onto a known PRBS from its own decisions,
then generating the error-free reference d[n] locally.

    seed    the `order` decisions starting at some bit become the LFSR state
    verify  predict the decisions that follow; locked once `verify_bits` of
            them are checked with at most `threshold` disagreeing (a wrong
            seed disagrees ~50% of the time, a right one only where the
            decisions themselves are wrong)
    retry   on too many disagreements, the next seed starts one bit later,
            re-checked against the buffered decisions before new ones
    locked  free-run the LFSR: the reference no longer depends on the
            decisions, so it stays error-free with a closed eye

Seeds advance one bit at a time so every start position gets tried. (Taking
each new seed from the latest decisions instead can cycle through the same
few positions forever when the decision errors are periodic, as with a
short PRBS and no noise.)

The lock may be one or more bits off the transmitter's labelling of the
data -- the receiver cannot tell, and need not. Line polarity inversion is
not handled.

Long PRBSs can false-lock at high decision error rates: a seed with one
wrong bit predicts a sequence that diverges from the truth only as the error
propagates through the taps, `order` bits per step. Over 256 bits a
one-bit-off PRBS31 seed disagrees just ~12% of the time (PRBS15 ~32%,
PRBS7 50%), indistinguishable from decision errors. Use a short PRBS, or a
verification window many times the order, when the eye is closed.
"""
from __future__ import annotations

import numpy as np
import numpy.typing as npt

from serdeskit.prbs.prbs import TAPS, Prbs


class PrbsSync:
    """Streams: feed() takes decisions in any block sizes."""

    def __init__(self, prbs: Prbs, verify_bits: int = 256, threshold: float = 0.3) -> None:
        self._order = prbs.order
        self._tap = TAPS[prbs.order]
        self._verify_bits = verify_bits
        self._max_errors = threshold * verify_bits
        self._buffer: list[int] = []  # decision bits from the current seed's start
        self._lfsr: list[int] = []  # last `order` bits of the predicted sequence
        self._pos = 0  # next buffered bit to check against the prediction
        self._errors = 0
        self.attempts = 0  # seeds tried
        self.locked_at: int | None = None  # bit index where verification passed
        self._n = 0  # bits fed so far

    @property
    def locked(self) -> bool:
        return self.locked_at is not None

    def _predict(self) -> int:
        bit = self._lfsr[-self._order] ^ self._lfsr[-self._tap]
        self._lfsr = self._lfsr[1:] + [bit]
        return bit

    def _seed(self) -> None:
        self._lfsr = self._buffer[: self._order]
        self._pos = self._order
        self._errors = 0
        self.attempts += 1

    def _verify(self) -> None:
        """Check buffered decisions against the prediction, sliding the seed
        one bit on failure, until the buffer is used up or the seed passes.
        """
        while len(self._buffer) >= self._order:
            if not self._lfsr:
                self._seed()
            while self._pos < len(self._buffer):
                self._errors += self._predict() != self._buffer[self._pos]
                self._pos += 1
                if self._errors > self._max_errors:
                    break
                if self._pos - self._order == self._verify_bits:
                    while self._pos < len(self._buffer):  # catch the LFSR up
                        self._predict()
                        self._pos += 1
                    self.locked_at = self._n
                    return
            else:
                return  # buffer used up, seed still in the running
            self._buffer.pop(0)  # failed: next seed starts one bit later
            self._lfsr = []

    def feed(self, decisions: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Per decision (+/-1): the reference symbol once locked, 0 before."""
        reference = np.zeros(len(decisions))
        for i, decision in enumerate(decisions):
            if self.locked:
                reference[i] = 2.0 * self._predict() - 1.0
            else:
                self._buffer.append(1 if decision > 0 else 0)
                self._verify()
            self._n += 1
        return reference
