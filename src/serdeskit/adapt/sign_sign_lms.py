"""SignSignLms: the dual-loop TX FFE adaptation of Stojanovic, JSSC 2005,
eqs. (1)-(2), with the paper's sign convention e = dLev - r:

    w_j  <- w_j  + step_tap  * sign(e) * d[n-j]     (j != 0)
    dLev <- dLev - step_dlev * sign(e)

The main tap is not updated from its own error: it is computed from the
others so that sum(|w|) = 1, the TX peak-swing constraint -- which pins the
overall scale the two loops would otherwise drift along together.

One adaptive sampler sits at +dLev, so only symbols with d[n] = +1 produce
updates. Each iteration feeds the link one block and takes a single
majority-vote step, like an up/down counter would.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
import numpy.typing as npt

from serdeskit.adapt.symbol_link import Received


class AdaptableLink(Protocol):
    """What the adaptation engine can see of a link: received samples with
    the data each one carries, and the TX taps it may change via the back
    channel.
    """

    @property
    def tap_weights(self) -> npt.NDArray[np.float64]: ...  # all taps, main included
    @property
    def main_tap(self) -> int: ...  # index of the main cursor in tap_weights
    def respond(self, symbols: npt.NDArray[np.float64]) -> Received: ...
    def set_tap_weights(self, tap_weights: npt.NDArray[np.float64]) -> None: ...


@dataclass(frozen=True, slots=True)
class AdaptationTrace:
    """Taps and dLev before the first iteration and after each one: row i
    is iteration i, so both have n_iterations + 1 rows.
    """

    tap_weights: npt.NDArray[np.float64]  # (n+1, n_taps), main included, link's ordering
    dlev: npt.NDArray[np.float64]  # (n+1,)


@dataclass(frozen=True)
class SignSignLms:
    step_tap: float
    step_dlev: float
    # Input-referred noise of the adaptive sampler, volts. Not just a
    # non-ideality: without it the residual ISI takes few discrete values and
    # sign(e) leaves a dead zone around the zero-forcing point.
    noise_rms: float = 0.0
    rng: np.random.Generator = field(default_factory=np.random.default_rng)

    def run(
        self,
        link: AdaptableLink,
        data: npt.NDArray[np.float64],
    ) -> AdaptationTrace:
        """Args:
            data: +/-1 symbols, shape (n_iterations, block) -- one block
                per iteration, sent back to back as one stream.
        """
        main = link.main_tap
        n_taps = len(link.tap_weights)

        taps, dlev = [link.tap_weights], [0.0]
        for block in data:
            rx = link.respond(block)

            # Up/down counters, one per tap plus one for dLev.
            tap_votes = np.zeros(n_taps)  # the main tap's is computed but unused
            dlev_vote = 0
            for n in range(len(rx.d)):
                if rx.d[n] <= 0:
                    continue  # data filtering: sampler sits at +dLev
                r = rx.r[n] + self.rng.normal(0.0, self.noise_rms)
                s = 1 if dlev[-1] > r else -1  # adaptive sampler: sign(e) = sign(dLev - r)
                dlev_vote += s
                for k in range(n_taps):
                    if 0 <= n + main - k < len(rx.d):  # outside this block: skip
                        tap_votes[k] += s * rx.d[n + main - k]

            new_taps = link.tap_weights.copy()
            for i in range(n_taps):
                if i != main:
                    new_taps[i] += self.step_tap * np.sign(tap_votes[i])
            # TX peak-swing constraint
            new_taps[main] = 1.0 - np.abs(np.delete(new_taps, main)).sum()
            link.set_tap_weights(new_taps)
            taps.append(link.tap_weights)
            dlev.append(dlev[-1] - self.step_dlev * float(np.sign(dlev_vote)))

        return AdaptationTrace(tap_weights=np.array(taps), dlev=np.array(dlev))
