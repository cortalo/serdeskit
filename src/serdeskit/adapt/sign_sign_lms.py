"""SignSignLms: the dual-loop TX FFE adaptation of Stojanovic, JSSC 2005,
eqs. (1)-(2), plus the RX DFE taps, all adapted together. With the paper's
sign convention e = dLev - y (y the DFE summer output):

    w_j     <- w_j     + step_tap  * sign(e) * d[n-j]   (TX, j != 0)
    alpha_k <- alpha_k - step_dfe  * sign(e) * d[n-k]   (DFE subtracts, hence -)
    dLev    <- dLev    - step_dlev * sign(e)

Each of these drives one cursor: TX w_j drives p_j to 0, alpha_k tracks
p_k, dLev tracks p_0. A TX post-tap at a cursor the DFE also covers would
share alpha_k's equation and leave the split between them undetermined, so
TX taps at offsets 1..n_dfe are held at 0 -- the DFE takes those cursors
without spending TX swing.

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
    @property
    def dfe_taps(self) -> npt.NDArray[np.float64]: ...  # alpha_1..alpha_N, volts
    def respond(self, symbols: npt.NDArray[np.float64]) -> Received: ...
    def set_tap_weights(self, tap_weights: npt.NDArray[np.float64]) -> None: ...
    def set_dfe_taps(self, taps: npt.NDArray[np.float64]) -> None: ...


@dataclass(frozen=True, slots=True)
class AdaptationTrace:
    """Taps and dLev before the first iteration and after each one: row i
    is iteration i, so both have n_iterations + 1 rows.
    """

    tap_weights: npt.NDArray[np.float64]  # (n+1, n_taps), main included, link's ordering
    dfe_taps: npt.NDArray[np.float64]  # (n+1, n_dfe)
    dlev: npt.NDArray[np.float64]  # (n+1,)


@dataclass(frozen=True)
class SignSignLms:
    step_tap: float
    step_dlev: float
    step_dfe: float = 1e-3
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
        n_dfe = len(link.dfe_taps)
        # TX taps the DFE covers (offsets 1..n_dfe) are held at 0.
        frozen = {main} | {i for i in range(n_taps) if 1 <= i - main <= n_dfe}

        taps, dfe, dlev = [link.tap_weights], [link.dfe_taps], [0.0]
        for block in data:
            rx = link.respond(block)

            # Up/down counters: one per TX tap, one per DFE tap, one for dLev.
            tap_votes = np.zeros(n_taps)  # frozen taps' are computed but unused
            dfe_votes = np.zeros(n_dfe)
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
                for k in range(n_dfe):
                    if n - k - 1 >= 0:  # alpha_{k+1} weights d[n - (k+1)]
                        dfe_votes[k] += s * rx.d[n - k - 1]

            new_taps = link.tap_weights.copy()
            for i in range(n_taps):
                if i not in frozen:
                    new_taps[i] += self.step_tap * np.sign(tap_votes[i])
                elif i != main:
                    new_taps[i] = 0.0
            # TX peak-swing constraint
            new_taps[main] = 1.0 - np.abs(np.delete(new_taps, main)).sum()
            link.set_tap_weights(new_taps)
            taps.append(link.tap_weights)
            link.set_dfe_taps(link.dfe_taps - self.step_dfe * np.sign(dfe_votes))
            dfe.append(link.dfe_taps)
            dlev.append(dlev[-1] - self.step_dlev * float(np.sign(dlev_vote)))

        return AdaptationTrace(
            tap_weights=np.array(taps), dfe_taps=np.array(dfe).reshape(len(dfe), n_dfe), dlev=np.array(dlev)
        )
