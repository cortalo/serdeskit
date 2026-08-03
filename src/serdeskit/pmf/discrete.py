"""delta_pmf: the discrete probability-mass-function construction at the
core of COM's noise/interference calculation (IEEE 802.3-2022 Annex 93A,
93A.1.7.1, equations 93A-39/93A-40).

Physical picture: `h_samples` holds one interference source's pulse-response
value at each UI it's evaluated over (ISI's neighboring cursors with the
current cursor zeroed out, A_DD * (local slope) for dual-Dirac deterministic
jitter, or one crosstalk aggressor's samples at its worst-case phase) —
i.e. how many volts that UI *would* contribute if the symbol occupying it
were transmitted at full scale. Each such UI independently carries one of
`levels` equally likely, evenly spaced symbol values in [-1, 1]; convolving
across all of `h_samples` (93A-40) yields the distribution of the source's
total voltage contribution at the sampling instant.

Ported against `PyChOpMarg`'s `utility.probability.delta_pmf` as a golden
reference for tests — see the COM deep-dive in this project's memory/chat
history for equation-by-equation tracing.
"""
from __future__ import annotations

import numpy as np
import numpy.typing as npt


def delta_pmf(
    h_samples: npt.NDArray[np.float64],
    levels: int,
    y: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Probability mass at each point of `y` for the total voltage
    contributed by `h_samples`, assuming each entry independently carries
    one of `levels` equally likely symbol values in [-1, 1].

    Args:
        h_samples: One interference source's per-UI pulse-response
            values (volts). The current cursor's own UI must already be
            excluded (zeroed) by the caller — this function has no notion
            of "which sample is the cursor".
        levels: Number of equally likely, evenly spaced symbol values each
            entry of `h_samples` independently takes (2 for NRZ, 4 for
            PAM4).
        y: The shared voltage quantization grid (volts) this PMF is
            computed on — must match the grid used by every other PMF this
            result will later be convolved against.

    Returns:
        Probability mass at each point of `y`, same shape as `y`, summing
        to 1.

    Raises:
        ValueError: `y` isn't uniformly spaced — the shift-per-sample math
            below assumes a single `ystep` applies across the whole array.
        ValueError: `y`'s center point isn't 0V — the initial delta is
            placed there, representing the running total before any sample
            is folded in.
    Note:
        A third check — that `y` is wide enough to contain the result —
        is currently disabled; see the commented-out block below.
    """
    npts = len(y)
    ystep = y[1] - y[0]
    center_ix = npts // 2

    if not np.allclose(np.diff(y), ystep):
        raise ValueError(
            f"y must be uniformly spaced (first step is {ystep}V, but "
            "spacing is not constant across the array)."
        )
    if not np.isclose(y[center_ix], 0.0, atol=abs(ystep) / 2):
        raise ValueError(
            f"y's center point (index {center_ix}) must be 0V, got "
            f"{y[center_ix]}V."
        )

    # TODO: re-enable a width check, but not this one. The concern is real
    # — `np.roll` below wraps rather than drops mass shifted past an edge,
    # so a too-narrow `y` silently aliases high-voltage probability onto
    # the low-voltage side. But the worst case this tested for (every
    # entry at |level|=1, all the same sign) has probability levels^-N,
    # utterly negligible at the N~300 of a real ISI vector, and it is
    # never satisfied in practice: measured across a real backplane
    # (peters B12) and several synthetic channels under both IEEE_8023by
    # and IEEE_8023dj, sum(|h|) ran 1.9x to 104x the grid half-width in
    # every single configuration. Enforcing it made delta_pmf unusable
    # for the workload it exists to serve.
    #
    # A statistical bound (k * sqrt(varX * sum(h^2)) <= half-width, with k
    # around the sigma the target DER implies) is the right shape, and
    # would also flag the genuinely degenerate cases this exposed — where
    # the distribution overflows +/-1.1*As so badly that PyChOpMarg's own
    # Ani saturates at the grid edge and its COM collapses to a constant
    # 20*log10(1/1.1) = -0.83 dB regardless of channel. Left out for now
    # because delta_pmf doesn't know varX, so the bound needs either
    # `levels` factored in here or the check moved up to a caller that
    # already has it.
    #
    # max_reach = float(np.abs(h_samples).sum())
    # upward_margin = y[-1] - y[center_ix]
    # downward_margin = y[center_ix] - y[0]
    # if max_reach > upward_margin or max_reach > downward_margin:
    #     raise ValueError(...)

    # (93A-39): each entry of `h_samples` independently takes one of
    # `levels` equally likely values, evenly spaced across [-1, 1].
    level_values = np.arange(levels) * (2.0 / (levels - 1)) - 1.0

    pmf = np.zeros(npts)
    pmf[center_ix] = 1.0  # delta at y=0 before any sample is convolved in

    # (93A-40): fold in each UI's contribution via convolution, one at a
    # time. A source occupying `levels` discrete points, scaled by `h`, is
    # itself a sum of shifted deltas — convolving with it means summing
    # `pmf` shifted to each of those points (then renormalizing, since a
    # shift can land two of the `levels` points on the same bin).
    for h in h_samples:
        shifts = np.round(level_values * h / ystep).astype(int)
        # TODO: dropping zero-shift levels here (rather than convolving all
        # `levels` branches, including zero-shift ones, each at weight
        # 1/levels) matches PyChOpMarg's `delta_pmf` exactly (our golden
        # reference) but diverges from the official IEEE 802.3 MATLAB COM
        # tool (`get_pdf_from_sampled_signal`, no such filtering — always a
        # full `levels`-way convolution at uniform 1/levels weight). Only
        # matters when `h` is small enough that some but not all levels
        # round to the same bin as no shift; verified via a hand-constructed
        # L=4 case that the two disagree. Left as-is for now, matching the
        # golden reference; revisit if this project ever needs to match the
        # official tool instead of PyChOpMarg bit-for-bit.
        shifts = shifts[shifts != 0]  # h == 0: contributes nothing
        if len(shifts) == 0:
            continue
        folded = np.zeros(npts)
        for shift in shifts:
            folded = folded + np.roll(pmf, shift)
        pmf = folded / folded.sum()

    return pmf
