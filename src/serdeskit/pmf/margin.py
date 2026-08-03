"""noise_margin: read a noise amplitude margin off a combined PMF at a
target error rate (IEEE 802.3-2022 Annex 93A, end of 93A.1.7 — used
directly in COM = 20*log10(As/Ani)).

Physical picture: `combine_pmfs` (chained across 93A-43/93A-44/93A-45)
produces the total interference-and-noise distribution at the sampling
instant. To turn that distribution into a single noise margin, integrate
it into a CDF and ask: "how far out into the (left) tail do I have to go
before the cumulative probability reaches the target error rate, DER0?"
That distance is Ani — the noise amplitude the link is expected to
tolerate at the target bit error rate.
"""
from __future__ import annotations

import numpy as np
import numpy.typing as npt


def noise_margin(
    pmf: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
    der0: float,
) -> float:
    """The noise amplitude margin Ani (volts): the (positive) distance out
    into `pmf`'s left tail at which cumulative probability first reaches
    `der0`.

    Args:
        pmf: A combined interference-and-noise PMF (e.g. `combine_pmfs`'s
            output), on `y`'s voltage grid.
        y: The voltage grid `pmf` is defined on (volts).
        der0: Target detector error ratio (e.g. 1e-4) — the point on the
            CDF's left tail this margin is read off at.

    Returns:
        Ani (volts): a positive noise amplitude margin.
    """
    cdf = np.cumsum(pmf)
    cdf /= cdf[-1]  # enforce a proper CDF regardless of pmf's own normalization
    ix = int(np.where(cdf >= der0)[0][0])
    return float(-y[ix])
