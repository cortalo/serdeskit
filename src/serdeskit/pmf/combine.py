"""combine_pmfs: combining two independent PMFs on the same voltage grid
(IEEE 802.3-2022 Annex 93A, equations 93A-43/93A-44/93A-45).

Physical picture: `delta_pmf` and `gaussian_pmf` each describe one
independent noise/interference source's own contribution to the sampled
voltage. The total sampled voltage is the *sum* of all these independent
contributions, and the distribution of a sum of independent random
variables is the convolution of their individual distributions — the same
fact `delta_pmf` already exploits internally, one UI at a time, to combine
sources of the same kind (93A-40). This function is that same operation
applied one level up: combining whole PMFs from *different* sources (the
Gaussian term with deterministic jitter's PMF, each crosstalk aggressor's
PMF with the next, and finally ISI/crosstalk/noise with each other) into
the total interference-and-noise distribution.
"""
from __future__ import annotations

import numpy as np
import numpy.typing as npt


def combine_pmfs(
    *pmfs: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """The PMF of the sum of two or more independent random variables,
    given their individual PMFs on the same shared voltage grid.

    Variadic rather than fixed at two so that combining, say, ISI, the
    Gaussian term, deterministic jitter, and however many crosstalk
    aggressors (93A-43/93A-44/93A-45) happen to be configured is one call
    with that many arguments, not a hand-nested chain of pairwise calls
    that grows every time a source is added.

    Args:
        *pmfs: Two or more independent sources' PMFs (e.g. `delta_pmf`'s
            or `gaussian_pmf`'s output), all on the *same* voltage grid
            (same length, same step). A single PMF is returned unchanged
            — the sum of one random variable is just that variable.

    Returns:
        The combined PMF, on that same shared grid, summing to 1.
    """
    combined = pmfs[0]
    for pmf in pmfs[1:]:
        convolved = np.convolve(combined, pmf, mode="same")
        combined = convolved / convolved.sum()
    return np.asarray(combined, dtype=np.float64)
