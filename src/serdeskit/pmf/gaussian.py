"""gaussian_pmf: the continuous (Gaussian) noise term in COM's
noise/interference calculation (IEEE 802.3-2022 Annex 93A, 93A.1.7.2,
equation 93A-42).

Physical picture: unlike `delta_pmf`'s sources (ISI, dual-Dirac
deterministic jitter, crosstalk), which each take one of a small, fixed
set of discrete voltage values, the Gaussian term lumps together every
noise source that's actually continuously distributed — transmitter noise,
random (as opposed to deterministic) jitter projected through the local
pulse-response slope, and receiver/thermal noise — combined into a single
variance, `varG` (93A-41). This function discretizes that Gaussian onto the
same voltage grid `delta_pmf` uses, so the two can later be convolved
together (93A-43).
"""
from __future__ import annotations

import numpy as np
import numpy.typing as npt


def gaussian_pmf(
    variance: float,
    y: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Probability mass at each point of `y` for a zero-mean Gaussian
    random variable with the given `variance`.

    Args:
        variance: The Gaussian's variance (volts^2) — `varG` from (93A-41):
            transmitter noise + random-jitter-projected-to-amplitude +
            receiver/thermal noise, combined.
        y: The shared voltage quantization grid (volts) this PMF is
            computed on — must match the grid used by every other PMF this
            result will later be convolved against.

    Returns:
        Probability mass at each point of `y`, same shape as `y`, summing
        to 1.

    Raises:
        ValueError: `variance` isn't positive — the density formula below
            divides by it (and takes its square root), silently producing
            NaNs (with only a RuntimeWarning) rather than failing at the
            actual source of the bad input.
        ValueError: `y` isn't uniformly spaced — only `y[1] - y[0]` is used
            as the grid step for the density -> mass conversion everywhere,
            so a non-uniform `y` would silently apply the wrong bin width
            wherever the spacing differs.
    """
    if variance <= 0.0:
        raise ValueError(f"variance must be positive, got {variance}")

    ystep = y[1] - y[0]
    if not np.allclose(np.diff(y), ystep):
        raise ValueError(
            f"y must be uniformly spaced (first step is {ystep}V, but "
            "spacing is not constant across the array)."
        )

    # (93A-42), density -> mass by multiplying by the grid step.
    density = np.exp(-(y**2) / (2.0 * variance)) / np.sqrt(2.0 * np.pi * variance)
    return np.asarray(density * ystep, dtype=np.float64)
