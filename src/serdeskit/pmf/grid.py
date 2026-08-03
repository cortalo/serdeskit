"""voltage_grid / filter_samples: the two setup steps every PMF in this
package shares (IEEE 802.3-2022 Annex 93A, 93A.1.7.1 Notes 1 and 2).

`voltage_grid` builds the shared voltage axis `delta_pmf` and
`gaussian_pmf` are computed on — deliberately not one of their own
arguments, since the same axis has to be common to every PMF that later
gets convolved together, which makes picking it a caller-level concern.

`filter_samples` drops pulse-response samples too small to matter before
they reach `delta_pmf`, which otherwise pays a full convolution for each.
"""
from __future__ import annotations

import numpy as np
import numpy.typing as npt


def voltage_grid(signal_amplitude: float) -> npt.NDArray[np.float64]:
    """Note 1 of 93A.1.7.1: a uniformly spaced, zero-centered voltage axis
    spanning +/-1.1 * `signal_amplitude`.

    The quantization step is 10 uV, capped at 2001 points total — a
    compromise the standard's own note motivates: too coarse an axis
    accumulates quantization error through the repeated convolutions,
    but the point count directly sets their cost.

    Args:
        signal_amplitude: As (volts), per 93A.1.6.c.

    Returns:
        The voltage axis (volts), odd-length so that exactly one point
        lands on 0 V — which `delta_pmf` requires, since that's where it
        seeds its initial delta.
    """
    ymax = 1.1 * signal_amplitude
    # 1_000, not 10_000: PyChOpMarg has two copies of this formula that
    # disagree on the cap. `calc_noise`'s uses 1_000 and is the one that
    # shapes real results; `delta_pmf`'s own uses 10_000 but only runs
    # when `y` is left unspecified, which this project never does.
    npts = 2 * min(int(ymax / 0.00001), 1_000) + 1
    return np.linspace(-ymax, ymax, npts)


def filter_samples(
    samples: npt.NDArray[np.float64], signal_amplitude: float, threshold: float = 0.001
) -> npt.NDArray[np.float64]:
    """Note 2 of 93A.1.7.1: drop samples whose magnitude is below
    `threshold * signal_amplitude`, on the grounds that they're
    measurement noise or numerical artifacts rather than real
    contributors.

    Args:
        samples: Pulse-response samples (volts), e.g. residual ISI or
            A_DD-scaled slopes.
        signal_amplitude: As (volts), per 93A.1.6.c.
        threshold: Relative magnitude below which a sample is dropped.
            Default: 0.001 (0.1%, per the note).

    Returns:
        The subset of `samples` at or above the threshold, in order.
    """
    return samples[np.abs(samples) > signal_amplitude * threshold]
