"""worst_case_phase_samples: the sub-UI phase selection a crosstalk
aggressor needs before its pulse response can feed `pmf.delta_pmf` (IEEE
802.3-2022 Annex 93A, equation 93A-33).

Physical picture: unlike the victim, a crosstalk aggressor has no cursor —
it's pure interference, not a signal being detected, so there's no
Muller-Mueller criterion to locate a sampling instant. Instead, every one
of the `samples_per_ui` sub-UI phases is tried, and the one whose per-UI
samples carry the most total energy is taken as the worst case: the phase
at which this aggressor does the most damage to the victim's eye.
"""
from __future__ import annotations

import numpy as np
import numpy.typing as npt


def worst_case_phase_samples(
    pulse_response: npt.NDArray[np.float64],
    samples_per_ui: int,
) -> npt.NDArray[np.float64]:
    """(93A-33): the per-UI samples at the sub-UI phase that maximizes
    total sampled energy.

    Args:
        pulse_response: A crosstalk aggressor's pulse response, volts.
        samples_per_ui: M, samples per unit interval.

    Returns:
        One sample per UI (volts), taken at the worst-case phase — ready
        to hand to `pmf.delta_pmf` as `h_samples`.
    """
    energies = [
        float((pulse_response[phase::samples_per_ui] ** 2).sum())
        for phase in range(samples_per_ui)
    ]
    best_phase = int(np.argmax(energies))
    return np.asarray(pulse_response[best_phase::samples_per_ui], dtype=np.float64)
