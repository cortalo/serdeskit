"""SignSignMuellerMuller: the comparator-only form of the Mueller-Muller
timing loop (as in Spagna, ISSCC 2010), with known data d. Error samplers at
+-dLev give

    ERR[n] = sign(|r[n]| - dLev) = d[n] sign(e[n]),   e[n] = r[n] - dLev d[n],

so the phase detector sign(e[n])d[n-1] - sign(e[n-1])d[n] becomes

    d[n]d[n-1] (ERR[n] - ERR[n-1])        (only differing ERRs vote)

and dLev tracks h_0 from the same ERRs:

    phase <- phase + step_phase * sign(sum of phase votes)
    dLev  <- dLev  + step_dlev  * sign(sum ERR[n])

Both loops run together: at h_1 = h_-1 the phase detector averages to zero
for any dLev, so dLev only sets the detector's gain, not the lock point.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from serdeskit.cdr.mueller_muller import PhaseAdjustableLink


@dataclass(frozen=True, slots=True)
class TimingTrace:
    """Before the first iteration and after each one: n_iterations + 1 rows."""

    phase: npt.NDArray[np.float64]  # UI
    dlev: npt.NDArray[np.float64]  # V


@dataclass(frozen=True)
class SignSignMuellerMuller:
    step_phase: float  # UI
    step_dlev: float  # V
    # Input-referred noise of the error samplers, volts. Without it the
    # residual ISI takes few discrete values and sign(e) leaves a dead zone.
    noise_rms: float = 0.0
    rng: np.random.Generator = field(default_factory=np.random.default_rng)

    def run(
        self,
        link: PhaseAdjustableLink,
        data: npt.NDArray[np.float64],
    ) -> TimingTrace:
        """Args:
            data: +/-1 symbols, shape (n_iterations, block) -- one block
                per iteration, sent back to back as one stream.
        """
        phase, dlev = [link.sampling_phase], [0.0]
        for block in data:
            rx = link.respond(block)

            # Known d picks the error sampler at d[n] * dLev. d = 0 (idle line,
            # or data not yet recovered) gives no error sample.
            err = np.zeros(len(rx.d))
            for n in range(len(rx.d)):
                if rx.d[n] != 0:
                    r = rx.r[n] + self.rng.normal(0.0, self.noise_rms)
                    err[n] = 1.0 if r * rx.d[n] > dlev[-1] else -1.0

            phase_vote = 0.0
            for n in range(1, len(rx.d)):
                phase_vote += rx.d[n] * rx.d[n - 1] * (err[n] - err[n - 1])

            link.set_sampling_phase(link.sampling_phase + self.step_phase * np.sign(phase_vote))
            phase.append(link.sampling_phase)
            dlev.append(dlev[-1] + self.step_dlev * float(np.sign(err.sum())))

        return TimingTrace(phase=np.array(phase), dlev=np.array(dlev))
