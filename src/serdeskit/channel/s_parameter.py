"""SParameterChannel: loads a Touchstone-described channel via scikit-rf and
exposes its differential S21 and impulse response. Actually filtering a
Signal through it (satisfying the Channel protocol's `process`) is a later
step, once eye extraction knows what to do with a non-zero channel delay.
"""
from __future__ import annotations

import numpy as np
import numpy.typing as npt
import skrf


class SParameterChannel:
    def __init__(self, network: skrf.Network) -> None:
        """A 4-port network is assumed single-ended, in the port order used
        by the ECEN 720 `peters_*`/`Case4_*` Touchstone files and by
        MATLAB's `s2sdd` default (see reference/ecen720/read_sparam.m):
        (TX+, RX+, TX-, RX-). scikit-rf's `se2gmm` instead expects each
        differential pair's two ends adjacent — (TX+, TX-, RX+, RX-) — so
        ports are renumbered [0,1,2,3] -> [0,2,1,3] before conversion. This
        matches the permutation PyBERT's `import_freq()` documents for the
        same file family ([4k, 4k+2, 4k+1, 4k+3] for lane k=0).
        """
        if network.nports == 4:
            network = network.copy()
            network.renumber([0, 1, 2, 3], [0, 2, 1, 3])
            network.se2gmm(p=2)  # mutates in place: ports become d0, d1, c0, c1
            network = network.subnetwork([0, 1])  # SDD: TX-diff -> RX-diff
        elif network.nports != 2:
            raise ValueError(f"expected a 2-port or 4-port network, got {network.nports}-port")
        self._network = network

    @classmethod
    def from_touchstone(cls, path: str) -> SParameterChannel:
        return cls(skrf.Network(path))

    def s21(self, freq: npt.NDArray[np.float64]) -> npt.NDArray[np.complex128]:
        """Differential S21, interpolated onto `freq` (Hz). `freq` must lie
        within the Touchstone file's measured band — no extrapolation policy
        has been decided yet.
        """
        # skrf.Network.interpolate builds a Frequency from a bare array
        # itself, defaulting to Hz but emitting a DeprecationWarning about
        # it — pass an explicit Frequency to avoid relying on that default.
        target = skrf.Frequency.from_f(freq, unit="Hz")
        interpolated = self._network.interpolate(target, coords="polar")
        return np.asarray(interpolated.s[:, 1, 0], dtype=np.complex128)

    def impulse_response(
        self, dt: float | None = None
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """(t, h) — the differential channel's impulse response. No
        windowing — matches MATLAB's `xfr_fn_to_imp.m` and serdespy's
        `freq2impulse`, neither of which window either.

        scikit-rf's own `impulse_response()` requires frequency data
        starting at 0 Hz and uniformly spaced to give an undistorted result
        (its docstring: without that, positions are right but "shapes will
        be distorted") — this is the same DC-extrapolation step MATLAB's
        `xfr_fn_to_imp.m` and serdespy's `freq2impulse`/`zero_pad` hand-roll
        themselves; here it's `extrapolate_to_dc()`. `bandpass=False` forces
        the real irfft path (baseband), rather than skrf silently falling
        back to a shape-distorting bandpass transform because our
        Touchstone data doesn't start at DC.

        Verified against this channel's own S21(0 Hz): the resulting
        step response (cumulative integral of the impulse response)
        settles to the same value.

        `dt` sets the output time step (s) by zero-padding the spectrum
        before the irfft (same technique `xfr_fn_to_imp.m` uses, padding out
        to `fmax=1/Ts`) — left at scikit-rf's own default (native resolution
        tied to this file's measured bandwidth, ~33 ps for a 15 GHz-wide
        file) when None. This isn't just a display choice: samples represent
        response *per sample*, not per unit time, so a finer `dt` spreads
        the same energy over more, proportionally smaller-valued samples —
        peak amplitude at two different `dt` isn't comparable. MATLAB's
        `read_sparam.m` uses `Ts=1ps`; matching that here (`dt=1e-12`)
        reproduces its ~5.3 mV peak for the B12 channel.
        """
        network = self._network.extrapolate_to_dc()
        n = None
        if dt is not None:
            df = network.f[1] - network.f[0]
            n = round(1.0 / (dt * df))
        t, h = network.impulse_response(window=None, n=n, bandpass=False)
        return np.asarray(t, dtype=np.float64), np.asarray(h[:, 1, 0], dtype=np.float64)
