"""SParameterChannel: loads a Touchstone-described channel via scikit-rf,
exposes its differential S21 and impulse response, and filters a Signal
through it (satisfying serdeskit.link.Channel's `process`).
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
import skrf
from scipy.signal import fftconvolve

from serdeskit.common.types import Signal


def differential_network(
    network: skrf.Network, port_order: Sequence[int] = (0, 2, 1, 3)
) -> skrf.Network:
    """A 4-port network is assumed single-ended, in the port order used
    by the ECEN 720 `peters_*`/`Case4_*` Touchstone files and by MATLAB's
    `s2sdd` default (see reference/ecen720/read_sparam.m): (TX+, RX+,
    TX-, RX-). scikit-rf's `se2gmm` instead expects each differential
    pair's two ends adjacent — (TX+, TX-, RX+, RX-) — so ports are
    renumbered [0,1,2,3] -> [0,2,1,3] (`port_order`'s default) before
    conversion. This matches the permutation PyBERT's `import_freq()`
    documents for the same file family ([4k, 4k+2, 4k+1, 4k+3] for lane
    k=0).

    Not every Touchstone file family uses that convention, though — e.g.
    the IEEE 802.3ck tools archive's `Std_BP_12inch_Meg7_*` backplane
    data is already adjacent-paired (TX+, TX-, RX+, RX-), per its own
    `Index_S4P-2019-3628.txt`. Pass `port_order=(0, 1, 2, 3)` (identity)
    for those — `se2gmm` can pair ports (0,1)/(2,3) directly, no
    renumbering first.

    Exposed as its own function — not just inlined in
    `SParameterChannel.__init__` — for callers that need the raw
    differential network before it's wrapped in a Channel, e.g.
    `evaluate.evaluate_channel` cascading it with a package model first.

    Args:
        network: A 2-port (returned unchanged) or 4-port network.
        port_order: The permutation applied before `se2gmm`, mapping
            `[0, 1, 2, 3]` to this. Default assumes the ECEN720/PyBERT
            interleaved convention; pass `(0, 1, 2, 3)` for files whose
            ports are already adjacent-paired.

    Returns:
        The differential two-port (SDD: Tx-diff -> Rx-diff).

    Raises:
        ValueError: `network` is neither 2-port nor 4-port.
    """
    if network.nports == 4:
        network = network.copy()
        network.renumber([0, 1, 2, 3], list(port_order))
        network.se2gmm(p=2)  # mutates in place: ports become d0, d1, c0, c1
        return network.subnetwork([0, 1])  # SDD: TX-diff -> RX-diff
    if network.nports != 2:
        raise ValueError(f"expected a 2-port or 4-port network, got {network.nports}-port")
    return network


@dataclass
class SParameterChannel:
    """`network` (a 4-port, converted to its differential two-port via
    `differential_network` -- see there for the port-order assumption; a
    2-port is used as-is) plus `gamma1`/`gamma2`, the reflection
    coefficients looking out of the near/far ends of the channel (e.g.
    `(R_d - R_0) / (R_d + R_0)` for a die impedance R_d against system
    reference impedance R_0) — used by `transfer_function` (93A-18) to
    account for reflections at imperfectly-terminated ends. Default 0.0
    (perfectly matched, no reflection) makes `transfer_function` reduce
    to plain S21.
    """

    network: skrf.Network
    gamma1: float = 0.0
    gamma2: float = 0.0
    _network: skrf.Network = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        self._network = differential_network(self.network)

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

    def transfer_function(self, freqs: npt.NDArray[np.float64]) -> npt.NDArray[np.complex128]:
        """(93A-18): the terminated two-port voltage transfer function,
        H21(f) — accounts for reflections at both ends when `gamma1`/
        `gamma2` are nonzero (plain S21 is only the exact answer under
        perfect termination, gamma1=gamma2=0).

        Frequencies beyond this channel's measured band are handled the
        same way MATLAB COM3.70's own `chdata(i).sdd21` construction does:
        extrapolate the network to DC, cubic-interpolate in-band, then
        hold the last in-band value constant (edge-pad) beyond it — no
        raised-cosine taper. PyChOpMarg's own `calc_H21` does apply one
        (across the *entire* band, not just the padded region), and this
        function used to transcribe it — but that has no MATLAB
        counterpart at all, at any point (with or without MATLAB's own,
        off-by-default causality correction — see docs/known-issues.md),
        and MATLAB, not PyChOpMarg, is this project's authoritative
        reference. Confirmed against MATLAB's own channel+package S21 for
        a real channel (`matlab_golden/data/channel_plus_package_c2c_thru.csv`,
        `tests/channel/test_transfer_function_vs_matlab.py`): matches to
        float precision without the taper.
        """
        in_band = self._network.extrapolate_to_dc().interpolate(
            freqs[freqs <= self._network.f[-1]], kind="cubic", coords="polar",
            basis="t", assume_sorted=True,
        )
        pad_len = len(freqs) - len(in_band.f)
        s11 = np.pad(in_band.s[:, 0, 0], (0, pad_len), mode="edge")
        s12 = np.pad(in_band.s[:, 0, 1], (0, pad_len), mode="edge")
        s21 = np.pad(in_band.s[:, 1, 0], (0, pad_len), mode="edge")
        s22 = np.pad(in_band.s[:, 1, 1], (0, pad_len), mode="edge")

        g1, g2 = self.gamma1, self.gamma2
        d_s = s11 * s22 - s12 * s21
        h = (s21 * (1 - g1) * (1 + g2)) / (1 - s11 * g1 - s22 * g2 + g1 * g2 * d_s)
        return np.asarray(h, dtype=np.complex128)

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

    def process(self, sig: Signal) -> Signal:
        """Convolve `sig` through this channel's impulse response.

        `impulse_response()`'s output is centered on t=0 (scikit-rf
        fftshifts it) and, unwindowed, has non-negligible Gibbs-ringing
        energy well before the main peak — neither MATLAB's `imp` (plain
        `ifft`, causal from index 0) nor a raw convolution kernel can use it
        as-is. `_trim_impulse` (a port of PyBERT's `trim_impulse`)
        re-centers it on the main lobe and keeps only the window capturing
        99.9% of the response's derivative energy, returning that window's
        delay relative to t=0 — which becomes part of the output Signal's
        `t0`, so callers never have to track the channel's group delay by
        hand (see Signal.t0's own docstring for why that matters).

        `mode="valid"`: a 'full' convolution's first/last `len(h_trimmed)-1`
        samples are where the kernel only partially overlaps real input
        (implicitly zero-padded) — the channel's response to data that
        hasn't started yet/has already ended, not steady-state ISI. 'valid'
        drops exactly that non-representative region on both ends (matching
        what MATLAB's `channel_data.m` does by hand, skipping a hardcoded
        55/500 symbols before plotting its eye) rather than leaving it in
        for eye extraction to slice into a handful of misleading traces.
        """
        dt = 1.0 / sig.fs
        _, h = self.impulse_response(dt=dt)
        h_trimmed, delay_samples = _trim_impulse(h)
        out = np.asarray(fftconvolve(sig.samples, h_trimmed, mode="valid"), dtype=np.float64)
        t0 = sig.t0 + (delay_samples + len(h_trimmed) - 1) * dt
        return Signal(samples=out, fs=sig.fs, t0=t0)


def _trim_impulse(
    h: npt.NDArray[np.float64], kept_energy: float = 0.999
) -> tuple[npt.NDArray[np.float64], int]:
    """Port of PyBERT's trim_impulse (utility/sigproc.py): re-center a
    possibly fftshift-centered impulse response on its main lobe, then keep
    only the window capturing `kept_energy` of the total first-derivative
    energy — discards the near-zero (or, unwindowed, Gibbs-ringing) tails on
    both sides without losing real precursor/postcursor ISI content.

    Returns (trimmed_h, start_offset): `start_offset` is the returned
    window's first sample's position, in samples, relative to the input
    array's own t=0 (its center, since impulse_response() is fftshift-ed) —
    i.e. the channel's group delay in samples, which can be negative
    (nonzero front porch before the main lobe).
    """
    n = len(h)
    half = n // 2
    if np.argmax(np.abs(h)) < n // 4:
        h = np.roll(h, half)

    diff_h = np.diff(h)
    total_energy = np.sum(diff_h**2)
    half_residual = 0.5 * (1 - kept_energy)
    e_beg_target = half_residual * total_energy
    e_end_target = (1 - half_residual) * total_energy

    ix_beg = 0
    energy = 0.0
    while energy < e_beg_target and ix_beg < n - 1:
        energy += diff_h[ix_beg] ** 2
        ix_beg += 1
    ix_end = ix_beg
    while energy < e_end_target and ix_end < n - 1:
        energy += diff_h[ix_end] ** 2
        ix_end += 1

    return h[ix_beg:ix_end], ix_beg - half
