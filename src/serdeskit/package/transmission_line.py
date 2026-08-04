"""package_transmission_line: a lossy package transmission line, modeled
as one or more uniform segments cascaded together (IEEE 802.3-2022 Annex
93A, equations 93A-9:14).

Physical picture: each segment is a uniform line of characteristic
impedance `zc` and length `zp` (mm), inserted into a `r0`-Ohm reference
system — the classic mismatched-transmission-line S-parameters, with a
frequency-dependent complex propagation constant gamma(f) standing in
for the usual textbook `j*beta` (lossless) or `alpha + j*beta` (constant
loss): here loss and phase both grow with frequency in a specific way
(skin-effect-like sqrt(f) loss, dielectric-loss-like f term with a log(f)
dispersion correction), per (93A-9)/(93A-10).

       Port 1 o──══════════──o Port 2      one segment: length zp (mm),
       (Z0=r0)  Zc, gamma(f)   (Z0=r0)      characteristic impedance zc

Multiple segments (package.z_c/z_p style: e.g. one for the die-side via
transition, one for the board-side breakout) cascade the same way
die_ladder_segment's pieces do.
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt
import skrf

from serdeskit.package.passive import _symmetric_network


def package_transmission_line(
    freqs: npt.NDArray[np.float64],
    r0: float,
    a1: float,
    a2: float,
    tau: float,
    gamma0: float,
    segments: Sequence[tuple[float, float]],
) -> skrf.Network:
    """(93A-9:14): one or more uniform lossy line segments, cascaded.

    Args:
        freqs: Frequencies to evaluate at (Hz).
        r0: System reference impedance (Ohms) — each segment's own port
            impedance; `2*r0` is what a segment's characteristic
            impedance `zc` is compared against to get its reflection
            coefficient, matching a differential (two conductors' worth
            of r0) convention.
        a1: Skin-effect-like loss coefficient (sqrt(ns)/mm).
        a2: Dielectric-loss-like coefficient (ns/mm).
        tau: Propagation delay per unit length (ns/mm).
        gamma0: Frequency-independent (DC) loss term (1/mm).
        segments: One (characteristic impedance zc (Ohms), length zp
            (mm)) pair per segment, cascaded in order.

    Returns:
        The cascaded line's two-port network.
    """
    gamma = _propagation_constant(freqs / 1e9, a1, a2, tau, gamma0)

    networks = [_segment_network(freqs, r0, gamma, zc, zp) for zc, zp in segments]
    line = networks[0]
    for network in networks[1:]:
        line = line ** network
    return line


def _propagation_constant(
    f_ghz: npt.NDArray[np.float64], a1: float, a2: float, tau: float, gamma0: float
) -> npt.NDArray[np.complex128]:
    """(93A-9)/(93A-10): the complex propagation constant gamma(f)
    (1/mm), at each frequency in `f_ghz` (GHz). `f_ghz == 0` is
    special-cased to `gamma0` alone — the dispersion term's `log(f)`
    would otherwise diverge at DC.
    """
    gamma1 = a1 * (1 + 1j)

    result = np.full(f_ghz.shape, gamma0, dtype=np.complex128)
    nonzero = f_ghz != 0
    f = f_ghz[nonzero]
    gamma2 = a2 * (1 - 1j * (2 / np.pi) * np.log(f)) + 1j * 2 * np.pi * tau
    result[nonzero] = gamma0 + gamma1 * np.sqrt(f) + gamma2 * f
    return result


def _segment_network(
    freqs: npt.NDArray[np.float64],
    r0: float,
    gamma: npt.NDArray[np.complex128],
    zc: float,
    zp: float,
) -> skrf.Network:
    """One uniform segment's S-parameters: a single reflection
    coefficient `rho` (the same impedance mismatch seen at both ends,
    since the segment is uniform), attenuated/phase-shifted by
    `exp(-gamma*zp)` over its length.
    """
    rho = (zc - 2 * r0) / (zc + 2 * r0)
    e = np.exp(-gamma * 2 * zp)
    s11 = np.asarray(rho * (1 - e) / (1 - rho**2 * e), dtype=np.complex128)
    s21 = np.asarray((1 - rho**2) * np.exp(-gamma * zp) / (1 - rho**2 * e), dtype=np.complex128)
    return _symmetric_network(freqs, s11, s21, r0)
