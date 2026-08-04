"""shunt_capacitor / series_inductor: the two elementary lumped-element
two-port S-parameter building blocks package/die parasitic models are
cascaded from (IEEE 802.3-2022 Annex 93A, equation 93A-8 for the
capacitor; the inductor is its series dual, not separately numbered in
the standard).

Plain functions, not classes: each is one closed-form formula with a
single shape (frequencies + one component value -> a two-port network),
and `skrf.Network` already provides the composition/cascade machinery
(`**`) these get combined with — there's no shared behavior here that
would justify a wrapper type of our own.
"""
from __future__ import annotations

import numpy as np
import numpy.typing as npt
import skrf


def shunt_capacitor(
    freqs: npt.NDArray[np.float64], capacitance: float, r0: float = 50.0
) -> skrf.Network:
    """(93A-8): the two-port S-parameter network for a shunt capacitance
    between two `r0`-Ohm reference-impedance ports.

    `r0` isn't a real resistor in this circuit — it's the reference
    impedance S-parameters are defined against (what "port 1"/"port 2"
    mean is "a Z0=r0 transmission line", not a physical part). The only
    real component is the capacitor, shunted to ground at the node
    between the two ports:

                  node
       Port 1 o────┬────o Port 2
       (Z0=r0)     │      (Z0=r0)
                   ═╪═ C
                    │
                   ─┴─  (reference / ground)

    S11 is the reflection looking into port 1 (port 2 terminated in
    `r0`); S21 is the transmission from port 1 to port 2. S22=S11 and
    S12=S21 because the capacitor sits exactly at the midpoint of a
    passive, reciprocal network — symmetric either way you look into it.

    Args:
        freqs: Frequencies to evaluate at (Hz).
        capacitance: The shunt capacitance (F).
        r0: Reference impedance (Ohms). Default: 50.0.

    Returns:
        A reciprocal, symmetric two-port network (S12=S21, S11=S22).
    """
    jwrc = 1j * 2 * np.pi * freqs * r0 * capacitance
    s11 = np.asarray(-jwrc / (2 + jwrc), dtype=np.complex128)
    s21 = np.asarray(2 / (2 + jwrc), dtype=np.complex128)
    return _symmetric_network(freqs, s11, s21, r0)


def series_inductor(
    freqs: npt.NDArray[np.float64], inductance: float, r0: float = 50.0
) -> skrf.Network:
    """The two-port S-parameter network for an inductance in series
    between two `r0`-Ohm reference-impedance ports — the series dual of
    `shunt_capacitor` (93A-8). No shunt branch to ground this time, just
    the inductor sitting directly in the through-line:

       Port 1 o────/\\/\\/\\/\\────o Port 2
       (Z0=r0)    L (inductance)     (Z0=r0)

    Args:
        freqs: Frequencies to evaluate at (Hz).
        inductance: The series inductance (H).
        r0: Reference impedance (Ohms). Default: 50.0.

    Returns:
        A reciprocal, symmetric two-port network (S12=S21, S11=S22).
    """
    w = 2 * np.pi * freqs
    jwrl = 1j * w * r0 * inductance
    w2l2 = (w * inductance) ** 2
    r2x2 = 2 * r0**2
    den = 2 * r2x2 + w2l2
    s11 = np.asarray((w2l2 + 2 * jwrl) / den, dtype=np.complex128)
    s21 = np.asarray(2 * (r2x2 - jwrl) / den, dtype=np.complex128)
    return _symmetric_network(freqs, s11, s21, r0)


def _symmetric_network(
    freqs: npt.NDArray[np.float64],
    s11: npt.NDArray[np.complex128],
    s21: npt.NDArray[np.complex128],
    r0: float,
) -> skrf.Network:
    s = np.zeros((len(freqs), 2, 2), dtype=np.complex128)
    s[:, 0, 0] = s11
    s[:, 1, 1] = s11
    s[:, 0, 1] = s21
    s[:, 1, 0] = s21
    return skrf.Network(s=s, f=freqs, z0=r0, f_unit="Hz")
