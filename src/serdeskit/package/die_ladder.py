"""die_ladder_segment: one rung of the on-die parasitic ladder network
(PyChOpMarg's sDieLadderSegment) — a shunt capacitor cascaded with a
series inductor:

       Port 1 o────┬────/\\/\\/\\/\\────o Port 2
       (Z0=r0)     │    L (inductance)     (Z0=r0)
                   ═╪═ C (capacitance)
                    │
                   ─┴─  (reference / ground)

A full die model cascades several of these end to end, one per parasitic
"stage" the standard's on-die model specifies (see Cd[]/Ls[] in
PyChOpMarg's own sDie) — not built here yet, this is just one rung.
"""
from __future__ import annotations

import numpy as np
import numpy.typing as npt
import skrf

from serdeskit.package.passive import series_inductor, shunt_capacitor


def die_ladder_segment(
    freqs: npt.NDArray[np.float64], capacitance: float, inductance: float, r0: float = 50.0
) -> skrf.Network:
    """A shunt capacitor cascaded with a series inductor, both between
    `r0`-Ohm reference-impedance ports — `**` is scikit-rf's own two-port
    cascade operator, so this is nothing more than combining the two
    already-golden-tested building blocks.

    Args:
        freqs: Frequencies to evaluate at (Hz).
        capacitance: The shunt capacitance (F).
        inductance: The series inductance (H).
        r0: Reference impedance (Ohms). Default: 50.0.

    Returns:
        A reciprocal, symmetric two-port network (S12=S21, S11=S22).
    """
    return shunt_capacitor(freqs, capacitance, r0) ** series_inductor(freqs, inductance, r0)
