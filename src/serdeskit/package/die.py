"""die_model: the full on-die parasitic network (PyChOpMarg's COM.sDie) —
several die_ladder_segment rungs cascaded end to end, plus a final shunt
"bump" capacitance (the die-to-package bond/bump parasitic):

   Port 1 o──[ladder rung 1]──[ladder rung 2]──...──[bump cap]──o Port 2
   (Z0=r0)                                                        (Z0=r0)

One rung per (capacitance, inductance) pair — matches PyChOpMarg's
per-index C_d[i]/L_s[i] on one side (Tx or Rx) of the die.
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt
import skrf

from serdeskit.package.die_ladder import die_ladder_segment
from serdeskit.package.passive import shunt_capacitor


def die_model(
    freqs: npt.NDArray[np.float64],
    r0: float,
    capacitances: Sequence[float],
    inductances: Sequence[float],
    bump_capacitance: float,
    flip: bool = False,
) -> skrf.Network:
    """The on-die parasitic network for one side (Tx or Rx) of the die:
    `len(capacitances)` ladder rungs cascaded together, then a final
    shunt bump capacitance.

    Args:
        freqs: Frequencies to evaluate at (Hz).
        r0: Reference impedance (Ohms).
        capacitances: One shunt capacitance (F) per ladder rung.
        inductances: One series inductance (H) per ladder rung, same
            length and order as `capacitances`.
        bump_capacitance: The final shunt capacitance (F), added after
            every rung.
        flip: Swap the two ports (S11<->S22, S12<->S21) before
            returning, without mutating any network this was built from.
            The Rx-side die needs this: cascading Tx package -> channel
            -> Rx package -> Rx die means the Rx die's "port 1" should
            face the incoming trace, but this model is built the same
            direction (silicon-outward) regardless of which side it's
            for — flipping is what turns it around for that use.
            Default: False.

    Returns:
        A two-port network — symmetric only if `flip` doesn't change
        anything, which isn't the case in general (a ladder ending in a
        single shunt cap isn't front-back symmetric, same situation as
        die_ladder_segment itself).
    """
    segments = [
        die_ladder_segment(freqs, capacitance, inductance, r0)
        for capacitance, inductance in zip(capacitances, inductances)
    ]
    die = segments[0]
    for segment in segments[1:]:
        die = die ** segment
    die = die ** shunt_capacitor(freqs, bump_capacitance, r0)

    return die.flipped() if flip else die
