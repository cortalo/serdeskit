"""Package: one side's (Tx or Rx) complete die+package parasitic model —
die_model, package_transmission_line, and a pad shunt_capacitor cascaded
together, then turned into a differential two-port via differential_pair
(PyChOpMarg's own sPkgTx/sPkgRx).

A dataclass rather than another free function: this is exactly the kind
of coherent parameter bundle CLAUDE.md's architecture note already
justifies elsewhere (ComParams, TwoStageCtle, ...) — ten related values
that always travel together, not a handful of similar-shaped calls.

Tx and Rx differ in two ways, both driven by `is_rx`: which end the
signal enters from (die-then-pad for Tx, pad-then-die for Rx — the
package sits on the *transmit* side of a Tx die and the *receive* side
of an Rx die) and, correspondingly, die_model's own `flip` (see its
docstring for why the Rx die needs turning around).
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import skrf

from serdeskit.package.die import die_model
from serdeskit.package.differential import differential_pair
from serdeskit.package.passive import shunt_capacitor
from serdeskit.package.transmission_line import package_transmission_line


@dataclass(frozen=True)
class Package:
    r0: float
    die_capacitances: Sequence[float]
    die_inductances: Sequence[float]
    bump_capacitance: float
    tline_a1: float
    tline_a2: float
    tline_tau: float
    tline_gamma0: float
    tline_segments: Sequence[tuple[float, float]]
    pad_capacitance: float
    is_rx: bool = False

    def network(self, freqs: npt.NDArray[np.float64]) -> skrf.Network:
        """This side's complete package model, as a differential two-port.

        Args:
            freqs: Frequencies to evaluate at (Hz).

        Returns:
            The differential two-port network to cascade with a raw
            channel's own network (Tx package ** channel ** Rx package).
        """
        die = die_model(
            freqs, self.r0, self.die_capacitances, self.die_inductances, self.bump_capacitance,
            flip=self.is_rx,
        )
        tline = package_transmission_line(
            freqs, self.r0, self.tline_a1, self.tline_a2, self.tline_tau, self.tline_gamma0,
            self.tline_segments,
        )
        pad = shunt_capacitor(freqs, self.pad_capacitance, self.r0)

        conductor = (pad ** tline ** die) if self.is_rx else (die ** tline ** pad)
        return differential_pair(conductor)
