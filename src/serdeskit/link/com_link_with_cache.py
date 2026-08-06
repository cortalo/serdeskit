from __future__ import annotations

from dataclasses import dataclass

from serdeskit.common.types import Signal
from serdeskit.link.link import Link
from serdeskit.link.system_grid import SystemGrid


@dataclass
class com_link_with_cache:
    link: Link
    unequalized_impulse_signal: Signal

    def sbr_pulse_response(self, grid: SystemGrid) -> Signal:
        ctle_impulse = self.link.ctle.process(self.unequalized_impulse_signal)  # type: ignore[union-attr]
        pulse = grid.box_car_integrate(ctle_impulse)
        eq_pulse = self.link.ffe.process(pulse)  # type: ignore[union-attr]
        return self.link.rx_ffe.process(eq_pulse)  # type: ignore[union-attr]
