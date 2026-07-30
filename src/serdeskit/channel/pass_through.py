"""PassThroughChannel: the simplest possible Channel — returns the signal
unchanged. Exists to exercise the Link -> Channel -> LinkResult pipeline
end to end before a real S-parameter channel model is built.
"""
from __future__ import annotations

from serdeskit.common.types import Signal


class PassThroughChannel:
    """Satisfies serdeskit.link.Channel implicitly — no inheritance needed."""

    def process(self, sig: Signal) -> Signal:
        return sig
