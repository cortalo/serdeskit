"""Dual-loop sign-sign LMS adaptation of a 4-tap TX FFE (Stojanovic, JSSC
2005, eqs. (1)-(2)) over a synthetic symbol-rate channel -- tap weights and
dLev vs. iteration, as in ECEN720 lecture 8, slide 38.

Run: python examples/plot_tx_ffe_adaptation.py
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from serdeskit.adapt import (
    SampledPulseChannel,
    SignSignLms,
    SymbolRateLink,
    TxFfe,
)


def main() -> None:
    rng = np.random.default_rng(0)
    link = SymbolRateLink(
        channel=SampledPulseChannel(cursors=np.array([0.05, 0.5, 0.25, 0.1]), n_pre=1),
        ffe=TxFfe(weights=np.array([0.0, 1.0, 0.0, 0.0]), main=1),
    )
    trace = SignSignLms(step_tap=1e-3, step_dlev=1e-3, noise_rms=0.01, rng=rng).run(
        link=link,
        data=rng.choice(np.array([-1.0, 1.0]), size=(1500, 512)),
    )

    fig, (ax_taps, ax_dlev) = plt.subplots(2, 1, sharex=True, figsize=(7, 6))
    for col, name in enumerate(["pre1", "main", "post1", "post2"]):
        ax_taps.plot(trace.tap_weights[:, col], label=name)
    ax_taps.set_ylabel("tap weight")
    ax_taps.legend()
    ax_taps.grid(True)

    ax_dlev.plot(trace.dlev)
    ax_dlev.set_ylabel("dLev")
    ax_dlev.set_xlabel("iteration")
    ax_dlev.grid(True)

    fig.suptitle("TX FFE sign-sign LMS, dual loop")
    plt.show()


if __name__ == "__main__":
    main()
