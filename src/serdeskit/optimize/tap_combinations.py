"""tx_tap_combinations: the Tx FFE tap-weight search space (PyChOpMarg's
mk_combs, plus the c0_min cursor-floor filter __init__ applies right
after calling it) — every combination of pre/post-cursor tap weights
opt_eq's grid search tries, one per candidate Tx equalization setting.
"""
from __future__ import annotations

import itertools
from collections.abc import Sequence

import numpy as np
import numpy.typing as npt


def tx_tap_combinations(
    bounds: Sequence[tuple[float, float, float]], min_cursor: float
) -> list[npt.NDArray[np.float64]]:
    """Every combination of tap weights across `bounds`, filtered to
    those that leave at least `min_cursor` for the cursor tap (93A-21's
    convention: cursor = 1 - sum(|taps|), so a combination is kept only
    if `1 - sum(|taps|) >= min_cursor`).

    Args:
        bounds: One (min, max, step) triple per pre/post-cursor tap
            position. A step of 0 locks that position to 0.0 (both min
            and max are ignored) — how an unused tap position is
            represented, not a single-point range at its min.
        min_cursor: The minimum cursor weight a combination must leave
            (see above); typically `ComParams.c0_min` in the standard's
            own terms.

    Returns:
        Every combination satisfying the cursor floor, in the same
        (unfiltered) order `bounds`' own cross product produces.
    """
    ranges = [
        np.arange(lo, hi + step, step) if step else np.array([0.0]) for lo, hi, step in bounds
    ]
    combos = (np.array(values) for values in itertools.product(*ranges))
    return [combo for combo in combos if (1 - np.abs(combo).sum()) >= min_cursor]
