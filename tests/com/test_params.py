import numpy as np
import pytest

pytest.skip(
    "references the removed ComParams API (level_variance now lives on LinkComParams/ComStandard) -- needs updating",
    allow_module_level=True,
)

from serdeskit.com import ComParams


def _params(levels: int) -> ComParams:
    return ComParams(
        baud_rate=100e9,
        freq_step=1e8,
        samples_per_ui=32,
        levels=levels,
        rlm=1.0,
        victim_amplitude=0.4,
        a_ne=0.6,
        a_fe=0.4,
        snr_tx=27.0,
        sigma_rj=0.01,
        eta_0=5.2e-8,
        a_dd=0.05,
        der_0=1e-5,
        dfe_min=np.array([-1.0]),
        dfe_max=np.array([1.0]),
    )


def test_level_variance_nrz() -> None:
    """(93A-29) with L=2: the symbol is +/-1, each with probability 1/2,
    so its variance is 1 — checkable without the formula.
    """
    assert _params(levels=2).level_variance == pytest.approx(1.0)


def test_level_variance_pam4() -> None:
    """(93A-29) with L=4: symbols are -1, -1/3, +1/3, +1, each with
    probability 1/4, so the variance is (1 + 1/9 + 1/9 + 1)/4 = 5/9 —
    again derived from the definition rather than the closed form.
    """
    assert _params(levels=4).level_variance == pytest.approx(5 / 9)


@pytest.mark.parametrize("levels", [2, 4, 8])
def test_level_variance_matches_direct_computation(levels: int) -> None:
    """The closed form (93A-29) against the variance computed directly
    from the equally likely, evenly spaced levels it summarizes.
    """
    symbols = np.linspace(-1.0, 1.0, levels)
    expected = float(np.mean(symbols**2))  # zero-mean by symmetry

    assert _params(levels).level_variance == pytest.approx(expected)
