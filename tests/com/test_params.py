import numpy as np
import pytest

from serdeskit.com import LinkComParams


def _params(levels: int) -> LinkComParams:
    """Every field level_variance doesn't touch is a placeholder --
    LinkComParams has no defaults (see its own docstring for why), so a
    test that only cares about `levels`/`level_variance` still has to
    supply the rest.
    """
    return LinkComParams(
        channel_path="unused.s4p",
        next_channel_paths=(),
        fext_channel_paths=(),
        port_order=(0, 2, 1, 3),
        gamma1=0.0,
        gamma2=0.0,
        tx_r0=50.0,
        tx_die_capacitances=(),
        tx_die_inductances=(),
        tx_bump_capacitance=0.0,
        tx_tline_a1=0.0,
        tx_tline_a2=0.0,
        tx_tline_tau=0.0,
        tx_tline_gamma0=0.0,
        tx_tline_segments=(),
        tx_pad_capacitance=0.0,
        rx_r0=50.0,
        rx_die_capacitances=(),
        rx_die_inductances=(),
        rx_bump_capacitance=0.0,
        rx_tline_a1=0.0,
        rx_tline_a2=0.0,
        rx_tline_tau=0.0,
        rx_tline_gamma0=0.0,
        rx_tline_segments=(),
        rx_pad_capacitance=0.0,
        ctle_zero_freq=1.0,
        ctle_pole1_freq=1.0,
        ctle_pole2_freq=1.0,
        ctle_shelf_freq=1.0,
        ctle_dc_gain_db=0.0,
        ctle_shelf_gain_db=0.0,
        ffe_tap_weights=np.array([]),
        ffe_n_post=0,
        tx_risetime=0.0,
        rx_afe_cutoff_freq=1.0,
        rx_ffe_tap_weights=np.array([1.0]),
        rx_ffe_n_pre=0,
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
