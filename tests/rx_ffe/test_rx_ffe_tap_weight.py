import numpy as np
import numpy.typing as npt
import pychopmarg.common
import pytest

from serdeskit.rx_ffe import TapWeightRxFfe


@pytest.fixture
def exact_pychopmarg_pi(monkeypatch: pytest.MonkeyPatch) -> None:
    """`Hffe_Rx`'s own phase matrix is built inside pychopmarg.com using a
    TWOPI bound at import time from pychopmarg.common — a different
    binding from the one tests/conftest.py's exact_pi fixture patches
    (pychopmarg.utility.filter's), so that fixture doesn't cover this
    formula. Patched directly here instead of reusing exact_pi, since
    Hffe_Rx doesn't live in utility.filter at all.
    """
    monkeypatch.setattr(pychopmarg.common, "PI", np.pi)
    monkeypatch.setattr(pychopmarg.common, "TWOPI", 2 * np.pi)


def _hffe_rx_reference(
    taps: npt.NDArray[np.float64], tap_delay: float, freqs: npt.NDArray[np.float64]
) -> npt.NDArray[np.complex128]:
    """PyChOpMarg's own COM.Hffe_Rx formula (`taps @ self.rx_ffe_phase_matrix`,
    where `rx_ffe_phase_matrix = exp(outer(arange(nRxTaps), -1j*TWOPI*ui*f))`)
    — reproduced directly rather than via a full COM instance, since
    building one requires an entire channel/package configuration this
    formula itself doesn't touch.
    """
    ns = np.arange(len(taps))
    return np.asarray(
        taps @ np.exp(np.outer(ns, -1j * pychopmarg.common.TWOPI * tap_delay * freqs)),
        dtype=np.complex128,
    )


@pytest.mark.usefixtures("exact_pychopmarg_pi")
def test_matches_pychopmarg_golden_reference() -> None:
    """Unlike Tx FFE's cursor-derived-from-taps convention, Rx FFE taps
    are given directly, cursor included — Hffe_Rx just does
    `taps @ rx_ffe_phase_matrix`, no normalization.
    """
    tap_weights = np.array([0.1, 1.0, -0.05, 0.02])  # cursor at index 1 here; RxFfe doesn't care which
    tap_delay = 1e-11
    freqs = np.linspace(0, 50e9, 501)

    rx_ffe = TapWeightRxFfe(tap_weights=tap_weights, tap_delay=tap_delay)

    expected = _hffe_rx_reference(tap_weights, tap_delay, freqs)
    actual = rx_ffe.transfer_function(freqs)

    np.testing.assert_allclose(actual, expected, atol=1e-12)


def test_single_tap_is_flat_response() -> None:
    """One tap (n=0 only): H(f) is a frequency-independent constant equal
    to that tap's weight.
    """
    tap_weights = np.array([1.0])
    rx_ffe = TapWeightRxFfe(tap_weights=tap_weights, tap_delay=1e-11)

    freqs = np.array([0.0, 1e9, 25e9, 50e9])
    actual = rx_ffe.transfer_function(freqs)

    np.testing.assert_allclose(actual, np.ones_like(freqs, dtype=complex), atol=1e-12)


def test_hand_verified_two_tap_response() -> None:
    """Two taps at a single frequency, worked out by hand: H(f) = b0 +
    b1*exp(-j*2*pi*T*f), independent of TapWeightRxFfe's own internals.
    """
    tap_weights = np.array([0.5, -0.25])
    tap_delay = 2e-11
    freqs = np.array([10e9])

    rx_ffe = TapWeightRxFfe(tap_weights=tap_weights, tap_delay=tap_delay)

    expected = 0.5 + (-0.25) * np.exp(-1j * 2 * np.pi * tap_delay * freqs)
    actual = rx_ffe.transfer_function(freqs)

    np.testing.assert_allclose(actual, expected, atol=1e-12)
