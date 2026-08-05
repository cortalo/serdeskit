import numpy as np
import numpy.typing as npt
import pytest
import skrf
from pychopmarg.utility.filter import calc_H21

from serdeskit.channel import SParameterChannel


def _network_with_reflection(freq: npt.NDArray[np.float64]) -> skrf.Network:
    fc = 8e9
    s21 = 1.0 / (1.0 + 1j * freq / fc)
    s11 = 0.05 * np.exp(-1j * freq / fc)
    s22 = 0.03 * np.exp(-1j * freq / fc)
    s = np.zeros((len(freq), 2, 2), dtype=complex)
    s[:, 0, 0] = s11
    s[:, 1, 1] = s22
    s[:, 1, 0] = s21
    s[:, 0, 1] = s21  # reciprocal
    return skrf.Network(f=freq, s=s, z0=50, f_unit="Hz")


@pytest.mark.usefixtures("exact_pi", "no_raised_cosine_taper")
def test_matches_pychopmarg_golden_reference() -> None:
    """(93A-18). The target frequency grid intentionally extends beyond the
    network's own measured band (1-20 GHz) and doesn't start at DC, to
    exercise calc_H21's extrapolate-to-DC + cubic-interpolate + edge-pad-
    beyond-band recipe, not just a simple in-band path.
    """
    freq = np.linspace(1e9, 20e9, 96)
    network = _network_with_reflection(freq)

    gamma1, gamma2 = 0.02, -0.015
    channel = SParameterChannel(network, gamma1=gamma1, gamma2=gamma2)

    target_freqs = np.linspace(0.0, 30e9, 301)
    expected = calc_H21(target_freqs, network, gamma1, gamma2)
    actual = channel.transfer_function(target_freqs)

    # `exact_pi` gives raised_cosine() full-precision PI, so this can assert
    # exact agreement rather than tolerating the ~7.7e-7 window-shape
    # deviation PyChOpMarg's truncated constant would otherwise introduce.
    np.testing.assert_allclose(actual, expected, atol=1e-9)


@pytest.mark.usefixtures("exact_pi", "no_raised_cosine_taper")
def test_default_gamma_matches_golden_reference_with_zero_reflection() -> None:
    """Default gamma1=gamma2=0.0 (perfectly matched, no reflection) must
    match calc_H21 called with g1=g2=0 too — not independent of the golden
    reference, but pins down that the "no gamma given" default wires
    through correctly rather than, say, defaulting to some other value.
    """
    freq = np.linspace(1e9, 20e9, 96)
    network = _network_with_reflection(freq)

    channel = SParameterChannel(network)

    target_freqs = np.linspace(0.0, 15e9, 151)  # in-band, keeps this test focused
    expected = calc_H21(target_freqs, network, 0.0, 0.0)
    actual = channel.transfer_function(target_freqs)

    # `exact_pi` gives raised_cosine() full-precision PI, so this can assert
    # exact agreement rather than tolerating the ~7.7e-7 window-shape
    # deviation PyChOpMarg's truncated constant would otherwise introduce.
    np.testing.assert_allclose(actual, expected, atol=1e-9)
