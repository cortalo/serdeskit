import numpy as np
import pytest

from serdeskit.common.types import Signal


def test_scale_multiplies_samples() -> None:
    sig = Signal(samples=np.array([1.0, -2.0, 3.0]), fs=100e9, t0=1e-9)

    scaled = sig.scale(0.4)

    np.testing.assert_allclose(scaled.samples, [0.4, -0.8, 1.2])


def test_scale_leaves_fs_and_t0_unchanged() -> None:
    sig = Signal(samples=np.array([1.0, 2.0]), fs=100e9, t0=1e-9)

    scaled = sig.scale(0.4)

    assert scaled.fs == sig.fs
    assert scaled.t0 == sig.t0


def test_scale_does_not_mutate_the_original() -> None:
    """Signal is frozen — scale() must return a new instance, not adjust
    `samples` in place.
    """
    original_samples = np.array([1.0, 2.0])
    sig = Signal(samples=original_samples, fs=100e9, t0=0.0)

    scaled = sig.scale(0.4)

    np.testing.assert_allclose(sig.samples, [1.0, 2.0])
    assert scaled is not sig


def test_scale_by_one_is_equal_but_a_new_object() -> None:
    sig = Signal(samples=np.array([1.0, 2.0]), fs=100e9, t0=0.0)

    scaled = sig.scale(1.0)

    np.testing.assert_allclose(scaled.samples, sig.samples)
    assert scaled is not sig


def test_scale_by_negative_flips_sign() -> None:
    sig = Signal(samples=np.array([1.0, -2.0]), fs=100e9, t0=0.0)

    scaled = sig.scale(-1.0)

    np.testing.assert_allclose(scaled.samples, [-1.0, 2.0])


@pytest.mark.parametrize("factor", [0.4, 2.0, -1.0, 0.0])
def test_scale_is_linear(factor: float) -> None:
    """Hand-verifiable, independent of any particular factor: scaling then
    reading a sample back out should match scaling that one value directly.
    """
    samples = np.array([0.5, -1.5, 3.0])
    sig = Signal(samples=samples, fs=100e9, t0=0.0)

    scaled = sig.scale(factor)

    np.testing.assert_allclose(scaled.samples, samples * factor)
