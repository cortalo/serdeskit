import numpy as np

from serdeskit.prbs import Prbs, PrbsSync


def _with_errors(symbols: np.ndarray, rate: float, seed: int) -> np.ndarray:
    flips = np.random.default_rng(seed).random(len(symbols)) < rate
    return np.where(flips, -symbols, symbols)


def test_error_free_decisions_lock_on_the_first_seed() -> None:
    prbs = Prbs(7)
    sync = PrbsSync(prbs, verify_bits=64)
    symbols = prbs.symbols(500, start=33)

    reference = sync.feed(symbols)

    assert sync.attempts == 1
    assert sync.locked_at == 7 + 64 - 1
    np.testing.assert_array_equal(reference[: sync.locked_at + 1], 0.0)
    np.testing.assert_array_equal(reference[sync.locked_at + 1 :], symbols[sync.locked_at + 1 :])


def test_locks_through_decision_errors_and_the_reference_is_then_error_free() -> None:
    prbs = Prbs(15)
    sync = PrbsSync(prbs)
    symbols = prbs.symbols(30_000, start=1234)

    reference = sync.feed(_with_errors(symbols, rate=0.15, seed=0))

    assert sync.locked_at is not None
    assert sync.attempts > 1  # 0.85^15 ~ 9%: a few seeds fail first
    after = slice(sync.locked_at + 1, None)
    np.testing.assert_array_equal(reference[after], symbols[after])


def test_random_data_never_locks() -> None:
    sync = PrbsSync(Prbs(7))
    noise = np.random.default_rng(1).choice(np.array([-1.0, 1.0]), size=20_000)

    reference = sync.feed(noise)

    assert not sync.locked
    np.testing.assert_array_equal(reference, 0.0)


def test_chunked_feed_matches_one_shot() -> None:
    prbs = Prbs(9)
    decisions = _with_errors(prbs.symbols(5000), rate=0.1, seed=2)

    whole = PrbsSync(prbs).feed(decisions)
    sync = PrbsSync(prbs)
    parts = [sync.feed(c) for c in np.split(decisions, [100, 101, 2500])]

    np.testing.assert_array_equal(np.concatenate(parts), whole)


def test_periodic_decision_errors_still_lock() -> None:
    """Errors at fixed positions of a short PRBS (deterministic ISI, no
    noise): seeds slide one bit at a time, so an error-free window is found.
    """
    prbs = Prbs(7)
    symbols = prbs.symbols(127 * 200)
    wrong = np.zeros(127, dtype=bool)
    wrong[::6] = True  # every 6th bit always wrong: no error-free 7-bit window...
    wrong[6] = False  # ...except across this gap (bits 1-11)
    decisions = np.where(np.tile(wrong, 200), -symbols, symbols)

    sync = PrbsSync(prbs)
    reference = sync.feed(decisions)

    assert sync.locked_at is not None
    np.testing.assert_array_equal(reference[sync.locked_at + 1 :], symbols[sync.locked_at + 1 :])
