import numpy as np
from pid_identity_sessions import exchange, whitening


def test_same_observations_order_only():
    out = exchange(np.array([[2., 2.], [-1., -1.]]))
    assert out["same_game"] == .5 and out["two_games"] == 1.
    assert out["gain"] == .5 and out["excess_covariance"] == 4.5
    flat = exchange(np.array([[2., -1.], [2., -5.]]))
    assert flat["gain"] == 0 and flat["excess_covariance"] == 0


def test_within_player_whitening_is_translation_invariant():
    z = np.random.default_rng(21).normal(size=(8, 7, 2, 5))
    valid = np.ones((8, 7), bool)
    valid[:2, 0] = False
    shifts = np.arange(8)[:, None, None, None] * np.ones((1, 1, 2, 5))
    np.testing.assert_allclose(whitening(z, valid), whitening(z + shifts, valid), atol=1e-10)
