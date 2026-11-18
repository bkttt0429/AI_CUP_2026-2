import numpy as np

from pid_tactical_temporal_transfer import agreed_residual, anchored_fit


def test_agreement_is_symmetric_and_requires_all_three_signs():
    x = np.array([[.2, .1, .3], [-.5, -.1, -.2], [.5, -.2, .1], [.2, .1, 0.]])
    np.testing.assert_allclose(agreed_residual(x), [.2, -.5, 0, 0])
    np.testing.assert_allclose(agreed_residual(-x), -agreed_residual(x))


def test_anchored_coefficient_can_be_zero_and_keeps_base_score():
    base = np.array([.4, -.1, .2, .9])
    assert anchored_fit(base, np.zeros(4)) == 0
    assert anchored_fit(base, -np.ones(4)) == 0
    assert anchored_fit(base, np.ones(4)) > 0
