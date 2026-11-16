"""Numerical identities independently checked against exact softmax likelihood."""
import importlib

import numpy as np
import pytest
import torch


def module():
    return importlib.import_module("aicup_pid.choice_signal")


def toy():
    return torch.tensor([[.25, .75]], dtype=torch.float64).log(), torch.tensor([[[0.], [2.]]], dtype=torch.float64), torch.tensor([1])


def test_binary_choice_moments_and_context():
    m = module()
    lp, phi, played = toy()
    U, F, n = m.choice_moments(lp, phi, played)
    np.testing.assert_allclose(U, [.5])
    np.testing.assert_allclose(F, [[.75]])
    assert n == 1
    U, F, _ = m.choice_moments(lp, m.context_features(phi, torch.tensor([.5])), played)
    np.testing.assert_allclose(U, [.5, .25])
    np.testing.assert_allclose(F, [[.75, .375], [.375, .1875]])


def test_common_action_offset_cancels():
    m = module()
    lp, phi, p = toy()
    a, b, _ = m.choice_moments(lp, phi, p)
    c, d, _ = m.choice_moments(lp, phi + 17, p)
    torch.testing.assert_close(a, c)
    torch.testing.assert_close(b, d)
    torch.testing.assert_close(m.exact_delta(lp, phi, p, torch.tensor([.2])), m.exact_delta(lp, phi + 17, p, torch.tensor([.2])))


def test_forced_move_zero_information_and_invalid_move():
    m = module()
    lp, phi, p = toy()
    lp[:] = torch.tensor([-float("inf"), 0.])
    U, F, _ = m.choice_moments(lp, phi, p)
    assert torch.count_nonzero(U) == torch.count_nonzero(F) == 0
    np.testing.assert_array_equal(m.score_candidates(U[None], F[None], np.array([[1.], [2.]])), [[0., 0.]])
    with pytest.raises(ValueError):
        m.choice_moments(lp, phi, torch.tensor([0]))


def test_fisher_matches_exact_hessian():
    m = module()
    lp, phi, p = toy()
    U, F, _ = m.choice_moments(lp, phi, p)
    z = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    fn = lambda x: m.exact_delta(lp, phi, p, x).sum()
    torch.testing.assert_close(torch.autograd.functional.jacobian(fn, z), U)
    torch.testing.assert_close(torch.autograd.functional.hessian(fn, z), -F)


def test_second_order_matches_small_exact_tilt():
    m = module()
    lp, phi, p = toy()
    U, F, _ = m.choice_moments(lp, phi, p)
    err = []
    for zz in [.02, .01]:
        z = torch.tensor([zz], dtype=torch.float64)
        approx = U @ z - .5 * z @ F @ z
        err.append(abs(float(m.exact_delta(lp, phi, p, z).sum() - approx)))
    assert 7.5 < err[0] / err[1] < 8.5


def test_singular_fisher_has_finite_ridge_solution():
    m = module()
    np.testing.assert_allclose(m.fit_candidate(np.array([2., 0.]), np.diag([1., 0.]), 2, .5), [1., 0.])
    for n, ridge in [(0, 1), (1, 0), (1, -1)]:
        with pytest.raises(ValueError):
            m.fit_candidate(np.ones(2), np.zeros((2, 2)), n, ridge)
    with pytest.raises(ValueError):
        m.fit_candidate(np.ones(2), np.diag([-1., 1.]), 1, 1.)


def test_context_permutation_preserves_phase_multisets():
    m = module()
    v = np.arange(12.)
    t = np.array([0, 1, 2, 3, 60, 61, 62, 63, 150, 151, 152, 153])
    out = m.permute_context(v, t, pool="pub2", key="example")
    np.testing.assert_array_equal(v, np.arange(12.))
    np.testing.assert_array_equal(out, m.permute_context(v, t, pool="pub2", key="example"))
    for indices in [slice(0, 4), slice(4, 8), slice(8, 12)]:
        np.testing.assert_array_equal(np.sort(out[indices]), v[indices])


def test_condition_relation_separates_equal_marginals():
    """A contextual preference is recoverable when marginal actions are identical."""
    m = module()
    psi = torch.tensor([[[-1.], [1.]]] * 4, dtype=torch.float64)
    lp = torch.full((4, 2), -np.log(2), dtype=torch.float64)
    v = torch.tensor([-1., -1., 1., 1.], dtype=torch.float64)
    phi = m.context_features(psi, v)
    a, b = torch.tensor([0, 0, 1, 1]), torch.tensor([1, 1, 0, 0])
    ua, fa, n = m.choice_moments(lp, phi, a)
    ub, fb, _ = m.choice_moments(lp, phi, b)
    z = np.stack([m.fit_candidate(ua, fa, n, 1.), m.fit_candidate(ub, fb, n, 1.)])
    sc = m.score_candidates(np.stack([ua, ub]), np.stack([fa, fb]), z)
    np.testing.assert_array_equal(sc.argmax(1), [0, 1])
    u0a, _, _ = m.choice_moments(lp, psi, a)
    u0b, _, _ = m.choice_moments(lp, psi, b)
    torch.testing.assert_close(u0a, u0b)
