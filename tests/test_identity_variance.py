import numpy as np

from aicup_pid.identity_variance import pair_margins, matched_pairs, crossed_endpoints


def test_implicit_fisher_matches_direct_and_reverses():
    rng = np.random.default_rng(82)
    x = rng.normal(size=(29, 8))
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    k = x @ x.T
    target, noise = np.arange(14), np.arange(14, 28)
    q = 28
    got = pair_margins(k, target, noise, q)
    mid = x[target].mean(0)
    delta = x[:7].mean(0) - x[7:14].mean(0)
    residual = x[noise].reshape(2, 7, -1)
    residual = (residual - residual.mean(1, keepdims=True)).reshape(14, -1)
    ridge = np.sum(residual ** 2) / 12
    w = np.linalg.solve(np.eye(8) + residual.T @ residual / ridge, delta)
    want = [np.dot(x[q] - mid, v / np.linalg.norm(v)) for v in (delta, w)]
    np.testing.assert_allclose(got, want)
    np.testing.assert_allclose(pair_margins(k, np.r_[target[7:], target[:7]], noise, q), -np.array(want))
    a, b = pair_margins(k, target, np.zeros(14, dtype=int), q)
    np.testing.assert_allclose(a, b)


def test_disjoint_pairs_and_crossed_cells():
    assert matched_pairs([(1, 2), (2, 3), (4, 5), (1, 6)]) == [(1, 2), (4, 5)]
    # Two true identities in each pair, two reference panels, two queries; already oriented.
    margins = np.array([[[1., -1.], [2., -2.]], [[3., -1.], [2., -1.]]])
    e = crossed_endpoints(margins)
    assert e["query_flip"] == 1 and e["reference_flip"] == 0
    assert e["accuracy"] == .5
