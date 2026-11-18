import numpy as np

from aicup_pid.corroboration import quarter_labels, pair_features


def test_mask_counts_colours_two_plies_and_padding():
    rng = np.random.default_rng(2026)
    for n in (1, 2, 7, 12, 83, 255, 256):
        moves = rng.integers(0, 2, size=256)
        a, b = quarter_labels("pub2|c:123:x", moves, n)
        assert np.all(a[n:] == -1) and np.all(b[n:] == -1)
        assert a[0] == 0
        assert np.array_equal(a[:n // 2 * 2:2], a[1:n // 2 * 2:2])
        assert np.array_equal(b[:n // 2 * 2:2], b[1:n // 2 * 2:2])
        if n % 2:
            assert a[n-1] == b[n-1]
        for c in (0, 1):
            for j in range(4):
                assert np.sum((a == j) & (moves == c)) == np.sum((b == j) & (moves == c))
        assert np.array_equal(b, quarter_labels("pub2|c:123:x", moves, n)[1])
        if n >= 83:
            assert np.any(a != b)


def test_candidate_floor_is_individual_and_swapping_is_antisymmetric():
    # [game, intact plus 4 chronological plus 4 shuffled views, nominee]
    v = np.array([[[4, 1], [2, 3], [3, 1], [4, 2], [4, 1],
                   [4, 1], [4, 1], [4, 1], [4, 1]]], dtype=float)
    z = pair_features(v)
    assert z['floor'][0] == -2 and z['sham_floor'][0] == 0
    assert z['persistent'][0] == 0 and z['sham_persistent'][0] == 3
    assert z['intact'][0] == 3
    for k, x in pair_features(v[:, :, ::-1]).items():
        np.testing.assert_allclose(x, -z[k])


def test_template_matches_explicit_leave_one_out_and_reverses():
    from aicup_pid.corroboration import template_margin
    rng = np.random.default_rng(885)
    a, b = rng.normal(size=(2, 3, 7, 5, 8))
    q = rng.normal(size=(3, 4, 8))
    got = template_margin(a, b, q)
    want = []
    for roster_a, roster_b, query in zip(a, b, q):
        folds = []
        for j in range(7):
            keep = np.arange(7) != j
            ca = roster_a[keep, 0].mean(0)
            cb = roster_b[keep, 0].mean(0)
            d = ca / np.linalg.norm(ca) - cb / np.linalg.norm(cb)
            traces = np.stack([query @ d, roster_a[j, 1:] @ d, roster_b[j, 1:] @ d])
            x, y, z = traces - traces.mean(1, keepdims=True)
            folds.append((x - (y + z) / 2) @ (y - z))
        want.append(np.mean(folds))
    np.testing.assert_allclose(got, want)
    np.testing.assert_allclose(template_margin(b, a, q), -got)


def test_vector_trace_matches_explicit_projection_and_distinct_reference_pairs():
    from aicup_pid.corroboration import vector_margins
    rng = np.random.default_rng(71)
    a, b = rng.normal(size=(2, 2, 7, 5, 8))
    q = rng.normal(size=(2, 4, 8))
    got = vector_margins(a, b, q)
    np.testing.assert_allclose(got[:, 0], got[:, 1] + got[:, 2])
    np.testing.assert_allclose(vector_margins(b, a, q), -got)
    for row in range(2):
        ca, cb = a[row, :, 0].mean(0), b[row, :, 0].mean(0)
        d = ca / np.linalg.norm(ca) - cb / np.linalg.norm(cb)
        d /= np.linalg.norm(d)
        proj = np.eye(len(d)) - np.outer(d, d)
        def trace(x):
            return ((x-x.mean(0)) @ proj).ravel()
        query = trace(q[row])
        def score(ref):
            t = np.array([trace(g[1:]) for g in ref])
            return query @ t.mean(0) - .5 * np.mean([x @ y for i, x in enumerate(t) for j, y in enumerate(t) if i != j])
        np.testing.assert_allclose(got[row, 2], score(a[row])-score(b[row]))
