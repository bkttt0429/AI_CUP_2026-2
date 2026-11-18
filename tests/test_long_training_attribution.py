import numpy as np
import pytest

from aicup_pid.long_training import sequence_variants, pair_scores, sibling_rows


def test_factorial_matches_literal_weighted_member_lists():
    rng = np.random.default_rng(2415)
    other, early, late = rng.normal(size=(3, 4, 8))
    base = (3*other + early)/4
    got = sequence_variants(base, early, late, .25)
    np.testing.assert_allclose(got["baseline"], base)
    np.testing.assert_allclose(got["replace"], (3*other+late)/4)
    np.testing.assert_allclose(got["mix"], (3*other+.5*early+.5*late)/4)
    np.testing.assert_allclose(got["weight"], (3*other+2*early)/5)
    np.testing.assert_allclose(got["add"], (3*other+early+late)/5)


def test_pair_standardisation_excludes_exactly_those_two_people():
    rng = np.random.default_rng(12)
    x = rng.normal(size=(5, 9))
    a, b = [0, 4, 1, 6, 8], [8, 5, 3, 0, 4]
    actual, sd = pair_scores(x, a, b)
    for row, (i, j) in enumerate(zip(a, b)):
        ref = np.delete(x[row], [i, j])
        np.testing.assert_allclose(actual[row], (x[row, [i, j]]-ref.mean())/ref.std())
        np.testing.assert_allclose(sd[row], ref.std())
    fixed, _ = pair_scores(x+.9, a, b, sd)
    np.testing.assert_allclose(fixed, actual)
    with pytest.raises(ValueError):
        pair_scores(x, a, a)


def test_siblings_are_the_other_two_games_in_the_same_enrollment():
    got = sibling_rows(np.repeat([3, 7], 3))
    assert got.tolist() == [[1, 2], [2, 0], [0, 1], [4, 5], [5, 3], [3, 4]]
    with pytest.raises(ValueError):
        sibling_rows([0, 0, 2, 2, 2, 2])
