import importlib

import numpy as np


def module():
    return importlib.import_module("aicup_pid.choice_eval")


def fixture():
    m = module()
    cfg = (m.GridPoint("A2", 1., 0.), m.GridPoint("A2", 1., .1), m.GridPoint("A2", 10., .1))
    # H0 prefers alpha=.1, ridge=1; H1 prefers the baseline.
    a = np.zeros((2, 3, 4, 3))
    a[:, 0] = .5
    a[:, 1, :2] = .8
    a[:, 1, 2:] = .4
    a[:, 2] = .3
    return m.GridScores("pub2", (0, 1), ("a", "b", "c", "d"), np.array([0, 0, 1, 1]), cfg, a, np.full((2, 4, 3), .5))


def test_alpha_zero_exactly_preserves_baseline():
    m = module()
    b = np.array([[1., 2., 3.], [0., 0., 0.]])
    np.testing.assert_array_equal(m.add_evidence(b, np.full_like(b, 8.), .4), b)
    np.testing.assert_array_equal(m.add_evidence(b, b * 5, 0), b)


def test_test_half_cannot_change_its_selected_grid():
    m = module()
    a = fixture()
    first = m.crossfit_grid(a)
    assert first["selection"]["A2"][0] == [0, 1]  # index by evaluated half
    a.score[:, :, 2:] = a.score[:, ::-1, 2:]
    second = m.crossfit_grid(a)
    assert first["selection"]["A2"][0][1] == second["selection"]["A2"][0][1]


def test_other_seed_never_enters_selection():
    m = module()
    a = fixture()
    first = m.crossfit_grid(a)
    a.score[1] = 1 - a.score[1]
    second = m.crossfit_grid(a)
    assert first["selection"]["A2"][0] == second["selection"]["A2"][0]


def test_size_balancing_and_tie_order():
    m = module()
    cfg = (m.GridPoint("A2", 1., .1), m.GridPoint("A2", 10., .1), m.GridPoint("A2", .1, 0.))
    score = np.full((3, 4, 3), .5)
    assert m.choose_grid(score, cfg, np.ones(4), "A2") == 2
    score[2] = .4
    assert m.choose_grid(score, cfg, np.ones(4), "A2") == 1
    score[0, :, :] = [.9, .3, .3]  # mean=.5, not a 3x weighted n=1
    assert m.choose_grid(score, cfg, np.ones(4), "A2") == 1


def test_crossfit_reports_losses_instead_of_only_recovered_questions():
    m = module()
    a = fixture()
    out = m.crossfit_grid(a)
    np.testing.assert_allclose(out["delta"]["A2"], -.05)
    np.testing.assert_allclose(out["per_owner_delta"]["A2"][:, :2], 0.)
    np.testing.assert_allclose(out["per_owner_delta"]["A2"][:, 2:], -.1)


def test_bootstrap_reselects_and_shares_owner_weights_across_seeds():
    m = module()
    a = fixture()
    # Differing H0 players make the source selection vary with owner weights.
    a.score[:, 1, 0] = .9
    a.score[:, 1, 1] = .1
    a.score[:, 1, 2:] = .7
    w = np.array([2, 0, 1, 1])
    out = m.crossfit_grid(a, weights=w)
    assert out["selection"]["A2"][0] == out["selection"]["A2"][1]
    assert out["selection"]["A2"][0] == [1, 1]
    np.testing.assert_allclose(out["delta"]["A2"], .3)
    report = m.bootstrap_procedure({"pub2": a}, seeds=(0, 1), draws=50, rng_seed=1)
    assert report["arms"]["A2"]["ci95"][0] < report["arms"]["A2"]["ci95"][1]
    assert len(report["selection_frequency"]["pub2"]["A2"]) > 1


def test_rank_ties_match_stable_submission_order():
    m = module()
    np.testing.assert_allclose(m.rank_scores(np.array([[1., 1., 1.]]), np.array([2])), [np.exp(-2)])


def test_positive_primary_result_requires_robustness_before_final_verdict():
    m = module()
    a = fixture()
    a.score[:, 1] = .8
    a.seeds = (2, 3)
    report = m.bootstrap_procedure({"pub2": a}, seeds=(2, 3), draws=10)
    assert report["verdict"] == "promising_pending_robustness"
