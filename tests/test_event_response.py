import importlib

import numpy as np


def module():
    return importlib.import_module("aicup_pid.event_response")


def record(n=48, subject=0):
    t = np.arange(n * 2)
    return {"t": t, "mover": t % 2, "subject": subject,
            "value": np.zeros(len(t)), "entropy": np.ones(len(t)),
            "surprisal": np.ones(len(t)), "residual": np.zeros((len(t), 2))}


def test_event_indices_use_opponent_before_subject_and_censor_lags():
    m = module()
    a = record(10, 0)
    ix = m.event_indices(a, 8)
    assert ix.tolist() == [[1, 0, 16], [3, 2, 18]]
    b = record(10, 1)
    assert m.event_indices(b, 8).tolist() == [[2, 1, 17], [4, 3, 19]]


def test_cell_permutation_preserves_context_and_is_deterministic():
    m = module()
    a = np.arange(12, dtype=float)
    cells = np.array([0] * 6 + [1] * 6)
    p, meta = m.permute_shock(a, cells, "game")
    np.testing.assert_array_equal(p, m.permute_shock(a, cells, "game")[0])
    assert sorted(p[:6]) == sorted(a[:6]) and sorted(p[6:]) == sorted(a[6:])
    assert meta["eligible"] == 12


def test_constant_response_has_zero_event_kernel():
    m = module()
    a = record()
    a["adjusted_r"] = np.ones_like(a["residual"])
    a["shock"] = np.sin(a["t"])
    out = m.response_moments(a, "game", 1.)
    np.testing.assert_allclose(out["E_num"], 0)
    np.testing.assert_allclose(out["P_num"], 0)


def test_nuisance_fit_does_not_read_or_change_heldout():
    m = module()
    a, b = record(), record()
    a["surprisal"] = 2 + .1 * np.sin(a["t"])
    fit = m.fit_nuisance([a])
    before = {k: np.array(v).copy() for k, v in fit.items()}
    b["surprisal"][:] = 1000
    m.adjust_record(b, fit)
    for k, v in before.items():
        np.testing.assert_array_equal(fit[k], v)


def test_equal_marginals_opposite_event_response_are_separable():
    m = module()
    a = record(100)
    rng = np.random.default_rng(7)
    shock = rng.permutation(np.tile([-1., 1.], 50))
    a["shock"] = np.zeros(200)
    a["shock"][1::2] = shock
    a["adjusted_r"] = np.zeros((200, 2))
    a["adjusted_r"][0, 0] = shock[-1]
    a["adjusted_r"][2::2, 0] = shock[:-1]
    b = dict(a, adjusted_r=-a["adjusted_r"])
    np.testing.assert_array_equal(np.sort(a["adjusted_r"][::2, 0]), np.sort(b["adjusted_r"][::2, 0]))
    aa, bb = m.response_moments(a, "a", 1.), m.response_moments(b, "b", 1.)
    assert m.pool_kernel([aa], "E")[0] > .1
    assert m.pool_kernel([bb], "E")[0] < -.1


def test_linear_drift_cancels_and_permutation_preserves_denominators():
    m = module()
    a = record(50)
    a["adjusted_r"] = np.column_stack((a["t"], a["t"] * 2.)).astype(float)
    a["shock"] = np.sin(a["t"] * .7)
    a["shock"][-12:] = 100.  # censored tail must not enter centering/shuffling
    out = m.response_moments(a, "drift", 1.)
    np.testing.assert_allclose(out["E_num"], 0, atol=1e-10)
    np.testing.assert_allclose(out["P_num"], 0, atol=1e-10)
    np.testing.assert_allclose(out["E_den"], out["P_den"], atol=1e-10)
    assert np.all(out["N"] == out["N"][0])


def test_zero_event_information_cannot_pass_repeatability_gate():
    m = module()
    a = record(50)
    a["adjusted_r"] = np.zeros((100, 2)); a["shock"] = np.sin(a["t"])
    row = m.response_moments(a, "zero", 1.)
    games = [{"key": f"p{p}g{g}", "owner": str(p), "role": "reference" if g < 7 else "heldout"} for p in range(4) for g in range(10)]
    out = m.evaluate_repeatability(games, {g["key"]: row for g in games}, draws=100)
    assert out["verdict"] == "pilot_repeatability_not_established"
    assert out["E_minus_P"]["ci95"] == [0., 0.]
    assert out["E_identity_margin_ci95"] == [0., 0.]
    assert out["arms"]["E"]["top1"] == .25
    assert out["learning_curves"]["E"]["reference7_minus1"]["diff"] == 0
