import importlib

import numpy as np
import pytest


def test_reference_vectors_reject_missing_and_nonfinite(tmp_path):
    m = importlib.import_module("pid_choice_eval")
    p = tmp_path / "vectors.npz"
    np.savez(p, key=np.array(["a", "b"]), emb=np.array([[1., 0.], [0., 1.]]))
    np.testing.assert_array_equal(m.load_vectors(p, ["b", "a"]), [[0., 1.], [1., 0.]])
    with pytest.raises(ValueError):
        m.load_vectors(p, ["missing"])
    np.savez(p, key=np.array(["a", "b"]), emb=np.array([[np.nan, 0.], [0., 1.]]))
    with pytest.raises(ValueError):
        m.load_vectors(p, ["a"])


def test_question_layout_excludes_references_and_balances_three_singles():
    m = importlib.import_module("pid_choice_eval")
    split = {"heldout": {"a": ["x", "y", "z"], "b": ["u", "v", "w"]}, "reference": {"a": ["r"], "b": ["s"]}}
    qs, owner, size = m.questions_from_split(split, ["a", "b"])
    assert len(qs) == 10
    assert not {"r", "s"}.intersection(k for q in qs for k in q)
    sc = np.array([1., 0., 0., .5, .25, 0., 0., 0., .1, .2])
    out = m.owner_size_scores(sc, owner, size, 2)
    np.testing.assert_allclose(out, [[1 / 3, .5, .25], [0., .1, .2]])


def test_baseline_loader_never_opens_query_csv(tmp_path):
    # A malformed minimal manifest must fail on candidate data, without trying
    # the old Pool loader (which always opens official queries).
    m = importlib.import_module("pid_choice_eval")
    with pytest.raises(ValueError, match="candidate"):
        m.validate_candidate_rows([{"pool": "pub2", "key": "pub2|q:x", "original_key": "q:x"}], "pub2")


def test_a0_preserves_fixed_grid_when_policy_taylor_rejects_ridge():
    m = importlib.import_module("pid_choice_eval")
    assert m.ridge_grid("A0", [1., 10.], [.1, 1., 10.]) == [.1, 1., 10.]
    assert m.ridge_grid("A2", [1., 10.], [.1, 1., 10.]) == [1., 10.]


def test_turn_order_guard_rejects_repeated_colour_and_supports_handicap():
    m = importlib.import_module("pid_choice_data_audit")
    m.validate_turn_order("(;SZ[19];B[dd];W[pp];B[])")
    m.validate_turn_order("(;SZ[19]HD[2];B[dd];W[qq])")
    with pytest.raises(ValueError, match="turn"):
        m.validate_turn_order("(;SZ[19];B[dd];B[pp])")
    with pytest.raises(ValueError, match="turn"):
        m.validate_turn_order("(;SZ[19];W[pp])")
