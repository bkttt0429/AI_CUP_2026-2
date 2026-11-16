import importlib
import numpy as np
import pytest


def test_subsample_is_deterministic_and_cannot_repeat_or_pad():
    m = importlib.import_module("pid_reference_policy")
    ix = m.position_indices(101, 32)
    assert len(ix) == len(set(ix)) == 32
    assert ix[0] == 0 and ix[-1] == 100
    assert np.array_equal(ix, m.position_indices(101, 32))
    with pytest.raises(ValueError):
        m.position_indices(20, 32)


def test_duplicate_physical_game_cannot_cross_players_or_splits():
    m = importlib.import_module("pid_reference_policy")
    g = {"owner": "a", "split": "train", "key": "1", "color": "B", "sgf": "(;SZ[19];B[dd];W[pp])"}
    with pytest.raises(ValueError, match="duplicate"):
        m.validate_games([g, dict(g, owner="b", split="test", key="2", color="W")], require_counts=False)


def test_split_selection_is_order_independent_and_never_uses_fold0():
    m = importlib.import_module("pid_reference_policy")
    rows = [{"owner": str(p), "key": f"{p}-{g}", "color": "B", "sgf": f"(;SZ[19]C[{p}-{g}];B[dd])"} for p in range(8) for g in range(10)]
    folds = {str(p): p % 5 for p in range(8)}
    a = m.select_games(rows, folds, {"train": 1, "dev": 1, "test": 1})
    b = m.select_games(rows[::-1], folds, {"train": 1, "dev": 1, "test": 1})
    assert a == b
    assert len(a) == 30
    assert all(folds[g["owner"]] != 0 for g in a)
    groups = {s: {g["owner"] for g in a if g["split"] == s} for s in ("train", "dev", "test")}
    assert not groups["train"] & (groups["dev"] | groups["test"])
