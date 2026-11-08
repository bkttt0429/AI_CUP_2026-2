import numpy as np
import pandas as pd

from aicup.sgf import parse_moves
from aicup_pid import ref
from aicup_pid.components import fit_ref, score_ref
from tests.synth import black_table, training_frame
from tests.test_orient import oriented_game


def test_reference_players_selection():
    frame, folds_df = training_frame(folds=(1, 2, 3, 4), players_per_fold=4, games_per_player=16)  # 8 Black games each
    games = black_table(frame)
    refs = ref.reference_players(games, folds_df, (3, 4), min_black=8, cap=5, seed=0)
    assert len(refs) == 8 and all(len(v) == 5 for v in refs.values())
    assert set(refs) <= set(folds_df[folds_df.fold.isin([3, 4])].player_id)
    assert ref.reference_players(games, folds_df, (3, 4), min_black=9) == {}
    some = next(iter(refs))
    assert ref.reference_players(games, folds_df, (3, 4), min_black=8, cap=5, seed=0, exclude={some}).keys() == set(refs) - {some}
    assert ref.reference_players(games, folds_df, (3, 4), min_black=8, cap=5, seed=1)[some] != refs[some] or len(refs[some]) == 8  # the draw depends on the seed


def test_ref_model_profiles_match_like_players():
    rng = np.random.RandomState(0)
    # references: players with four distinct corner habits
    refs = {f"r{i}": [parse_moves(oriented_game(rng, i % 4, 40)) for _ in range(12)] for i in range(8)}
    model = ref.RefModel(orient_raw=1).fit(refs)
    assert model.players == sorted(refs) and model.logp.shape[0] == 8 and model.logp.dtype == np.float32
    v = model.vectors([parse_moves(oriented_game(rng, 0, 40))])
    assert v.shape == (1, 8) and abs(v.mean()) < 1e-9 and abs(v.std() - 1) < 1e-9
    # candidates and questions with the same habits: the question's profile is closest to its own-corner candidate
    cands = {f"c{c}": [parse_moves(oriented_game(rng, c, 40)) for _ in range(10)] for c in range(4)}
    C = np.stack([model.profile(cands[p]) for p in sorted(cands)])
    for c in range(4):
        q = [parse_moves(oriented_game(rng, c, 40)) for _ in range(5)]
        S = ref.cosine_scores(np.stack([model.profile(q)]), C)
        assert S.shape == (1, 4) and int(np.argmax(S[0])) == c
    assert model.profile([]).tolist() == [0.0] * 8 and model.state()["references"] == 8


def test_fit_and_score_ref_helpers():
    rng = np.random.RandomState(1)
    moves, refs, by_player = {}, {}, {}
    for i in range(6):
        for j in range(8):
            moves[f"r:{i}:{j}"] = parse_moves(oriented_game(rng, i % 4, 30))
            refs.setdefault(f"r{i}", []).append(f"r:{i}:{j}")
    for c in range(3):
        for j in range(6):
            moves[f"c:{c}:{j}"] = parse_moves(oriented_game(rng, c, 30))
            by_player.setdefault(f"p{c}", []).append(f"c:{c}:{j}")
    moves["q:0"], moves["q:1"] = parse_moves(oriented_game(rng, 2, 30)), parse_moves(oriented_game(rng, 2, 30))
    model = fit_ref(refs, moves, {"ref_orient_raw": 1})
    S = score_ref(model, by_player, [["q:0", "q:1"]], moves)
    assert S.shape == (1, 3) and int(np.argmax(S[0])) == 2 and np.all(np.abs(S) <= 1 + 1e-9)
    assert score_ref(model, by_player, [], moves).shape == (0, 3)


def test_ref_model_without_orientation_is_symmetry_invariant():
    import pytest
    from aicup_pid.symmetry import transform_sgf
    from tests.synth import habit_of, random_game
    rng = np.random.RandomState(3)
    raw = {f"r{i}": [random_game(rng, habit_of(i % 3), 60) for _ in range(10)] for i in range(6)}
    refs = {p: [parse_moves(s) for s in gs] for p, gs in raw.items()}
    m0, m2 = ref.RefModel(orient_raw=0).fit(refs), ref.RefModel(orient_raw=2).fit(refs)
    assert m0.orient is None and m0.state()["orient"] is None and m0.logp.shape[1] < m2.logp.shape[1]
    q = [random_game(rng, habit_of(1), 60) for _ in range(3)]
    rot = [transform_sgf(s, k) for s, k in zip(q, (1, 5, 6))]
    assert np.allclose(m0.profile([parse_moves(s) for s in q]), m0.profile([parse_moves(s) for s in rot]))
    assert fit_ref({p: [f"{p}:{j}" for j in range(10)] for p in raw}, {f"{p}:{j}": refs[p][j] for p in raw for j in range(10)}, {"ref_orient_raw": 0}).orient is None
    with pytest.raises(ValueError):
        ref.RefModel(orient_raw=-1)
