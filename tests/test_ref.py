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
