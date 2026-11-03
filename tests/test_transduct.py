import numpy as np

from aicup_pid import transduct
from aicup_pid.components import features_all, fit_from_parsed, fit_local, parse_all, score_components
from tests.synth import black_table, sgf_lookup, training_frame


def test_margins_of():
    S = np.array([[3.0, 1.0, 0.0], [0.5, 0.4, 2.0]])
    assert transduct.margins_of(S, [0, 2]).tolist() == [2.0, 1.5]
    assert transduct.margins_of(np.zeros((0, 3)), []).shape == (0,)


def test_self_train_keeps_or_improves_planted_habits():
    frame, _ = training_frame(folds=(1,), players_per_fold=6, games_per_player=60, seed=11)  # 30 Black games each
    sgfs = {k: s for k, (s, c) in sgf_lookup(frame).items() if c == "B"}
    by_all = black_table(frame).groupby("player_id")["game_key"].apply(sorted).to_dict()
    by_player = {p: ks[:10] for p, ks in by_all.items()}  # 10 candidate games
    q_keys, truth = [], []
    for p, ks in by_all.items():  # two questions per player from the remaining games
        q_keys += [ks[10:15], ks[15:25]]
        truth += [p, p]
    moves, feats = parse_all(sgfs), features_all(sgfs)
    cfg = {"depth": 8, "local_start": 0}
    opening, style = fit_from_parsed(by_player, moves, feats, cfg)
    local = fit_local(by_player, moves, cfg)
    S = score_components(transduct.THREE, opening, style, q_keys, moves, feats, local)
    weights = [1.0, 0.5, 0.3]
    base = transduct.fuse(S, weights)
    idx, S_f, info = transduct.self_train(by_player, moves, feats, cfg, q_keys, S, weights, tau=0.0, rounds=1, assign=False)
    players = info["players"]
    assert idx.shape == (12, 5) and S_f.shape == (12, 6) and info["rounds"][0]["confident"] == 12 and info["rounds"][0]["games_added"] == 90
    before = np.mean([players[int(np.argmax(base[q]))] == truth[q] for q in range(12)])
    after = np.mean([players[idx[q, 0]] == truth[q] for q in range(12)])
    assert after >= before - 1e-9
    # a huge threshold: nobody is confident, nothing changes
    idx0, S0, info0 = transduct.self_train(by_player, moves, feats, cfg, q_keys, S, weights, tau=1e9, rounds=2)
    assert info0["rounds"] == [{"round": 1, "confident": 0, "games_added": 0, "margin_threshold": 1e9}] and np.allclose(S0, base)
    # two rounds run when something is confident
    _, _, info2 = transduct.self_train(by_player, moves, feats, cfg, q_keys, S, weights, tau=0.0, rounds=2)
    assert len(info2["rounds"]) == 2
