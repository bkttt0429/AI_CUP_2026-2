import numpy as np
import pandas as pd
import pytest

from aicup_pid import pools
from tests.synth import black_table, training_frame


def test_make_pool_structure_and_reproducibility(tmp_path):
    frame, folds_df = training_frame(folds=(1, 2), players_per_fold=5, games_per_player=40)  # 20 Black games each
    games = black_table(frame)
    elig = pools.eligible_players(games, folds_df, [1, 2], min_black=15)
    assert len(elig) == 10 and list(elig.columns) == ["player_id", "rank", "n_black"]
    cands, qs, info = pools.make_pool(games, folds_df, [1, 2], n_players=4, cand_games=10, sizes=(2, 3, 5), seed=7, pool_id="t")
    assert info["players"] == 4 and info["eligible"] == 10 and len(cands) == 40
    assert set(qs["n_games"]) == {2, 3, 5} and info["questions_by_n"] == {"2": 4, "3": 4, "5": 4}  # 10 remaining games: 2 + 3 + 5
    for p, g in qs.groupby("player_id"):
        used = [k for ks in g["game_keys"] for k in ks]
        assert len(used) == len(set(used)) and not set(used) & set(cands.loc[cands["player_id"] == p, "game_key"])
    assert qs["question_id"].tolist() == [f"t_q_{i:04d}" for i in range(1, 13)]
    again = pools.make_pool(games, folds_df, [1, 2], n_players=4, cand_games=10, sizes=(2, 3, 5), seed=7, pool_id="t")
    assert again[0].equals(cands) and again[1]["game_keys"].tolist() == qs["game_keys"].tolist()
    other = pools.make_pool(games, folds_df, [1, 2], n_players=4, cand_games=10, sizes=(2, 3, 5), seed=8, pool_id="t")
    assert other[0]["game_key"].tolist() != cands["game_key"].tolist()
    with pytest.raises(ValueError, match="no player"):
        pools.make_pool(games, folds_df, [1, 2], cand_games=100)


def test_official_readers_score_and_submission(tmp_path):
    frame, _ = training_frame(folds=(1,), players_per_fold=3, games_per_player=4)
    cand = frame[frame["color"] == "B"].rename(columns={"sgf_content": "sgf_content"})[["game_id", "player_id", "sgf_content", "color"]]
    path = tmp_path / "cands.csv"
    cand.to_csv(path, index=False)
    c = pools.read_candidates(path)
    assert list(c.columns) == ["player_id", "game_id", "sgf", "color"] and len(c) == 6
    bad = cand.copy()
    bad.loc[bad.index[0], "color"] = "W"
    bad.to_csv(tmp_path / "bad.csv", index=False)
    with pytest.raises(ValueError, match="not Black"):
        pools.read_candidates(tmp_path / "bad.csv")
    assert pools.question_score(["a", "b", "c", "d", "e"], "a") == 1.0 and pools.question_score(["a", "b", "c", "d", "e"], "c") == pytest.approx(np.exp(-2))
    assert pools.question_score(["a", "b", "c", "d", "e"], "z") == 0.0
    s = pools.summarise([["a", "b", "c", "d", "e"], ["x", "a", "c", "d", "e"], ["z", "y", "w", "v", "u"]], ["a", "a", "a"], [5, 10, 5])
    assert s["top1"] == pytest.approx(1 / 3) and s["top5"] == pytest.approx(2 / 3) and s["mrr"] == pytest.approx((1 + 0.5) / 3)
    assert s["score"] == pytest.approx((1 + np.exp(-1)) / 3) and s["by_n"]["10"]["score"] == pytest.approx(np.exp(-1))
    rows = [("pub_q_0001", ["p1", "p2", "p3", "p4", "p5"]), ("priv_q_0001", ["q1", "q2", "q3", "q4", "q5"])]
    out = tmp_path / "submission_player.csv"
    table = pools.write_submission(rows, out)
    assert list(table.columns) == pools.COLUMNS and len(table) == 2
    pool_players = {"pub_": {f"p{i}" for i in range(1, 9)}, "priv_": {f"q{i}" for i in range(1, 9)}}
    assert pools.validate_submission(out, ["pub_q_0001", "priv_q_0001"], pool_players, expect_rows=2) == []
    assert any("rows" in p for p in pools.validate_submission(out, ["pub_q_0001", "priv_q_0001"], pool_players, expect_rows=800))
    assert any("order" in p for p in pools.validate_submission(out, ["priv_q_0001", "pub_q_0001"], pool_players))
    bad_rows = [("pub_q_0001", ["p1", "p1", "p3", "p4", "p5"]), ("priv_q_0001", ["p1", "q2", "q3", "q4", "q5"])]
    pools.write_submission(bad_rows, out)
    problems = pools.validate_submission(out, ["pub_q_0001", "priv_q_0001"], pool_players)
    assert any("distinct" in p for p in problems) and any("not in the priv_ pool" in p for p in problems)
    pools.write_submission([("priv_q_0001", ["q1", "q2", "q3", "q4", "q5"]), ("pub_q_0001", ["p1", "p2", "p3", "p4", "p5"])], out)
    assert any("public question after a private" in p for p in pools.validate_submission(out, ["priv_q_0001", "pub_q_0001"], pool_players))
    pd.DataFrame({"question_id": ["pub_q_0001"], "top_1": ["p1"]}).to_csv(out, index=False)
    assert pools.validate_submission(out, ["pub_q_0001"], pool_players)[0].startswith("header")


def test_one_per_player_fills_sizes_greedily():
    frame, folds_df = training_frame(folds=(1, 2), players_per_fold=6, games_per_player=60)  # 30 Black games each
    cands, qs, info = pools.make_pool_one_per_player(black_table(frame), folds_df, [1, 2], n_players=8, cand_games=10, per_size=2,
                                                     sizes=(5, 10, 15, 20), seed=0)
    assert info["players"] == 8 and info["one_per_player"] is True and info["per_size"] == 2
    assert len(qs) == 8 and qs["player_id"].is_unique
    assert qs["n_games"].value_counts().to_dict() == {20: 2, 15: 2, 10: 2, 5: 2}
    assert (cands.groupby("player_id").size() == 10).all()
    for r in qs.itertuples():
        own = set(cands[cands["player_id"] == r.player_id]["game_key"])
        assert len(r.game_keys) == r.n_games and not set(r.game_keys) & own
    assert info["questions_by_n"] == {"5": 2, "10": 2, "15": 2, "20": 2}
    # reproducible
    cands2, qs2, _ = pools.make_pool_one_per_player(black_table(frame), folds_df, [1, 2], n_players=8, cand_games=10, per_size=2, seed=0)
    assert cands2.equals(cands) and qs2["game_keys"].tolist() == qs["game_keys"].tolist()


def test_one_per_player_short_players_get_the_largest_affordable_size():
    frame, folds_df = training_frame(folds=(1,), players_per_fold=3, games_per_player=32)  # 16 Black games: 6 spare after 10 candidates
    cands, qs, info = pools.make_pool_one_per_player(black_table(frame), folds_df, [1], n_players=3, cand_games=10, per_size=1,
                                                     sizes=(5, 10, 15, 20), seed=0)
    assert len(qs) == 3 and sorted(qs["n_games"]) == [5, 5, 5]
    assert info["questions_by_n"] == {"5": 3, "10": 0, "15": 0, "20": 0}


def test_file_order_split_takes_the_earliest_games_as_candidates():
    frame, folds_df = training_frame(folds=(1,), players_per_fold=4, games_per_player=40)  # 20 Black games each
    games = black_table(frame)
    cands, qs, info = pools.make_pool(games, folds_df, [1], n_players=4, cand_games=10, sizes=(5,), seed=0, order="file")
    assert info["order"] == "file"
    for p, g in cands.groupby("player_id"):
        own = sorted(games[games["player_id"] == p]["game_key"])
        assert sorted(g["game_key"]) == own[:10]
        q = qs[qs["player_id"] == p]["game_keys"].iloc[0]
        assert list(q) == own[10:15]
    shuffled, _, info_s = pools.make_pool(games, folds_df, [1], n_players=4, cand_games=10, sizes=(5,), seed=0)
    assert info_s["order"] == "shuffle" and sorted(shuffled["game_key"]) != sorted(cands["game_key"])
    c1, q1, i1 = pools.make_pool_one_per_player(games, folds_df, [1], n_players=4, cand_games=10, per_size=4, sizes=(5, 10), seed=0, order="file")
    assert i1["order"] == "file" and all(list(r.game_keys)[0] > max(c1[c1["player_id"] == r.player_id]["game_key"]) for r in q1.itertuples())
    with pytest.raises(ValueError):
        pools.make_pool(games, folds_df, [1], n_players=4, cand_games=10, order="random")

