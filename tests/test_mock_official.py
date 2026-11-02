import json

import pandas as pd

import pid_mock_official
from aicup_pid import pools
from tests.synth import black_table, sgf_lookup, training_frame


def test_mock_official_writes_official_format(tmp_path, monkeypatch):
    frame, folds_df = training_frame(folds=(1, 2, 3, 4), players_per_fold=5, games_per_player=60)  # 30 Black games per player
    lookup = sgf_lookup(frame)
    monkeypatch.setattr(pid_mock_official.pools, "black_games", lambda train_dir=None: black_table(frame))
    monkeypatch.setattr(pid_mock_official, "load_games", lambda keys, train_dir=None: {k: lookup[k] for k in keys})
    folds_csv = tmp_path / "folds.csv"
    folds_df.to_csv(folds_csv, index=False)
    out = tmp_path / "mock"
    rc = pid_mock_official.main(["--pools", "1,2", "3,4", "--out-dir", str(out), "--n-players", "8", "--cand-games", "10", "--per-size", "2",
                                 "--folds-csv", str(folds_csv)])
    assert rc == 0
    cands = pools.read_candidates(out / "player_identification_candidates_public.csv")
    qs = pools.read_questions(out / "player_identification_test_public.csv")
    assert cands["player_id"].str.startswith("pub_p_").all() and cands["game_id"].str.startswith("pub_g_").all() and len(cands) == 80
    assert cands["game_id"].is_unique and (cands["color"] == "B").all()
    assert len(qs) == 8 and all(q["declared_games"] == len(q["games"]) for q in qs) and sorted(q["declared_games"] for q in qs) == [5, 5, 10, 10, 15, 15, 20, 20]
    assert all(c == "B" for q in qs for _, c in q["games"])
    truth = pd.read_csv(out / "truth_public.csv", dtype=str)
    assert list(truth.columns) == ["question_id", "player_id", "train_player_id", "rank", "n_games"]
    assert set(truth["player_id"]) <= set(cands["player_id"]) and truth["question_id"].tolist() == [q["question_id"] for q in qs]
    assert truth["player_id"].is_unique
    cand_sgfs = set(cands["sgf"])
    assert not any(s in cand_sgfs for q in qs for s, _ in q["games"])  # question games are not candidate games
    priv = pools.read_candidates(out / "player_identification_candidates_private.csv")
    assert priv["player_id"].str.startswith("priv_p_").all() and (out / "truth_private.csv").exists()
    info = json.loads((out / "mock_info.json").read_text())
    assert [p["folds"] for p in info["pools"]] == [[1, 2], [3, 4]] and info["pools"][0]["one_per_player"] and "pid_git_head" in info
    assert pid_mock_official.main(["--pools", "0,1", "--out-dir", str(tmp_path / "x"), "--folds-csv", str(folds_csv)]) == 2
