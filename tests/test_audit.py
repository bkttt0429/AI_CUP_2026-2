"""scripts/pid_audit.py: the report-only leak audit of the simulated pools (planted duplicates are found, symmetric copies included)."""
import json
import re

import pytest

import pid_audit
from aicup_pid import pools
from aicup_pid.opening import parse
from tests.synth import black_table, sgf_lookup, sgf_of, training_frame


def _patch(monkeypatch, frame, folds_df, tmp_path):
    lookup = sgf_lookup(frame)
    monkeypatch.setattr(pid_audit.pools, "black_games", lambda train_dir=None: black_table(frame))
    monkeypatch.setattr(pid_audit, "load_games", lambda keys, train_dir=None: {k: lookup[k] for k in keys})
    folds_csv = tmp_path / "folds.csv"
    folds_df.to_csv(folds_csv, index=False)
    return folds_csv


def _transposed(sgf):
    return re.sub(r";([BW])\[([a-s])([a-s])\]", r";\1[\3\2]", sgf)


def test_whole_game_and_head_are_symmetry_invariant():
    moves = [("B", 15, 3), ("W", 3, 15), ("B", 16, 16), ("W", 2, 2), ("B", 15, 9), ("W", 9, 9)]
    variants = [sgf_of(moves), _transposed(sgf_of(moves)), sgf_of([(c, 18 - x, y) for c, x, y in moves]), sgf_of([(c, x, 18 - y) for c, x, y in moves])]
    wholes = {pid_audit.whole_game(parse(s)) for s in variants}
    heads = {pid_audit.head(parse(s), 4) for s in variants}
    assert len(wholes) == 1 and len(next(iter(wholes))) == 6 and len(heads) == 1 and len(next(iter(heads))) == 4
    assert pid_audit.whole_game(parse(sgf_of(moves[:5]))) not in wholes, "a different game is a different key"


def test_pid_audit_cli_finds_planted_duplicates(tmp_path, monkeypatch, capsys):
    frame, folds_df = training_frame(folds=(1, 2, 3, 4), players_per_fold=5, games_per_player=40)  # 20 Black games per player
    # the audit builds the same pool as pid_cv.py, so plant duplicates where they will be seen: pool 1,2 only
    cands, qs, _ = pools.make_pool(black_table(frame), folds_df, [1, 2], n_players=8, cand_games=10, seed=0, pool_id="p12")
    key_of = dict(zip(frame["rank"] + ":" + frame["game_id"], frame.index))
    p, o = qs["player_id"].iloc[0], qs["player_id"].iloc[-1]
    assert p != o
    own_cand = cands[cands["player_id"] == p]["game_key"].iloc[0]
    other_cand = cands[cands["player_id"] == o]["game_key"].iloc[0]
    q_keys = [k for ks in qs[qs["player_id"] == p]["game_keys"] for k in ks]
    assert len(q_keys) >= 5
    sgf = frame.loc[key_of[own_cand], "sgf_content"]
    frame.loc[key_of[q_keys[0]], "sgf_content"] = sgf  # exact copy of an own candidate game
    frame.loc[key_of[q_keys[1]], "sgf_content"] = _transposed(sgf)  # the same game under a board symmetry
    frame.loc[key_of[q_keys[2]], "sgf_content"] = frame.loc[key_of[q_keys[3]], "sgf_content"]  # two query games of the same player
    frame.loc[key_of[other_cand], "sgf_content"] = frame.loc[key_of[q_keys[4]], "sgf_content"]  # a query game among another player's candidates
    folds_csv = _patch(monkeypatch, frame, folds_df, tmp_path)
    out = tmp_path / "pid_audit.json"
    args = ["--pools", "1,2", "3,4", "--n-players", "8", "--cand-games", "10", "--depth", "8", "--workers", "1", "--folds-csv", str(folds_csv), "--out", str(out)]
    assert pid_audit.main(args) == 0
    printed = capsys.readouterr().out
    res = json.loads(out.read_text())
    assert len(res["pools"]) == 2 and res["n_players"] == 8 and res["cand_games"] == 10 and res["depth"] == 8
    planted, clean = res["pools"]
    assert planted["folds"] == [1, 2] and planted["pool"]["players"] == 8 and planted["query_games"] == sum(qs["n_games"])
    assert planted["query_dup_in_own_candidates"] >= 2 and planted["query_dup_in_other_candidates"] >= 1 and planted["query_repeats_within_player"] >= 1
    assert planted["unique_candidate_games"] == planted["candidate_games"]  # the planted copies are query games, the candidates stay distinct
    assert planted["first12_in_own_candidates"] > 0 and planted["questions_without_duplicates"] <= planted["questions"]
    assert clean["query_dup_in_own_candidates"] == clean["query_dup_in_other_candidates"] == clean["query_repeats_within_player"] == 0
    assert clean["unique_candidate_games"] == clean["candidate_games"] and clean["questions_without_duplicates"] == clean["questions"]
    for pool in res["pools"]:
        assert set(pool["without_duplicates"]) == {"opening", "style"} and set(pool["opening_by_depth"]) == {str(d) for d in pid_audit.DEPTHS}
        for s in list(pool["without_duplicates"].values()) + list(pool["opening_by_depth"].values()) + [pool["negative_control"]]:
            assert 0 <= s["score"] <= 1
        # planted habits are recovered by the real model, not by candidates shuffled between players
        assert pool["negative_control"]["score"] < pool["opening_by_depth"]["8"]["score"]
        assert 0 <= pool["first12_in_random_other_candidates"] <= 1
    assert "shuffled candidates" in printed and "wrote" in printed
    with pytest.raises(SystemExit):
        pid_audit.main(args + ["--pools", "0,1"])
