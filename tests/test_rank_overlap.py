"""scripts/pid_rank_overlap.py: identifying a rank question's player among all training players (AI_CUP_2026-1 issue #22)."""
import json

import numpy as np
import pandas as pd
import pytest

import pid_rank_overlap as ro
from aicup.metric import LEVELS
from aicup_pid import opening
from tests.synth import habit_of, random_game, training_frame


def _training_table(frame, depth):
    t = frame[["player_id", "rank", "color"]].copy()
    t["game_key"] = frame["rank"] + ":" + frame["game_id"]
    t["moves"] = [opening.parse_prefix(s, depth) for s in frame["sgf_content"]]
    return t.reset_index(drop=True)


def _questions(frame, fold, tmp_path):
    """Per player of `fold`: the last 10 games (mixed colours) as one 10-game question and two disjoint 5-game questions."""
    rows = []
    for pid, g in frame[frame["fold"] == fold].groupby("player_id"):
        keys = (g["rank"] + ":" + g["game_id"]).tolist()[-10:]
        rank = g["rank"].iloc[0]
        rows.append({"question_id": f"f{fold}_{rank}_{pid[:8]}_n10_0", "rank": rank, "player_id": pid, "n_games": 10, "game_keys": ";".join(keys)})
        for j in range(2):
            rows.append({"question_id": f"f{fold}_{rank}_{pid[:8]}_n5_{j}", "rank": rank, "player_id": pid, "n_games": 5, "game_keys": ";".join(keys[j * 5:(j + 1) * 5])})
    path = tmp_path / f"questions_fold{fold}.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path, pd.DataFrame(rows)


def test_white_target_tokens_are_symmetry_invariant_and_whites_moves():
    rng = np.random.RandomState(3)
    sgf = random_game(rng, habit_of(5), n_moves=30)
    moves = opening.parse(sgf)
    base = opening.tokens(moves, 12, color="W")
    assert len(base) == 6 and all(len(fctx) % 2 == 1 and len(own) == (len(fctx) - 1) // 2 for fctx, own, _ in base)  # White moves 2, 4, ...
    for k in range(8):
        sym = [(c, *opening._sym(k, x, y)) for c, x, y in moves]
        assert opening.tokens(sym, 12, color="W") == base, k
    assert opening.tokens(moves, 12, color="B") != base
    assert opening.parse_prefix(sgf, 7) == moves[:7] and opening.parse_prefix(sgf, 1000) == moves
    with pytest.raises(ValueError):
        opening.OpeningModel(color="X")
    assert opening.OpeningModel(color="W").fit({"p": [moves]}).state()["color"] == "W"


def test_rank_overlap_cli(tmp_path, monkeypatch, capsys):
    frame, folds_df = training_frame(folds=(1, 2, 3, 4), players_per_fold=10, games_per_player=40)  # 40 players (all ten ranks per fold), 20 Black + 20 White each
    table = _training_table(frame, 8)
    monkeypatch.setattr(ro, "load_training", lambda train_dir, depth, workers=1: table)
    folds_csv = tmp_path / "folds.csv"
    folds_df.to_csv(folds_csv, index=False)
    calib_csv, _ = _questions(frame, 1, tmp_path)
    eval_csv, evalq = _questions(frame, 2, tmp_path)
    # a fused_opp proba table that always predicts "D": the identified rank should beat it
    proba = pd.DataFrame({"question_id": evalq["question_id"].str.replace(r"^f\d+_", "", regex=True), "player_id": evalq["player_id"], "n_games": evalq["n_games"], "rank": evalq["rank"]})
    for lv in LEVELS:
        proba[f"fused_opp_avg_p_{lv}"] = 0.91 if lv == "D" else 0.01
    proba_csv = tmp_path / "proba.csv"
    proba.to_csv(proba_csv, index=False)
    out = tmp_path / "rank_overlap.json"
    args = ["--train-dir", str(tmp_path), "--folds-csv", str(folds_csv), "--questions", str(calib_csv), str(eval_csv), "--fused-proba", str(proba_csv),
            "--siblings", "--depth", "8", "--n-boot", "50", "--out", str(out)]
    assert ro.main(args) == 0
    printed = capsys.readouterr().out
    res = json.loads(out.read_text())
    assert res["candidates"] == {"players": 40, "games": 1600, "held_out_games": 200}
    c, e = res["calibration"], res["evaluation"]
    assert c["fold"] == 1 and e["fold"] == 2 and c["questions"] == 30 and e["questions"] == 30
    assert e["top1"] >= 0.5 and e["top5"] >= e["top1"] and set(e["top1_by_n"]) == {"5", "10"}, "planted openings identify the player among 40"
    assert 0 <= e["id_rank_score"] <= 1 and c["precision_target"] == 0.98 and len(c["curve"]) >= 1
    if np.isfinite(c["tau"]):
        assert e["precision_at_tau"] is None or 0 <= e["precision_at_tau"] <= 1
    comb = res["combined"]
    assert comb["questions"] == 30 and comb["fused_opp"] < 0.3 and comb["combined"] >= comb["fused_opp"] and isinstance(comb["pass"], bool)
    assert set(comb["combined_vs_fused_opp"]) >= {"diff", "ci95"}
    assert set(res["siblings"]) == {"5"} and res["siblings"]["5"]["with_sibling"] == 20 and 0 <= res["siblings"]["5"]["top1_same_player"] <= 1
    assert "calibration (fold 1)" in printed and "evaluation (fold 2)" in printed and "wrote" in printed
    # calibration and evaluation must not share players; --test-csv needs --tau
    assert ro.main(args[:5] + ["--questions", str(calib_csv), str(calib_csv), "--out", str(out)]) == 2
    assert ro.main(["--train-dir", str(tmp_path), "--folds-csv", str(folds_csv), "--test-csv", str(eval_csv), "--out", str(out)]) == 2
    # official-format test questions from fold 3 players (all their games are candidates here) are identified
    rows = []
    for i, (pid, g) in enumerate(frame[frame["fold"] == 3].groupby("player_id")):
        games = list(zip(g["sgf_content"], g["color"]))[-6:]
        row = {"question_id": f"pub_q_{i + 1:04d}", "num_games": len(games)}
        for j in range(20):
            row[f"sgf_{j + 1}"] = games[j][0] if j < len(games) else ""
            row[f"color_{j + 1}"] = games[j][1] if j < len(games) else ""
        row["truth"] = pid
        rows.append(row)
    test_csv = tmp_path / "rank_test.csv"
    pd.DataFrame(rows).drop(columns=["truth"]).to_csv(test_csv, index=False)
    out2, table_csv = tmp_path / "overlap_test.json", tmp_path / "overlap_test.csv"
    assert ro.main(["--train-dir", str(tmp_path), "--folds-csv", str(folds_csv), "--test-csv", str(test_csv), "--tau", "1.0", "--depth", "8",
                    "--out", str(out2), "--out-table", str(table_csv)]) == 0
    res2, tab = json.loads(out2.read_text()), pd.read_csv(table_csv, dtype=str)
    assert res2["mode"] == "test" and res2["questions"] == 10 and set(tab.columns) >= {"question_id", "n_games", "top1_player", "top1_rank", "margin", "above_tau", "top5_players"}
    assert (tab["top1_player"].to_numpy() == np.array([r["truth"] for r in rows])).mean() >= 0.8 and tab["top1_rank"].isin(LEVELS).all()
    assert 0 <= res2["share_above_tau"] <= 1 and set(res2["share_above_tau_by_n"]) == {"6"}
