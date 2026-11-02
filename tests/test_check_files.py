import json

import pandas as pd
import pytest

import pid_check_files
from tests.synth import official_files, sgf_of, training_frame
from tests.test_dedup import _write_training


def _fusion(tmp_path):
    p = tmp_path / "fusion.json"
    p.write_text(json.dumps({"components": ["opening", "style"], "weights": [1.0, 0.5], "cfg": {"depth": 8}}))
    return str(p)


def test_cli_end_to_end_with_planted_overlap(tmp_path, capsys):
    frame, _ = training_frame(folds=(1,), players_per_fold=6, games_per_player=40, seed=5)  # 6 players, 20 Black games each
    train = tmp_path / "train"
    _write_training(frame, train)
    cp1, qp1, _ = official_files(tmp_path, frame, "pub_")  # every public game is a training game
    frame2, _ = training_frame(folds=(3,), players_per_fold=6, games_per_player=40, seed=6)
    cp2, qp2, _ = official_files(tmp_path, frame2, "priv_")  # none of the private games is
    out = tmp_path / "checks.json"
    rc = pid_check_files.main(["--candidates", str(cp1), str(cp2), "--questions", str(qp1), str(qp2), "--fusion", _fusion(tmp_path),
                               "--train-dir", str(train), "--out", str(out), "--model-games", "6", "--n-null", "5"])
    assert rc == 0
    res = json.loads(out.read_text())
    pub, priv = res["pools"]
    assert pub["prefix"] == "pub_" and priv["prefix"] == "priv_"
    assert pub["n_candidates"] == 6 and pub["n_candidate_games"] == 96 and pub["n_questions"] == 3 and pub["num_games"] == {"2": 2, "4": 1}
    assert pub["training_overlap"]["candidates"]["n_dup_games"] == 96 and pub["training_overlap"]["candidates"]["players_single_training_id"] == 6
    assert pub["training_overlap"]["questions"]["n_dup_games"] == 8
    assert priv["training_overlap"]["candidates"]["n_dup_games"] == 0 and priv["training_overlap"]["questions"]["n_dup_games"] == 0
    assert res["flags"]["A1_training_overlap"] is True and res["flags"]["A3i_opening_shift"] is False
    assert res["training"]["n_games"] == len(frame) and res["training"]["lengths_black"]["n"] == 120
    assert pub["question_vs_candidate_dups"] == 0 and pub["question_vs_question_dups"] == 0 and res["cross_pool_candidate_dups"] == 0
    assert set(pub["opening_shift"]) == {"4", "8"} and pub["opening_shift"]["4"]["n_query"] == 8 and len(pub["opening_shift"]["4"]["null"]) == 5
    si = pub["self_identification"]
    assert si["n_candidates"] == 6 and si["n_questions"] == 6 and si["scores"]["fusion"]["by_n"].keys() == {"5"}  # 16 games: 6 model + one 5-game question
    ml = pub["main_line"]
    assert len(ml["top1"]) == 3 and all(t[0].startswith("pub_q_") and t[1].startswith("pub_p_") for t in ml["top1"])
    assert ml["top1_collisions"]["n_questions"] == 3 and {"p05", "p25", "p50", "p75", "p95"} <= set(ml["margin_quantiles"])
    assert pub["legality"]["candidates"]["n_bad"] == 0 and pub["legality"]["questions"]["n"] == 8
    assert pub["lengths"]["candidates"]["n"] == 96 and pub["records"]["candidates"]["properties"]["B"] == 96 and pub["records"]["candidates"]["game_id_unique"] is True
    assert "pid_git_head" in res and res["fusion"]["weights"] == [1.0, 0.5]
    printed = capsys.readouterr().out
    assert "A1" in printed and "self-identification" in printed


def test_cli_no_training_and_bad_games(tmp_path):
    frame, _ = training_frame(folds=(1,), players_per_fold=6, games_per_player=40, seed=7)
    cp, qp, _ = official_files(tmp_path, frame, "pub_")
    q = pd.read_csv(qp, dtype=str, keep_default_na=False)
    q.loc[0, ["sgf_1", "sgf_2"]] = "garbage"  # question 1 has 2 games: both unusable
    q.loc[1, "sgf_1"] = sgf_of([("W", 3, 3), ("B", 15, 15), ("W", 3, 15)])  # question 2: one illegal game
    q.to_csv(qp, index=False)
    out = tmp_path / "checks.json"
    rc = pid_check_files.main(["--candidates", str(cp), "--questions", str(qp), "--fusion", _fusion(tmp_path), "--no-training", "--out", str(out),
                               "--model-games", "6", "--n-null", "3"])
    assert rc == 0
    res = json.loads(out.read_text())
    pub = res["pools"][0]
    assert res["training"] is None and res["flags"]["A1_training_overlap"] is None and pub["training_overlap"] is None
    assert pub["legality"]["questions"]["n_bad"] == 3 and len(pub["legality"]["questions"]["errors"]) == 3
    assert pub["main_line"]["top1"][0] == ["pub_q_0001", None] and pub["main_line"]["top1"][1][1].startswith("pub_p_")
    assert pub["main_line"]["top1_collisions"]["n_questions"] == 2 and pub["opening_shift"]["4"]["n_query"] == 5


def test_cli_cross_pool_duplicates(tmp_path):
    frame, _ = training_frame(folds=(1,), players_per_fold=4, games_per_player=40, seed=8)
    cp1, qp1, _ = official_files(tmp_path, frame, "pub_")
    cp2, qp2, _ = official_files(tmp_path, frame, "priv_")  # the same games under other ids
    out = tmp_path / "checks.json"
    rc = pid_check_files.main(["--candidates", str(cp1), str(cp2), "--questions", str(qp1), str(qp2), "--fusion", _fusion(tmp_path), "--no-training",
                               "--out", str(out), "--model-games", "6", "--n-null", "3"])
    assert rc == 0
    res = json.loads(out.read_text())
    assert res["cross_pool_candidate_dups"] == 64 and res["pools"][0]["question_vs_candidate_dups"] == 0


def test_cli_three_component_document(tmp_path):
    frame, _ = training_frame(folds=(1,), players_per_fold=4, games_per_player=40, seed=9)
    cp, qp, _ = official_files(tmp_path, frame, "pub_")
    f3 = tmp_path / "fusion3.json"
    f3.write_text(json.dumps({"components": ["opening", "style", "local"], "weights": [1.0, 0.5, 0.3], "cfg": {"depth": 8, "local_start": 0}}))
    out = tmp_path / "checks3.json"
    assert pid_check_files.main(["--candidates", str(cp), "--questions", str(qp), "--fusion", str(f3), "--no-training", "--out", str(out), "--model-games", "6", "--n-null", "3"]) == 0
    res = json.loads(out.read_text())
    assert res["fusion"]["components"] == ["opening", "style", "local"] and len(res["pools"][0]["main_line"]["top1"]) == 3
    assert res["pools"][0]["self_identification"]["n_candidates"] == 4  # check 7 stays on the production pair
    (tmp_path / "bad.json").write_text(json.dumps({"components": ["opening", "style"], "weights": [1.0]}))
    assert pid_check_files.main(["--candidates", str(cp), "--questions", str(qp), "--fusion", str(tmp_path / "bad.json"), "--no-training", "--out", str(out)]) == 2


def test_cli_refuses_unpaired_files(tmp_path):
    with pytest.raises(SystemExit):
        pid_check_files.main(["--candidates", "a.csv", "b.csv", "--questions", "c.csv", "--fusion", "f.json", "--out", str(tmp_path / "x.json")])
    assert pid_check_files.main(["--candidates", "missing.csv", "--questions", "missing2.csv", "--fusion", _fusion(tmp_path), "--no-training",
                                 "--out", str(tmp_path / "x.json")]) == 2
