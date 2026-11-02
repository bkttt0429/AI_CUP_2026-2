import json

import pandas as pd
import pytest

import pid_cv
import predict_pid
from aicup_pid import pools
from tests.synth import black_table, official_files, sgf_lookup, training_frame


def _patch_data(monkeypatch, frame, folds_df, tmp_path):
    lookup = sgf_lookup(frame)
    monkeypatch.setattr(pid_cv.pools, "black_games", lambda train_dir=None: black_table(frame))
    monkeypatch.setattr(pid_cv, "load_games", lambda keys, train_dir=None: {k: lookup[k] for k in keys})
    folds_csv = tmp_path / "folds.csv"
    folds_df.to_csv(folds_csv, index=False)
    return folds_csv


def test_pid_cv_cli(tmp_path, monkeypatch, capsys):
    frame, folds_df = training_frame(folds=(1, 2, 3, 4), players_per_fold=5, games_per_player=40)  # 20 Black games per player
    folds_csv = _patch_data(monkeypatch, frame, folds_df, tmp_path)
    out, fus = tmp_path / "pid_cv.json", tmp_path / "pid_fusion.json"
    args = ["--pools", "1,2", "3,4", "--n-players", "8", "--cand-games", "10", "--depth", "8", "--n-boot", "100",
            "--folds-csv", str(folds_csv), "--out", str(out), "--fusion-out", str(fus)]
    assert pid_cv.main(args) == 0
    printed = capsys.readouterr().out
    res = json.loads(out.read_text())
    assert len(res["pools"]) == 2 and res["pools"][0]["pool"]["folds"] == [1, 2] and res["pools"][0]["pool"]["players"] == 8
    for p in res["pools"]:
        assert set(p["scores"]) == {"opening", "style", "fusion"} and len(p["fusion_weights"]) == 2
        for s in p["scores"].values():
            assert 0 <= s["score"] <= 1 and set(s["by_n"]) <= {"5", "10", "15", "20"}
    assert set(res["comparisons"]["pre_specified"]) == {"fusion vs opening"} and "pass" in res["comparisons"]["pre_specified"]["fusion vs opening"]
    assert res["questions"] == sum(p["pool"]["questions"] for p in res["pools"])
    assert res["pools"][0]["scores"]["opening"]["top1"] >= 0.5  # planted openings are recoverable from 10 candidate games
    doc = json.loads(fus.read_text())
    assert doc["components"] == ["opening", "style"] and len(doc["weights"]) == 2 and doc["fitted_on_pools"] == [[1, 2], [3, 4]]
    assert "[pre_specified] fusion vs opening" in printed and "wrote" in printed
    # fold 0 is refused; one pool gives equal weights
    assert pid_cv.main([*args, "--pools", "0,1"]) == 2 and "fold 0" in capsys.readouterr().err
    assert pid_cv.main(["--pools", "1,2", "--n-players", "8", "--cand-games", "10", "--depth", "8", "--n-boot", "50", "--folds-csv", str(folds_csv),
                        "--out", str(tmp_path / "one.json")]) == 0
    assert json.loads((tmp_path / "one.json").read_text())["pools"][0]["fusion_weights"] is None


def test_predict_pid_cli(tmp_path, capsys):
    frame_pub, _ = training_frame(folds=(1,), players_per_fold=6, games_per_player=40, seed=5)
    frame_priv, _ = training_frame(folds=(2,), players_per_fold=6, games_per_player=40, seed=6)
    cp1, qp1, pub_players = official_files(tmp_path, frame_pub, "pub_")
    cp2, qp2, priv_players = official_files(tmp_path, frame_priv, "priv_")
    fus = tmp_path / "fusion.json"
    fus.write_text(json.dumps({"components": ["opening", "style"], "weights": [1.0, 0.5], "cfg": {"depth": 8}}))
    out = tmp_path / "submission_player.csv"
    args = ["--candidates", str(cp1), str(cp2), "--questions", str(qp1), str(qp2), "--fusion", str(fus), "--out", str(out), "--expect-rows", "6"]
    assert predict_pid.main(args) == 0
    sub = pd.read_csv(out, dtype=str)
    assert list(sub.columns) == pools.COLUMNS and sub["question_id"].tolist() == [f"pub_q_{i:04d}" for i in (1, 2, 3)] + [f"priv_q_{i:04d}" for i in (1, 2, 3)]
    assert all(set(sub.iloc[i, 1:]) <= set(pub_players) for i in range(3)) and all(set(sub.iloc[i, 1:]) <= set(priv_players) for i in range(3, 6))
    assert sub.apply(lambda r: len(set(r[1:])) == 5, axis=1).all()
    rep = json.loads(out.with_suffix(".report.json").read_text())
    assert rep["problems"] == [] and len(rep["pools"]) == 2 and rep["pools"][0]["n_candidates"] == 6
    # the planted openings make the true player the Top-1 for most questions
    truth = {f"pub_q_{i + 1:04d}": f"pub_p_{pid}" for i, pid in enumerate(frame_pub[frame_pub['color'] == 'B']['player_id'].unique()[:3])}
    assert sum(sub.set_index("question_id").loc[q, "top_1"] == t for q, t in truth.items()) >= 2
    # candidate game ids that repeat (unique only within a player, as in the training files) must not drop any game
    cands = pd.read_csv(cp1, dtype=str, keep_default_na=False)
    cands["game_id"] = [f"g_{i % 16:03d}" for i in range(len(cands))]
    cp_rep = tmp_path / "cands_repeated.csv"
    cands.to_csv(cp_rep, index=False)
    assert predict_pid.main(["--candidates", str(cp_rep), "--questions", str(qp1), "--fusion", str(fus), "--out", str(tmp_path / "rep.csv")]) == 0
    rep_r = json.loads((tmp_path / "rep.csv").with_suffix(".report.json").read_text())["pools"][0]
    assert rep_r["n_candidate_games_used"] == rep_r["n_candidate_games"] == 96 and rep_r["candidate_game_ids_unique"] is False
    assert pd.read_csv(tmp_path / "rep.csv", dtype=str)["top_1"].tolist() == sub["top_1"].tolist()[:3]
    # the assignment modes: on applies it; auto applies it only when the Top-1 picks are (nearly) distinct
    assert predict_pid.main([*args, "--assign", "on"]) == 0
    rep_on = json.loads(out.with_suffix(".report.json").read_text())["pools"][0]["assign"]
    assert rep_on["mode"] == "on" and rep_on["applied"] is True and rep_on["top1_collisions"]["n_questions"] == 3 and "n_top1_changed" in rep_on
    assert predict_pid.main([*args, "--assign", "auto"]) == 0
    rep_auto = json.loads(out.with_suffix(".report.json").read_text())["pools"][0]["assign"]
    assert rep_auto["applied"] == rep_auto["top1_collisions"]["one_to_one_likely"]
    assert rep["pools"][0]["assign"] == {"mode": "off", "applied": False, "top1_collisions": rep["pools"][0]["assign"]["top1_collisions"], "n_top1_changed": 0}
    # a fusion document may name a subset of the components (the opening-only fallback); unknown names or a weight mismatch are refused
    (tmp_path / "opening.json").write_text(json.dumps({"components": ["opening"], "weights": [1.0], "cfg": {"depth": 8}}))
    assert predict_pid.main([*args[:-4], "--fusion", str(tmp_path / "opening.json"), "--out", str(tmp_path / "op.csv")]) == 0
    assert json.loads((tmp_path / "op.csv").with_suffix(".report.json").read_text())["fusion"]["components"] == ["opening"]
    (tmp_path / "bad.json").write_text(json.dumps({"components": ["foo"], "weights": [1.0]}))
    assert predict_pid.main([*args[:-4], "--fusion", str(tmp_path / "bad.json"), "--out", str(out)]) == 2
    (tmp_path / "bad2.json").write_text(json.dumps({"components": ["opening", "style"], "weights": [1.0]}))
    assert predict_pid.main([*args[:-4], "--fusion", str(tmp_path / "bad2.json"), "--out", str(out)]) == 2
    with pytest.raises(SystemExit):
        predict_pid.main(["--candidates", str(cp1), "--questions", str(qp1), str(qp2), "--fusion", str(fus), "--out", str(out)])
