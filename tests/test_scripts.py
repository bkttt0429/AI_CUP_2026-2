import json

import pandas as pd
import pytest

import pid_cv
import predict_pid
from aicup_pid import pools
from tests.synth import black_table, sgf_lookup, training_frame


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


def _official_files(tmp_path, frame, prefix, n_q=3):
    """Candidates (all Black games of each player but the last four) and questions (those four, 2 or 4 games) in the official format."""
    black = frame[frame["color"] == "B"]
    cand_rows, q_rows = [], []
    for i, (pid, g) in enumerate(black.groupby("player_id")):
        games = g["sgf_content"].tolist()
        cand_rows += [{"game_id": f"g_{len(cand_rows) + 1:07d}", "player_id": f"{prefix}p_{pid}", "sgf_content": s, "color": "B"} for s in games[:-4]]
        if i < n_q:
            qg = games[-4:] if i % 2 else games[-2:]
            row = {"question_id": f"{prefix}q_{i + 1:04d}", "num_games": len(qg)}
            for j in range(20):
                row[f"sgf_{j + 1}"] = qg[j] if j < len(qg) else ""
                row[f"color_{j + 1}"] = "B" if j < len(qg) else ""
            q_rows.append(row)
    cp, qp = tmp_path / f"cands_{prefix}csv", tmp_path / f"test_{prefix}csv"
    pd.DataFrame(cand_rows).to_csv(cp, index=False)
    pd.DataFrame(q_rows).to_csv(qp, index=False)
    return cp, qp, [f"{prefix}p_{pid}" for pid in black["player_id"].unique()]


def test_predict_pid_cli(tmp_path, capsys):
    frame_pub, _ = training_frame(folds=(1,), players_per_fold=6, games_per_player=40, seed=5)
    frame_priv, _ = training_frame(folds=(2,), players_per_fold=6, games_per_player=40, seed=6)
    cp1, qp1, pub_players = _official_files(tmp_path, frame_pub, "pub_")
    cp2, qp2, priv_players = _official_files(tmp_path, frame_priv, "priv_")
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
    # a fusion document with other components is refused
    (tmp_path / "bad.json").write_text(json.dumps({"components": ["opening"], "weights": [1.0]}))
    assert predict_pid.main([*args[:-4], "--fusion", str(tmp_path / "bad.json"), "--out", str(out)]) == 2
    with pytest.raises(SystemExit):
        predict_pid.main(["--candidates", str(cp1), "--questions", str(qp1), str(qp2), "--fusion", str(fus), "--out", str(out)])
