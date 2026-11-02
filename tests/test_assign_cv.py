import json

import pid_assign_cv
from tests.synth import black_table, sgf_lookup, training_frame


def test_assign_cv_cli(tmp_path, monkeypatch, capsys):
    frame, folds_df = training_frame(folds=(1, 2, 3, 4), players_per_fold=5, games_per_player=60)  # 30 Black games per player
    lookup = sgf_lookup(frame)
    monkeypatch.setattr(pid_assign_cv.pools, "black_games", lambda train_dir=None: black_table(frame))
    monkeypatch.setattr(pid_assign_cv, "load_games", lambda keys, train_dir=None: {k: lookup[k] for k in keys})
    folds_csv = tmp_path / "folds.csv"
    folds_df.to_csv(folds_csv, index=False)
    out = tmp_path / "assign_cv.json"
    args = ["--pools", "1,2", "3,4", "--seeds", "0", "1", "--n-players", "8", "--cand-games", "10", "--per-size", "2", "--depth", "8",
            "--n-boot", "50", "--control", "--folds-csv", str(folds_csv), "--out", str(out)]
    assert pid_assign_cv.main(args) == 0
    res = json.loads(out.read_text())
    assert len(res["pools"]) == 4 and [p["pool"]["folds"] for p in res["pools"]] == [[1, 2], [1, 2], [3, 4], [3, 4]] and [p["pool"]["seed"] for p in res["pools"]] == [0, 1, 0, 1]
    for p in res["pools"]:
        assert set(p["scores"]) == {"fusion", "assign"} and p["pool"]["questions"] == 8 and len(p["fusion_weights"]) == 2
        assert p["assignment"]["n_questions"] == 8 and 0 <= p["assignment"]["n_top1_changed"] <= 8 and p["assignment"]["distinct_top1_before"] <= 8
        for s in p["scores"].values():
            assert 0 <= s["score"] <= 1 and set(s["by_n"]) <= {"5", "10", "15", "20"}
    cmp = res["comparisons"]["pre_specified"]["assign vs fusion"]
    assert {"diff", "ci95", "per_pool_diff", "diff_n5", "pass"} <= set(cmp) and len(cmp["per_pool_diff"]) == 2
    assert set(res["overall_score"]) == {"fusion", "assign"} and res["questions"] == 32
    ctrl = res["control"]
    assert len(ctrl) == 2 and all({"folds", "questions", "distinct_top1", "fusion", "assign", "diff"} <= set(c) for c in ctrl)
    assert "[pre_specified] assign vs fusion" in capsys.readouterr().out
    # fold 0 refused
    assert pid_assign_cv.main([*args, "--pools", "0,1"]) == 2
