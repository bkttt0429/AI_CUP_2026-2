import json

import pid_assign_cv
import pid_selftrain_cv
from tests.synth import black_table, sgf_lookup, training_frame


def test_selftrain_cv_cli(tmp_path, monkeypatch, capsys):
    frame, folds_df = training_frame(folds=(1, 2, 3, 4), players_per_fold=5, games_per_player=60)
    lookup = sgf_lookup(frame)
    monkeypatch.setattr(pid_assign_cv.pools, "black_games", lambda train_dir=None: black_table(frame))
    monkeypatch.setattr(pid_assign_cv, "load_games", lambda keys, train_dir=None: {k: lookup[k] for k in keys})
    folds_csv = tmp_path / "folds.csv"
    folds_df.to_csv(folds_csv, index=False)
    out = tmp_path / "selftrain.json"
    args = ["--pools", "1,2", "3,4", "--seeds", "0", "--n-players", "8", "--cand-games", "10", "--per-size", "2", "--depth", "8", "--local-start", "0",
            "--tau", "0.5", "--n-boot", "50", "--folds-csv", str(folds_csv), "--out", str(out)]
    assert pid_selftrain_cv.main(args) == 0
    res = json.loads(out.read_text())
    assert len(res["pools"]) == 2
    for p in res["pools"]:
        assert set(p["scores"]) == {"assign3", "self3", "self3_report", "self3_two_rounds", "self3_noassign"} and len(p["fusion3_weights"]) == 3
        st = p["self_training"]
        assert 0 <= st["confident"] <= 8 and 0 <= st["confident_wrong"] <= st["confident"] and len(st["rounds"]) == 1
    cmp = res["comparisons"]
    assert set(cmp["pre_specified"]) == {"self3 vs assign3"} and set(cmp["report_only"]) == {"self3_report vs assign3", "self3_two_rounds vs assign3", "self3_noassign vs assign3"}
    assert {"diff", "ci95", "per_pool_diff", "diff_n5", "pass"} <= set(cmp["pre_specified"]["self3 vs assign3"])
    assert res["questions"] == 16 and res["tau"] == 0.5 and "pid_git_head" in res
    assert "[pre_specified] self3 vs assign3" in capsys.readouterr().out
    assert pid_selftrain_cv.main([*args, "--pools", "0,1"]) == 2
