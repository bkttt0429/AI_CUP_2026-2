import json

import pid_local_cv
from tests.synth import black_table, sgf_lookup, training_frame


def test_local_cv_cli(tmp_path, monkeypatch, capsys):
    frame, folds_df = training_frame(folds=(1, 2, 3, 4), players_per_fold=5, games_per_player=40)
    lookup = sgf_lookup(frame)
    monkeypatch.setattr(pid_local_cv.pools, "black_games", lambda train_dir=None: black_table(frame))
    monkeypatch.setattr(pid_local_cv, "load_games", lambda keys, train_dir=None: {k: lookup[k] for k in keys})
    folds_csv = tmp_path / "folds.csv"
    folds_df.to_csv(folds_csv, index=False)
    out = tmp_path / "local_cv.json"
    fus = tmp_path / "pid_fusion3.json"
    args = ["--pools", "1,2", "3,4", "--n-players", "8", "--cand-games", "10", "--depth", "8", "--start", "0", "--n-boot", "50",
            "--folds-csv", str(folds_csv), "--out", str(out), "--fusion-out", str(fus)]
    assert pid_local_cv.main(args) == 0
    doc = json.loads(fus.read_text())
    assert doc["components"] == ["opening", "style", "local"] and len(doc["weights"]) == 3 and doc["fitted_on_pools"] == [[1, 2], [3, 4]]
    assert doc["cfg"] == {"depth": 8, "alpha": 2.0, "beta": 5.0, "shrink": 0.1, "local_alpha": 2.0, "local_beta": 5.0, "local_start": 0}
    res = json.loads(out.read_text())
    assert len(res["pools"]) == 2
    for p in res["pools"]:
        assert set(p["scores"]) == {"opening", "style", "local", "fusion", "fusion3"} and len(p["fusion_weights"]) == 2 and len(p["fusion3_weights"]) == 3
    cmp = res["comparisons"]
    assert set(cmp["pre_specified"]) == {"fusion3 vs fusion"} and set(cmp["report_only"]) == {"local vs style", "local vs opening"}
    assert {"diff", "ci95", "per_pool_diff", "diff_n5", "pass"} <= set(cmp["pre_specified"]["fusion3 vs fusion"])
    assert res["local_cfg"] == {"alpha": 2.0, "beta": 5.0, "start": 0} and "pid_git_head" in res
    assert "[pre_specified] fusion3 vs fusion" in capsys.readouterr().out
    assert pid_local_cv.main([*args, "--pools", "0,1"]) == 2
