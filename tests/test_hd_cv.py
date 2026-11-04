import json

import numpy as np

import pid_hd_cv
from tests.synth import black_table, sgf_lookup, training_frame


def test_hd_cv_cli(tmp_path, monkeypatch, capsys):
    frame, folds_df = training_frame(folds=(1, 2, 3, 4), players_per_fold=5, games_per_player=40)
    rng = np.random.RandomState(0)
    rate = {p: r for p, r in zip(sorted(frame["player_id"].unique()), np.tile([0.0, 0.2, 0.8, 1.0, 0.5], 4))}  # a planted per-player HD[1] habit
    frame["sgf_content"] = [s.replace("(;SZ[19]", f"(;SZ[19]HD[{int(rng.rand() < rate[p])}]", 1) for p, s in zip(frame["player_id"], frame["sgf_content"])]
    lookup = sgf_lookup(frame)
    monkeypatch.setattr(pid_hd_cv.pools, "black_games", lambda train_dir=None: black_table(frame))
    monkeypatch.setattr(pid_hd_cv, "load_games", lambda keys, train_dir=None: {k: lookup[k] for k in keys})
    folds_csv = tmp_path / "folds.csv"
    folds_df.to_csv(folds_csv, index=False)
    out = tmp_path / "hd_cv.json"
    fus = tmp_path / "pid_fusion4.json"
    args = ["--pools", "1,2", "3,4", "--n-players", "8", "--cand-games", "10", "--depth", "8", "--start", "0", "--n-boot", "50",
            "--folds-csv", str(folds_csv), "--out", str(out), "--fusion-out", str(fus)]
    assert pid_hd_cv.main(args) == 0
    doc = json.loads(fus.read_text())
    assert doc["components"] == ["opening", "style", "local", "hd"] and len(doc["weights"]) == 4 and doc["fitted_on_pools"] == [[1, 2], [3, 4]]
    assert doc["cfg"] == {"depth": 8, "alpha": 2.0, "beta": 5.0, "shrink": 0.1, "local_alpha": 2.0, "local_beta": 5.0, "local_start": 0, "hd_alpha": 2.0, "hd_beta": 5.0}
    res = json.loads(out.read_text())
    assert len(res["pools"]) == 2
    for p in res["pools"]:
        assert set(p["scores"]) == {"opening", "style", "local", "hd", "fusion", "fusion3", "fusion4"}
        assert len(p["fusion_weights"]) == 2 and len(p["fusion3_weights"]) == 3 and len(p["fusion4_weights"]) == 4
        assert set(p["hd_counts"]) == {"candidates", "questions"} and sum(p["hd_counts"]["candidates"].values()) == 80
        assert p["scores"]["hd"]["score"] > 0.15  # the planted habit alone beats chance (1/8 Top-1)
    cmp = res["comparisons"]
    assert set(cmp["pre_specified"]) == {"fusion4 vs fusion3"} and set(cmp["report_only"]) == {"hd vs style", "fusion4 vs fusion"}
    assert {"diff", "ci95", "per_pool_diff", "diff_n5", "pass"} <= set(cmp["pre_specified"]["fusion4 vs fusion3"])
    assert res["cfg"]["hd_alpha"] == 2.0 and "pid_git_head" in res
    assert "[pre_specified] fusion4 vs fusion3" in capsys.readouterr().out
    assert pid_hd_cv.main([*args, "--pools", "0,1"]) == 2
