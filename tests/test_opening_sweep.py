import json

import pid_opening_sweep
from tests.synth import black_table, sgf_lookup, training_frame


def test_opening_sweep_cli(tmp_path, monkeypatch, capsys):
    frame, folds_df = training_frame(folds=(1, 2, 3, 4), players_per_fold=5, games_per_player=40)  # 20 Black games per player
    lookup = sgf_lookup(frame)
    monkeypatch.setattr(pid_opening_sweep.pools, "black_games", lambda train_dir=None: black_table(frame))
    monkeypatch.setattr(pid_opening_sweep, "load_games", lambda keys, train_dir=None: {k: lookup[k] for k in keys})
    folds_csv = tmp_path / "folds.csv"
    folds_df.to_csv(folds_csv, index=False)
    out = tmp_path / "sweep.json"
    grid = json.dumps([{"name": "d8", "depth": 8}, {"name": "a4", "alpha": 4.0}, {"name": "full_off", "alpha_full": 1000.0}])
    args = ["--pools", "1,2", "3,4", "--n-players", "8", "--cand-games", "10", "--depth", "6", "--grid", grid, "--n-boot", "50",
            "--folds-csv", str(folds_csv), "--out", str(out)]
    assert pid_opening_sweep.main(args) == 0
    res = json.loads(out.read_text())
    assert res["base"]["name"] == "base" and res["base"]["cfg"] == {"depth": 6, "alpha": 2.0, "beta": 5.0, "alpha_full": 1.0, "shrink": 0.1}
    assert [c["name"] for c in res["configs"]] == ["base", "d8", "a4", "full_off"]
    for c in res["configs"]:
        assert set(c["cfg"]) == {"depth", "alpha", "beta", "alpha_full", "shrink"} and len(c["pools"]) == 2
        for p in c["pools"]:
            assert set(p["scores"]) == {"opening", "fusion"} and 0 <= p["scores"]["fusion"]["score"] <= 1
        assert set(c["overall_score"]) == {"opening", "fusion"}
    assert res["configs"][1]["cfg"]["depth"] == 8 and res["configs"][3]["cfg"]["alpha_full"] == 1000.0
    cmp = res["comparisons"]["pre_specified"]
    assert set(cmp) == {"d8 vs base", "a4 vs base", "full_off vs base"}
    assert all({"diff", "ci95", "per_pool_diff", "diff_n5", "pass"} <= set(v) for v in cmp.values())
    assert res["questions"] > 0 and "pid_git_head" in res
    printed = capsys.readouterr().out
    assert "[pre_specified] d8 vs base" in printed
    assert pid_opening_sweep.main([*args, "--pools", "0,1"]) == 2
