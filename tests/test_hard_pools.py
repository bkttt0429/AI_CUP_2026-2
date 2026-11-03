import json

import numpy as np

import pid_hard_pools
from tests.synth import black_table, sgf_lookup, training_frame


def test_nearest_pool_greedy():
    sim = np.array([[1, 0.9, 0.1, 0.2], [0.9, 1, 0.15, 0.1], [0.1, 0.15, 1, 0.8], [0.2, 0.1, 0.8, 1]])
    assert pid_hard_pools.nearest_pool(sim, ["a", "b", "c", "d"], 2) == ["a", "b"]
    assert pid_hard_pools.nearest_pool(sim, ["a", "b", "c", "d"], 3) == ["a", "b", "d"]  # d: mean(0.2, 0.1) > c: mean(0.1, 0.15)
    assert len(pid_hard_pools.nearest_pool(sim, ["a", "b", "c", "d"], 10)) == 4


def test_hard_pools_cli(tmp_path, monkeypatch, capsys):
    frame, folds_df = training_frame(folds=(1, 2), players_per_fold=8, games_per_player=40)  # 16 players, 20 Black games each
    lookup = sgf_lookup(frame)
    monkeypatch.setattr(pid_hard_pools.pools, "black_games", lambda train_dir=None: black_table(frame))
    monkeypatch.setattr(pid_hard_pools, "load_games", lambda keys, train_dir=None: {k: lookup[k] for k in keys})
    folds_csv = tmp_path / "folds.csv"
    folds_df.to_csv(folds_csv, index=False)
    f2, f3 = tmp_path / "f2.json", tmp_path / "f3.json"
    f2.write_text(json.dumps({"components": ["opening", "style"], "weights": [1.0, 0.5]}))
    f3.write_text(json.dumps({"components": ["opening", "style", "local"], "weights": [1.0, 0.5, 0.3], "cfg": {"depth": 8, "local_start": 0}}))
    out = tmp_path / "hard.json"
    ranks = sorted(frame["rank"].unique())
    args = ["--folds", "1", "2", "--n-players", "6", "--cand-games", "10", "--bands", ",".join(ranks[:3]), "--nearest", "6", "--nearest-k", "2",
            "--fusion", str(f2), "--fusion3", str(f3), "--folds-csv", str(folds_csv), "--out", str(out)]
    assert pid_hard_pools.main(args) == 0
    res = json.loads(out.read_text())
    names = [p["name"] for p in res["pools"]]
    assert names == ["random", f"band:{','.join(ranks[:3])}", "nearest:6"] and res["players_used"] == 16
    for p in res["pools"]:
        assert set(p["scores"]) == {"opening", "style", "local", "fusion", "fusion3"} and p["players"] <= 6 and p["questions"] > 0
        assert sum(p["ranks"].values()) == p["players"]
    band = res["pools"][1]
    assert all(band["ranks"][r] == 0 for r in band["ranks"] if r not in ranks[:3])
    assert res["pools"][2]["mean_pairwise_similarity"] >= res["pools"][0]["mean_pairwise_similarity"]  # the greedy pool is at least as self-similar
    assert "nearest:6" in capsys.readouterr().out
    assert pid_hard_pools.main([*args, "--folds", "0"]) == 2
    assert pid_hard_pools.main([*args, "--bands", "9D"]) == 2
