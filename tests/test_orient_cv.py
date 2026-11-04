import json

import numpy as np

import pid_orient_cv
from aicup.sgf import parse_moves
from aicup_pid.opening import canonical
from tests.synth import LETTERS, black_table, sgf_lookup, training_frame


def reorient(sgf, k):
    """The synthetic game in its canonical orientation (random_game applies a random symmetry per game), then under board
    symmetry k (0..7): a fixed k per player plants an orientation habit."""
    moves = parse_moves(sgf)
    cs, _ = canonical(moves, 12)
    out = []
    for (c, *_), (x, y) in zip(moves, cs):
        if k & 1:
            x = 18 - x
        if k & 2:
            y = 18 - y
        if k & 4:
            x, y = y, x
        out.append(f";{c}[{LETTERS[x]}{LETTERS[y]}]")
    return "(;SZ[19]" + "".join(out) + ")"


def test_orient_cv_cli(tmp_path, monkeypatch, capsys):
    frame, folds_df = training_frame(folds=(1, 2, 3, 4), players_per_fold=5, games_per_player=40)
    players = sorted(frame["player_id"].unique())
    habit = {p: i % 8 for i, p in enumerate(players)}  # a planted per-player orientation habit
    frame["sgf_content"] = [reorient(s, habit[p]) for p, s in zip(frame["player_id"], frame["sgf_content"])]
    lookup = sgf_lookup(frame)
    monkeypatch.setattr(pid_orient_cv.pools, "black_games", lambda train_dir=None: black_table(frame))
    monkeypatch.setattr(pid_orient_cv, "load_games", lambda keys, train_dir=None: {k: lookup[k] for k in keys})
    folds_csv = tmp_path / "folds.csv"
    folds_df.to_csv(folds_csv, index=False)
    out, fus, fus5 = tmp_path / "orient_cv.json", tmp_path / "pid_fusion3o.json", tmp_path / "pid_fusion5.json"
    args = ["--pools", "1,2", "3,4", "--n-players", "8", "--cand-games", "10", "--depth", "8", "--start", "0", "--n-boot", "50",
            "--folds-csv", str(folds_csv), "--out", str(out), "--fusion-out", str(fus), "--fusion5-out", str(fus5)]
    assert pid_orient_cv.main(args) == 0
    doc = json.loads(fus.read_text())
    assert doc["components"] == ["opening", "style", "local", "orient"] and len(doc["weights"]) == 4 and doc["fitted_on_pools"] == [[1, 2], [3, 4]]
    assert doc["cfg"]["orient_depth"] == 12 and doc["cfg"]["local_start"] == 0 and doc["cfg"]["hd_alpha"] == 2.0
    assert json.loads(fus5.read_text())["components"] == ["opening", "style", "local", "orient", "hd"]
    res = json.loads(out.read_text())
    assert len(res["pools"]) == 2
    for p in res["pools"]:
        assert set(p["scores"]) == {"opening", "style", "local", "orient", "hd", "fusion", "fusion3", "fusion3o", "fusion5"}
        assert len(p["fusion3o_weights"]) == 4 and len(p["fusion5_weights"]) == 5
        assert p["scores"]["orient"]["score"] > 0.3  # the planted habit alone beats chance (1/8 Top-1) clearly
        assert set(p["first_move_modal_share"]) == {"median", "ge_0.9"}
    cmp = res["comparisons"]
    assert set(cmp["pre_specified"]) == {"fusion3o vs fusion3"} and set(cmp["report_only"]) == {"orient vs style", "orient vs opening", "fusion5 vs fusion3o"}
    assert {"diff", "ci95", "per_pool_diff", "diff_n5", "pass"} <= set(cmp["pre_specified"]["fusion3o vs fusion3"])
    assert "[pre_specified] fusion3o vs fusion3" in capsys.readouterr().out
    assert pid_orient_cv.main([*args, "--pools", "0,1"]) == 2
