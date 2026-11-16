import importlib
from pathlib import Path

import numpy as np
import pytest


def module():
    return importlib.import_module("pid_choice_manifest")


def test_pool_namespace_preserves_original_key():
    m = module()
    assert m.namespaced_key("pub2", "c:0:1") != m.namespaced_key("priv2", "c:0:1")
    assert m.namespaced_key("pub2", "c:0:1") == "pub2|c:0:1"


def test_holdout_matches_existing_and_excludes_all_query_games():
    from aicup_pid.v2 import candidate_holdout
    m = module()
    rows = [{"pool": "pub2", "owner": p, "key": f"pub2|c:{p}:{i}", "original_key": f"c:{p}:{i}", "color": "B"} for p in ["a", "b"] for i in range(10)]
    manifest = {"games": rows}
    split = m.seed_split(manifest, "pub2", 3)
    original = {p: [(f"c:{p}:{i}", "B") for i in range(10)] for p in ["a", "b"]}
    expected = candidate_holdout(original, 3, 3)
    for p in original:
        assert split["heldout"][p] == ["pub2|" + k for k, _ in expected[p]]
        assert len(split["reference"][p]) == 7
    held = {k for v in split["heldout"].values() for k in v}
    assert not held.intersection(k for v in split["reference"].values() for k in v)


def test_changed_source_and_symlink_target_refused(tmp_path):
    m = module()
    source = tmp_path / "source"
    source.write_text("original")
    link = tmp_path / "link"
    link.symlink_to(source)
    record = m.source_record(link)
    m.verify_source(record)
    other = tmp_path / "other"
    other.write_text("original")
    link.unlink()
    link.symlink_to(other)
    with pytest.raises(ValueError):
        m.verify_source(record)
    source.write_text("modified")
    with pytest.raises(ValueError):
        m.verify_source(m.source_record(other) | {"path": str(source)})


def test_wrong_projection_refused(tmp_path):
    m = module()
    p = tmp_path / "p.npz"
    np.savez(p, blocks=np.array(["g", "x", "e"]), fit_folds=np.array([3, 4]), W=np.zeros((1152, 64)), sd=np.ones(1152))
    assert m.validate_projection(p).shape == (384, 32)
    np.savez(p, blocks=np.array(["g", "x", "e"]), fit_folds=np.array([0, 4]), W=np.zeros((1152, 64)), sd=np.ones(1152))
    with pytest.raises(ValueError):
        m.validate_projection(p)


def test_calibration_has_100_players_ten_games_and_no_fold0(tmp_path):
    import csv
    m = module()
    folds = tmp_path / "folds.csv"
    with open(folds, "w") as f:
        w = csv.writer(f)
        w.writerow(["player_id", "fold"])
        w.writerows([(str(p), 0 if p == 0 else 3) for p in range(101)])
    with open(tmp_path / "train_1D.csv", "w") as f:
        w = csv.writer(f)
        w.writerow(["player_id", "game_id", "rank", "color", "sgf_content"])
        w.writerows([(str(p), str(p * 100 + g), "1D", "B", "(;SZ[19]" + ";B[aa];W[bb]" * 10 + ")") for p in range(101) for g in range(11)])
    games, _ = m.select_calibration(tmp_path, folds)
    assert len(games) == 1000
    assert all(g["fold"] == 3 for g in games)
    for owner in {g["owner"] for g in games}:
        gs = [g for g in games if g["owner"] == owner]
        assert len(gs) == 10
        assert sum(g["role"] == "reference" for g in gs) == 7


@pytest.fixture(scope="module")
def replay_module():
    import torch
    import pid_trunk_extract as old
    new = importlib.import_module("pid_choice_extract")
    config = torch.load(old.te.CKPT, map_location="cpu", weights_only=True)["config"]
    old.te.init_worker(config)
    return new, old


def test_replay_before_move_and_white(replay_module):
    new, old = replay_module
    sgf = "(;SZ[19];B[dd];W[pp];B[dp];W[pd];B[];W[])"
    r = new.replay_choices("x", sgf, "W")
    _, bins, globs, played, mover, t = old.replay_target(("x", sgf, "W"))
    for name, expected in [("bins", bins), ("globs", globs), ("played", played), ("mover", mover), ("t", t)]:
        np.testing.assert_array_equal(r[name], expected)
    assert r["legal"].shape == (3, 362)
    assert r["legal"][:, 361].all()
    assert not r["legal"][0, 3 * 19 + 3]  # occupied by preceding black move


def test_replay_capture_suicide_ko_handicap_and_no_target(replay_module):
    new, _ = replay_module
    # White at bb has one liberty at bc; black captures then white passes.
    r = new.replay_choices("capture", "(;SZ[19];B[ab];W[bb];B[ba];W[pp];B[cb];W[qq];B[bc];W[])", "B")
    assert r["legal"][-1, 2 * 19 + 1]
    h = new.replay_choices("handicap", "(;SZ[19]HD[2];B[dd];W[qq];B[])", "W")
    assert not h["legal"][0, 3 * 19 + 15]  # pd setup stone; recorded B is actual W
    assert not h["legal"][0, 15 * 19 + 3]  # dp setup stone
    ko = new.replay_choices("ko", "(;SZ[19];B[ab];W[bb];B[ba];W[ac];B[cb];W[cc];B[pp];W[bd];B[bc];W[qq];B[rr];W[bb])", "W")
    assert not ko["legal"][-2, 1 * 19 + 1]
    assert ko["legal"][-1, 1 * 19 + 1]
    suicide = new.replay_choices("suicide", "(;SZ[19];B[ab];W[pp];B[ba];W[qq];B[cb];W[rr];B[bc];W[ss])", "W")
    assert not suicide["legal"][-1, 1 * 19 + 1]
    with pytest.raises(ValueError):
        new.replay_choices("no-target", "(;SZ[19];B[dd])", "W")
    with pytest.raises(Exception):
        new.replay_choices("occupied", "(;SZ[19];B[dd];W[dd])", "W")


def test_shard_interruption_and_changed_resume(tmp_path):
    e = importlib.import_module("pid_choice_extract")
    arrays = {"keys": np.array(["pub2|a"]), "n": np.array([1])}
    path = e.write_shard(tmp_path, "a", arrays, {"source": "x"})
    assert path.exists()
    assert e.valid_shard(path, {"source": "x"})
    assert e.write_shard(tmp_path, "a", arrays, {"source": "x"}) == path
    with pytest.raises(ValueError):
        e.write_shard(tmp_path, "a", arrays, {"source": "different"})
    path.write_bytes(b"partial")
    assert not e.valid_shard(path, {"source": "x"})
