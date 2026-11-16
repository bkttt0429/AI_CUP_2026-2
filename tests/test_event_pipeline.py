import importlib

import numpy as np
import pytest


def test_both_sides_replay_is_ordered_and_preserves_before_move_inputs():
    import torch
    import pid_trunk_extract as old
    m = importlib.import_module("pid_event_probe")
    old.te.init_worker(torch.load(old.te.CKPT, map_location="cpu", weights_only=True)["config"])
    g = {"key": "calibration|x", "sgf": "(;SZ[19];B[dd];W[pp];B[];W[])", "color": "W"}
    both = m.replay_both(g)
    np.testing.assert_array_equal(both["t"], [0, 1, 2, 3])
    np.testing.assert_array_equal(both["mover"], [0, 1, 0, 1])
    for color, side in (("B", 0), ("W", 1)):
        r = m.replay_choices_checked(g["key"], g["sgf"], color)
        for f in ("bins", "globs", "legal", "played"):
            np.testing.assert_array_equal(both[f][both["mover"] == side], r[f])


def test_event_raw_loader_rejects_missing_sidecar_and_changed_bytes(tmp_path):
    m = importlib.import_module("pid_event_probe")
    path = tmp_path / "events/raw/calibration_p0000.npz"
    path.parent.mkdir(parents=True)
    games = [{"pool": "calibration", "key": "calibration|x"}]
    meta = {"protocol": "p", "part": 0, "keys": ["calibration|x"]}
    arrays = {f: np.zeros((4, 2)) if f == "residual" else np.arange(4) for f in m.FIELDS}
    arrays.update(keys=np.array(["calibration|x"]), n=np.array([4]), subject=np.array([0]))
    with pytest.raises(ValueError, match="missing"):
        m.load_raw(tmp_path, {"games": games}, {"id": "p"})
    m._write(path, arrays, meta)
    assert len(m.load_raw(tmp_path, {"games": games}, {"id": "p"})["calibration|x"]["t"]) == 4
    with open(path, "ab") as f:
        f.write(b"changed")
    with pytest.raises(ValueError, match="corrupted"):
        m.load_raw(tmp_path, {"games": games}, {"id": "p"})
