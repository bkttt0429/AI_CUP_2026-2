import numpy as np
import pytest

from aicup_pid import components, trunkstyle


def test_trunk_model_matches_question_to_its_player(tmp_path):
    rng = np.random.RandomState(0)
    centers = rng.randn(3, 256) * 2
    store, by_player = {}, {}
    for p in range(3):
        for j in range(20):
            k = f"c:{p}:{j}"
            store[k] = (centers[p] + rng.randn(256)).astype(np.float32)
            by_player.setdefault(f"p{p}", []).append(k)
        for j in range(5):
            store[f"q:{p}:{j}"] = (centers[p] + rng.randn(256)).astype(np.float32)
    model = trunkstyle.fit_trunk(by_player, store, {"trunk_feats": "all", "trunk_shrink": 0.1})
    assert model.players == ["p0", "p1", "p2"] and len(model.columns) == 256
    S = trunkstyle.score_trunk(model, [[f"q:{p}:{j}" for j in range(5)] for p in range(3)], store)
    assert S.shape == (3, 3) and (S.argmax(axis=1) == np.arange(3)).all()
    assert trunkstyle.score_trunk(model, [], store).shape == (0, 3)
    with pytest.raises(KeyError):
        trunkstyle.score_trunk(model, [["missing"]], store)
    m2 = trunkstyle.fit_trunk(by_player, store, {"trunk_feats": "mean"})
    assert len(m2.columns) == 128 and trunkstyle.COLUMNS["local"][0] == 128
    keys = sorted(store)
    np.savez(tmp_path / "part_s000_p0000.npz", keys=np.array(keys), feats=np.stack([store[k] for k in keys]), n_target=np.zeros(len(keys), np.int16))
    (tmp_path / "part_s000_p0001.npz.tmp.npz").write_bytes(b"")  # an unfinished part is ignored
    loaded = trunkstyle.load_store(tmp_path)
    assert set(loaded) == set(store) and np.allclose(loaded["q:1:3"], store["q:1:3"])


def test_trunk_is_an_optional_component():
    assert "trunk" in components.OPTIONAL and components.ALL_COMPONENTS[-2:] == ("trunk", "emb")
