import numpy as np

from aicup_pid import fusion
from aicup_pid.stats import paired_player_bootstrap


def test_standardise_topk_and_fusion_weights():
    S = np.array([[1.0, 2.0, 3.0], [5.0, 5.0, 5.0]])
    Z = fusion.standardise(S)
    assert np.allclose(Z[0].mean(), 0) and np.allclose(Z[0].std(), 1) and np.allclose(Z[1], 0)
    assert fusion.topk(S, 2).tolist() == [[2, 1], [0, 1]]
    rng = np.random.RandomState(0)
    y = rng.randint(0, 30, 80)
    good = rng.randn(80, 30)
    good[np.arange(80), y] += 2.5
    noise = rng.randn(80, 30)
    f = fusion.Fusion().fit([good, noise], y)
    assert f.w[0] > 1.0 and abs(f.w[1]) < 0.3
    s = f.scores([good, noise])
    assert s.shape == (80, 30) and (fusion.topk(s, 1)[:, 0] == y).mean() > 0.6
    assert f.state()["weights"] == f.w.tolist()


def test_paired_player_bootstrap():
    pid = np.repeat([f"p{i}" for i in range(50)], 4)
    a = np.ones(200) * 0.5
    b = np.ones(200) * 0.3
    d = paired_player_bootstrap(a, b, pid, n=200)
    assert abs(d["diff"] - 0.2) < 1e-12 and abs(d["ci95"][0] - 0.2) < 1e-12 and d["n_players"] == 50 and d["n_questions"] == 200
    rng = np.random.RandomState(1)
    d2 = paired_player_bootstrap(rng.randn(200), rng.randn(200), pid, n=500)
    assert d2["ci95"][0] < d2["diff"] < d2["ci95"][1]
