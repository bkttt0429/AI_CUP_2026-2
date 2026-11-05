import numpy as np

from aicup_pid import probe


def test_deltas_and_decompositions():
    d = probe.deltas((2, 1, 3, 4, 5))  # swap ranks 1 and 2
    assert abs(d[0] - (np.exp(-1) - 1)) < 1e-12 and abs(d[1] - (1 - np.exp(-1))) < 1e-12 and d[2] == d[3] == d[4] == d[5] == 0
    assert probe.deltas((1, 2, 3, 4, 5)).tolist() == [0.0] * 6
    c = probe.deltas(probe.CYCLE)
    assert c[4] > 0 and all(x < 0 for x in c[:4]) and c[5] == 0  # rank 5 to the top, the rest one down
    assert probe.deficit_decompositions(1.0) == [(0, 0, 0, 0, 0)]
    assert probe.deficit_decompositions(0.9873479)  # the 11/04 Public score has at least one decomposition


def test_histograms_decode_uniquely_up_to_max_group_size():
    assert probe.bounds_from_score(0.9873479) == (5, 1, 2, 1, 1)
    assert len(probe.histograms(3, bounds=(5, 5, 5, 5, 5))) == 56  # C(3 + 5, 5) without effective bounds
    assert len(probe.histograms(40)) == 6 * 2 * 3 * 2 * 2 and probe.histograms(2)[0] == (2, 0, 0, 0, 0, 0)
    assert probe.min_gap(5, bounds=(5, 5, 5, 5, 5)) == 0.0  # (1,1,1,1,1,0) and (0,0,0,0,0,5) collide: the deltas sum to zero
    size = probe.max_group_size()
    assert size >= 40
    rng = np.random.RandomState(0)
    for n in (8, 20, 40):
        for _ in range(10):
            rest = [int(rng.randint(b + 1)) for b in probe.BOUNDS]
            while sum(rest) > n:
                rest = [int(rng.randint(b + 1)) for b in probe.BOUNDS]
            hist = (n - sum(rest), *rest)
            observed = probe.histogram_sum(hist) + rng.uniform(-4e-5, 4e-5)
            assert probe.decode_histogram(observed, n) == [hist]


def test_reorder_fix_and_resolve():
    top5 = ["a", "b", "c", "d", "e"]
    assert probe.reorder(top5) == ["e", "a", "b", "c", "d"]
    assert probe.fixed_list(top5, 3) == ["c", "a", "b", "d", "e"]
    assert abs(probe.gain_of_fix(2) - (1 - np.exp(-1))) < 1e-12 and probe.gain_of_fix(1) == 0.0
    groups = [{"members": ["q1", "q2", "q3", "q4"], "hist": (2, 1, 0, 0, 1, 0)},
              {"members": ["q1", "q2"], "hist": (2, 0, 0, 0, 0, 0)},
              {"members": ["q3"], "hist": (0, 1, 0, 0, 0, 0)}]
    positions, derived = probe.resolve(groups)
    assert positions == {"q1": 1, "q2": 1, "q3": 2, "q4": 5}  # q4 inferred twice over: parent minus the two children
    assert any(sorted(g["members"]) == ["q3", "q4"] for g in derived) and any(g["members"] == ["q4"] for g in derived)
