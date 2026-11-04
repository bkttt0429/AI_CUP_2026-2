import numpy as np
import pytest

from aicup_pid import components, hd
from aicup_pid.components import fit_from_parsed, fit_hd, parse_all, features_all, score_components
from tests.synth import habit_of, random_game


def with_hd(sgf, n):
    return sgf.replace("(;SZ[19]", f"(;SZ[19]HD[{n}]", 1)


def test_hd_class():
    assert hd.hd_class("(;SZ[19];B[pd])") == 0  # dataset v1.0 record without HD
    assert hd.hd_class("(;SZ[19]HD[0];B[pd])") == 0
    assert hd.hd_class("(;SZ[19]HD[1];B[pd])") == 1
    assert hd.hd_class("(;SZ[19]HD[2];B[pd])") == 2 and hd.hd_class("(;SZ[19]HD[9];B[pd])") == 2


def test_hd_model_recovers_planted_rates():
    rng = np.random.RandomState(0)
    rates = {"p0": 0.0, "p1": 0.1, "p2": 0.6, "p3": 1.0}
    lib = {p: [with_hd(random_game(rng, habit_of(0), 30), int(rng.rand() < r)) for _ in range(50)] for p, r in rates.items()}
    model = hd.HDModel(alpha=2.0, beta=5.0).fit(lib)
    assert model.players == ["p0", "p1", "p2", "p3"] and model.logp.shape == (4, hd.K) and np.all(model.logp < 0)
    assert np.allclose(np.exp(model.logp).sum(axis=1), 1.0)
    always = [with_hd(random_game(rng, habit_of(0), 30), 1) for _ in range(5)]
    never = [with_hd(random_game(rng, habit_of(0), 30), 0) for _ in range(5)]
    assert model.players[int(np.argmax(model.log_scores(always)))] == "p3"
    assert model.players[int(np.argmax(model.log_scores(never)))] == "p0"
    s = model.log_scores(always)
    assert s[3] > s[2] > s[1] > s[0]  # monotone in the planted rate
    assert model.log_scores([]).shape == (4,) and np.all(model.log_scores([]) == 0)
    assert np.isfinite(model.log_scores([with_hd(always[0], 3)])).all()  # an unseen class scores finitely (background mass)
    st = model.state()
    assert st["alpha"] == 2.0 and st["players"] == model.players and sum(st["class_counts"]) == 200
    with pytest.raises(ValueError):
        hd.HDModel(alpha=0)


def test_score_components_with_hd():
    rng = np.random.RandomState(1)
    sgfs, by_player = {}, {}
    for i, (p, r) in enumerate((("a", 0.0), ("b", 1.0))):
        for j in range(6):
            k = f"c:{i}:{j}"
            sgfs[k] = with_hd(random_game(rng, habit_of(i), 30), int(rng.rand() < r))
            by_player.setdefault(p, []).append(k)
    sgfs["q:0"] = with_hd(random_game(rng, habit_of(0), 30), 1)
    sgfs["q:1"] = with_hd(random_game(rng, habit_of(1), 30), 1)
    moves, feats = parse_all(sgfs), features_all(sgfs)
    opening, style = fit_from_parsed(by_player, moves, feats, {"depth": 8})
    model = fit_hd(by_player, sgfs, {"hd_alpha": 2.0, "hd_beta": 5.0})
    assert model.players == opening.players == ["a", "b"] and model.alpha == 2.0
    S = score_components(("opening", "style", "hd"), opening, style, [["q:0", "q:1"]], moves, feats, hd=model, sgfs=sgfs)
    assert set(S) == {"opening", "style", "hd"} and S["hd"].shape == (1, 2) and S["hd"][0, 1] > S["hd"][0, 0]
    assert "hd" in components.OPTIONAL and components.ALL_COMPONENTS == ("opening", "style", "local", "hd", "orient", "opening_raw")
    with pytest.raises(ValueError):
        score_components(("opening", "style", "hd"), opening, style, [["q:0"]], moves, feats)
    assert score_components(("hd",), opening, style, [], moves, feats, hd=model, sgfs=sgfs)["hd"].shape == (0, 2)
