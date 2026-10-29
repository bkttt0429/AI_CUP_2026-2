import numpy as np

from aicup_pid import opening
from tests.synth import habit_of, random_game


def _sym_sgf(sgf, k):
    moves = opening.parse(sgf)
    out = []
    for c, x, y in moves:
        out.append((c, *opening._sym(k, x, y)))
    return "(;SZ[19]" + "".join(f";{c}[{'abcdefghijklmnopqrs'[x]}{'abcdefghijklmnopqrs'[y]}]" for c, x, y in out) + ")"


def test_canonical_form_is_symmetry_invariant_and_tokens_are_blacks_moves():
    rng = np.random.RandomState(1)
    sgf = random_game(rng, habit_of(3), n_moves=30)
    base = opening.tokens(opening.parse(sgf), 12)
    assert len(base) == 6 and all(len(fctx) % 2 == 0 and len(bctx) == len(fctx) // 2 for fctx, bctx, _ in base)  # Black moves 1, 3, 5, ...
    assert base[0][0] == () and base[0][1] == () and base[1][1] == (base[0][2],) and base[1][0][0] == base[0][2]
    for k in range(8):
        assert opening.tokens(opening.parse(_sym_sgf(sgf, k)), 12) == base, k
    # a pass keeps its sentinel coordinate under every symmetry
    cs, _ = opening.canonical(opening.parse("(;SZ[19];B[pd];W[];B[dp])"), 12)
    assert cs[1] == opening.PASS and len(cs) == 3


def test_opening_model_identifies_players_with_planted_openings():
    rng = np.random.RandomState(2)
    players = {f"p{i:02d}": habit_of(i) for i in range(16)}  # 16 distinct (first, second) corner habits
    cand = {p: [opening.parse(random_game(rng, h, 30)) for _ in range(30)] for p, h in players.items()}
    model = opening.OpeningModel(depth=8, alpha=1.0, beta=5.0).fit(cand)
    assert model.players == sorted(players)
    hits = 0
    for p, h in players.items():
        query = [opening.parse(random_game(rng, h, 30)) for _ in range(5)]
        s = model.log_scores(query)
        assert s.shape == (16,) and np.all(np.isfinite(s))
        hits += model.players[int(np.argmax(s))] == p
    assert hits == 16
    # an opening never seen by anyone is finite (background smoothing) and equal across candidates with no such context
    weird = opening.parse("(;SZ[19];B[jj];W[dp];B[ji])")
    s = model.log_scores([weird])
    assert np.all(np.isfinite(s))
    assert opening.OpeningModel(depth=4).fit({"a": [weird]}).log_scores([weird])[0] > model.log_scores([weird]).max()
