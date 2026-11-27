"""Label-free probe on the official candidates: how much do the OPPONENT's (White's) moves and the game's ending tell
about the player? MI(player; token) with a label-shuffle null for: White's first move (raw / canonical), White's first
two moves, Black+White first two moves canonical, game length bucket, who moved last (resignation proxy), the last
move's region. Plus the same for Black's second raw move (orientation extension)."""
import collections, re, sys
import numpy as np, pandas as pd
sys.path.insert(0, "/home/u2507115/BK_AICUP_2026-2/src")
import aicup_pid  # noqa: E402
from aicup.sgf import parse_moves  # noqa: E402
from aicup_pid.opening import canonical, coords  # noqa: E402

R2 = "/home/u2507115/BK_AICUP_2026-2"


def mi(lab, tok):
    df = pd.crosstab(lab, tok).values.astype(float)
    p = df / df.sum(); px = p.sum(1, keepdims=True); py = p.sum(0, keepdims=True)
    nz = p > 0
    return float((p[nz] * np.log2(p[nz] / (px @ py)[nz])).sum())


def info(lab, tok, rng, n_null=10):
    lab = np.asarray(lab); tok = np.array([str(t) for t in tok])
    return mi(lab, tok), float(np.mean([mi(rng.permutation(lab), tok) for _ in range(n_null)]))


for pool, cf in (("pub_", "player_identification_candidates_public.csv"), ("priv_", "player_identification_candidates_private.csv")):
    rng = np.random.default_rng(0)
    c = pd.read_csv(f"{R2}/data/tests/{cf}", dtype=str, keep_default_na=False)
    lab = c.player_id.values
    toks = collections.defaultdict(list)
    for s in c.sgf_content:
        m = parse_moves(s); cs = coords(m); can, k = canonical(m, 12)
        b = [i for i, (col, *_) in enumerate(m) if col == "B"]; w = [i for i, (col, *_) in enumerate(m) if col == "W"]
        toks["black 1st raw"].append(cs[b[0]] if b else None)
        toks["black 2nd raw"].append(cs[b[1]] if len(b) > 1 else None)
        toks["black 1st+2nd raw"].append((cs[b[0]], cs[b[1]]) if len(b) > 1 else None)
        toks["white 1st raw"].append(cs[w[0]] if w else None)
        toks["white 1st canonical"].append(can[w[0]] if w else None)
        toks["white 1st+2nd canonical"].append((can[w[0]], can[w[1]]) if len(w) > 1 else None)
        toks["first 4 moves canonical"].append(tuple(can[:4]))
        toks["length bucket (20 moves)"].append(len(m) // 20)
        toks["last mover"].append(m[-1][0] if m else None)
        toks["last move region"].append((m[-1][0], cs[-1][0] // 5, cs[-1][1] // 5) if m else None)
        toks["length parity x last mover"].append((len(m) % 2, m[-1][0]) if m else None)
    print(pool, "games", len(c))
    for name, tok in toks.items():
        m_, null = info(lab, tok, rng)
        print(f"{pool} {name:28s} I = {m_:.3f} bits  null {null:.3f}  net {m_ - null:+.3f}")
    lm = collections.Counter(toks["last mover"]); print(pool, "last mover counts", dict(lm))
