"""Label-free probe: is the RAW board orientation (which our models canonicalise away) a per-player habit on the official
candidates? Per pool: per-player distribution of the raw first Black move, its corner, and the symmetry index that
aicup_pid.opening.canonical picks for the first 12 moves; concentration and mutual information against a label-shuffle
null. Also pass encoding variants and the raw first WHITE move (opponent/server habit)."""
import collections, re, sys
import numpy as np, pandas as pd
sys.path.insert(0, "/home/u2507115/BK_AICUP_2026-2/src")
import aicup_pid  # noqa: E402
from aicup.sgf import parse_moves  # noqa: E402
from aicup_pid.opening import canonical  # noqa: E402

R2 = "/home/u2507115/BK_AICUP_2026-2"
first_b = re.compile(r";B\[([a-t]{0,2})\]")
first_w = re.compile(r";W\[([a-t]{0,2})\]")


def corner(pt):
    if len(pt) != 2: return "pass"
    x, y = ord(pt[0]) - 97, ord(pt[1]) - 97
    return ("L" if x < 9 else "C" if x == 9 else "R") + ("T" if y < 9 else "M" if y == 9 else "B")


def info(labels, tokens, rng, n_null=20):
    """Mutual information I(player; token) in bits (plug-in) and the mean of a label-shuffle null."""
    def mi(lab, tok):
        df = pd.crosstab(lab, tok).values.astype(float)
        p = df / df.sum(); px = p.sum(1, keepdims=True); py = p.sum(0, keepdims=True)
        nz = p > 0
        return float((p[nz] * np.log2(p[nz] / (px @ py)[nz])).sum())
    lab = np.asarray(labels); tok = np.asarray(tokens)
    return mi(lab, tok), float(np.mean([mi(rng.permutation(lab), tok) for _ in range(n_null)]))


for pool, cf, qf in (("pub_", "player_identification_candidates_public.csv", "player_identification_test_public.csv"),
                     ("priv_", "player_identification_candidates_private.csv", "player_identification_test_private.csv")):
    rng = np.random.default_rng(0)
    c = pd.read_csv(f"{R2}/data/tests/{cf}", dtype=str, keep_default_na=False)
    fb = c.sgf_content.map(lambda s: (first_b.search(s) or [None, ""])[1] if first_b.search(s) else "")
    fw = c.sgf_content.map(lambda s: first_w.search(s).group(1) if first_w.search(s) else "")
    sym = c.sgf_content.map(lambda s: canonical(parse_moves(s), 12)[1])
    lab = c.player_id.values
    print(pool, "raw first Black move: top points", collections.Counter(fb).most_common(8))
    for name, tok in (("first Black point", fb.values), ("first Black corner", fb.map(corner).values), ("symmetry index (first 12 moves)", sym.values),
                      ("first White point", fw.values), ("first White corner", fw.map(corner).values)):
        m, null = info(lab, tok, rng)
        modal = pd.DataFrame({"p": lab, "t": tok}).groupby("p").t.agg(lambda s: s.value_counts(normalize=True).iloc[0])
        print(pool, f"{name}: I(player; token) = {m:.3f} bits (shuffle null {null:.3f}, log2(400) = 8.64); per-player modal share median {modal.median():.2f}, "
              f">= 0.9 for {int((modal >= 0.9).sum())} players, <= 0.5 for {int((modal <= 0.5).sum())}")
    # pass encoding and odd tokens
    enc = collections.Counter()
    for s in c.sgf_content:
        enc["B[]"] += s.count("B[]"); enc["W[]"] += s.count("W[]"); enc["[tt]"] += s.count("[tt]")
    print(pool, "pass encodings in candidates", dict(enc))
    # questions: first-move corner distribution, to see they are not re-oriented
    q = pd.read_csv(f"{R2}/data/tests/{qf}", dtype=str, keep_default_na=False)
    qfb = [first_b.search(q.loc[i, f"sgf_{j}"]).group(1) for i in range(len(q)) for j in range(1, int(q.loc[i, "num_games"]) + 1)]
    print(pool, "questions raw first Black corner", collections.Counter(map(corner, qfb)).most_common(9))
