"""Tail evidence from single reference games (METHOD_NOTES 2.117): the opponents of a question against the opponents of a
candidate, and its relatives.

A sequence network returns a vector for the side flagged as the subject of a game; with the other colour flagged it describes
the OPPONENT (scripts/pid_v2_opp_embed.py). A player meets some opponents again, so the opponent of a question game may also
be the opponent in one of the owner's reference games. On the official candidates (leave-one-game-out, scripts/pid_probe_opp.py)
the nearest opponent vector belongs to the same candidate in 3.4% of the games; 0.15% if nothing were there. The event is
rare and then decisive, so only the tail counts:
  per network cos(o_g, o_r) of a question game g and a reference game r; the networks' cosines are averaged;
  m[g, c] = the maximum over the reference games of candidate c;  z[g, .] = m standardised over the candidates;
  score[q, c] = sum over the games g of question q of max(z[g, c] - tau, 0).
The score is added to the fused score with its own weight. It is NOT standardised again: most of its entries are zero.
The same statistic on the subject's OWN vectors says that one reference game looks like the question game (the prototype
averages that away), and on the sum of both sides that the same pairing met again.
"""
import numpy as np


def nearest_z(cands, questions, sides):
    """(z [G, C], {question game key: row}): per question game and candidate, the best single reference game under the SUM
    over `sides` of the network-averaged cosines, standardised over the candidates. cands {player: [(key, colour)]},
    questions [[(key, colour)]], sides = [[{key: vector} per network] per side]; candidates in sorted id order. Games without
    a vector are skipped; a candidate without any takes the row mean (z = 0)."""
    players = sorted(cands)
    have = set(sides[0][0])
    ckeys = [k for p in players for k, _ in cands[p] if k in have]
    cidx = np.array([i for i, p in enumerate(players) for k, _ in cands[p] if k in have])
    qkeys = sorted({k for q in questions for k, _ in q if k in have})
    if set(qkeys) & set(ckeys):
        raise ValueError("a question game is among the reference games")
    unit = lambda X: X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)  # noqa: E731
    cos = np.zeros((len(qkeys), len(ckeys)))
    for nets in sides:
        one_side = np.zeros_like(cos)
        for one in nets:
            one_side += unit(np.stack([one[k] for k in qkeys]).astype(np.float64)) @ unit(np.stack([one[k] for k in ckeys]).astype(np.float64)).T
        cos += one_side / len(nets)
    order = np.argsort(cidx, kind="stable")
    sorted_c = cidx[order]
    starts = np.flatnonzero(np.r_[True, sorted_c[1:] != sorted_c[:-1]])
    m = np.full((len(qkeys), len(players)), np.nan)
    m[:, sorted_c[starts]] = np.maximum.reduceat(cos[:, order], starts, axis=1)
    mu = np.nanmean(m, axis=1, keepdims=True)
    sd = np.nanstd(m, axis=1, keepdims=True)
    sd[sd < 1e-12] = 1.0
    z = np.where(np.isnan(m), 0.0, (m - mu) / sd)
    return z, {k: i for i, k in enumerate(qkeys)}


def opp_z(cands, questions, opp):
    """nearest_z for one side: opp = one or several {key: opponent-side vector}."""
    return nearest_z(cands, questions, [list(opp) if isinstance(opp, (list, tuple)) else [opp]])


def opp_scores(cands, questions, opp, tau=3.0, zq=None):
    """[Q, C]: sum over a question's games of the part of z above tau. zq = opp_z(...) / nearest_z(...) when several tau are wanted."""
    z, row = zq if zq is not None else opp_z(cands, questions, opp)
    h = np.maximum(z - tau, 0.0)
    out = np.zeros((len(questions), z.shape[1]))
    for qi, q in enumerate(questions):
        for k, _ in q:
            if k in row:
                out[qi] += h[row[k]]
    return out
