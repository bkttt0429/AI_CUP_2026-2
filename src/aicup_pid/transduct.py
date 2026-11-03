"""Transductive self-training on a closed candidate pool (fork plan, robustness line).

After a first pass (fused scores, optionally the one-to-one assignment), the questions whose winning margin is at least
`tau` are treated as solved: their games are added to the matched candidate's library, the component models are
refitted on the enlarged libraries and every question is rescored with the same fusion weights. The pool is closed, so
every confident answer also sharpens the models the unconfident questions are compared against. Each round rebuilds the
libraries from the original candidate games plus the currently confident questions (no accumulation of stale answers).
"""
import numpy as np

from aicup_pid.assign import assign_topk
from aicup_pid.components import fit_from_parsed, fit_local, score_components
from aicup_pid.fusion import Fusion, topk

THREE = ("opening", "style", "local")


def margins_of(S_f, top1):
    """Winning margin of each question: its Top-1's score minus the best other candidate's ([Q])."""
    S = np.asarray(S_f, dtype=float)
    top1 = np.asarray(top1)
    if S.shape[0] == 0:
        return np.zeros(0)
    rows = np.arange(len(top1))
    best = S[rows, top1]
    masked = S.copy()
    masked[rows, top1] = -np.inf
    return best - masked.max(axis=1)


def fuse(S, weights, comps=THREE):
    fus = Fusion()
    fus.w = np.asarray(weights, dtype=float)
    return fus.scores([S[c] for c in comps])


def self_train(by_player, moves, feats, cfg, q_keys, S, weights, tau=3.0, rounds=1, assign=True, comps=THREE, k=5):
    """Self-training from the component scores S of the questions q_keys against the libraries by_player.

    Returns (Top-k indices [Q, k], final fused scores [Q, C], info) with info["rounds"] = per round {confident, games_added}
    and info["confident"] = the last round's confident question indices (for the precision of the confident set)."""
    players = sorted(by_player)
    S_f = fuse(S, weights, comps)
    idx = assign_topk(S_f, k) if assign else topk(S_f, k)
    history, confident = [], np.zeros(0, dtype=int)
    for r in range(rounds):
        confident = np.flatnonzero(margins_of(S_f, idx[:, 0]) >= tau)
        history.append({"round": r + 1, "confident": int(len(confident)), "games_added": int(sum(len(q_keys[q]) for q in confident)),
                        "margin_threshold": float(tau)})
        if len(confident) == 0:
            break
        library = {p: list(keys) for p, keys in by_player.items()}
        for q in confident:
            library[players[idx[q, 0]]] = library[players[idx[q, 0]]] + list(q_keys[q])
        opening, style = fit_from_parsed(library, moves, feats, cfg)
        local = fit_local(library, moves, cfg) if "local" in comps else None
        S_new = score_components(comps, opening, style, q_keys, moves, feats, local)
        S_f = fuse(S_new, weights, comps)
        idx = assign_topk(S_f, k) if assign else topk(S_f, k)
    return idx, S_f, {"rounds": history, "players": players, "confident": confident.tolist()}
