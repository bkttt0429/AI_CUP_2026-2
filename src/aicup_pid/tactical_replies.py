"""Opportunity-conditioned response profiles and a symmetric two-candidate comparator."""
import hashlib

import numpy as np
from katago.game.board import Board
from scipy.optimize import minimize
from scipy.special import expit, logit


CONTEXTS = ("friendly_atari", "opponent_atari", "two_liberties", "contact", "open")
ACTIONS = ("nearby", "capture", "new_atari", "connect", "self_atari")


def opportunity(board, player, previous):
    """Classify the opponent's last chain before inspecting the player's move."""
    if previous == Board.PASS_LOC or board.board[previous] != Board.get_opp(player):
        raise ValueError("expected the previous opponent stone")
    adjacent = [board.num_liberties(previous + step) for step in board.adj
                if board.board[previous + step] == player]
    opposing = board.num_liberties(previous)
    if 1 in adjacent:
        return 0
    if opposing == 1:
        return 1
    if opposing == 2 or 2 in adjacent:
        return 2
    return 3 if adjacent else 4


def play_checked(board, player, loc):
    if board.pla != player or (loc != Board.PASS_LOC and board.would_be_suicide(player, loc)):
        raise ValueError("unexpected player or suicide")
    board.play(player, loc)  # also checks occupation, bounds and simple ko


def play_response(board, player, loc, previous):
    if loc == Board.PASS_LOC:
        play_checked(board, player, loc)
        return (False,) * len(ACTIONS)
    other = Board.get_opp(player)
    adj = [loc + step for step in board.adj]
    friendly = {board.group_head[p] for p in adj if board.board[p] == player}
    enemies = [p for p in adj if board.board[p] == other]
    healthy = [p for p in enemies if board.num_liberties(p) > 1]
    play_checked(board, player, loc)
    near = max(abs(board.loc_x(loc) - board.loc_x(previous)), abs(board.loc_y(loc) - board.loc_y(previous))) <= 2
    capture = any(board.board[p] == Board.EMPTY for p in enemies)
    attack = any(board.board[p] == other and board.num_liberties(p) == 1 for p in healthy)
    return near, capture, attack, len(friendly) >= 2, board.num_liberties(loc) == 1


def counts(context, responses, phase, seed):
    """Counts per context followed by five successes; shuffle whole context labels within phases."""
    context = np.asarray(context, dtype=int)
    responses = np.asarray(responses, dtype=int).reshape(len(context), len(ACTIONS))
    phase = np.asarray(phase, dtype=int)
    if len(phase) != len(context) or np.any((context < 0) | (context >= len(CONTEXTS))):
        raise ValueError("invalid opportunity sequence")
    perm = context.copy()
    rng = np.random.default_rng(seed)
    for label in np.unique(phase):
        mask = phase == label
        perm[mask] = rng.permutation(perm[mask])
    summary = []
    obs = np.column_stack([np.ones(len(context), dtype=int), responses])
    for labels in (context, perm):
        table = np.zeros((len(CONTEXTS), len(ACTIONS) + 1), dtype=np.int32)
        np.add.at(table, labels, obs)
        summary.append(table)
    return tuple(summary)


def profile(reference, population):
    """Fit marginal and conditional Bernoulli probabilities, pooling sparse opportunities."""
    ref = np.asarray(reference, dtype=float)
    pop = np.asarray(population, dtype=float)
    prior_m = (pop.sum(0)[1:] + .5) / (pop[:, 0].sum() + 1.)
    prior_c = (pop[:, 1:] + .5) / (pop[:, :1] + 1.)
    sums = ref.sum(-2)
    marginal = (sums[..., 1:] + 32. * prior_m) / (sums[..., :1] + 32.)
    shifted = expit(logit(prior_c) + logit(marginal)[..., None, :] - logit(prior_m))
    conditional = (ref[..., 1:] + 32. * shifted) / (ref[..., :1] + 32.)
    return marginal, conditional


def evidence(query, probabilities):
    """Per-observation marginal score and extra conditional score for each response."""
    q = np.asarray(query, dtype=float)
    total = q[..., :1]
    success = q[..., 1:]
    denom = np.maximum(1., total.sum(-2))
    pm, pc = probabilities
    def ll(p):
        return (success * np.log(p) + (total - success) * np.log1p(-p)).sum(-2) / denom
    marginal = ll(pm[..., None, :])
    return marginal, ll(pc) - marginal


def owner_folds(pool, owners):
    order = sorted(range(len(owners)), key=lambda i: hashlib.sha256(
        f"tactical-owner-v1|{pool}|{owners[i]}".encode()).digest())
    folds = np.empty(len(owners), dtype=int)
    folds[order] = np.arange(len(owners)) % 4
    return folds


def eligible_players(folds, a, b):
    return (folds != a) & (folds != b)


def fit_comparator(true_minus_other):
    """Return weights in original units. Swapping the candidates reverses the log odds."""
    x = np.asarray(true_minus_other, dtype=float)
    if x.ndim != 2 or not len(x) or not np.isfinite(x).all():
        raise ValueError("empty or non-finite training pairs")
    scale = np.maximum(np.sqrt(np.mean(x ** 2, axis=0)), 1e-8)
    z = x / scale
    def objective(w):
        logits = z @ w
        loss = np.logaddexp(0., -logits).mean() + .05 * (w @ w)
        grad = -(expit(-logits) @ z) / len(z) + .1 * w
        return loss, grad
    result = minimize(objective, np.zeros(z.shape[1]), method="L-BFGS-B", jac=True,
                      bounds=[(0., None)] + [(None, None)] * (z.shape[1] - 1),
                      options={"ftol": 1e-12, "gtol": 1e-8})
    if not result.success:
        raise ValueError(f"comparator fit failed: {result.message}")
    return result.x / scale


def swapped_ranks(base_rank, swap):
    r = np.asarray(base_rank)
    return np.where(np.asarray(swap, dtype=bool) & (r < 2), 1 - r, r)
