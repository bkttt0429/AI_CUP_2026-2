"""Global one-to-one assignment of questions to candidates (fork plan line B3).

The official pools have 400 questions for 400 candidates, 100 per size, so very likely one question per candidate. When
that holds, the Top-1 picks should form a permutation: assign_topk replaces the per-question argmax by the assignment that
maximises the summed fused scores under the one-to-one constraint (Hungarian algorithm; a per-row constant such as the
log-softmax normaliser does not change it), and keeps the rest of each row's order for Top-2..k. With more questions than
candidates the unassigned rows keep their argmax. assignment_report counts what the assignment changed.
"""
import numpy as np
from scipy.optimize import linear_sum_assignment

from aicup_pid.fusion import topk


def assign_topk(S, k=5, capacity=1):
    """[Q, min(k, C)] candidate indices: Top-1 from the one-to-one assignment, then the row's remaining order.
    capacity > 1: every candidate may take up to `capacity` questions (the columns are tiled), for pools with several
    questions per candidate."""
    S = np.asarray(S, dtype=float)
    base = topk(S, k)
    Q, C = S.shape
    if Q == 0 or C == 0:
        return base
    capacity = int(capacity)
    if capacity < 1:
        raise ValueError("capacity must be >= 1")
    rows, cols = linear_sum_assignment(np.tile(S, (1, capacity)) if capacity > 1 else S, maximize=True)
    cols = cols % C
    out = base.copy()
    width = base.shape[1]
    for q, c in zip(rows, cols):
        if out[q, 0] == c:
            continue
        rest = [x for x in base[q].tolist() if x != c]
        out[q] = np.array([c, *rest][:width])
    return out


def assignment_report(S, assigned):
    """{n_questions, n_candidates, distinct_top1_before, distinct_top1_after, n_top1_changed}."""
    S = np.asarray(S, dtype=float)
    Q, C = S.shape
    if Q == 0:
        return {"n_questions": 0, "n_candidates": int(C), "distinct_top1_before": 0, "distinct_top1_after": 0, "n_top1_changed": 0}
    before = np.argmax(S, axis=1)
    after = np.asarray(assigned)[:, 0]
    return {"n_questions": int(Q), "n_candidates": int(C), "distinct_top1_before": int(len(set(before.tolist()))),
            "distinct_top1_after": int(len(set(after.tolist()))), "n_top1_changed": int((before != after).sum())}


def assign_blocks(S, blocks, apply, k=5):
    """assign_topk block by block (11-11 release: the original questions and the added ones are separate one-to-one sets
    over the same candidates). blocks [Q] labels; apply {label: bool}: a block with False keeps its per-question Top-k."""
    S = np.asarray(S, dtype=float)
    blocks = np.asarray(blocks)
    out = topk(S, k)
    for b in sorted(set(blocks.tolist())):
        if apply.get(b):
            rows = np.flatnonzero(blocks == b)
            out[rows] = assign_topk(S[rows], k)
    return out


def assign_soft(S, lam, k=5, copies=3):
    """Top-k with a penalty for sharing a candidate: the j-th question a candidate takes costs (j - 1) * lam of fused score
    (assignment over `copies` penalised copies of every column). lam = 0 is the per-question Top-1 while collisions stay within
    `copies`; a large lam is assign_topk. For pools where most players have one question but some have several (the v2 added
    pools): a collision survives only when both questions prefer the candidate by more than lam over their next choice."""
    S = np.asarray(S, dtype=float)
    base = topk(S, k)
    Q, C = S.shape
    if Q == 0 or C == 0:
        return base
    big = np.concatenate([S - j * float(lam) for j in range(int(copies))], axis=1)
    rows, cols = linear_sum_assignment(big, maximize=True)
    cols = cols % C
    out = base.copy()
    width = base.shape[1]
    for q, c in zip(rows, cols):
        if out[q, 0] != c:
            out[q] = np.array([c, *[x for x in base[q].tolist() if x != c]][:width])
    return out


def sinkhorn_prices(S, tau, iters=300):
    """log v [C] <= 0: the price of every candidate in the entropic assignment with row sums 1 and column sums <= 1
    (METHOD_NOTES 2.122). A candidate that several questions want gets a negative price; an uncontested one keeps 0."""
    S = np.asarray(S, dtype=float)
    K = np.exp((S - S.max(axis=1, keepdims=True)) / float(tau))
    v = np.ones(S.shape[1])
    for _ in range(int(iters)):
        u = 1.0 / np.maximum(K @ v, 1e-300)
        v = np.minimum(1.0, 1.0 / np.maximum(K.T @ u, 1e-300))
    return np.log(np.maximum(v, 1e-300))


def assign_sinkhorn(S, tau, k=5, lam=None):
    """Top-k of the scores minus tau x the Sinkhorn price of each candidate: unlike assign_soft / assign_topk it changes the
    whole order, not only the Top-1. With lam, the Top-1 is the one of assign_soft(S, lam) and the rest follows the priced
    order (a collision between two confident questions survives, as in assign_soft)."""
    S = np.asarray(S, dtype=float)
    Q, C = S.shape
    if Q == 0 or C == 0:
        return topk(S, k)
    order = topk(S + float(tau) * sinkhorn_prices(S, tau)[None, :], min(C, k + 3))
    out = order[:, :min(k, C)].copy()
    if lam is not None:
        first = assign_soft(S, lam, k)[:, 0]
        for q, c in enumerate(first):
            if out[q, 0] != c:
                out[q] = np.array([c, *[x for x in order[q].tolist() if x != c]][:out.shape[1]])
    return out
