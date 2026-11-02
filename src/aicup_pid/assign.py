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


def assign_topk(S, k=5):
    """[Q, min(k, C)] candidate indices: Top-1 from the one-to-one assignment, then the row's remaining order."""
    S = np.asarray(S, dtype=float)
    base = topk(S, k)
    Q, C = S.shape
    if Q == 0 or C == 0:
        return base
    rows, cols = linear_sum_assignment(S, maximize=True)
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
