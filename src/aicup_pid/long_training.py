"""Checkpoint/weight interventions and fixed-rival score measurements."""
import numpy as np


def sequence_variants(base, early, late, share):
    """Input matrices are weighted means of member-standardized roster scores."""
    if not 0 < share < 1 or base.shape != early.shape or base.shape != late.shape:
        raise ValueError("invalid family weight or incompatible sequence matrices")
    change = share * (late - early)
    return {"baseline": base, "replace": base + change, "add": (base + share*late) / (1+share),
            "weight": (base + share*early) / (1+share), "mix": base + .5*change}


def pair_scores(scores, truth, rival, scale=None):
    """Center both finalists on the other candidates and optionally fix the SD."""
    s = np.asarray(scores, dtype=np.float64)
    a, b = np.asarray(truth), np.asarray(rival)
    if s.ndim != 2 or s.shape[1] < 4 or a.shape != (len(s),) or b.shape != a.shape or np.any(a == b):
        raise ValueError("two distinct candidate columns required for each query")
    row = np.arange(len(s))
    pair = np.column_stack([s[row, a], s[row, b]])
    n = s.shape[1]-2
    mu = (s.sum(1)-pair.sum(1)) / n
    var = ((s*s).sum(1) - (pair*pair).sum(1)) / n - mu*mu
    sd = np.sqrt(np.maximum(var, 0.))
    divisor = sd if scale is None else np.asarray(scale)
    if np.any(divisor < 1e-12) or not np.isfinite(divisor).all():
        raise ValueError("degenerate rival background")
    return (pair-mu[:, None]) / divisor[:, None], sd


def sibling_rows(owners):
    ids = np.asarray(owners)
    if ids.ndim != 1 or len(ids) % 3 or not np.all(ids.reshape(-1, 3) == ids[::3, None]):
        raise ValueError("expected three ordered games per owner")
    row = np.arange(len(ids))
    return row[:, None]//3*3 + (row[:, None] % 3 + [1, 2]) % 3
