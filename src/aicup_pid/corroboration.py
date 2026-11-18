"""Within-game temporal deletions and candidate-specific corroboration scores."""
import hashlib

import numpy as np


def quarter_labels(key, movers, length):
    """Contiguous quarters and a matched control, in units of two consecutive plies."""
    movers = np.asarray(movers)
    if movers.ndim != 1 or not 0 < length <= len(movers):
        raise ValueError("invalid observed game length")
    units = (length + 1) // 2
    ordered = np.arange(units) * 4 // units
    shuffled = ordered.copy()
    seed = int.from_bytes(hashlib.sha256(("pid-quarter-v1|" + key).encode()).digest()[:8], "big")
    rng = np.random.default_rng(seed)
    n = length // 2
    patterns = 2 * movers[:2*n:2] + movers[1:2*n:2]
    for pattern in np.unique(patterns):
        ix = np.flatnonzero(patterns == pattern)
        shuffled[ix] = rng.permutation(ordered[ix])
    result = np.full((2, len(movers)), -1, dtype=np.int8)
    result[:, :length] = np.repeat([ordered, shuffled], 2, axis=1)[:, :length]
    return result


def pair_features(scores):
    """Antisymmetric contrasts for both nominees, using nine aligned roster-z views."""
    x = np.asarray(scores, dtype=float)
    if x.ndim != 3 or x.shape[1:] != (9, 2) or not np.isfinite(x).all():
        raise ValueError("expected nine finite two-nominee score views")
    gaps = x[:, :, 0] - x[:, :, 1]
    out = {"intact": gaps[:, 0]}
    for name, sl in (("", slice(1, 5)), ("sham_", slice(5, 9))):
        floor = x[:, sl, :].min(1)
        out[name + "floor"] = floor[:, 0] - floor[:, 1] - gaps[:, 0]
        margins = gaps[:, sl]
        out[name + "persistent"] = np.maximum(0., margins.min(1)) + np.minimum(0., margins.max(1))
    return out


def template_margin(reference_a, reference_b, query):
    """One centered four-quarter trace per game, projected using independent references."""
    def held_out_directions(reference):
        full = reference[:, :, 0]
        other = full.sum(1, keepdims=True) - full
        return other / np.maximum(np.linalg.norm(other, axis=-1, keepdims=True), 1e-12)

    direction = held_out_directions(reference_a) - held_out_directions(reference_b)
    query_trace = np.einsum("bjd,bkd->bjk", direction, query)
    a_trace = np.einsum("bjd,bjkd->bjk", direction, reference_a[:, :, 1:])
    b_trace = np.einsum("bjd,bjkd->bjk", direction, reference_b[:, :, 1:])
    target = query_trace - .5 * (a_trace + b_trace)
    difference = a_trace - b_trace
    return ((target - target.mean(2, keepdims=True)) *
            (difference - difference.mean(2, keepdims=True))).sum(2).mean(1)


def vector_margins(reference_a, reference_b, query):
    """Whole, parallel and perpendicular temporal contrasts, with cross-game offsets."""
    def normalize(x):
        return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)

    d = normalize(normalize(reference_a[:, :, 0].mean(1)) - normalize(reference_b[:, :, 0].mean(1)))
    q = query - query.mean(1, keepdims=True)
    q_parallel = np.einsum("bkd,bd->bk", q, d)

    def candidate_score(query_trace, ref_traces):
        between = ref_traces.sum(1)
        n = ref_traces.shape[1]
        squared_mean = ((between ** 2).sum(-1) - (ref_traces ** 2).sum(axis=(1, 2))) / (n * (n - 1))
        return (query_trace * (between / n)).sum(-1) - .5 * squared_mean

    def score(reference):
        traces = reference[:, :, 1:] - reference[:, :, 1:].mean(2, keepdims=True)
        parallel = np.einsum("bjkd,bd->bjk", traces, d)
        whole = candidate_score(q.reshape(len(q), -1), traces.reshape(len(q), traces.shape[1], -1))
        return np.column_stack([whole, candidate_score(q_parallel, parallel)])

    whole, parallel = (score(reference_a) - score(reference_b)).T
    return np.column_stack([whole, parallel, whole - parallel])
