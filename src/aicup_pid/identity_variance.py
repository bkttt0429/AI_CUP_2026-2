"""Independent repeated measurements and a two-template noise-adjusted discriminant."""
import numpy as np


def matched_pairs(ordered_edges):
    selected, used = [], set()
    for a, b in ordered_edges:
        if a != b and a not in used and b not in used:
            selected.append((a, b))
            used.update((a, b))
    return selected


def pair_margins(kernel, target, noise, query):
    """Unit-direction midpoint margins. The second direction suppresses within-template scatter."""
    t, n = np.asarray(target), np.asarray(noise)
    m = len(t) // 2
    if m < 2 or len(t) != 2 * m or len(n) != 2 * m:
        raise ValueError("two equal reference sets of at least two observations required")
    d = np.r_[np.ones(m), -np.ones(m)] / m
    ktt = kernel[np.ix_(t, t)].astype(float)
    knt = kernel[np.ix_(n, t)].astype(float)
    knn = kernel[np.ix_(n, n)].astype(float)
    h = np.kron(np.eye(2), np.eye(m) - np.ones((m, m)) / m)
    scatter = h @ knn @ h
    ridge = np.trace(scatter) / (2 * m - 2)
    dist = d @ ktt @ d
    if dist <= 1e-12:
        return 0., 0.
    numerator = (kernel[query, t] - ktt.mean(0)) @ d
    iso = numerator / np.sqrt(dist)
    if ridge <= 1e-12:
        return iso, iso
    projected = h @ knt @ d
    theta = np.linalg.solve(scatter + ridge * np.eye(len(t)), projected)
    numerator -= (kernel[query, n] - knt.mean(1)) @ h @ theta
    norm2 = dist - 2 * theta @ projected + theta @ scatter @ theta
    if norm2 <= 1e-12:
        raise ValueError("degenerate adjusted direction")
    return iso, numerator / np.sqrt(norm2)


def crossed_endpoints(margins):
    """The three axes index the true player, reference panel and query game."""
    x = np.asarray(margins)
    if x.shape != (2, 2, 2) or not np.isfinite(x).all():
        raise ValueError("expected two players, two panels and two queries")
    return {"reference_flip": float(np.mean(np.sign(x[:, 0]) != np.sign(x[:, 1]))),
            "query_flip": float(np.mean(np.sign(x[:, :, 0]) != np.sign(x[:, :, 1]))),
            "accuracy": float(np.mean((x > 0) + .5 * (x == 0)))}


def unit(x):
    x = np.asarray(x, dtype=float)
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    if np.any(n < 1e-12):
        raise ValueError("zero vector")
    return x / n


def bootstrap_matrix(values, strata, seed=111901):
    """Bootstrap independent rows within each stratum, retaining paired column contrasts."""
    x = np.asarray(values, dtype=float)
    strata = np.asarray(strata)
    if x.ndim != 2 or len(x) != len(strata) or not len(x) or not np.isfinite(x).all():
        raise ValueError("invalid cluster matrix")
    rng = np.random.default_rng(seed)
    sums = np.zeros((5000, x.shape[1]))
    for group in np.unique(strata):
        xx = x[strata == group]
        weights = rng.multinomial(len(xx), np.full(len(xx), 1 / len(xx)), 5000)
        sums += weights @ xx
    intervals = np.quantile(sums / len(x), [.025, .975], axis=0)
    return [{"mean": float(a), "ci95": b.tolist()} for a, b in zip(x.mean(0), intervals.T)]
