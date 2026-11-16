"""Within-seed cross-fitting; bootstrap repeats selection, not just evaluation."""
from dataclasses import dataclass

import numpy as np

from .fusion import standardise


@dataclass(frozen=True)
class GridPoint:
    arm: str
    ridge: float | None
    alpha: float


@dataclass
class GridScores:
    pool: str
    seeds: tuple
    owners: tuple
    halves: np.ndarray
    configs: tuple
    score: np.ndarray  # seed, grid, owner, size (1/2/3 equally weighted)
    base: np.ndarray

    def __post_init__(self):
        s, g, c = len(self.seeds), len(self.configs), len(self.owners)
        if self.score.shape != (s, g, c, 3) or self.base.shape != (s, c, 3) or self.halves.shape != (c,):
            raise ValueError("grid shape mismatch")
        if set(self.halves.tolist()) != {0, 1} or len(set(self.owners)) != c or len(set(self.seeds)) != s:
            raise ValueError("need distinct owners/seeds and both halves")
        if not np.isfinite(self.score).all() or not np.isfinite(self.base).all():
            raise ValueError("nonfinite grid")


def add_evidence(baseline, new_score, alpha):
    b, s = np.asarray(baseline, float), np.asarray(new_score, float)
    if b.shape != s.shape or not np.isfinite(b).all() or not np.isfinite(s).all() or alpha < 0:
        raise ValueError("invalid fusion inputs")
    return b.copy() if alpha == 0 else b + alpha * standardise(s)


def ranks(scores, truth):
    s, y = np.asarray(scores), np.asarray(truth)
    own = s[np.arange(len(y)), y, None]
    return ((s > own) | ((s == own) & (np.arange(s.shape[1])[None] < y[:, None]))).sum(1)


def rank_scores(scores, truth):
    r = ranks(scores, truth)
    return np.where(r < 5, np.exp(-r.astype(float)), 0.)


def choose_grid(scores, configs, weights, arm):
    w = np.asarray(weights, dtype=float)
    if w.shape != scores.shape[1:2] or np.any(w < 0) or not np.isfinite(w).all() or w.sum() <= 0:
        raise ValueError("invalid source-half weights")
    ix = [i for i, p in enumerate(configs) if p.arm == arm]
    if not ix:
        raise ValueError("missing arm")
    means = np.einsum("gc,c->g", scores.mean(2), w) / w.sum()
    best = max(means[ix])
    tied = [i for i in ix if abs(means[i] - best) <= 1e-12]
    return min(tied, key=lambda i: (configs[i].alpha, -(configs[i].ridge or 0.), i))


def crossfit_grid(data, weights=None):
    w = np.ones(len(data.owners)) if weights is None else np.asarray(weights, float)
    if w.shape != data.halves.shape or not np.isfinite(w).all() or np.any(w < 0):
        raise ValueError("invalid owner weights")
    if any(w[data.halves == h].sum() <= 0 for h in (0, 1)):
        raise ValueError("empty weighted half")
    selections, deltas, means = {}, {}, {}
    for arm in sorted({c.arm for c in data.configs}):
        d = np.zeros_like(data.base)
        sel = []
        for si, _ in enumerate(data.seeds):
            by_half = []
            for h in (0, 1):
                source_weights = w * (data.halves != h)
                chosen = choose_grid(data.score[si], data.configs, source_weights, arm)
                mask = data.halves == h
                d[si, mask] = data.score[si, chosen, mask] - data.base[si, mask]
                by_half.append(chosen)
            sel.append(by_half)
        selections[arm], deltas[arm] = sel, d
        means[arm] = float(np.einsum("scn,c->", d, w) / (len(data.seeds) * 3 * w.sum()))
    return {"selection": selections, "per_owner_delta": deltas, "delta": means}


def bootstrap_procedure(pools, *, seeds, draws=5000, rng_seed=20261116):
    if draws < 2:
        raise ValueError("need multiple bootstrap draws")
    for data in pools.values():
        if tuple(data.seeds) != tuple(seeds):
            raise ValueError("requested seeds must exactly match saved grid")
    arms = sorted(set.intersection(*[{c.arm for c in d.configs} for d in pools.values()]))
    points = {p: crossfit_grid(d) for p, d in pools.items()}
    rng = np.random.default_rng(rng_seed)
    boot = {a: np.empty(draws) for a in arms}
    frequency = {p: {a: {} for a in arms} for p in pools}
    for b in range(draws):
        outcomes = []
        for p, d in pools.items():
            w = np.zeros(len(d.owners))
            for h in (0, 1):
                ix = np.flatnonzero(d.halves == h)
                # The same multiplicities apply to all seeds of this owner.
                w[ix] = rng.multinomial(len(ix), np.full(len(ix), 1 / len(ix)))
            out = crossfit_grid(d, w)
            outcomes.append(out)
            for arm in arms:
                for selection in out["selection"][arm]:
                    for i in selection:
                        k = str(i)
                        frequency[p][arm][k] = frequency[p][arm].get(k, 0) + 1
        for arm in arms:
            boot[arm][b] = np.mean([o["delta"][arm] for o in outcomes])
    report = {"seeds": list(seeds), "draws": draws, "rng_seed": rng_seed, "arms": {}, "contrasts": {},
              "selection_frequency": frequency, "pools": {},
              "scope": "conditional on fixed candidate pool, baseline and projection; not new independent players"}
    for arm in arms:
        report["arms"][arm] = {"diff": float(np.mean([v["delta"][arm] for v in points.values()])), "ci95": np.quantile(boot[arm], [.025, .975]).tolist()}
    for a, b in [("A1", "A0"), ("A2", "A1"), ("A2", "R"), ("A2", "A2perm")]:
        if a in arms and b in arms:
            report["contrasts"][a + "-" + b] = {"diff": report["arms"][a]["diff"] - report["arms"][b]["diff"],
                                                     "ci95": np.quantile(boot[a] - boot[b], [.025, .975]).tolist()}
    for p, out in points.items():
        report["pools"][p] = {"delta": out["delta"], "selection": out["selection"],
                               "by_seed": {str(seed): {a: float(d[si].mean()) for a, d in out["per_owner_delta"].items()} for si, seed in enumerate(seeds)},
                               "by_size": {a: d.mean(axis=(0, 1)).tolist() for a, d in out["per_owner_delta"].items()}}
    main = report["arms"].get("A2")
    if main:
        positive_pools = all(p["delta"]["A2"] > 0 for p in report["pools"].values())
        report["verdict"] = "promising_pending_robustness" if main["diff"] >= .002 and main["ci95"][0] > 0 and positive_pools else "below_practical_threshold" if main["ci95"][1] < .002 else "inconclusive"
    return report
