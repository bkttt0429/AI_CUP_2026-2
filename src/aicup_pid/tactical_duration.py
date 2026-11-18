"""Binary local replies under tactical opportunities and preceding exchange duration."""
import numpy as np
from scipy.special import expit, logit


def advance_streak(previous, current, streak):
    if previous is None or current is None or max(abs(a - b) for a, b in zip(previous, current)) > 2:
        return 0
    return min(2, streak + 1)


def duration_tables(context, streak, response, phase, seed):
    c, s, y, phase = (np.asarray(v, dtype=int) for v in (context, streak, response, phase))
    if any(v.shape != c.shape for v in (s, y, phase)):
        raise ValueError("misaligned decisions")
    rng = np.random.default_rng(seed)
    cp, sp = c.copy(), s.copy()
    for p in np.unique(phase):
        use = phase == p
        cp[use] = rng.permutation(c[use])
        for k in np.unique(c[use]):
            take = use & (c == k)
            sp[take] = rng.permutation(s[take])
    out = []
    for key, run in ((c, s), (cp, s), (c, sp)):
        table = np.zeros((5, 3, 2), dtype=int)
        np.add.at(table[:, :, 0], (key, run), 1)
        np.add.at(table[:, :, 1], (key, run), y)
        out.append(table)
    return tuple(out)


def duration_probabilities(ref, pop):
    def rate(table):
        return (table[..., 1] + .5) / (table[..., 0] + 1.)
    def shrink(table, prior):
        return (table[..., 1] + 32. * prior) / (table[..., 0] + 32.)
    pop_c = pop.sum(axis=1)
    ref_c = ref.sum(axis=1)
    prior = rate(pop_c.sum(axis=0))
    marginal = shrink(ref_c.sum(axis=0), prior)
    prior_c = rate(pop_c)
    context = shrink(ref_c, expit(logit(prior_c) + logit(marginal) - logit(prior)))
    adjustment = (logit(context) - logit(prior_c))[:, None]
    temporal = shrink(ref, expit(logit(rate(pop)) + adjustment))
    return marginal, context[:, None], temporal


def log_scores(query, probs):
    n, y = query[..., 0], query[..., 1]
    return np.array([(y * np.log(p) + (n - y) * np.log1p(-p)).sum() / max(1, n.sum()) for p in probs])
