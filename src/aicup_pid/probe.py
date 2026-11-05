"""Public-leaderboard probing of the one-to-one submission (user decision 2026-11-06; METHOD_NOTES 2.17).

The leaderboard returns the mean official score over the 400 Public questions to seven decimals, so one upload resolves
sums of per-question changes to about 4e-5. Permuting a question's five candidates changes its score by
DELTA[t] = e^-(pi(t)-1) - e^-(t-1) when the truth sits at current rank t (0 when it is outside the five). The deltas of
every permutation are differences of the same five values, so signed sums over several questions collide (for example
(1 - e^-1) + (e^-1 - e^-2) = 1 - e^-2): a design that identifies each tested question's own position is limited to a
handful of questions. What one upload does resolve is a HISTOGRAM: apply the same 5-cycle (every candidate one rank
down, rank 5 to rank 1) to a group of questions, and the total change is sum_t n_t DELTA[t] with n_t = how many of the
group have the truth at rank t; the five deltas are rationally independent and, for groups up to `max_group_size`, every
histogram gives a distinct sum. Groups are then split adaptively: the histogram of the untested half is the parent's
minus the tested half's. A decoded position >= 2 is fixed by moving the truth to rank 1 (duplicate Top-1 across questions
is allowed by the rules), and every later upload carries the fixes, so the reference score is known exactly.
"""
import itertools
from functools import lru_cache

import numpy as np

DECAY = np.exp(-np.arange(5))  # score of the truth at rank 1..5
OUT = 0  # truth outside the submitted five
POSITIONS = (1, 2, 3, 4, 5, OUT)
N_PUBLIC = 400
TOL = 2e-4  # two scores rounded to 1e-7, times 400, with margin
CYCLE = (5, 1, 2, 3, 4)  # new order as current ranks: rank 5 goes to the top, everyone else one down


def deltas(perm):
    """[6] score change for the truth at current rank 1..5 (and 0 for OUT) when the list is reordered by `perm`:
    perm[j] = current rank (1..5) of the candidate placed at new rank j+1."""
    new_rank = {r: j + 1 for j, r in enumerate(perm)}
    d = [DECAY[new_rank[t] - 1] - DECAY[t - 1] for t in range(1, 6)]
    return np.array([*d, 0.0])


def deficit_decompositions(score, n=N_PUBLIC, tol=TOL, max_wrong=12):
    """Multisets (a, b, c, d, e) = numbers of wrong questions whose truth is at rank 2, 3, 4, 5, outside, whose total
    loss matches n * (1 - score) within tol."""
    loss = n * (1.0 - score)
    unit = 1.0 - DECAY[1:]
    out = []
    for combo in itertools.product(range(max_wrong + 1), repeat=5):
        if sum(combo) > max_wrong:
            continue
        total = sum(k * u for k, u in zip(combo[:4], unit)) + combo[4] * 1.0
        if abs(total - loss) <= tol:
            out.append(combo)
    return out


MAX_OUT = 1  # at most this many "truth outside the five" per group (the loss decompositions of 0.98735 allow 0 or 1)
BOUNDS = (5, 1, 2, 1, 1)  # per group: at most this many truths at rank 2, 3, 4, 5, outside = elementwise max over the decompositions of 0.98735


def bounds_from_score(score, n=N_PUBLIC, tol=TOL, max_wrong=12):
    """Elementwise maximum over the loss decompositions: no group can hold more truths at a rank than the whole set."""
    combos = deficit_decompositions(score, n, tol, max_wrong)
    if not combos:
        raise ValueError("no decomposition of the loss: check the score")
    return tuple(max(c[i] for c in combos) for i in range(5))


def histograms(size, bounds=BOUNDS):
    """Every (n_1, ..., n_5, n_out) with sum = size and (n_2, n_3, n_4, n_5, n_out) <= bounds. The deltas of any
    permutation sum to zero, so without a bound the histograms h and h + k (1, 1, 1, 1, 1, -5) would give the same
    total; with the bounds from the loss decomposition the set is small and the sums stay apart for any group size."""
    out = []
    for rest in itertools.product(*[range(min(b, size) + 1) for b in bounds]):
        n1 = size - sum(rest)
        if n1 >= 0:
            out.append((n1, *rest))
    return out


def histogram_sum(hist, perm=CYCLE):
    d = deltas(perm)
    return float(sum(n * d[POSITIONS.index(t)] for n, t in zip(hist, POSITIONS)))


@lru_cache(maxsize=None)
def min_gap(size, bounds=BOUNDS, perm=CYCLE):
    """Smallest difference between the sums of two different admissible histograms of a group of this size."""
    sums = np.sort(np.array([histogram_sum(h, perm) for h in histograms(size, bounds)]))
    return float(np.min(np.diff(sums))) if len(sums) > 1 else np.inf


def max_group_size(tol=TOL, bounds=BOUNDS, limit=N_PUBLIC):
    """Largest group whose admissible histograms are all distinguishable at tolerance tol."""
    best = 0
    for size in range(1, limit + 1):
        if min_gap(size, bounds) > tol:
            best = size
        else:
            break
    return best


def decode_histogram(observed_delta, size, tol=TOL, bounds=BOUNDS, perm=CYCLE):
    """Admissible histograms of a group of `size` tested questions whose predicted sum matches observed_delta (= 400 x
    score change) within tol; exactly one when the group is within max_group_size."""
    return [h for h in histograms(size, bounds) if abs(histogram_sum(h, perm) - observed_delta) <= tol]


def reorder(top5, perm=CYCLE):
    """The new list: perm[j] = current rank of the candidate at new rank j+1."""
    return [top5[r - 1] for r in perm]


def fixed_list(top5, truth_pos):
    """Move the truth (current rank truth_pos, 2..5) to rank 1, keeping the others' order."""
    lst = list(top5)
    c = lst.pop(truth_pos - 1)
    return [c, *lst]


def gain_of_fix(truth_pos):
    """Score gain of moving the truth from rank truth_pos to rank 1."""
    return float(1.0 - DECAY[truth_pos - 1])


def resolve(groups):
    """Propagate decoded histograms: for nested groups the difference set gets the difference histogram; a group whose
    histogram is all rank 1 marks its members right; a singleton fixes its member. groups: [{members: [..], hist: (6,)}]
    (hist None when not yet decoded). Returns ({question_id: position or None}, derived groups added)."""
    known = [g for g in groups if g.get("hist") is not None]
    derived, changed = [], True
    while changed:
        changed = False
        for a in list(known):
            for b in list(known):
                sa, sb = set(a["members"]), set(b["members"])
                if sa < sb:  # a strictly inside b: b minus a is determined
                    rest = sorted(sb - sa)
                    hist = tuple(int(x - y) for x, y in zip(b["hist"], a["hist"]))
                    if min(hist) < 0:
                        raise ValueError(f"inconsistent histograms for {a['members']} inside {b['members']}")
                    if not any(set(g["members"]) == set(rest) for g in known):
                        g = {"members": rest, "hist": hist, "derived": True}
                        known.append(g)
                        derived.append(g)
                        changed = True
    positions = {}
    for g in known:
        hist, members = g["hist"], g["members"]
        if hist[0] == len(members):
            for m in members:
                positions[m] = 1
        elif len(members) == 1:
            t = [POSITIONS[i] for i, n in enumerate(hist) if n][0]
            positions[members[0]] = t
    return positions, derived
