"""Fit only on separate games and owners, then evaluate all frozen finalist decisions."""
import hashlib
from itertools import combinations_with_replacement
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pid_tactical_inputs import CACHE, PLAN, POOLS, ROOT, atomic_json, atomic_npz, digest_file, guard
from aicup_pid.tactical_replies import ACTIONS, CONTEXTS, eligible_players, evidence, fit_comparator, owner_folds, profile, swapped_ranks

ARMS = ("marginal", "shuffled", "easy", "hard")
REPORT = ROOT / "results/final/tactical_replies_20261118.json"


def generator(label):
    key = hashlib.sha256(("tactical-pairs-v1|" + label).encode()).digest()
    return np.random.default_rng(int.from_bytes(key[:8], "big"))


def draw_other(owners, eligible, rng):
    choices = np.flatnonzero(eligible)
    return np.array([rng.choice(choices[choices != p]) for p in owners])


def pair_features(scores, q, probs, ix, a, b):
    def channels(ids):
        return evidence(q[ix], (probs[0][ids], probs[1][ids]))
    ma, ca = channels(a)
    mb, cb = channels(b)
    return np.column_stack((scores[ix, a] - scores[ix, b], ma - mb, ca - cb))


def load_pool(pool, source, baseline):
    path = CACHE / f"base_{pool}.npz"
    if digest_file(path) != baseline["pools"][pool]["cache_sha"]:
        raise ValueError("baseline cache changed")
    with np.load(path, allow_pickle=False) as z:
        b = dict(z)
    idx = {key: i for i, key in enumerate(source["keys"])}
    refs = np.array([[idx[k] for k in row] for row in b["refs"]])
    qidx = np.array([idx[k] for k in b["keys"]])
    all_idx = np.r_[refs.ravel(), qidx]
    if (len(set(all_idx)) != 6000 or any(not source["keys"][k].startswith(pool + "|c:") for k in all_idx)
            or not np.array_equal(b["truth"], np.repeat(np.arange(600), 3))):
        raise ValueError("unexpected role overlap or pool/owner order")
    b["q"] = source["intact"][qidx]
    b["ref"] = source["intact"][refs].sum(1)
    b["perm"] = source["shuffled"][refs].sum(1)
    b["signatures"] = source["sig"][qidx]
    return b


def deploy(pool, b):
    scores, truth, q = b["scores"], b["truth"], b["q"]
    folds = owner_folds(pool, b["owners"])
    pair = np.argsort(-scores, axis=1, kind="stable")[:, :2]
    pair_folds = np.sort(folds[pair], axis=1)
    is_fit_game = np.arange(len(q)) % 3 == 0
    eval_rows = np.flatnonzero(~is_fit_game)
    inside = (pair == truth[:, None]).any(1)
    margins = {arm: np.full(len(q), np.nan) for arm in ARMS}
    raw = np.full((len(q), 3), np.nan)  # marginal, intact conditional, shuffled conditional
    random_raw = np.full((len(q), 2), np.nan)
    chan = np.full((len(q), len(ACTIONS)), np.nan)
    choices = draw_other(truth, np.ones(len(folds), bool), generator(pool + "|evaluation random rivals"))
    random_folds = np.sort(folds[np.column_stack((truth, choices))], axis=1)
    fit_records = {}
    for fa, fb in combinations_with_replacement(range(4), 2):
        keep = eligible_players(folds, fa, fb)
        pair_ok = keep[pair].all(1)
        fit = np.flatnonzero(is_fit_game & inside & pair_ok)
        a = truth[fit]
        rival = np.where(pair[fit, 0] == a, pair[fit, 1], pair[fit, 0])
        if not len(fit) or not np.all(keep[a] & keep[rival]):
            raise ValueError("fit owner or finalist outside allowed folds")
        probs = profile(b["ref"], b["ref"][keep].sum(0))
        shuf = profile(b["perm"], b["perm"][keep].sum(0))
        e = draw_other(a, keep, generator(f"{pool}|fit|{fa}|{fb}"))
        x = pair_features(scores, q, probs, fit, a, rival)
        xp = pair_features(scores, q, shuf, fit, a, rival)
        xe = pair_features(scores, q, probs, fit, a, e)
        weights = {"hard": fit_comparator(x), "marginal": fit_comparator(x[:, :6]),
                   "shuffled": fit_comparator(xp), "easy": fit_comparator(xe)}
        test = np.flatnonzero(~is_fit_game & (pair_folds == [fa, fb]).all(1))
        first, second = pair[test].T
        if np.any(keep[first] | keep[second]):
            raise ValueError("evaluation finalist seen in comparator fitting")
        u = pair_features(scores, q, probs, test, first, second)
        up = pair_features(scores, q, shuf, test, first, second)
        for arm, features in (("hard", u), ("easy", u), ("marginal", u[:, :6]), ("shuffled", up)):
            margins[arm][test] = features @ weights[arm]
        # All quantities above are functions only of the observed query and nominated players at evaluation.
        raw[test, 0] = u[:, 1:6].mean(1)
        raw[test, 1] = raw[test, 0] + u[:, 6:].mean(1)
        raw[test, 2] = up[:, 1:6].mean(1) + up[:, 6:].mean(1)
        chan[test] = u[:, 6:]
        # Uniform reference for a post-scoring diagnostic only. Its folds are excluded too.
        rr = np.flatnonzero(~is_fit_game & (random_folds == [fa, fb]).all(1))
        xu = pair_features(scores, q, probs, rr, truth[rr], choices[rr])
        random_raw[rr, 0] = xu[:, 1:6].mean(1)
        random_raw[rr, 1] = random_raw[rr, 0] + xu[:, 6:].mean(1)
        fit_records[f"{fa},{fb}"] = {"eligible_owners": int(keep.sum()), "fit_pairs": len(fit), "test_pairs": len(test),
                                      "weights_in_raw_units": {arm: w.tolist() for arm, w in weights.items()}}
    base = b["rank"][eval_rows]
    results = {"truth": truth[eval_rows], "pair": pair[eval_rows], "base": base,
               "keys": b["keys"][eval_rows], "physical": b["signatures"][eval_rows], "folds": folds,
               "raw": raw[eval_rows], "random_raw": random_raw[eval_rows], "random_rival": choices[eval_rows],
               "context_channels": chan[eval_rows], "opportunities": q[eval_rows, :, 0],
               "base_gap": (scores[np.arange(len(q)), pair[:, 0]] - scores[np.arange(len(q)), pair[:, 1]])[eval_rows]}
    for arm in ARMS:
        gap = margins[arm][eval_rows]
        results["gap_" + arm] = gap
        results["rank_" + arm] = swapped_ranks(base, gap < 0)
    if any(not np.isfinite(v).all() for v in results.values() if v.dtype.kind == "f"):
        raise ValueError("unscored evaluations")
    return results, fit_records


def competition(rank):
    return np.where(rank < 5, np.exp(-rank.astype(float)), 0.)


def summarise(results):
    def cat(k):
        return np.concatenate([results[p][k] for p in POOLS])
    truth = cat("truth")
    cluster = truth + np.repeat([0, 600], [len(results[p]["truth"]) for p in POOLS])
    if len(cluster) != 2400 or not np.all(np.bincount(cluster, minlength=1200) == 2):
        raise ValueError("two evaluation games per query owner required")
    rng = np.random.default_rng(111805)
    weights = np.hstack([rng.multinomial(600, np.full(600, 1 / 600), size=5000) for _ in POOLS]).astype(float)
    all_rows = np.ones(len(truth), bool)
    def interval(values, mask=all_rows):
        v = np.asarray(values)[mask]
        ids = cluster[mask]
        if not len(v):
            raise ValueError("empty reported group")
        sums = np.bincount(ids, weights=v, minlength=1200)
        n = np.bincount(ids, minlength=1200)
        means = (weights @ sums) / (weights @ n)
        return {"mean": float(v.mean()), "ci95": np.quantile(means, [.025, .975]).tolist(), "rows": len(v)}
    base, pair = cat("base"), cat("pair")
    inside = base < 2
    if not np.array_equal(inside, (pair == truth[:, None]).any(1)):
        raise ValueError("finalist membership does not match full-ensemble ranks")
    sign = np.where(pair[:, 0] == truth, 1, -1)
    base_sc = competition(base)
    arms, contrasts = {}, {}
    subsets = {"both": all_rows, "pub2": cluster < 600, "priv2": cluster >= 600}
    for arm in ARMS:
        rank = cat("rank_" + arm)
        want = competition(rank)
        detail = {}
        for name, mask in subsets.items():
            detail[name] = {**interval(want - base_sc, mask), "rank_score": float(want[mask].mean()),
                            "rescued": int((mask & (base != 0) & (rank == 0)).sum()),
                            "broken": int((mask & (base == 0) & (rank != 0)).sum()),
                            "swapped": int((mask & (cat("gap_" + arm) < 0)).sum())}
        arms[arm] = detail
    for other in ("marginal", "shuffled", "easy"):
        contrasts["hard_minus_" + other] = interval(competition(cat("rank_hard")) - competition(cat("rank_" + other)))
    raw = cat("raw") * sign[:, None]
    random = cat("random_raw")
    def win(x):
        return (x > 0).astype(float) + .5 * (x == 0)
    diagnostics = {"current_pair_base_accuracy": interval((base == 0).astype(float), inside)}
    for tag, v in (("marginal", raw[:, 0]), ("conditional", raw[:, 1]), ("shuffled", raw[:, 2]),
                   ("uniform_marginal", random[:, 0]), ("uniform_conditional", random[:, 1])):
        diagnostics[tag] = {"true_minus_rival": interval(v, inside), "pair_accuracy": interval(win(v), inside)}
    diagnostics["conditional_increment"] = interval(raw[:, 1] - raw[:, 0], inside)
    diagnostics["conditional_minus_shuffled"] = interval(raw[:, 1] - raw[:, 2], inside)
    diagnostics["conditional_accuracy_uniform_minus_actual"] = interval(win(random[:, 1]) - win(raw[:, 1]), inside)
    diagnostics["extra_by_response"] = {action: interval(cat("context_channels")[:, i] * sign, inside)
                                        for i, action in enumerate(ACTIONS)}
    return {"rows": len(truth), "owners": 1200, "base_rank_score": float(base_sc.mean()), "base_top1": int((base == 0).sum()),
            "true_in_top_two": int(inside.sum()), "rank2": int((base == 1).sum()),
            "opportunities_in_test": dict(zip(CONTEXTS, map(int, cat("opportunities").sum(0)))),
            "arms": arms, "contrasts": contrasts, "descriptive": diagnostics}


def main():
    budget = guard()
    ext = json.loads((CACHE / "extract.json").read_text())
    baseline = json.loads((CACHE / "baseline.json").read_text())
    if ext["plan"] != digest_file(PLAN) or baseline["plan"] != ext["plan"] or ext["data"] != digest_file(CACHE / "tactics.npz"):
        raise ValueError("upstream data or protocol mismatch")
    with np.load(CACHE / "tactics.npz", allow_pickle=False) as z:
        source = dict(z)
    results, fits, artifacts = {}, {}, {}
    for pool in POOLS:
        b = load_pool(pool, source, baseline)
        results[pool], fits[pool] = deploy(pool, b)
        path = CACHE / f"decisions_{pool}.npz"
        atomic_npz(path, **results[pool])
        artifacts[pool] = digest_file(path)
    summary = summarise(results)
    h = summary["arms"]["hard"]["both"]
    summary["adopt"] = bool(h["mean"] >= .002 and h["ci95"][0] > 0)
    summary["conditional_above_both_controls"] = all(summary["contrasts"]["hard_minus_" + a]["ci95"][0] > 0
                                                      for a in ("marginal", "shuffled"))
    files = [Path(__file__), ROOT / "scripts/pid_tactical_inputs.py", ROOT / "scripts/pid_tactical_extract.py",
             ROOT / "src/aicup_pid/tactical_replies.py", PLAN]
    rec = {"summary": summary, "fit": fits, "extract": ext, "baseline_document_sha": baseline["ensemble"],
           "baseline_reproduction": {p: {k:v for k,v in baseline["pools"][p].items() if k != "sources"} for p in POOLS},
           "code": {str(f.relative_to(ROOT)): digest_file(f) for f in files}, "decisions": artifacts, "budget_start": budget}
    atomic_json(REPORT, rec)
    print(json.dumps({"hard": h, "contrasts": summary["contrasts"], "verifier_pending": True}), flush=True)


if __name__ == "__main__":
    main()
