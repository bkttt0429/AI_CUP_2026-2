"""Independent arithmetic audit from retained counts, reference sets and frozen ensemble scores."""
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pid_tactical_inputs import CACHE, POOLS, ROOT, atomic_json, digest_file, guard

OUT = ROOT / "results/final"


def load(name):
    with np.load(CACHE / name, allow_pickle=False) as z:
        return dict(z)


def report(name):
    return json.loads((OUT / f"{name}_20261118.json").read_text())


def close(x, y):
    np.testing.assert_allclose(x, y, rtol=1e-8, atol=1e-9)


def tilt(p, personal, overall):
    odds = (p / (1 - p)) * (personal / (1 - personal)) / (overall / (1 - overall))
    return odds / (1. + odds)


def logmean(count, success, p):
    return (success * np.log(p) + (count - success) * np.log1p(-p)).sum(axis=0) / max(1., count[:, 0].sum())


def channel_scores(q, r, pop):
    n, y = q[:, :1], q[:, 1:]
    prob = (pop[:, 1:].sum(0) + .5) / (pop[:, 0].sum() + 1)
    pc = (pop[:, 1:] + .5) / (pop[:, :1] + 1)
    marginal = (r[:, 1:].sum(0) + 32 * prob) / (r[:, 0].sum() + 32)
    posterior = (r[:, 1:] + 32 * tilt(pc, marginal, prob)) / (r[:, :1] + 32)
    return np.stack([logmean(n, y, marginal), logmean(n, y, posterior)])


def binary_scores(q, r, pop):
    def fraction(t):
        return (t[..., 1] + .5) / (t[..., 0] + 1.)
    def pooled(t, p):
        return (t[..., 1] + 32 * p) / (t[..., 0] + 32)
    g, c = fraction(pop.sum((0, 1))), fraction(pop.sum(1))
    m = pooled(r.sum((0, 1)), g)
    cm = pooled(r.sum(1), tilt(c, m, g))
    tm = pooled(r, tilt(fraction(pop), cm[:, None], c[:, None]))
    n, y = q[..., 0], q[..., 1]
    return np.array([np.sum(y * np.log(p) + (n - y) * np.log1p(-p)) / max(1., n.sum()) for p in (m, cm[:, None], tm)])


def rank_value(r):
    return np.where(r <= 4, np.exp(-np.asarray(r, dtype=float)), 0.)


def boot_check(metric, ids, delta, weights):
    count = np.bincount(ids, minlength=weights.shape[1])
    num = np.bincount(ids, weights=delta, minlength=weights.shape[1])
    values = np.matmul(weights, num) / np.matmul(weights, count)
    close(metric["mean"], delta.mean())
    close(metric["ci95"], np.percentile(values, (2.5, 97.5)))


def verify_pool(pool, ix, joint, perturb, temporal, first, third):
    base = load(f"base_{pool}.npz")
    r = load(f"decisions_{pool}.npz")
    t = load(f"temporal_decisions_{pool}.npz")
    refs = np.array([[ix[k] for k in row] for row in base["refs"]])
    qidx = np.array([ix[k] for k in base["keys"]])
    use = np.where(np.arange(len(qidx)) % 3 != 0)[0]
    if len(set(qidx) | set(refs.ravel())) != 6000 or len(set(qidx) & set(refs.ravel())):
        raise AssertionError("overlap in roles")
    order = np.argsort(-base["scores"][use], kind="stable", axis=1)
    np.testing.assert_array_equal(r["pair"], order[:, :2])
    np.testing.assert_array_equal(r["keys"], t["keys"])
    np.testing.assert_array_equal(r["keys"], base["keys"][use])
    from hashlib import sha256
    owner_order = sorted(range(600), key=lambda i: sha256(f"tactical-owner-v1|{pool}|{base['owners'][i]}".encode()).digest())
    fold = np.empty(600, dtype=int)
    for k, j in enumerate(owner_order):
        fold[j] = k % 4
    np.testing.assert_array_equal(fold, r["folds"])
    tensors = [table[refs].sum(1) for table in (joint, perturb)]
    time_refs = temporal[refs]
    # Precompute population tables per allowed pair of folds. Only reference games contribute.
    priors = {}
    for i, row in enumerate(use):
        a, b = order[i, :2]
        f = tuple(sorted((int(fold[a]), int(fold[b]))))
        label = ",".join(map(str, f))
        if f not in priors:
            take = ~np.isin(fold, f)
            priors[f] = [v[take].sum(0) for v in tensors], time_refs[take].sum((0, 1))
        pops, pop_time = priors[f]
        qg = qidx[row]
        diff = []
        for ref, pop in zip(tensors, pops):
            diff.append(channel_scores(joint[qg], ref[a], pop) - channel_scores(joint[qg], ref[b], pop))
        gap = base["scores"][row, a] - base["scores"][row, b]
        features = [np.r_[gap, d[0], d[1] - d[0]] for d in diff]
        for arm, v in (("hard", features[0]), ("easy", features[0]), ("marginal", features[0][:6]), ("shuffled", features[1])):
            close(r["gap_" + arm][i], v @ first["fit"][pool][label]["weights_in_raw_units"][arm])
        close(r["raw"][i], [diff[0][0].mean(), diff[0][1].mean(), diff[1][1].mean()])
        tm = third["fits"][pool][label]
        for arm_idx in (0, 2):
            residuals = []
            for take in (slice(None), slice(0, 3), slice(3, None)):
                qa = temporal[qg, 0]
                d = binary_scores(qa, time_refs[a, take, arm_idx].sum(0), pop_time[arm_idx])
                d -= binary_scores(qa, time_refs[b, take, arm_idx].sum(0), pop_time[arm_idx])
                residuals.append(d[2] - d[1])
            full, left, right = residuals
            agree = full if np.sign(full) == np.sign(left) == np.sign(right) else 0.
            arms = (("gated", agree), ("ungated", full)) if arm_idx == 0 else (("shuffled", agree),)
            for arm, val in arms:
                close(t[arm][i], tm["base_coefficient"] * gap + tm[arm] * val)
    for data, arms, prefix in ((r, ("hard", "easy", "marginal", "shuffled"), "gap_"), (t, ("gated", "ungated", "shuffled"), "")):
        check_rank = np.argmax(order == data["truth"][:, None], axis=1)
        np.testing.assert_array_equal(check_rank, data["base"])
        for arm in arms:
            edited = order.copy()
            swap = data[prefix + arm] < 0
            edited[swap, :2] = edited[swap, 1::-1]
            check_rank = np.argmax(edited == data["truth"][:, None], axis=1)
            np.testing.assert_array_equal(check_rank, data["rank_" + arm])
    return r, t


def verify_replication(replica):
    d = load("duration_decisions.npz")
    n = replica["players"]
    if len(set(d["signatures"].ravel())) != n * 10:
        raise AssertionError("duplicate independent games")
    vector_file = next(Path(p) for p in replica["audit"]["inputs"] if p.endswith("/data/seq/fold4.npz"))
    with np.load(vector_file, allow_pickle=False) as z:
        lookup = dict(zip(z["key"], z["emb"]))
    v = np.array([[lookup[k] for k in keys] for keys in d["keys"]], dtype=float)
    ref_vec, q_vec = v[:, :7].mean(1), v[:, 7:].reshape(n * 3, -1)
    sims = (q_vec @ ref_vec.T) / (np.linalg.norm(q_vec, axis=1)[:, None] * np.linalg.norm(ref_vec, axis=1))
    order = np.argsort(-sims, axis=1, kind="stable")
    np.testing.assert_array_equal(order[:, :2], d["top2"])
    refs = d["tables"][:, :7].sum(1)
    totals = refs.sum(0)
    queries = d["tables"][:, 7:, 0].reshape(3 * n, 5, 3, 2)
    for i, p in enumerate(d["truth"]):
        same = order[i][(d["rank"][order[i]] == d["rank"][p]) & (order[i] != p)][0]
        assert same == d["matched"][i]
        for name, candidates in (("same", (p, same)), ("actual", d["top2"][i])):
            a, b = candidates
            pop = totals - refs[a] - refs[b]
            diff = [binary_scores(queries[i], refs[a, k], pop[k]) - binary_scores(queries[i], refs[b, k], pop[k]) for k in range(3)]
            close(d[name][i], [*diff[0], diff[1][1], diff[2][2]])
    weights = np.random.default_rng(111816).multinomial(n, np.full(n, 1 / n), 5000)
    for name, scope in (("actual", "actual_finalists"), ("same", "same_rank_nearest_other")):
        x = d[name]
        ids = d["truth"]
        if name == "actual":
            mask = np.any(d["top2"] == ids[:, None], axis=1)
            x = (x * np.where(d["top2"][:, 0] == ids, 1., -1.)[:, None])[mask]
            ids = ids[mask]
        for key, a, b in (("context_minus_marginal", 1, 0), ("context_minus_shuffle", 1, 3),
                          ("duration_minus_context", 2, 1), ("duration_minus_shuffle", 2, 4)):
            boot_check(replica[scope][key], ids, x[:, a] - x[:, b], weights)
    return len(queries)


def main():
    budget = guard()
    one, two, three = (report(s) for s in ("tactical_replies", "tactical_duration", "tactical_temporal_transfer"))
    for p, sha in one["code"].items():
        assert digest_file(ROOT / p) == sha
    assert digest_file(ROOT / "scripts/pid_tactical_temporal_transfer.py") == three["code"]
    assert digest_file(CACHE / "duration_decisions.npz") == two["decisions"]
    j = load("tactics.npz")
    v = load("temporal_counts.npz")
    np.testing.assert_array_equal(j["keys"], v["keys"])
    np.testing.assert_array_equal(j["intact"][:, :, :2], v["tables"][:, 0].sum(2))
    ix = {k: i for i, k in enumerate(j["keys"])}
    assert len(ix) == len(set(j["sig"])) == 12000
    records, timing = [], []
    for p in POOLS:
        for path, sha in ((f"decisions_{p}.npz", one["decisions"][p]), (f"temporal_decisions_{p}.npz", three["data"][p])):
            assert digest_file(CACHE / path) == sha
        r, t = verify_pool(p, ix, j["intact"], j["shuffled"], v["tables"], one, three)
        records.append(r)
        timing.append(t)
    boot = np.random.default_rng(111805)
    weights = np.hstack([boot.multinomial(600, np.full(600, 1 / 600), 5000) for _ in POOLS])
    ids = np.r_[records[0]["truth"], records[1]["truth"] + 600]
    for data, arms, published in ((records, ("hard", "easy", "marginal", "shuffled"), one["summary"]["arms"]),
                                  (timing, ("gated", "ungated", "shuffled"), three["effects"])):
        base = np.concatenate([r["base"] for r in data])
        for arm in arms:
            val = np.concatenate([r["rank_" + arm] for r in data])
            metric = published[arm].get("both", published[arm])
            boot_check(metric, ids, rank_value(val) - rank_value(base), weights)
            assert metric["rescued"] == int(((val == 0) & (base != 0)).sum())
            assert metric["broken"] == int(((val != 0) & (base == 0)).sum())
    boot_check(three["gated_minus_shuffled"], ids, np.concatenate(
        [rank_value(r["rank_gated"]) - rank_value(r["rank_shuffled"]) for r in timing]), weights)
    n = verify_replication(two)
    # Post-hoc decomposition of the observed positive mean, without changing any decision.
    base_rank = np.concatenate([t["base"] for t in timing])
    oriented = np.concatenate([t["raw"] * np.where(t["pair"][:, 0] == t["truth"], 1., -1.) for t in timing])
    by_rank = {}
    for rank in (0, 1):
        mask = base_rank == rank
        den = weights @ np.bincount(ids[mask], minlength=1200)
        data = {"games": int(mask.sum())}
        for label, val in (("signed_duration_increment", oriented), ("favors_true", (oriented > 0) + .5 * (oriented == 0))):
            sums = np.bincount(ids[mask], weights=val[mask], minlength=1200)
            data[label] = {"mean": float(val[mask].mean()), "ci95": np.quantile((weights @ sums) / den, [.025, .975]).tolist()}
        by_rank[str(rank + 1)] = data
    atomic_json(OUT / "tactical_studies_check_20261118.json", {
        "passed": True, "primary_full_ensemble_queries": len(ids), "independent_queries": n,
        "exact_histories_in_first_study": len(ix), "rankings_recomputed": 7 * len(ids),
        "posthoc_duration_evidence_by_original_true_rank": by_rank,
        "verified": ["full score order and seven reranked lists", "five response and temporal likelihood arithmetic",
                     "saved comparator margins and split-reference gates", "independent roster retrieval and matched rivals",
                     "paired owner intervals and first-place flips"],
        "reports": {p.name: digest_file(p) for p in [OUT / f"{name}_20261118.json" for name in
                     ("tactical_replies", "tactical_duration", "tactical_temporal_transfer")]}, "budget": budget})
    print("AUDIT PASSED", len(ids), n, flush=True)


if __name__ == "__main__":
    main()
