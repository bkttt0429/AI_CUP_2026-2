"""Test one anchored timing correction on the accepted ensemble's actual finalists."""
from concurrent.futures import ProcessPoolExecutor
from itertools import combinations_with_replacement
import json
from pathlib import Path
import sys

import numpy as np
from scipy.optimize import brentq
from scipy.special import expit

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pid_tactical_inputs import CACHE, ROOT, RUN, POOLS, atomic_json, atomic_npz, digest_file, guard, read_manifest
from pid_tactical_duration import extract, duration_probabilities
from pid_tactical_score import competition
from aicup_pid.tactical_replies import eligible_players, fit_comparator, owner_folds, swapped_ranks

PLAN = ROOT / "docs/superpowers/plans/2026-11-18-duration-transfer.md"
REPORT = ROOT / "results/final/tactical_temporal_transfer_20261118.json"
ARMS = ("gated", "ungated", "shuffled")


def agreed_residual(columns):
    x = np.asarray(columns)
    agree = ((x[:, 0] * x[:, 1]) > 0) & ((x[:, 0] * x[:, 2]) > 0)
    return np.where(agree, x[:, 0], 0.)


def anchored_fit(base_logits, residual):
    rms = np.sqrt(np.mean(residual ** 2))
    if rms < 1e-12:
        return 0.
    x = residual / rms
    def derivative(w):
        return -(x * expit(-base_logits - w * x)).mean() + .1 * w
    if derivative(0.) >= 0:
        return 0.
    # mean(abs(x)) <= 1 after RMS scaling; at w=11 the derivative must be positive.
    return brentq(derivative, 0., 11.) / rms


def signed_increment(reference, pop, query, pair):
    probs = [duration_probabilities(r, pop) for r in reference]
    pc = np.stack([p[1] for p in probs])
    pt = np.stack([p[2] for p in probs])
    n, y = query[..., 0], query[..., 1]
    def candidate(ids):
        yes = np.log(pt[ids] / pc[ids])
        no = np.log((1. - pt[ids]) / (1. - pc[ids]))
        return (y * yes + (n - y) * no).sum(axis=(1, 2)) / np.maximum(1, n.sum(axis=(1, 2)))
    return candidate(pair[:, 0]) - candidate(pair[:, 1])


def evaluate_pool(pool, b, reference, query):
    ranks = b["rank"]
    truth, scores = b["truth"], b["scores"]
    pair = np.argsort(-scores, axis=1, kind="stable")[:, :2]
    rows = np.arange(len(truth))
    gap = scores[rows, pair[:, 0]] - scores[rows, pair[:, 1]]
    folds = owner_folds(pool, b["owners"])
    groups = np.sort(folds[pair], axis=1)
    held = rows % 3 != 0
    inside = (truth[:, None] == pair).any(1)
    sign = np.where(truth == pair[:, 0], 1., -1.)
    attrs = {k: np.full(len(rows), np.nan) for k in ("raw", "raw_shuffled", "half1", "half2", *ARMS)}
    models = {}
    for a, c in combinations_with_replacement(range(4), 2):
        keep = eligible_players(folds, a, c)
        fit = (~held) & inside & keep[pair].all(1)
        test = held & (groups == [a, c]).all(1)
        assert np.all(keep[truth[fit]]) and not np.any(keep[pair[test]])
        selected = fit | test
        xf, xt = fit[selected], test[selected]
        g, orient = gap[fit] * sign[fit], sign[fit]
        scale = float(fit_comparator(g[:, None])[0])
        base_logits = g * scale
        models[f"{a},{c}"] = {"fit": int(fit.sum()), "test": int(test.sum()), "base_coefficient": scale}
        for arm_id in (0, 2):
            ref = reference[:, :, arm_id]
            all_ref = ref.sum(1)
            pop = all_ref[keep].sum(0)
            columns = np.column_stack([signed_increment(r, pop, query[selected], pair[selected])
                                       for r in (all_ref, ref[:, :3].sum(1), ref[:, 3:].sum(1))])
            full, agree = columns[:, 0], agreed_residual(columns)
            if arm_id == 0:
                attrs["raw"][test], attrs["half1"][test], attrs["half2"][test] = columns[xt].T
                arms = (("ungated", full), ("gated", agree))
            else:
                attrs["raw_shuffled"][test] = full[xt]
                arms = (("shuffled", agree),)
            for name, feature in arms:
                weight = anchored_fit(base_logits, orient * feature[xf])
                attrs[name][test] = gap[test] * scale + weight * feature[xt]
                models[f"{a},{c}"][name] = weight
    result = {k: v[held] for k, v in attrs.items()}
    if not all(np.isfinite(v).all() for v in result.values()):
        raise ValueError("incomplete finalist decisions")
    result.update({"truth": truth[held], "pair": pair[held], "base": ranks[held], "keys": b["keys"][held]})
    for arm in ARMS:
        result["rank_" + arm] = swapped_ranks(result["base"], result[arm] < 0)
    return result, models


def main():
    budget = guard()
    m = read_manifest(RUN, verify=False)
    binfo = json.loads((CACHE / "baseline.json").read_text())
    # The first experiment already verified uniqueness of the full physical histories.
    ext = json.loads((CACHE / "extract.json").read_text())
    if ext["data"] != digest_file(CACHE / "tactics.npz"):
        raise ValueError("unverified source signatures")
    with np.load(CACHE / "tactics.npz", allow_pickle=False) as z:
        keys, signatures = z["keys"], z["sig"]
    games = {g["key"]: g for g in m["games"] if g["pool"] in POOLS}
    if set(keys) != set(games) or len(set(signatures)) != 12000:
        raise ValueError("source rows changed")
    rows = [(k, games[k]["color"], games[k]["sgf"], sig) for k, sig in zip(keys, signatures)]
    index = {k: i for i, k in enumerate(keys)}
    with ProcessPoolExecutor(max_workers=10) as workers:
        tables = np.asarray(list(workers.map(extract, rows, chunksize=32)))
    # The single nearby channel must replay identically to the original five-channel audit.
    with np.load(CACHE / "tactics.npz", allow_pickle=False) as z:
        if not np.array_equal(tables[:, 0].sum(2), z["intact"][:, :, :2]):
            raise ValueError("nearby replay changed")
    if not np.array_equal(tables[:, 0].sum(2), tables[:, 2].sum(2)):
        raise ValueError("control changed conditional response margins")
    atomic_npz(CACHE / "temporal_counts.npz", keys=keys, tables=tables)
    results, models, artifacts = {}, {}, {}
    for pool in POOLS:
        path = CACHE / f"base_{pool}.npz"
        if binfo["pools"][pool]["cache_sha"] != digest_file(path):
            raise ValueError("baseline cache changed")
        with np.load(path, allow_pickle=False) as z:
            b = dict(z)
        refs = np.array([[index[k] for k in ks] for ks in b["refs"]])
        qi = np.array([index[k] for k in b["keys"]])
        if len(set(np.r_[refs.ravel(), qi])) != 6000:
            raise ValueError("roles overlap")
        results[pool], models[pool] = evaluate_pool(pool, b, tables[refs], tables[qi, 0])
        path = CACHE / f"temporal_decisions_{pool}.npz"
        atomic_npz(path, **results[pool])
        artifacts[pool] = digest_file(path)
    def cat(key):
        return np.concatenate([results[p][key] for p in POOLS])
    ids = cat("truth") + np.repeat([0, 600], 1200)
    rng = np.random.default_rng(111805)
    boot = np.hstack([rng.multinomial(600, np.full(600, 1 / 600), size=5000) for _ in POOLS]).astype(float)
    def metric(x, mask):
        means = (boot @ np.bincount(ids[mask], weights=x[mask], minlength=1200)) / (boot @ np.bincount(ids[mask], minlength=1200))
        return {"mean": float(x[mask].mean()), "ci95": np.quantile(means, [.025, .975]).tolist(), "rows": int(mask.sum())}
    base = cat("base")
    inside, all_rows = base < 2, np.ones(len(base), bool)
    signe = np.where(cat("pair")[:, 0] == cat("truth"), 1., -1.)
    raw = cat("raw")
    agree = (raw * cat("half1") > 0) & (raw * cat("half2") > 0)
    secondary = {"temporal_minus_context": metric(raw * signe, inside),
                 "temporal_minus_shuffled": metric((raw - cat("raw_shuffled")) * signe, inside),
                 "all_reference_signs_agree": int(agree.sum()), "both_halves_agree": int((cat("half1") * cat("half2") > 0).sum())}
    effects = {}
    for arm in ARMS:
        r = cat("rank_" + arm)
        effects[arm] = {**metric(competition(r) - competition(base), all_rows), "score": float(competition(r).mean()),
                        "rescued": int(((r == 0) & (base > 0)).sum()), "broken": int(((r > 0) & (base == 0)).sum()),
                        "swapped": int((cat(arm) < 0).sum())}
    gate_diff = metric(competition(cat("rank_gated")) - competition(cat("rank_shuffled")), all_rows)
    adopt = effects["gated"]["mean"] >= .002 and effects["gated"]["ci95"][0] > 0
    rec = {"plan": digest_file(PLAN), "code": digest_file(__file__), "source_duration_report": digest_file(
           ROOT / "results/final/tactical_duration_20261118.json"), "base_score": float(competition(base).mean()),
           "data": artifacts, "counts_sha": digest_file(CACHE / "temporal_counts.npz"), "fits": models,
           "effects": effects, "gated_minus_shuffled": gate_diff, "secondary": secondary, "adopt": bool(adopt),
           "beats_shuffled": bool(gate_diff["ci95"][0] > 0), "budget_start": budget}
    atomic_json(REPORT, rec)
    print(json.dumps({"gated": effects["gated"], "adopt": bool(adopt)}))


if __name__ == "__main__":
    main()
