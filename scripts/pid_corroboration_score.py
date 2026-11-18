"""Fit and score frozen temporal specialists on the full committee's real nominees."""
from itertools import combinations_with_replacement
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pid_corroboration_views import CACHE, FAMILIES, PLAN, ROOT, POOLS, atomic_json, atomic_npz, digest_file, guard
from pid_identity_inputs import CACHE as BASE_INFO, baseline_path
from pid_tactical_temporal_transfer import anchored_fit
from aicup_pid.corroboration import pair_features, template_margin
from aicup_pid.tactical_replies import eligible_players, fit_comparator, owner_folds, swapped_ranks

TEMPLATE_PLAN = ROOT / "docs/superpowers/plans/2026-11-18-temporal-templates.md"
REPORT = ROOT / "results/final/corroboration_20261118.json"
ARMS = ("floor", "sham_floor", "intact", "persistent", "sham_persistent", "template", "sham_template")


def read_views():
    banks, metadata = [], {}
    for family in FAMILIES:
        info = json.loads((CACHE / f"{family}.json").read_text())
        path = CACHE / f"views_{family}.npz"
        if info["protocol"] != digest_file(PLAN) or info["views_sha"] != digest_file(path):
            raise ValueError("stored probe vectors or protocol changed")
        with np.load(path) as z:
            keys, vectors = z["keys"], z["views"].astype(np.float32)
        vectors /= np.maximum(np.linalg.norm(vectors, axis=-1, keepdims=True), 1e-12)
        banks.append(({k: i for i, k in enumerate(keys)}, vectors))
        metadata[family] = info
    return banks, metadata


def template_batch(query, reference_a, reference_b):
    return np.column_stack([
        template_margin(reference_a[:, :, cols], reference_b[:, :, cols], query[:, cols[1:]])
        for cols in ([0, 1, 2, 3, 4], [0, 5, 6, 7, 8])
    ])


def measure(bank, b, order):
    idx, views = bank
    full = views[:, 0]
    ri = np.array([[idx[k] for k in row] for row in b["refs"]])
    qi = np.array([idx[k] for k in b["keys"]])
    if ri.shape != (600, 7) or qi.shape != (1800,) or len(set(np.r_[ri.ravel(), qi])) != 6000:
        raise ValueError("seven references and disjoint queries required")
    centers = full[ri].mean(1)
    centers /= np.maximum(np.linalg.norm(centers, axis=1, keepdims=True), 1e-12)
    nominees = np.empty((len(qi), 9, 2), dtype=np.float32)
    template = np.empty((len(qi), 2), dtype=np.float32)
    for start in range(0, len(qi), 64):
        sl = slice(start, start+64)
        q = views[qi[sl]]
        scores = q @ centers.T
        scores = (scores - scores.mean(-1, keepdims=True)) / np.maximum(scores.std(-1, keepdims=True), 1e-12)
        pair = order[sl]
        nominees[sl] = np.take_along_axis(scores, pair[:, None, :], axis=2)
        a, c = views[ri[pair[:, 0]]], views[ri[pair[:, 1]]]
        template[sl] = template_batch(q, a, c)
    return nominees, template


def nominees_and_gap(b):
    truth, base = b["truth"], b["rank"]
    order = np.argsort(-b["scores"], axis=1, kind="stable")[:, :2]
    rows = np.arange(len(truth))
    if not np.array_equal(truth, np.repeat(np.arange(600), 3)) or not np.array_equal(
            (base < 2), (truth[:, None] == order).any(1)):
        raise ValueError("query order or finalist membership mismatch")
    gap = b["scores"][rows, order[:, 0]] - b["scores"][rows, order[:, 1]]
    return order, gap


def calibrate(b, pool, order, gap, raw, arms):
    truth, base = b["truth"], b["rank"]
    held = np.arange(len(truth)) % 3 != 0
    folds = owner_folds(pool, b["owners"])
    groups = np.sort(folds[order], axis=1)
    orient = np.where(order[:, 0] == truth, 1., -1.)
    fitted = np.full((len(truth), len(arms)), np.nan)
    weights = {}
    for a, c in combinations_with_replacement(range(4), 2):
        keep = eligible_players(folds, a, c)
        train = (~held) & (base < 2) & keep[order].all(1)
        test = held & (groups == [a, c]).all(1)
        if keep[order[test]].any() or not keep[truth[train]].all():
            raise ValueError("fit/test finalist folds overlap")
        signed_gap = orient[train] * gap[train]
        scale = float(fit_comparator(signed_gap[:, None])[0])
        weights[f"{a},{c}"] = {"n": int(train.sum()), "base": scale}
        for j, name in enumerate(arms):
            weight = anchored_fit(scale * signed_gap, orient[train] * raw[name][train])
            fitted[test, j] = scale * gap[test] + weight * raw[name][test]
            weights[f"{a},{c}"][name] = float(weight)
    if not np.isfinite(fitted[held]).all():
        raise ValueError("missing out-of-fold decision")
    return held, fitted[held], weights


def score(banks, b, pool, seed):
    truth, base = b["truth"], b["rank"]
    order, gap = nominees_and_gap(b)
    output = [measure(bank, b, order) for bank in banks]
    nominees = np.stack([o[0] for o in output])
    template = np.mean([o[1] for o in output], axis=0)
    raw = pair_features(nominees.mean(0))
    raw.update(template=template[:, 0], sham_template=template[:, 1])
    held, fitted, weights = calibrate(b, pool, order, gap, raw, ARMS)
    cuts = np.quantile(gap[~held], [.2, .4, .6, .8])
    out = {"keys": b["keys"][held], "cluster": truth[held] + 600*POOLS.index(pool),
           "seed": np.full(held.sum(), seed), "base": base[held], "order": order[held], "gap": gap[held],
           "bin": np.searchsorted(cuts, gap[held], side="right"), "cuts": cuts,
           "nominees": nominees[:, held], "raw": np.stack([raw[k][held] for k in ARMS], axis=1),
           "fitted": fitted, "new": np.stack([swapped_ranks(base[held], f < 0) for f in fitted.T], axis=1)}
    return out, weights


def main():
    from pid_corroboration_summary import summarize
    budget = guard()
    banks, probe_info = read_views()
    fixed = json.loads((BASE_INFO / "inputs.json").read_text())
    records, sources = [], {}
    for seed in (1, 3):
        for pool in POOLS:
            path = baseline_path(pool, seed)
            if digest_file(path) != fixed["pools"][pool][str(seed)]:
                raise ValueError("accepted baseline score cache changed")
            with np.load(path) as z:
                b = dict(z)
            rec, weights = score(banks, b, pool, seed)
            target = CACHE / f"score_{pool}_{seed}.npz"
            atomic_npz(target, **rec)
            sources[f"{pool}/{seed}"] = {"baseline": digest_file(path), "record": digest_file(target), "fit": weights}
            records.append(rec)
    results = summarize(records, ARMS)
    atomic_json(REPORT, {"budget": budget, "protocol": digest_file(PLAN), "template_protocol": digest_file(TEMPLATE_PLAN),
                         "probes": probe_info, "sources": sources, **results})
    print(json.dumps(results["combined"], indent=1), flush=True)


if __name__ == "__main__":
    main()
