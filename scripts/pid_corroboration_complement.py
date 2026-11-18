"""Score temporal information perpendicular to the two contenders' intact difference."""
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pid_corroboration_score import ROOT, POOLS, CACHE, baseline_path, read_views, nominees_and_gap, calibrate, digest_file, atomic_json, atomic_npz, guard
from pid_corroboration_replicate import projections, hits, seed_split, read_manifest, RUN
from pid_tactical_score import competition
from aicup_pid.corroboration import vector_margins
from aicup_pid.tactical_replies import swapped_ranks
from aicup_pid.identity_variance import bootstrap_matrix

ARMS = ("whole", "parallel", "perpendicular", "sham_perpendicular")
PLAN = ROOT / "docs/superpowers/plans/2026-11-18-temporal-complement.md"
OUT = ROOT / "results/final/corroboration_complement_20261118.json"


def margins(query, a, b):
    direct = vector_margins(a[:, :, :5], b[:, :, :5], query[:, 1:5])
    cols = [0, 5, 6, 7, 8]
    sham = vector_margins(a[:, :, cols], b[:, :, cols], query[:, 5:])[:, 2]
    return np.column_stack([direct, sham])


def describe(records):
    get = lambda k: np.concatenate([r[k] for r in records])
    ids, base, raw, new, seed = (get(k) for k in ("cluster", "base", "raw", "new", "seed"))
    rng = np.random.default_rng(111825)
    boot = np.hstack([rng.multinomial(600, np.full(600, 1/600), 5000) for _ in POOLS])
    def interval(x, use):
        sums = np.bincount(ids[use], weights=x[use], minlength=1200)
        den = np.bincount(ids[use], minlength=1200)
        return {"mean": float(x[use].mean()), "ci95": np.quantile((boot @ sums)/(boot @ den), [.025, .975]).tolist()}
    changes = competition(new) - competition(base)[:, None]
    all_rows = np.ones(len(ids), bool)
    score, raw_hits = {}, {}
    for j, arm in enumerate(ARMS):
        vals = changes[:, j]
        score[arm] = {**interval(vals, all_rows), "splits": {str(s): interval(vals, seed == s) for s in np.unique(seed)},
                      "rescued": int(((base == 1) & (new[:, j] == 0)).sum()), "broken": int(((base == 0) & (new[:, j] == 1)).sum())}
        success = hits(raw[:, j] * np.where(base == 0, 1, -1))
        raw_hits[arm] = {k: interval(success, mask) for k, mask in (("true_first", base == 0), ("true_second", base == 1), ("in_pair", base < 2))}
    return {"combined": score, "standalone": raw_hits, "perpendicular_minus_shuffled": interval(changes[:, 2]-changes[:, 3], all_rows)}


def main():
    budget = guard()
    banks, _ = read_views()
    screen = json.loads((ROOT / "results/final/corroboration_20261118.json").read_text())
    previous = json.loads((ROOT / "results/final/corroboration_replication_20261118.json").read_text())
    manifest = read_manifest(RUN, verify=False)
    records, sources, paired, strata = [], {}, [], []
    for pool_i, pool in enumerate(POOLS):
        for seed in (1, 3):
            path = baseline_path(pool, seed)
            if digest_file(path) != screen["sources"][f"{pool}/{seed}"]["baseline"]:
                raise ValueError("baseline changed")
            with np.load(path) as data:
                b = dict(data)
            order, gap = nominees_and_gap(b)
            raw = projections(banks, b["refs"], b["keys"], order, margins)
            held, pred, weights = calibrate(b, pool, order, gap, dict(zip(ARMS, raw.T)), ARMS)
            base = b["rank"][held]
            new = np.stack([swapped_ranks(base, col < 0) for col in pred.T], axis=1)
            rec = {"raw": raw[held], "fitted": pred, "new": new, "base": base, "cluster": b["truth"][held]+600*pool_i,
                   "seed": np.full(len(base), seed), "keys": b["keys"][held]}
            records.append(rec)
            target = CACHE / f"complement_{pool}_{seed}.npz"
            atomic_npz(target, **rec)
            sources[f"{pool}/{seed}"] = {"fit": weights, "record": digest_file(target)}
        path = CACHE / f"bilateral_{pool}.npz"
        if digest_file(path) != previous["source"][f"{pool}/bilateral"]["record"]:
            raise ValueError("bilateral pair selection changed")
        with np.load(path) as z:
            pairs = z["players"] - 600*pool_i
            query = z["keys"]
        split = seed_split(manifest, pool, 0)
        refs = np.asarray([split["reference"][p] for p in sorted(split["reference"])])
        raw = projections(banks, refs, query.ravel(), np.repeat(pairs, 2, axis=0), margins).reshape(-1, 2, 4)
        raw[:, 1] *= -1
        paired.append(hits(raw).mean(1))
        strata.extend([pool_i] * len(pairs))
    values = np.vstack(paired)
    bilateral = dict(zip((*ARMS, "advantage"), bootstrap_matrix(np.column_stack([values, values[:, 2]-values[:, 3]]), strata, seed=111827)))
    details = describe(records)
    eff = details["combined"]["perpendicular"]
    atomic_json(OUT, {"protocol": digest_file(PLAN), "budget": budget, "sources": sources, "arms": ARMS,
                      **details, "bilateral": bilateral, "screen_pass": bool(eff["mean"] >= .002 and eff["ci95"][0] > 0)})
    print("perpendicular competition", eff, "bilateral", bilateral, flush=True)


if __name__ == "__main__":
    main()
