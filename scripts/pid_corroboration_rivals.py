"""Compare the same query against nominated and query-independent rivals."""
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pid_corroboration_complement import margins
from pid_corroboration_replicate import BASE_INFO, CACHE, POOLS, ROOT, RUN, atomic_json, atomic_npz, digest_file, guard, hits, projections, read_manifest, read_views, seed_split, unit

PLAN = ROOT / "docs/superpowers/plans/2026-11-18-same-query-rivals.md"
OUT = ROOT / "results/final/corroboration_rivals_20261118.json"


def describe(records, rank):
    base = np.concatenate([r["rank"] for r in records])
    use = base == rank
    ids = np.concatenate([r["cluster"] for r in records])[use]
    raw = np.concatenate([r["raw"] for r in records])[use]
    rivals = np.concatenate([r["rivals"] for r in records])[use]
    flags = hits(raw)
    actual, near, other_game = flags[:, :, 0].T
    sham = flags[:, :, 1]
    rng = np.random.default_rng(111828)
    boot = np.hstack([rng.multinomial(600, np.full(600, 1/600), 5000) for _ in POOLS])
    den = boot @ np.bincount(ids, minlength=1200)
    out = {"games": len(ids), "owners": len(set(ids)), "same_nearest": int((rivals[:, 0] == rivals[:, 1]).sum()),
           "same_other_game": int((rivals[:, 0] == rivals[:, 2]).sum())}
    values = {"actual": actual, "nearest": near, "other_game": other_game,
              "nearest_minus_actual": near-actual, "other_game_minus_actual": other_game-actual,
              "nearest_minus_actual_shuffled": sham[:, 1]-sham[:, 0],
              "difference_of_differences": near-actual-sham[:, 1]+sham[:, 0]}
    for name, v in values.items():
        sampled = (boot @ np.bincount(ids, weights=v, minlength=1200)) / den
        out[name] = {"mean": float(v.mean()), "ci95": np.quantile(sampled, [.025, .975]).tolist()}
    return out


def main():
    guard()
    banks, _ = read_views()
    m = read_manifest(RUN, verify=False)
    earlier = json.loads((ROOT / "results/final/corroboration_replication_20261118.json").read_text())
    records, hashes = [], {}
    for s in (0, 2):
        for pi, p in enumerate(POOLS):
            split = seed_split(m, p, s)
            owners = sorted(split["reference"])
            ref = np.array([split["reference"][o] for o in owners])
            similarity = np.zeros((600, 600))
            for ix, views in banks:
                ri = np.array([[ix[k] for k in row] for row in ref])
                center = unit(views[ri, 0].mean(1))
                similarity += center @ center.T
            np.fill_diagonal(similarity, -np.inf)
            nearest = np.argmax(similarity, axis=1)
            previous = earlier["source"][f"{p}/{s}"]
            archive = BASE_INFO / f"confirm_{p}_{s}.npz"
            fresh_path = CACHE / f"fresh_{p}_{s}.npz"
            if digest_file(archive) != previous["nominees"] or digest_file(fresh_path) != previous["records"]:
                raise ValueError("nominees or previously frozen new-query roles changed")
            with np.load(archive) as z:
                keys, pair, old = z["keys"], z["order"], z["rows"]
            truth = old[:, 0] - 600*pi
            if not np.array_equal(truth, np.repeat(np.arange(600), 2)):
                raise ValueError("adjacent query slots do not correspond to each owner")
            best_wrong = np.where(pair[:, 0] == truth, pair[:, 1], pair[:, 0])
            with np.load(fresh_path) as z:
                select = np.isin(keys, z["keys"]) & (old[:, 1] < 2)
            pos = np.flatnonzero(select)
            if len(set(keys)) != len(keys) or set(keys[pos]) & set(ref.ravel()):
                raise ValueError("reference/query clash")
            rivals = np.column_stack([best_wrong[pos], nearest[truth[pos]], best_wrong[pos ^ 1]])
            if (rivals == truth[pos, None]).any() or np.any(keys[pos] == keys[pos ^ 1]):
                raise ValueError("self or current game used as other-game rival")
            decision = [projections(banks, ref, keys[pos], np.column_stack([truth[pos], r]), margins)[:, 2:] for r in rivals.T]
            rec = {"keys": keys[pos], "cluster": old[pos, 0], "rank": old[pos, 1], "rivals": rivals, "raw": np.stack(decision, axis=1)}
            records.append(rec)
            path = CACHE / f"rivals_{p}_{s}.npz"
            atomic_npz(path, **rec)
            hashes[f"{p}/{s}"] = digest_file(path)
    summary = {k: describe(records, rank) for k, rank in (("true_second", 1), ("true_first", 0))}
    atomic_json(OUT, {**summary, "records": hashes, "protocol": digest_file(PLAN)})
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
