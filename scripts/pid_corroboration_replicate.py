"""Frozen temporal templates on newly queried games and bidirectional matched pairs."""
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pid_corroboration_score import ROOT, POOLS, CACHE, BASE_INFO, baseline_path, read_views, template_batch, digest_file, atomic_json, atomic_npz, guard
from pid_corroboration_views import RUN, read_manifest
from pid_choice_manifest import seed_split
from pid_reference_policy import physical_signature
from aicup_pid.identity_variance import bootstrap_matrix, matched_pairs, unit

PLAN = ROOT / "docs/superpowers/plans/2026-11-18-temporal-replication.md"
OUT = ROOT / "results/final/corroboration_replication_20261118.json"


def projections(banks, refs, queries, pairs, per_batch=template_batch):
    results = []
    for lookup, views in banks:
        ix = np.array([[lookup[k] for k in rr] for rr in refs])
        qi = np.array([lookup[k] for k in queries])
        tables = []
        for i in range(0, len(qi), 64):
            sl = slice(i, i+64)
            a, b = pairs[sl].T
            tables.append(per_batch(views[qi[sl]], views[ix[a]], views[ix[b]]))
        results.append(np.concatenate(tables))
    return np.mean(results, axis=0)


def neighbor_pairs(banks, refs):
    similarity = np.zeros((len(refs), len(refs)))
    for ix, views in banks:
        r = np.array([[ix[k] for k in rr] for rr in refs])
        centers = unit(views[r, 0].mean(1))
        similarity += centers @ centers.T
    np.fill_diagonal(similarity, -np.inf)
    near = np.argsort(-similarity, kind="stable")[:, :5]
    edges = sorted((-similarity[a, b], a, b) for a in range(len(near)) for b in near[a]
                   if a < b and a in near[b])
    return matched_pairs([(a, b) for _, a, b in edges])


def hits(signed):
    return (signed > 0).astype(float) + .5 * (signed == 0)


def paired_summary(records, restricted):
    raw = np.concatenate([r["signed"] for r in records])
    pool = np.concatenate([r["pool"] for r in records])
    use = np.concatenate([r["matched"] for r in records]) if restricted else np.ones(len(raw), bool)
    real, sham = hits(raw[use]).transpose(2, 0, 1)
    rates = np.column_stack([real.mean(1), sham.mean(1), (real-sham).mean(1), (real == 1).all(1), (real == 0).all(1)])
    effects = bootstrap_matrix(rates, pool[use], seed=111823)
    return {"pairs": int(use.sum()), "players": int(2*use.sum()), "per_pool_pairs": np.bincount(pool[use], minlength=2).tolist(),
            **dict(zip(("accuracy", "shuffled_accuracy", "advantage", "both_correct", "neither_correct"), effects))}


def finalist_summary(records):
    ids, base = (np.concatenate([r[k] for r in records]) for k in ("owner", "rank"))
    signed = np.concatenate([r["signed"] for r in records])
    rng = np.random.default_rng(111820)
    boot = np.hstack([rng.multinomial(600, np.full(600, 1/600), 5000) for _ in POOLS])
    out = {"unique_games": len(base), "true_in_pair": int((base < 2).sum())}
    values, intervals = [], []
    def format_rates(raw, sampled):
        return {name: {"mean": float(raw[i]), "ci95": np.quantile(sampled[:, i], [.025, .975]).tolist()}
                for i, name in enumerate(("accuracy", "shuffled_accuracy", "advantage"))}
    for rank, label in ((0, "true_first"), (1, "true_second")):
        use = base == rank
        h = hits(signed[use])
        h = np.column_stack([h, h[:, 0] - h[:, 1]])
        counts = np.bincount(ids[use], minlength=1200)
        sums = np.column_stack([np.bincount(ids[use], weights=x, minlength=1200) for x in h.T])
        resampled = (boot @ sums) / (boot @ counts)[:, None]
        avg = h.mean(0)
        out[label] = {"games": int(use.sum()), **format_rates(avg, resampled)}
        values.append(avg)
        intervals.append(resampled)
    out["balanced"] = format_rates(np.mean(values, axis=0), np.mean(intervals, axis=0))
    return out


def main():
    budget = guard()
    banks, _ = read_views()
    manifest = read_manifest(RUN, verify=False)
    games = [g for g in manifest["games"] if g["pool"] in POOLS]
    if len(games) != 12000 or len({physical_signature(g) for g in games}) != len(games):
        raise ValueError("physical candidate histories overlap")
    previous = json.loads((ROOT / "results/final/identity_fisher_confirmation_20261118.json").read_text())
    screen = json.loads((ROOT / "results/final/corroboration_20261118.json").read_text())
    if screen["screen_passed"]:
        raise ValueError("diagnostic follow-up assumes failed reranking screen")
    colour = {g["key"]: g["color"] for g in games}
    with np.load(CACHE / "views_o12Fs0e.npz") as z:
        lengths = dict(zip(z["keys"], z["lengths"]))
    seen, used, pairs_out, finals_out, sources = set(), set(), [], [], {}
    for p in POOLS:
        for s in (1, 3):
            path = baseline_path(p, s)
            if digest_file(path) != screen["sources"][f"{p}/{s}"]["baseline"]:
                raise ValueError("exploration query roles changed")
            with np.load(path) as z:
                seen.update(z["keys"].tolist())
    for s in (0, 2):
        for pi, p in enumerate(POOLS):
            split = seed_split(manifest, p, s)
            owners = sorted(split["reference"])
            ref = np.asarray([split["reference"][o] for o in owners])
            hold = np.asarray([split["heldout"][o] for o in owners])
            if len(set(np.r_[ref.ravel(), hold.ravel()])) != 6000 or ref.shape != (600, 7) or hold.shape != (600, 3):
                raise ValueError("recovered reference and query roles overlap")
            with np.load(baseline_path(p, 1)) as old:
                if owners != old["owners"].tolist():
                    raise ValueError("owner indexing differs")
            path = BASE_INFO / f"confirm_{p}_{s}.npz"
            sha = digest_file(path)
            if sha != previous["archives"][f"{p}/{s}"]:
                raise ValueError("nominee and rank archive changed")
            with np.load(path) as old:
                keys, rows, order = old["keys"], old["rows"], old["order"]
                if not np.array_equal(keys, hold[:, 1:].ravel()) or not np.array_equal(rows[:, 0], np.repeat(np.arange(600)+pi*600, 2)):
                    raise ValueError("new query keys or owners differ from frozen nominees")
                fresh = np.array([k not in seen and k not in used for k in keys])
                if not np.array_equal(old["fresh"], fresh):
                    raise ValueError("independent role and deduplication checks disagree")
            used.update(keys[fresh])
            order, keys, rows = order[fresh], keys[fresh], rows[fresh]
            owner, rank = rows[:, 0], rows[:, 1]
            if not np.array_equal(rank == 0, order[:, 0] == (owner % 600)) or not np.array_equal(rank == 1, order[:, 1] == (owner % 600)):
                raise ValueError("ground-truth finalist order disagrees with stored rank")
            score = projections(banks, ref, keys, order)
            rec = {"owner": owner, "rank": rank, "keys": keys, "order": order, "signed": score*np.where(rank == 0, 1, -1)[:, None]}
            finals_out.append(rec)
            target = CACHE / f"fresh_{p}_{s}.npz"
            atomic_npz(target, **rec)
            sources[f"{p}/{s}"] = {"nominees": sha, "records": digest_file(target)}
            if s != 0:
                continue
            pairs, keypairs = [], []
            selection = neighbor_pairs(banks, ref)
            for a, b in selection:
                both = [[k for k in hold[ob] if k not in seen] for ob in (a, b)]
                if all(both):
                    pairs.append([a, b])
                    keypairs.append([side[0] for side in both])
            pairs, keypairs = np.asarray(pairs), np.asarray(keypairs)
            chosen = projections(banks, ref, keypairs.ravel(), np.repeat(pairs, 2, axis=0)).reshape(-1, 2, 2)
            chosen[:, 1] *= -1
            matched = [colour[k] == colour[l] and abs(int(lengths[k])-int(lengths[l])) <= 16 for k, l in keypairs]
            rec = {"signed": chosen, "players": pairs + 600*pi, "keys": keypairs, "matched": np.array(matched),
                   "pool": np.full(len(pairs), pi)}
            pairs_out.append(rec)
            target = CACHE / f"bilateral_{p}.npz"
            atomic_npz(target, **rec)
            sources[f"{p}/bilateral"] = {"record": digest_file(target), "before_freshness": len(selection)}
    both = paired_summary(pairs_out, False)
    report = {"overall": both, "same_colour_near_length": paired_summary(pairs_out, True),
              "nominees": finalist_summary(finals_out), "source": sources, "earlier_query_keys": len(seen),
              "physical_candidates": len(games), "budget": budget, "protocol": digest_file(PLAN)}
    report["bilateral_replicated"] = bool(both["accuracy"]["ci95"][0] > .5 and both["advantage"]["ci95"][0] > 0)
    atomic_json(OUT, report)
    print(json.dumps({k: report[k] for k in ("overall", "nominees", "bilateral_replicated")}), flush=True)


if __name__ == "__main__":
    main()
